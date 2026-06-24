"""
Stage 2: Query Loop — Autonomous Reasoning Engine
==================================================
The main loop that integrates the FSM with a DeepSeek LLM (via langchain-openai
ChatOpenAI), tool execution, and reflection.

Flow per iteration:
  1. FSM transitions to THINK → LLM decides next action
  2. FSM transitions to ACT → tool executor runs the chosen tool
  3. FSM transitions to OBSERVE → collect tool result
  4. FSM transitions to REFLECT → LLM evaluates progress
  5. Loop back to THINK or exit to DONE

Uses tenacity for retry logic on LLM calls.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import logging
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from coding_agent.config import config
from coding_agent.core.fsm import FSM, State

logger = logging.getLogger(__name__)


# ── Retry decorator for LLM calls ──────────────────────────────────

llm_retry = retry(
    stop=stop_after_attempt(config.query_loop_max_retries),
    wait=wait_exponential(
        min=config.query_loop_retry_min_wait,
        max=config.query_loop_retry_max_wait,
    ),
    retry=retry_if_exception_type(
        (ConnectionError, TimeoutError, ValueError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _build_llm() -> ChatOpenAI:
    """Build the ChatOpenAI client for DeepSeek API."""
    if not config.deepseek_api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. "
            "Please configure it in your .env file."
        )
    return ChatOpenAI(
        model=config.deepseek_model,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        timeout=config.llm_request_timeout,
    )


class QueryLoop:
    """
    The main autonomous reasoning loop.

    Args:
        fsm: An FSM instance.
        tool_executor: Callable[[str, Dict], Any] — executes a tool call.
        skill_router: Optional callable to route intents to skills.
        context_providers: Optional list of callables that return extra
                           context dicts injected into the system prompt.
    """

    def __init__(
        self,
        fsm: Optional[FSM] = None,
        tool_executor: Optional[Callable] = None,
        skill_router: Optional[Callable] = None,
        context_providers: Optional[List[Callable]] = None,
    ):
        self.fsm = fsm or FSM()
        self._llm = _build_llm()
        self._tool_executor = tool_executor
        self._skill_router = skill_router
        self._context_providers = context_providers or []

        # Conversation history (raw messages)
        self._messages: List[BaseMessage] = []

        # System prompt (built once and extended per-iteration)
        self._system_prompt: Optional[str] = None

        # Iteration tracking
        self._iteration = 0
        self._max_iterations = config.fsm_max_iterations

        # Callbacks
        self.on_state_change: Optional[Callable[[State, State], None]] = None
        self.on_tool_call: Optional[Callable[[str, Dict, Any], None]] = None

    # ── Public API ───────────────────────────────────────────────

    def set_system_prompt(self, prompt: str) -> None:
        """Set the base system prompt for the agent."""
        self._system_prompt = prompt

    @llm_retry
    def _llm_invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Invoke the LLM with tenacity retry."""
        return self._llm.invoke(messages)  # type: ignore[return-value]

    def run(
        self,
        user_input: str,
        tools: Optional[List[BaseTool]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full reasoning loop for a single user request.

        Args:
            user_input: The user's query or instruction.
            tools: List of langchain BaseTool instances the LLM can call.

        Returns:
            Dict with keys: final_state, iterations, response, history
        """
        self.fsm.reset(State.INIT)
        self._iteration = 0

        # ── Seed the conversation ────────────────────────────
        self._messages = []
        self._add_to_history(SystemMessage(content=self._build_system_prompt()))
        self._add_to_history(HumanMessage(content=user_input))

        # ── Main cognitive cycle ─────────────────────────────
        response_text = ""
        final_tool_results = []

        while self._iteration < self._max_iterations:
            self._iteration += 1
            context = {
                "_iteration": self._iteration,
                "_loop_count": self._iteration - 1,
                "_done": False,
                "_has_tool_result": bool(final_tool_results),
            }

            # 1. Transition FSM
            prev_state = self.fsm.current_state
            new_state = self.fsm.transition(context)
            self._notify_state_change(prev_state, new_state)
            logger.info(
                "[Iter %d/%d] FSM: %s → %s",
                self._iteration, self._max_iterations,
                prev_state.name, new_state.name,
            )

            # 2. Handle each state
            if new_state == State.THINK:
                response_text = self._think(tools=tools)

            elif new_state == State.ACT:
                tool_result = self._act(response_text)
                if tool_result is not None:
                    final_tool_results.append(tool_result)
                    context["_has_tool_result"] = True

            elif new_state == State.OBSERVE:
                self._observe()

            elif new_state == State.REFLECT:
                done = self._reflect(response_text)
                context["_done"] = done
                if done:
                    # One more transition to DONE
                    self.fsm.transition(context)
                    break

            elif new_state == State.DONE:
                break

        # ── Final summary ────────────────────────────────────
        return {
            "final_state": self.fsm.current_state.name,
            "iterations": self._iteration,
            "response": response_text,
            "tool_results": final_tool_results,
            "conversation_summary": self._summarize_conversation(),
        }

    # ── State Handlers ────────────────────────────────────────────

    def _think(self, tools: Optional[List[BaseTool]] = None) -> str:
        """THINK state: invoke LLM to decide the next action."""
        llm_with_tools = self._llm
        if tools:
            llm_with_tools = self._llm.bind_tools(tools)  # type: ignore[attr-defined]

        # Inject skill context if router available
        if self._skill_router and self._messages:
            last_human = next(
                (m for m in reversed(self._messages) if isinstance(m, HumanMessage)),
                None,
            )
            if last_human:
                try:
                    route_result = self._skill_router(last_human.content)
                    if route_result and route_result.get("skill_name"):
                        logger.info(
                            "Skill routed: %s (confidence=%.2f)",
                            route_result["skill_name"],
                            route_result.get("score", 0),
                        )
                except Exception as e:
                    logger.warning("Skill router error: %s", e)

        response: AIMessage = self._llm_invoke(self._messages)  # type: ignore[assignment]
        self._add_to_history(response)
        return response.content or ""

    def _act(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """ACT state: if LLM requested a tool call, execute it."""
        last_msg = self._messages[-1] if self._messages else None
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            # No tool call → treat as "done thinking"
            logger.debug("No tool call in LLM response, skipping ACT.")
            # Manually push to OBSERVE
            self._add_to_history(
                ToolMessage(
                    content="[No tool call requested]",
                    tool_call_id="noop",
                )
            )
            return None

        results = []
        for tc in last_msg.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})

            logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_args))

            # Route to executor
            if self._tool_executor is not None:
                try:
                    result = self._tool_executor(tool_name, tool_args)
                except Exception as e:
                    result = {"success": False, "error": str(e), "result": None}
            else:
                result = {"success": False, "error": "No tool executor registered"}

            results.append(result)

            # Add tool result message
            self._add_to_history(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tc.get("id", "unknown"),
                )
            )

            if self.on_tool_call:
                self.on_tool_call(tool_name, tool_args, result)

        return results[-1] if results else None

    def _observe(self) -> None:
        """OBSERVE state: currently a pass-through; could log or transform results."""
        logger.debug("Observing tool results...")

    def _reflect(self, last_response: str) -> bool:
        """
        REFLECT state: ask the LLM whether the task is complete.
        Returns True if DONE, False if more work needed.
        """
        # Append a meta-prompt for reflection
        reflection_msg = HumanMessage(
            content=(
                "[REFLECTION] Based on the conversation so far, "
                "is the original task fully completed? "
                "Answer with exactly 'YES' or 'NO' followed by a brief reason."
            )
        )
        self._add_to_history(reflection_msg)

        try:
            response: AIMessage = self._llm_invoke(self._messages)  # type: ignore[assignment]
            answer = (response.content or "").strip().upper()
            self._add_to_history(response)
            done = answer.startswith("YES")
            logger.info("Reflection result: %s (done=%s)", answer[:50], done)
            return done
        except Exception as e:
            logger.warning("Reflection failed, assuming done: %s", e)
            return True

    # ── Helpers ─────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build / refresh the system prompt with dynamic context."""
        parts = [
            "You are an expert software engineering automation agent.",
            "You have access to tools that you can call to complete tasks.",
            "",
            "Follow this cognitive cycle:",
            "  1. Think about what needs to be done.",
            "  2. Call a tool (if needed) to make progress.",
            "  3. Observe the result.",
            "  4. Reflect on whether the task is done.",
            "  5. Repeat until done, then respond to the user.",
            "",
        ]

        # Add dynamic context from providers
        for provider in self._context_providers:
            try:
                extra = provider()
                if isinstance(extra, dict):
                    for key, value in extra.items():
                        parts.append(f"[{key}]\n{json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value}\n")
            except Exception as e:
                logger.warning("Context provider error: %s", e)

        return "\n".join(parts)

    def _add_to_history(self, message: BaseMessage) -> None:
        """Add a message to conversation history."""
        self._messages.append(message)

    def _notify_state_change(self, old: State, new: State) -> None:
        if self.on_state_change:
            try:
                self.on_state_change(old, new)
            except Exception as e:
                logger.warning("State change callback error: %s", e)

    def _summarize_conversation(self) -> Dict[str, Any]:
        """Produce a short summary of the conversation for the final output."""
        return {
            "total_messages": len(self._messages),
            "iterations": self._iteration,
            "last_state": self.fsm.current_state.name,
        }

    def get_conversation_history(self) -> List[BaseMessage]:
        """Return the raw conversation message list."""
        return self._messages

    def reset(self) -> None:
        """Reset the loop for a new task."""
        self.fsm.reset()
        self._messages = []
        self._iteration = 0
        logger.info("QueryLoop reset.")
