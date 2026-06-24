"""
coding_agent Main Entry Point
==============================
Demonstrates the complete flow:
  User Input → Security Filter → FSM Reasoning → Skill Routing
  → Tool Calling → Context Compression → Memory → Output

Usage:
    1. Set up .env with your DEEPSEEK_API_KEY
    2. pip install -r requirements.txt
    3. python main.py

Modes:
    - demo: runs 3 pre-built demo scenarios (no API key needed)
    - interactive: interactive chat loop (requires API key)
    - eval: run the evaluation suite
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import os
import sys
import json

from coding_agent.config import config, setup_logging
from coding_agent.core.fsm import FSM, State
from coding_agent.core.query_loop import QueryLoop
from coding_agent.core.compressor import ContextCompressor
from coding_agent.core.cache_manager import CacheManager
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tools.executor import ToolExecutor
from coding_agent.skills.catalog import catalog_summary
from coding_agent.skills.router import SkillRouter
from coding_agent.memory.short_term import ShortTermMemory
from coding_agent.memory.long_term import LongTermMemory
from coding_agent.memory.reflection import ReflectionPipeline
from coding_agent.security.filter import InputFilter
from coding_agent.security.sandbox import ShellSandbox
from coding_agent.security.audit import AuditLogger
from coding_agent.monitoring.tracer import Tracer
from coding_agent.agents.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


# ── Component initialiser ──────────────────────────────────────────

class CodingAgent:
    """
    Aggregates all coding_agent components into a single pipeline.

    Flow: input → security filter → skill router → FSM + query loop
          → tool executor → memory → tracer → output
    """

    def __init__(self):
        # ── Initialise all components ──────────────────────────
        self.config = config
        self.fsm = FSM()
        self.tool_registry = ToolRegistry.get_instance()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.input_filter = InputFilter()
        self.shell_sandbox = ShellSandbox()
        self.audit_logger = AuditLogger()
        self.tracer = Tracer()
        self.cache_manager = CacheManager()
        self.compressor = ContextCompressor()
        self.short_term_memory = ShortTermMemory(maxlen=config.short_term_maxlen)
        self.long_term_memory = LongTermMemory()

        # Skill-path → registered-tool-name mapping for demo mode
        # (Allows FSM-only demo to execute real tools based on skill routing)
        self._skill_to_tool_map: Dict[str, str] = {
            "backend.code_review.review_code": "read_file",
            "frontend.component_gen.create_component": "write_file",
            "devops.container.write_dockerfile": "write_file",
            "backend.api_dev.create_endpoint": "write_file",
            "backend.api_dev.generate_model": "write_file",
            "backend.db_dev.write_query": "write_file",
            "data.pipeline.etl_script": "write_file",
        }
        self.reflection_pipeline = ReflectionPipeline(
            self.short_term_memory,
            self.long_term_memory,
        )
        self.skill_router = SkillRouter()

        # ── Wire up security hooks into tool executor ──────────
        self._setup_security_hooks()

        # ── Build the skill index ──────────────────────────────
        self._init_skill_router()

        logger.info("CodingAgent initialised with all components.")

    # ── Main inference ─────────────────────────────────────────

    def process(
        self,
        user_input: str,
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single user input through the full pipeline.

        Args:
            user_input: Raw user input text.
            use_llm: If True, use DeepSeek LLM for reasoning.
                     If False, use the simpler FSM-only flow (demo mode).

        Returns:
            Dict with pipeline results.
        """
        # 1. Security filter
        is_safe, reason = self.input_filter.detect_injection(user_input)
        self.audit_logger.log_event(
            "user_input", "user",
            {"input_preview": user_input[:100], "safe": is_safe},
        )
        if not is_safe:
            self.audit_logger.log_security_violation(
                "injection", "user", {"input": user_input[:100], "reason": reason},
            )
            return {
                "status": "blocked",
                "error": f"Input blocked by security filter: {reason}",
                "skill_route": None,
                "tool_results": [],
                "iterations": 0,
            }

        # 2. Store in short-term memory
        self.short_term_memory.add("user", user_input)

        # 3. Skill routing
        try:
            route_result = self.skill_router.route(user_input)
        except Exception as e:
            logger.warning("Skill routing failed: %s", e)
            route_result = None

        # 4. Process through FSM / QueryLoop
        if use_llm and config.deepseek_api_key:
            result = self._process_with_llm(user_input)
        else:
            result = self._process_fsm_only(user_input, route_result)

        result["skill_route"] = route_result

        # 5. Reflection pipeline check
        try:
            self.reflection_pipeline.step()
        except Exception as e:
            logger.warning("Reflection step failed: %s", e)

        return result

    # ── LLM path ──────────────────────────────────────────────

    def _process_with_llm(self, user_input: str) -> Dict[str, Any]:
        """Full LLM-driven processing."""
        # Build system prompt from cached components
        system_prompt = self.cache_manager.get_or_build(
            "system_prompt",
            builder=self._build_system_prompt,
            description="Agent system prompt",
        )

        # Set up the query loop
        loop = QueryLoop(
            fsm=self.fsm,
            tool_executor=self._safe_tool_executor,
            skill_router=self.skill_router.route,
        )
        loop.set_system_prompt(system_prompt)
        loop.on_state_change = lambda old, new: self.tracer.record(
            "state_transition",
            f"{old.name}->{new.name}",
            0.0,
            metadata={"from": old.name, "to": new.name},
        )

        # Run
        result = loop.run(user_input)

        # Store in memory
        self.short_term_memory.add("assistant", result.get("response", ""))

        return {
            "status": "completed",
            "response": result.get("response", ""),
            "tool_results": result.get("tool_results", []),
            "iterations": result.get("iterations", 0),
            "conversation_summary": result.get("conversation_summary", {}),
        }

    # ── FSM-only path (no LLM, for demo/testing) ──────────────

    def _process_fsm_only(
        self,
        user_input: str,
        route_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simplified processing without LLM.
        Uses the FSM + Skill routing to demonstrate the flow.
        """
        self.fsm.reset(State.INIT)
        iterations = 0
        tool_results = []
        current_state = self.fsm.current_state

        # Log initial state
        self.audit_logger.log_state_transition("START", current_state.name)

        while current_state != State.DONE and iterations < config.fsm_max_iterations:
            iterations += 1
            prev = current_state

            # Manual state handling
            if current_state == State.THINK:
                # Use skill suggestion if available
                if route_result and not tool_results:
                    thought = (
                        f"Intent recognised: {route_result['skill_name']} "
                        f"(path: {route_result['path']}, confidence: {route_result['score']})"
                    )
                else:
                    thought = "Processing user request..."

                self.short_term_memory.add("assistant", thought)
                current_state = self.fsm.transition({
                    "_iteration": iterations,
                    "_loop_count": iterations - 1,
                    "_done": False,
                    "_has_tool_result": bool(tool_results),
                })

            elif current_state == State.ACT:
                # Execute the matched skill's tool
                if route_result and route_result.get("path"):
                    skill_path = route_result["path"]
                    # Map skill path to actual registered tool name for demo mode
                    mapped_tool = self._skill_to_tool_map.get(skill_path, "")
                    if not mapped_tool:
                        mapped_tool = route_result.get("skill_name", "read_file")
                    # If still not registered, fall back to read_file
                    if not self.tool_registry.tool_exists(mapped_tool):
                        mapped_tool = "read_file"
                    with self.tracer.trace("tool_call", mapped_tool):
                        tr_result = self._safe_tool_executor(mapped_tool, {})
                        tool_results.append(tr_result)
                        self.audit_logger.log_tool_call(mapped_tool, {}, tr_result)
                else:
                    pass  # no tool to call

                current_state = self.fsm.transition({
                    "_has_tool_result": len(tool_results) > 0,
                })

            elif current_state == State.OBSERVE:
                current_state = self.fsm.transition()

            elif current_state == State.REFLECT:
                done = len(tool_results) > 0
                current_state = self.fsm.transition({
                    "_done": done,
                    "_loop_count": iterations,
                })

            else:  # INIT or others
                current_state = self.fsm.transition()

            # Trace state change
            self.tracer.record(
                "state_transition",
                f"{prev.name}->{current_state.name}",
                0.0,
            )
            self.audit_logger.log_state_transition(prev.name, current_state.name)

        response = "Task processed. See tool results for details."

        return {
            "status": "completed",
            "response": response,
            "tool_results": tool_results,
            "iterations": iterations,
            "states_visited": [s.name for s in self.fsm.history] + [current_state.name],
            "conversation_summary": {"short_term_size": self.short_term_memory.size},
        }

    # ── Security-wrapped tool executor ─────────────────────────

    def _safe_tool_executor(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Security-wrapped tool executor for the query loop."""
        # 1. Validate tool name
        if not self.input_filter.validate_tool_name(tool_name):
            self.audit_logger.log_security_violation(
                "blocked_tool", "agent",
                {"tool": tool_name, "params": params},
            )
            return {"success": False, "error": f"Tool '{tool_name}' is not in the whitelist"}

        # 2. Check parameters for injection
        is_safe, reason = self.input_filter.detect_tool_param_injection(tool_name, params)
        if not is_safe:
            self.audit_logger.log_security_violation(
                "param_injection", "agent",
                {"tool": tool_name, "reason": reason},
            )
            return {"success": False, "error": f"Parameter injection detected: {reason}"}

        # 3. Route shell_exec through sandbox
        if tool_name == "shell_exec":
            command = params.get("command", "")
            with self.tracer.trace("tool_call", "shell_exec"):
                result = self.shell_sandbox.execute(command)
        else:
            with self.tracer.trace("tool_call", tool_name):
                result = self.tool_executor.execute(tool_name, params)

        # 4. Audit
        self.audit_logger.log_tool_call(tool_name, params, result.to_dict())

        return result.to_dict()

    # ── Setup ─────────────────────────────────────────────────

    def _setup_security_hooks(self) -> None:
        """Add security hooks to the tool executor."""

        def pre_exec_hook(tool_name: str, params: Dict[str, Any]) -> None:
            # Validate tool name
            if not self.input_filter.validate_tool_name(tool_name):
                raise PermissionError(f"Tool '{tool_name}' is not allowed")

            # Check params
            is_safe, reason = self.input_filter.detect_tool_param_injection(
                tool_name, params
            )
            if not is_safe:
                raise PermissionError(f"Injection detected: {reason}")

        self.tool_executor.add_pre_hook(pre_exec_hook)

    def _init_skill_router(self) -> None:
        """Build the skill index (best-effort)."""
        try:
            summary = catalog_summary()
            logger.info("Skill catalogue: %s", summary)
            self.skill_router.build_index()
        except Exception as e:
            logger.warning("Skill router init skipped: %s", e)

    @staticmethod
    def _build_system_prompt() -> str:
        """Build the agent system prompt (cached)."""
        return (
            "You are coding_agent, an autonomous software engineering assistant.\n"
            "You have access to file system and shell tools.\n"
            "Follow the cognitive cycle: THINK → ACT → OBSERVE → REFLECT → DONE.\n"
            "Be concise and precise in your responses."
        )

    def get_status(self) -> Dict[str, Any]:
        """Return component status."""
        return {
            "fsm_state": self.fsm.current_state.name,
            "stm_size": self.short_term_memory.size,
            "ltm_count": self.long_term_memory.count(),
            "tool_registry": [n for n, _ in self.tool_registry.list_all()],
            "cache_stats": self.cache_manager.stats(),
            "audit_events": self.audit_logger.total_events,
            "tracer_stats": self.tracer.get_statistics(),
        }


# ── Demo scenarios ─────────────────────────────────────────────────

DEMO_SCENARIOS = [
    {
        "title": "Demo 1: Code Review Request",
        "input": "Review the code in config.py for potential issues.",
    },
    {
        "title": "Demo 2: File Operation",
        "input": "Create a new Python file called 'hello.py' with a hello world function.",
    },
    {
        "title": "Demo 3: Project Analysis",
        "input": "List all Python files in the project and summarize their purposes.",
    },
]


def run_demo_mode(agent: CodingAgent) -> None:
    """Run the 3 pre-built demo scenarios."""

    print("\n" + "=" * 60)
    print("  coding_agent — Demo Mode (FSM only, no LLM)")
    print("=" * 60)

    for scenario in DEMO_SCENARIOS:
        print(f"\n{'─' * 40}")
        print(f"  {scenario['title']}")
        print(f"  Input: \"{scenario['input']}\"")
        print(f"{'─' * 40}")

        result = agent.process(scenario["input"], use_llm=False)

        print(f"  Status:      {result['status']}")
        print(f"  Iterations:  {result['iterations']}")

        if result.get("skill_route"):
            sr = result["skill_route"]
            print(f"  Skill route: {sr.get('skill_name', 'N/A')} "
                  f"(score={sr.get('score', 0):.3f})")

        if result.get("tool_results"):
            for tr in result["tool_results"]:
                print(f"  Tool:        {tr.get('tool_name', '?')} -- "
                      f"{'[OK]' if tr.get('success') else '[FAIL]'} "
                      f"({tr.get('duration_ms', 0):.1f}ms)")

        print(f"  States:      {' -> '.join(result.get('states_visited', []))}")

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"  Agent status: {agent.get_status()}")
    print(f"{'=' * 60}\n")


def run_interactive_mode(agent: CodingAgent) -> None:
    """Run an interactive chat loop (requires DeepSeek API key)."""
    if not config.deepseek_api_key:
        print("\nERROR: DEEPSEEK_API_KEY not set. Interactive mode requires an API key.")
        print("Set it in .env or run with DEEPSEEK_API_KEY=your_key python main.py\n")
        return

    print("\n" + "=" * 60)
    print("  coding_agent — Interactive Mode (LLM-enabled)")
    print("  Type 'exit' to quit, 'status' for agent status")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if user_input.lower() == "status":
            print(json.dumps(agent.get_status(), indent=2, default=str))
            continue

        # Process
        result = agent.process(user_input, use_llm=True)

        if result["status"] == "blocked":
            print(f"[BLOCKED] {result.get('error', '')}")
            continue

        print(f"\nAgent: {result.get('response', '(no response)')}")

        if result.get("skill_route"):
            sr = result["skill_route"]
            print(f"  [Routed: {sr.get('skill_name', 'N/A')}]")

        tool_results = result.get("tool_results", [])
        if tool_results:
            for tr in tool_results:
                status = "✓" if tr.get("success") else "✗"
                print(f"  [Tool: {tr.get('tool_name', '?')} {status}]")


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="coding_agent — Backend Engineering Automation Coding Agent"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="demo",
        choices=["demo", "interactive", "eval"],
        help="Run mode (default: demo)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(r"""
   ____          _       _
  / ___|___   __| | __ _(_)_ __
 | |   / _ \ / _` |/ _` | | '_ \
 | |__| (_) | (_| | (_| | | | | |
  \____\___/ \__,_|\__, |_|_| |_|
                   |___/
  Backend Engineering Automation Coding Agent
  ===========================================
""")

    # Init agent
    try:
        agent = CodingAgent()
    except Exception as e:
        print(f"ERROR initialising agent: {e}")
        sys.exit(1)

    # Route mode
    if args.mode == "demo":
        run_demo_mode(agent)
    elif args.mode == "interactive":
        run_interactive_mode(agent)
    elif args.mode == "eval":
        from coding_agent.monitoring.eval import Evaluator
        evaluator = Evaluator()
        report = evaluator.run(verbose=True)
        evaluator.export_report(report)

    # Cleanup
    agent.audit_logger.close()


if __name__ == "__main__":
    main()
