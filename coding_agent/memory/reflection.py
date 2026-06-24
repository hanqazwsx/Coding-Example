"""
Stage 6: Reflection Pipeline
=============================
Monitors short-term memory and, when a threshold is reached, triggers
a DeepSeek-based reflection that produces a summary stored in long-term memory.

This gives the agent the ability to "learn from experience" across sessions.

Flow:
  1. After each agent turn, the reflection pipeline checks short-term memory size.
  2. If the size >= threshold, it extracts recent messages for reflection.
  3. Calls DeepSeek to produce a reflection summary.
  4. Stores the summary in long-term memory (Chroma).
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import logging
import time

from coding_agent.config import config
from coding_agent.memory.short_term import ShortTermMemory
from coding_agent.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)


class ReflectionPipeline:
    """
    Watches short-term memory and triggers LLM-based reflection.

    Args:
        short_term: ShortTermMemory instance.
        long_term: LongTermMemory instance.
        threshold: Number of messages before triggering reflection.
        summarizer_fn: Optional custom summarizer; uses DeepSeek if None.
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        threshold: int = config.reflection_threshold,
        summarizer_fn: Optional[Callable[[str], str]] = None,
    ):
        self._stm = short_term
        self._ltm = long_term
        self._threshold = threshold
        self._summarizer_fn = summarizer_fn
        self._last_reflection_time: float = 0.0
        self._reflection_count = 0

    # ── Public API ──────────────────────────────────────────────

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def reflection_count(self) -> int:
        return self._reflection_count

    @property
    def time_since_last_reflection(self) -> float:
        """Seconds since the last reflection."""
        if self._last_reflection_time == 0:
            return float("inf")
        return time.time() - self._last_reflection_time

    def step(self, force: bool = False) -> Optional[str]:
        """
        Check conditions and trigger reflection if needed.

        Args:
            force: If True, reflect regardless of threshold.

        Returns:
            Summary string if reflection occurred, None otherwise.
        """
        if not force and self._stm.size < self._threshold:
            return None

        # Gather recent messages for reflection
        messages = self._stm.get_all()
        if not messages:
            return None

        summary = self._generate_reflection(messages)

        # Store in long-term memory
        metadata = {
            "type": "reflection",
            "message_count": len(messages),
            "threshold_triggered": not force,
        }
        entry_id = self._ltm.add_experience(summary, metadata=metadata)

        self._last_reflection_time = time.time()
        self._reflection_count += 1

        logger.info(
            "Reflection #%d stored (id=%s, msgs=%d, summary=%d chars)",
            self._reflection_count,
            (entry_id or "N/A")[:8],
            len(messages),
            len(summary),
        )

        # Optionally: clear or trim short-term memory after reflection
        # (keeping only the most recent few messages for continuity)
        recent = self._stm.get_recent(4)
        self._stm.clear()
        for m in recent:
            self._stm.add(
                role=m["role"],
                content=m["content"],
                metadata=m.get("metadata"),
            )

        return summary

    def reflect_on(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Immediately reflect on a given text and store in LTM.
        Useful for one-off reflections outside the normal cycle.
        """
        try:
            summary = self._llm_reflect(text)
        except Exception as e:
            logger.warning("Reflect-on failed: %s", e)
            return None

        entry_id = self._ltm.add_experience(summary, metadata=metadata)
        self._reflection_count += 1
        return summary

    # ── Internal ────────────────────────────────────────────────

    def _generate_reflection(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a reflection summary from a list of messages.
        Uses the custom summarizer if provided, otherwise DeepSeek.
        """
        text = self._format_messages(messages)

        if self._summarizer_fn is not None:
            return self._summarizer_fn(text)

        return self._llm_reflect(text)

    def _llm_reflect(self, text: str) -> str:
        """
        Use DeepSeek to produce a structured reflection.
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatOpenAI(
                model=config.deepseek_model,
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                temperature=0.2,
                max_tokens=512,
                timeout=30,
            )

            system_msg = SystemMessage(
                content=(
                    "You are a reflection assistant for an autonomous coding agent. "
                    "Given the conversation history below, produce a concise "
                    "reflection that includes:\n"
                    "1. What was the goal / task?\n"
                    "2. What actions were taken?\n"
                    "3. What was the outcome (success/failure)?\n"
                    "4. What lessons or insights can be learned?\n"
                    "5. Any follow-up items?\n\n"
                    "Be specific and factual."
                )
            )
            human_msg = HumanMessage(
                content=f"Conversation history:\n\n{text}"
            )

            response = llm.invoke([system_msg, human_msg])
            summary = (response.content or "").strip()
            if not summary:
                summary = "[Reflection produced empty output]"

            return summary

        except Exception as e:
            logger.warning("LLM reflection failed: %s", e)
            return f"[Reflection failed: {e}]\n\nRaw: {text[:1000]}"

    @staticmethod
    def _format_messages(messages: List[Dict[str, Any]], max_len: int = 800) -> str:
        """Format messages into a reflection-friendly text block."""
        lines = []
        for i, m in enumerate(messages):
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > max_len:
                content = content[:max_len] + "..."
            lines.append(f"[{i}] {role}: {content}")
        return "\n".join(lines[-40:])  # last 40 messages max
