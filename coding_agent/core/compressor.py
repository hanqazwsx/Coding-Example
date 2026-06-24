"""
Stage 5: Hierarchical Context Compressor
=========================================
Implements a sliding-window + summarization approach for conversation history.

Strategy:
  1. Keep the last N (configurable) message turns in full fidelity.
  2. Older messages are compressed into a single summary via DeepSeek LLM.
  3. The compressed summary replaces the old messages in the context.

This reduces token usage while preserving essential context for the agent.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from coding_agent.config import config

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Compresses conversation history using a sliding window + summary approach.

    Args:
        keep_last_n: Number of recent message pairs to keep uncompressed.
        summarizer_fn: Optional callable that takes text and returns a summary.
                       If None, uses DeepSeek API.
    """

    def __init__(
        self,
        keep_last_n: int = config.compressor_keep_last_n,
        summarizer_fn: Optional[Callable[[str], str]] = None,
    ):
        self._keep_last_n = keep_last_n
        self._summarizer_fn = summarizer_fn
        self._compressed_summary: Optional[str] = None

    # ── Public API ──────────────────────────────────────────────

    def compress(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compress a list of message dicts.

        Args:
            messages: Each message is a dict with keys like
                      {"role": "user"|"assistant"|"tool", "content": "..."}

        Returns:
            A compressed list where:
              - Old messages are replaced by a single "system" entry (the summary).
              - Recent messages stay as-is.
        """
        if len(messages) <= self._keep_last_n:
            return messages  # nothing to compress

        # Split: old (to be summarised) vs recent (keep raw)
        split_idx = len(messages) - self._keep_last_n
        old_part = messages[:split_idx]
        recent_part = messages[split_idx:]

        # Generate or reuse summary
        summary = self._summarize(old_part)

        # Return: [summary_entry] + recent_raw
        compressed: List[Dict[str, Any]] = [
            {"role": "system", "content": f"[Context Summary]\n{summary}"}
        ]
        compressed.extend(recent_part)

        self._compressed_summary = summary
        logger.debug(
            "Context compressed: %d msgs → %d + summary (kept last %d)",
            len(messages),
            len(recent_part),
            self._keep_last_n,
        )
        return compressed

    def compress_langchain(
        self,
        messages: List,
    ) -> List:
        """
        Compress a list of langchain BaseMessage objects.
        Converts to dict, compresses, converts back.
        """
        # Convert to dict
        dict_msgs = []
        for m in messages:
            content = m.content if hasattr(m, "content") else str(m)
            role = "unknown"
            if hasattr(m, "type"):
                role = m.type  # type of BaseMessage: "human", "ai", "tool", "system"
            dict_msgs.append({"role": role, "content": content})

        compressed_dicts = self.compress(dict_msgs)

        # Convert back to dict format (caller can reconstruct)
        return compressed_dicts

    def get_summary(self) -> Optional[str]:
        """Return the last generated summary."""
        return self._compressed_summary

    def reset(self) -> None:
        """Clear the cached summary."""
        self._compressed_summary = None

    # ── Internal ────────────────────────────────────────────────

    def _summarize(self, messages: List[Dict[str, Any]]) -> str:
        """
        Summarise a list of messages using either the injected summarizer
        or the DeepSeek API.
        """
        text = self._format_messages_for_summary(messages)

        if self._summarizer_fn is not None:
            return self._summarizer_fn(text)

        return self._llm_summarize(text)

    def _llm_summarize(self, text: str) -> str:
        """Use DeepSeek to generate a concise summary."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatOpenAI(
                model=config.compressor_summary_model,
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                temperature=0.1,
                max_tokens=512,
                timeout=30,
            )

            system_msg = SystemMessage(
                content=(
                    "You are a context compression assistant. "
                    "Summarise the following conversation history concisely, "
                    "retaining all important decisions, tool calls, results, "
                    "and user requirements. Focus on facts, not fluff."
                )
            )
            human_msg = HumanMessage(
                content=f"Please summarise:\n\n{text}"
            )

            response = llm.invoke([system_msg, human_msg])
            summary = (response.content or "").strip()
            if not summary:
                summary = "[Summary generation returned empty]"

            logger.info("Generated context summary (%d chars)", len(summary))
            return summary

        except Exception as e:
            logger.warning("LLM summarization failed: %s", e)
            # Fallback: simple truncation
            return self._truncate_summary(text)

    @staticmethod
    def _format_messages_for_summary(messages: List[Dict[str, Any]]) -> str:
        """Format messages into a plain text block for the summarizer."""
        lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate very long messages
            if isinstance(content, str) and len(content) > 2000:
                content = content[:2000] + "... [truncated]"
            lines.append(f"[{i}] ({role}): {content}")
        return "\n".join(lines)

    @staticmethod
    def _truncate_summary(text: str, max_chars: int = 2000) -> str:
        """Fallback: simple truncation when LLM is unavailable."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [truncated]"
