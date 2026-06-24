"""
Stage 6: Short-Term Memory
===========================
Maintains a bounded deque of recent conversation messages.
Once the deque reaches its maximum length, old messages are evicted (FIFO).

This is the agent's "working memory" — used directly for LLM context.
"""

from __future__ import annotations
from typing import Any, Deque, Dict, List, Optional
from collections import deque
import logging
import json
import time

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Bounded FIFO message queue for recent conversation history.

    Each message is a dict:
        {
            "role": str,       # "user" | "assistant" | "tool" | "system"
            "content": str,    # message content
            "timestamp": float,  # when it was added
            "metadata": dict,    # optional extra info
        }

    Args:
        maxlen: Maximum number of messages to retain. When full, oldest
                messages are automatically evicted.
    """

    def __init__(self, maxlen: int = 50):
        self._maxlen = maxlen
        self._messages: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._evicted_count = 0

    # ── Public API ──────────────────────────────────────────────

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def size(self) -> int:
        """Current number of stored messages."""
        return len(self._messages)

    @property
    def evicted_count(self) -> int:
        """Total number of messages evicted due to maxlen."""
        return self._evicted_count

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to the queue.

        Args:
            role: "user", "assistant", "tool", or "system".
            content: Message text.
            metadata: Optional extra fields.
        """
        # Track evictions
        if len(self._messages) == self._maxlen:
            self._evicted_count += 1

        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        logger.debug("STM add [%s]: %s...", role, content[:60])

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all messages (oldest first)."""
        return list(self._messages)

    def get_recent(self, k: int) -> List[Dict[str, Any]]:
        """Return the most recent k messages."""
        total = len(self._messages)
        if k >= total:
            return list(self._messages)
        return list(self._messages)[total - k:]

    def get_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Return all messages matching a given role."""
        return [m for m in self._messages if m["role"] == role]

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()
        logger.debug("STM cleared.")

    def to_llm_messages(self) -> List[Dict[str, str]]:
        """
        Convert to simple role/content dicts suitable for LLM API calls.
        """
        result = []
        for m in self._messages:
            # Skip system messages (injected separately)
            if m["role"] == "system":
                continue
            # Map roles
            role_map = {
                "user": "user",
                "assistant": "assistant",
                "tool": "tool",
            }
            llm_role = role_map.get(m["role"], "user")
            result.append({"role": llm_role, "content": m["content"]})
        return result

    def to_string(self, max_msg_len: int = 200) -> str:
        """Format all messages as a single string (for summarization)."""
        lines = []
        for m in self._messages:
            content = m["content"]
            if len(content) > max_msg_len:
                content = content[:max_msg_len] + "..."
            lines.append(f"[{m['role']}] {content}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        role_counts = {}
        for m in self._messages:
            role_counts[m["role"]] = role_counts.get(m["role"], 0) + 1
        return (
            f"ShortTermMemory(size={len(self._messages)}/{self._maxlen}, "
            f"evicted={self._evicted_count}, "
            f"roles={role_counts})"
        )
