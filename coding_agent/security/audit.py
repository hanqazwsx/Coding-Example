"""
Stage 8: Security Audit Trail
==============================
Provides an append-only log of all security-relevant events:
  - User inputs received
  - Tool calls attempted (and whether allowed/blocked)
  - State transitions
  - Filter/sandbox violations
  - Worker agent activity

Logs are written to both a local file and an in-memory buffer.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import json
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Append-only security audit trail.

    Args:
        log_dir: Directory to write audit log files.
        max_buffer_size: Max in-memory entries before auto-flush.
        enabled: Set False to disable auditing (not recommended).
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_buffer_size: int = 100,
        enabled: bool = True,
    ):
        self._log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "audit_logs",
        )
        self._max_buffer_size = max_buffer_size
        self._enabled = enabled
        self._buffer: List[Dict[str, Any]] = []
        self._total_events = 0

        # Ensure log directory exists
        if self._enabled:
            os.makedirs(self._log_dir, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────

    @property
    def total_events(self) -> int:
        return self._total_events

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def log_event(
        self,
        event_type: str,
        actor: str,
        details: Dict[str, Any],
    ) -> None:
        """
        Record an audit event.

        Args:
            event_type: Category (user_input, tool_call, state_transition,
                        security_violation, filter_block, etc.)
            actor: Who/what performed the action (user, agent_id, system).
            details: Event-specific data.
        """
        if not self._enabled:
            return

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "actor": actor,
            "details": self._sanitize_for_log(details),
        }

        self._buffer.append(event)
        self._total_events += 1

        # Log to Python logger
        logger.debug("AUDIT [%s] %s: %s", event_type, actor, str(details)[:120])

        # Auto-flush if buffer is full
        if len(self._buffer) >= self._max_buffer_size:
            self.flush()

    def log_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        actor: str = "agent",
        allowed: bool = True,
    ) -> None:
        """Convenience: log a tool call event."""
        self.log_event(
            event_type="tool_call",
            actor=actor,
            details={
                "tool": tool_name,
                "params": params,
                "result_summary": {
                    "success": result.get("success"),
                    "error": result.get("error", "")[:200],
                },
                "allowed": allowed,
            },
        )

    def log_security_violation(
        self,
        violation_type: str,
        actor: str,
        details: Dict[str, Any],
    ) -> None:
        """Convenience: log a security violation."""
        self.log_event(
            event_type=f"security_violation_{violation_type}",
            actor=actor,
            details=details,
        )

    def log_state_transition(
        self,
        from_state: str,
        to_state: str,
        actor: str = "fsm",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience: log a state transition."""
        self.log_event(
            event_type="state_transition",
            actor=actor,
            details={
                "from": from_state,
                "to": to_state,
                "context_summary": str(context)[:200] if context else "",
            },
        )

    def flush(self) -> None:
        """Flush buffered events to disk."""
        if not self._enabled or not self._buffer:
            return

        filename = f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl"
        filepath = os.path.join(self._log_dir, filename)

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                for event in self._buffer:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            logger.info("Audit flushed %d events to %s", len(self._buffer), filename)
            self._buffer.clear()
        except Exception as e:
            logger.error("Audit flush failed: %s", e)

    def get_recent_events(
        self,
        n: int = 20,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent events, optionally filtered by type."""
        events = self._buffer
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        return events[-n:]

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics."""
        type_counts: Dict[str, int] = {}
        for e in self._buffer:
            et = e["event_type"]
            type_counts[et] = type_counts.get(et, 0) + 1

        return {
            "total_events": self._total_events,
            "buffered": len(self._buffer),
            "types": type_counts,
            "log_dir": self._log_dir,
        }

    def close(self) -> None:
        """Flush and release resources."""
        self.flush()
        logger.info("AuditLogger closed. Total events: %d", self._total_events)

    # ── Internal ────────────────────────────────────────────────

    @staticmethod
    def _sanitize_for_log(details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive data (API keys, secrets) from log entries.
        """
        sanitized = {}
        SENSITIVE_KEYS = {"api_key", "api-key", "apikey", "secret", "password",
                          "token", "authorization", "auth", "credential", "key"}

        for k, v in details.items():
            if isinstance(k, str) and k.lower() in SENSITIVE_KEYS:
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, dict):
                sanitized[k] = AuditLogger._sanitize_for_log(v)
            elif isinstance(v, str) and len(v) > 500:
                sanitized[k] = v[:500] + "..."
            else:
                sanitized[k] = v
        return sanitized
