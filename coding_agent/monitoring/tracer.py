"""
Stage 9: Full-Link Instrumentation Tracer
==========================================
Records timing and metadata for every major operation:
  - State transitions
  - Tool calls
  - Skill routing
  - Agent actions

Stores traces in both a log file and a Chroma "trace" collection.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import logging
import json
import os
import time
import uuid
from datetime import datetime
from functools import wraps

from coding_agent.config import config

logger = logging.getLogger(__name__)


# ── Trace entry model ──────────────────────────────────────────────

class TraceEntry:
    """A single trace record."""

    def __init__(
        self,
        trace_type: str,
        name: str,
        duration_ms: float,
        input_summary: str = "",
        output_summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.trace_id = uuid.uuid4().hex[:12]
        self.timestamp = datetime.utcnow().isoformat()
        self.trace_type = trace_type
        self.name = name
        self.duration_ms = round(duration_ms, 2)
        self.input_summary = input_summary[:200]
        self.output_summary = output_summary[:200]
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "type": self.trace_type,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "input": self.input_summary,
            "output": self.output_summary,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"[{self.trace_type}] {self.name} "
            f"({self.duration_ms}ms)"
        )


class Tracer:
    """
    Instrumentation tracer with Chroma + log file storage.

    Usage:
        tracer = Tracer()

        # As context manager:
        with tracer.trace("tool_call", "read_file"):
            result = read_file(...)

        # As decorator:
        @tracer.traced("skill_route", "skill_router")
        def route(intent): ...

        # Manual:
        tracer.record("state_transition", "INIT->THINK", 0.0)
    """

    def __init__(self, log_dir: Optional[str] = None):
        self._log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trace_logs",
        )
        self._entries: List[TraceEntry] = []
        self._chroma_collection = None
        self._chroma_initialised = False
        self._enabled = True

        # Ensure log dir exists
        os.makedirs(self._log_dir, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────

    def record(
        self,
        trace_type: str,
        name: str,
        duration_ms: float,
        input_summary: str = "",
        output_summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceEntry:
        """
        Record a trace entry manually.

        Args:
            trace_type: Category (state_transition, tool_call, skill_route, etc.)
            name: Human-readable name.
            duration_ms: Duration in milliseconds.
            input_summary: Short summary of input.
            output_summary: Short summary of output.
            metadata: Extra structured data.

        Returns:
            The created TraceEntry.
        """
        if not self._enabled:
            # Return a dummy entry
            return TraceEntry(trace_type, name, duration_ms)

        entry = TraceEntry(
            trace_type=trace_type,
            name=name,
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=metadata,
        )
        self._entries.append(entry)

        # Log to file
        self._write_to_file(entry)

        # Store in Chroma (async-ish)
        self._store_in_chroma(entry)

        logger.debug("Trace: %s", entry)
        return entry

    def trace(self, trace_type: str, name: str):
        """
        Context manager for tracing a block of code.

        Usage:
            with tracer.trace("tool_call", "read_file"):
                do_something()
        """
        class _TraceContext:
            def __init__(self, outer: Tracer, ttype: str, tname: str):
                self.outer = outer
                self.ttype = ttype
                self.tname = tname
                self.start = 0.0
                self.input = ""

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = (time.perf_counter() - self.start) * 1000
                output = str(exc_val) if exc_val else "OK"
                self.outer.record(
                    trace_type=self.ttype,
                    name=self.tname,
                    duration_ms=duration,
                    output_summary=output[:200],
                    metadata={"exception": exc_type.__name__ if exc_type else None},
                )

        return _TraceContext(self, trace_type, name)

    def traced(self, trace_type: str, name: Optional[str] = None):
        """
        Decorator for tracing a function.

        Usage:
            @tracer.traced("tool_call")
            def my_function(...): ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = name or func.__name__
                start = time.perf_counter()
                input_summary = (
                    f"args={str(args)[:100]}, kwargs={str(kwargs)[:100]}"
                )
                try:
                    result = func(*args, **kwargs)
                    duration = (time.perf_counter() - start) * 1000
                    self.record(
                        trace_type=trace_type,
                        name=func_name,
                        duration_ms=duration,
                        input_summary=input_summary,
                        output_summary=str(result)[:200],
                    )
                    return result
                except Exception as e:
                    duration = (time.perf_counter() - start) * 1000
                    self.record(
                        trace_type=trace_type,
                        name=func_name,
                        duration_ms=duration,
                        input_summary=input_summary,
                        output_summary=f"ERROR: {e}",
                        metadata={"error": str(e)},
                    )
                    raise
            return wrapper
        return decorator

    def get_traces(
        self,
        trace_type: Optional[str] = None,
        n: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent traces, optionally filtered by type."""
        entries = self._entries
        if trace_type:
            entries = [e for e in entries if e.trace_type == trace_type]
        return [e.to_dict() for e in entries[-n:]]

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregated trace statistics."""
        if not self._entries:
            return {"total": 0, "by_type": {}}

        by_type: Dict[str, List[float]] = {}
        for e in self._entries:
            by_type.setdefault(e.trace_type, []).append(e.duration_ms)

        stats = {
            "total": len(self._entries),
            "by_type": {
                t: {
                    "count": len(durations),
                    "avg_ms": round(sum(durations) / len(durations), 2),
                    "max_ms": round(max(durations), 2),
                    "min_ms": round(min(durations), 2),
                }
                for t, durations in by_type.items()
            },
        }
        return stats

    def export_json(self, filepath: Optional[str] = None) -> str:
        """Export all traces as JSON."""
        path = filepath or os.path.join(self._log_dir, "traces_export.json")
        data = [e.to_dict() for e in self._entries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Exported %d traces to %s", len(data), path)
        return path

    def clear(self) -> None:
        """Clear in-memory traces."""
        self._entries.clear()
        logger.debug("Traces cleared.")

    def disable(self) -> None:
        """Disable tracing (no-ops until re-enabled)."""
        self._enabled = False

    def enable(self) -> None:
        """Re-enable tracing."""
        self._enabled = True

    # ── Internal ────────────────────────────────────────────────

    def _write_to_file(self, entry: TraceEntry) -> None:
        """Write a trace entry to the daily log file."""
        filename = f"traces_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        filepath = os.path.join(self._log_dir, filename)
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning("Trace file write failed: %s", e)

    def _store_in_chroma(self, entry: TraceEntry) -> None:
        """Optionally store trace in Chroma trace collection."""
        if not self._chroma_initialised:
            self._init_chroma()
            self._chroma_initialised = True

        if self._chroma_collection is None:
            return

        try:
            self._chroma_collection.add(
                documents=[f"{entry.trace_type}: {entry.name}: {entry.output_summary}"],
                metadatas=[{
                    "trace_type": entry.trace_type,
                    "name": entry.name,
                    "duration_ms": entry.duration_ms,
                    "timestamp": entry.timestamp,
                }],
                ids=[entry.trace_id],
            )
        except Exception as e:
            logger.debug("Chroma trace store skipped: %s", e)

    def _init_chroma(self) -> None:
        """Initialise Chroma trace collection (lazy)."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=config.chroma_persist_dir)
            try:
                self._chroma_collection = client.get_collection(
                    name=config.chroma_collection_traces,
                )
            except Exception:
                self._chroma_collection = client.create_collection(
                    name=config.chroma_collection_traces,
                )
        except Exception as e:
            logger.debug("Chroma not available for traces: %s", e)
            self._chroma_collection = None
