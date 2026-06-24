"""
Stage 5: Static Content Cache (Prompt Cache Alternative)
=========================================================
DeepSeek does not support prompt cache_control headers.
This module provides an in-memory cache that stores static content
(system prompts, tool definitions, skill catalog, etc.) and avoids
reconstructing them on each call.

Simulates the effect of prompt caching by:
  1. Pre-building and storing static content in memory.
  2. Returning cached content via get_or_build().
  3. Tracking hit/miss statistics.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class CacheEntry:
    """A single cache entry with metadata."""

    def __init__(self, key: str, value: Any, description: str = ""):
        self.key = key
        self.value = value
        self.description = description
        self.hits = 0
        self.created_at = __import__("time").time()

    def __repr__(self) -> str:
        return f"CacheEntry(key={self.key}, hits={self.hits})"


class CacheManager:
    """
    In-memory cache for static content that does not change during a session.

    Thread-safe for read operations (writes are single-threaded during init).

    Usage:
        cache = CacheManager()
        system_prompt = cache.get_or_build(
            "system_prompt",
            builder=lambda: "You are a helpful assistant...",
        )
        tools_def = cache.get_or_build(
            "tool_defs",
            builder=lambda: json.dumps(tool_specs),
        )
        stats = cache.stats()
    """

    def __init__(self):
        self._store: Dict[str, CacheEntry] = {}
        self._misses = 0

    # ── Core API ────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """
        Get a cached value by key. Returns None if not found.
        Increments hit counter if found.
        """
        entry = self._store.get(key)
        if entry is not None:
            entry.hits += 1
            return entry.value
        self._misses += 1
        return None

    def set(self, key: str, value: Any, description: str = "") -> None:
        """Store a value in the cache."""
        self._store[key] = CacheEntry(key=key, value=value, description=description)
        logger.debug("Cache SET: %s (%s)", key, description)

    def get_or_build(
        self,
        key: str,
        builder: Callable[[], Any],
        description: str = "",
        force_rebuild: bool = False,
    ) -> Any:
        """
        Return cached value if available; otherwise build, cache, and return.

        Args:
            key: Cache key.
            builder: Zero-argument callable that produces the value.
            description: Human-readable description for logging.
            force_rebuild: If True, always call builder and overwrite cache.

        Returns:
            The cached or freshly-built value.
        """
        if not force_rebuild:
            cached = self.get(key)
            if cached is not None:
                return cached

        # Miss: build and cache
        value = builder()
        self.set(key, value, description=description)
        return value

    def invalidate(self, key: str) -> None:
        """Remove a single key from the cache."""
        self._store.pop(key, None)
        logger.debug("Cache invalidated: %s", key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()
        self._misses = 0
        logger.info("Cache cleared.")

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_entries = len(self._store)
        total_hits = sum(e.hits for e in self._store.values())
        return {
            "entries": total_entries,
            "hits": total_hits,
            "misses": self._misses,
            "hit_rate": round(
                total_hits / (total_hits + self._misses), 4
            ) if (total_hits + self._misses) > 0 else 0,
            "keys": list(self._store.keys()),
        }

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"CacheManager(entries={stats['entries']}, "
            f"hit_rate={stats['hit_rate']})"
        )


# ── Helper: content fingerprinting ─────────────────────────────────

def fingerprint(content: Any) -> str:
    """
    Generate a SHA-256 fingerprint of arbitrary content.
    Useful for cache invalidation when content might have changed.
    """
    raw = json.dumps(content, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
