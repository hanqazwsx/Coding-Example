"""
Stage 6: Long-Term Memory
==========================
Stores compressed experiences in ChromaDB as vector embeddings.
Each "experience" is a text summary of a completed conversation or
significant event, vectorised for similarity-based retrieval.

Supports:
  - add_experience(summary, metadata) → stored as a Chroma document
  - search(query, k) → returns most semantically similar past experiences
  - get_all() → list all stored experiences for inspection
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import time
import uuid

from coding_agent.config import config

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Vector-based long-term memory using ChromaDB.

    Each memory entry is a dict with:
        id, summary (text), timestamp, metadata (dict)

    Args:
        persist_dir: ChromaDB persistence directory.
        collection_name: Name of the Chroma collection.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = config.chroma_collection_memory,
    ):
        self._persist_dir = persist_dir or config.chroma_persist_dir
        self._collection_name = collection_name
        self._collection = None
        self._initialised = False

    # ── Initialisation ──────────────────────────────────────────

    def _ensure_initialised(self) -> None:
        """Lazy init of ChromaDB (avoids import errors at module level)."""
        if self._initialised:
            return

        try:
            import chromadb
        except ImportError:
            logger.warning("chromadb not available. Long-term memory disabled.")
            self._initialised = True  # mark as "attempted"
            return

        try:
            client = chromadb.PersistentClient(path=self._persist_dir)
            # Try to get existing collection, create if not found
            try:
                self._collection = client.get_collection(
                    name=self._collection_name,
                )
                logger.info(
                    "LTM: loaded existing collection '%s' (%d entries)",
                    self._collection_name,
                    self._collection.count(),
                )
            except Exception:
                self._collection = client.create_collection(
                    name=self._collection_name,
                )
                logger.info(
                    "LTM: created collection '%s'", self._collection_name
                )

            # Try to use sentence-transformers embedding if available
            try:
                from chromadb.utils import embedding_functions
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=config.embedding_model,
                )
                # Can't change embedding fn on existing collection easily;
                # just note it
                logger.debug("LTM embedding available: %s", config.embedding_model)
            except Exception:
                logger.debug("LTM using Chroma default embedding (all-MiniLM-L6-v2)")

        except Exception as e:
            logger.warning("Failed to initialise LTM Chroma: %s", e)
            self._collection = None

        self._initialised = True

    # ── Public API ──────────────────────────────────────────────

    def add_experience(
        self,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a compressed experience as a vector entry.

        Args:
            summary: Text summary of the experience.
            metadata: Optional dict (e.g. {"task": "API endpoint", "success": true}).

        Returns:
            The entry ID, or None if storage failed.
        """
        self._ensure_initialised()
        if self._collection is None:
            logger.warning("LTM not available, cannot store experience.")
            return None

        entry_id = str(uuid.uuid4())
        meta = {
            "timestamp": time.time(),
            "type": "experience",
            **(metadata or {}),
        }

        try:
            self._collection.add(
                documents=[summary],
                metadatas=[meta],
                ids=[entry_id],
            )
            logger.debug("LTM add: %s (len=%d chars)", entry_id[:8], len(summary))
            return entry_id
        except Exception as e:
            logger.warning("LTM add failed: %s", e)
            return None

    def search(
        self,
        query: str,
        k: int = config.long_term_search_k,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the k most similar past experiences.

        Args:
            query: Natural language query.
            k: Number of results.

        Returns:
            List of dicts: [{id, summary, metadata, score}, ...]
        """
        self._ensure_initialised()
        if self._collection is None:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, max(1, self._collection.count() or 1)),
            )

            entries = []
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(ids)):
                score = 1.0 / (1.0 + distances[i]) if distances else 0.0
                entries.append({
                    "id": ids[i],
                    "summary": documents[i] if documents else "",
                    "metadata": metadatas[i] if metadatas else {},
                    "score": round(score, 4),
                })

            return entries

        except Exception as e:
            logger.warning("LTM search failed: %s", e)
            return []

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all stored experiences (for inspection)."""
        self._ensure_initialised()
        if self._collection is None:
            return []

        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.get(limit=count)
            entries = []
            for i in range(len(results["ids"])):
                entries.append({
                    "id": results["ids"][i],
                    "summary": (results["documents"] or [""])[i] if results.get("documents") else "",
                    "metadata": (results["metadatas"] or [{}])[i] if results.get("metadatas") else {},
                })
            return entries
        except Exception as e:
            logger.warning("LTM get_all failed: %s", e)
            return []

    def count(self) -> int:
        """Number of stored experiences."""
        self._ensure_initialised()
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all experiences from long-term memory."""
        self._ensure_initialised()
        if self._collection is None:
            return
        try:
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)
            logger.info("LTM cleared (%d entries removed)", len(all_ids))
        except Exception as e:
            logger.warning("LTM clear failed: %s", e)
