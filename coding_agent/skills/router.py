"""
Stage 4: Semantic Intent Routing (BM25 + Chroma)
=================================================
Routes a user's natural-language intent to the best matching skill using
hybrid sparse (BM25) + dense (Chroma) retrieval.

If DeepSeek embedding is unavailable, falls back to sentence-transformers.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging
import json

from coding_agent.config import config

logger = logging.getLogger(__name__)


class SkillRouter:
    """
    Routes user intent → best skill using hybrid retrieval.

    Usage:
        router = SkillRouter()
        router.build_index()          # index the skill catalog
        result = router.route("create a new api endpoint")
        # → {"skill_name": "create_endpoint", "path": "backend.api_dev...",
        #     "params": {...}, "score": 0.92}
    """

    def __init__(self):
        self._bm25 = None
        self._chroma_collection = None
        self._embedding_fn = None
        self._skills_list: List[Dict[str, Any]] = []
        self._index_built = False

    # ── Public API ──────────────────────────────────────────────

    def build_index(self) -> None:
        """
        Build the BM25 + Chroma index from the skill catalog.
        Must be called before route().
        """
        from coding_agent.skills.catalog import flatten_catalog

        self._skills_list = flatten_catalog()
        if not self._skills_list:
            logger.warning("Skill catalog is empty, nothing to index.")
            return

        # Build BM25 index (sparse)
        self._build_bm25()

        # Build Chroma index (dense)
        self._build_chroma()

        self._index_built = True
        logger.info(
            "SkillRouter index built: %d skills indexed (BM25 + Chroma).",
            len(self._skills_list),
        )

    def route(self, intent: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """
        Route the user's intent to the best matching skill.

        Args:
            intent: Natural language intent (user query).
            top_k: How many candidates to consider from each retriever.

        Returns:
            Dict with keys: skill_name, path, description, params, score
            or None if no match found.
        """
        if not self._index_built:
            self.build_index()

        if not self._skills_list:
            return None

        # 1. BM25 scores
        bm25_scores = self._search_bm25(intent, top_k)

        # 2. Chroma scores (if available)
        chroma_scores = self._search_chroma(intent, top_k)

        # 3. Hybrid merge (normalised score sum)
        merged: Dict[str, float] = {}
        for path, score in bm25_scores:
            merged[path] = merged.get(path, 0) + score * 0.4   # BM25 weight
        for path, score in chroma_scores:
            merged[path] = merged.get(path, 0) + score * 0.6   # Chroma weight

        if not merged:
            # Fallback: keyword search
            from coding_agent.skills.catalog import search_skills
            fallback = search_skills(intent)
            if fallback:
                merged = {path: score for path, score in fallback[:top_k]}

        if not merged:
            logger.info("No matching skill found for intent: %s", intent[:60])
            return None

        # Pick the best
        best_path = max(merged, key=merged.get)
        best_score = merged[best_path]

        # Find the skill data
        skill_data = self._find_skill_by_path(best_path)
        if skill_data is None:
            return None

        logger.info(
            "Routed '%s' → %s (score=%.3f)",
            intent[:40], best_path, best_score,
        )

        return {
            "skill_name": skill_data.get("name", ""),
            "path": best_path,
            "description": skill_data.get("description", ""),
            "params": skill_data.get("parameters", []),
            "score": round(best_score, 4),
        }

    # ── BM25 (sparse) ───────────────────────────────────────────

    def _build_bm25(self) -> None:
        """Tokenise skill descriptions and build the BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 not installed. BM25 retrieval disabled. "
                "Install with: pip install rank-bm25"
            )
            self._bm25 = None
            return

        tokenized_corpus = [
            self._tokenize(
                f"{s.get('name', '')} {s.get('description', '')} "
                f"{s.get('domain', '')} {s.get('capability', '')} "
                f"{' '.join(s.get('examples', []))}"
            )
            for s in self._skills_list
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Search BM25 index. Returns [(path, score)]."""
        if self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Pair with paths
        path_scores = [
            (self._skills_list[i]["path"], float(scores[i]))
            for i in range(len(scores))
        ]
        # Sort descending
        path_scores.sort(key=lambda x: x[1], reverse=True)
        return path_scores[:top_k]

    # ── Chroma (dense) ──────────────────────────────────────────

    def _build_chroma(self) -> None:
        """Build Chroma collection from skill embeddings."""
        try:
            import chromadb
        except ImportError:
            logger.warning("chromadb not installed. Dense retrieval disabled.")
            self._chroma_collection = None
            return

        # Embedding function
        embed_fn = self._get_embedding_fn()
        if embed_fn is None:
            logger.warning("No embedding function available. Chroma disabled.")
            self._chroma_collection = None
            return

        try:
            client = chromadb.PersistentClient(path=config.chroma_persist_dir)
            # Delete existing collection if present (to rebuild)
            try:
                client.delete_collection(config.chroma_collection_skills)
            except Exception:
                pass
            collection = client.create_collection(
                name=config.chroma_collection_skills,
                embedding_function=embed_fn,
            )

            # Add all skills
            texts = []
            metadatas = []
            ids = []
            for i, s in enumerate(self._skills_list):
                text = (
                    f"{s.get('name', '')} {s.get('description', '')} "
                    f"{s.get('domain', '')} {s.get('capability', '')} "
                    f"{' '.join(s.get('examples', []))}"
                )
                texts.append(text)
                metadatas.append({
                    "path": s.get("path", ""),
                    "domain": s.get("domain", ""),
                    "capability": s.get("capability", ""),
                })
                ids.append(f"skill_{i}")

            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            self._chroma_collection = collection
            logger.info("Chroma skill index built with %d entries.", len(texts))

        except Exception as e:
            logger.warning("Failed to build Chroma index: %s", e)
            self._chroma_collection = None

    def _search_chroma(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Search Chroma index. Returns [(path, score)]."""
        if self._chroma_collection is None:
            return []

        try:
            results = self._chroma_collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self._skills_list)),
            )

            paths = []
            scores = []
            if results.get("metadatas") and results["metadatas"][0]:
                paths = [m.get("path", "") for m in results["metadatas"][0]]
            if results.get("distances") and results["distances"][0]:
                # Convert distance to similarity score (1 / (1 + distance))
                scores = [
                    1.0 / (1.0 + d) for d in results["distances"][0]
                ]
            return list(zip(paths, scores))
        except Exception as e:
            logger.warning("Chroma search error: %s", e)
            return []

    # ── Helpers ─────────────────────────────────────────────────

    def _get_embedding_fn(self):
        """
        Returns a callable embedding function.
        Priority: DeepSeek API → sentence-transformers.
        """
        # Try DeepSeek embedding first (via langchain-openai)
        if config.use_deepseek_embedding and config.deepseek_api_key:
            try:
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings(
                    model="deepseek-embedding",
                    api_key=config.deepseek_api_key,
                    base_url=config.deepseek_base_url,
                )

                # Wrap for Chroma
                class _DeepSeekEmbedder:
                    def __call__(self, input):
                        if isinstance(input, str):
                            input = [input]
                        return embeddings.embed_documents(input)

                logger.info("Using DeepSeek embedding.")
                return _DeepSeekEmbedder()

            except Exception as e:
                logger.warning("DeepSeek embedding failed: %s. Falling back.", e)

        # Fallback: sentence-transformers (local)
        try:
            from chromadb.utils import embedding_functions

            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.embedding_model,
            )
            logger.info("Using sentence-transformers embedding: %s", config.embedding_model)
            return ef
        except Exception as e:
            logger.warning("sentence-transformers not available: %s", e)
            return None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenisation."""
        import re
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    @staticmethod
    def _find_skill_by_path(path: str) -> Optional[Dict[str, Any]]:
        """Find a skill dict by its dotted path."""
        from coding_agent.skills.catalog import get_skill_by_path
        return get_skill_by_path(path)
