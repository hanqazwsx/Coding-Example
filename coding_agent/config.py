"""
Stage 1 / Stage 8: Central Configuration
=========================================
Centralizes all configurable parameters: API keys, paths, limits, security settings.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── LLM / DeepSeek ──────────────────────────────────────────
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    llm_request_timeout: int = 120

    # ── ChromaDB ────────────────────────────────────────────────
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    chroma_collection_skills: str = "skill_vectors"
    chroma_collection_memory: str = "long_term_memory"
    chroma_collection_traces: str = "trace_logs"

    # ── FSM / Query Loop ────────────────────────────────────────
    fsm_max_iterations: int = 15
    query_loop_max_retries: int = 3
    query_loop_retry_min_wait: float = 1.0
    query_loop_retry_max_wait: float = 10.0

    # ── Context Compression (Stage 5) ──────────────────────────
    compressor_keep_last_n: int = 6          # sliding window: keep last N raw messages
    compressor_summary_model: str = "deepseek-chat"

    # ── Memory (Stage 6) ────────────────────────────────────────
    short_term_maxlen: int = 50
    reflection_threshold: int = 20           # trigger reflection after N messages
    long_term_search_k: int = 5

    # ── Multi-Agent (Stage 7) ───────────────────────────────────
    orchestrator_max_workers: int = 4
    worker_max_iterations: int = 10

    # ── Security (Stage 8) ──────────────────────────────────────
    tool_whitelist: List[str] = field(default_factory=lambda: [
        "read_file", "write_file", "shell_exec",
    ])
    tool_blacklist: List[str] = field(default_factory=list)
    shell_allowed_commands: List[str] = field(default_factory=lambda: [
        "ls", "cat", "grep", "find", "head", "tail", "wc",
        "echo", "sort", "uniq", "cut", "tr",
    ])
    shell_blocked_patterns: List[str] = field(default_factory=lambda: [
        "curl", "wget", "nc", "ncat", "ssh", "telnet",
        "python -c", "bash -c", "sh -c",
    ])
    project_root: str = field(
        default_factory=lambda: os.path.abspath(os.path.dirname(__file__))
    )

    # ── Logging ─────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    # ── Embedding ───────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_deepseek_embedding: bool = False


# Global singleton
config = Config()


def setup_logging() -> None:
    """Initialise root logger (call once at application entry point)."""
    if logging.root.handlers:
        return  # already configured
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

