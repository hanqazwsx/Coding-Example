# coding_agent — Backend Engineering Automation Coding Agent

**coding_agent** is a self-developed automation coding scheduling framework with FSM reasoning, Tool Calling, Skill routing, hierarchical context compression, long/short-term memory, multi-agent cluster orchestration, and full security auditing. Designed for notebook environments with optimised resource usage.

## Core Capabilities

| Capability | Description |
|---|---|
| **FSM Reasoning Engine** | 6-state cognitive cycle: INIT → THINK → ACT → OBSERVE → REFLECT → DONE |
| **Tool Calling** | Pydantic-validated tool registry with unified `ToolResult` envelope |
| **Semantic Skill Routing** | Hybrid BM25 + Chroma retrieval to match user intent to skills |
| **Context Compression** | Sliding window + LLM summarization for token-efficient histories |
| **Static Content Cache** | In-memory cache replacing DeepSeek's unsupported prompt cache |
| **Short-Term Memory** | Bounded deque (FIFO) for conversation working memory |
| **Long-Term Memory** | Chroma vector store for compressed experiences across sessions |
| **Multi-Agent Orchestration** | Master/worker fork-join with ThreadPoolExecutor |
| **Security Audit** | Injection filtering, shell sandbox, command allowlist, audit trail |
| **Full-Link Tracing** | Timing instrumentation for all state transitions, tool calls, and routing |

## Technology Stack

- **Language:** Python 3.10+
- **Core:** langchain, langchain-openai, pydantic, tenacity
- **Vector Store:** chromadb (persistent or in-memory)
- **Retrieval:** rank-bm25 (sparse) + chromadb (dense)
- **Embedding:** sentence-transformers (local, default: `all-MiniLM-L6-v2`)
- **LLM Backend:** DeepSeek API (`deepseek-chat` via OpenAI-compatible interface)
- **Infrastructure:** Fully synchronous, thread-based parallelism (suitable for Jupyter notebooks)

## Directory Structure

```
coding_agent/
├── __init__.py              # Package version & doc
├── config.py                # Central settings (singleton, .env-backed)
├── main.py                  # Entry point: demo / interactive / eval modes
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
├── setup.bat                # Windows environment setup
├── setup.sh                 # Linux/Mac environment setup
│
├── core/                    # Core engine
│   ├── fsm.py               # Finite State Machine (6 states, configurable transitions)
│   ├── query_loop.py        # LLM-driven reasoning loop with tenacity retry
│   ├── compressor.py        # Hierarchical context compressor (sliding window + summary)
│   └── cache_manager.py     # Static content cache (system prompt, tool defs)
│
├── tools/                   # Tool Calling layer
│   ├── schema.py            # ToolSpec + ToolResult pydantic models
│   ├── registry.py          # Singleton tool registry (CRUD + 3 built-in tools)
│   └── executor.py          # Parameter validation, execution, pre/post hooks
│
├── skills/                  # Skill catalog and routing
│   ├── catalog.py           # 3-level catalog: domain > capability > skill (8 skills)
│   └── router.py            # BM25 + Chroma hybrid semantic intent router
│
├── memory/                  # Memory system
│   ├── short_term.py        # Bounded deque message buffer (FIFO)
│   ├── long_term.py         # Chroma vector store for experience retrieval
│   └── reflection.py        # DeepSeek-based reflection pipeline → LTM
│
├── agents/                  # Multi-agent orchestration
│   ├── orchestrator.py      # Master agent: task decomposition + fork-join
│   └── worker.py            # Worker agent: independent sub-task executor
│
├── security/                # Security layer
│   ├── filter.py            # Input injection detection (command, path traversal)
│   ├── sandbox.py           # Shell sandbox (command allowlist, timeout, cwd lock)
│   └── audit.py             # Append-only audit log (JSONL files)
│
└── monitoring/              # Instrumentation
    ├── tracer.py            # Full-link tracing (state transitions, tool calls, routing)
    └── eval.py              # Task evaluation suite (4 test tasks, metrics)
```

## Installation

### Prerequisites

- Python 3.10 or later
- pip

### Step 1: Create a virtual environment

**Windows:**
```batch
python -m venv venv
call venv\Scripts\activate.bat
```

**Linux / Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install --upgrade pip
pip install -r coding_agent/requirements.txt
pip install chromadb sentence-transformers
```

Alternatively, use the setup scripts:

```bash
# Windows
coding_agent\setup.bat

# Linux/Mac
bash coding_agent/setup.sh
```

### Step 3: Configure environment

```bash
cp coding_agent/.env.example coding_agent/.env
# Edit .env → set your DEEPSEEK_API_KEY
```

Example `.env`:

```ini
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
CHROMA_PERSIST_DIR=./chroma_db
LOG_LEVEL=INFO
```

## Quick Start

### Demo Mode (no API key required)

Run 3 pre-built demo scenarios showing FSM cycling, skill routing, and tool execution:

```bash
cd d:/MyTest
python -m coding_agent.main demo
```

### Interactive Mode (requires DeepSeek API key)

Full LLM-driven interactive session:

```bash
python -m coding_agent.main interactive
```

Example interaction:

```
You: Read the config.py file and summarize its contents
Agent: I'll read config.py and summarize it for you.
  [Routed: review_code]
  [Tool: read_file OK]

You: Create a new endpoint for user authentication
Agent: I'll help you create an authentication endpoint.
  [Routed: create_endpoint]
  [Tool: write_file OK]
```

### Evaluation Mode

Run 4 test tasks and produce a metrics report:

```bash
python -m coding_agent.monitoring.eval
```

Example output:

```
==================================================
EVALUATION REPORT
==================================================
  Tasks:    4/4 completed
  Rate:     100.0%
  Tool SR:  100.0%
  Avg turn: 3.2
  Avg dur:  40ms
  Total:    161ms
==================================================
```

## Core Module Explanations

### 1. FSM (Finite State Machine) — `core/fsm.py`

Defines a 6-state cognitive cycle:

```
INIT → THINK → ACT → OBSERVE → REFLECT → DONE
                 ↑________________________|
```

- Each `Transition` specifies a `from_state`, `to_state`, and optional `condition` (callable).
- The `FSM.transition(context)` method evaluates all transitions from the current state and picks the first valid one.
- Context dict (`_context`) accumulates state across transitions and is cleared on `reset()`.

### 2. Tool Calling — `tools/`

- **Registry**: Singleton holding `(ToolSpec, callable)` pairs. Built-in: `read_file`, `write_file`, `shell_exec`.
- **Schema**: `ToolSpec` defines parameters (name, type, required). `ToolResult` is the unified return envelope: `{success, result, error, duration_ms, tool_name, metadata}`.
- **Executor**: Validates params against `ToolSpec`, runs pre/post hooks (used for security filtering), catches exceptions, and returns `ToolResult`.

### 3. Skill Router — `skills/`

- **Catalog**: 3-level hierarchy — 4 domains (backend, frontend, data, devops), 6 capabilities, 8 skills.
- **Router**: Hybrid retrieval pipeline:
  1. **BM25** (sparse): keyword matching via `rank_bm25`, weight 0.4
  2. **Chroma** (dense): vector similarity via `sentence-transformers`, weight 0.6
  3. Merged score → returns best skill + parameters suggestion

### 4. Context Compression — `core/compressor.py`

- Keeps the last N (default 6) message turns in full fidelity.
- Older messages are fed to DeepSeek for summarization.
- The compressed summary is injected as a `system`-role message in the LLM context.

### 5. Memory System — `memory/`

- **Short-Term**: Python `deque(maxlen=50)` — FIFO eviction, supports `get_recent(k)`.
- **Long-Term**: ChromaDB collection storing vectorised experience summaries.
- **Reflection Pipeline**: When short-term memory reaches 20 messages, calls DeepSeek to produce a structured reflection (goal, actions, outcome, lessons), stores in long-term memory, and clears short-term (keeping last 4 messages for continuity).

### 6. Multi-Agent — `agents/`

- **Orchestrator**: Uses DeepSeek to decompose a high-level task into 2-4 subtasks (JSON output), then dispatches them via `ThreadPoolExecutor(max_workers=4)`.
- **Worker**: Each worker has its own `FSM`, `QueryLoop`, and `ShortTermMemory`. Results collected via `as_completed()`.

### 7. Security — `security/`

- **InputFilter**: Regex-based detection of command injection (`;`, `|`, `` ` ``, `$()`), path traversal (`../`).
- **ShellSandbox**: Command allowlist (default: `ls, cat, grep, find, head, tail, wc, echo, sort, uniq, cut, tr`), blocks network commands (`curl, wget, nc, ssh`), validates ALL chained segments (`;`, `&&`, `||`, `|`), forces cwd to project root, enforces 30s timeout.
- **AuditLogger**: Append-only JSONL log of all user inputs, tool calls, state transitions, and security violations. Auto-flushes every 100 events.

### 8. Monitoring — `monitoring/`

- **Tracer**: Context manager + decorator for measuring operation durations. Stores traces in daily JSONL files and optionally in Chroma's `trace_logs` collection.
- **Evaluator**: Runs predefined test tasks, computes task completion rate, tool call success rate, average turns, and average duration. Outputs a JSON report.

## Configuration Reference

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (required for LLM mode) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API endpoint (OpenAI-compatible) |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Configurable Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `llm_temperature` | `0.3` | LLM sampling temperature |
| `llm_max_tokens` | `4096` | Max tokens per LLM response |
| `fsm_max_iterations` | `15` | Max FSM cycles per task |
| `query_loop_max_retries` | `3` | Tenacity retry count for LLM calls |
| `compressor_keep_last_n` | `6` | Sliding window size (messages kept raw) |
| `short_term_maxlen` | `50` | Max short-term memory messages |
| `reflection_threshold` | `20` | Messages before auto-reflection |
| `orchestrator_max_workers` | `4` | Max concurrent worker agents |
| `tool_whitelist` | `[read_file, write_file, shell_exec]` | Allowed tools |
| `tool_blacklist` | `[]` | Blocked tools |
| `shell_allowed_commands` | `[ls, cat, grep, ...]` | Shell sandbox allowlist |

## Known Limitations & Notes

### Notebook Resource Optimisation

- **ThreadPoolExecutor** limited to 4 workers (avoids notebook resource contention).
- **ChromaDB** runs in persistent mode (supports in-memory for notebooks without disk I/O).
- **Short-term memory** capped at 50 messages; long-term memory only stores compressed reflections.
- All LLM calls use **tenacity** with exponential backoff (max 3 retries).

### DeepSeek Prompt Cache Alternative

DeepSeek does not support Anthropic-style `cache_control` headers. The `cache_manager.py` module provides an in-memory replacement:

- System prompts, tool definitions, and other static content are built once and cached in a dict.
- `get_or_build(key, builder_fn)` avoids repeated construction.
- Hit/miss statistics available via `stats()`.
- Content fingerprinting (SHA-256) available for manual cache invalidation.

### Shell Sandbox Bypass Protections

The sandbox (`security/sandbox.py`) protects against:

| Attack Vector | Protection |
|---|---|
| Command chaining via `;` | All segments validated individually |
| Command chaining via `&&`/`||` | All segments validated individually |
| Pipes with unsafe commands | All pipe segments validated |
| Network commands | Substring match against blocked patterns |
| Path escape | `cwd` locked to project root |
| Runaway processes | 30s timeout enforced |
| Encoded commands | Not currently detected (future work) |

### Demo Mode Limitations

The FSM-only demo mode routes skill catalog entries to actual registered tools via a static mapping (`_skill_to_tool_map`). Parameters are not supplied in demo mode, so tool calls will fail with validation errors. This is expected — the demo demonstrates the control flow (FSM cycling, skill routing, security hooks) without requiring an LLM.

## Full Example Run

Below is a trace from an interactive session. This demonstrates the complete pipeline:

```
User: List all Python files in the project

  [Security Filter] → Injection check: PASS
  [Skill Router]    → write_dockerfile (score=0.807) [Note: best available match]
  [FSM] INIT → THINK → ACT → OBSERVE → REFLECT → DONE
  [Tool Executor]   → shell_exec("ls *.py") → [OK] (57ms)
  [Audit]          → logged as tool_call #3
  [Tracer]         → state_transition: 5 iterations, 15 transitions
  [Memory]         → Short-term: 2 messages | Long-term: 0 experiences

Agent: Found 8 Python files. Here's the summary:
  - main.py: Application entry point
  - config.py: Central configuration
  - core/fsm.py: FSM engine
  ...
```

## Code Logic Audit Report (Resolved)

After a thorough audit of all 29 source files, the following issues were identified and fixed:

| # | File | Issue | Severity | Status |
|---|---|---|---|---|
| 1 | `core/fsm.py` | FSM context dict not cleared on `reset()` — cross-task data bleed | HIGH | Fixed |
| 2 | `security/sandbox.py` | Allowlist bypass via `;`/`&&` command chaining | HIGH | Fixed |
| 3 | `security/filter.py` | Single-semicolon `;` not detected as injection | HIGH | Fixed |
| 4 | `agents/worker.py` | Broken `@lc_tool` decorator wrapping tools | HIGH | Fixed |
| 5 | `main.py` | FSM-only demo bypassed ShellSandbox | MEDIUM | Fixed |
| 6 | `monitoring/eval.py`, `main.py` | Unicode chars crashed on Windows CP-437/1252 | MEDIUM | Fixed |
| 7 | `main.py` | Skill names don't match registered tool names | MEDIUM | Fixed |
| 8 | Multiple files | Unused imports (`State`, `search_skills`, `json`, `builtins`) | LOW | Fixed |
| 9 | `tools/schema.py` | `to_dict()` dropped `metadata` field | LOW | Fixed |
| 10 | `config.py` | `basicConfig` at module level (side effect) | LOW | Fixed |

No issues found in: async/sync compatibility, DeepSeek API format consistency, or circular imports.
