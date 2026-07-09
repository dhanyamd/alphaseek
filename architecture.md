# AlphaSeek Architecture

## Overview

AlphaSeek is an LLM-driven quantitative research platform. A multi-agent team explores raw datasets, searches academic literature, designs experiments, writes and runs Python code in a hardened sandbox, and answers the user's research question — all without hardcoded assumptions about data schema, column names, or evaluation metrics.

---

## Pipeline (One Research Session)

```
User prompt
  |
  v
+------------------------------------------------------------------+
| 0. DATA PROFILER   Backend reads raw file (NPZ/CSV/PARQUET/...), |
|                    LLM interprets metadata -> DataReport          |
+------------------------------------------------------------------+
| 1. LITERATURE      Researcher plans queries -> OpenAlex          |
|                    search -> arXiv full-text -> PaperBriefs       |
+------------------------------------------------------------------+
| 2. SYNTHESIST      ExperimentPlan from DataReport + briefs       |
|                    (Pydantic schema auto-injected in prompt)      |
+------------------------------------------------------------------+
| 3. CODER           ONE Python script per round, runs in          |
|                    sandbox, prints JSON results to stdout,        |
|                    saves arrays to af.OUT/*.npz                   |
+------------------------------------------------------------------+
| 4. VISUALIZER      Plotly charts from whatever .npz keys exist   |
+------------------------------------------------------------------+
| 5. EXPORTER        Pine Script / MQL5 (only if prompt asks)      |
+------------------------------------------------------------------+
| 6. REPORTER        Answers the user's question in plain text     |
+------------------------------------------------------------------+
```

Stages 0-1 run once. Stages 2-4 repeat for N rounds (configurable, default 8). Stages 5-6 run once at the end.

---

## Directory Layout

```
alphaseek/
+-- backend/
|   +-- app/
|   |   +-- agent/
|   |   |   +-- agents.py         # 7 agent classes (Profiler->Reporter)
|   |   |   +-- llm.py            # LLM client (OpenAI + Anthropic, multi-provider failover)
|   |   |   +-- memory.py         # Experiment memory (flat list, no metric assumptions)
|   |   |   +-- orchestrator.py   # Research loop, yields SSE events
|   |   |   +-- reader.py         # Paper reading (PDF full-text -> structured brief)
|   |   |   +-- research_tools.py # OpenAlex + arXiv search, PDF fetching
|   |   |   +-- evaluate.py       # unused (legacy)
|   |   |   +-- rag.py            # unused (legacy)
|   |   +-- quant/
|   |   |   +-- dataset.py        # Dataset builder (yfinance/Polygon, env-configurable)
|   |   |   +-- docker_sandbox.py # Docker/fallback execution, artifact collection
|   |   |   +-- schemas.py        # DataReport + ExperimentPlan (Pydantic)
|   |   |   +-- provision.py      # Agent-declared pip packages -> cached Docker layers
|   |   |   +-- convert.py        # No-op: raw uploads pass through, coder handles conversion
|   |   |   +-- backtest.py       # FactorError only
|   |   +-- main.py               # FastAPI app: sessions, uploads, SSE stream, manual run
|   |   +-- pg.py                 # Postgres persistence
|   |   +-- store.py              # Facade -> pg.py
|   |   +-- bus.py                # Event bus (Redis pub/sub + optional Redpanda)
|   |   +-- storage.py            # Artifact store (local disk + optional S3)
|   |   +-- settings.py           # All env vars, single source of truth
|   +-- data/                     # Cached dataset (market.npz, built by dataset.py)
|
+-- frontend/
|   +-- app/
|   |   +-- page.tsx              # Main workspace: sessions, editor, artifact viewer, event feed
|   |   +-- layout.tsx            # Root layout
|   +-- components/
|   |   +-- CodeBlock.tsx         # Syntax-highlighted code display
|   |   +-- EquityChart.tsx       # Mini equity curve sparkline
|   |   +-- StrategyViewer.tsx    # Pine/MQL5 strategy display
|   |   +-- TVChart.tsx           # TradingView-like chart widget
|   +-- lib/
|       +-- api.ts                # API client (REST + SSE)
|
+-- sandbox/
|   +-- runner.py                 # Thin executor: af.DATA / af.OUT / af.uploads(), captures stdout
|   +-- Dockerfile                # Sandbox image (quant stack, no network at runtime)
|
+-- architecture.md               # This file
```

---

## Key Design Decisions

### 1. The LLM Drives Everything

No hidden grading, no RiskCritic, no hardcoded schema. The LLM:
- **Profiles** the data (decides what each column means)
- **Synthesizes** the experiment plan (chooses methodology and evaluation)
- **Writes** all Python code (the sandbox provides no metric helpers)
- **Renders** charts (writes Plotly code)
- **Exports** strategies (Pine Script / MQL5 when asked)
- **Answers** the user (writes a summary)

### 2. The Sandbox Contract

The in-container runner (`sandbox/runner.py`) is intentionally thin. It provides:

```python
import alphaseek as af

af.DATA        # path to the dataset file (agent loads + inspects it)
af.OUT         # directory to save artifacts / arrays (agent picks names)
af.uploads()   # paths to any user-uploaded files
```

There is **no** `af.submit`, **no** `af.manifest`, **no** metric computation.
The agent is expected to:
- load `af.DATA` and inspect keys/columns
- compute forward returns from whatever price column exists
- evaluate the strategy however the research question demands
- save arrays for visualization to `af.OUT` (e.g. `np.savez(f"{af.OUT}/result.npz", ...)`)
- print a single JSON result line (metrics / conclusions) to stdout

The Docker sandbox runs with `--network none`, as non-root, with a read-only filesystem (except `af.OUT`).

### 3. Data Discovery at Runtime

- No `px_` prefix, no required keys (`tickers`, `dates`, `fwd`, `close`)
- The dataset builder (`dataset.py`) writes standard names, but **any uploaded file with different names works identically**
- The profiler reads raw file metadata -> sends to LLM -> gets `DataReport`
- The coder prompt includes actual column names from the data report (not placeholders)

### 4. Multi-Provider LLM

`backend/app/agent/llm.py` supports:
- OpenAI-compatible APIs (Groq, OpenRouter, etc.) + Anthropic native SDK
- Primary + secondary provider with per-role model routing
- Automatic failover on rate limits / errors
- JSON mode when the provider supports it

### 5. Streaming Events

Every pipeline step yields a typed event dict that is:
1. Persisted to Postgres (JSONB)
2. Published to Redis pub/sub (or Redpanda for durable replay)
3. Streamed to the frontend via SSE

The frontend reconstructs the session on refresh by replaying stored events, then tails the live bus.

---

## Configuration

All via environment variables in `backend/.env`:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | Primary LLM provider |
| `LLM_API_KEY_2` / `LLM_BASE_URL_2` / `LLM_MODELS_2` | Secondary LLM provider |
| `LLM_MODEL_<ROLE>` | Per-role model overrides |
| `ALPHASEEK_DATASET_PATH` | Path to default `.npz` dataset |
| `ALPHASEEK_UNIVERSE` | Ticker list for dataset builder |
| `ALPHASEEK_DATA_SOURCE` | `yfinance` (default) or `polygon` |
| `POLYGON_API_KEY` | Polygon.io API key |
| `EXA_API_KEY` | Web search for literature |
| `ARTIFACT_S3_BUCKET` | S3 bucket for artifacts/charts |
| `REDIS_URL` | Redis for live event streaming |
| `DOCKER_MEMORY` / `DOCKER_CPUS` | Sandbox resource limits |

---

## Data Flow (Docker Execution)

```
Agent code        ->  /in/research.py (mounted ro)
Default dataset   ->  /data/market.npz (mounted ro, via DATA_PATH env)
Uploaded file     ->  /data/default (mounted ro, via DATA_PATH env)
Artifacts out     ->  /out/ (mounted rw, copied to backend/artifacts/)
Arrays for viz    ->  /out/*.npz (agent picks file/key names)
```

`DATA_PATH` is set via `-e` in the Docker run command — `/data/market.npz` for the default dataset, `/data/default` for uploads.

---

## Removal History (Refactoring Log)

| Removed | Reason |
|---|---|
| `mode` (factor/general) | LLM reads user prompt and decides; no backend branch needed |
| `RiskCritic` | LLM judges results directly; no separate critic agent |
| `px_` prefix / reserved keys | Data keys are whatever the file contains -- no prefix assumptions |
| `af.submit()` / `af.manifest()` | Runner is thin; LLM computes all metrics itself |
| `manifest.npz` convention | Visualizer globs `af.OUT/*.npz` dynamically |
| Hidden `fwd` fallback | No implicit forward returns; agent computes them explicitly |
| `market.npz` hardcoded path | Configurable via `ALPHASEEK_DATASET_PATH` env var |
| `_RESERVED` / `_tolerant_load` | No key aliasing; agent discovers actual names at runtime |
| SQLite | Postgres only (removed db.py, facade -> pg.py) |
