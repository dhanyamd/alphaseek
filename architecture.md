# AlphaSeek Architecture

## Overview

AlphaSeek is an LLM-driven quantitative research platform. A multi-agent team explores raw datasets, searches academic literature, designs experiments, writes and runs Python code in a hardened sandbox, and answers the user's research question — all without hardcoded assumptions about data schema, column names, or evaluation metrics.

---

## Pipeline (One Research Session)

```
User prompt
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│ 0. DATA PROFILER   Backend reads raw file (NPZ/CSV/...),    │
│                    LLM interprets metadata → DataReport      │
├──────────────────────────────────────────────────────────────┤
│ 1. LITERATURE      Researcher plans queries → OpenAlex       │
│                    search → arXiv full-text → PaperBriefs    │
├──────────────────────────────────────────────────────────────┤
│ 2. SYNTHESIST      ExperimentPlan from DataReport + briefs   │
│                    (Pydantic schema auto-injected in prompt)  │
├──────────────────────────────────────────────────────────────┤
│ 3. CODER           ONE Python script per round, runs in      │
│                    sandbox, calls af.submit(signal, fwd_arr)  │
├──────────────────────────────────────────────────────────────┤
│ 4. VISUALIZER      Plotly charts from manifest.npz arrays    │
├──────────────────────────────────────────────────────────────┤
│ 5. REPORTER        Answers the user's question in plain text │
└──────────────────────────────────────────────────────────────┘
```

Stages 0 and 1 run once per session. Stages 2-4 repeat for N rounds (configurable, default 8). Stage 5 runs once at the end.

---

## Directory Layout

```
alphaseek/
├── backend/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── agents.py         # 5 agent classes (Profiler→Reporter)
│   │   │   ├── llm.py            # LLM client (OpenAI + Anthropic, multi-provider failover)
│   │   │   ├── memory.py         # Experiment memory (flat list, no metric assumptions)
│   │   │   ├── orchestrator.py   # Research loop, yields SSE events
│   │   │   ├── reader.py         # Paper reading (PDF full-text → structured brief)
│   │   │   ├── research_tools.py # OpenAlex + arXiv search, PDF fetching
│   │   │   ├── evaluate.py       # unused (legacy)
│   │   │   └── rag.py            # unused (legacy)
│   │   ├── quant/
│   │   │   ├── dataset.py        # Dataset builder (yfinance/Polygon, env-configurable)
│   │   │   ├── docker_sandbox.py # Docker/fallback execution, artifact collection
│   │   │   ├── schemas.py        # DataReport + ExperimentPlan (Pydantic)
│   │   │   ├── provision.py      # Agent-declared pip packages → cached Docker layers
│   │   │   ├── convert.py        # No-op: raw uploads pass through, coder handles conversion
│   │   │   └── backtest.py       # FactorError only
│   │   ├── main.py               # FastAPI app: sessions, uploads, SSE stream, manual run
│   │   ├── pg.py                 # Postgres persistence
│   │   ├── store.py              # Facade → pg.py
│   │   ├── bus.py                # Event bus (Redis pub/sub + optional Redpanda)
│   │   ├── storage.py            # Artifact store (local disk + optional S3)
│   │   └── settings.py           # All env vars, single source of truth
│   └── data/                     # Cached dataset (market.npz, built by dataset.py)
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx              # Main workspace: sessions, editor, artifact viewer, event feed
│   │   └── layout.tsx            # Root layout
│   ├── components/
│   │   ├── CodeBlock.tsx         # Syntax-highlighted code display
│   │   ├── EquityChart.tsx       # Mini equity curve sparkline
│   │   ├── StrategyViewer.tsx    # Pine/MQL5 strategy display
│   │   └── TVChart.tsx           # TradingView-like chart widget
│   └── lib/
│       └── api.ts                # API client (REST + SSE)
│
├── sandbox/
│   ├── runner.py                 # In-container runner: alphaseek.* API, grading, artifact capture
│   └── Dockerfile                # Sandbox image (quant stack, no network at runtime)
│
└── architecture.md               # This file
```

---

## Key Design Decisions

### 1. The LLM Drives Everything

No hidden grading, no RiskCritic, no hardcoded schema. The LLM:
- **Profiles** the data (decides what each column means)
- **Synthesizes** the experiment plan (chooses methodology and evaluation)
- **Writes** all Python code (the sandbox only provides `af.submit()`)
- **Renders** charts (writes Plotly code)
- **Answers** the user (writes a summary)

### 2. The Sandbox Contract

The in-container runner (`sandbox/runner.py`) exposes a minimal API:

```python
import alphaseek as af

data = np.load(af.DATA)           # raw data — any key names
signal = ...                        # (T, N) signal
fwd = ...                           # forward returns from price columns
m = af.submit(signal, fwd_arr=fwd)  # metrics — fwd_arr REQUIRED, no hidden fallback

np.savez(f"{af.OUT}/manifest.npz", key1=arr1, ...)  # arrays for visualizer
```

- `af.submit()` with `fwd_arr=None` returns an error — no implicit forward returns
- The agent must compute `fwd` from price columns explicitly
- Metric names (`sharpe`, `mean_ic`, etc.) are mathematical computations, not LLM judgments
- The Docker sandbox has no network, runs as non-root, is read-only

### 3. Data Discovery at Runtime

- No `px_` prefix, no required keys (`tickers`, `dates`, `fwd`)
- The dataset builder (`dataset.py`) writes `close`, `volume`, `returns`, `fwd`, `tickers`, `dates` — but any uploaded file with different names works identically
- The profiler (`orchestrator.py:_profile`) reads raw metadata → sends to LLM → gets `DataReport`
- The coder prompt includes actual column names from the data report

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
| `ARTIFACT_S3_BUCKET` | S3 bucket for charts/manifests |
| `REDIS_URL` | Redis for live event streaming |
| `DOCKER_MEMORY` / `DOCKER_CPUS` | Sandbox resource limits |

---

## Data Flow (Docker Execution)

```
Agent code        →  /in/research.py (mounted ro)
Default dataset   →  /data/market.npz (mounted ro, via DATA_PATH env)
Uploaded file     →  /data/default (mounted ro, via DATA_PATH env)
Artifacts out     →  /out/ (mounted rw, copied to backend/artifacts/)
Manifest arrays   →  /out/manifest.npz (for viz stage)
```

`DATA_PATH` is set via `-e` in the Docker run command — `/data/market.npz` for the default dataset, `/data/default` for uploads.

---

## Removal History (Refactoring Log)

| Removed | Reason |
|---|---|
| `mode` (factor/general) | LLM reads user prompt and decides; no backend branch needed |
| `RiskCritic` | LLM judges results directly; no separate critic agent |
| `px_` prefix / reserved keys | Data keys are whatever the file contains — no prefix assumptions |
| Hidden `fwd` fallback | Explicit `fwd_arr` required; no silent default |
| `Exporter` class | Pine Script / MQL5 export was factor-mode-only and orphaned |
| `market.npz` hardcoded path | Configurable via `ALPHASEEK_DATASET_PATH` env var |
| `_RESERVED` / `_tolerant_load` | No key aliasing; agent discovers actual names at runtime |
| SQLite | Postgres only (removed db.py, facade → pg.py) |
