# AlphaSeek

An AI quant-research platform: a multi-agent team autonomously reads finance
papers, invents and tests a trading strategy, or performs general quantitative
analysis — writing and self-repairing real code against real market data in a
hardened sandbox. Two research modes: **Factor Research** (signal → backtest →
blind grade) and **General Analysis** (regression, clustering, portfolio opt,
statistics). Results stream live to a web IDE.

---

## System architecture

```
   Browser ───── prompt + mode ─────► FastAPI ──── SSE/WS stream ────► UI
      ▲                                       │
      └─── live events (Redis pub/sub ────────┘
            or Redpanda durable log)

                        │  dispatches by mode
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  RESEARCH PIPELINE (shared literature stage, mode-specific rounds)    │
 │                                                                       │
 │  Researcher ─► Synthesist ─┬── factor ──► Coder ─► Viz ─► Critic ─┐ │
 │      │          ^          │                          Archivist ─►│ │
 │  OpenAlex       │           └── general ─► Coder ─► Viz ─►          │ │
 │  arXiv/Jina     │                                    Archivist ─►   │ │
 │              memory (per-session state)                  Reporter ──┘ │
 │                                                           │           │
 │  Docker sandbox: agent writes ALL code                    │ (Exporter │
 │  af.submit(signal) → blind grade (factor)                 │  to Pine/ │
 │  af.backtest() / print results (general)                  │  MQL5 for │
 └──────────────────────┬────────────────────────────────────  factor)  ┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   Postgres (log)  S3 / local     Qdrant (RAG
                   disk (charts)   paper corpus)
```



**Flow:** The user selects Factor Research or General Analysis. After literature
review (Researcher → paper fetch → Qdrant RAG), each round runs:
- **Factor mode:** Synthesist → Coder (`af.submit`) → Visualizer → RiskCritic → Archivist → Exporter
- **General mode:** Synthesist → Coder (`print results`) → Visualizer → Archivist

Every event streams live via Redis pub/sub or Redpanda and is durably logged to
Postgres. Charts go to S3/local disk.

**Design principles**
- **Autonomous coder, minimal contract.** The sandbox exposes `af.DATA`,
  `af.OUT`, `af.backtest`, `af.submit`, `af.manifest`, `af.uploads`. The agent
  writes *every* feature, estimator, and signal itself — no DSL, no
  pre-computed features, no hardcoded strategy logic.
- **Two pipelines, no branching inside stages.** Mode selects which stages run;
  no stage has `if mode == "factor"` logic.
- **Grading integrity (factor mode).** Forward returns are never exposed;
  `af.submit` grades against hidden data, look-ahead guard rejects
  future-peeking signals.
- **Math first, visuals last.** Coder saves a manifest (`np.savez`); Visualizer
  loads and plots — a chart bug can't lose a result.
- **Nothing canned.** No mocks, no silent fallbacks — every failure visible.

---

## Tech stack

| Layer | Technology |
|---|---|---|
| Frontend | Next.js, CodeMirror, Geist |
| API | FastAPI, SSE + WebSockets |
| Persistence | **Postgres** (JSONB event log, CQRS) |
| Streaming | **Redis** (pub/sub for live UI) + **Redpanda** (durable log, replay, consumer groups) |
| Vector store / RAG | **Qdrant** (Cloud or local) + fastembed (ONNX, no torch) |
| Object storage | **S3 + presigned URLs** (local disk fallback) |
| LLM routing | multi-provider client + optional **LiteLLM gateway** |
| Sandbox | Docker (`--network none`), quant stack: numpy, pandas, scipy, scikit-learn, statsmodels, empyrical, arch, cvxpy, quantstats, plotly, riskfolio-lib, skfolio, pyportfolioopt |
| Literature | OpenAlex / Semantic Scholar / arXiv + Jina Reader |
| Data | **yfinance** (default, no key) or **Polygon.io** (free tier, 15-min delayed) + user uploads (CSV, Parquet, Excel, JSON, NPZ) |

**Backend layout**
```
backend/app/
  main.py          FastAPI: sessions, SSE + WebSocket streaming, uploads, artifacts
  settings.py      one place all service URLs/creds/toggles resolve from env
  store.py         persistence facade → pg.py (Postgres)
  pg.py            Postgres: users, sessions, JSONB event log
  bus.py           dual-transport event bus: Redis pub/sub + Redpanda
  storage.py       artifact store (local + S3)
  agent/
    orchestrator.py  runs the multi-agent pipeline, dispatches factor/general rounds
    agents.py        Researcher, Synthesist, CodingAgent (mode-aware), Visualizer,
                     RiskCritic, Reporter, Exporter
    evaluate.py      statistical grading (DSR, PSR, bootstrap) — factor mode only
    memory.py        per-session Memory (what was tried, what worked)
    llm.py           multi-provider client (routing / failover / pacing)
    rag.py           Qdrant agentic RAG (index / search / retrieve→judge→refine)
    reader.py        PaperQA-style reader (chunk → embed-rank → 1 LLM reduce)
    research_tools.py OpenAlex/S2/arXiv search + Jina full-text
  quant/
    dataset.py       yfinance → raw market.npz (env-configurable universe)
    convert.py       user-uploaded files → NPZ (pandas-based column detection)
    docker_sandbox.py runs agent code in a hardened container + provisioning
    provision.py     agent-declared pip deps → cached, allowlisted image layers
sandbox/
  runner.py          the `alphaseek` module (6-item contract) + grading + LA guard
  Dockerfile         base image with the quant + portfolio stack
docker-compose.yml   Redis, Postgres, Qdrant, Redpanda, LiteLLM
litellm-config.yaml  LLM gateway routing/failover
```

---

## How to run

### Prerequisites
Python 3.12, Node 18+, Docker, `uv`.

### 1. Backend deps
```bash
cd backend
uv venv && uv pip install -r requirements.txt
cp .env.example .env            # then set LLM_API_KEY etc.
```
Minimum config in `backend/.env`:
```
LLM_API_KEY=...                 # e.g. a Gemini AI Studio key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-2.5-flash
# optional secondary bucket for failover:
LLM_API_KEY_2=...  LLM_BASE_URL_2=https://openrouter.ai/api/v1  LLM_MODELS_2=openai/gpt-oss-120b:free
```

### 2. Build the sandbox image (once)
```bash
cd sandbox
docker build -t alphaseek-sandbox:base .
docker tag alphaseek-sandbox:base alphaseek-sandbox:latest
```

### 3. Infrastructure (all free, local Docker)
```bash
docker compose up -d            # Redis, Postgres, Qdrant, Redpanda, LiteLLM
docker compose ps               # all healthy
```
Then enable the pieces you want in `backend/.env`:
```
DATABASE_URL=postgresql://alphaseek:alphaseek@localhost:5432/alphaseek   # uses Postgres
REDIS_URL=redis://localhost:6379/0
# Redpanda for durable event log (replaces Redis pub/sub for streaming):
USE_REDPANDA_BUS=1
REDPANDA_BROKERS=localhost:9092
# Qdrant Cloud (or omit for local Docker Qdrant):
QDRANT_CLUSTER_ENDPOINT=https://<cluster>.cloud.qdrant.io
QDRANT_API_KEY=...
# S3 artifacts (or omit for local disk):
ARTIFACT_S3_BUCKET=my-bucket   AWS_REGION=us-east-1
```
To route LLM calls through the gateway, put `GEMINI_API_KEY` / `OPENROUTER_API_KEY`
in a repo-root `.env`, `docker compose up -d litellm`, then set
`LLM_BASE_URL=http://localhost:4000` in `backend/.env`.

### 4. Run the backend
```bash
cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000
```
First start downloads market data from yfinance into `data/market.npz`.
Health: `curl localhost:8000/api/health`.

### 5. Run the frontend
```bash
cd frontend && npm install && npm run dev        # http://localhost:3000
```

### Configuration reference
| Env | Default | Purpose |
|-----|---------|---------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | — | primary provider |
| `LLM_API_KEY_2` / `LLM_BASE_URL_2` / `LLM_MODELS_2` | — | secondary failover bucket |
| `LLM_MODEL_<ROLE>` | — | per-role model chain (RESEARCHER/CODER/VIZ/CRITIC/REPORTER/READER) |
| `LLM_MIN_INTERVAL_PRIMARY` | 0 | seconds between calls (rate-limit pacing) |
| `DATABASE_URL` | — | Postgres connection string (required) |
| `REDIS_URL` | redis://localhost:6379/0 | Redis for pub/sub + arq queue |
| `USE_REDIS_BUS` | 1 | live streaming via Redis pub/sub |
| `USE_REDPANDA_BUS` | 0 | live streaming via Redpanda (durable log, replay) |
| `REDPANDA_BROKERS` | localhost:9092 | Redpanda/Kafka broker list |
| `QDRANT_CLUSTER_ENDPOINT` / `QDRANT_API_KEY` | (local :6333) | Qdrant Cloud |
| `ARTIFACT_S3_BUCKET` / `AWS_REGION` | (local disk) | S3 artifact storage |
| `ALPHASEEK_UNIVERSE` | 85 large-caps | comma-separated tickers |
| `ALPHASEEK_YEARS` | 6 | years of daily history |
| `ALPHASEEK_DATA_SOURCE` | yfinance | data provider: `yfinance` or `polygon` |
| `POLYGON_API_KEY` | — | Polygon.io API key (required when source=polygon, free at polygon.io) |

---

## Roadmap
Data-ingestion pillar (Redpanda → dbt) for streaming market feeds; Langfuse/
OpenTelemetry tracing; Firecracker/E2B executor tier; Prometheus + Grafana
observability; Prefect/Dagster for dataset pipeline orchestration.
