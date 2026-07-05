# AlphaSeek

An AI quant-research platform: a multi-agent team autonomously reads finance
papers, invents a trading strategy, writes and self-repairs real quant code
against real market data in a hardened sandbox, grades the resulting factor, and
renders interactive visualizations — streamed live to a web IDE. A working clone
of QuantPad's research agent.

---

## System architecture

```
   Browser ──── prompt ────► FastAPI ──── enqueue ────► arq worker
      ▲                                                     │
      └──── live events (WebSocket ◄ Redis bus) ────────────┘
                                                            │  runs the pipeline
                                                            ▼
 ┌─ RESEARCH PIPELINE ──────────────────────────────────────────────────────┐
 │                                                                           │
 │  Researcher ─► Reader ─► Synthesist ─► Coder ─► Visualizer ─► Critic ─►    │
 │      │           │                       │                        Reporter │
 │  OpenAlex     Qdrant RAG            Docker sandbox                          │
 │  arXiv/Jina   (+ fastembed)     (isolated · no network)                    │
 │                                  agent writes all code,                    │
 │                                  af.submit → blind grade                   │
 └──────────────┬──────────────────────────────────┬────────────────────────┘
                │ events                            │ artifacts
                ▼                                   ▼
          Postgres (log)                      S3 / local disk

   LLM calls (every stage) ─► app/agent/llm.py ─► providers (Gemini / OpenRouter)
                                 multi-provider · failover · pacing
                                 (optionally via a LiteLLM gateway)
```

**Flow:** a prompt is queued; a worker runs the pipeline — the *research* half
(Researcher → Reader → Synthesist) reads papers and produces a grounded plan,
the *build* half (Coder → Visualizer) writes and runs real code in an isolated
sandbox and renders charts, and (Critic → Reporter) grade and explain it. Every
event streams live to the browser and is logged to Postgres; artifacts go to S3.

**Design principles**
- **Autonomous coder, minimal contract.** The sandbox exposes exactly six things
  (`af.DATA`, `af.OUT`, `af.backtest`, `af.submit`, `af.manifest`, `af.uploads`).
  The agent loads the raw data file, inspects it, and writes *every* feature,
  estimator, and signal itself with numpy/pandas/scipy/sklearn — no DSL, no
  pre-computed features, no hardcoded strategy logic.
- **Grading integrity.** Forward returns are never exposed; `af.submit` grades
  against hidden data, and a look-ahead guard rejects future-peeking signals.
- **Math first, visuals last.** The coder saves a result manifest (`np.savez`); a
  separate Visualizer stage loads it and plots — a chart bug can't lose a result.
- **Nothing canned.** No mocks, no silent fallbacks — every failure is a visible event.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, CodeMirror, Geist |
| API | FastAPI, SSE + WebSockets |
| Persistence | **Postgres** (JSONB events) with SQLite fallback |
| Queue / streaming | **Redis** (arq job queue + pub/sub bus) |
| Vector store / RAG | **Qdrant** (Cloud or local) + fastembed (ONNX, no torch) |
| Object storage | **S3 + presigned URLs** (local disk fallback) |
| LLM routing | multi-provider client + optional **LiteLLM gateway** |
| Sandbox | Docker (`--network none`), quant stack: numpy, pandas, scipy, scikit-learn, statsmodels, empyrical, alphalens, arch, cvxpy, quantstats, plotly |
| Literature | OpenAlex / Semantic Scholar / arXiv + Jina Reader |
| Data | yfinance (raw daily panels) |

**Backend layout**
```
backend/app/
  main.py          FastAPI: sessions, SSE + WebSocket streaming, uploads, artifacts
  settings.py      one place all service URLs/creds/toggles resolve from env
  store.py         persistence facade → pg.py (Postgres) or db.py (SQLite)
  bus.py           Redis pub/sub event bus
  tasks.py         arq worker (run `arq app.tasks.WorkerSettings`)
  storage.py       artifact store (local + S3)
  agent/
    orchestrator.py  runs the multi-agent pipeline, streams events
    agents.py        Researcher, Synthesist, CodingAgent, Visualizer, Critic, Reporter
    llm.py           multi-provider client (routing / failover / pacing)
    rag.py           Qdrant agentic RAG (index / search / retrieve→judge→refine)
    reader.py        PaperQA-style reader (chunk → embed-rank → 1 LLM reduce)
    research_tools.py OpenAlex/S2/arXiv search + Jina full-text
  quant/
    dataset.py       yfinance → raw market.npz (env-configurable universe)
    docker_sandbox.py runs agent code in a hardened container + provisioning
    provision.py     agent-declared pip deps → cached, allowlisted image layers
sandbox/
  runner.py          the `alphaseek` module (6-item contract) + grading + LA guard
  Dockerfile         base image with the quant stack
docker-compose.yml   Redis, Postgres, Qdrant, LiteLLM
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

### 3. (Optional) v3 infrastructure
```bash
docker compose up -d            # Redis, Postgres, Qdrant, LiteLLM
docker compose ps               # all healthy
```
Then enable the pieces you want in `backend/.env`:
```
DATABASE_URL=postgresql://alphaseek:alphaseek@localhost:5432/alphaseek   # use Postgres
REDIS_URL=redis://localhost:6379/0
USE_REDIS_BUS=1                                                          # WS via Redis
# Qdrant Cloud (or omit for local Docker Qdrant):
QDRANT_CLUSTER_ENDPOINT=https://<cluster>.cloud.qdrant.io
QDRANT_API_KEY=...
# S3 artifacts (or omit for local disk):
ARTIFACT_S3_BUCKET=my-bucket   AWS_REGION=us-east-1
```
To run research in worker processes instead of the API thread:
```bash
cd backend && .venv/bin/python -m arq app.tasks.WorkerSettings
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
| `DATABASE_URL` | (SQLite) | Postgres connection string; unset ⇒ SQLite |
| `REDIS_URL` | redis://localhost:6379/0 | queue + pub/sub |
| `USE_REDIS_BUS` | 0 | WebSocket streams via Redis instead of DB tail |
| `QDRANT_CLUSTER_ENDPOINT` / `QDRANT_API_KEY` | (local :6333) | Qdrant Cloud |
| `ARTIFACT_S3_BUCKET` / `AWS_REGION` | (local disk) | S3 artifact storage |
| `ALPHASEEK_UNIVERSE` | 85 large-caps | comma-separated tickers |
| `ALPHASEEK_YEARS` | 6 | years of daily history |

---

## Roadmap
Data-ingestion pillar (Kafka → Snowflake → dbt) for streaming market feeds;
Langfuse/OpenTelemetry tracing; Firecracker/E2B executor tier; Pine Script /
MQL5 export of validated strategies.
