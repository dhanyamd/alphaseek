# AlphaSeek

An AI quant-research platform: a multi-agent team autonomously reads finance
papers, invents a trading strategy, writes and self-repairs real quant code
against real market data in a hardened sandbox, grades the resulting factor, and
renders interactive visualizations — streamed live to a web IDE. A working clone
of QuantPad's research agent.

---

## System architecture (current)

```
                         ┌───────────────────────────────────────────────┐
  Browser (Next.js IDE)  │  chat · CodeMirror editor · artifact viewer    │
   left: files/sessions  │  right: agent pipeline strip + live feed       │
        │  ▲              └───────────────────────────────────────────────┘
   HTTP │  │ SSE (event stream)
        ▼  │
  ┌──────────────────────────── FastAPI (app/main.py) ────────────────────────┐
  │  /api/login  /api/sessions  /api/sessions/{id}/stream (SSE)                │
  │  /api/sessions/{id}/upload  /run  /api/artifacts/{name}                    │
  │  a worker THREAD runs the research generator; every event is persisted to  │
  │  SQLite; the SSE endpoint TAILS the DB (disconnect/refresh-proof)          │
  └───────────────┬───────────────────────────────────────────┬──────────────┘
                  │                                            │
        ┌─────────▼──────────┐                        ┌────────▼─────────┐
        │  SQLite (db.py)    │                        │ Orchestrator     │
        │  users/sessions/   │                        │ (agent/orchestr.)│
        │  events            │                        └────────┬─────────┘
        └────────────────────┘                                 │  streams handoffs
                                                               ▼
   LITERATURE (once/prompt)          ROUNDS (per iteration)
   ┌──────────────────────┐   ┌───────────────────────────────────────────────┐
   │ Researcher.queries   │   │ Synthesist  → connect papers → novel plan +    │
   │ → OpenAlex/S2/arXiv   │   │               pip requirements                 │
   │ → rank (fastembed)    │   │ Provision   → cached dep-layer image (allow-   │
   │ → Reader (Jina +      │   │               listed, network-gated build)     │
   │   embed + 1 LLM call) │   │ Quant Coder → MATH ONLY, agentic run/fix loop, │
   │ → PaperBriefs         │   │               records a result manifest        │
   └──────────────────────┘   │ Visualizer  → loads manifest, renders plotly   │
                              │ Risk Critic → grades edge + overfit            │
                              │ Reporter    → grounded answer + next steps     │
                              └───────────────────────┬───────────────────────┘
                                                      │ run code
                                          ┌───────────▼───────────────┐
                                          │  Docker sandbox            │
                                          │  --network none, non-root, │
                                          │  read-only, cpu/mem caps   │
                                          │  runs sandbox/runner.py     │
                                          │  mounts: runner, data(ro),  │
                                          │  uploads(ro), manifest      │
                                          └───────────┬───────────────┘
                                                      │ import alphaseek_data as ad
                                          ┌───────────▼───────────────┐
                                          │  data/market.npz (yfinance)│
                                          │  RAW panels only: close,   │
                                          │  volume, returns + hidden  │
                                          │  fwd (grading target)      │
                                          └────────────────────────────┘
```

**Key design decisions**
- **No pre-computed features, no hardcoded strategy logic.** The sandbox exposes
  only raw point-in-time panels (`ad.data['close']`, `ad.returns()`, `ad.volume()`)
  and a runtime schema (`ad.describe()`). The agent derives *every* feature itself
  from the papers. The universe/history are env config (`ALPHASEEK_UNIVERSE`,
  `ALPHASEEK_YEARS`), not baked in.
- **Grading integrity.** Forward returns are never exposed; a look-ahead guard
  rejects any submitted signal whose IC is implausibly high (future-peeking).
- **Math first, visuals last.** The coder writes pure math and records a manifest;
  a separate Visualizer stage loads that manifest and plots — a chart bug can
  never lose a validated result.
- **Model routing + failover.** Per-role model chains across providers
  (`LLM_MODEL_<ROLE>`, primary + secondary buckets) with client-side pacing and
  `reasoning_effort` control for thinking models.
- **Nothing canned.** No mocks, no silent fallbacks — every failure is a visible
  event.

**Repo layout**
```
backend/
  app/main.py            FastAPI: sessions, SSE streaming, uploads, artifacts
  app/db.py              SQLite persistence (users/sessions/events)
  app/agent/
    orchestrator.py      runs the multi-agent pipeline, streams events
    agents.py            Researcher, Synthesist, CodingAgent, Visualizer, Critic, Reporter
    llm.py               multi-provider OpenAI-compatible client (routing/failover/pacing)
    reader.py            PaperQA-style reader (chunk → embed-rank → 1 LLM reduce)
    research_tools.py    OpenAlex/S2/arXiv search + Jina full-text
    memory.py, evaluate.py
  app/quant/
    dataset.py           yfinance → raw market.npz (env-configurable universe)
    docker_sandbox.py    runs agent code in a hardened container + provisioning
    provision.py         agent-declared pip deps → cached, allowlisted image layers
    backtest.py          FactorError type
sandbox/
  runner.py              in-container data API (ad.*) + grading + look-ahead guard
  Dockerfile             base image with the quant stack (empyrical/alphalens/arch/cvxpy/...)
frontend/                Next.js IDE (chat, editor, artifact viewer, agent strip)
docker-compose.yml       v3 infra: Redis, Postgres, Qdrant
```

---

## How to run

### Prerequisites
- Python 3.12, Node 18+, Docker, `uv`

### 1. Backend deps
```bash
cd backend
uv venv && uv pip install -r requirements.txt
cp .env.example .env            # then set LLM_API_KEY etc.
```
Set at least one provider in `.env`:
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

### 3. Run the backend
```bash
cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000
```
First start downloads market data from yfinance into `data/market.npz` (a few
seconds). Health: `curl localhost:8000/api/health`.

### 4. Run the frontend
```bash
cd frontend && npm install && npm run dev        # http://localhost:3000
```

### 5. v3 infrastructure (Redis / Postgres / Qdrant)
```bash
docker compose up -d            # from repo root
docker compose ps               # all healthy
```

### Configuration reference
| Env | Default | Purpose |
|-----|---------|---------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | — | primary provider |
| `LLM_API_KEY_2` / `LLM_BASE_URL_2` / `LLM_MODELS_2` | — | secondary failover bucket |
| `LLM_MODEL_<ROLE>` | — | per-role model chain (RESEARCHER/CODER/VIZ/CRITIC/REPORTER/READER) |
| `LLM_MIN_INTERVAL_PRIMARY` | 0 | seconds between calls (rate-limit pacing) |
| `ALPHASEEK_UNIVERSE` | 85 large-caps | comma-separated tickers |
| `ALPHASEEK_YEARS` | 6 | years of daily history |

---

## Roadmap (v3, in progress)
Infra foundation up (`docker-compose.yml`). Staged wiring: Postgres event store →
arq worker queue + Redis pub/sub → WebSocket streaming → LiteLLM gateway →
Langfuse/OpenTelemetry observability → S3 artifacts → Qdrant paper archive.
The data-ingestion pillar (Kafka → Snowflake → dbt) is a separate subsystem.
