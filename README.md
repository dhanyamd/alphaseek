# AlphaSeek

An AI quant-research platform: a multi-agent team autonomously reads finance
papers, profiles a dataset, designs and runs an experiment, renders charts, and
answers the user's research question — writing and self-repairing real code
against real data in a hardened sandbox. **No hardcoded assumptions**: the LLM
discovers the data schema, picks the evaluation metric, and writes all code.

---

## System architecture

```
   Browser ---- prompt ----> FastAPI ----- SSE stream ----> UI
      ^                                                     |
      +--------- live events (Redis pub/sub + Redpanda) ----+


  +------------------------------------------------------------------+
  |  RESEARCH PIPELINE (no mode)                                     |
  |                                                                   |
  |  DataProfiler -----> Researcher -------> Synthesist -\            |
  |       |             (OpenAlex/arXiv)        |         |           |
  |       |               paper briefs          |         |           |
  |       v                                     v         |           |
  |  DataReport -------------------------> ExperimentPlan |           |
  |                                                |      |           |
  |                                          CodingAgent |           |
  |                                               |       |           |
  |                                         sandbox run -+           |
  |                                               |                  |
  |                                          Visualizer              |
  |                                               |                  |
  |                                         Memory (best)            |
  |                                               |                  |
  |  Exporter (Pine/MQL5, if asked) <---- best run                   |
  |  Reporter <---- answers the question                              |
  +------------------------------------------------------------------+
                       |           |             |
                       v           v             v
                 Postgres (log)  S3/disk   OpenAlex/arXiv
                                 (charts)   (on demand)
```

**Flow (one research session):**

1. **Data Profiler** — backend reads the raw file (any format), LLM interprets
   metadata into a `DataReport` grounded in the user's question.
2. **Literature** — Researcher plans queries → OpenAlex/arXiv search → read top
   papers → `PaperBriefs`.
3. **Per round** (N times): Synthesist → CodingAgent → Visualizer → Memory.
4. **Exporter** (only if the prompt asks for Pine Script / MQL5).
5. **Reporter** — answers the user's question.

Every event streams live via Redis pub/sub or Redpanda and is durably logged to
Postgres. Charts/arrays go to S3/local disk.

**Design principles**

- **Autonomous coder, minimal contract.** The sandbox exposes only
  `af.DATA` (dataset path), `af.OUT` (output dir), `af.uploads()` (uploaded
  files). The agent loads the data, discovers column names, computes forward
  returns, evaluates the strategy, prints metrics, and saves arrays — all itself.
- **No hardcoded mode.** There is no factor/general switch. The LLM reads the
  user's prompt and the data report, then decides the methodology and the
  evaluation metric (Sharpe for price data, accuracy for classification, R² for
  regression, etc.).
- **No backend grading.** `runner.py` is a thin executor — it captures stdout,
  collects saved artifacts/arrays, and prints one JSON line. It computes no
  metrics and enforces no schema. The metrics the user sees are whatever the
  coder printed.
- **Dynamic data discovery.** Uploaded files (CSV/Parquet/NPZ/...) are saved raw
  and profiled at runtime by extension. No conversion, no assumed column names.
  The default `market.npz` (yfinance/Polygon) is only a fallback when nothing is
  uploaded.
- **Math first, visuals last.** The coder saves arrays to `af.OUT` as `.npz`;
  the Visualizer loads whatever keys exist and renders Plotly charts that answer
  the research question. A chart bug can't lose a result.
- **Self-repair loop.** On a sandbox error, the error text becomes `feedback`
  for the next round's coder, which rewrites the script. No silent fallbacks.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, CodeMirror, Geist |
| API | FastAPI, SSE + WebSockets |
| Persistence | **Postgres** (JSONB event log) |
| Streaming | **Redis** (pub/sub for live UI) + **Redpanda** (durable log, replay) |
| Object storage | **S3 + presigned URLs** (local disk fallback) |
| LLM routing | multi-provider client (OpenAI-compatible + Anthropic) with failover |
| Sandbox | Docker (`--network none`), quant stack: numpy, pandas, scipy, scikit-learn, statsmodels, arch, cvxpy, quantstats, plotly, riskfolio-lib, skfolio, pyportfolioopt |
| Literature | OpenAlex / arXiv + full-text fetch |
| Data | **yfinance** (default, no key) or **Polygon.io** (free tier) + user uploads (CSV, Parquet, Excel, JSON, NPZ) |

**Backend layout**
```
backend/app/
  main.py           FastAPI: sessions, SSE + WebSocket streaming, uploads, artifacts
  settings.py       one place all service URLs/creds/toggles resolve from env
  store.py          persistence facade → pg.py (Postgres)
  pg.py             Postgres: users, sessions, JSONB event log
  bus.py            dual-transport event bus: Redis pub/sub + Redpanda
  storage.py        artifact store (local + S3)
  agent/
    orchestrator.py  runs the multi-agent pipeline, streams events, manages state
    agents.py        DataProfiler, Researcher, Synthesist, CodingAgent,
                     Visualizer, Exporter, Reporter
    memory.py        per-session Memory (what was tried, what worked)
    llm.py           multi-provider client (routing / failover / pacing)
    reader.py        paper reader (full-text → LLM reduce)
    research_tools.py OpenAlex/arXiv search + full-text fetch
  quant/
    dataset.py       optional default dataset: yfinance/Polygon → market.npz
    docker_sandbox.py runs agent code in a hardened container + provisioning
    provision.py     agent-declared pip deps → cached, allowlisted image layers
    schemas.py       DataReport + ExperimentPlan (Pydantic)
sandbox/
  runner.py          thin executor: af.DATA / af.OUT / af.uploads(), captures stdout
  Dockerfile         base image with the quant + portfolio stack
docker-compose.yml   Redis, Postgres, Redpanda
```

---

## The sandbox contract

`runner.py` is intentionally thin. The agent writes ALL its own code. The
sandbox only provides file handles and captures output:

```python
import alphaseek as af
af.DATA      -> path to the dataset file (agent loads + inspects it)
af.OUT       -> directory to save artifacts / arrays (agent picks names)
af.uploads() -> paths to any user-uploaded files
```

The agent is expected to:
- load `af.DATA` (np.load / pd.read_csv / etc.) and inspect the keys/columns
- compute forward returns from whatever price column exists
- evaluate the strategy however the research question demands
- save arrays it wants visualized to `af.OUT` (e.g. `np.savez`)
- print a single JSON result line (metrics / conclusions) to stdout

There is **no** `af.submit`, **no** `af.manifest`, **no** metric computation,
**no** hardcoded `manifest.npz` name. The runner prints one JSON line containing
the agent's output + the list of files saved in `af.OUT`.

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
LLM_API_KEY=...                 # e.g. an Anthropic or OpenAI key
LLM_BASE_URL=https://api.anthropic.com/v1/
LLM_MODEL=claude-...
# optional secondary bucket for failover:
LLM_API_KEY_2=...  LLM_BASE_URL_2=https://openrouter.ai/api/v1  LLM_MODELS_2=openai/gpt-oss-120b:free
```

### 2. Build the sandbox image (once)
```bash
cd sandbox
docker build -t alphaseek-sandbox:base .
docker tag alphaseek-sandbox:base alphaseek-sandbox:latest
```

### 3. Infrastructure (local Docker)
```bash
docker compose up -d            # Redis, Postgres, Redpanda
docker compose ps               # all healthy
```
`docker compose up` starts **only the backing services**, not the app itself.
You still run the backend (step 4) and frontend (step 5) as separate processes.
Then enable the pieces you want in `backend/.env`:
```
DATABASE_URL=postgresql://alphaseek:alphaseek@localhost:5432/alphaseek
REDIS_URL=redis://localhost:6379/0
# Redpanda for durable event log (replaces Redis pub/sub for streaming):
USE_REDPANDA_BUS=1
REDPANDA_BROKERS=localhost:9092
# S3 artifacts (or omit for local disk):
ARTIFACT_S3_BUCKET=my-bucket   AWS_REGION=us-east-1
```

### 4. Run the backend
```bash
cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000
```
On first start (if no file is uploaded and `data/market.npz` is missing), the
backend downloads the **default market dataset**. The source is configured by
`ALPHASEEK_DATA_SOURCE` (default **`yfinance`**, no API key needed). Set
`ALPHASEEK_DATA_SOURCE=polygon` + `POLYGON_API_KEY` to use Polygon.io instead.
This requires network access on first run and only produces the fallback
`market.npz` — uploaded files are always used in preference.
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
| `LLM_MODEL_<ROLE>` | — | per-role model chain (RESEARCHER/CODER/VIZ/EXPORTER/REPORTER/READER) |
| `DATABASE_URL` | — | Postgres connection string (required) |
| `REDIS_URL` | redis://localhost:6379/0 | Redis for pub/sub |
| `USE_REDPANDA_BUS` | 0 | live streaming via Redpanda (durable log, replay) |
| `REDPANDA_BROKERS` | localhost:9092 | Redpanda/Kafka broker list |
| `ARTIFACT_S3_BUCKET` / `AWS_REGION` | (local disk) | S3 artifact storage |
| `ALPHASEEK_DATASET_PATH` | data/market.npz | path to the default dataset |
| `ALPHASEEK_UNIVERSE` | 96 large-caps | comma-separated tickers |
| `ALPHASEEK_YEARS` | 6 | years of daily history |
| `ALPHASEEK_DATA_SOURCE` | yfinance | data provider: `yfinance` or `polygon` |
| `POLYGON_API_KEY` | — | Polygon.io API key (required when source=polygon) |

---

## Roadmap
Langfuse/OpenTelemetry tracing; Firecracker/E2B executor tier; Prometheus +
Grafana observability; Prefect/Dagster for dataset pipeline orchestration.
