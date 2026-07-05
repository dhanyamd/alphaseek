"""AlphaSeek API — FastAPI app: users, sessions, and the live research stream.

Endpoints:
    POST /api/login                 {name}            -> ensure user
    GET  /api/sessions?user=        -> list a user's research sessions
    POST /api/sessions              {user,seed,iterations} -> create a session
    GET  /api/sessions/{id}         -> a session + its full event log (replay)
    GET  /api/sessions/{id}/stream  -> SSE: run the loop, stream + persist events
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app import bus, storage
from app import store as db          # SQLite or Postgres, chosen by DATABASE_URL
from app.agent.llm import get_llm
from app.agent.orchestrator import research
from app.quant.dataset import dataset_meta, dataset_status, ensure_dataset_async
from app.settings import settings

UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_ROOT.mkdir(exist_ok=True)

load_dotenv()
db.init_db()
db.close_stale_runs()    # a restart orphans daemon workers — close out stale runs
ensure_dataset_async()   # build the real market cache if missing

_RUNNING: set[int] = set()

app = FastAPI(title="AlphaSeek", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"], allow_headers=["*"],
)


class LoginIn(BaseModel):
    name: str


class SessionIn(BaseModel):
    user: str
    seed: str
    iterations: int = 8


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "llm_mode": get_llm().mode, "model": get_llm().model,
            "dataset": {**dataset_status(), **dataset_meta()}}


@app.post("/api/sessions/{session_id}/upload")
async def upload(session_id: int, file: UploadFile):
    """Attach a file (trade log CSV, etc.) to a session — visible to the agents."""
    sdir = UPLOAD_ROOT / str(session_id)
    sdir.mkdir(exist_ok=True)
    safe = Path(file.filename or "upload.csv").name
    dest = sdir / safe
    dest.write_bytes(await file.read())
    db.append_event(session_id, {"type": "upload", "filename": safe})
    return {"ok": True, "filename": safe,
            "files": sorted(p.name for p in sdir.iterdir())}


class RunIn(BaseModel):
    filename: str = "script.py"
    code: str


@app.post("/api/sessions/{session_id}/run")
def manual_run(session_id: int, body: RunIn) -> dict:
    """Run a script from the editor (the Run button). Persists code + result events."""
    from app.quant.backtest import FactorError
    from app.quant.docker_sandbox import run_factor_code

    sdir = UPLOAD_ROOT / str(session_id)
    db.append_event(session_id, {"type": "code", "agent": "You", "step": 0,
                                 "filename": body.filename, "code": body.code})
    try:
        bt = run_factor_code(body.code, uploads_dir=sdir if sdir.is_dir() else None)
        ev = {"type": "backtest", "agent": "Backtester", "step": 0,
              "name": body.filename, "result": bt,
              "engine": bt.get("engine", ""), "exploration": not bt.get("submitted")}
        db.append_event(session_id, ev)
        return ev
    except FactorError as e:
        ev = {"type": "run_error", "step": 0, "message": str(e)[:400]}
        db.append_event(session_id, ev)
        return ev


@app.get("/api/artifacts/{name}")
def artifact(name: str):
    """Serve a chart/artifact. Redirects to a presigned S3 URL when configured,
    else serves the local copy."""
    from fastapi.responses import FileResponse, RedirectResponse

    safe = Path(name).name                      # no path traversal
    s3_url = storage.url(safe)
    if s3_url:
        return RedirectResponse(s3_url)
    path = storage.local_path(safe)
    if not path.is_file():
        return {"error": "not found"}
    return FileResponse(path)


@app.post("/api/login")
def login(body: LoginIn) -> dict:
    db.ensure_user(body.name.strip())
    return {"ok": True, "user": body.name.strip()}


@app.get("/api/sessions")
def sessions(user: str) -> dict:
    return {"sessions": db.list_sessions(user)}


@app.post("/api/sessions")
def new_session(body: SessionIn) -> dict:
    sid = db.create_session(body.user.strip(), body.seed.strip(), body.iterations)
    return {"id": sid}


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: int) -> dict:
    s = db.get_session(session_id)
    return s or {"error": "not found"}


def _rebuild_memory(events: list[dict]):
    """Reconstruct the Archivist's memory from a session's stored events so a
    follow-up prompt continues where the last run left off."""
    from app.agent.memory import Memory

    mem = Memory()
    pending: dict = {}
    for ev in events:
        if ev.get("type") == "backtest":
            pending[ev.get("name")] = ev.get("result", {})
        elif ev.get("type") == "verdict" and ev.get("name") in pending:
            bt = pending.pop(ev["name"])
            if "sharpe" in bt and "mean_ic" in bt:
                mem.add(ev["name"], bt.get("expr", ev["name"]), bt, ev.get("verdict", {}))
    return mem


@app.get("/api/sessions/{session_id}/stream")
async def stream(session_id: int, prompt: str | None = None):
    """Run one research pass for this session, streaming each step as SSE.

    With `prompt`, this is a FOLLOW-UP turn in the conversation: the Archivist's
    memory is rebuilt from the session's history and the new prompt steers the
    next rounds. Without it, the session's original seed is used (first turn).
    """
    s = db.get_session(session_id)
    if not s:
        async def err():
            yield {"event": "error", "data": '{"message":"session not found"}'}
        return EventSourceResponse(err())

    goal = (prompt or "").strip() or s["seed"]
    mem = _rebuild_memory(s["events"]) if s["events"] else None

    import json
    import threading

    # Execution is DECOUPLED from the stream: the worker persists every event to
    # the DB itself, so a browser refresh/disconnect never kills or loses a run.
    # The SSE generator simply tails the DB.
    start_after = db.max_event_id(session_id)
    if prompt:
        db.append_event(session_id, {"type": "user", "text": goal})

    sdir = UPLOAD_ROOT / str(session_id)
    upload_names = sorted(p.name for p in sdir.iterdir()) if sdir.is_dir() else []

    def worker() -> None:
        best = None
        try:
            for event in research(goal, iterations=s["iterations"], mem=mem,
                                  uploads_dir=sdir if upload_names else None,
                                  uploads=[f"/uploads/{n}" for n in upload_names]):
                db.append_event(session_id, event)
                bus.publish(session_id, event)      # live fan-out (best-effort)
                if event["type"] == "done":
                    best = event.get("best")
        except Exception as e:  # noqa: BLE001 — surface crashes to the UI
            db.append_event(session_id, {"type": "error", "fatal": True,
                                         "message": f"{type(e).__name__}: {e}"})
        finally:
            db.set_status(session_id, "done", best)
            _RUNNING.discard(session_id)

    if session_id not in _RUNNING:
        _RUNNING.add(session_id)
        db.set_status(session_id, "running")
        threading.Thread(target=worker, daemon=True).start()

    async def gen():
        last = start_after
        while True:
            rows = await asyncio.to_thread(db.events_after, session_id, last)
            for eid, ev in rows:
                last = eid
                yield {"data": json.dumps(ev)}
            if not rows:
                if session_id not in _RUNNING:
                    break
                await asyncio.sleep(0.3)
        yield {"data": json.dumps({"type": "close"})}

    return EventSourceResponse(gen())


@app.websocket("/api/sessions/{session_id}/ws")
async def stream_ws(websocket: WebSocket, session_id: int):
    """WebSocket transport for the live event stream. Replays the stored history,
    then tails the Redis bus (or the DB if the bus is off) — refresh-proof."""
    await websocket.accept()
    last = 0
    try:
        for eid, ev in db.events_after(session_id, 0):   # replay history
            last = eid
            await websocket.send_json(ev)
        if settings.use_redis_bus:
            for ev in bus.subscribe(session_id):         # live via Redis pub/sub
                await websocket.send_json(ev)
                if ev.get("type") in ("done", "close"):
                    break
        else:
            while True:                                  # live via DB tail
                rows = await asyncio.to_thread(db.events_after, session_id, last)
                for eid, ev in rows:
                    last = eid
                    await websocket.send_json(ev)
                if not rows and session_id not in _RUNNING:
                    break
                await asyncio.sleep(0.3)
        await websocket.send_json({"type": "close"})
    except WebSocketDisconnect:
        return
