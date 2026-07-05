"""arq task queue — research runs execute in worker processes, off the API.

Run a worker with:  arq app.tasks.WorkerSettings
The API enqueues jobs (see enqueue_research); each event is persisted to the
store and published to the Redis bus for live streaming.
"""
from __future__ import annotations

from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings

from app.settings import settings

UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "uploads"


def _redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def run_research_task(ctx, session_id: int, goal: str, iterations: int) -> None:
    """Execute one research pass; persist + publish every event. The pipeline is
    blocking (LLM + Docker), so it runs in a worker thread off the event loop."""
    import anyio

    from app import bus, store
    from app.agent.orchestrator import research

    def _run() -> None:
        best = None
        sdir = UPLOAD_ROOT / str(session_id)
        names = sorted(p.name for p in sdir.iterdir()) if sdir.is_dir() else []
        mem = _rebuild_memory(store.get_session(session_id) or {})
        store.set_status(session_id, "running")
        try:
            for ev in research(goal, iterations=iterations, mem=mem,
                               uploads_dir=sdir if names else None,
                               uploads=[f"/uploads/{n}" for n in names]):
                store.append_event(session_id, ev)
                bus.publish(session_id, ev)
                if ev.get("type") == "done":
                    best = ev.get("best")
        except Exception as e:  # noqa: BLE001
            err = {"type": "error", "fatal": True, "message": f"{type(e).__name__}: {e}"}
            store.append_event(session_id, err)
            bus.publish(session_id, err)
        finally:
            store.set_status(session_id, "done", best)
            bus.publish(session_id, {"type": "close"})

    await anyio.to_thread.run_sync(_run)


def _rebuild_memory(session: dict):
    from app.agent.memory import Memory

    mem, pending = Memory(), {}
    for ev in session.get("events", []):
        if ev.get("type") == "backtest":
            pending[ev.get("name")] = ev.get("result", {})
        elif ev.get("type") == "verdict" and ev.get("name") in pending:
            bt = pending.pop(ev["name"])
            if "sharpe" in bt and "mean_ic" in bt:
                mem.add(ev["name"], bt.get("expr", ev["name"]), bt, ev.get("verdict", {}))
    return mem


async def enqueue_research(session_id: int, goal: str, iterations: int) -> None:
    pool = await create_pool(_redis_settings())
    await pool.enqueue_job("run_research_task", session_id, goal, iterations)


class WorkerSettings:
    functions = [run_research_task]
    redis_settings = _redis_settings()
