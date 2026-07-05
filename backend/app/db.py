"""SQLite persistence — users, research sessions, and their event logs.

Keeps things dead simple (stdlib sqlite3): a "user" is just a name, a "session"
is one research run (seed + settings), and every step the agent emits is stored
as an event so past sessions can be replayed in the UI.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "alphaseek.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                name TEXT PRIMARY KEY,
                created REAL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT, seed TEXT, iterations INTEGER,
                status TEXT, created REAL, best_json TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER, ts REAL, payload TEXT
            );
            """
        )


def close_stale_runs() -> None:
    """A restart orphans daemon workers — mark any 'running' session done."""
    with _conn() as c:
        c.execute("UPDATE sessions SET status='done' WHERE status='running'")


def ensure_user(name: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO users(name, created) VALUES(?,?)", (name, time.time()))


def create_session(user: str, seed: str, iterations: int) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO sessions(user, seed, iterations, status, created) VALUES(?,?,?,?,?)",
            (user, seed, iterations, "pending", time.time()),
        )
        return int(cur.lastrowid)


def list_sessions(user: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, seed, iterations, status, created, best_json FROM sessions "
            "WHERE user=? ORDER BY created DESC",
            (user,),
        ).fetchall()
    out = []
    for r in rows:
        best = json.loads(r["best_json"]) if r["best_json"] else None
        out.append({"id": r["id"], "seed": r["seed"], "iterations": r["iterations"],
                    "status": r["status"], "created": r["created"], "best": best})
    return out


def get_session(session_id: int) -> dict | None:
    with _conn() as c:
        s = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not s:
            return None
        events = c.execute(
            "SELECT payload FROM events WHERE session_id=? ORDER BY id", (session_id,)
        ).fetchall()
    return {
        "id": s["id"], "user": s["user"], "seed": s["seed"], "iterations": s["iterations"],
        "status": s["status"], "created": s["created"],
        "best": json.loads(s["best_json"]) if s["best_json"] else None,
        "events": [json.loads(e["payload"]) for e in events],
    }


def events_after(session_id: int, after_id: int) -> list[tuple[int, dict]]:
    """Events with id > after_id — lets live streams tail the DB (reconnect-safe)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, payload FROM events WHERE session_id=? AND id>? ORDER BY id",
            (session_id, after_id),
        ).fetchall()
    return [(r["id"], json.loads(r["payload"])) for r in rows]


def max_event_id(session_id: int) -> int:
    with _conn() as c:
        r = c.execute("SELECT COALESCE(MAX(id),0) m FROM events WHERE session_id=?",
                      (session_id,)).fetchone()
    return int(r["m"])


def append_event(session_id: int, event: dict) -> None:
    with _conn() as c:
        c.execute("INSERT INTO events(session_id, ts, payload) VALUES(?,?,?)",
                  (session_id, time.time(), json.dumps(event)))


def set_status(session_id: int, status: str, best: dict | None = None) -> None:
    with _conn() as c:
        if best is not None:
            c.execute("UPDATE sessions SET status=?, best_json=? WHERE id=?",
                      (status, json.dumps(best), session_id))
        else:
            c.execute("UPDATE sessions SET status=? WHERE id=?", (status, session_id))
