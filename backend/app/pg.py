"""Postgres persistence — same interface as db.py, selected when DATABASE_URL is
set. JSONB event payloads, real concurrency. Enabled via app/store.py.
"""

from __future__ import annotations

import time

import psycopg
from psycopg.types.json import Jsonb

from app.settings import settings


def _conn() -> psycopg.Connection:
    return psycopg.connect(settings.database_url, row_factory=psycopg.rows.dict_row)


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                name TEXT PRIMARY KEY, created DOUBLE PRECISION);
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY, "user" TEXT, seed TEXT, iterations INTEGER,
                mode TEXT DEFAULT 'factor', status TEXT, created DOUBLE PRECISION, best_json JSONB);
            CREATE TABLE IF NOT EXISTS events (
                id BIGSERIAL PRIMARY KEY, session_id INTEGER, ts DOUBLE PRECISION,
                payload JSONB);
            CREATE INDEX IF NOT EXISTS events_session_id_idx ON events(session_id, id);
        """)
        # migrations for existing tables
        _add_column_if_missing(c, "sessions", "mode", "TEXT DEFAULT 'factor'")
        _add_column_if_missing(c, "sessions", "iterations", "INTEGER")


def _add_column_if_missing(c: psycopg.Connection, table: str, col: str, col_def: str) -> None:
    exists = c.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_name=%s AND column_name=%s",
        (table, col),
    ).fetchone()
    if not exists:
        c.execute(f'ALTER TABLE "{table}" ADD COLUMN {col} {col_def}')


def close_stale_runs() -> None:
    with _conn() as c:
        c.execute("UPDATE sessions SET status='done' WHERE status='running'")


def ensure_user(name: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO users(name, created) VALUES(%s,%s) ON CONFLICT DO NOTHING",
            (name, time.time()),
        )


def create_session(user: str, seed: str, iterations: int, mode: str) -> int:
    with _conn() as c:
        row = c.execute(
            'INSERT INTO sessions("user", seed, iterations, mode, status, created) '
            "VALUES(%s,%s,%s,%s,%s,%s) RETURNING id",
            (user, seed, iterations, mode, "pending", time.time()),
        ).fetchone()
        return int(row["id"])


def list_sessions(user: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, seed, iterations, mode, status, created, best_json FROM sessions "
            'WHERE "user"=%s ORDER BY created DESC',
            (user,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "seed": r["seed"],
            "iterations": r["iterations"],
            "mode": r["mode"],
            "status": r["status"],
            "created": r["created"],
            "best": r["best_json"],
        }
        for r in rows
    ]


def get_session(session_id: int) -> dict | None:
    with _conn() as c:
        s = c.execute("SELECT * FROM sessions WHERE id=%s", (session_id,)).fetchone()
        if not s:
            return None
        events = c.execute(
            "SELECT payload FROM events WHERE session_id=%s ORDER BY id", (session_id,)
        ).fetchall()
    return {
        "id": s["id"],
        "user": s["user"],
        "seed": s["seed"],
        "iterations": s["iterations"],
        "mode": s["mode"],
        "status": s["status"],
        "created": s["created"],
        "best": s["best_json"],
        "events": [e["payload"] for e in events],
    }


def events_after(session_id: int, after_id: int) -> list[tuple[int, dict]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, payload FROM events WHERE session_id=%s AND id>%s ORDER BY id",
            (session_id, after_id),
        ).fetchall()
    return [(r["id"], r["payload"]) for r in rows]


def max_event_id(session_id: int) -> int:
    with _conn() as c:
        r = c.execute(
            "SELECT COALESCE(MAX(id),0) m FROM events WHERE session_id=%s", (session_id,)
        ).fetchone()
    return int(r["m"])


def append_event(session_id: int, event: dict) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO events(session_id, ts, payload) VALUES(%s,%s,%s)",
            (session_id, time.time(), Jsonb(event)),
        )


def set_status(session_id: int, status: str, best: dict | None = None) -> None:
    with _conn() as c:
        if best is not None:
            c.execute(
                "UPDATE sessions SET status=%s, best_json=%s WHERE id=%s",
                (status, Jsonb(best), session_id),
            )
        else:
            c.execute("UPDATE sessions SET status=%s WHERE id=%s", (status, session_id))
