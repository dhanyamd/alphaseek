"""Redis pub/sub event bus — live streaming decoupled from persistence.

Workers publish each event here; SSE/WebSocket endpoints subscribe. Durable
replay still comes from the event store (store.py); this is only the live fan-out.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

from app.settings import settings

_redis = None


def _client():
    global _redis
    if _redis is None:
        import redis

        _redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def channel(session_id: int) -> str:
    return f"alphaseek:events:{session_id}"


def publish(session_id: int, event: dict) -> None:
    try:
        _client().publish(channel(session_id), json.dumps(event))
    except Exception:  # noqa: BLE001 — live fan-out is best-effort; DB is the record
        pass


def subscribe(session_id: int) -> Iterator[dict]:
    """Yield events published to a session (blocking generator)."""
    ps = _client().pubsub()
    ps.subscribe(channel(session_id))
    try:
        for msg in ps.listen():
            if msg.get("type") == "message":
                try:
                    yield json.loads(msg["data"])
                except (ValueError, TypeError):
                    continue
    finally:
        ps.close()
