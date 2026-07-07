"""Event bus — live streaming decoupled from persistence.

Two transports:
  - Redis pub/sub (default, lightweight, fire-and-forget)
  - Redpanda/Kafka (durable log, replay, consumer groups)

Durable replay always comes from the event store (store.py); this bus is only
the *live* fan-out.  When Redpanda is enabled and the consumer connects before
a publish, it still gets the message (unlike Redis pub/sub which drops it).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from app.settings import settings

_redis = None
_kafka = None


def _redis_client():
    global _redis
    if _redis is None:
        import redis as _r

        _redis = _r.Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _kafka_producer():
    global _kafka
    if _kafka is None:
        from kafka import KafkaProducer

        _brokers = (
            settings.redpanda_brokers.split(",")
            if "," in settings.redpanda_brokers
            else [settings.redpanda_brokers]
        )
        _kafka = KafkaProducer(
            bootstrap_servers=_brokers,
            acks="all",
            compression_type="gzip",
            retries=3,
        )
    return _kafka


def _kafka_consumer(session_id: int):
    from kafka import KafkaConsumer

    return KafkaConsumer(
        _channel(session_id),
        bootstrap_servers=settings.redpanda_brokers,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id=f"alphaseek-{session_id}",
        consumer_timeout_ms=1000,
        value_deserializer=lambda m: json.loads(m.decode()),
    )


def _channel(session_id: int) -> str:
    return f"alphaseek:events:{session_id}"


def publish(session_id: int, event: dict) -> None:
    """Publish an event to listeners — best-effort for both transports."""
    payload = json.dumps(event)

    # Redis pub/sub (fire-and-forget, always on for live UI)
    try:
        _redis_client().publish(_channel(session_id), payload)
    except Exception:  # noqa: BLE001
        pass

    # Redpanda (durable log — survives subscriber disconnects)
    if settings.use_redpanda_bus:
        try:
            _kafka_producer().send(_channel(session_id), payload.encode())
            _kafka_producer().flush(timeout=2)
        except Exception:  # noqa: BLE001
            pass


def subscribe(session_id: int) -> Iterator[dict]:
    """Yield events published to a session (blocking generator).

    Uses Redpanda when configured (durable, supports late-joining consumers).
    Falls back to Redis pub/sub otherwise.
    """
    if settings.use_redpanda_bus:
        consumer = _kafka_consumer(session_id)
        try:
            while True:
                msgs = consumer.poll(timeout_ms=1000)
                for _tp, batch in msgs.items():
                    for msg in batch:
                        ev = msg.value
                        yield ev
                        if ev.get("type") in ("done", "close"):
                            return
        finally:
            consumer.close()
        return

    # Redis pub/sub fallback
    ps = _redis_client().pubsub()
    ps.subscribe(_channel(session_id))
    try:
        for msg in ps.listen():
            if msg.get("type") == "message":
                try:
                    yield json.loads(msg["data"])
                except (ValueError, TypeError):
                    continue
    finally:
        ps.close()
