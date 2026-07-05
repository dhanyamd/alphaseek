"""Persistence facade — Postgres when DATABASE_URL is set, else SQLite.

The rest of the app imports from here, so switching backends is one env var
(DATABASE_URL) with no code changes.
"""
from __future__ import annotations

from app.settings import settings

if settings.use_postgres:
    from app import pg as _b
else:
    from app import db as _b

init_db = _b.init_db
close_stale_runs = _b.close_stale_runs
ensure_user = _b.ensure_user
create_session = _b.create_session
list_sessions = _b.list_sessions
get_session = _b.get_session
events_after = _b.events_after
max_event_id = _b.max_event_id
append_event = _b.append_event
set_status = _b.set_status
