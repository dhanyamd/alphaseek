"""Persistence facade — Postgres only. DATABASE_URL is required.

SQLite has been removed. Set DATABASE_URL=postgres://... in backend/.env.
"""

from __future__ import annotations

from app import pg as _b

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
