#!/usr/bin/env bash
# Launch both AlphaSeek servers. Ctrl-C stops both.
set -e
cd "$(dirname "$0")"
( cd backend && .venv/bin/uvicorn app.main:app --port 8000 ) &
BACK=$!
( cd frontend && npm run dev ) &
FRONT=$!
trap "kill $BACK $FRONT 2>/dev/null" EXIT
echo "AlphaSeek: backend :8000  ·  frontend http://localhost:3000"
wait
