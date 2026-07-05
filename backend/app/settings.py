"""Central runtime configuration — every external service and tunable, read from
the environment with safe defaults. No service URL, credential, or knob is
hardcoded in application code; they resolve here.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env BEFORE reading any variable, regardless of import order elsewhere.
load_dotenv()


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # --- persistence: SQLite today, Postgres when DATABASE_URL is set (v3) ---
    database_url: str = os.getenv("DATABASE_URL", "").strip()          # postgres://... enables PG
    sqlite_path: str = os.getenv("ALPHASEEK_SQLITE", "alphaseek.db")

    # --- redis: queue broker + live pub/sub + cache ---
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # --- vector store: Qdrant Cloud (endpoint + key) or local Docker (no key) ---
    qdrant_url: str = (os.getenv("QDRANT_CLUSTER_ENDPOINT", "").strip()
                       or os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip()

    # --- object storage: artifacts (charts/manifests); local disk when unset ---
    s3_bucket: str = os.getenv("ARTIFACT_S3_BUCKET", "").strip()       # bucket enables S3
    s3_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_prefix: str = os.getenv("ARTIFACT_S3_PREFIX", "artifacts")

    # --- streaming transport: DB-tailing SSE today, Redis pub/sub when enabled ---
    use_redis_bus: bool = _b("USE_REDIS_BUS", False)

    @property
    def use_postgres(self) -> bool:
        return bool(self.database_url)

    @property
    def use_s3(self) -> bool:
        return bool(self.s3_bucket)


settings = Settings()
