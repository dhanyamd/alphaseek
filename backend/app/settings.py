"""Central runtime configuration — every external service and tunable, read from
the environment with safe defaults. No service URL, credential, or knob is
hardcoded in application code; they resolve here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV = str(Path(__file__).resolve().parents[1] / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV, env_file_encoding="utf-8", extra="ignore")

    # --- persistence: Postgres only ---
    database_url: str  # required — postgres://user:pass@host:5432/dbname

    # --- redis: queue broker + live pub/sub + cache ---
    redis_url: str = "redis://localhost:6379/0"

    # --- vector store: Qdrant Cloud (endpoint + key) or local Docker (no key) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # --- object storage: artifacts (charts/manifests); local disk when unset ---
    artifact_s3_bucket: str = ""  # bucket enables S3
    aws_region: str = "us-east-1"
    artifact_s3_prefix: str = "artifacts"

    # --- streaming transport: Redis pub/sub (default) or Redpanda (durable log) ---
    use_redis_bus: bool = True
    use_redpanda_bus: bool = False
    redpanda_brokers: str = "localhost:9092"

    # --- web search: Exa neural search (free tier: 1000 req/mo) ---
    exa_api_key: str = ""

    # --- sandbox security ---
    allow_inprocess: bool = False  # must be explicitly enabled for dev fallback
    sandbox_timeout: int = 120  # max seconds per sandbox run
    docker_memory: str = "2g"  # memory cap per container
    docker_cpus: str = "2"  # CPU count per container

    # --- embedding model (single source of truth for name + dim) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    @property
    def use_postgres(self) -> bool:
        return bool(self.database_url)

    @property
    def use_s3(self) -> bool:
        return bool(self.artifact_s3_bucket)


settings = Settings()
