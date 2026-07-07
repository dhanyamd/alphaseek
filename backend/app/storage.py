"""Artifact storage — local disk by default, S3 + presigned URLs when configured.

Enable S3 by setting ARTIFACT_S3_BUCKET (see settings.py). The local copy is
always kept so the API can serve artifacts even without S3.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.settings import settings

LOCAL_DIR = Path(__file__).resolve().parent.parent / "artifacts"
LOCAL_DIR.mkdir(exist_ok=True)

_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3", region_name=settings.aws_region)
    return _s3_client


def _key(name: str) -> str:
    return f"{settings.artifact_s3_prefix}/{name}"


def put(src: Path, name: str) -> str:
    """Store an artifact; return the name used to fetch it. Uploads to S3 too
    when configured (local copy always kept as a fallback)."""
    dest = LOCAL_DIR / name
    if Path(src) != dest:
        shutil.copy(src, dest)
    if settings.use_s3:
        try:
            _s3().upload_file(str(dest), settings.artifact_s3_bucket, _key(name))
        except Exception:  # noqa: BLE001 — never fail a run on artifact upload
            pass
    return name


def url(name: str) -> str | None:
    """Presigned S3 URL when configured, else None (served locally by the API)."""
    if settings.use_s3:
        try:
            return _s3().generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.artifact_s3_bucket, "Key": _key(name)},
                ExpiresIn=3600,
            )
        except Exception:  # noqa: BLE001
            return None
    return None


def put_text(text: str, name: str) -> str:
    """Save text content as an artifact (local + S3 when configured)."""
    dest = LOCAL_DIR / name
    dest.write_text(text)
    if settings.use_s3:
        try:
            _s3().upload_file(str(dest), settings.artifact_s3_bucket, _key(name))
        except Exception:  # noqa: BLE001
            pass
    return name


def local_path(name: str) -> Path:
    return LOCAL_DIR / name
