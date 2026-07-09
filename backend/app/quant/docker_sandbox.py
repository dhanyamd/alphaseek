"""Run agent-written research SCRIPTS — Docker-hardened, artifact-collecting.

Both engines execute the SAME `sandbox/runner.py` contract (script imports
`alphaseek`, loads af.DATA, saves artifacts to af.OUT, prints JSON results):

  * docker     — hardened container: no network, read-only fs, non-root,
                 memory/CPU caps. /in (code, ro) and /out (artifacts, rw) mounts.
  * inprocess  — dev fallback: the same runner via the local Python.
                 Requires ALLOW_INPROCESS=1 environment variable. NEVER runs
                 in-process without explicit opt-in (security: agent code
                 executes with full host access when Docker is unavailable).

Artifacts (PNGs etc.) are copied to backend/artifacts/ and served by the API.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from app.quant.backtest import FactorError
from app.settings import settings

IMAGE = "alphaseek-sandbox:latest"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "sandbox" / "runner.py"
ARTIFACT_STORE = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACT_STORE.mkdir(exist_ok=True)

_DOCKER_MEM = settings.docker_memory
_DOCKER_CPUS = settings.docker_cpus
_SANDBOX_TIMEOUT = settings.sandbox_timeout


class SandboxUnavailable(Exception):
    """Docker infrastructure failed (daemon/image) — distinct from bad code."""


_docker_ok: bool | None = None


def docker_available() -> bool:
    """Is the Docker daemon reachable? (The image itself is checked by running —
    a missing image raises SandboxUnavailable and falls back per call.)"""
    global _docker_ok
    if _docker_ok is not None:
        return _docker_ok
    if shutil.which("docker") is None:
        _docker_ok = False
        return False
    try:
        p = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Os}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _docker_ok = p.returncode == 0 and bool(p.stdout.strip())
    except Exception:  # noqa: BLE001
        _docker_ok = False
    return _docker_ok


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def run_factor_code(
    code: str,
    timeout: int | None = None,
    uploads_dir: Path | None = None,
    image: str = IMAGE,
    manifest_src: Path | None = None,
) -> dict:
    """Execute an agent research script; return stdout + artifacts.

    `image` selects a provisioned sandbox image (base or a dep-layered one).
    `manifest_src` (optional) mounts a prior run's saved arrays so a
    visualization script can load them without recomputing.
    """
    global _docker_ok
    import logging
    import time as _time

    log = logging.getLogger(__name__)
    timeout = timeout or _SANDBOX_TIMEOUT
    t0 = _time.time()
    tmp = Path(tempfile.mkdtemp(prefix="alphaseek_"))
    out = tmp / "out"
    out.mkdir()
    (tmp / "research.py").write_text(code)
    try:
        if docker_available():
            try:
                result = _exec_docker(tmp, out, timeout, uploads_dir, image)
                result["engine"] = "docker"
            except SandboxUnavailable:
                _docker_ok = None
                result = _fallback_local(tmp, out, timeout, uploads_dir)
        else:
            result = _fallback_local(tmp, out, timeout, uploads_dir)

        if not result.get("ok"):
            err = result.get("error", "unknown error")
            tail = (result.get("stdout") or "")[-500:]
            raise FactorError(f"{err}\n--- script output ---\n{tail}" if tail else err)

        # persist artifacts with unique names the API can serve (local + S3 if set)
        from app import storage

        stored = []
        data_artifacts = []
        for fn in result.get("artifacts", []):
            src = out / fn
            if src.is_file():
                dest = f"{uuid.uuid4().hex[:8]}_{fn}"
                storage.put(src, dest)
                stored.append(dest)
                if fn.lower().endswith(".npz"):
                    data_artifacts.append(dest)
        result["artifacts"] = stored
        # keep any saved .npz arrays so a later viz run can mount them
        if data_artifacts:
            result["data_artifacts"] = data_artifacts
        result["elapsed_s"] = round(_time.time() - t0, 2)
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _parse(p: subprocess.CompletedProcess) -> dict:
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        try:
            return json.loads(ln)
        except json.JSONDecodeError:
            continue
    raise SandboxUnavailable(f"runner produced no JSON. stderr: {p.stderr[:300]}")


def _sess_file(uploads_dir: Path | None) -> Path | None:
    """Return the path to the first file in the uploads directory, if any."""
    if uploads_dir is not None:
        ud = Path(uploads_dir)
        if ud.is_dir():
            for f in sorted(ud.iterdir()):
                if f.is_file():
                    return f
    return None


def _exec_docker(
    tmp: Path, out: Path, timeout: int, uploads_dir: Path | None, image: str = IMAGE
) -> dict:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--memory",
        _DOCKER_MEM,
        "--cpus",
        _DOCKER_CPUS,
        "--pids-limit",
        "256",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        # runner + market data are MOUNTED, so code/data updates need no rebuild
        "-v",
        f"{RUNNER}:/app/runner.py:ro",
    ]
    sess_file = _sess_file(uploads_dir)
    if sess_file:
        cmd += ["-v", f"{sess_file}:/data/default:ro"]
        cmd += ["-e", "DATA_PATH=/data/default"]
    else:
        cmd += ["-v", f"{DATA_DIR}:/data:ro"]
        cmd += ["-e", "DATA_PATH=/data/market.npz"]
    cmd += [
        "-v",
        f"{tmp}:/in:ro",
        "-v",
        f"{out}:/out",
    ]
    if uploads_dir and uploads_dir.is_dir():
        cmd += ["-v", f"{uploads_dir}:/uploads:ro"]
    cmd += [image, "/in/research.py"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise FactorError("script timed out in sandbox (possible infinite loop)") from e
    return _parse(p)


def _exec_local(tmp: Path, out: Path, timeout: int, uploads_dir: Path | None) -> dict:
    sess_file = _sess_file(uploads_dir)
    data_path = str(sess_file or DATA_DIR / "market.npz")
    env = dict(
        os.environ,
        ARTIFACTS_DIR=str(out),
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(tmp / "mpl"),
        DATA_PATH=data_path,
        UPLOADS_DIR=str(uploads_dir or ""),
    )
    try:
        p = subprocess.run(
            [sys.executable, str(RUNNER), str(tmp / "research.py")],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise FactorError("script timed out (possible infinite loop)") from e
    return _parse(p)


def _fallback_local(tmp: Path, out: Path, timeout: int, uploads_dir: Path | None) -> dict:
    """Run in-process only if ALLOW_INPROCESS is explicitly set. Refuses otherwise."""
    import logging

    log = logging.getLogger(__name__)

    if not os.getenv("ALLOW_INPROCESS", "").strip().lower() in ("1", "true", "yes"):
        raise SandboxUnavailable(
            "Docker unavailable and ALLOW_INPROCESS is not set. "
            "Set ALLOW_INPROCESS=1 to enable local execution (insecure — "
            "agent code runs with full host access)."
        )
    log.warning(
        "Running agent code in-process (ALLOW_INPROCESS=1) — no sandbox "
        "isolation. This is insecure and should only be used in development."
    )
    result = _exec_local(tmp, out, timeout, uploads_dir)
    result["engine"] = "inprocess"
    return result
