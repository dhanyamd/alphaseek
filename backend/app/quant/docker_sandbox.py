"""Run agent-written research SCRIPTS — Docker-hardened, artifact-collecting.

Both engines execute the SAME `sandbox/runner.py` contract (script imports
`alphaseek_data`, calls ad.submit(signal), may print + save charts):

  * docker     — hardened container: no network, read-only fs, non-root,
                 memory/CPU caps. /in (code, ro) and /out (artifacts, rw) mounts.
  * inprocess  — dev fallback: the same runner via the local Python.

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

IMAGE = "alphaseek-sandbox:latest"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "sandbox" / "runner.py"
ARTIFACT_STORE = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACT_STORE.mkdir(exist_ok=True)


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
        p = subprocess.run(["docker", "version", "--format", "{{.Server.Os}}"],
                           capture_output=True, text=True, timeout=10)
        _docker_ok = p.returncode == 0 and bool(p.stdout.strip())
    except Exception:  # noqa: BLE001
        _docker_ok = False
    return _docker_ok


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def run_factor_code(code: str, seed_num: int = 7, timeout: int = 120,
                    uploads_dir: Path | None = None, image: str = IMAGE,
                    manifest_src: Path | None = None) -> dict:
    """Execute an agent research script; return metrics + stdout + artifacts.

    `image` selects a provisioned sandbox image (base or a dep-layered one).
    `manifest_src` mounts a prior run's result manifest at /in/manifest.json so
    a visualization script can load it via ad.manifest() without recomputing.
    """
    global _docker_ok
    import time as _time
    t0 = _time.time()
    tmp = Path(tempfile.mkdtemp(prefix="alphaseek_"))
    out = tmp / "out"
    out.mkdir()
    (tmp / "research.py").write_text(code)
    if manifest_src and Path(manifest_src).is_file():
        shutil.copy(manifest_src, tmp / "manifest.json")
    try:
        if docker_available():
            try:
                result = _exec_docker(tmp, out, timeout, uploads_dir, image)
                result["engine"] = "docker"
            except SandboxUnavailable:
                _docker_ok = None
                result = _exec_local(tmp, out, timeout, uploads_dir)
                result["engine"] = "inprocess"
        else:
            result = _exec_local(tmp, out, timeout, uploads_dir)
            result["engine"] = "inprocess"

        if not result.get("ok"):
            err = result.get("error", "unknown error")
            tail = (result.get("stdout") or "")[-500:]
            raise FactorError(f"{err}\n--- script output ---\n{tail}" if tail else err)
        result.setdefault("submitted", True)

        # persist artifacts with unique names the API can serve (local + S3 if set)
        from app import storage
        stored = []
        for fn in result.get("artifacts", []):
            src = out / fn
            if src.is_file():
                dest = f"{uuid.uuid4().hex[:8]}_{fn}"
                storage.put(src, dest)
                stored.append(dest)
        result["artifacts"] = stored
        # keep the result manifest so a later viz run can mount it
        man = out / "manifest.json"
        if man.is_file():
            dest = ARTIFACT_STORE / f"{uuid.uuid4().hex[:8]}_manifest.json"
            shutil.copy(man, dest)
            result["manifest_path"] = str(dest)
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


def _exec_docker(tmp: Path, out: Path, timeout: int, uploads_dir: Path | None,
                 image: str = IMAGE) -> dict:
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "2g", "--cpus", "2", "--pids-limit", "256",
        "--read-only", "--tmpfs", "/tmp",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        # runner + market data are MOUNTED, so code/data updates need no rebuild
        "-v", f"{RUNNER}:/app/runner.py:ro",
        "-v", f"{DATA_DIR}:/data:ro",
        "-v", f"{tmp}:/in:ro", "-v", f"{out}:/out",
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
    env = dict(os.environ, ARTIFACTS_DIR=str(out), MPLBACKEND="Agg",
               MPLCONFIGDIR=str(tmp / "mpl"), DATA_PATH=str(DATA_DIR / "market.npz"),
               UPLOADS_DIR=str(uploads_dir or ""),
               MANIFEST_PATH=str(tmp / "manifest.json"))
    try:
        p = subprocess.run(
            [sys.executable, str(RUNNER), str(tmp / "research.py")],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise FactorError("script timed out (possible infinite loop)") from e
    return _parse(p)
