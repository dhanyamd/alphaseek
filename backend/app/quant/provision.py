"""Dependency provisioning — the agent picks libraries, we install them safely.

The sandbox that runs agent-written research code has NO network (so the code
can never exfiltrate or attack). But the agent must be free to use any quant
library. We reconcile the two by splitting resolution from execution:

    agent declares requirements  ->  we build a cached image LAYER that pip-
                                     installs them (network ON, PyPI only, no
                                     research code runs)  ->  the research code
                                     executes in that image with network OFF.

Installs are gated to a known quant/scientific allowlist — an unknown package
name is skipped and surfaced, never silently installed (supply-chain safety).
The base image already ships the common stack, so most runs provision nothing.
"""

from __future__ import annotations

import hashlib
import re
import subprocess

BASE_IMAGE = "alphaseek-sandbox:base"

# Already baked into the base image — never need provisioning.
PREINSTALLED = {
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "sklearn",
    "statsmodels",
    "empyrical-reloaded",
    "empyrical",
    "alphalens-reloaded",
    "alphalens",
    "arch",
    "cvxpy",
    "quantstats",
    "matplotlib",
    "plotly",
    "cvxportfolio",
    "skfolio",
    "riskfolio-lib",
    "pyportfolioopt",
}

# Human-readable description of the quant stack for agent prompts — derived
# from PREINSTALLED so the two can never drift.
STACK_STRING = (
    "numpy, pandas, scipy, scikit-learn, statsmodels, "
    "empyrical (empyrical-reloaded), alphalens (alphalens-reloaded), "
    "arch (GARCH), cvxpy, quantstats, "
    "cvxportfolio, skfolio, riskfolio-lib, pyportfolioopt (import as pypfopt)"
)

# Vetted quant/scientific packages the agent may pull in on demand. Anything
# outside this set is skipped (and reported) rather than installed blindly.
ALLOWLIST = PREINSTALLED | {
    "vectorbt",
    "backtrader",
    "pyportfolioopt",
    "riskfolio-lib",
    "riskparityportfolio",
    "ffn",
    "pandas-ta",
    "ta",
    "yfinance",
    "pandas-datareader",
    "pykalman",
    "hmmlearn",
    "linearmodels",
    "cvxopt",
    "numba",
    "seaborn",
    "networkx",
    "scikit-optimize",
    "hurst",
    "tslearn",
    "pmdarima",
    "filterpy",
    "quandl",
    "empyrical-reloaded",
    "alphalens-reloaded",
    "jquantstats",
    # Production portfolio construction
    "cvxportfolio",
    "skfolio",
}

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")


class ProvisionResult:
    def __init__(
        self, image: str, installed: list[str], skipped: list[str], cached: bool, error: str = ""
    ):
        self.image = image
        self.installed = installed
        self.skipped = skipped  # (name, reason) rendered as strings
        self.cached = cached
        self.error = error


def _norm(pkg: str) -> str:
    """Bare distribution name, lowercased (drop version pins/extras)."""
    return re.split(r"[<>=!~\[ ]", pkg.strip(), 1)[0].lower()


def provision(requirements: list[str], timeout: int = 300) -> ProvisionResult:
    """Return a sandbox image that has `requirements` installed.

    Base image when nothing extra is needed; otherwise a cached layer tagged by
    the requirement-set hash. Never raises for a bad package — reports it.
    """
    wanted, skipped = [], []
    for raw in requirements or []:
        name = _norm(raw)
        if not name or not _NAME.match(name):
            skipped.append(f"{raw} (invalid name)")
        elif name in PREINSTALLED:
            continue  # already in the base image
        elif name not in ALLOWLIST:
            skipped.append(f"{raw} (not on quant allowlist)")
        else:
            wanted.append(name)

    wanted = sorted(set(wanted))
    if not wanted:
        return ProvisionResult(BASE_IMAGE, [], skipped, cached=True)

    tag = "alphaseek-sandbox:" + hashlib.sha1(",".join(wanted).encode()).hexdigest()[:12]
    if _image_exists(tag):
        return ProvisionResult(tag, wanted, skipped, cached=True)

    dockerfile = (
        f"FROM {BASE_IMAGE}\nUSER root\n"
        f"RUN pip install --no-cache-dir {' '.join(wanted)}\nUSER sandbox\n"
    )
    try:
        p = subprocess.run(
            ["docker", "build", "-t", tag, "-"],
            input=dockerfile,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ProvisionResult(
            BASE_IMAGE, [], skipped, cached=False, error=f"provision timed out installing {wanted}"
        )
    if p.returncode != 0:
        return ProvisionResult(
            BASE_IMAGE, [], skipped, cached=False, error=(p.stderr or p.stdout)[-400:]
        )
    return ProvisionResult(tag, wanted, skipped, cached=False)


def _image_exists(tag: str) -> bool:
    try:
        p = subprocess.run(
            ["docker", "image", "inspect", tag], capture_output=True, text=True, timeout=10
        )
        return p.returncode == 0
    except Exception:  # noqa: BLE001
        return False
