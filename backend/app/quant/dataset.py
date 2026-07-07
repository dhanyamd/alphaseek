"""Real market dataset — daily bars via yfinance or Polygon.io, cached as one .npz.

Stores ONLY raw point-in-time panels (adjusted close, volume, daily returns) plus
next-period forward returns for grading — NO pre-computed features. The agent
derives every feature itself from these panels. The cache
(backend/data/market.npz) is mounted read-only into the sandbox.

The universe, history length, and data source are config, not hardcoded logic:
    ALPHASEEK_UNIVERSE    comma-separated tickers (default: a liquid large-cap set)
    ALPHASEEK_YEARS       years of daily history to download (default 6)
    ALPHASEEK_BURN_DAYS   warm-up discard (default 130)
    ALPHASEEK_DATA_SOURCE yfinance (default) or polygon
    POLYGON_API_KEY       required when source=polygon (free: 5 calls/min)
Rebuild by deleting the cache file.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NPZ_PATH = DATA_DIR / "market.npz"

# Default universe — a sensible out-of-box set; override entirely via env.
_DEFAULT_UNIVERSE = (
    "AAPL,MSFT,NVDA,GOOGL,META,AMZN,ORCL,CRM,ADBE,INTC,CSCO,AMD,QCOM,TXN,AVGO,IBM,"
    "NOW,INTU,AMAT,MU,T,VZ,TMUS,CMCSA,NFLX,DIS,WMT,PG,KO,PEP,HD,MCD,NKE,SBUX,COST,"
    "TGT,LOW,MDLZ,JPM,BAC,WFC,GS,MS,AXP,C,BLK,SCHW,USB,SPGI,JNJ,PFE,UNH,ABBV,MRK,"
    "TMO,ABT,LLY,DHR,BMY,AMGN,XOM,CVX,COP,SLB,EOG,MPC,PSX,VLO,CAT,GE,HON,UPS,RTX,"
    "BA,UNP,LMT,DE,MMM,NEE,DUK,SO,LIN,FCX,NEM,SHW"
)
TICKERS = [
    t.strip().upper()
    for t in os.getenv("ALPHASEEK_UNIVERSE", _DEFAULT_UNIVERSE).split(",")
    if t.strip()
]
DEFAULT_YEARS = int(os.getenv("ALPHASEEK_YEARS", "6"))
BURN_DAYS = int(os.getenv("ALPHASEEK_BURN_DAYS", "130"))
SOURCE = os.getenv("ALPHASEEK_DATA_SOURCE", "yfinance").strip().lower()
POLYGON_API_KEY = os.getenv("MASSIVE_API_KEY", "") or os.getenv("POLYGON_API_KEY", "").strip()
MASSIVE_API_BASE = os.getenv("MASSIVE_API_BASE", "https://api.massive.com").rstrip("/")
MASSIVE_RATE_LIMIT = int(os.getenv("MASSIVE_RATE_LIMIT", "5"))
MASSIVE_RATE_PERIOD = int(os.getenv("MASSIVE_RATE_PERIOD", "61"))

_build_state = {"status": "missing", "error": ""}
_lock = threading.Lock()


def build(years: int = DEFAULT_YEARS) -> dict:
    """Download the raw dataset from the configured source; write NPZ."""
    if SOURCE == "polygon":
        if not POLYGON_API_KEY:
            raise ValueError(
                "POLYGON_API_KEY not set. Get a free key at https://polygon.io/dashboard/signup"
            )
        return _build_polygon(years)
    return _build_yfinance(years)


def _build_yfinance(years: int) -> dict:
    import pandas as pd
    import yfinance as yf

    raw = yf.download(TICKERS, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    close: pd.DataFrame = raw["Close"].dropna(axis=1, thresh=int(len(raw) * 0.9))
    volume: pd.DataFrame = raw["Volume"][close.columns]
    ret = close.pct_change()
    fwd = ret.shift(-1)

    valid = slice(BURN_DAYS, -1)
    arrays = {
        "px_close": np.nan_to_num(close.values[valid], nan=0.0),
        "px_volume": np.nan_to_num(volume.values[valid], nan=0.0),
        "px_returns": np.nan_to_num(ret.values[valid], nan=0.0),
        "fwd": np.nan_to_num(fwd.values[valid], nan=0.0),
        "tickers": np.array(list(close.columns)),
        "dates": np.array([d.strftime("%Y-%m-%d") for d in close.index[valid]]),
    }

    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(NPZ_PATH, **arrays)
    T, N = arrays["fwd"].shape
    return {"days": T, "stocks": N, "inputs": sorted(k[3:] for k in arrays if k.startswith("px_"))}


# ---------------------------------------------------------------------------
# Polygon.io backend (free tier: 5 API calls/min)
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Token-bucket rate limiter. Ensures at most max_calls per period seconds."""

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self.timestamps: list[float] = []

    def wait(self) -> None:
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.period]
        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.period - (now - self.timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.timestamps.append(time.time())


_massive_limiter = _RateLimiter(MASSIVE_RATE_LIMIT, MASSIVE_RATE_PERIOD)


def _polygon_aggs(ticker: str, from_date: str, to_date: str) -> list[dict]:
    """Fetch daily bars for one ticker via Massive (ex-Polygon) API."""
    _massive_limiter.wait()
    url = f"{MASSIVE_API_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
    resp = requests.get(
        url, params={"apiKey": POLYGON_API_KEY, "adjusted": "true", "sort": "asc", "limit": 5000}
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK" or "results" not in data:
        return []
    return data["results"]


def _build_polygon(years: int) -> dict:
    import pandas as pd

    end = time.strftime("%Y-%m-%d")
    start_dt = time.localtime(time.time() - years * 365.25 * 86400)
    start = time.strftime("%Y-%m-%d", start_dt)

    ticker_data: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        results = _polygon_aggs(ticker, start, end)
        if not results:
            continue
        df = pd.DataFrame(
            {
                "close": [r["c"] for r in results],
                "volume": [r["v"] for r in results],
            },
            index=pd.to_datetime([r["t"] for r in results], unit="ms"),
        )
        ticker_data[ticker] = df

    if not ticker_data:
        raise ValueError("Polygon returned zero tickers — check your API key and ticker symbols.")

    all_dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in ticker_data.values()])))

    close_frames: list[pd.DataFrame] = []
    volume_frames: list[pd.DataFrame] = []
    for ticker, df in ticker_data.items():
        aligned = df.reindex(all_dates)
        close_frames.append(aligned["close"])
        volume_frames.append(aligned["volume"])

    close = pd.concat(close_frames, axis=1)
    close.columns = list(ticker_data.keys())
    volume = pd.concat(volume_frames, axis=1)
    volume.columns = list(ticker_data.keys())

    close = close.dropna(axis=1, thresh=int(len(close) * 0.9))
    volume = volume[close.columns]
    ret = close.pct_change()
    fwd = ret.shift(-1)

    valid = slice(BURN_DAYS, -1)
    arrays = {
        "px_close": np.nan_to_num(close.values[valid], nan=0.0),
        "px_volume": np.nan_to_num(volume.values[valid], nan=0.0),
        "px_returns": np.nan_to_num(ret.values[valid], nan=0.0),
        "fwd": np.nan_to_num(fwd.values[valid], nan=0.0),
        "tickers": np.array(list(close.columns)),
        "dates": np.array([d.strftime("%Y-%m-%d") for d in close.index[valid]]),
    }

    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(NPZ_PATH, **arrays)
    T, N = arrays["fwd"].shape
    return {"days": T, "stocks": N, "inputs": sorted(k[3:] for k in arrays if k.startswith("px_"))}


def ensure_dataset_async() -> None:
    """Kick off a background build if the cache is missing (non-blocking)."""
    if NPZ_PATH.exists():
        _build_state["status"] = "ready"
        return

    def worker() -> None:
        with _lock:
            if NPZ_PATH.exists():
                _build_state["status"] = "ready"
                return
            _build_state["status"] = "building"
            try:
                meta = build()
                _build_state.update(status="ready", **{})
                _build_state["meta"] = meta
            except Exception as e:  # noqa: BLE001
                _build_state.update(status="error", error=str(e)[:300])

    threading.Thread(target=worker, daemon=True).start()


def dataset_status() -> dict:
    if NPZ_PATH.exists():
        _build_state["status"] = "ready"
    d = dict(_build_state)
    d["data_source"] = SOURCE
    return d


def dataset_meta() -> dict:
    """Introspect whatever is actually in the cache — no assumptions about which
    columns exist. Works for the default market data or any other panel set."""
    if not NPZ_PATH.exists():
        return {"source": "missing", "data_source": SOURCE}
    with np.load(NPZ_PATH, allow_pickle=False) as z:
        _RESERVED = {"fwd", "tickers", "dates"}
        _PREFIX = "px_"
        px_keys = [k for k in z.files if k.startswith(_PREFIX)]
        if px_keys:
            cols = [k[len(_PREFIX) :] for k in px_keys]
        else:
            # No px_ convention — treat every non-metadata key as a panel
            px_keys = [k for k in z.files if k not in _RESERVED and not k.startswith("f_")]
            cols = list(px_keys)

        # Discover shape from fwd if it exists, else from the first panel
        T, N = 0, 0
        if "fwd" in z.files:
            arr = z["fwd"]
            if arr.ndim == 2:
                T, N = arr.shape
            elif arr.ndim == 1:
                T, N = arr.shape[0], 1
        elif px_keys:
            arr = z[px_keys[0]]
            if arr.ndim == 2:
                T, N = arr.shape
            elif arr.ndim == 1:
                T, N = arr.shape[0], 1

        start = str(z["dates"][0]) if "dates" in z.files and len(z["dates"]) > 0 else "?"
        end = str(z["dates"][-1]) if "dates" in z.files and len(z["dates"]) > 0 else "?"
        return {
            "source": "real",
            "data_source": SOURCE,
            "days": int(T),
            "stocks": int(N),
            "start": start,
            "end": end,
            "inputs": sorted(cols),
        }
