"""Real market dataset — daily bars via yfinance, cached as one .npz.

Stores ONLY raw point-in-time panels (adjusted close, volume, daily returns) plus
next-period forward returns for grading — NO pre-computed features. The agent
derives every feature itself from these panels. The cache
(backend/data/market.npz) is mounted read-only into the sandbox.

The universe and history length are config, not hardcoded logic:
    ALPHASEEK_UNIVERSE   comma-separated tickers (default: a liquid large-cap set)
    ALPHASEEK_YEARS      years of daily history to download (default 6)
Rebuild by deleting the cache file.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

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
TICKERS = [t.strip().upper() for t in
           os.getenv("ALPHASEEK_UNIVERSE", _DEFAULT_UNIVERSE).split(",") if t.strip()]
DEFAULT_YEARS = int(os.getenv("ALPHASEEK_YEARS", "6"))

_build_state = {"status": "missing", "error": ""}
_lock = threading.Lock()


def build(years: int = DEFAULT_YEARS) -> dict:
    """Download the raw dataset; write NPZ. Returns summary metadata."""
    import pandas as pd
    import yfinance as yf

    raw = yf.download(TICKERS, period=f"{years}y", interval="1d",
                      auto_adjust=True, progress=False)
    close: pd.DataFrame = raw["Close"].dropna(axis=1, thresh=int(len(raw) * 0.9))
    volume: pd.DataFrame = raw["Volume"][close.columns]
    ret = close.pct_change()

    fwd = ret.shift(-1)

    # We store ONLY raw, point-in-time market data — NO pre-defined features. The
    # agent derives and computes every feature/signal itself from these panels,
    # using logic it takes from the papers. Row t is knowable at t (past-aligned);
    # forward returns (fwd) stay hidden for grading, and a look-ahead IC guard in
    # the runner rejects any signal that implies peeking at the future.
    valid = slice(130, -1)
    arrays = {
        "px_close": np.nan_to_num(close.values[valid], nan=0.0),
        "px_volume": np.nan_to_num(volume.values[valid], nan=0.0),
        "px_returns": np.nan_to_num(ret.values[valid], nan=0.0),   # ret[t]=close[t]/close[t-1]-1
        "fwd": np.nan_to_num(fwd.values[valid], nan=0.0),
        "tickers": np.array(list(close.columns)),
        "dates": np.array([d.strftime("%Y-%m-%d") for d in close.index[valid]]),
    }

    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(NPZ_PATH, **arrays)
    T, N = arrays["fwd"].shape
    return {"days": T, "stocks": N,
            "inputs": sorted(k[3:] for k in arrays if k.startswith("px_"))}


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
    if NPZ_PATH.exists() and _build_state["status"] != "building":
        _build_state["status"] = "ready"
    return dict(_build_state)


def dataset_meta() -> dict:
    """Introspect whatever is actually in the cache — no assumptions about which
    columns exist. Works for the default market data or any other panel set."""
    if not NPZ_PATH.exists():
        return {"source": "missing"}
    with np.load(NPZ_PATH, allow_pickle=False) as z:
        T, N = z["fwd"].shape
        cols = [k[3:] for k in z.files if k.startswith("px_")]   # discovered, not fixed
        return {"source": "real", "days": int(T), "stocks": int(N),
                "start": str(z["dates"][0]), "end": str(z["dates"][-1]),
                "inputs": sorted(cols)}
