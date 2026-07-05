
"""In-container runner — executes an agent-written RESEARCH SCRIPT on REAL data.

The script imports the built-in data API (mirroring QuantPad's quantpad_data):

    import alphaseek_data as ad
    f = ad.features()                        # dict of (T,N) arrays — real US equities
    res = ad.backtest(signal)                # metrics for any candidate signal
    ad.submit(signal)                        # final signal for official grading
    plt.savefig(f"{ad.ARTIFACTS}/x.png")     # charts collected as artifacts
    ad.uploads()                             # user-uploaded files, if any

Market data is loaded from DATA_PATH (an .npz mounted read-only). Forward
returns are NEVER exposed raw — signals are graded blind, so scripts cannot
cheat. Captures stdout, collects artifacts, prints one JSON result line.
"""
import io
import json
import os
import sys
import types
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

ART_DIR = os.environ.get("ARTIFACTS_DIR", "/out")
DATA_PATH = os.environ.get("DATA_PATH", "/data/market.npz")
UPLOADS_DIR = os.environ.get("UPLOADS_DIR", "/uploads")
MANIFEST_OUT = os.path.join(ART_DIR, "manifest.json")   # coder writes here
MANIFEST_IN = os.environ.get("MANIFEST_PATH", "/in/manifest.json")  # viz reads here


def _jsonify(x):
    """Make numpy/pandas objects JSON-serializable for the result manifest."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    try:                                    # pandas Series/Index/etc.
        import pandas as pd
        if isinstance(x, (pd.Series, pd.Index)):
            return x.tolist()
        if isinstance(x, pd.DataFrame):
            return {str(c): x[c].tolist() for c in x.columns}
    except Exception:  # noqa: BLE001
        pass
    return x


def _std(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def _rank(x):
    order = np.asarray(x).argsort(1).argsort(1).astype(float)
    return 2 * (order / (x.shape[1] - 1)) - 1


def load_market():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"market dataset not found at {DATA_PATH} — build it on the backend first")
    z = np.load(DATA_PATH, allow_pickle=False)
    feats = {k[2:]: z[k].astype(float) for k in z.files if k.startswith("f_")}
    fwd = z["fwd"].astype(float)
    tickers = [str(t) for t in z["tickers"]] if "tickers" in z.files else []
    dates = [str(d) for d in z["dates"]] if "dates" in z.files else []
    raw = {k[3:]: z[k].astype(float) for k in z.files if k.startswith("px_")}
    return feats, fwd, tickers, dates, raw


def compute_metrics(signal, fwd):
    s = _std(np.nan_to_num(np.asarray(signal, dtype=float)))
    w = s - s.mean(1, keepdims=True)
    w = w / (np.abs(w).sum(1, keepdims=True) + 1e-9)
    port = (w * fwd).sum(1)
    sr, fr = _rank(signal), _rank(fwd)
    sr -= sr.mean(1, keepdims=True); fr -= fr.mean(1, keepdims=True)
    ic = (sr * fr).sum(1) / (np.sqrt((sr**2).sum(1) * (fr**2).sum(1)) + 1e-9)
    mean_ic = float(np.nanmean(ic))
    if abs(mean_ic) > 0.15:                 # look-ahead guard
        raise ValueError(
            f"Look-ahead detected: mean rank-IC {mean_ic:.3f} is implausibly high "
            "(real equity factors run ~0.01-0.05). Your signal is using FUTURE "
            "information. Build signals causally from ad.prices()/ad.returns() using "
            "PAST rows only (e.g. ad.roll_mean); never index future rows or use fwd.")
    eq = np.cumprod(1 + np.nan_to_num(port))
    dd = eq / np.maximum.accumulate(eq) - 1
    third = max(len(ic) // 3, 1)
    # net of costs: 5 bps per unit of daily turnover (simple linear cost model)
    turn_series = np.abs(np.diff(w, axis=0)).sum(1)
    port_net = port.copy()
    port_net[1:] -= 0.0005 * turn_series
    return {
        "sharpe": float(port.mean() / (port.std() + 1e-12) * np.sqrt(252)),
        "sharpe_net": float(port_net.mean() / (port_net.std() + 1e-12) * np.sqrt(252)),
        "ann_return": float((1 + port).prod() ** (252 / len(port)) - 1),
        "max_drawdown": float(dd.min()),
        "hit_rate": float((port > 0).mean()),
        "mean_ic": float(np.nanmean(ic)),
        "ic_early": float(np.nanmean(ic[:third])),
        "ic_late": float(np.nanmean(ic[-third:])),
        "ic_decay": float(np.nanmean(ic[-third:]) - np.nanmean(ic[:third])),
        "turnover": float(np.abs(np.diff(w, axis=0)).sum(1).mean()),
        "equity_curve": [round(float(x), 4) for x in eq[:: max(1, len(eq) // 120)]],
        "n_days": int(len(port)),
    }


def _roll_mean(x, w):
    """Causal rolling mean along time (axis 0): value at t uses rows [t-w+1, t].

    Works on a 1-D series (e.g. market returns) or a 2-D (T, N) panel."""
    x = np.asarray(x, dtype=float)
    flat = x.ndim == 1
    if flat:
        x = x[:, None]
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.full_like(x, np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    out[:w - 1] = out[w - 1]
    return out[:, 0] if flat else out


def _roll_std(x, w):
    """Causal rolling std along time (axis 0)."""
    m = _roll_mean(x, w)
    m2 = _roll_mean(np.asarray(x, dtype=float) ** 2, w)
    return np.sqrt(np.maximum(m2 - m**2, 1e-12))


PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#111112", "plot_bgcolor": "#111112",
    "font": {"family": "Geist, system-ui, sans-serif", "size": 12, "color": "#d4d4d8"},
    "colorway": ["#fafafa", "#7ee787", "#79c0ff", "#ff7b72", "#d2a8ff", "#ffa657"],
    "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
}


def make_data_module(feats, fwd, tickers, dates, raw=None):
    mod = types.ModuleType("alphaseek_data")
    submitted = {}
    raw = raw or {}

    # Equal-weight market return, PAST-aligned: entry t is the return realized
    # over day t-1 -> t (knowable at t, same timing as the features). Safe for
    # regime analysis; contains no future information relative to grading.
    mkt = np.zeros(fwd.shape[0])
    mkt[1:] = fwd.mean(1)[:-1]

    def market_returns():
        return mkt.copy()

    def monte_carlo(signal, n_paths=400, seed=0):
        """Bootstrap the signal's daily portfolio returns into equity paths.

        Returns dict with downsampled sample paths and percentile bands (p5/p50/p95)
        plus terminal stats — everything needed for a QuantPad-style fan chart.
        Only aggregate portfolio returns are used; forward returns stay hidden.
        """
        s = _std(np.nan_to_num(np.asarray(signal, dtype=float)))
        w = s - s.mean(1, keepdims=True)
        w = w / (np.abs(w).sum(1, keepdims=True) + 1e-9)
        port = (w * fwd).sum(1)
        rng = np.random.default_rng(seed)
        T = len(port)
        idx = rng.integers(0, T, size=(int(n_paths), T))
        paths = np.cumprod(1 + port[idx], axis=1)
        step = max(1, T // 100)
        x = list(range(0, T, step))              # day index for every series below
        terminal = paths[:, -1]
        out = {
            "x": x,
            "sample_paths": [p[::step].round(4).tolist() for p in paths[:40]],
            "terminal_mean": float(terminal.mean()),
            "terminal_p5": float(np.percentile(terminal, 5)),
            "terminal_p95": float(np.percentile(terminal, 95)),
            "prob_loss": float((terminal < 1.0).mean()),
            "max_drawdown_p95": float(np.percentile(
                (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1), 5)),
        }
        for q in (5, 25, 50, 75, 95):            # p5..p95 — all same length as x
            out[f"p{q}"] = np.percentile(paths, q, axis=0)[::step].round(4).tolist()
        return out

    def _validated_backtest(signal):
        sig = np.asarray(signal, dtype=float)
        if sig.shape != fwd.shape:
            raise ValueError(
                f"ad.backtest(signal) needs the FULL history: shape {fwd.shape}, got {sig.shape}. "
                "Never slice the time axis — compute your signal on the complete (T, N) arrays; "
                "to vary a lookback, transform the full array (e.g. rolling ops), not a slice.")
        return compute_metrics(sig, fwd)

    # RAW point-in-time panels, SCHEMA-DRIVEN — the agent inspects whatever
    # columns exist (they differ per dataset) and computes its own features.
    mod.data = {k: v.copy() for k, v in raw.items()}   # {column_name: (T, N) array}
    mod.columns = sorted(raw)

    def describe():
        """Runtime schema of the loaded data — inspect before writing code."""
        lines = [f"rows(T)={fwd.shape[0]} cols(N)={fwd.shape[1]} tickers/dates provided"]
        for k in sorted(raw):
            v = raw[k]
            last = v[-1][np.isfinite(v[-1])]
            rng = f"[{last.min():.4g}, {last.max():.4g}]" if last.size else "[]"
            lines.append(f"  data['{k}']: (T,N) float, last-row range {rng}")
        return "\n".join(lines)

    mod.describe = describe
    # Convenience aliases for the common market columns (still schema-checked).
    mod.prices = lambda: raw["close"].copy() if "close" in raw else None
    mod.volume = lambda: raw["volume"].copy() if "volume" in raw else None
    mod.returns = lambda: raw["returns"].copy() if "returns" in raw else None
    mod.backtest = _validated_backtest
    mod.submit = lambda signal: submitted.__setitem__("signal", np.asarray(signal, dtype=float))
    mod.rank = _rank
    mod.zscore = _std
    mod.FEATURES = sorted(feats)
    mod.TICKERS = tickers
    mod.DATES = dates
    mod.ARTIFACTS = ART_DIR
    def ic_series(signal):
        """Daily rank-IC of a (T,N) signal vs next-day returns — for stability charts."""
        sr, fr = _rank(signal), _rank(fwd)
        sr = sr - sr.mean(1, keepdims=True); fr = fr - fr.mean(1, keepdims=True)
        return (sr * fr).sum(1) / (np.sqrt((sr**2).sum(1) * (fr**2).sum(1)) + 1e-9)

    # --- result manifest: the seam between the math stage and the viz stage ---
    manifest: dict = {}

    def record(**named):
        """Save named results (metrics, arrays, series) for the Visualizer.

        The math stage calls ad.record(equity_scaled=..., sweep_grid=...) with
        everything a chart could need. The Visualizer later loads these via
        ad.manifest() and plots them — it never recomputes the math."""
        manifest.update({k: _jsonify(v) for k, v in named.items()})
        with open(MANIFEST_OUT, "w") as fh:
            json.dump(manifest, fh)
        return sorted(manifest)

    def load_manifest():
        """Load the recorded manifest (viz stage). Numeric fields come back as
        lists — wrap in np.asarray as needed."""
        if not os.path.isfile(MANIFEST_IN):
            raise FileNotFoundError(
                "ad.manifest() found no manifest — the math stage must call "
                "ad.record(...) first. This script runs in the visualization stage.")
        with open(MANIFEST_IN) as fh:
            return json.load(fh)

    mod.record = record
    mod.manifest = load_manifest
    mod.market_returns = market_returns
    mod.monte_carlo = monte_carlo
    mod.ic_series = ic_series
    mod.roll_mean = _roll_mean
    mod.roll_std = _roll_std
    mod.PLOTLY_LAYOUT = dict(PLOTLY_LAYOUT)
    mod.uploads = lambda: (
        [os.path.join(UPLOADS_DIR, f) for f in sorted(os.listdir(UPLOADS_DIR))]
        if os.path.isdir(UPLOADS_DIR) else []
    )
    mod.__doc__ = "AlphaSeek data API: features(), backtest(), submit(), rank, zscore, uploads()"
    return mod, submitted


def main():
    code_path = sys.argv[1]
    with open(code_path) as fh:
        code = fh.read()

    os.makedirs(ART_DIR, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    try:  # default dark theme so charts match the UI; scripts may override
        import matplotlib as mpl
        mpl.rcParams.update({
            "figure.facecolor": "#111112", "axes.facecolor": "#111112",
            "savefig.facecolor": "#111112", "axes.edgecolor": "#3a3a3e",
            "axes.labelcolor": "#d4d4d8", "text.color": "#d4d4d8",
            "xtick.color": "#8a8a8f", "ytick.color": "#8a8a8f",
            "grid.color": "#26262a", "axes.grid": True, "grid.alpha": 0.6,
            "axes.prop_cycle": mpl.cycler(color=[
                "#fafafa", "#7ee787", "#79c0ff", "#ff7b72", "#d2a8ff", "#ffa657"]),
            "figure.dpi": 110, "font.size": 9,
        })
    except Exception:
        pass

    stdout = io.StringIO()
    try:
        feats, fwd, tickers, dates, raw = load_market()
        mod, submitted = make_data_module(feats, fwd, tickers, dates, raw)
        sys.modules["alphaseek_data"] = mod
        ns = {"__name__": "__main__"}
        with redirect_stdout(stdout), redirect_stderr(stdout):
            exec(compile(code, "research.py", "exec"), ns)
        if "signal" not in submitted and callable(ns.get("factor")):
            submitted["signal"] = np.asarray(ns["factor"](feats), dtype=float)
        if "signal" in submitted:
            signal = submitted["signal"]
            if signal.shape != fwd.shape:
                raise ValueError(f"submitted signal shape {signal.shape}, expected {fwd.shape}")
            result = compute_metrics(signal, fwd)
            result["ok"] = True
            result["submitted"] = True
        else:
            # exploration run — no final signal yet; stdout/artifacts still returned
            result = {"ok": True, "submitted": False}
    except Exception as e:  # noqa: BLE001
        import traceback
        lines = code.splitlines()
        frames = [fr for fr in traceback.extract_tb(e.__traceback__)
                  if fr.filename == "research.py"]
        loc = ""
        for fr in frames[-2:]:
            src = lines[fr.lineno - 1].strip() if 0 < fr.lineno <= len(lines) else ""
            loc += f"\n  research.py line {fr.lineno}: {src}"
        result = {"ok": False, "error": f"{type(e).__name__}: {e}{loc}"}

    result["stdout"] = stdout.getvalue()[-4000:]
    result["artifacts"] = sorted(
        fn for fn in os.listdir(ART_DIR)
        if fn.lower().endswith((".png", ".svg", ".jpg", ".html"))
    ) if os.path.isdir(ART_DIR) else []
    if os.path.isfile(MANIFEST_OUT):        # keys the Visualizer can chart
        try:
            with open(MANIFEST_OUT) as fh:
                result["manifest_keys"] = sorted(json.load(fh))
        except Exception:  # noqa: BLE001
            pass
    print(json.dumps(result))


if __name__ == "__main__":
    main()
