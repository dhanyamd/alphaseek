
"""In-container runner — executes an agent-written RESEARCH SCRIPT on REAL data.

Minimal contract (the agent writes all its own code with pandas/numpy/scipy/...):

    import alphaseek as af
    d = np.load(af.DATA)                      # raw market panels — load + inspect
    sig = ...                                 # the agent's own (T,N) signal
    m = af.submit(sig)                        # OFFICIAL graded metrics (call once)
    np.savez(f"{af.OUT}/manifest.npz", ...)   # arrays for the visualization stage

Forward returns are NEVER exposed — signals are graded against hidden data and a
look-ahead guard rejects future-peeking. Captures stdout, collects artifacts,
prints one JSON result line.
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
MANIFEST_OUT = os.path.join(ART_DIR, "manifest.npz")    # coder saves arrays here
MANIFEST_IN = os.environ.get("MANIFEST_PATH", "/in/manifest.npz")  # viz reads here




def _std(x):
    """Cross-sectional z-score. Works on a single day (1-D, N) or a panel (T, N)."""
    x = np.asarray(x, dtype=float)
    flat = x.ndim == 1
    if flat:
        x = x[None, :]
    out = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
    return out[0] if flat else out


def _rank(x):
    """Cross-sectional rank in [-1, 1]. Works on a single day (1-D) or a panel (T, N)."""
    x = np.asarray(x, dtype=float)
    flat = x.ndim == 1
    if flat:
        x = x[None, :]
    order = x.argsort(1).argsort(1).astype(float)
    out = 2 * (order / (x.shape[1] - 1)) - 1
    return out[0] if flat else out


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
            "information. Build it CAUSALLY from the raw panels (past rows only); "
            "never index future rows.")
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






def make_alphaseek(fwd, raw, tickers, dates):
    """The MINIMAL sandbox contract. The agent writes all the code itself with
    pandas/numpy/scipy/sklearn; the sandbox only provides:
      alphaseek.DATA  -> path to a .npz of raw market panels (load + inspect it)
      alphaseek.OUT   -> directory to save artifacts + a manifest.npz for charts
      alphaseek.backtest(signal) -> metrics dict (evaluation, no side effects)
      alphaseek.submit(signal)   -> the OFFICIAL graded metrics (call once)
      alphaseek.manifest()       -> arrays saved by the math stage (viz stage)
      alphaseek.uploads()        -> any user-uploaded file paths
    Forward returns stay hidden; a look-ahead guard rejects future-peeking signals.
    """
    mod = types.ModuleType("alphaseek")
    submitted = {}

    # A public data file the agent loads itself — raw panels only, NO fwd.
    public = {f"px_{k}": v for k, v in raw.items()}
    public["tickers"] = np.asarray(tickers)
    public["dates"] = np.asarray(dates)
    data_path = "/tmp/market.npz"
    np.savez(data_path, **public)

    def _grade(signal):
        sig = np.asarray(signal, dtype=float)
        if sig.shape != fwd.shape:
            raise ValueError(
                f"signal must be the FULL (T, N) panel {fwd.shape}, got {sig.shape}. "
                "Compute it on the complete arrays; never slice the time axis.")
        return compute_metrics(sig, fwd)          # includes the look-ahead guard

    def submit(signal):
        metrics = _grade(signal)                   # grade first (may raise)
        submitted["signal"] = np.asarray(signal, dtype=float)
        submitted["metrics"] = metrics
        return metrics

    def manifest():
        if not os.path.isfile(MANIFEST_IN):
            raise FileNotFoundError(
                "alphaseek.manifest() found nothing — the math stage must save arrays "
                "to alphaseek.OUT/manifest.npz (np.savez) first.")
        return dict(np.load(MANIFEST_IN))

    mod.DATA = data_path
    mod.OUT = ART_DIR
    mod.backtest = _grade
    mod.submit = submit
    mod.manifest = manifest
    mod.uploads = lambda: (
        [os.path.join(UPLOADS_DIR, f) for f in sorted(os.listdir(UPLOADS_DIR))]
        if os.path.isdir(UPLOADS_DIR) else [])
    mod.__doc__ = ("alphaseek sandbox: DATA (npz path), OUT (dir), backtest(signal), "
                   "submit(signal), manifest(), uploads().")
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
        _feats, fwd, tickers, dates, raw = load_market()
        mod, submitted = make_alphaseek(fwd, raw, tickers, dates)
        sys.modules["alphaseek"] = mod
        ns = {"__name__": "__main__"}
        with redirect_stdout(stdout), redirect_stderr(stdout):
            exec(compile(code, "research.py", "exec"), ns)
        if "signal" in submitted:
            result = dict(submitted["metrics"])     # already graded in submit()
            result["ok"] = True
            result["submitted"] = True
        else:
            # exploration run — no submit yet; stdout/artifacts still returned
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
    if os.path.isfile(MANIFEST_OUT):        # arrays the Visualizer can chart
        try:
            result["manifest_keys"] = sorted(np.load(MANIFEST_OUT).files)
        except Exception:  # noqa: BLE001
            pass
    print(json.dumps(result))


if __name__ == "__main__":
    main()
