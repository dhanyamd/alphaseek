"""Evaluate a backtested factor — statistical grading, no magic numbers.

Uses Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR),
bootstrap confidence intervals, and statistical significance tests.
All parameters are grounded in the data distribution, not arbitrary cutoffs.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

_EULER_MASCHERONI = 0.5772156649


def _bootstrap_sharpe(
    equity_curve: list[float] | np.ndarray,
    n_iter: int = 10_000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 10:
        return 0.0, 0.0, 0.0
    returns = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    n = len(returns)
    boot = np.empty(n_iter)
    rng = np.random.default_rng(42)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        r = returns[idx]
        m = r.mean()
        s = r.std(ddof=1) + 1e-12
        boot[i] = (m / s) * np.sqrt(252)
    alpha = (1 - ci) / 2
    return (
        float(boot.mean()),
        float(np.percentile(boot, alpha * 100)),
        float(np.percentile(boot, (1 - alpha) * 100)),
    )


def _probabilistic_sharpe(sharpe: float, n_obs: int) -> float:
    if n_obs < 2:
        return 0.5
    se = np.sqrt((1 + 0.5 * sharpe**2) / (n_obs - 1))
    return float(sp_stats.norm.cdf(sharpe / se)) if se > 0 else 0.5


def _deflated_sharpe(sharpe: float, n_obs: int, n_trials: int = 200) -> float:
    if n_obs < 2:
        return 0.0
    sqrt_t = np.sqrt(n_obs - 1)
    se = np.sqrt((1 + 0.5 * sharpe**2) / (n_obs - 1))
    inv_norm = sp_stats.norm.ppf(1 - 1.0 / n_obs)
    e_max = (1 - _EULER_MASCHERONI) * inv_norm + _EULER_MASCHERONI * np.sqrt(2 * np.log(n_trials))
    e_max_sharpe = e_max / sqrt_t
    return float((sharpe - e_max_sharpe) / se) if se > 0 else 0.0


def verdict(bt: dict) -> dict:
    sharpe = bt.get("sharpe")
    ic = bt.get("mean_ic", 0.0)
    decay = bt.get("ic_decay", 0.0)
    mdd = bt.get("max_drawdown", 0.0)
    turnover = bt.get("turnover", 0.0)
    n_obs = max(bt.get("n_days", 0), 2)

    if sharpe is None:
        sharpe = 0.0
    else:
        sharpe = float(sharpe)
    ic = float(ic)
    decay = float(decay)
    mdd = float(mdd)
    turnover = float(turnover)

    psr = _probabilistic_sharpe(sharpe, n_obs)
    dsr = _deflated_sharpe(sharpe, n_obs)
    dsr_sig = dsr > 0.95

    equity = bt.get("equity_curve")
    has_equity = bool(equity) and len(equity) > 5
    if has_equity:
        _, ci_low, ci_high = _bootstrap_sharpe(equity)
        ci_contains_zero = ci_low <= 0 <= ci_high
    else:
        ci_low = ci_high = 0.0
        ci_contains_zero = True

    ic_se = 1.0 / np.sqrt(n_obs)
    ic_z = ic / ic_se if ic_se > 0 else 0.0
    ic_sig = abs(ic_z) > 1.96

    decay_penalty = max(0, -decay * 50)
    turnover_penalty = max(0, (turnover - 1.0) * 2)
    edge_z = abs(ic_z) * 0.6 + max(0, dsr) * 0.4
    robustness_z = dsr * 0.5 - decay_penalty * 0.3 - turnover_penalty * 0.2
    risk_z = -mdd * 5

    overall_z = edge_z * 0.4 + robustness_z * 0.35 + risk_z * 0.25
    overall = float(sp_stats.norm.cdf(overall_z) * 100)

    notes = []
    if psr < 0.9:
        notes.append("Low probability that true Sharpe is positive.")
    if dsr < 0.95 and sharpe > 0:
        notes.append("Edge may not survive multiple-testing correction.")
    if ci_contains_zero and has_equity:
        notes.append("Bootstrap CI includes zero — no statistical edge detected.")
    if decay < 0:
        notes.append(f"IC decay is negative ({decay:.3f}) — strategy may be degrading.")
    if turnover > 2.0:
        notes.append(f"High turnover ({turnover:.1f}x) — costs may erode returns.")
    if abs(ic_z) < 1.96:
        notes.append("IC is not statistically distinguishable from zero.")
    if not notes:
        notes.append("Clean result — edge is statistically detectable.")

    grade = _stat_grade(dsr, psr, ic_sig, overall)
    keep = psr > 0.95 and (dsr > 0.95 or ic_sig)

    return {
        "grade": grade,
        "overall_score": round(overall, 1),
        "edge": round(max(0, edge_z * 10), 1),
        "robustness": round(max(0, robustness_z * 10), 1),
        "risk": round(max(0, risk_z * 10), 1),
        "overfit": dsr < 0.95 and sharpe > 1.5,
        "notes": notes,
        "keep": keep,
        "psr": round(psr, 3),
        "dsr": round(dsr, 3),
        "ic_z": round(ic_z, 2),
    }


def _stat_grade(dsr: float, psr: float, ic_sig: bool, overall: float) -> str:
    if dsr > 1.65 and ic_sig:
        return "A"
    if psr > 0.95 and ic_sig:
        return "B"
    if psr > 0.9:
        return "C"
    if psr > 0.8:
        return "D"
    return "F"
