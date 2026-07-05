"""Evaluate a backtested factor — the "Verdict" (A–F) + overfit detection.

Mirrors QuantPad's verdict: score a factor across EDGE, ROBUSTNESS, and RISK,
combine into a letter grade, and flag suspicious (overfit) results. This is how
the agent decides whether a factor is worth keeping in memory.
"""
from __future__ import annotations


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def verdict(bt: dict) -> dict:
    """Score a backtest result. Returns grades, sub-scores, notes, overfit flag."""
    sharpe = bt["sharpe"]
    ic = bt["mean_ic"]
    decay = bt["ic_decay"]
    mdd = bt["max_drawdown"]
    turnover = bt["turnover"]

    # EDGE — is there real predictive power? Calibrated for REAL daily equity
    # data, where mean IC 0.01-0.03 and Sharpe 0.5-1.5 are genuinely good.
    edge = _clamp(55 * max(sharpe, 0) + 2200 * max(ic, 0))

    # ROBUSTNESS — does the edge hold up (low decay, not insane turnover)?
    robustness = _clamp(70 + 1500 * decay - 20 * max(turnover - 1.0, 0))

    # RISK — shallow drawdowns score higher (mdd is negative)
    risk = _clamp(100 + 220 * mdd)   # -0.20 mdd -> 56 ; -0.40 -> 12

    overall = 0.5 * edge + 0.3 * robustness + 0.2 * risk

    # OVERFIT heuristics — great return but no IC, or absurd Sharpe, or huge turnover
    overfit = False
    notes = []
    if sharpe > 2.2 and ic < 0.015:
        overfit = True
        notes.append("Suspiciously high Sharpe with near-zero IC — likely overfit.")
    if turnover > 3.0:
        notes.append("Very high turnover — costs would erode the edge.")
    if decay < -0.02:
        notes.append("IC is decaying over the sample — fragile.")
    if ic <= 0.005 and sharpe <= 0.2:
        notes.append("No detectable edge — this factor is essentially noise.")
    if not notes:
        notes.append("Clean result — edge is real and reasonably stable.")

    return {
        "grade": _grade(overall),
        "overall_score": round(overall, 1),
        "edge": round(edge, 1),
        "robustness": round(robustness, 1),
        "risk": round(risk, 1),
        "overfit": overfit,
        "notes": notes,
        "keep": overall >= 55 and not overfit and ic > 0.008,  # worth remembering?
    }


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 68:
        return "B"
    if score >= 55:
        return "C"
    if score >= 42:
        return "D"
    return "F"
