"""Pydantic models for the quant data contract.

The NPZ format has no fixed key set — the agent discovers available panels
at runtime via ``d.files``.  ``NpzManifest`` only describes the naming
convention, not specific keys.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class NpzManifest(BaseModel):
    """Naming convention for NPZ datasets.

    Every ``px_*`` key is a panel; ``tickers`` and ``dates`` are metadata;
    ``fwd`` is hidden forward returns for grading; ``ret_key`` is the
    convention for computed returns.  Which panels exist is
    data-dependent and discovered at runtime.
    """

    prefix: str = "px_"
    tickers: str = "tickers"
    dates: str = "dates"
    fwd: str = "fwd"
    ret_key: str = "px_returns"


class DatasetPlan(BaseModel):
    """LLM-generated transformation plan for converting any uploaded
    dataset into the NPZ contract.  Every field is nullable — the LLM
    fills only what it can identify.  Code never assumes any field is set.
    """

    format: str = "wide"
    date_col: str | None = None
    ticker_col: str | None = None
    price_col: str | None = None
    returns_col: str | None = None
    value_cols: list[str] = []
    pivot_col: str | None = None

    @field_validator("format", mode="before")
    @classmethod
    def _norm_format(cls, v: str) -> str:
        v = str(v).strip().lower()
        return v if v in ("wide", "long") else "wide"


class VerdictOut(BaseModel):
    """What a statistical verdict contains — all fields optional so consumers
    check existence rather than testing sentinel values."""

    grade: str | None = None
    overall_score: float | None = None
    edge: float | None = None
    robustness: float | None = None
    risk: float | None = None
    overfit: bool | None = None
    notes: list[str] | None = None
    keep: bool | None = None
    psr: float | None = None
    dsr: float | None = None
    ic_z: float | None = None
