"""Convert uploaded datasets to NPZ for the sandbox runner.

The LLM analyses the raw file — column names, dtypes, sample values, shape —
and produces a Pydantic-validated ``DatasetPlan``.  The plan drives every
transformation decision: what is the date column, what is the ticker column,
is it wide or long format, which column holds prices vs returns, etc.

Zero column names, formats, or shapes are hardcoded anywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.llm import get_llm
from app.quant.schemas import DatasetPlan, NpzManifest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM dataset analyser
# ---------------------------------------------------------------------------


def _analyze_dataset(df: pd.DataFrame) -> DatasetPlan:
    """Ask the LLM to produce a transformation plan, validated by Pydantic."""
    col_lines = []
    for c in df.columns:
        col = df[c]
        samples = col.dropna().head(5).to_list()
        col_lines.append(
            f"  {c!r}: dtype={col.dtype}, nunique={col.nunique()}/{len(col)}, sample={samples}"
        )
    col_info = "\n".join(col_lines)
    idx = df.index
    idx_info = f"Index: name={idx.name!r}, dtype={idx.dtype}, sample={list(idx[:3])}"

    prompt = (
        f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns.\n"
        f"{idx_info}\n\nColumns:\n{col_info}\n\n"
        "Produce a transformation plan as JSON:\n"
        "{\n"
        '  "format": "wide" or "long",\n'
        '  "date_col": "<col name for dates>" or null (use index),\n'
        '  "ticker_col": "<col identifying entities/tickers>" or null,\n'
        '  "price_col": "<col with raw prices>" or null,\n'
        '  "returns_col": "<col with pct returns>" or null,\n'
        '  "value_cols": ["<numeric columns to keep as data panels>"],\n'
        '  "pivot_col": "<value col to pivot on>" or null\n'
        "}\n\n"
        "- wide: each column is already a separate entity.\n"
        "- long: rows from multiple entities are stacked; ticker_col identifies them.\n"
        "- date_col: time axis column. null means the index IS the time axis.\n"
        "- price_col: monetary prices suitable for computing returns.\n"
        "- returns_col: pre-computed percentage returns.\n"
        "- value_cols: every numeric column worth keeping.\n"
        "- pivot_col: when long, which value column to pivot into wide.\n"
        "Return ONLY valid JSON."
    )
    raw = get_llm().chat_json(
        "You are a quant data engineer. Analyze dataset structure precisely.",
        prompt,
        temperature=0.1,
        role="dataset-analyzer",
        max_tokens=500,
    )

    # Strip any column names the LLM hallucinated
    all_cols = set(df.columns)
    for key in ("date_col", "ticker_col", "price_col", "returns_col", "pivot_col"):
        if isinstance(raw.get(key), str) and raw[key] not in all_cols:
            raw[key] = None
    if isinstance(raw.get("value_cols"), list):
        raw["value_cols"] = [c for c in raw["value_cols"] if c in all_cols]

    # Infer format if missing
    if raw.get("format") not in ("wide", "long"):
        raw["format"] = "long" if raw.get("ticker_col") else "wide"

    plan = DatasetPlan.model_validate(raw)
    log.info("Dataset plan: %s", plan.model_dump_json())
    return plan


# ---------------------------------------------------------------------------
# Plan execution — driven entirely by the Pydantic-validated plan
# ---------------------------------------------------------------------------


def _execute_plan(
    df: pd.DataFrame, plan: DatasetPlan
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Build NPZ arrays from a DataFrame + validated plan. No column name
    is ever assumed — every decision comes from the plan."""
    m = NpzManifest()
    arrays: dict[str, np.ndarray] = {}

    # 1) Set date index if plan says so
    if plan.date_col and plan.date_col in df.columns:
        df = df.copy()
        df[plan.date_col] = pd.to_datetime(df[plan.date_col], errors="coerce")
        df = df.set_index(plan.date_col).sort_index()

    # 2) Long → wide pivot
    if plan.format == "long" and plan.ticker_col and plan.ticker_col in df.columns:
        tcol = plan.ticker_col
        tickers = sorted(df[tcol].dropna().unique().astype(str))
        cols_to_pivot = [c for c in (plan.value_cols or []) if c in df.columns and c != tcol]
        if not cols_to_pivot and plan.pivot_col and plan.pivot_col in df.columns:
            cols_to_pivot = [plan.pivot_col]
        if not cols_to_pivot:
            cols_to_pivot = [c for c in df.select_dtypes(include="number").columns if c != tcol]
        for vc in cols_to_pivot:
            try:
                wide = (
                    df.pivot_table(index=df.index, columns=tcol, values=vc, aggfunc="first")
                    .sort_index()
                    .fillna(0.0)
                )
                arrays[f"{m.prefix}{vc}"] = wide.values.astype(float)
            except Exception:
                log.warning("pivot failed for column %r", vc, exc_info=True)
        dates = [str(d) for d in df.index.unique().sort_values()]
    else:
        # 3) Wide format — each value column is a panel
        tickers = []
        cols_to_save = [
            c
            for c in (plan.value_cols or [])
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cols_to_save:
            cols_to_save = list(df.select_dtypes(include="number").columns)
        for c in cols_to_save:
            arrays[f"{m.prefix}{c}"] = np.nan_to_num(df[c].values.astype(float), nan=0.0)
            tickers.append(str(c))
        dates = [str(d) for d in df.index]

    # 4) Forward returns — only if plan identified a price or returns column
    price_key = f"{m.prefix}{plan.price_col}" if plan.price_col else None
    ret_key = f"{m.prefix}{plan.returns_col}" if plan.returns_col else None

    if price_key and price_key in arrays:
        px = arrays[price_key]
        returns = np.zeros_like(px)
        returns[1:] = px[1:] / np.where(px[:-1] != 0, px[:-1], 1e-12) - 1.0
        returns = np.nan_to_num(returns, nan=0.0)
        fwd = np.zeros_like(returns)
        fwd[:-1] = returns[1:]
        arrays[m.ret_key] = returns
        arrays[m.fwd] = np.nan_to_num(fwd, nan=0.0)
    elif ret_key and ret_key in arrays:
        existing = arrays[ret_key]
        fwd = np.zeros_like(existing)
        fwd[:-1] = existing[1:]
        arrays[m.fwd] = np.nan_to_num(fwd, nan=0.0)

    # 5) Metadata
    arrays[m.tickers] = np.array(tickers if tickers else ["col0"])
    arrays[m.dates] = np.array(dates if dates else [str(i) for i in range(len(df))])

    return arrays, tickers, dates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upload_to_npz(src: Path, dst: Path) -> dict:
    """Convert any file to NPZ. The LLM decides the transformation."""
    m = NpzManifest()

    if src.suffix == ".npz":
        import shutil

        shutil.copy(src, dst)
        with np.load(dst, allow_pickle=False) as z:
            keys = list(z.files)
            ref_keys = [k for k in keys if k.startswith(m.prefix)] or keys
            if not ref_keys:
                raise ValueError("NPZ file is empty")
            ref = z[ref_keys[0]]
            T = ref.shape[0]
            N = ref.shape[1] if ref.ndim >= 2 else 1
        return {"days": int(T), "stocks": int(N), "format": "npz"}

    df = _read(src)
    plan = _analyze_dataset(df)
    arrays, tickers, dates = _execute_plan(df, plan)

    np.savez_compressed(dst, **arrays)
    return {"days": len(dates), "stocks": len(tickers), "format": src.suffix[1:]}


def _read(src: Path) -> pd.DataFrame:
    """Load a file into a DataFrame. No index assumptions — the LLM decides."""
    readers: dict = {
        ".csv": lambda: pd.read_csv(src),
        ".parquet": lambda: pd.read_parquet(src),
        ".xlsx": lambda: pd.read_excel(src),
        ".xls": lambda: pd.read_excel(src),
        ".json": lambda: pd.read_json(src),
    }
    reader = readers.get(src.suffix)
    if reader is None:
        raise ValueError(f"Unsupported format: {src.suffix}")

    result = reader()
    if isinstance(result, pd.Series):
        result = result.to_frame()
    elif not isinstance(result, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(result)}")

    if result.empty or result.shape[1] == 0:
        raise ValueError("Uploaded file contains no columns")
    return result
