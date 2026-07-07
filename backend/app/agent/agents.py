"""The research team — role-routed models, no mocks, no silent fallbacks.

Pipeline (one literature stage per prompt, then experiment rounds):

    Researcher.plan_queries  -> scholarly queries for the goal
    (orchestrator: search + read papers -> PaperBriefs)
    Synthesist.synthesize    -> connects 2-3 briefs into ONE novel, testable
                                plan: hypothesis, novelty, methodology,
                                acceptance, validation targets, and the pip
                                REQUIREMENTS the code will need
    (orchestrator: provision the requirements into a sandbox image)
    CodingAgent              -> implements the methodology as MATH ONLY in an
                                agentic run/fix loop; records a result manifest;
                                writes no charts
    Visualizer               -> a separate stage that LOADS the manifest and
                                renders interactive charts (never recomputes)
    RiskCritic               -> hard-nosed review + next direction
    Reporter                 -> answers the user's question, grounded on what ran

Each role can run on a different model (LLM_MODEL_<ROLE> in .env). Every failure
is surfaced; nothing is canned.
"""

from __future__ import annotations

# import ast  # removed — no more babysitter code
import re

from app.agent.llm import LLMError, get_llm
from app.quant.dataset import dataset_meta
from app.quant.provision import STACK_STRING


def dataset_schema() -> dict:
    """Canonical schema for prompts and lint — introspected from the cache.

    Every field is discovered from the actual .npz file. Nothing is hardcoded.
    """
    from app.quant.dataset import NPZ_PATH

    meta = dataset_meta()
    panels = list(meta.get("inputs") or [])
    days = meta.get("days")
    stocks = meta.get("stocks")
    span = f"{meta.get('start', '?')}..{meta.get('end', '?')}" if days else "unknown"

    # Discover actual NPZ keys — never assume a prefix
    npz_keys: list[str] = []
    if NPZ_PATH.exists():
        import numpy as np

        with np.load(NPZ_PATH, allow_pickle=False) as z:
            npz_keys = list(z.files)
    if not npz_keys:
        npz_keys = [f"px_{p}" for p in panels] + ["tickers", "dates"]
    return {
        "panels": panels,
        "npz_keys": npz_keys,
        "days": days,
        "stocks": stocks,
        "span": span,
    }


def env_spec(uploads: list[str] | None = None, replaced_default: bool = False) -> str:
    """Sandbox contract — how the agent reads data and what it can do.

    When *replaced_default* is true, the uploaded file has been converted to NPZ
    and IS ``af.DATA`` — the agent loads it with ``np.load(af.DATA)`` as normal.
    When uploads exist but *replaced_default* is false, the upload is the primary
    source and the default market npz is a fallback.
    """
    if not replaced_default:
        schema = dataset_schema()
        keys_str = ", ".join(schema["npz_keys"]) or "(discover with d.files)"
        data_line = (
            f"{schema['stocks']} instruments x {schema['days']} periods ({schema['span']})"
            if schema["days"]
            else "backend-provided dataset"
        )
    else:
        keys_str = "px_<panel> for each uploaded panel — discover ALL keys at runtime"
        data_line = "uploaded dataset — discover dimensions at runtime with d['key'].shape"
    uploads_line = ""
    if uploads and replaced_default:
        fnames = ", ".join(uploads)
        uploads_line = f"""
  - af.DATA -> YOUR uploaded dataset (converted to NPZ).
    Load it with numpy:  d = np.load(af.DATA); print('keys:', list(d.files))
    Then discover dimensions with d['key'].shape.
    The original file ({fnames}) is also available via af.uploads() for pandas.
"""
        data_access = f"""```python
import numpy as np
d = np.load(af.DATA)
print("DATA keys:", list(d.files))
for k in d.files:
    print(f"  {{k}}: {{d[k].shape}}, dtype={{d[k].dtype}}")
```"""
    elif uploads:
        fnames = ", ".join(uploads)
        uploads_line = f"""
  - af.uploads() -> paths: [{fnames}].  THIS is your PRIMARY data for this session.
    Load it with pandas:  df = pd.read_csv(af.uploads()[0])  (or .read_parquet, .read_excel, etc.)
    Inspect columns, dtypes, shape at runtime — never assume column names.
    The default npz (af.DATA) is SECONDARY — only use it if the upload doesn't contain what you need.
"""
        data_access = f"""```python
import pandas as pd
path = af.uploads()[0]
df = pd.read_csv(path)  # or read_parquet, read_excel, read_json
print("Columns:", list(df.columns))
print("Shape:", df.shape)
print(df.dtypes)
print(df.head())
```"""
    else:
        uploads_line = "\n  - af.uploads() -> paths of any user-uploaded files (pandas-readable)."
        data_access = f"""```python
import numpy as np
d = np.load(af.DATA)
keys = list(d.files)
print(keys)                      # discover what exists — do NOT assume
arr = d[keys[0]]                 # access by the exact string from `keys`
print(arr.shape)                 # discover dimensions at runtime
```"""

    return f"""SANDBOX CONTRACT — you write ALL the code yourself. Installed and
importable: {STACK_STRING}, matplotlib, plotly. No network, no pip.

`import alphaseek as af` gives you:
  - af.DATA : path to a .npz of RAW market panels ({data_line}).
      EXACT keys: {keys_str}
      Determine each array's dimensions at runtime with d['key'].shape.
  - af.backtest(signal) -> metrics dict {{sharpe, sharpe_net, mean_ic, ic_decay,
      turnover, max_drawdown, equity_curve}}. Use it to evaluate signal-type research.
  - af.submit(signal) -> the OFFICIAL graded metrics; call ONCE when finished (for
      signal-based work only).  For non-signal research (regression, clustering,
      stats tests, etc.) just print results — do NOT call submit.
      `signal` is a FULL (T, N) array. Grading is BLIND (forward returns hidden) —
      build signals CAUSALLY (past rows only) or the look-ahead guard rejects them.
      Internally: z-score -> demean -> equal-weight long/short. Magnitude ignored.
  - af.submit_weights(weights) -> submit DIRECT portfolio weights (T, N), bypassing
      the signal-to-weight pipeline. Use this when you build your OWN portfolio
      construction. The runner preserves relative magnitudes you set (Kelly sizing,
      volatility targeting, risk parity, etc.) and only ensures dollar-neutrality.
  - af.OUT : a directory. Save arrays the chart stage needs with
      np.savez(f"{{af.OUT}}/manifest.npz", equity=..., ic=...). Do NOT draw charts.
{uploads_line}
DATA ACCESS PATTERN — discover the structure of your data at runtime:
{data_access}
INSTRUCTION: Every data key lives inside the npz archive. Access it ONLY via
dictionary subscript: array = d['key_name']. Replace 'key_name' with whatever
d.files returns. NEVER use a key name as a bare variable.
For uploaded files, use pandas and discover column names at runtime — never hardcode them.

LIBRARY USAGE — pick the RIGHT tool for each task:
  - numpy/pandas: data loading, alignment, rolling windows, vectorized ops
  - scipy: optimization (minimize, linprog), interpolation, signal processing
    (scipy.signal), spatial statistics (scipy.stats), linear algebra (scipy.linalg)
  - scikit-learn: regressions (LinearRegression, Ridge, Lasso, ElasticNet),
    classifiers, PCA/decomposition, preprocessing (StandardScaler), metrics
  - statsmodels: time-series models (ARIMA, VAR), OLS with diagnostics,
    panel regressions, hypothesis tests
  - arch: GARCH volatility models, mean-variance spanning tests
  - cvxpy: convex optimization (min-variance, risk parity, max Sharpe).
    cvxpy is your primary portfolio tool.
  - cvxportfolio: full portfolio optimization + backtesting with transaction
    cost models and market simulators (by Stanford CVXPY group)
  - skfolio: scikit-learn compatible portfolio optimization — 20+ methods,
    HRP, ensemble stacking, cross-validation for portfolios
  - riskfolio-lib: 22 risk measures, Nested Clustered Optimization,
    Black-Litterman, Entropy Pooling
  - pyportfolioopt (import as pypfopt): efficient frontier, Black-Litterman,
    HRP, covariance shrinkage
  - empyrical: standard performance metrics (Sharpe, Sortino, drawdown)
  - quantstats: advanced performance analytics and attribution
Do NOT limit yourself to numpy/pandas when a specialized library fits better."""


CHART_CRAFT = """CHART DESIGN SYSTEM (plotly, interactive, house style — follow exactly):
Always: fig.update_layout(template='plotly_dark', title=..., and axis titles). Name every
trace; add units; hovertemplates where useful. One figure per HTML file.

Pattern A — 3D SURFACE (parameter sweeps, term structures):
  go.Figure(go.Surface(z=Z, x=xs, y=ys, colorscale="Viridis",
      contours={"z": {"show": True, "usecolormap": True, "project_z": True}}))
  Overlay the optimum: go.Scatter3d(x=[x*], y=[y*], z=[z*], mode="markers+text").

Pattern B — MONTE CARLO FAN:
  faint sample paths (high transparency, low opacity) with showlegend=False;
  confidence band with fill="tonexty";
  bold median; annotate terminal stats in the title.

Pattern C — ANNOTATED HEATMAP (regimes, correlations):
  go.Heatmap(z=grid, x=cols, y=rows, colorscale="Viridis", colorbar={"title": "..."})
  + text annotations of each cell value; axis titles naming the regime dimensions."""


# --------------------------------------------------------------------------- Researcher
class Researcher:
    """Plans the literature search — the queries most likely to surface the
    canonical papers for the goal."""

    plan_system = (
        "You are the senior RESEARCHER on a quant team. Your job is to find the "
        "canonical papers that actually matter for the goal — not by keyword matching "
        "but by understanding the economic mechanism behind each paper.\n\n"
        "HOW QUANT RESEARCHERS SEARCH:\n"
        "- Start with the economic rationale: what market friction, behavioral bias, "
        "or structural constraint could create a tradable anomaly?\n"
        "- Search for the seminal papers first (the ones every follow-up cites), "
        "then forward-cite chase for modern improvements.\n"
        "- Look for construction details that matter: formation periods, holding "
        "windows, breakpoints, weighting schemes, universe filters, lag conventions. "
        "Academic abstracts hide these — the full paper or code appendix is what matters.\n"
        "- Prioritize papers that report out-of-sample tests and economic rationale "
        "over pure data-mining exercises.\n"
        "- Known pitfalls and failure modes are as valuable as positive results. "
        '"This factor decayed after publication" is a finding worth citing.\n\n'
        "YOUR SEARCH STRATEGY:\n"
        "- Mix 1 broad query (capture the territory) with 2-3 narrow queries "
        "(specific methods, authors, or known survey papers like Cochrane 2011).\n"
        "- Target each query at a different source type: academic (seminal), "
        "practitioner (implementation), recent (out-of-sample decay).\n"
        '- Respond with ONLY JSON: {"queries": ["...", "..."]}'
    )

    def plan_queries(self, seed: str, memory_summary: str) -> list[str]:
        out = get_llm().chat_json(
            self.plan_system,
            f"RESEARCH GOAL: {seed}\n"
            f"Findings so far:\n{memory_summary or '(first experiment)'}\n\nJSON only.",
            temperature=0.4,
            role="researcher",
        )
        queries = [str(q).strip() for q in out.get("queries", []) if str(q).strip()]
        return queries[:4]


# --------------------------------------------------------------------------- Synthesist
class Synthesist:
    """Connects the papers into ONE novel, testable research plan — the step
    where the team reasons across 2-3 sources and invents something to try."""

    system = (
        "You are the SYNTHESIST on a quant team — the person who turns raw literature "
        "into a testable research plan. Your job is to design an experiment that will "
        "tell you whether the hypothesis is true, regardless of whether it's a trading "
        "strategy, a statistical relationship, or a data description.\n\n"
        "HOW TO DESIGN A QUANT RESEARCH PLAN:\n"
        "1. Start with the economic mechanism or research question. What relationship "
        "are we testing? Why should it exist?\n"
        "2. State exactly what would FALSIFY the hypothesis before the code runs. "
        'e.g. "If the correlation is not significantly different from zero, the '
        'mechanism is wrong." A good experiment has falsification built in.\n'
        "3. Choose the right methodology for the question:\n"
        "   - Trading factor: signal → backtest → Sharpe/IC (use af.backtest, af.submit)\n"
        "   - Regression: OLS/logit/panel model → coefficients → t-stats/p-values\n"
        "   - Classification: sklearn model → accuracy/precision → confusion matrix\n"
        "   - Time series: ARIMA/GARCH → parameters → diagnostics\n"
        "   - Portfolio optimization: cvxpy/cvxportfolio → weights → efficient frontier\n"
        "   - Data analysis: correlations, distributions, summary statistics\n"
        "4. Specify failure modes upfront: what could go wrong and how you'll know.\n"
        "5. Choose parameters based on the economic mechanism, not what performed best.\n\n"
        "PRECISION RULES:\n"
        f"- Available libraries: {STACK_STRING}. The coder uses whichever fits.\n"
        "- If your methodology needs anything beyond this stack, list it in "
        "`requirements` (max 8 packages). Leave empty if not needed.\n"
        "- CRITICAL: Methodology steps must reference operations on available data. "
        "The coder discovers available keys at runtime. Do NOT assume specific column keys.\n"
        "- VALIDATION TARGETS must be concrete numbers: "
        '"reproduce the 0.08 IC from Jegadeesh & Titman 1993" or '
        '"p-value < 0.05" not "check if it works."\n\n'
        "THE OUTPUT CONTRACT — JSON only, no other text:\n"
        "{"
        '"name": "snake_case_id", '
        '"hypothesis": "one-sentence research question naming the mechanism/relationship", '
        '"novelty": "what is new here and which papers it connects", '
        '"methodology": ["3-6 concrete ordered steps. Each step names the exact '
        "operation, the window/parameter, the library, AND why this choice follows "
        'from the research question"], '
        '"validation_targets": ["specific numbers or criteria to validate the findings"], '
        '"acceptance": "the numeric evidence that would confirm or refute the hypothesis", '
        '"requirements": ["extra pip imports beyond the stack"], '
        '"references": ["short cites of the briefs that informed this"]}'
    )

    def synthesize(
        self,
        seed: str,
        briefs_context: str,
        memory_summary: str,
        feedback: str,
        tried: list[str],
        uploads: list[str],
        replaced_default: bool = False,
    ) -> dict:
        # Constraints stated in the goal ("uncorrelated with momentum", "only
        # low-vol") are the model's to read and honor — we do not pre-parse intent.
        if uploads and replaced_default:
            uploads_note = (
                f"\nUser uploaded dataset IS the primary data (converted to NPZ at af.DATA). "
                f"Your methodology must use the uploaded file. "
                f"Original file{'s' if len(uploads) > 1 else ''}: {uploads}"
            )
        elif uploads:
            uploads_note = (
                f"\nUser uploads available via af.uploads(): {uploads}" if uploads else ""
            )
        else:
            uploads_note = ""
        schema = dataset_schema()
        user = (
            f"RESEARCH GOAL: {seed}\n"
            f"AVAILABLE DATA — discover at runtime via np.load(af.DATA).files: "
            f"{schema['npz_keys']}\n"
            f"  Dimensions: {schema['stocks']} stocks x {schema['days']} days "
            f"({schema['span']})\n"
            "  These are the ONLY data keys that exist. Every feature must be derived\n"
            "  from these panels — there is nothing else. If the goal asks for data\n"
            "  not in this list, you MUST design a proxy from what IS available.\n"
            "  State your proxy explicitly in the methodology.\n"
            f"{uploads_note}\n"
            "Honor any feature or approach constraints stated in the goal.\n\n"
            f"PAPER BRIEFS (from the team's Reader):\n{briefs_context or '(no relevant literature retrieved — design from domain knowledge, do not degrade)'}\n\n"
            f"Findings so far:\n{memory_summary or '(first experiment)'}\n"
            f"Critic's direction: {feedback or '(none)'}\n"
            f"Already tried (must differ): {', '.join(tried) if tried else 'none'}\n\n"
            "Synthesize the single most informative NEXT experiment. JSON only."
        )
        return self._validate(
            get_llm().chat_json(
                self.system, user, temperature=0.5, role="researcher", max_tokens=4096
            )
        )

    @staticmethod
    def _validate(out: dict) -> dict:
        """Check the hand-off contract. Core fields (hypothesis, methodology) are
        the Synthesist's actual output — if they are missing the synthesis failed,
        and we surface that rather than fabricating a blank plan (no silent
        fallback). Genuinely optional fields are normalized, not invented."""
        if not str(out.get("hypothesis", "")).strip() or not out.get("methodology"):
            raise LLMError(
                "Synthesist returned no hypothesis/methodology — the "
                "synthesis failed to produce a usable plan."
            )
        out["name"] = (
            re.sub(r"[^a-zA-Z0-9_]+", "_", str(out.get("name", "experiment")))[:48] or "experiment"
        )
        out["requirements"] = [
            str(r).strip() for r in out.get("requirements", []) if str(r).strip()
        ][:8]
        for optional in ("novelty", "acceptance"):
            out.setdefault(optional, "")
        for optional in ("validation_targets", "references"):
            out.setdefault(optional, [])
        return out


RUN_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a complete Python research script in the sandbox. "
                "Call this when: you have a complete script implementing the methodology. "
                "The script MUST: load af.DATA, compute signals, call af.submit(signal), "
                "and save manifest arrays to af.OUT/manifest.npz. "
                "Do NOT call for: partial code, explanations, or questions. "
                "Returns: submitted flag, official metrics, manifest keys, stdout, or error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python source code. Must include: "
                            "import alphaseek as af, load af.DATA, compute signal as (T,N) "
                            "array, call af.submit(signal), and optionally save manifest. "
                            "No markdown fences — raw Python only."
                        ),
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }
]

RUN_TOOL_GENERAL = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a complete Python research script in the sandbox. "
                "Call this when: you have a complete script implementing the analysis. "
                "The script MUST: load data, perform the analysis, print results. "
                "Do NOT call af.submit() — this is not signal-based work. "
                "Returns: stdout, manifest keys, or error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python source code. Must include: "
                            "import alphaseek as af, load data, perform analysis, "
                            "print results. No markdown fences — raw Python only."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    }
]


# Preflight removed. The prompt teaches correct patterns and the error feedback
# loop handles mistakes naturally — same as Claude Code's approach.


def enrich_error_feedback(err: str) -> dict:
    """Turn a sandbox traceback into actionable retry hints for the coder."""
    schema = dataset_schema()
    keys = schema["npz_keys"]
    out: dict = {"available_data_keys": keys}

    prefix = (
        f"AVAILABLE DATA KEYS (af.DATA): {keys}\n"
        f"  Dimensions: {schema['stocks']} stocks x {schema['days']} days ({schema['span']})\n"
        "  These are the ONLY data keys. Derive ALL features from these panels.\n\n"
    )

    if "KeyError" in err or "not a file in the archive" in err:
        out["hint"] = prefix + (
            f"DATA KEY ERROR — the key you tried does not exist. "
            f"Use ONLY: {keys}. "
            "Derive any needed features from the available panels."
        )
    elif "Invalid format specifier" in err:
        out["hint"] = prefix + (
            "PRINT FORMAT ERROR — backtest logic is likely fine. "
            "The f-string format spec is invalid Python. Simplify your print "
            "statements. Wrap them in try/except so a print bug never wastes "
            "a good backtest."
        )
    elif "SyntaxError" in err:
        out["hint"] = (
            prefix + "Fix unmatched braces/quotes/parentheses only; resend the complete script."
        )
    elif "Look-ahead detected" in err:
        out["hint"] = prefix + (
            "Signal uses future data — the look-ahead guard rejected it. "
            "Build causally: use .shift(1) on returns, compute rolling windows on "
            "past rows only, and never index future dates."
        )
    elif "NameError" in err:
        out["hint"] = prefix + (
            "A variable or function is used before being defined. "
            "Check the traceback line and ensure the variable is computed before use."
        )
    elif "ValueError" in err and "shape" in err.lower():
        out["hint"] = prefix + (
            "Array shape mismatch. af.backtest() and af.submit() require a FULL "
            "(T, N) signal matching the shape of the data panels. "
            "Do not slice the time dimension."
        )
    elif "IndexError" in err:
        if "uploads" in err:
            out["hint"] = prefix + (
                "af.uploads() is EMPTY — there are NO user-uploaded files. "
                "Do NOT call af.uploads(). Only use data from af.DATA "
                "(np.load), discover its keys via d.files, and derive all "
                "features from the available panels."
            )
        else:
            out["hint"] = prefix + (
                "Index out of bounds — likely indexing past the end of an array. "
                "Use .iloc[-1] or check array shapes before indexing."
            )
    elif "KeyError" in err and ("archive" in err or "is not a file" in err):
        # NPZ key access failure — model guessed a key name instead of using
        # the exact strings from d.files
        out["hint"] = (
            prefix + "You accessed a NON-EXISTENT key in the NPZ. The error message "
            "lists the AVAILABLE keys — use one of THOSE exact strings, wrapped in "
            "d['...']. Do NOT guess names like 'close' or 'returns'; copy the exact "
            "key from your earlier print(list(d.files)) output (e.g. d['px_close'])."
        )
    else:
        out["hint"] = prefix + (
            "Read the traceback line number. Surgical fix only — change the failing "
            "lines, do not rewrite the whole script."
        )
    return out


def _extract_code_arg(arguments: str) -> str:
    import codecs
    import json as _json

    try:
        return _json.loads(arguments).get("code", "")
    except Exception:  # noqa: BLE001
        m = re.search(r'"code"\s*:\s*"(.*)"', arguments, re.S)
        if not m:
            return ""
        raw = m.group(1)
        try:
            return codecs.decode(raw, "unicode_escape")
        except Exception:  # noqa: BLE001
            return raw.replace("\\n", "\n").replace('\\"', '"')


_COMMON_RULES = """1. **Causal signals only**. Shift returns by 1 before any computation when working
   with price/return data. No look-ahead.

2. **Load data once, access by exact key string**. Load with np.load(af.DATA),
   print the key list, then access each array by its exact string name from that
   list — wrapped in d[...]. Never use a key name as a bare variable. For uploaded
   files, discover column names at runtime — never assume them.

3. **Vectorized only**. No Python loops over time or instruments. Use numpy/pandas.

4. **Know your shapes**. Print .shape or .columns for every data source you load.

5. **Use the professional quant stack — it is already installed**. Do NOT
   reinvent methods with raw numpy/pandas when a dedicated library exists.
   Import the right tool for the task:
   - Factor analysis & IC:        `import alphalens` (alphalens-reloaded)
   - Performance metrics:          `import empyrical` (empyrical-reloaded)
   - Regime switching / HMM:       `import hmmlearn` (GaussianHMM, GMMHMM)
   - GARCH & volatility models:    `import arch`
   - Cross-sectional / panel reg:  `import statsmodels.api as sm` (Fama-MacBeth,
                                    Newey-West, OLS)
   - Portfolio optimization:       `import cvxpy` (convex), `import riskfolio`
                                    (CVaR/mean-risk), `import pypfopt`
                                    (PyPortfolioOpt), `import cvxportfolio`
   - Modern portfolio analytics:   `import skfolio`
   - ML (clustering, dim-red, etc):`from sklearn import ...` (cluster,
                                    decomposition, ensemble)
   - Backtesting:                  `import vectorbt`, `import quantstats`
   - Visualization:                `import plotly.graph_objects as go`,
                                    `import matplotlib.pyplot as plt`,
                                    `import seaborn as sns`
   - Time-series stats:            `import scipy.stats as st`
   `import` the specific library your methodology requires on EVERY run. If an
   import fails, try an alternative from the list above — do not silently fall
   back to hand-rolled numpy loops for things these libraries do better."""

FACTOR_RULES = f"""{_COMMON_RULES}

6. **Always call af.submit(signal)** at the end with your (T,N) factor signal.
   This is what produces graded metrics. If you do not submit, the run is wasted.

# Quant Engineering Standards

**Before writing analysis logic, build the causal skeleton:**
- Load data from af.DATA (np.load), shift returns by 1.
- Hold out the last 20% of periods as OOS for predictive work.
- Verify shapes and check for NaNs.

**Then implement the factor methodology from the plan:**
- Derive every feature from raw panels. No pre-computed features exist.

**Quant rigor:**
- Report Sharpe, IC, turnover, drawdown.
- A Sharpe > 2 on the first try is usually a data leak — verify causality.

**Manifest contract:**
- np.savez(f"{{af.OUT}}/manifest.npz", equity=equity_curve, ic=ic_series, ...)
- Do NOT draw charts in this stage.

# Preferences
- Print a one-line summary: Sharpe, net Sharpe, IC, turnover, drawdown.
- Keep each script focused: one methodology per run.
- When a run fails: make a surgical fix, do not rewrite the whole script."""

GENERAL_RULES = f"""{_COMMON_RULES}

6. **Print your results** — never call af.submit().  This is a general quant
   analysis (regression, clustering, time-series, portfolio, stats, etc.).
   Save any arrays you want visualised via np.savez(f"{{af.OUT}}/manifest.npz", ...).

# Quant Analysis Standards

**Before writing analysis logic:**
- Load data (af.DATA or af.uploads()). Inspect columns, dtypes, shapes, missing values.
- Hold out the last 20% of periods as OOS when building predictive models.

**Then implement the methodology from the plan:**
- Use pandas for EDA, sklearn for ML, scipy/statsmodels for statistics,
  cvxpy/cvxportfolio for optimisation, arch for GARCH, etc.
- Every parameter choice needs a statistical or economic justification.

**Quant rigor:**
- Report uncertainty: p-values, confidence intervals, standard errors.
- Test robustness: split-sample, cross-validation, sensitivity analysis.

**Manifest contract:**
- np.savez(f"{{af.OUT}}/manifest.npz", your_key=your_array, ...)
- Saved arrays become charts in the visualizer stage.

# Preferences
- Print a clear summary of your findings.
- Keep each script focused: one methodology per run.
- When a run fails: make a surgical fix, do not rewrite the whole script."""


class CodingAgent:
    """An autonomous quant-coding agent — the MATH stage. Writes no charts."""

    MAX_STEPS = 5

    def __init__(self, mode: str = "factor"):
        self.mode = mode

    def _system(self, uploads: list[str] | None = None, replaced_default: bool = False) -> str:
        if self.mode == "factor":
            rules = FACTOR_RULES
            final_run = (
                "  Final run: call af.submit(signal) with your (T,N) factor signal, "
                "save manifest to af.OUT/manifest.npz, and print a results summary.\n"
            )
        else:
            rules = GENERAL_RULES
            final_run = (
                "  Final run: print all relevant results (tables, stats, regression "
                "output, etc). Save arrays for visualization via "
                'np.savez(f"{af.OUT}/manifest.npz", ...).\n'
            )
        return (
            "You are a senior quant researcher and engineer. Your job is to take a "
            "research plan grounded in real papers, implement its mathematics, "
            "iterate against data, and produce a rigorous quantitative answer. "
            "You write every line of code — there is no framework.\n\n" + rules + "\n\n"
            "## Sandbox Contract\n" + env_spec(uploads, replaced_default) + "\n\n"
            "## Workflow — How the Tool Works\n"
            "You have exactly one tool: run_python(code). Each call is a fresh "
            "sandbox — nothing persists between calls except what you output.\n\n"
            "Write the COMPLETE analysis in a SINGLE script. Do not split it across "
            "runs. Inside the script, load the data, check shapes with .shape (one "
            "print line), then implement the full methodology and save results. "
            "Only if the script raises an error should you use a second run to fix it.\n\n"
            + final_run
            + "  Do not waste runs on data exploration — discover shapes inline and "
            "implement everything in one script.\n\n"
            "At most "
            + str(self.MAX_STEPS)
            + " runs. One clean run is better than five messy ones.\n"
            "When done, reply with a 2-3 sentence summary of what you found "
            "and do NOT call the tool.\n\n"
            "IMPORTANT — Data keys are accessed ONLY via d['key'] after "
            "np.load(af.DATA). Never use them as bare variable names."
        )

    def run(
        self,
        step: int,
        idea: dict,
        uploads: list[str],
        uploads_dir,
        tried_code: list[str],
        briefs_context: str = "",
        lessons: list[str] | None = None,
        image: str | None = None,
        replaced_default: bool = False,
    ):
        """Generator: yields UI events; returns {'code','bt','summary'} or None."""
        import json as _json

        from app.quant.backtest import FactorError
        from app.quant.docker_sandbox import run_factor_code

        llm = get_llm()
        uploads_note = f"\nUser uploads via af.uploads(): {uploads}" if uploads else ""
        prior = ("\n".join(c[:300] for c in tried_code[-2:])) if tried_code else "(none)"
        method = "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(idea.get("methodology", [])))
        targets = "\n".join(f"  - {t}" for t in idea.get("validation_targets", []))
        lit = (
            f"\nLITERATURE (implement the math these briefs describe):\n{briefs_context}\n"
            if briefs_context
            else ""
        )
        lessons_note = ""
        if lessons:
            lessons_note = (
                "\nLESSONS FROM EARLIER FAILED RUNS (do not repeat these):\n"
                + "\n".join(f"  - {ls}" for ls in lessons[-3:])
                + "\n"
            )
        if self.mode == "factor":
            action = (
                "Implement the MATH now. Compute the signal, np.savez the chart arrays "
                "to af.OUT/manifest.npz, and END with af.submit(final_signal) — that "
                "last line is mandatory or the run is wasted. Write no charts."
            )
        else:
            action = (
                "IMPLEMENT the METHODOLOGY above in a single script. Do not split work "
                "across multiple runs — write one complete script that does everything.\n"
                "End your script with:\n"
                '  np.savez(f"{af.OUT}/manifest.npz", ic_series=ic, regimes=reg, cum_ret=cum, ...)\n'
                "Save EVERY array you want charted (IC series, regime states, cumulative "
                "returns, volatility estimates, etc). If you do not save a manifest, the "
                "visualizer has nothing to render and the run is wasted.\n"
                "Print p-values, confidence intervals, effect sizes. Do NOT call af.submit()."
            )
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"EXPERIMENT: {idea['name']}\nHYPOTHESIS: {idea['hypothesis']}\n"
                    f"NOVELTY: {idea.get('novelty', '')}\n"
                    f"METHODOLOGY:\n{method or '  (design it yourself, minimally)'}\n"
                    f"VALIDATION TARGETS (reproduce & compare):\n{targets or '  (none given)'}\n"
                    f"ACCEPTANCE: {idea.get('acceptance', '')}{uploads_note}{lit}{lessons_note}"
                    f"Code from earlier rounds (yours must differ):\n{prior}\n\n" + action
                ),
            }
        ]
        final = None
        summary = ""
        last_error = ""
        for turn in range(1, self.MAX_STEPS + 1):
            # Until we have a graded result, FORCE the tool call so the model
            # can't waste a turn "explaining" instead of running code; once it has
            # submitted, allow a free turn so it can end with a text summary.
            # Big max_tokens so a full script's tool-call JSON is never truncated.
            _run_tool = RUN_TOOL if self.mode == "factor" else RUN_TOOL_GENERAL
            resp = llm.chat_tools(
                self._system(uploads, replaced_default),
                messages,
                _run_tool,
                role="coder",
                max_tokens=16000,
                tool_choice="auto" if final is not None else "required",
            )
            if not resp["tool_calls"]:
                if final is not None and resp["content"].strip():
                    summary = resp["content"].strip()[:700]
                    yield {
                        "type": "agent_msg",
                        "step": step,
                        "agent": "Quant Coder",
                        "title": "summary",
                        "content": summary,
                    }
                    break
                yield {
                    "type": "run_error",
                    "step": step,
                    "attempt": turn,
                    "message": "coder did not call run_python — asked it to write and run code",
                }
                messages.append({"role": "assistant", "content": resp["content"] or "(no output)"})
                if self.mode == "factor":
                    _push = "CALL run_python now with the complete research script that ends in af.submit(final_signal)."
                else:
                    _push = (
                        "CALL run_python now with the complete analysis script that prints results."
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": "You did not call the run_python tool. Do not explain — "
                        + _push,
                    }
                )
                continue

            tc = resp["tool_calls"][0]
            code = _extract_code_arg(tc["arguments"])
            if not code.strip():
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp["content"] or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": "run_python", "arguments": tc["arguments"]},
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": _json.dumps(
                            {
                                "error": (
                                    "your tool arguments could not be parsed (empty code). "
                                    'Resend as valid JSON: {"code": "..."} with newlines '
                                    "as \\n and double quotes escaped."
                                )
                            }
                        ),
                    }
                )
                yield {
                    "type": "run_error",
                    "step": step,
                    "message": "unparseable tool call — asked the model to resend",
                    "attempt": turn,
                }
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": resp["content"] or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": "run_python", "arguments": tc["arguments"]},
                        }
                    ],
                }
            )
            yield {
                "type": "code",
                "step": step,
                "agent": "Quant Coder",
                "filename": f"run{turn}.py",
                "code": code,
                "model": resp.get("model", ""),
            }

            yield {
                "type": "handoff",
                "step": step,
                "agent": "Backtester",
                "action": f"executing run {turn}",
            }
            try:
                bt = run_factor_code(
                    code, uploads_dir=uploads_dir, image=image or "alphaseek-sandbox:latest"
                )
                brief = {
                    k: round(bt[k], 4)
                    for k in (
                        "sharpe",
                        "sharpe_net",
                        "mean_ic",
                        "ic_decay",
                        "turnover",
                        "max_drawdown",
                    )
                    if k in bt
                }
                tool_result = {
                    "submitted": bool(bt.get("submitted")),
                    **brief,
                    "manifest_keys": bt.get("manifest_keys", []),
                    "stdout": (bt.get("stdout") or "")[-2000:],
                    "available_data_keys": dataset_schema()["npz_keys"],
                }
                got = bt.get("manifest_keys") or []
                if self.mode == "factor" and not bt.get("submitted"):
                    if got:
                        tool_result["hint"] = (
                            f"You already recorded {got} and the math ran cleanly; send the "
                            "SAME script with the single line af.submit(<your final (T,N) signal>) "
                            "added at the end."
                        )
                    else:
                        schema = dataset_schema()
                        tool_result["hint"] = (
                            f"AVAILABLE DATA: {schema['npz_keys']} "
                            f"({schema['stocks']} stocks x {schema['days']} days)\n"
                            "Code ran clean but produced no signal — read stdout and fix."
                        )
                elif self.mode != "factor" and not got:
                    tool_result["hint"] = (
                        "WARNING: You printed results but did NOT save any arrays via "
                        'np.savez(f"{af.OUT}/manifest.npz", ...). The visualizer NEEDS '
                        "saved arrays to produce charts. On your next run, include ALL "
                        "analysis in ONE script and end with np.savez to save key results. "
                        "You have limited runs — do not waste them on exploration."
                    )
                yield {
                    "type": "backtest",
                    "step": step,
                    "agent": "Backtester",
                    "name": idea["name"],
                    "result": bt,
                    "engine": bt.get("engine", ""),
                    "exploration": not bt.get("submitted"),
                }
                if self.mode == "factor":
                    if bt.get("submitted"):
                        final = {"code": code, "bt": bt}
                elif bt.get("manifest_keys"):
                    final = {"code": code, "bt": bt}
            except FactorError as e:
                err = str(e)[:2000]
                tool_result = enrich_error_feedback(err)
                tool_result["error"] = err
                if err[:120] == last_error[:120]:
                    tool_result["hint"] = (
                        "SAME error as your previous run — re-read the "
                        "traceback line and take a different approach."
                    )
                last_error = err
                yield {"type": "run_error", "step": step, "message": str(e)[:400], "attempt": turn}
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": _json.dumps(tool_result)}
            )

        if final is not None:
            final["summary"] = summary
            final["last_error"] = last_error
        return final


# --------------------------------------------------------------------------- Visualizer
class Visualizer:
    """The chart stage: writes a SMALL script that loads the result manifest and
    renders interactive charts. It never recomputes the math, so a chart bug can
    never lose a validated result."""

    def _system(self) -> str:
        return (
            "You are the VISUALIZATION ENGINEER on a quant team. The math is DONE and "
            "its results are saved in a manifest. Write a short, self-contained script "
            "that loads the manifest and renders outstanding INTERACTIVE plotly charts "
            "that tell the story of the result.\n\n" + CHART_CRAFT + "\n\n"
            "## CRITICAL RULES — NEVER VIOLATE\n"
            "IMPORTANT: The math is done. Your script ONLY renders charts. "
            "Any recomputation risks corrupting the result.\n"
            "1. DO NOT call af.backtest(), af.submit(), or af.manifest(). "
            "These would re-run or re-submit research code.\n"
            "2. DO NOT recompute signals or factors. Load only what the coder saved.\n"
            "3. Pre-loaded variables are the ONLY data available. Each manifest key "
            "is a top-level numpy array. "
            "Do not invent or derive new arrays.\n"
            "4. numpy is available as np.\n\n"
            "## RULES — always follow\n"
            "- ALWAYS use .shape to discover array dimensions at runtime. "
            "Never assume a specific shape or dimension count.\n"
            "- Compute T from a time-series array: T = array.shape[0]. "
            "Do NOT use an undefined 'T'.\n"
            "- Choose the chart type the RESULTS justify: lines for equity/IC, "
            "heatmaps for signal panels, surfaces for parameter sweeps.\n"
            '- Save each figure: fig.write_html(f"{af.OUT}/<name>.html", '
            'include_plotlyjs="cdn").\n'
            "- Output ONLY the complete Python script, starting with import numpy and import plotly. No explanations."
        )

    def render(self, goal: str, idea: dict, manifest_keys: list[str], stdout: str) -> str:
        llm = get_llm()
        var_list = ", ".join(manifest_keys) or "(none)"
        user = (
            f"USER'S GOAL: {goal}\nEXPERIMENT: {idea['name']} — {idea['hypothesis']}\n"
            f"PRE-LOADED VARIABLES (numpy arrays, already in scope; use .shape to discover dimensions at runtime): {var_list}\n"
            f"What the math printed:\n{(stdout or '')[:600]}\n\n"
            "Write the visualization script using the pre-loaded variables. Code only."
        )
        out = llm.chat(self._system(), user, temperature=0.3, max_tokens=12000, role="viz")
        return _unfence(out)


def _unfence(text: str) -> str:
    """Unwrap a ``` ... ``` markdown code fence (any language) if the model wrapped
    its code. Removes only the delivery wrapper; the code inside is untouched."""
    t = text.strip()
    m = re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", t, re.S)
    return (m.group(1) if m else t).strip()


# --------------------------------------------------------------------------- Exporter
class Exporter:
    """Translates a VALIDATED Python strategy into deployable platform code
    (TradingView Pine Script, MetaTrader MQL5). The model does the entire
    translation — we template nothing. These run on their own platforms, not our
    sandbox, so we generate and hand them over; we do not backtest them here."""

    # target -> (platform description for the model, file extension)
    TARGETS = {
        "pine": (
            "Pine Script v5 for TradingView (a `strategy(...)` script with entries/exits)",
            "pine",
        ),
        "mql5": (
            "MQL5 for MetaTrader 5 (a self-contained Expert Advisor: "
            "OnInit/OnTick, order handling)",
            "mq5",
        ),
    }

    def _system(self, platform_desc: str) -> str:
        return (
            "You are a trading-systems engineer. You are given a research strategy "
            "that was already validated in Python on historical data. Re-express its "
            f"EXACT logic — the same features, windows, thresholds, and entry/exit "
            f"rules — as {platform_desc}. Use that platform's NATIVE data model and "
            "built-in indicators (the Python data API does not exist there). Keep it "
            "faithful, self-contained, and runnable; open with a short header comment "
            "naming the strategy and its source. Output ONLY the code."
        )

    def export(self, goal: str, name: str, hypothesis: str, code: str, lang: str) -> str:
        platform_desc, _ext = self.TARGETS[lang]
        user = (
            f"STRATEGY: {name}\nHYPOTHESIS: {hypothesis}\nRESEARCH GOAL: {goal}\n\n"
            f"VALIDATED PYTHON (translate its signal logic, not its data plumbing):\n"
            f"{code[:6000]}\n\nWrite the {lang} version. Code only."
        )
        out = get_llm().chat(
            self._system(platform_desc), user, temperature=0.2, max_tokens=3000, role="exporter"
        )
        return _unfence(out)


# --------------------------------------------------------------------------- RiskCritic
class RiskCritic:
    system = (
        "You are the skeptical RISK CRITIC on a quant team working on REAL equity data. "
        "net-of-cost Sharpe is what matters for tradeability; heavy turnover erodes "
        "edges. Judge BOTH edge quality AND whether the experiment served the user's "
        "goal and its own acceptance criteria. Respond with ONLY JSON: "
        '{"assessment": "one sharp sentence", "suggestion": "one concrete direction for '
        'the next experiment"}.'
    )

    def review(self, idea: dict, bt: dict, v: dict, goal: str = "") -> dict:
        llm = get_llm()
        sharpe = bt.get("sharpe")
        mean_ic = bt.get("mean_ic")
        ic_decay = bt.get("ic_decay")
        turnover = bt.get("turnover")
        mdd = bt.get("max_drawdown")
        sharpe_s = f"{sharpe:.2f}" if sharpe is not None else "N/A"
        ic_s = f"{mean_ic:.4f}" if mean_ic is not None else "N/A"
        decay_s = f"{ic_decay:.4f}" if ic_decay is not None else "N/A"
        turn_s = f"{turnover:.2f}" if turnover is not None else "N/A"
        mdd_s = f"{mdd:.1%}" if mdd is not None else "N/A"
        user = (
            f"USER'S GOAL: {goal}\n"
            f"Experiment `{idea.get('name', '?')}` — {idea.get('hypothesis', '')}\n"
            f"Acceptance criteria: {idea.get('acceptance', '(none)')}\n"
            f"Result: Sharpe {sharpe_s} (net {bt.get('sharpe_net', 0):.2f}), "
            f"mean IC {ic_s}, IC decay {decay_s}, "
            f"turnover {turn_s}, max drawdown {mdd_s}.\n"
            f"Verdict: grade {v.get('grade', '?')}, "
            f"overfit={v.get('overfit', False)}, keep={v.get('keep', False)}.\n"
            "JSON only."
        )
        out = llm.chat_json(self.system, user, temperature=0.4, role="critic")
        out.setdefault("assessment", (v.get("notes") or ["Reviewed."])[0])
        out.setdefault("suggestion", "Try a different construction.")
        return out


# --------------------------------------------------------------------------- Reporter
class Reporter:
    system = (
        "You are the REPORTER on a quant research team. Write a direct, specific answer "
        "to the user's question in 2-4 sentences. Describe ONLY what the experiments "
        "actually did and measured — never invent methodology. Include the winning "
        "factor's numbers and the honest caveat; cite the papers that grounded the "
        "work when briefs are listed. Also propose next steps. Respond "
        'with ONLY JSON: {"answer": "...", "next_steps": ["2-3 concrete follow-up '
        'research prompts the user could run next, each a single actionable sentence"]}.'
    )

    def report(
        self,
        goal: str,
        memory_summary: str,
        best: dict | None,
        tested: int,
        coder_summary: str = "",
        final_stdout: str = "",
        briefs_context: str = "",
        mode: str = "factor",
    ) -> dict:
        llm = get_llm()
        if best:
            sharpe_s = (
                f"Sharpe {best['sharpe']:.2f}, net {best.get('sharpe_net', 0):.2f}"
                if "sharpe" in best
                else ""
            )
            ic_s = f", IC {best['mean_ic']:.4f}" if "mean_ic" in best else ""
            grade_s = f", grade {best['verdict']['grade']}" if best.get("verdict") else ""
            best_line = f"Best: {best['name']} ({sharpe_s}{ic_s}{grade_s})"
        elif mode == "general":
            best_line = "General analysis completed. See experiment summaries below."
        else:
            best_line = "No factor produced a usable edge."
        user = (
            f"USER'S QUESTION: {goal}\n\nExperiments run: {tested}\n{best_line}\n"
            f"What the final script actually did (coder's summary): {coder_summary or '(none)'}\n"
            f"Final run stdout:\n{final_stdout[:500] or '(none)'}\n"
            f"Literature the team read:\n{briefs_context[:900] or '(none)'}\n"
            f"All findings:\n{memory_summary or '(none)'}\n\n"
            "Answer the question, grounded strictly in the above. JSON only."
        )
        out = llm.chat_json(self.system, user, temperature=0.3, role="reporter")
        out.setdefault("answer", best_line)
        out.setdefault("next_steps", [])
        return out
