"""The research team — role-routed models, no hardcoded data assumptions.

Pipeline (one literature stage, then experiment rounds):

    0. DataProfiler         -> explores raw file(s), produces DataReport
    1. Researcher.plan_queries -> scholarly queries
       (orchestrator: search + read papers -> PaperBriefs)
    2. Synthesist.synthesize -> plan grounded in the DataReport + briefs
    3. CodingAgent           -> implements the plan in a run/fix loop;
                                saves result manifest, prints conclusions
    4. Visualizer            -> renders interactive charts from manifest
    5. Reporter              -> answers the user's question

Every role can run on a different model (LLM_MODEL_<ROLE> in .env).
Nothing is canned; failures are surfaced.
"""

from __future__ import annotations

import json
import re

from app.agent.llm import LLMError, get_llm
from app.quant.dataset import dataset_meta
from app.quant.provision import STACK_STRING
from app.quant.schemas import DataReport, ExperimentPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unfence(text: str) -> str:
    """Strip markdown code fences so the result is clean Python (or JSON)."""
    text = re.sub(r"^```\w*\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def dataset_schema() -> dict:
    """Discover actual keys from the default dataset — no assumptions about structure."""
    from app.quant.dataset import NPZ_PATH

    meta = dataset_meta()
    return {
        "keys": meta.get("keys", []),
        "shapes": meta.get("shapes", {}),
        "dtypes": meta.get("dtypes", {}),
    }


def env_spec(replaced_default: bool, uploads: list[str]) -> str:
    """Describe the files available in the sandbox — no fixed column/key assumptions."""
    parts = [
        "SANDBOX ENVIRONMENT",
        "  af.DATA      -> /data/default (default dataset — load with numpy, pandas, etc.)",
        "  af.OUT       -> /out (save artifacts + manifest to this directory)",
        f"  PREINSTALLED -> {STACK_STRING}",
    ]
    if uploads:
        flist = ", ".join(uploads)
        parts.append(f"  af.uploads() -> {flist} (your uploaded files — load with numpy/pandas)")
    parts.append("  Inspect af.DATA keys at runtime — no fixed column names are assumed.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 0. DATA PROFILER
# ---------------------------------------------------------------------------


class DataProfiler:
    """Interpret raw file metadata into an enriched DataReport.

    The backend reads the raw file (any format — NPZ, CSV, etc.) and extracts
    basic metadata (keys, columns, shapes, dtypes). The LLM turns that into a
    meaningful report the Synthesist can use to design an experiment.
    """

    @staticmethod
    def interpret(metadata: dict, question: str) -> DataReport:
        import json

        system = (
            "You are a DATA ANALYST. Given raw file metadata and a research question, "
            "produce a structured report describing what the data contains, what each "
            "column likely represents, basic quality notes, and how it maps to the "
            "research question. The dataset could be any format (tabular, array, etc.). "
            "Be specific — reference actual column/key names from the metadata."
        )
        user = (
            f"RESEARCH QUESTION: {question}\n\n"
            f"RAW METADATA:\n{json.dumps(metadata, indent=2)}\n\n"
            "Produce a data report as JSON with at least these keys:\n"
            "- summary: str — what the dataset contains\n"
            "- columns: dict — each column/key name, its shape, dtype, and likely meaning\n"
            "- data_plan: str — how to use these columns for the research question, "
            "including any preprocessing needed\n"
            "Output ONLY valid JSON. No markdown, no extra text."
        )
        llm = get_llm()
        raw = _unfence(llm.chat(system, user, temperature=0.2, max_tokens=3000, role="synthesist"))
        keys = metadata.get("keys") or metadata.get("columns") or []
        try:
            obj = json.loads(raw)
            return DataReport(content=obj, raw_keys=list(keys), source="llm")
        except (json.JSONDecodeError, TypeError):
            return DataReport(content=metadata, raw_keys=list(keys), source="raw")


# ---------------------------------------------------------------------------
# 1. RESEARCHER / PAPER READER  (unchanged logic, minor prompt tweaks)
# ---------------------------------------------------------------------------


class Researcher:
    """Plan and execute literature searches via OpenAlex + arXiv."""

    _COUNTER: int = 0

    @staticmethod
    def plan_queries(goal: str) -> list[str]:
        Researcher._COUNTER += 1
        system = (
            "You are a FINANCE & QUANT RESEARCH LIBRARIAN. "
            "Given a research goal, propose 3 search queries for OpenAlex."
        )
        user = (
            f"Goal: {goal}\n\n"
            "Return a JSON list of 3 search query strings. "
            "Each query should target a different facet of the problem. "
            'Output ONLY valid JSON like ["query1", "query2", "query3"]. No markdown.'
        )
        llm = get_llm()
        raw = _unfence(llm.chat(system, user, temperature=0.0, max_tokens=500, role="researcher"))
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [goal]

    @staticmethod
    def elaborate(idea: str) -> str:
        """Expand a research idea with literature context (brief)."""
        system = (
            "You are a QUANT RESEARCH ANALYST. "
            "Given a research idea, expand it with context from the literature you have read."
        )
        user = (
            f"Research idea: {idea}\n\nProvide 1-2 paragraphs of relevant background and context."
        )
        llm = get_llm()
        return _unfence(llm.chat(system, user, temperature=0.3, max_tokens=1000, role="researcher"))


class PaperReaderAgent:
    """Wrap a paper brief for LLM ingestion."""

    def __init__(self, source: str, title: str, summary: str, context: str = ""):
        self.source = source
        self.title = title
        self.summary = summary
        self.context = context

    def to_context(self) -> str:
        return f"--- Paper: {self.title} ---\nSource: {self.source}\n{self.summary}\n"


# ---------------------------------------------------------------------------
# 2. GROUNDED SYNTHESIST
# ---------------------------------------------------------------------------


class Synthesist:
    """Design an experiment plan grounded in the data report + paper briefs.

    The output is an ExperimentPlan (Pydantic schema auto-injected into prompt).
    """

    @staticmethod
    def synthesize(goal: str, data_report: DataReport, briefs: str) -> ExperimentPlan:
        system = (
            "You are a QUANTITATIVE RESEARCH SYNTHESIST. "
            "You design experiments that are achievable with the given data and grounded "
            "in the literature."
        )
        schema_str = ExperimentPlan.json_schema()
        example_str = ExperimentPlan.json_example()
        constraints = f"""
CONSTRAINTS (non-negotiable):
- The experiment MUST be achievable with the columns DESCRIBED in the data report below.
- The plan MUST reference ACTUAL column names from the data report.
- NEVER propose using columns that don't appear in the data report.
- The plan MUST include: goal, data_columns (mapping roles to actual names),
  methodology (step-by-step), evaluation (how to judge success), and constraints.
- The evaluation metric MUST match the data type (Sharpe for price data,
  accuracy for classification, R^2 for regression, etc.).
- If the data report shows no price columns, do NOT propose factor research.

OUTPUT SCHEMA (your response must match this JSON schema exactly):
{schema_str}

EXAMPLE:
{example_str}
"""
        user = (
            f"RESEARCH QUESTION: {goal}\n\n"
            f"DATA REPORT:\n{json.dumps(data_report.content, indent=2)}\n\n"
            f"LITERATURE BRIEFS:\n{briefs}\n\n"
            "Design the experiment plan now. Return ONLY the JSON object. No markdown, no extra text."
        )
        llm = get_llm()
        raw = _unfence(
            llm.chat(
                system + "\n\n" + constraints,
                user,
                temperature=0.3,
                max_tokens=4000,
                role="synthesist",
            )
        )
        try:
            obj = json.loads(raw)
            return ExperimentPlan(**obj)
        except (json.JSONDecodeError, TypeError) as exc:
            return ExperimentPlan(
                goal=goal,
                methodology=[f"plan parsing failed: {exc}", raw[:500]],
            )


# ---------------------------------------------------------------------------
# 3. CODING AGENT
# ---------------------------------------------------------------------------


class CodingAgent:
    """Implement the experiment plan — write ONE script per sandbox run.

    The coder gets the plan (with actual column names), the data report,
    and file paths. It writes one Python script per iteration, runs it in
    the sandbox, and adapts based on stdout + errors.
    """

    def __init__(self):
        self.cache = {"command": "", "_sys": "", "_last": ""}

    def _system(self, uploads: list[str], replaced_default: bool) -> str:
        parts = [
            "You are a SENIOR QUANT RESEARCHER AND ENGINEER. "
            "You write Python code that implements quantitative experiments.",
            "",
            "CONSTRAINTS:",
            "- Write ONE script per response. The script must be complete and runnable.",
            "- The sandbox gives you ONLY file paths — no scoring, no hidden helpers.",
            "  af.DATA    -> path to the dataset (load + inspect it yourself)",
            "  af.OUT     -> directory to save artifacts / arrays",
            "  af.uploads() -> list of uploaded file paths",
            "- Load af.DATA with the right reader (np.load / pd.read_csv / etc.) and",
            "  DISCOVER the actual column/key names at runtime. Never assume names.",
            "- You MUST compute EVERYTHING yourself: load data, pick the price/feature",
            "  columns, compute forward returns if needed, evaluate the strategy, and",
            "  print the metrics the research question demands (Sharpe for price data,",
            "  accuracy for classification, R^2 for regression, etc.).",
            "- Print a single JSON result line to stdout (metrics + conclusions), e.g.:",
            '  print(json.dumps({"sharpe": 1.2, "max_drawdown": -0.1, "conclusion": "..."}))',
            '- Save arrays you want visualized: np.savez(f"{af.OUT}/anything.npz", key=arr).',
            "  Any .npz in af.OUT is picked up for charting — name keys whatever you like.",
            "- NEVER hardcode column or key names — read them from the file or the plan.",
            '- If the plan specifies column roles (e.g. "price": "close"), USE those names.',
            "- Verify columns exist at runtime: print(df.columns.tolist()).",
            "- Handle errors gracefully — print error and adapt, don't crash.",
            "- Output ONLY the Python script. No markdown fences, no explanations.",
        ]
        parts.append("")
        parts.append(env_spec(replaced_default, uploads))
        return "\n".join(parts)

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
    ) -> tuple[str, str]:
        """Generate one research script given the experiment plan and data.

        Returns (script_code, thinking_text).
        """
        _sys = self._system(uploads, replaced_default)
        self.cache["_sys"] = _sys

        # Build context from the experiment plan
        plan = idea.get("experiment_plan", {})
        if isinstance(plan, ExperimentPlan):
            plan_str = json.dumps(
                {
                    "goal": plan.goal,
                    "data_columns": plan.data_columns,
                    "methodology": plan.methodology,
                    "evaluation": plan.evaluation,
                    "constraints": plan.constraints,
                },
                indent=2,
            )
        else:
            plan_str = json.dumps(plan, indent=2)

        data_report = idea.get("data_report", {})
        if isinstance(data_report, DataReport):
            report_str = json.dumps(data_report.content, indent=2)
        else:
            report_str = json.dumps(data_report, indent=2)

        feedback = ""
        if tried_code:
            fb = idea.get("last_feedback", "")
            fb_stdout = idea.get("last_stdout", "")
            fb_err = idea.get("last_error", "")
            if fb:
                feedback = f"FEEDBACK FROM PREVIOUS RUN:\n{fb}\n"
            if fb_stdout:
                feedback += f"STDOUT:\n{fb_stdout[-1500:]}\n"
            if fb_err:
                feedback += f"ERROR:\n{fb_err}\n"

        goal = plan.goal if isinstance(plan, ExperimentPlan) else idea.get("goal", "")
        user = (
            f"RESEARCH QUESTION: {goal}\n"
            f"EXPERIMENT PLAN:\n{plan_str}\n\n"
            f"DATA REPORT:\n{report_str}\n\n"
            f"FILE(S): {', '.join(uploads) if uploads else 'af.DATA (default dataset)'}\n"
            f"ITERATION: {'initial' if not tried_code else f'repair #{len(tried_code)}'}\n\n"
            f"{feedback}\n"
            "Write the script now. Code only."
        )
        llm = get_llm()
        result = _unfence(llm.chat(_sys, user, temperature=0.3, max_tokens=8000, role="coder"))
        tried_code.append(result)
        return result, ""


# ---------------------------------------------------------------------------
# 4. VISUALIZER
# ---------------------------------------------------------------------------


class Visualizer:
    """Render interactive charts from a manifest NPZ.

    The manifest keys are whatever the coder chose — no expected structure.
    """

    @staticmethod
    def render(manifest: list[dict], goal: str = "", evaluation: str = "") -> str:
        if not manifest:
            return ""
        system = (
            "You are a DATA VISUALIZATION ENGINEER. "
            "Given arrays from a quant experiment, render interactive Plotly charts "
            "that DIRECTLY support the user's research question and the planned "
            "evaluation — NOT generic dataset plots."
        )
        keys_summary = {str(i): list(m.keys()) for i, m in enumerate(manifest[:3])}
        user = (
            f"USER'S RESEARCH QUESTION:\n{goal}\n\n"
            f"PLANNED EVALUATION:\n{evaluation}\n\n"
            f"AVAILABLE MANIFEST KEYS (shapes only):\n{json.dumps(keys_summary, indent=2)}\n\n"
            "Write ONE Python script that creates the charts that best answer the "
            "research question above (e.g. equity curve if evaluating returns, "
            "confusion matrix if classifying, scatter if regression). Save each chart "
            "as plotly_html_{i}.html in af.OUT. The arrays are available as "
            "af.OUT/*.npz (load with np.load)."
        )
        llm = get_llm()
        return _unfence(llm.chat(system, user, temperature=0.4, max_tokens=6000, role="visualizer"))


# ---------------------------------------------------------------------------
# 5. EXPORTER  (generates Pine Script / MQL5 when the user asks for it)
# ---------------------------------------------------------------------------


class Exporter:
    """Translate a backtested strategy into Pine Script / MQL5.

    Only invoked when the user's prompt asks for a platform export
    (e.g. "generate Pine Script", "export to MQL5"). The LLM does the
    translation — there is no hardcoded schema; it reads the result dict.
    """

    @staticmethod
    def export(result: dict, stdout: str) -> tuple[str, str]:
        system = (
            "You are a TRADING STRATEGY EXPORTER. "
            "Translate a validated quant strategy into executable Pine Script (TradingView) "
            "and MQL5 (MetaTrader 5) code. Use the provided metrics and run output to "
            "reconstruct the signal logic faithfully."
        )
        user = (
            f"RESULT KEYS: {list(result.keys())}\n"
            f"RUN OUTPUT (last 2500 chars):\n{stdout[-2500:]}\n\n"
            "Generate BOTH:\n"
            "1) Pine Script (TradingView) — an `indicator()` or `strategy()` with the signal.\n"
            "2) MQL5 (MetaTrader 5) — an Expert Advisor implementing the same logic.\n\n"
            'Output as JSON: {"pine": "...", "mql5": "..."}'
        )
        llm = get_llm()
        raw = _unfence(llm.chat(system, user, temperature=0.3, max_tokens=5000, role="exporter"))
        try:
            data = json.loads(raw)
            return data.get("pine", ""), data.get("mql5", "")
        except (json.JSONDecodeError, TypeError):
            return "", ""


# ---------------------------------------------------------------------------
# 6. REPORTER
# ---------------------------------------------------------------------------


class Reporter:
    """Synthesize all results into a final answer to the user."""

    @staticmethod
    def report(goal: str, plan: ExperimentPlan, run_stdout: str) -> dict:
        system = (
            "You are a QUANTITATIVE RESEARCH REPORTER. "
            "Summarize what was done, what was found, and what it means for the user."
        )
        user = (
            f"RESEARCH GOAL: {goal}\n"
            f"EXPERIMENT PLAN:\n{plan.model_dump_json(indent=2)}\n\n"
            f"RUN OUTPUT:\n{run_stdout[-3000:]}\n\n"
            "Write a concise report covering:\n"
            "- What was tested\n"
            "- What the data showed\n"
            "- The key results (with numbers)\n"
            "- Interpretation / conclusion\n"
            'Output as JSON: {"summary": "...", "results": {...}, "conclusion": "..."}'
        )
        llm = get_llm()
        raw = _unfence(llm.chat(system, user, temperature=0.3, max_tokens=2000, role="reporter"))
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"summary": raw[:500], "conclusion": "report generation failed"}
