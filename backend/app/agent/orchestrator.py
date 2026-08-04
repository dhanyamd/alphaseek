"""The lead orchestrator — runs the multi-agent research team, streaming handoffs.

    0. DATA PROFILER : explores raw file(s), produces DataReport
    1. LITERATURE    : Researcher plans queries -> search -> read top papers
    2. per round     : Synthesist -> Coder -> Visualizer -> Memory
    3. finally       : Reporter answers the user's question

No hardcoded grading, no RiskCritic, no fixed schema expectations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.agent.agents import (
    CodingAgent,
    DataProfiler,
    Exporter,
    Reporter,
    Researcher,
    Synthesist,
    Visualizer,
)
from app.agent.llm import LLMError, get_llm
from app.agent.memory import Memory
from app.agent.reader import rank_papers, read_paper
from app.agent.research_tools import fetch_full_text, search
from app.quant.backtest import FactorError
from app.quant.dataset import dataset_meta
from app.quant.docker_sandbox import ARTIFACT_STORE, docker_available, run_factor_code
from app.quant.schemas import DataReport, ExperimentPlan


def _sanitize(v: dict) -> dict:
    import numpy as np

    out: dict = {}
    for k, val in v.items():
        out[k] = _sanitize_dict_or_scalar(val)
    return out


def _sanitize_dict_or_scalar(v):
    import numpy as np

    if isinstance(v, dict):
        return _sanitize(v)
    if isinstance(v, (list, tuple)):
        return [_sanitize_dict_or_scalar(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.datetime64, np.timedelta64)):
        return str(v)
    if isinstance(v, np.complexfloating):
        return str(v)
    if isinstance(v, np.void):
        return str(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


TEAM = [
    {"name": "Data Profiler", "role": "explores the raw dataset"},
    {"name": "Researcher", "role": "reads literature"},
    {"name": "Synthesist", "role": "designs experiment grounded in data report"},
    {"name": "Quant Coder", "role": "implements the plan"},
    {"name": "Visualizer", "role": "renders interactive charts"},
    {"name": "Exporter", "role": "translates strategy to Pine Script / MQL5 (on request)"},
]

MAX_PAPERS_READ = 3
READ_CONCURRENCY = 3
LESSONS_WINDOW = 3


def research(
    seed: str,
    iterations: int = 8,
    mem: Memory | None = None,
    uploads_dir=None,
    uploads: list[str] | None = None,
) -> Iterator[dict]:
    llm = get_llm()
    uploads = uploads or []
    profiler, synthesist = DataProfiler(), Synthesist()
    coder, reporter, visualizer = CodingAgent(), Reporter(), Visualizer()
    mem = mem or Memory()
    state = {
        "feedback": "",
        "best": None,
        "tried_code": [],
        "briefs": [],
        "lessons": [],
        "last_plan": None,
        "data_report": DataReport(content={}, raw_keys=[], source="default"),
    }
    engine = "docker" if docker_available() else "inprocess"
    yield {
        "type": "start",
        "seed": seed,
        "model": llm.model,
        "iterations": iterations,
        "team": TEAM,
        "engine": engine,
        "dataset": dataset_meta(),
        "resumed": len(mem.entries) > 0,
        "uploads": uploads,
    }

    if not llm.configured:
        yield {
            "type": "error",
            "fatal": True,
            "message": "No LLM configured — set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL in backend/.env.",
        }
        return

    # DATA PROFILING — explore the dataset, produce a free-form DataReport
    try:
        yield from _profile(seed, uploads, uploads_dir, state)
    except LLMError as e:
        yield {"type": "error", "fatal": True, "message": f"Profiling failed — run stopped: {e}"}
        return

    # LITERATURE — once per prompt
    try:
        yield from _literature(seed, Researcher(), mem, state)
    except LLMError as e:
        yield {"type": "error", "fatal": True, "message": f"Literature failure — run stopped: {e}"}
        return

    # RESEARCH ROUNDS — LLM-driven, no hardcoded mode
    for step in range(1, iterations + 1):
        yield {"type": "round", "step": step, "total": iterations}
        try:
            yield from _round_core(
                step, seed, engine, uploads, uploads_dir, synthesist, coder, visualizer, mem, state
            )
        except LLMError as e:
            yield {
                "type": "error",
                "step": step,
                "fatal": True,
                "message": f"LLM failure — run stopped: {e}",
            }
            return

    # EXPORTER — only when the user's prompt asks for a platform export
    wants_export = bool(
        re.search(r"\b(pine\s?script|tradingview|mq5|mql5|metatrader|export)\b", seed, re.I)
    )
    if wants_export and state["best"] is not None:
        yield from _export(seed, state)

    # REPORTER
    yield {
        "type": "handoff",
        "step": iterations,
        "agent": "Archivist",
        "action": "writing the answer",
    }
    try:
        report = _run_report(seed, mem, state)
        answer = report["summary"]
    except LLMError as e:
        answer = f"(Reporter failed: {e})"
    yield {
        "type": "done",
        "keepers": _sanitize({"items": []})["items"],
        "best": _sanitize(state["best"]) if state["best"] else None,
        "tested": len(mem.entries),
        "suggestion": state["feedback"],
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# STAGES
# ---------------------------------------------------------------------------


def _profile(seed: str, uploads: list[str], uploads_dir, state) -> Iterator[dict]:
    """Profile the dataset — backend reads raw file (any format), LLM interprets."""
    import numpy as np

    yield {
        "type": "handoff",
        "step": 0,
        "agent": "Data Profiler",
        "action": "exploring the dataset",
    }

    raw: dict = {}
    from app.quant.dataset import NPZ_PATH

    src = NPZ_PATH if NPZ_PATH.exists() else None
    if uploads_dir:
        ud = Path(str(uploads_dir))
        for f in sorted(ud.iterdir()):
            if f.is_file():
                src = f
                break

    if src:
        ext = src.suffix.lower()
        try:
            if ext == ".npz":
                with np.load(src, allow_pickle=False) as z:
                    raw["format"] = "npz"
                    raw["keys"] = list(z.files)
                    raw["shapes"] = {k: list(z[k].shape) for k in z.files}
                    raw["dtypes"] = {k: str(z[k].dtype) for k in z.files}
            elif ext == ".csv":
                import pandas as pd

                df = pd.read_csv(src, nrows=1000)
                raw["format"] = "csv"
                raw["columns"] = list(df.columns)
                raw["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
                raw["rows"] = len(df)
                raw["head"] = df.head(5).to_dict(orient="records")
                raw["describe"] = df.describe(include="all").to_dict() if len(df) > 0 else {}
            elif ext in (".parquet", ".pq"):
                import pandas as pd

                df = pd.read_parquet(src)
                if len(df) > 1000:
                    df = df.head(1000)
                raw["format"] = "parquet"
                raw["columns"] = list(df.columns)
                raw["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
                raw["rows"] = len(df)
                raw["head"] = df.head(5).to_dict(orient="records")
                raw["describe"] = df.describe(include="all").to_dict() if len(df) > 0 else {}
            else:
                raw["format"] = ext
                raw["path"] = str(src)
        except Exception as e:
            raw["read_error"] = str(e)

    enriched = DataProfiler.interpret(raw, seed)
    state["data_report"] = enriched
    yield {
        "type": "profile",
        "step": 0,
        "agent": "Data Profiler",
        "report": enriched.content,
        "stdout": "",
    }


def _literature(seed, researcher: Researcher, mem, state) -> Iterator[dict]:
    """Plan queries → search → read top papers into briefs."""
    yield {
        "type": "handoff",
        "step": 0,
        "agent": "Researcher",
        "action": "planning the literature search",
    }
    try:
        queries = researcher.plan_queries(seed)
    except LLMError:
        queries = [seed]
    if not queries:
        queries = [seed]

    candidates, seen = [], set()
    for query in queries:
        papers = search(query, limit=8)
        yield {
            "type": "search",
            "step": 0,
            "agent": "Researcher",
            "query": query,
            "results": [
                {"title": p.title, "year": p.year, "citations": p.citations, "source": p.source}
                for p in papers
            ],
        }
        for p in papers:
            key = p.title.strip().lower()
            if key and key not in seen:
                seen.add(key)
                candidates.append(p)

    candidates = rank_papers(seed, candidates)[:MAX_PAPERS_READ]
    if not candidates:
        return

    yield {
        "type": "handoff",
        "step": 0,
        "agent": "Researcher",
        "action": f"reading {len(candidates)} papers",
    }
    read_briefs = []
    with ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as pool:
        futures = {pool.submit(_read_one, seed, p): p for p in candidates}
        for fut in as_completed(futures):
            paper = futures[fut]
            try:
                brief = fut.result()
            except LLMError as e:
                yield {
                    "type": "error",
                    "step": 0,
                    "fatal": False,
                    "message": f"reader failed on '{paper.title[:60]}': {e}",
                }
                continue
            if brief is None:
                continue
            read_briefs.append(brief)
            yield {
                "type": "reading",
                "step": 0,
                "agent": "Researcher",
                "title": brief.title,
                "year": brief.year,
                "relevant": brief.relevant,
                "claim": brief.claim,
                "basis": "full text" if paper.full_text else "abstract",
                "citations": paper.citations,
                "method_steps": brief.method_steps,
                "reported_numbers": brief.reported_numbers,
            }
            if brief.relevant:
                state["briefs"].append(brief)

    if not state["briefs"] and read_briefs:
        state["briefs"] = read_briefs


def _read_one(seed: str, paper):
    paper.full_text = fetch_full_text(paper)
    if not paper.full_text and not paper.abstract:
        return None
    return read_paper(seed, paper)


def _briefs_context(state) -> str:
    return "\n\n".join(b.to_context() for b in state["briefs"])


def _round_core(
    step, seed, engine, uploads, uploads_dir, synthesist, coder, visualizer, mem, state
):
    """Core of one research round — LLM-driven, no hardcoded mode."""

    # SYNTHESIST — grounded in data report + literature
    yield {
        "type": "handoff",
        "step": step,
        "agent": "Synthesist",
        "action": "designing experiment from data report + literature",
    }
    plan = synthesist.synthesize(
        goal=seed,
        data_report=state["data_report"],
        briefs=_briefs_context(state),
    )
    state["last_plan"] = plan
    yield {
        "type": "agent_msg",
        "step": step,
        "agent": "Synthesist",
        "title": plan.goal[:80] if plan.goal else "experiment",
        "content": json.dumps(
            {"goal": plan.goal, "methodology": plan.methodology, "evaluation": plan.evaluation},
            indent=2,
        ),
        "detail": "\n".join(f"{i + 1}. {m}" for i, m in enumerate(plan.methodology)),
    }

    image = "alphaseek-sandbox:latest"

    # CODING AGENT — generate code from the plan
    yield {
        "type": "handoff",
        "step": step,
        "agent": "Quant Coder",
        "action": "implementing the plan",
    }
    idea = {
        "experiment_plan": plan,
        "data_report": state["data_report"],
        "last_feedback": state["feedback"],
        "last_stdout": state.get("final_stdout", ""),
        "last_error": "",
    }
    code, _ = coder.run(
        step,
        idea,
        uploads,
        uploads_dir,
        state["tried_code"],
        briefs_context=_briefs_context(state),
        lessons=state["lessons"],
        image=image,
    )
    yield {
        "type": "code",
        "step": step,
        "agent": "Quant Coder",
        "filename": "research.py",
        "code": code,
    }

    # Run in sandbox
    try:
        bt = run_factor_code(code, uploads_dir=uploads_dir, image=image)
    except FactorError as e:
        msg = str(e)
        lesson = msg.splitlines()[0][:200] if msg else "sandbox error"
        if lesson not in state["lessons"]:
            state["lessons"] = (state["lessons"] + [lesson])[-LESSONS_WINDOW:]
        state["feedback"] = msg
        state["tried_code"].append(code)
        yield {"type": "run_error", "step": step, "message": msg}
        return

    bt_safe = _sanitize(bt)
    state["coder_summary"] = bt.get("summary", "")
    state["final_stdout"] = bt_safe.get("stdout", "")

    # VISUALIZER — render charts from whatever .npz arrays the coder saved
    data_artifacts = bt.get("data_artifacts", [])
    if data_artifacts:
        try:
            yield {
                "type": "handoff",
                "step": step,
                "agent": "Visualizer",
                "action": "rendering charts",
            }
            # Load the saved arrays so the visualizer prompt knows the keys
            import numpy as _np

            manifest = {}
            for art in data_artifacts:
                try:
                    with _np.load(ARTIFACT_STORE / art, allow_pickle=False) as _z:
                        manifest.update({k: _z[k] for k in _z.files})
                except Exception:  # noqa: BLE001
                    continue
            viz_code = visualizer.render(
                [{k: manifest[k].shape for k in manifest}],
                goal=seed,
                evaluation=plan.evaluation if plan else "",
            )

            import re as _re

            _clean = []
            for _line in viz_code.split("\n"):
                _s = _line.strip()
                if _s.startswith("import alphaseek") or _s.startswith("from alphaseek"):
                    continue
                _clean.append(_line)
            viz_code = "\n".join(_clean)

            # The coder saved arrays to af.OUT; load them in the viz script
            preamble = ["import alphaseek as af", "import numpy as np", "import glob, os"]
            preamble.append("_m = {}")
            preamble.append("for _f in glob.glob(os.path.join(af.OUT, '*.npz')):")
            preamble.append("    with np.load(_f) as _z: _m.update({k: _z[k] for k in _z.files})")
            for k in manifest:
                preamble.append(f"{k.replace('-', '_')} = _m.get('{k}')")
            viz_code = "\n".join(preamble) + "\n\n" + viz_code
            yield {
                "type": "code",
                "step": step,
                "agent": "Visualizer",
                "filename": "viz.py",
                "code": viz_code,
                "revised": True,
            }

            viz = run_factor_code(viz_code, uploads_dir=uploads_dir, image=image)
            if viz.get("artifacts"):
                bt["artifacts"] = viz["artifacts"]
                yield {
                    "type": "backtest",
                    "step": step,
                    "agent": "Visualizer",
                    "result": _sanitize(bt),
                    "engine": viz.get("engine", engine),
                    "exploration": False,
                }
        except (LLMError, FactorError) as e:
            yield {
                "type": "error",
                "step": step,
                "fatal": False,
                "message": f"viz failed (results stand): {e}",
            }

    state["tried_code"].append(code)
    bt["code"] = code
    bt_safe = _sanitize(bt)

    # MEMORY — store result
    mem.add(plan.goal[:60] if plan.goal else "experiment", bt_safe)
    if state["best"] is None:
        state["best"] = _sanitize({"name": plan.goal[:60], "code": code, **bt_safe})
    yield {
        "type": "memory",
        "step": step,
        "agent": "Archivist",
        "summary": mem.summary(),
        "keepers": len(mem.keepers()) if hasattr(mem, "keepers") else 0,
    }


def _run_report(seed, mem, state):
    """Final report — uses last experiment plan if available."""
    plan = state.get("last_plan") or ExperimentPlan(
        goal=seed, data_columns={}, methodology=[], evaluation="", constraints=[]
    )
    return Reporter.report(goal=seed, plan=plan, run_stdout=state.get("final_stdout", ""))


def _export(seed: str, state) -> Iterator[dict]:
    """Generate Pine Script / MQL5 when the user asked for a platform export.

    Triggered only by keyword match in the seed prompt. The Exporter LLM reads
    the best run's result dict + stdout and emits both platform codes.
    """
    from app.storage import put_text

    yield {
        "type": "handoff",
        "step": 0,
        "agent": "Exporter",
        "action": "translating strategy to Pine Script / MQL5",
    }
    best = state["best"] or {}
    # best holds a mix of metadata + metrics; pass what we have
    pine, mql5 = Exporter.export(best, state.get("final_stdout", ""))
    if not pine and not mql5:
        yield {
            "type": "error",
            "fatal": False,
            "message": "Exporter failed to generate platform code.",
        }
        return

    for lang, code in (("pine", pine), ("mql5", mql5)):
        if not code:
            continue
        fname = f"export_{lang}_{abs(hash(seed)) % 100000}.txt"
        try:
            put_text(code, fname)
        except Exception:  # noqa: BLE001
            fname = None
        yield {
            "type": "export",
            "step": 0,
            "agent": "Exporter",
            "lang": lang,
            "code": code,
            "filename": fname,
        }
