"""The lead orchestrator — runs the multi-agent research team, streaming handoffs.

Per prompt:

    LITERATURE (once) : Researcher plans queries -> search -> read top papers
                        into PaperBriefs (relevance-gated, parallel)
    per round         : Synthesist connects papers into a novel plan (+ the pip
                        requirements it needs) -> PROVISION installs them into a
                        sandbox image -> Quant Coder implements the MATH (agentic
                        loop, records a manifest) -> Visualizer renders charts
                        from the manifest -> Risk Critic -> Archivist
    finally           : Reporter answers the user's question, grounded on what ran

The Critic's suggestion feeds the next synthesis. Accepts an existing Memory so a
session continues across prompts, and an uploads dir the sandbox mounts read-only.

Failure policy: LLM failures are surfaced as fatal error events and stop the run
(no silent fallbacks); a single paper failing to read, or a provisioning miss, is
a visible non-fatal event. Sandbox failures feed the coder's repair loop and
become lessons (Reflexion, sliding window of 3).
"""
from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.agent.agents import (CodingAgent, Reporter, Researcher, RiskCritic,
                              Synthesist, Visualizer)
from app.agent.evaluate import verdict
from app.agent.llm import LLMError, get_llm
from app.agent.memory import Memory
from app.agent.reader import rank_papers, read_paper
from app.agent.research_tools import fetch_full_text, search
from app.quant.dataset import dataset_meta
from app.quant.docker_sandbox import docker_available
from app.quant.provision import provision

TEAM = [
    {"name": "Researcher", "role": "reads literature"},
    {"name": "Synthesist", "role": "connects papers into a novel plan"},
    {"name": "Quant Coder", "role": "implements the math"},
    {"name": "Backtester", "role": "runs it in the sandbox"},
    {"name": "Visualizer", "role": "renders interactive charts"},
    {"name": "Risk Critic", "role": "grades and checks overfit"},
    {"name": "Archivist", "role": "maintains memory"},
]

MAX_PAPERS_READ = 2         # per prompt — one reduce LLM call each
READ_CONCURRENCY = 1        # sequential: gentler on tight per-minute LLM rate limits
LESSONS_WINDOW = 3          # Reflexion Ω: sliding window of failure lessons


def research(
    seed: str,
    iterations: int = 8,
    seed_num: int = 7,
    mem: Memory | None = None,
    uploads_dir=None,
    uploads: list[str] | None = None,
) -> Iterator[dict]:
    llm = get_llm()
    uploads = uploads or []
    researcher, synthesist = Researcher(), Synthesist()
    coder, critic, reporter, visualizer = CodingAgent(), RiskCritic(), Reporter(), Visualizer()
    mem = mem or Memory()
    state = {"feedback": "", "best": None, "tried_code": [],
             "briefs": [], "lessons": []}

    engine = "docker" if docker_available() else "inprocess"
    yield {"type": "start", "seed": seed, "mode": llm.mode, "model": llm.model,
           "iterations": iterations, "team": TEAM, "engine": engine,
           "dataset": dataset_meta(), "resumed": len(mem.entries) > 0,
           "uploads": uploads}

    if not llm.configured:
        yield {"type": "error", "fatal": True,
               "message": "No LLM configured — set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL in backend/.env."}
        return

    # LITERATURE — once per prompt: plan queries, search, read into briefs.
    try:
        yield from _literature(seed, researcher, mem, state)
    except LLMError as e:
        yield {"type": "error", "fatal": True,
               "message": f"LLM failure during literature stage — run stopped: {e}"}
        return

    for step in range(1, iterations + 1):
        yield {"type": "round", "step": step, "total": iterations}
        try:
            yield from _round(step, seed, engine, uploads, uploads_dir,
                              synthesist, coder, critic, visualizer, mem, state)
        except LLMError as e:
            yield {"type": "error", "step": step, "fatal": True,
                   "message": f"LLM failure — run stopped: {e}"}
            return

    # REPORTER: answer the user's question directly.
    yield {"type": "handoff", "step": iterations, "agent": "Archivist",
           "action": "writing the answer"}
    try:
        report = reporter.report(seed, mem.summary(top_k=6), state["best"], len(mem.entries),
                                 coder_summary=state.get("coder_summary", ""),
                                 final_stdout=state.get("final_stdout", ""),
                                 briefs_context=_briefs_context(state))
        answer = report["answer"]
        next_steps = report.get("next_steps", [])[:3]
    except LLMError as e:
        answer = f"(Reporter failed: {e})"
        next_steps = []
    keepers = [{"name": e.name, "expr": e.expr, "grade": e.grade,
                "sharpe": e.sharpe, "mean_ic": e.mean_ic} for e in mem.keepers()]
    yield {"type": "done", "keepers": keepers, "best": state["best"],
           "tested": len(mem.entries), "suggestion": state["feedback"], "answer": answer,
           "next_steps": next_steps}


# --------------------------------------------------------------------- stages
def _literature(seed, researcher, mem, state) -> Iterator[dict]:
    """Plan queries -> search (streamed) -> read top papers into briefs (parallel)."""
    yield {"type": "handoff", "step": 0, "agent": "Researcher",
           "action": "planning the literature search"}
    # Query planning is a nicety — if the LLM is unavailable, search on the seed
    # itself rather than aborting the whole run before any work is done.
    try:
        queries = researcher.plan_queries(seed, mem.summary())
    except LLMError:
        queries = []
    if not queries:
        queries = [seed]

    candidates, seen = [], set()
    for query in queries:
        papers = search(query, limit=5)
        yield {"type": "search", "step": 0, "agent": "Researcher", "query": query,
               "results": [{"title": p.title, "year": p.year,
                            "citations": p.citations, "source": p.source}
                           for p in papers]}
        for p in papers:
            key = p.title.strip().lower()
            if key and key not in seen:
                seen.add(key)
                candidates.append(p)

    # Rank by semantic relevance of the paper to the goal (not raw popularity),
    # so the on-point paper is read before any famous-but-tangential survey.
    candidates = rank_papers(seed, candidates)[:MAX_PAPERS_READ]
    if not candidates:
        return

    # Read the top papers CONCURRENTLY — each read is fetch (Jina, ~30s) + embed
    # + one LLM reduce, all independent. Serial reads dominated wall-clock.
    yield {"type": "handoff", "step": 0, "agent": "Researcher",
           "action": f"reading {len(candidates)} papers"}
    with ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as pool:
        futures = {pool.submit(_read_one, seed, p): p for p in candidates}
        for fut in as_completed(futures):
            paper = futures[fut]
            try:
                brief = fut.result()
            except LLMError as e:
                yield {"type": "error", "step": 0, "fatal": False,
                       "message": f"reader failed on '{paper.title[:60]}': {e}"}
                continue
            if brief is None:
                continue
            yield {"type": "reading", "step": 0, "agent": "Researcher",
                   "title": brief.title, "year": brief.year,
                   "relevant": brief.relevant, "claim": brief.claim,
                   "basis": "full text" if paper.full_text else "abstract",
                   "citations": paper.citations,
                   "method_steps": brief.method_steps,
                   "reported_numbers": brief.reported_numbers}
            if brief.relevant:
                state["briefs"].append(brief)


def _read_one(seed: str, paper):
    """Fetch full text and reduce to a brief — the unit of parallel work."""
    paper.full_text = fetch_full_text(paper)
    if not paper.full_text and not paper.abstract:
        return None
    return read_paper(seed, paper)


def _briefs_context(state) -> str:
    return "\n\n".join(b.to_context() for b in state["briefs"])


def _round(step, seed, engine, uploads, uploads_dir,
           synthesist, coder, critic, visualizer, mem, state) -> Iterator[dict]:
    # 1) SYNTHESIST — connect the papers into a novel, testable plan ----------
    yield {"type": "handoff", "step": step, "agent": "Synthesist",
           "action": "connecting papers into a novel plan"}
    idea = synthesist.synthesize(seed, _briefs_context(state), mem.summary(),
                                 state["feedback"], mem.tried_exprs(), uploads)
    method_lines = "\n".join(f"{i+1}. {m}" for i, m in enumerate(idea.get("methodology", [])))
    yield {"type": "agent_msg", "step": step, "agent": "Synthesist",
           "title": idea["name"], "content": idea["hypothesis"],
           "novelty": idea.get("novelty", ""), "detail": method_lines,
           "validation_targets": idea.get("validation_targets", []),
           "acceptance": idea.get("acceptance", ""),
           "references": idea.get("references", []),
           "requirements": idea.get("requirements", [])}

    # 2) PROVISION — install the agent-declared libraries into a sandbox image.
    image = "alphaseek-sandbox:latest"
    reqs = idea.get("requirements", [])
    if reqs:
        yield {"type": "handoff", "step": step, "agent": "Quant Coder",
               "action": f"provisioning libraries: {', '.join(reqs)}"}
        prov = provision(reqs)
        image = prov.image
        yield {"type": "provision", "step": step, "agent": "Quant Coder",
               "installed": prov.installed, "skipped": prov.skipped,
               "cached": prov.cached, "error": prov.error}

    # 3+4) CODING AGENT — MATH ONLY, agentic loop; records a chart manifest.
    yield {"type": "handoff", "step": step, "agent": "Quant Coder",
           "action": "starting agentic coding session"}
    final = yield from _observe_lessons(
        coder.run(step, idea, uploads, uploads_dir, state["tried_code"],
                  briefs_context=_briefs_context(state), lessons=state["lessons"],
                  image=image), state)
    if final is None:
        state["feedback"] = "The last idea never produced a submitted signal — go simpler."
        return
    code, bt = final["code"], final["bt"]
    state["coder_summary"] = final.get("summary", "")
    state["final_stdout"] = bt.get("stdout", "")

    # 5) VISUALIZER — a SEPARATE stage that loads the manifest and plots. It
    # cannot break the math (already submitted); charts are chosen from results.
    manifest_keys = bt.get("manifest_keys", [])
    manifest_path = bt.get("manifest_path")
    if manifest_keys and manifest_path:
        try:
            yield {"type": "handoff", "step": step, "agent": "Visualizer",
                   "action": "rendering interactive charts from results"}
            viz_code = visualizer.render(seed, idea, manifest_keys, bt.get("stdout", ""))
            yield {"type": "code", "step": step, "agent": "Visualizer",
                   "filename": f"{idea['name']}_viz.py", "code": viz_code, "revised": True}
            from app.quant.backtest import FactorError
            from app.quant.docker_sandbox import run_factor_code
            try:
                viz = run_factor_code(viz_code, uploads_dir=uploads_dir, image=image,
                                      manifest_src=manifest_path)
                if viz.get("artifacts"):
                    bt["artifacts"] = viz["artifacts"]      # charts attach to the result
                    yield {"type": "backtest", "step": step, "agent": "Visualizer",
                           "name": idea["name"], "result": bt,
                           "engine": viz.get("engine", engine), "exploration": False}
            except FactorError as e:
                yield {"type": "run_error", "step": step,
                       "message": f"visualization failed (result stands, no charts): {str(e)[:200]}",
                       "attempt": 0}
        except LLMError:
            pass    # charts are a bonus — never fail the round over them

    state["tried_code"].append(code)
    bt["expr"] = idea.get("construction", idea["name"])
    bt["code"] = code

    # 6) RISK CRITIC -------------------------------------------------------
    yield {"type": "handoff", "step": step, "agent": "Risk Critic",
           "action": "grading the result"}
    v = verdict(bt)
    review = critic.review(idea, bt, v, goal=seed)
    state["feedback"] = review["suggestion"]
    yield {"type": "verdict", "step": step, "agent": "Risk Critic",
           "name": idea["name"], "verdict": v, "review": review}

    # 7) ARCHIVIST ---------------------------------------------------------
    expr_label = bt.get("expr", idea["name"])
    mem.add(idea["name"], expr_label, bt, v)
    if state["best"] is None or bt["sharpe"] > state["best"]["sharpe"]:
        state["best"] = {"name": idea["name"], "expr": expr_label, **bt, "verdict": v}
    yield {"type": "memory", "step": step, "agent": "Archivist",
           "kept": v["keep"], "summary": mem.summary(), "keepers": len(mem.keepers())}


def _observe_lessons(gen, state) -> Iterator[dict]:
    """Pass the coder's events through; harvest run errors as lessons (Ω=3)."""
    while True:
        try:
            ev = next(gen)
        except StopIteration as stop:
            return stop.value
        if ev.get("type") == "run_error" and ev.get("message"):
            lesson = ev["message"].splitlines()[0][:200]
            if lesson not in state["lessons"]:
                state["lessons"] = (state["lessons"] + [lesson])[-LESSONS_WINDOW:]
        yield ev
