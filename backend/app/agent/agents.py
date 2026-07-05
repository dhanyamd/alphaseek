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

import re

from app.agent.llm import LLMError, get_llm
from app.quant.dataset import dataset_meta

# The quant stack baked into the sandbox base image — the agent's default palette.
STACK = ("numpy, pandas, scipy, scikit-learn, statsmodels, "
         "empyrical (empyrical-reloaded), alphalens (alphalens-reloaded), "
         "arch (GARCH), cvxpy, quantstats")


def env_spec() -> str:
    meta = dataset_meta()
    data_line = (
        f"{meta.get('stocks', '?')} instruments x {meta.get('days', '?')} periods "
        f"({meta.get('start', '?')}..{meta.get('end', '?')})"
        if meta.get("source") == "real" else "backend-provided dataset"
    )
    return f"""SANDBOX CONTRACT — you write ALL the code yourself. Installed and
importable: {STACK}, matplotlib, plotly. No network, no pip.

`import alphaseek as af` gives you exactly four things:
  - af.DATA : path to a .npz of RAW market panels ({data_line}). LOAD and INSPECT it:
        import numpy as np
        d = np.load(af.DATA); print(d.files)      # discover the columns yourself
      Panels are `px_<name>` (T, N) arrays (e.g. px_close), plus 'tickers','dates'.
      There are NO pre-computed features — derive every feature from these raw panels.
  - af.backtest(signal) -> metrics dict {{sharpe, sharpe_net, mean_ic, ic_decay,
      turnover, max_drawdown, equity_curve}}. Use it to evaluate as you iterate.
  - af.submit(signal) -> the OFFICIAL graded metrics; call ONCE when finished.
      `signal` is a FULL (T, N) array. Grading is BLIND (forward returns hidden) —
      build signals CAUSALLY (past rows only) or the look-ahead guard rejects them.
  - af.OUT : a directory. Save arrays the chart stage needs with
      np.savez(f"{{af.OUT}}/manifest.npz", equity=..., ic=...). Do NOT draw charts.
  - af.uploads() -> paths of any user-uploaded files (pandas-readable).

Everything else — features, regressions, optimization, statistics — is your own
code with numpy/pandas/scipy/sklearn/statsmodels/arch/cvxpy."""


CHART_CRAFT = """CHART DESIGN SYSTEM (plotly, interactive, house style — follow exactly):
Always: fig.update_layout(template='plotly_dark', title=..., and axis titles). Name every
trace; add units; hovertemplates where useful. One figure per HTML file.

Pattern A — 3D SURFACE (parameter sweeps, term structures):
  fig = go.Figure(go.Surface(z=Z, x=xs, y=ys, colorscale="Viridis",
      contours={"z": {"show": True, "usecolormap": True, "project_z": True}}))
  fig.update_layout(scene={"xaxis_title": "...", "yaxis_title": "...",
      "zaxis_title": "...", "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 0.9}}})
  Overlay the optimum: go.Scatter3d(x=[x*], y=[y*], z=[z*], mode="markers+text").

Pattern B — MONTE CARLO FAN:
  faint sample paths: go.Scatter(x=mc["x"], y=path, line={"color": "rgba(121,192,255,0.08)"},
      showlegend=False) for each path;
  band: p95 then p5 with fill="tonexty", fillcolor="rgba(126,231,135,0.15)";
  bold median p50; annotate terminal stats (prob_loss, p5/p95) in the title.

Pattern C — ANNOTATED HEATMAP (regimes, correlations):
  go.Heatmap(z=grid, x=cols, y=rows, colorscale="Viridis", colorbar={"title": "..."})
  + text annotations of each cell value; axis titles naming the regime dimensions."""


# --------------------------------------------------------------------------- Researcher
class Researcher:
    """Plans the literature search — the queries most likely to surface the
    canonical papers for the goal."""

    plan_system = (
        "You are the senior RESEARCHER on a quant team. Given a research goal, plan "
        "the literature search: the 2-4 scholarly queries most likely to surface the "
        "canonical papers (seminal factor papers, construction details, known "
        "pitfalls). Mix one broad query with narrower ones naming specific methods "
        'or authors when the goal implies them. Respond with ONLY JSON: '
        '{"queries": ["...", "..."]}'
    )

    def plan_queries(self, seed: str, memory_summary: str) -> list[str]:
        out = get_llm().chat_json(
            self.plan_system,
            f"RESEARCH GOAL: {seed}\n"
            f"Findings so far:\n{memory_summary or '(first experiment)'}\n\nJSON only.",
            temperature=0.4, role="researcher",
        )
        queries = [str(q).strip() for q in out.get("queries", []) if str(q).strip()]
        return queries[:4]


# --------------------------------------------------------------------------- Synthesist
class Synthesist:
    """Connects the papers into ONE novel, testable research plan — the step
    where the team reasons across 2-3 sources and invents something to try."""

    system = (
        "You are the SYNTHESIST on a quant team working on real US equity data. You "
        "receive PAPER BRIEFS the team has read. Your job: connect 2-3 of them into "
        "ONE precise, NOVEL, testable experiment that serves the user's goal — reuse "
        "their real construction details (windows, formation periods, weighting, "
        "volatility targeting, hazard models) and name where each idea comes from. "
        "The novelty is the specific combination or twist you propose, not a vague "
        "aspiration. Obey any feature constraints in the goal. Where briefs report "
        "numbers, set validation targets so the code can reproduce and compare them.\n"
        f"The sandbox already has: {STACK}. List in `requirements` ONLY extra pip "
        "packages beyond that stack that your methodology genuinely needs (usually "
        "none). Respond with ONLY JSON: {"
        '"name": "snake_case_id", '
        '"hypothesis": "one-sentence economic thesis", '
        '"novelty": "what is new here and which papers it connects", '
        '"methodology": ["3-6 concrete ordered steps naming exact features, windows, '
        'estimators, comparisons and statistics"], '
        '"validation_targets": ["specific numbers from the briefs to reproduce/compare"], '
        '"acceptance": "the numeric evidence that would confirm or refute the hypothesis", '
        '"requirements": ["extra pip packages, usually empty"], '
        '"references": ["short cites of the briefs that informed this"]}'
    )

    def synthesize(self, seed: str, briefs_context: str, memory_summary: str,
                   feedback: str, tried: list[str], uploads: list[str]) -> dict:
        # Constraints stated in the goal ("uncorrelated with momentum", "only
        # low-vol") are the model's to read and honor — we do not pre-parse intent.
        uploads_note = f"\nUser uploads available via af.uploads(): {uploads}" if uploads else ""
        user = (
            f"RESEARCH GOAL: {seed}\n"
            f"Data: raw daily prices, returns, and volume (T,N) — the coder computes "
            f"all features/signals itself from these.{uploads_note}\n"
            "Honor any feature or approach constraints stated in the goal.\n\n"
            f"PAPER BRIEFS (from the team's Reader):\n{briefs_context or '(no relevant literature retrieved — design from domain knowledge, do not degrade)'}\n\n"
            f"Findings so far:\n{memory_summary or '(first experiment)'}\n"
            f"Critic's direction: {feedback or '(none)'}\n"
            f"Already tried (must differ): {', '.join(tried) if tried else 'none'}\n\n"
            "Synthesize the single most informative NEXT experiment. JSON only."
        )
        return self._validate(get_llm().chat_json(self.system, user,
                                                  temperature=0.5, role="researcher"))

    @staticmethod
    def _validate(out: dict) -> dict:
        """Check the hand-off contract. Core fields (hypothesis, methodology) are
        the Synthesist's actual output — if they are missing the synthesis failed,
        and we surface that rather than fabricating a blank plan (no silent
        fallback). Genuinely optional fields are normalized, not invented."""
        if not str(out.get("hypothesis", "")).strip() or not out.get("methodology"):
            raise LLMError("Synthesist returned no hypothesis/methodology — the "
                           "synthesis failed to produce a usable plan.")
        out["name"] = re.sub(r"[^a-zA-Z0-9_]+", "_", str(out.get("name", "experiment")))[:48] or "experiment"
        out["requirements"] = [str(r).strip() for r in out.get("requirements", []) if str(r).strip()][:8]
        for optional in ("novelty", "acceptance"):
            out.setdefault(optional, "")
        for optional in ("validation_targets", "references"):
            out.setdefault(optional, [])
        return out


RUN_TOOL = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": ("Execute a complete Python research script in the sandbox. Returns "
                        "JSON: submitted flag, official metrics when af.submit(signal) was "
                        "called, recorded manifest keys, captured stdout, or the error with "
                        "traceback."),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string",
                                    "description": "full python source, executed verbatim"}},
            "required": ["code"],
        },
    },
}]


def preflight(code: str) -> list[str]:
    """Deterministic pre-flight checks — catch cheap, high-signal mistakes before
    spending a sandbox run. Returns findings as feedback strings (never rewrites
    the code): the model still fixes them. This is a linter, not enforcement.
    """
    import ast

    # Syntax-only: universal and data-agnostic (no assumptions about column or
    # feature names — those vary per dataset). Everything else is caught at
    # runtime by the sandbox and fed back with a real traceback.
    try:
        ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno}). Fix the syntax and resend."]
    return []


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


ENGINEERING = """ENGINEERING STANDARDS:
- Implement the plan's method faithfully from the raw panels — you compute every
  feature/estimator yourself (empyrical for metrics, arch for GARCH,
  statsmodels/sklearn for regressions/hazard models, cvxpy for optimization).
- Where the plan compares variants, af.backtest each and print an aligned findings
  table; where the papers report numbers, print ours-vs-theirs.
- Vectorized numpy — no per-element Python loops over the full panel.
- Save the arrays a chart will need with np.savez(f"{af.OUT}/manifest.npz", ...).
- Draw NO charts here — a separate stage does that."""


class CodingAgent:
    """An autonomous quant-coding agent — the MATH stage. Writes no charts."""

    MAX_STEPS = 5

    def _system(self) -> str:
        return (
            "You are a senior quant researcher and engineer. You are handed a research "
            "plan grounded in real papers; implement its MATHEMATICS as production code, "
            "iterate against the backtest, and submit the graded signal. You write every "
            "line yourself — there is no framework or helper library to lean on beyond "
            "the tiny sandbox contract below.\n\n"
            + env_spec() + "\n\n" + ENGINEERING + "\n\n"
            "HOW YOU WORK\n"
            "- ONE tool: run_python(code) — a fresh sandbox each call (no state persists).\n"
            "- Load and inspect af.DATA first, then build the paper's construction with "
            "your own numpy/pandas. Iterate: run, read stdout/tracebacks, fix, refine.\n"
            "- QUOTING: double-quoted f-strings with single-quoted keys "
            "(f\"ic={m['mean_ic']:.4f}\") — never nest same-type quotes.\n"
            "- Your FINAL run must implement the full method, print the findings table, "
            "np.savez the chart arrays to af.OUT/manifest.npz, and call "
            "af.submit(final_signal) — that submit is REQUIRED or the run is wasted.\n"
            "- At most " + str(self.MAX_STEPS) + " runs — be efficient.\n"
            "- When done, reply with a 2-3 sentence summary WITHOUT calling the tool."
        )

    def run(self, step: int, idea: dict, uploads: list[str], uploads_dir,
            tried_code: list[str], briefs_context: str = "",
            lessons: list[str] | None = None, image: str | None = None):
        """Generator: yields UI events; returns {'code','bt','summary'} or None."""
        import json as _json

        from app.quant.backtest import FactorError
        from app.quant.docker_sandbox import run_factor_code

        llm = get_llm()
        uploads_note = f"\nUser uploads via af.uploads(): {uploads}" if uploads else ""
        prior = ("\n".join(c[:300] for c in tried_code[-2:])) if tried_code else "(none)"
        method = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(idea.get("methodology", [])))
        targets = "\n".join(f"  - {t}" for t in idea.get("validation_targets", []))
        lit = f"\nLITERATURE (implement the math these briefs describe):\n{briefs_context}\n" \
            if briefs_context else ""
        lessons_note = ""
        if lessons:
            lessons_note = "\nLESSONS FROM EARLIER FAILED RUNS (do not repeat these):\n" + \
                "\n".join(f"  - {ls}" for ls in lessons[-3:]) + "\n"
        messages: list[dict] = [{
            "role": "user",
            "content": (f"EXPERIMENT: {idea['name']}\nHYPOTHESIS: {idea['hypothesis']}\n"
                        f"NOVELTY: {idea.get('novelty', '')}\n"
                        f"METHODOLOGY:\n{method or '  (design it yourself, minimally)'}\n"
                        f"VALIDATION TARGETS (reproduce & compare):\n{targets or '  (none given)'}\n"
                        f"ACCEPTANCE: {idea.get('acceptance', '')}{uploads_note}{lit}{lessons_note}"
                        f"Code from earlier rounds (yours must differ):\n{prior}\n\n"
                        "Implement the MATH now. Compute the signal, np.savez the chart arrays to af.OUT/manifest.npz, and "
                        "END with af.submit(final_signal) — that last line "
                        "is mandatory or the run is wasted. Write no charts."),
        }]
        final = None
        summary = ""
        last_error = ""

        for turn in range(1, self.MAX_STEPS + 1):
            # Until we have a graded result, FORCE the tool call so the model
            # can't waste a turn "explaining" instead of running code; once it has
            # submitted, allow a free turn so it can end with a text summary.
            # Big max_tokens so a full script's tool-call JSON is never truncated.
            resp = llm.chat_tools(self._system(), messages, RUN_TOOL, role="coder",
                                  max_tokens=8000,
                                  tool_choice="auto" if final is not None else "required")
            if not resp["tool_calls"]:
                # A summary AFTER we already have a submitted result ends the turn.
                if final is not None and resp["content"].strip():
                    summary = resp["content"].strip()[:700]
                    yield {"type": "agent_msg", "step": step, "agent": "Quant Coder",
                           "title": "summary", "content": summary}
                    break
                # No result yet and no tool call: the model explained instead of
                # coding (or a failover returned empty). Make it visible and push
                # it to actually call run_python rather than silently ending.
                yield {"type": "run_error", "step": step, "attempt": turn,
                       "message": "coder did not call run_python — asked it to write and run code"}
                messages.append({"role": "assistant",
                                 "content": resp["content"] or "(no output)"})
                messages.append({"role": "user",
                                 "content": ("You did not call the run_python tool. Do not explain — "
                                             "CALL run_python now with the complete research script "
                                             "that ends in af.submit(final_signal).")})
                continue

            tc = resp["tool_calls"][0]
            code = _extract_code_arg(tc["arguments"])
            if not code.strip():
                messages.append({"role": "assistant", "content": resp["content"] or None,
                                 "tool_calls": [{"id": tc["id"], "type": "function",
                                                 "function": {"name": "run_python",
                                                              "arguments": tc["arguments"]}}]})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": _json.dumps({"error": (
                                     "your tool arguments could not be parsed (empty code). "
                                     "Resend as valid JSON: {\"code\": \"...\"} with newlines "
                                     "as \\n and double quotes escaped.")})})
                yield {"type": "run_error", "step": step,
                       "message": "unparseable tool call — asked the model to resend",
                       "attempt": turn}
                continue

            messages.append({"role": "assistant", "content": resp["content"] or None,
                             "tool_calls": [{"id": tc["id"], "type": "function",
                                             "function": {"name": "run_python",
                                                          "arguments": tc["arguments"]}}]})
            yield {"type": "code", "step": step, "agent": "Quant Coder",
                   "filename": f"{idea['name']}_run{turn}.py", "code": code,
                   "model": resp.get("model", "")}

            # Pre-flight lint — bounce cheap, certain-to-crash mistakes (syntax,
            # bare-API names) back to the model WITHOUT spending a sandbox run.
            lint = preflight(code)
            if lint:
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": _json.dumps({"lint_errors": lint,
                                     "note": "Fixed by you and resent — the sandbox did not run."})})
                yield {"type": "run_error", "step": step,
                       "message": "pre-flight: " + " | ".join(lint), "attempt": turn}
                continue

            yield {"type": "handoff", "step": step, "agent": "Backtester",
                   "action": f"executing run {turn}"}
            try:
                bt = run_factor_code(code, uploads_dir=uploads_dir, image=image or "alphaseek-sandbox:latest")
                brief = {k: round(bt[k], 4) for k in
                         ("sharpe", "sharpe_net", "mean_ic", "ic_decay", "turnover",
                          "max_drawdown") if k in bt}
                tool_result = {"submitted": bool(bt.get("submitted")), **brief,
                               "manifest_keys": bt.get("manifest_keys", []),
                               "stdout": (bt.get("stdout") or "")[-600:]}
                if not bt.get("submitted"):
                    got = bt.get("manifest_keys") or []
                    tool_result["hint"] = (
                        "This run did NOT call af.submit(final_signal) — nothing was graded. "
                        + (f"You already recorded {got} and the math ran cleanly; send the "
                           "SAME script with the single line af.submit(<your final (T,N) signal>) "
                           "added at the end." if got else
                           "Finish the methodology, np.savez chart arrays to af.OUT/manifest.npz, then "
                           "af.submit(final_signal)."))
                yield {"type": "backtest", "step": step, "agent": "Backtester",
                       "name": idea["name"], "result": bt,
                       "engine": bt.get("engine", ""),
                       "exploration": not bt.get("submitted")}
                if bt.get("submitted"):
                    final = {"code": code, "bt": bt}
            except FactorError as e:
                err = str(e)[:1000]
                tool_result = {"error": err}
                if err[:120] == last_error[:120]:
                    tool_result["hint"] = ("SAME error as your previous run — re-read the "
                                           "traceback line and take a different approach.")
                last_error = err
                yield {"type": "run_error", "step": step, "message": str(e)[:400],
                       "attempt": turn}
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": _json.dumps(tool_result)})

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
            "that tell the story of the result.\n\n"
            + env_spec() + "\n\n" + CHART_CRAFT + "\n\n"
            "RULES\n"
            "- Start with `d = af.manifest()` — a dict of the recorded arrays/values. "
            "Wrap numeric fields in np.asarray as needed.\n"
            "- Use ONLY manifest data (via af.manifest()). Do NOT recompute signals, "
            "do NOT call af.backtest or af.submit.\n"
            "- Choose the charts the RESULTS justify: a 3D surface for a sweep grid, a "
            "fan for MC paths, an annotated heatmap for regimes, lines for equity/IC.\n"
            "- Save each figure: fig.write_html(f\"{af.OUT}/<name>.html\", "
            "include_plotlyjs=\"cdn\").\n"
            "- Output ONLY the complete Python script."
        )

    def render(self, goal: str, idea: dict, manifest_keys: list[str], stdout: str) -> str:
        llm = get_llm()
        keys = ", ".join(manifest_keys) or "(none recorded)"
        user = (f"USER'S GOAL: {goal}\nEXPERIMENT: {idea['name']} — {idea['hypothesis']}\n"
                f"MANIFEST KEYS available via af.manifest(): {keys}\n"
                f"What the math printed:\n{(stdout or '')[:600]}\n\n"
                "Write the visualization script. Load the manifest, chart the story. Code only.")
        out = llm.chat(self._system(), user, temperature=0.3, max_tokens=4000, role="viz")
        return _unfence(out)


def _unfence(text: str) -> str:
    """Unwrap a ```python ... ``` markdown fence if the model wrapped its code.

    This only removes the delivery wrapper (like parsing tool-call JSON); the
    model's code inside is untouched."""
    t = text.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
    return (m.group(1) if m else t).strip()


# --------------------------------------------------------------------------- RiskCritic
class RiskCritic:
    system = (
        "You are the skeptical RISK CRITIC on a quant team working on REAL equity data. "
        "Calibration: mean IC 0.01-0.03 is meaningful; Sharpe above ~2 is suspicious; "
        "net-of-cost Sharpe is what matters for tradeability; heavy turnover erodes "
        "edges. Judge BOTH edge quality AND whether the experiment served the user's "
        "goal and its own acceptance criteria. Respond with ONLY JSON: "
        '{"assessment": "one sharp sentence", "suggestion": "one concrete direction for '
        'the next experiment"}.'
    )

    def review(self, idea: dict, bt: dict, v: dict, goal: str = "") -> dict:
        llm = get_llm()
        user = (
            f"USER'S GOAL: {goal}\n"
            f"Experiment `{idea['name']}` — {idea['hypothesis']}\n"
            f"Acceptance criteria: {idea.get('acceptance', '(none)')}\n"
            f"Result: Sharpe {bt['sharpe']:.2f} (net {bt.get('sharpe_net', 0):.2f}), "
            f"mean IC {bt['mean_ic']:.4f}, IC decay {bt['ic_decay']:.4f}, "
            f"turnover {bt['turnover']:.2f}, max drawdown {bt['max_drawdown']:.1%}.\n"
            f"Verdict: grade {v['grade']}, overfit={v['overfit']}, keep={v['keep']}.\n"
            "JSON only."
        )
        out = llm.chat_json(self.system, user, temperature=0.4, role="critic")
        out.setdefault("assessment", v["notes"][0] if v.get("notes") else "Reviewed.")
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

    def report(self, goal: str, memory_summary: str, best: dict | None, tested: int,
               coder_summary: str = "", final_stdout: str = "",
               briefs_context: str = "") -> dict:
        llm = get_llm()
        best_line = (
            f"Best: {best['name']} (Sharpe {best['sharpe']:.2f}, net {best.get('sharpe_net', 0):.2f}, "
            f"IC {best['mean_ic']:.4f}, grade {best['verdict']['grade']})"
            if best else "No factor produced a usable edge."
        )
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
