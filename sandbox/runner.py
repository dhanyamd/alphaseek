"""In-container runner — executes an agent-written RESEARCH SCRIPT on REAL data.

The runner is intentionally THIN. The agent writes ALL its own code with
pandas/numpy/scipy/sklearn and does EVERYTHING: it loads the data, discovers
column names, computes forward returns, evaluates its own strategy, saves
arrays, and prints results. The runner only provides file handles and captures
output — it makes no assumptions about schema, column names, or metrics.

    import alphaseek as af
    af.DATA    -> path to the dataset file (agent loads + inspects it)
    af.OUT     -> directory to save artifacts / arrays (agent picks names)
    af.uploads() -> paths to any user-uploaded files

The agent is expected to:
  - load af.DATA (np.load / pd.read_csv / etc.) and inspect the keys/columns
  - compute forward returns from whatever price column exists
  - evaluate the strategy however the research question demands
  - save any arrays it wants visualized to af.OUT (e.g. np.savez)
  - print a single JSON result line (metrics / conclusions) to stdout

Captures stdout, collects artifacts, prints one JSON result line.
"""

import io
import json
import os
import sys
import types
from contextlib import redirect_stderr, redirect_stdout

ART_DIR = os.environ.get("ARTIFACTS_DIR", "/out")
DATA_PATH = os.environ.get("DATA_PATH")
UPLOADS_DIR = os.environ.get("UPLOADS_DIR", "/uploads")


def make_alphaseek():
    """The MINIMAL sandbox contract — paths only, no hidden logic.

    The agent writes all the code itself; the runner just exposes:
      alphaseek.DATA    -> path to the dataset (agent loads + inspects)
      alphaseek.OUT     -> directory to save artifacts / arrays
      alphaseek.uploads() -> any user-uploaded file paths
    """
    mod = types.ModuleType("alphaseek")

    mod.DATA = DATA_PATH
    mod.OUT = ART_DIR
    mod.uploads = lambda: (
        [os.path.join(UPLOADS_DIR, f) for f in sorted(os.listdir(UPLOADS_DIR))]
        if os.path.isdir(UPLOADS_DIR)
        else []
    )
    mod.__doc__ = (
        "alphaseek sandbox: DATA (dataset path), OUT (output dir), "
        "uploads() (list of uploaded file paths)."
    )
    return mod


def main():
    code_path = sys.argv[1]
    with open(code_path) as fh:
        code = fh.read()

    # Strip stray markdown code fences the LLM sometimes wraps around the script
    # (e.g. ```python ... ```), which would otherwise crash exec().
    import re

    _fence = re.compile(r"^```(?:python|py)?\s*\n", re.IGNORECASE)
    _fence_end = re.compile(r"\n```\s*$", re.IGNORECASE)
    if _fence.match(code.strip()):
        code = _fence.sub("", code.strip())
    if _fence_end.search(code):
        code = _fence_end.sub("", code)
    code = re.sub(r"```(?:python|py)?\s*\n", "", code)
    code = re.sub(r"\n```", "", code)

    os.makedirs(ART_DIR, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    try:  # default dark theme so charts match the UI; scripts may override
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "figure.facecolor": "#111112",
                "axes.facecolor": "#111112",
                "savefig.facecolor": "#111112",
                "axes.edgecolor": "#3a3a3e",
                "axes.labelcolor": "#d4d4d8",
                "text.color": "#d4d4d8",
                "xtick.color": "#8a8a8f",
                "ytick.color": "#8a8a8f",
                "grid.color": "#26262a",
                "axes.grid": True,
                "grid.alpha": 0.6,
                "axes.prop_cycle": mpl.cycler(
                    color=[
                        "#fafafa",
                        "#7ee787",
                        "#79c0ff",
                        "#ff7b72",
                        "#d2a8ff",
                        "#ffa657",
                    ]
                ),
                "figure.dpi": 110,
                "font.size": 9,
            }
        )
    except Exception:
        pass

    stdout = io.StringIO()
    try:
        mod = make_alphaseek()
        sys.modules["alphaseek"] = mod
        # Auto-inject the import so the model can't forget it.
        code = "import alphaseek as af\n" + code
        ns = {"__name__": "__main__"}
        with redirect_stdout(stdout), redirect_stderr(stdout):
            exec(compile(code, "research.py", "exec"), ns)
    except Exception:
        # If the user's code raises, capture the traceback so the coder model
        # can see what broke and fix it.
        import traceback

        traceback.print_exc(file=stdout)
        stdout.write("\n(exception — coder should fix the error above)\n")

    out_text = stdout.getvalue()

    # Collect results. The agent is expected to print a JSON result line itself;
    # if it didn't, we surface whatever stdout it produced.
    result: dict = {"ok": True, "submitted": False}
    for ln in reversed(out_text.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                parsed = json.loads(ln)
                if isinstance(parsed, dict):
                    result = parsed
                    result["ok"] = True
                    result["submitted"] = True
                break
            except json.JSONDecodeError:
                continue

    if "Traceback" in out_text and not result.get("submitted"):
        result = {"ok": False, "error": out_text[-1500:]}

    result["stdout"] = out_text[-4000:]
    result["artifacts"] = (
        sorted(
            fn
            for fn in os.listdir(ART_DIR)
            if fn.lower().endswith((".png", ".svg", ".jpg", ".html", ".npz"))
        )
        if os.path.isdir(ART_DIR)
        else []
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
