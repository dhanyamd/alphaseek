"""The agent's memory — what it has tried and learned, fed back into each prompt.

This is the "context engineering / memory systems" the JD calls out. After every
factor is evaluated, we record it. Before proposing the next one, we summarize
the best and worst results so the model builds on what worked and avoids what
failed. Memory is per research session.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    name: str
    expr: str
    grade: str | None = None
    sharpe: float | None = None
    mean_ic: float | None = None
    keep: bool = False


@dataclass
class Memory:
    entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, name: str, expr: str, bt: dict, v: dict | None = None) -> None:
        self.entries.append(
            MemoryEntry(
                name=name,
                expr=expr,
                grade=v.get("grade") if v else None,
                sharpe=float(round(bt["sharpe"], 2)) if bt.get("sharpe") is not None else None,
                mean_ic=float(round(bt["mean_ic"], 4)) if bt.get("mean_ic") is not None else None,
                keep=bool(v["keep"]) if v and "keep" in v else False,
            )
        )

    def tried_exprs(self) -> list[str]:
        return [e.expr for e in self.entries] + [e.name for e in self.entries]

    def summary(self, top_k: int = 4) -> str:
        """A compact natural-language summary for the LLM prompt."""
        if not self.entries:
            return ""
        ranked = sorted(
            self.entries, key=lambda e: e.sharpe if e.sharpe is not None else -999, reverse=True
        )
        lines = []
        for e in ranked[:top_k]:
            verdict = "WORKED" if e.keep else "failed"
            sharpe_s = f"Sharpe {e.sharpe}" if e.sharpe is not None else "non-signal"
            ic_s = f", IC {e.mean_ic}" if e.mean_ic is not None else ""
            grade_s = f"grade {e.grade}" if e.grade else ""
            lines.append(f"- {e.name} ({e.expr}) -> {verdict}, {sharpe_s}{ic_s}, {grade_s}")
        worst = ranked[-1]
        if worst not in ranked[:top_k]:
            w_sharpe = f"Sharpe {worst.sharpe}" if worst.sharpe is not None else "non-signal"
            lines.append(f"- worst so far: {worst.name} ({worst.expr}) {w_sharpe}")
        return "\n".join(lines)

    def keepers(self) -> list[MemoryEntry]:
        return [e for e in self.entries if e.keep]


def rebuild_memory(events: list[dict]) -> Memory:
    """Reconstruct the Archivist's memory from a session's stored events so a
    follow-up prompt continues where the last run left off."""
    mem = Memory()
    pending: dict = {}
    for ev in events:
        if ev.get("type") == "backtest":
            pending[ev.get("name")] = ev.get("result", {})
        elif ev.get("type") == "verdict" and ev.get("name") in pending:
            bt = pending.pop(ev["name"])
            mem.add(ev["name"], bt.get("expr", ev["name"]), bt, ev.get("verdict", {}))
    # Remaining backtests without a matching verdict (general mode) still get recorded
    for name, bt in pending.items():
        mem.add(name, bt.get("expr", name), bt)
    return mem
