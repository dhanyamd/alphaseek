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
    grade: str
    sharpe: float
    mean_ic: float
    keep: bool


@dataclass
class Memory:
    entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, name: str, expr: str, bt: dict, v: dict) -> None:
        self.entries.append(MemoryEntry(
            name=name, expr=expr, grade=v["grade"],
            sharpe=round(bt["sharpe"], 2), mean_ic=round(bt["mean_ic"], 4), keep=v["keep"],
        ))

    def tried_exprs(self) -> list[str]:
        return [e.expr for e in self.entries] + [e.name for e in self.entries]

    def summary(self, top_k: int = 4) -> str:
        """A compact natural-language summary for the LLM prompt."""
        if not self.entries:
            return ""
        ranked = sorted(self.entries, key=lambda e: e.sharpe, reverse=True)
        lines = []
        for e in ranked[:top_k]:
            verdict = "WORKED" if e.keep else "failed"
            lines.append(f"- {e.name} ({e.expr}) -> {verdict}, grade {e.grade}, "
                         f"Sharpe {e.sharpe}, IC {e.mean_ic}")
        worst = ranked[-1]
        if worst not in ranked[:top_k]:
            lines.append(f"- worst so far: {worst.name} ({worst.expr}) Sharpe {worst.sharpe}")
        return "\n".join(lines)

    def keepers(self) -> list[MemoryEntry]:
        return [e for e in self.entries if e.keep]
