"""The agent's memory — what it has tried and learned, fed back into each prompt.

Records every experiment run — no assumptions about metric names, data structure,
or evaluation criteria. The LLM decides what matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    name: str
    result: dict


@dataclass
class Memory:
    entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, name: str, bt: dict, expr: str = "", v: dict | None = None) -> None:
        self.entries.append(MemoryEntry(name=name, result=bt))

    def summary(self, top_k: int = 4) -> str:
        """A compact natural-language summary for the LLM prompt."""
        if not self.entries:
            return ""
        names = [e.name for e in self.entries[-top_k:]]
        return f"Tried {len(self.entries)} experiments. Recent: {', '.join(names)}."

    def keepers(self) -> list[MemoryEntry]:
        return [e for e in self.entries if e.result]


def rebuild_memory(events: list[dict]) -> Memory:
    mem = Memory()
    for ev in events:
        if ev.get("type") == "memory":
            name = ev.get("name", ev.get("summary", ""))
            result = ev.get("result", {})
            mem.add(name, result)
    return mem
