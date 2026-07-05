"""The Reader — turns a paper's full text into a structured, citable brief.

Follows PaperQA's retrieve→map→reduce shape, adapted for free-tier budgets:

  1. chunk    — 4,000-char overlapping windows over the full text
  2. map      — rank chunks by SEMANTIC relevance to the goal: local ONNX
                embeddings (fastembed, no torch/GPU) + cosine similarity,
                with a small lexical boost for method/results sections.
                Zero LLM calls; PaperQA uses an LLM scorer per chunk — this
                is the documented budget adaptation.
  3. reduce   — ONE LLM call turns the top chunks into a PaperBrief

At this corpus size (a few papers per run, ~60 chunks each) exact numpy
cosine search is strictly better than a vector-DB server; the retrieval
interface is one function, so Qdrant is a drop-in swap when a persistent
cross-session paper archive justifies it.

A brief that the model marks irrelevant is dropped — a paper is evidence only
when it actually informs the goal (relevance gate, no padding).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.llm import get_llm
from app.agent.research_tools import Paper

CHUNK_CHARS = 4_000
CHUNK_OVERLAP = 400
TOP_CHUNKS = 8

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was we with what which how does do can not".split()
)


@dataclass
class PaperBrief:
    title: str
    year: int | None
    relevant: bool
    claim: str = ""
    method_steps: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    reported_numbers: list[str] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)

    def cite(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title

    def to_context(self) -> str:
        """Compact rendering injected into downstream agent prompts."""
        steps = "\n".join(f"    {i + 1}. {s}" for i, s in enumerate(self.method_steps))
        nums = "; ".join(self.reported_numbers) or "(none extracted)"
        pits = "; ".join(self.pitfalls) or "(none)"
        return (f"PAPER: {self.cite()}\n"
                f"  claim: {self.claim}\n"
                f"  method:\n{steps or '    (not extracted)'}\n"
                f"  reported numbers: {nums}\n"
                f"  pitfalls: {pits}")


def read_paper(goal: str, paper: Paper) -> PaperBrief:
    """Produce a PaperBrief from a paper's text (full text when available)."""
    text = paper.full_text or paper.abstract
    excerpts = _top_chunks(goal, text) if paper.full_text else [text]
    return _brief_from_excerpts(goal, paper, excerpts)


def rank_papers(goal: str, papers: list[Paper]) -> list[Paper]:
    """Order candidates by semantic closeness of title+abstract to the goal.

    Citation count is a weak tiebreak only — a famous but off-topic survey must
    never outrank the on-point paper. Papers with no text to embed sink to the
    bottom. This is what enforces 'read the relevant ones only'.
    """
    import numpy as np

    scored = [p for p in papers if (p.title or p.abstract)]
    if len(scored) <= 1:
        return scored
    model = _get_embedder()
    q = np.asarray(next(iter(model.query_embed(goal))))
    docs = [f"{p.title}. {p.abstract}"[:2_000] for p in scored]
    d = np.asarray(list(model.embed(docs)))
    cosine = (d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q) + 1e-12)
    cite_norm = np.log1p([max(p.citations or 0, 0) for p in scored])
    cite_norm = cite_norm / (cite_norm.max() + 1e-12)
    order = np.argsort(cosine + 0.05 * cite_norm)[::-1]
    return [scored[i] for i in order.tolist()]


# ------------------------------------------------------------------ map (rank)
_embedder = None


def _get_embedder():
    """Lazy singleton — the ONNX model loads once per process (~2s warm)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedder


def _chunk(text: str) -> list[str]:
    step = CHUNK_CHARS - CHUNK_OVERLAP
    return [text[i:i + CHUNK_CHARS] for i in range(0, max(len(text) - CHUNK_OVERLAP, 1), step)]


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS}


_METHOD_TERMS = frozenset({"method", "methodology", "construct", "estimate", "regression",
                           "portfolio", "returns", "table", "results", "sharpe", "alpha"})


def _top_chunks(goal: str, text: str) -> list[str]:
    """Semantic retrieval: embedding cosine to the goal + method-section boost."""
    import numpy as np

    chunks = _chunk(text)
    if len(chunks) <= TOP_CHUNKS:
        return chunks

    model = _get_embedder()
    q = np.asarray(next(iter(model.query_embed(goal))))
    d = np.asarray(list(model.embed(chunks)))
    cosine = (d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q) + 1e-12)
    boost = np.array([len(_METHOD_TERMS & _terms(c)) / len(_METHOD_TERMS) for c in chunks])
    scores = cosine + 0.1 * boost

    top = sorted(np.argsort(scores)[-TOP_CHUNKS:].tolist())   # document order
    return [chunks[i] for i in top]


# --------------------------------------------------------------- reduce (brief)
_SYSTEM = (
    "You are the READER on a quant research team. From paper excerpts, extract a "
    "faithful structured brief for the research goal. Never invent content that is "
    "not in the excerpts; use empty values when the excerpts do not say. Respond "
    "with ONLY JSON: {"
    '"relevant": true/false (does this paper actually inform the goal?), '
    '"claim": "the paper\'s central claim in one sentence", '
    '"method_steps": ["the method as concrete ordered steps, equations as pseudocode"], '
    '"parameters": {"name": "value"} (windows, thresholds, formation periods), '
    '"reported_numbers": ["exact headline results with figures"], '
    '"pitfalls": ["caveats or failure conditions the paper states"]}'
)


def _brief_from_excerpts(goal: str, paper: Paper, excerpts: list[str]) -> PaperBrief:
    body = "\n---\n".join(excerpts)[:26_000]
    user = (f"RESEARCH GOAL: {goal}\n\n"
            f"PAPER: {paper.label()}  [{'full text' if paper.full_text else 'abstract only'}]\n\n"
            f"EXCERPTS:\n{body}\n\nExtract the brief. JSON only.")
    out = get_llm().chat_json(_SYSTEM, user, temperature=0.2, role="reader")
    return PaperBrief(
        title=paper.title,
        year=paper.year,
        relevant=bool(out.get("relevant", False)),
        claim=str(out.get("claim", ""))[:300],
        method_steps=[str(s)[:300] for s in out.get("method_steps", [])][:8],
        parameters=out.get("parameters", {}) if isinstance(out.get("parameters"), dict) else {},
        reported_numbers=[str(n)[:200] for n in out.get("reported_numbers", [])][:6],
        pitfalls=[str(p)[:200] for p in out.get("pitfalls", [])][:5],
    )
