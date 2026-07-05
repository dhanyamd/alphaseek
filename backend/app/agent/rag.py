"""Agentic RAG over the paper corpus, backed by Qdrant.

Persistent, cross-session vector store of paper chunks (fastembed vectors) plus
an agentic retrieval loop: retrieve → judge coverage → refine the query →
retrieve again, instead of one-shot search. Reuses the reader's embedder/chunker.
"""
from __future__ import annotations

import hashlib

from app.settings import settings

COLLECTION = "alphaseek_papers"
VECTOR_SIZE = 384                      # BAAI/bge-small-en-v1.5

_client = None


def _qc():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        _client = QdrantClient(url=settings.qdrant_url,
                               api_key=settings.qdrant_api_key or None,
                               check_compatibility=False)
        names = [c.name for c in _client.get_collections().collections]
        if COLLECTION not in names:
            _client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    return _client


def _pid(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:15], 16)


def index_paper(paper) -> int:
    """Embed a paper's chunks and upsert into Qdrant (idempotent by content id)."""
    from qdrant_client.models import PointStruct

    from app.agent.reader import _chunk, _get_embedder

    text = paper.full_text or paper.abstract
    if not text:
        return 0
    chunks = _chunk(text)
    vecs = list(_get_embedder().embed(chunks))
    points = [
        PointStruct(id=_pid(f"{paper.title}:{i}"),
                    vector=[float(x) for x in v],
                    payload={"title": paper.title, "year": paper.year, "chunk": ch})
        for i, (ch, v) in enumerate(zip(chunks, vecs))
    ]
    _qc().upsert(COLLECTION, points)
    return len(points)


def search_chunks(query: str, k: int = 8) -> list[dict]:
    """Vector search over all indexed papers; returns scored chunks."""
    from app.agent.reader import _get_embedder

    qv = [float(x) for x in next(iter(_get_embedder().query_embed(query)))]
    hits = _qc().query_points(COLLECTION, query=qv, limit=k).points
    return [{"title": h.payload.get("title"), "year": h.payload.get("year"),
             "chunk": h.payload.get("chunk"), "score": float(h.score)} for h in hits]


_REFINE_SYS = (
    "You steer literature retrieval for a quant team. Given the goal and the "
    "papers already retrieved, write ONE refined search query targeting the gaps "
    '(methods/authors not yet covered). Respond with ONLY JSON: {"query": "..."}'
)


def agentic_retrieve(goal: str, rounds: int = 2, per_round: int = 8) -> list[dict]:
    """Retrieve → judge coverage → refine query → retrieve again. Returns the
    unique chunks gathered, most relevant first."""
    from app.agent.llm import LLMError, get_llm

    seen: set[tuple] = set()
    gathered: list[dict] = []
    query = goal
    for _ in range(max(1, rounds)):
        for h in search_chunks(query, per_round):
            key = (h["title"], h["chunk"][:80])
            if key not in seen:
                seen.add(key)
                gathered.append(h)
        if len(gathered) >= per_round:          # enough coverage
            break
        try:
            out = get_llm().chat_json(
                _REFINE_SYS,
                f"GOAL: {goal}\nAlready have: {sorted({h['title'] for h in gathered})}\nJSON only.",
                role="researcher")
            query = str(out.get("query") or query)
        except LLMError:
            break
    return sorted(gathered, key=lambda h: h["score"], reverse=True)
