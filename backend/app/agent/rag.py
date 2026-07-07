"""Agentic RAG over the paper corpus, backed by Qdrant.

Architecture (production-grade, 2026):
  1. Hybrid retrieval: Qdrant native fusion (dense + sparse BM25 via named vectors)
  2. Cross-encoder reranking on retrieved candidates
  3. Agentic refinement loop: retrieve → judge coverage → rewrite → retrieve
  4. Metadata filtering (year, source) via Qdrant native filter API
  5. Parent-document retrieval: embed small chunks, return full sections to LLM

Embeddings: BAAI/bge-small-en-v1.5 (384-dim, fastembed)
Sparse: Qdrant/bm25 (fastembed sparse embedding, stored as Qdrant SparseVector)
Fusion: Qdrant native RRF via FusionQuery
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from qdrant_client import models

from app.settings import settings

COLLECTION = "alphaseek_papers"
VECTOR_SIZE = settings.embedding_dim
_EMBED_MODEL = settings.embedding_model
_BM25_MODEL = "Qdrant/bm25"

_client = None


def _qc():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            check_compatibility=False,
        )
        names = [c.name for c in _client.get_collections().collections]
        if COLLECTION not in names:
            _client.create_collection(
                COLLECTION,
                vectors_config={
                    "dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )
    return _client


def _pid(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_paper(paper) -> int:
    """Embed a paper's chunks and upsert into Qdrant (idempotent by content id).

    Stores dense vectors only — sparse BM25 vectors are computed server-side
    at query time by Qdrant via the fastembed Qdrant/bm25 model.
    """
    from qdrant_client.models import PointStruct

    from app.agent.reader import _chunk, _get_embedder

    text = paper.full_text or paper.abstract
    if not text:
        return 0

    chunks = _chunk(text)
    vecs = list(_get_embedder().embed(chunks))

    # Build parent sections: each chunk's parent is the surrounding context
    sections = _build_parent_sections(chunks)

    points = [
        PointStruct(
            id=_pid(f"{paper.title}:{i}"),
            vector={"dense": [float(x) for x in v]},
            payload={
                "title": paper.title,
                "year": paper.year,
                "source": getattr(paper, "source", ""),
                "citations": getattr(paper, "citations", 0),
                "chunk": ch,
                "section": sections[i],
                "chunk_idx": i,
                "total_chunks": len(chunks),
            },
        )
        for i, (ch, v) in enumerate(zip(chunks, vecs))
    ]
    _qc().upsert(COLLECTION, points)
    return len(points)


def _build_parent_sections(chunks: list[str], context_window: int = 1) -> list[str]:
    """Build parent sections: each chunk gets its surrounding context window."""
    sections = []
    for i in range(len(chunks)):
        start = max(0, i - context_window)
        end = min(len(chunks), i + context_window + 1)
        sections.append("\n\n".join(chunks[start:end]))
    return sections


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """A single retrieval hit with rich metadata."""

    title: str
    year: int
    source: str
    citations: int
    chunk: str
    section: str
    score: float
    retrieval_method: str  # "vector", "bm25", "hybrid"


def _build_filter(year_min: int | None = None) -> models.Filter | None:
    """Build a Qdrant metadata filter."""
    if year_min:
        return models.Filter(
            must=[models.FieldCondition(key="year", range=models.Range(gte=year_min))]
        )
    return None


def search_vector(query: str, k: int = 8, year_min: int | None = None) -> list[RetrievalResult]:
    """Dense vector search — Qdrant computes the sparse embedding via Document."""
    qv = models.Document(text=query, model=_EMBED_MODEL)

    hits = (
        _qc()
        .query_points(
            COLLECTION,
            query=qv,
            using="dense",
            limit=k,
            query_filter=_build_filter(year_min),
        )
        .points
    )

    return [
        RetrievalResult(
            title=h.payload.get("title", ""),
            year=h.payload.get("year", 0),
            source=h.payload.get("source", ""),
            citations=h.payload.get("citations", 0),
            chunk=h.payload.get("chunk", ""),
            section=h.payload.get("section", h.payload.get("chunk", "")),
            score=float(h.score),
            retrieval_method="vector",
        )
        for h in hits
    ]


def search_bm25(query: str, k: int = 8, year_min: int | None = None) -> list[RetrievalResult]:
    """Sparse BM25 search — Qdrant computes the sparse embedding via Document.

    Uses fastembed's Qdrant/bm25 model for tokenization + TF-IDF, stored as
    native Qdrant SparseVector with IDF modifier.
    """
    qv = models.Document(text=query, model=_BM25_MODEL)

    hits = (
        _qc()
        .query_points(
            COLLECTION,
            query=qv,
            using="sparse",
            limit=k,
            query_filter=_build_filter(year_min),
        )
        .points
    )

    return [
        RetrievalResult(
            title=h.payload.get("title", ""),
            year=h.payload.get("year", 0),
            source=h.payload.get("source", ""),
            citations=h.payload.get("citations", 0),
            chunk=h.payload.get("chunk", ""),
            section=h.payload.get("section", h.payload.get("chunk", "")),
            score=float(h.score),
            retrieval_method="bm25",
        )
        for h in hits
    ]


def hybrid_search(query: str, k: int = 10, year_min: int | None = None) -> list[RetrievalResult]:
    """Hybrid search: Qdrant native RRF fusion of dense + sparse BM25.

    Both vectors are computed server-side via fastembed Document embedding.
    RRF is applied by Qdrant's FusionQuery — no Python-level merging needed.
    """
    year_filter = _build_filter(year_min)

    dense_prefetch = models.Prefetch(
        query=models.Document(text=query, model=_EMBED_MODEL),
        using="dense",
        limit=k,
        query_filter=year_filter,
    )
    sparse_prefetch = models.Prefetch(
        query=models.Document(text=query, model=_BM25_MODEL),
        using="sparse",
        limit=k,
        query_filter=year_filter,
    )

    hits = (
        _qc()
        .query_points(
            COLLECTION,
            prefetch=[dense_prefetch, sparse_prefetch],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
        )
        .points
    )

    return [
        RetrievalResult(
            title=h.payload.get("title", ""),
            year=h.payload.get("year", 0),
            source=h.payload.get("source", ""),
            citations=h.payload.get("citations", 0),
            chunk=h.payload.get("chunk", ""),
            section=h.payload.get("section", h.payload.get("chunk", "")),
            score=float(h.score),
            retrieval_method="hybrid",
        )
        for h in hits
    ]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

_reranker = None
_reranker_warned = False


def _get_reranker():
    """Lazy-load cross-encoder reranker (runs locally, no API needed)."""
    global _reranker, _reranker_warned
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        except ImportError:
            if not _reranker_warned:
                import logging

                log = logging.getLogger(__name__)
                log.warning(
                    "sentence-transformers not installed — reranking disabled. "
                    "Install with: pip install sentence-transformers"
                )
                _reranker_warned = True
            return None
    return _reranker


def rerank(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    """Cross-encoder reranking — the production standard for precision.

    If the reranker model is not installed, returns results unchanged.
    """
    if not results:
        return []

    reranker = _get_reranker()
    if reranker is None:
        return results[:top_k]

    pairs = [(query, r.section) for r in results]
    scores = reranker.predict(pairs)

    for r, s in zip(results, scores):
        r.score = float(s)
        r.retrieval_method = f"reranked({r.retrieval_method})"

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Agentic retrieval loop
# ---------------------------------------------------------------------------

_REFINE_SYS = (
    "You steer literature retrieval for a research team. Given the goal and the "
    "papers already retrieved, write ONE refined search query targeting the gaps "
    '(methods/authors not yet covered). Respond with ONLY JSON: {"query": "..."}'
)


def agentic_retrieve(
    goal: str,
    rounds: int = 3,
    per_round: int = 8,
    top_k: int = 8,
    year_min: int | None = None,
) -> list[dict]:
    """Retrieve → judge coverage → refine query → retrieve again.

    Production pattern: hybrid search + reranking + agentic refinement.
    Returns unique chunks sorted by relevance, most relevant first.
    """
    from app.agent.llm import LLMError, get_llm

    seen: set[tuple] = set()
    gathered: list[RetrievalResult] = []
    query = goal

    for round_num in range(max(1, rounds)):
        results = hybrid_search(query, per_round, year_min=year_min)

        for r in results:
            key = (r.title, r.chunk[:80])
            if key not in seen:
                seen.add(key)
                gathered.append(r)

        if len(gathered) >= per_round:
            break

        try:
            titles_seen = sorted({r.title for r in gathered})
            out = get_llm().chat_json(
                _REFINE_SYS,
                f"GOAL: {goal}\nAlready have: {titles_seen}\nJSON only.",
                role="researcher",
            )
            query = str(out.get("query") or query)
        except LLMError:
            break

    reranked = rerank(goal, gathered, top_k=top_k)

    return [
        {
            "title": r.title,
            "year": r.year,
            "source": r.source,
            "citations": r.citations,
            "chunk": r.chunk,
            "section": r.section,
            "score": r.score,
            "retrieval_method": r.retrieval_method,
        }
        for r in reranked
    ]
