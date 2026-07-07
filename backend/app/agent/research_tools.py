"""Literature retrieval — parallel, relevance-gated, typed, no junk.

Four sources searched in PARALLEL for speed:
  * OpenAlex (keyless, ~100k req/day, citations + OA PDF links)
  * Semantic Scholar Graph API (AI tl;drs, throttles hard)
  * arXiv restricted to q-fin categories
  * Exa (neural web search, catches SSRN/blogs/repos arXiv misses)

Design rules:
  * All sources are searched concurrently via ThreadPoolExecutor.
  * Results are merged, deduplicated by title similarity, and ranked by citation count.
  * An empty result list is an honest answer — we never pad with off-topic hits.
  * Full text via Jina Reader (HTML or PDF); None when no source has it.
  * All network access happens on the backend; the code sandbox stays isolated.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

HEADERS = {"User-Agent": "AlphaSeek-Research/0.2 (quant research agent)"}
TIMEOUT = 12


@dataclass
class Paper:
    title: str
    year: int | None
    source: str  # "openalex" | "semanticscholar" | "arxiv"
    abstract: str = ""
    citations: int | None = None
    arxiv_id: str | None = None
    tldr: str = ""  # Semantic Scholar's AI one-liner, when present
    pdf_url: str | None = None  # best open-access PDF, when present
    alt_pdf_urls: list[str] = field(default_factory=list)  # other OA copies
    full_text: str | None = field(default=None, repr=False)

    def label(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title


# ---------------------------------------------------------------------------
# Parallel search — all sources at once
# ---------------------------------------------------------------------------


def search(query: str, limit: int = 5) -> list[Paper]:
    """Search all scholarly sources in parallel, merge and deduplicate.

    Returns the top `limit` papers ranked by citation count (higher = more
    authoritative). Deduplicates by normalized title similarity.
    """
    results_by_source: dict[str, list[Paper]] = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_openalex, query, limit): "openalex",
            pool.submit(_semantic_scholar, query, limit): "semanticscholar",
            pool.submit(_arxiv_qfin, query, limit): "arxiv",
            pool.submit(_exa, query, limit): "exa",
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                papers = future.result()
                if papers:
                    results_by_source[source] = papers
            except Exception:  # noqa: BLE001 — source down is not fatal
                pass

    # Merge all results
    all_papers = []
    for papers in results_by_source.values():
        all_papers.extend(papers)

    if not all_papers:
        return []

    # Deduplicate by normalized title similarity
    merged = _deduplicate_papers(all_papers)

    # Rank: citations (desc) → year (desc) → has abstract
    merged.sort(
        key=lambda p: (
            p.citations or 0,
            p.year or 0,
            1 if p.abstract else 0,
        ),
        reverse=True,
    )

    return merged[:limit]


def _normalize_title(title: str) -> str:
    """Normalize title for deduplication: lowercase, remove punctuation, collapse whitespace."""
    t = re.sub(r"[^\w\s]", "", title.lower())
    return re.sub(r"\s+", " ", t).strip()


def _deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Remove duplicate papers by title similarity, keeping the richer version."""
    seen: dict[str, Paper] = {}
    for paper in papers:
        key = _normalize_title(paper.title)
        if not key:
            continue
        if key in seen:
            existing = seen[key]
            # Keep the version with more data
            if (paper.citations or 0) > (existing.citations or 0):
                seen[key] = paper
            elif paper.abstract and not existing.abstract:
                seen[key] = paper
            elif paper.pdf_url and not existing.pdf_url:
                seen[key] = paper
        else:
            seen[key] = paper
    return list(seen.values())


# ---------------------------------------------------------------------------
# Full text fetching
# ---------------------------------------------------------------------------


def fetch_full_text(paper: Paper, max_chars: int = 120_000) -> str | None:
    """Fetch full text via Jina Reader (free, keyless, handles HTML and PDF).

    Candidates in order: arXiv HTML render, then the open-access PDF.
    Returns None honestly when no source yields a real document.
    """
    candidates: list[str] = []
    if paper.arxiv_id:
        candidates.append(f"https://arxiv.org/html/{paper.arxiv_id}")
    if paper.pdf_url:
        candidates.append(paper.pdf_url)
    candidates.extend(paper.alt_pdf_urls)
    if paper.arxiv_id:  # Jina parses PDFs — last resort
        candidates.append(f"https://arxiv.org/pdf/{paper.arxiv_id}")
    for url in candidates:
        text = _jina_read(url)
        if text and len(text) > 2_000:  # a real document, not an error page
            return text[:max_chars]
    return None


def _jina_read(url: str) -> str | None:
    """URL -> clean markdown via r.jina.ai (parses PDFs with PDF.js)."""
    try:
        r = requests.get(f"https://r.jina.ai/{url}", headers=HEADERS, timeout=45)
        if r.status_code != 200:
            return None
        return _squash(r.text)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Source implementations
# ---------------------------------------------------------------------------


def _openalex(query: str, limit: int) -> list[Paper]:
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": limit,
                "select": (
                    "title,publication_year,cited_by_count,"
                    "abstract_inverted_index,best_oa_location,locations,ids"
                ),
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        papers = []
        for w in r.json().get("results", [])[:limit]:
            oa = w.get("best_oa_location") or {}
            best_pdf = oa.get("pdf_url")
            alt_pdfs, arxiv_id = [], None
            for loc in w.get("locations") or []:
                url = loc.get("pdf_url")
                if url and url != best_pdf:
                    alt_pdfs.append(url)
                m = re.search(
                    r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?$",
                    loc.get("landing_page_url") or "",
                )
                arxiv_id = arxiv_id or (m.group(1) if m else None)
            papers.append(
                Paper(
                    title=w.get("title") or "",
                    year=w.get("publication_year"),
                    source="openalex",
                    abstract=_from_inverted(w.get("abstract_inverted_index"))[:1_500],
                    citations=w.get("cited_by_count"),
                    arxiv_id=arxiv_id,
                    pdf_url=best_pdf,
                    alt_pdf_urls=alt_pdfs[:3],
                )
            )
        return papers
    except Exception:  # noqa: BLE001 — source down is not a fatal error
        return []


def _from_inverted(inv: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]} — rebuild the text."""
    if not inv:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inv.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


def _semantic_scholar(query: str, limit: int) -> list[Paper]:
    try:
        retries = [2.5, 6.0, 0.0]
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,year,citationCount,abstract,externalIds,tldr,openAccessPdf",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        for wait in retries:
            if r.status_code == 429 and wait:
                time.sleep(wait)
                r = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": limit,
                        "fields": "title,year,citationCount,abstract,externalIds,tldr,openAccessPdf",
                    },
                    headers=HEADERS,
                    timeout=TIMEOUT,
                )
            else:
                break
        r.raise_for_status()
        papers = []
        for p in r.json().get("data", [])[:limit]:
            ids = p.get("externalIds") or {}
            papers.append(
                Paper(
                    title=p.get("title", ""),
                    year=p.get("year"),
                    source="semanticscholar",
                    abstract=(p.get("abstract") or "")[:1_500],
                    citations=p.get("citationCount"),
                    arxiv_id=ids.get("ArXiv"),
                    tldr=((p.get("tldr") or {}).get("text") or "")[:300],
                    pdf_url=(p.get("openAccessPdf") or {}).get("url"),
                )
            )
        return papers
    except Exception:  # noqa: BLE001 — source down is not a fatal error
        return []


def _arxiv_qfin(query: str, limit: int) -> list[Paper]:
    """arXiv search restricted to quantitative-finance categories only.

    Three-phase precision strategy:
      1. Title-only search with multi-term AND (strictest, most precise).
      2. Abstract-only search if title search returns < limit (moderate).
      3. Broad all-field search as last resort (least precise).

    Results are deduplicated by arxiv_id across phases.
    """
    try:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        seen_ids: set[str] = set()
        all_papers: list[Paper] = []

        # Phase 1: Title-only multi-term AND — most precise.
        terms = [t.strip() for t in re.sub(r"[^\w\s]", " ", query).split() if len(t.strip()) > 2]
        if terms:
            title_query = " AND ".join(f"ti:{t}" for t in terms[:6])
            title_query = f"cat:q-fin.* AND ({title_query})"
            papers = _arxiv_fetch(title_query, limit, ns)
            for p in papers:
                pid = p.arxiv_id or p.title
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)

        # Phase 2: Abstract search if phase 1 was too narrow.
        if len(all_papers) < limit // 2 and terms:
            abs_query = " AND ".join(f"abs:{t}" for t in terms[:6])
            abs_query = f"cat:q-fin.* AND ({abs_query})"
            papers = _arxiv_fetch(abs_query, limit, ns)
            for p in papers:
                pid = p.arxiv_id or p.title
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)

        # Phase 3: Broad all-field search as fallback.
        if len(all_papers) < 1:
            fallback = f"cat:q-fin.* AND all:{query}"
            papers = _arxiv_fetch(fallback, limit, ns)
            for p in papers:
                pid = p.arxiv_id or p.title
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)

        return all_papers[:limit]
    except Exception:  # noqa: BLE001
        return []


def _arxiv_fetch(search_query: str, limit: int, ns: dict) -> list[Paper]:
    """Execute a single arXiv API query and parse results."""
    r = requests.get(
        "http://export.arxiv.org/api/query",
        params={
            "search_query": search_query,
            "max_results": limit,
            "sortBy": "relevance",
        },
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    papers = []
    for e in ET.fromstring(r.text).findall("a:entry", ns)[:limit]:
        arxiv_id = (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id) or None
        published = e.findtext("a:published", "", ns) or ""
        papers.append(
            Paper(
                title=_squash(e.findtext("a:title", "", ns)),
                year=int(published[:4]) if published[:4].isdigit() else None,
                source="arxiv",
                abstract=_squash(e.findtext("a:summary", "", ns))[:1500],
                arxiv_id=arxiv_id,
            )
        )
    return papers


# ---------------------------------------------------------------------------
# Exa — neural web search (catches SSRN, blogs, GitHub repos)
# ---------------------------------------------------------------------------


def _exa(query: str, limit: int) -> list[Paper]:
    """Search via Exa neural web search. Free tier: 1000 req/mo.

    Exa indexes academic content AND general web — catches papers on SSRN,
    quant blogs, GitHub repos, and forum discussions that arXiv misses.
    Requires EXA_API_KEY in backend/.env.
    """
    from app.settings import settings

    api_key = getattr(settings, "exa_api_key", "")
    if not api_key:
        return []
    try:
        from exa_py import Exa

        exa = Exa(api_key=api_key)
        results = exa.search(
            query,
            type="auto",
            num_results=limit,
            category="research paper",
            contents={"text": {"maxCharacters": 2000}},
        )
        papers = []
        for r in results.results:
            # Extract year from published date
            year = None
            if r.published_date:
                try:
                    year = int(r.published_date[:4])
                except (ValueError, IndexError):
                    pass
            # Try to extract arxiv ID from URL
            arxiv_id = None
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?$", r.url or "")
            if m:
                arxiv_id = m.group(1)
            papers.append(
                Paper(
                    title=r.title or "",
                    year=year,
                    source="exa",
                    abstract=(r.text or "")[:1_500],
                    arxiv_id=arxiv_id,
                    pdf_url=r.url if r.url and r.url.endswith(".pdf") else None,
                )
            )
        return papers
    except Exception:  # noqa: BLE001 — Exa down is not fatal
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _squash(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
