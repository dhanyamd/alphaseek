"""Literature retrieval — relevance-gated, typed, no junk.

Three scholarly sources, tried in reliability order:
  * OpenAlex (primary — keyless, ~100k req/day, citations + OA PDF links)
  * Semantic Scholar Graph API (secondary — adds AI tl;drs, throttles hard)
  * arXiv restricted to q-fin categories (tertiary)

Design rules:
  * An empty result list is an honest answer — we never pad with off-topic hits.
  * Full text via Jina Reader (HTML or PDF); None when no source has it.
  * All network access happens on the backend; the code sandbox stays isolated.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

HEADERS = {"User-Agent": "AlphaSeek-Research/0.2 (quant research agent)"}
TIMEOUT = 12


@dataclass
class Paper:
    title: str
    year: int | None
    source: str                     # "semanticscholar" | "arxiv"
    abstract: str = ""
    citations: int | None = None
    arxiv_id: str | None = None
    tldr: str = ""                  # Semantic Scholar's AI one-liner, when present
    pdf_url: str | None = None      # best open-access PDF, when present
    alt_pdf_urls: list[str] = field(default_factory=list)   # other OA copies
    full_text: str | None = field(default=None, repr=False)

    def label(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title


def search(query: str, limit: int = 5) -> list[Paper]:
    """Search scholarly literature. Returns relevant papers or an empty list."""
    for source in (_openalex, _semantic_scholar, _arxiv_qfin):
        papers = source(query, limit)
        if papers:
            return papers
    return []


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
    if paper.arxiv_id:                          # Jina parses PDFs — last resort
        candidates.append(f"https://arxiv.org/pdf/{paper.arxiv_id}")
    for url in candidates:
        text = _jina_read(url)
        if text and len(text) > 2_000:          # a real document, not an error page
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


# --------------------------------------------------------------------- sources
def _openalex(query: str, limit: int) -> list[Paper]:
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": limit,
                    "select": ("title,publication_year,cited_by_count,"
                               "abstract_inverted_index,best_oa_location,locations,ids")},
            headers=HEADERS, timeout=TIMEOUT,
        )
        r.raise_for_status()
        papers = []
        for w in r.json().get("results", [])[:limit]:
            oa = w.get("best_oa_location") or {}
            best_pdf = oa.get("pdf_url")
            alt_pdfs, arxiv_id = [], None
            for loc in (w.get("locations") or []):
                url = loc.get("pdf_url")
                if url and url != best_pdf:
                    alt_pdfs.append(url)
                m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?$",
                              loc.get("landing_page_url") or "")
                arxiv_id = arxiv_id or (m.group(1) if m else None)
            papers.append(Paper(
                title=w.get("title") or "",
                year=w.get("publication_year"),
                source="openalex",
                abstract=_from_inverted(w.get("abstract_inverted_index"))[:1_500],
                citations=w.get("cited_by_count"),
                arxiv_id=arxiv_id,
                pdf_url=best_pdf,
                alt_pdf_urls=alt_pdfs[:3],
            ))
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
        for attempt, wait in enumerate((2.5, 6.0, 0.0)):    # free tier throttles hard
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": limit,
                        "fields": "title,year,citationCount,abstract,externalIds,tldr,openAccessPdf"},
                headers=HEADERS, timeout=TIMEOUT,
            )
            if r.status_code == 429 and wait:
                time.sleep(wait)
                continue
            break
        r.raise_for_status()
        papers = []
        for p in r.json().get("data", [])[:limit]:
            ids = p.get("externalIds") or {}
            papers.append(Paper(
                title=p.get("title", ""),
                year=p.get("year"),
                source="semanticscholar",
                abstract=(p.get("abstract") or "")[:1_500],
                citations=p.get("citationCount"),
                arxiv_id=ids.get("ArXiv"),
                tldr=((p.get("tldr") or {}).get("text") or "")[:300],
                pdf_url=(p.get("openAccessPdf") or {}).get("url"),
            ))
        return papers
    except Exception:  # noqa: BLE001 — source down is not a fatal error
        return []


def _arxiv_qfin(query: str, limit: int) -> list[Paper]:
    """arXiv search restricted to quantitative-finance categories only."""
    try:
        r = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"cat:q-fin.* AND all:{query}",
                    "max_results": limit, "sortBy": "relevance"},
            headers=HEADERS, timeout=TIMEOUT,
        )
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers = []
        for e in ET.fromstring(r.text).findall("a:entry", ns)[:limit]:
            arxiv_id = (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id) or None
            published = e.findtext("a:published", "", ns) or ""
            papers.append(Paper(
                title=_squash(e.findtext("a:title", "", ns)),
                year=int(published[:4]) if published[:4].isdigit() else None,
                source="arxiv",
                abstract=_squash(e.findtext("a:summary", "", ns))[:1_500],
                arxiv_id=arxiv_id,
            ))
        return papers
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------- helpers
def _squash(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()



