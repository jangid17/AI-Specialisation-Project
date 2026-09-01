"""Search arXiv through its public Atom API (https://info.arxiv.org/help/api/)."""

import re
import threading
import time
import xml.etree.ElementTree as ET

import requests

from research_assistant import config
from research_assistant.models import Paper

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}
SEARCH_FIELDS = {"All fields": "all", "Title": "ti", "Abstract": "abs", "Author": "au"}
SORT_OPTIONS = {"Relevance": "relevance", "Newest first": "submittedDate", "Recently updated": "lastUpdatedDate"}

# A few categories a reader is likely to filter by; any arXiv category code also works in a query.
CATEGORIES = {
    "cs.AI": "Artificial intelligence",
    "cs.CL": "Computation and language",
    "cs.CV": "Computer vision",
    "cs.LG": "Machine learning",
    "cs.IR": "Information retrieval",
    "cs.RO": "Robotics",
    "cs.CR": "Cryptography and security",
    "cs.SE": "Software engineering",
    "cs.HC": "Human-computer interaction",
    "stat.ML": "Machine learning (statistics)",
    "math.OC": "Optimization and control",
    "eess.SP": "Signal processing",
    "physics.comp-ph": "Computational physics",
    "quant-ph": "Quantum physics",
    "q-bio.QM": "Quantitative methods (biology)",
    "econ.EM": "Econometrics",
}

_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)
_lock = threading.Lock()
_last_call = 0.0


class ArxivError(RuntimeError):
    pass


def parse_arxiv_id(text: str) -> str | None:
    """Pull an arXiv identifier out of an ID, an abs/pdf/html URL, or 'arXiv:...' text."""
    m = _ID_RE.search(text.strip())
    return m.group(1) if m else None


def build_query(terms: str, field: str = "all", category: str | None = None) -> str:
    """Turn free text into an arXiv search_query. Quoted phrases are kept; other words are ANDed."""
    parts = []
    for phrase, word in re.findall(r'"([^"]+)"|(\S+)', terms):
        token = f'"{phrase}"' if phrase else word
        if token.upper() in ("AND", "OR", "ANDNOT"):
            continue
        parts.append(f"{field}:{token}")
    query = " AND ".join(parts)
    if category:
        query = f"({query}) AND cat:{category}" if query else f"cat:{category}"
    return query


def _text(node, path: str) -> str:
    found = node.find(path, NS)
    return re.sub(r"\s+", " ", found.text).strip() if found is not None and found.text else ""


def parse_feed(xml_text: str) -> tuple[list[Paper], int]:
    """Parse an Atom feed into papers and the total number of matches."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ArxivError(f"arXiv returned a response that is not valid XML: {e}") from e
    total = int(_text(root, "os:totalResults") or 0)
    papers = []
    for entry in root.findall("atom:entry", NS):
        raw_id = _text(entry, "atom:id")
        m = _ID_RE.search(raw_id.rsplit("/abs/", 1)[-1])
        if not m:  # arXiv reports query errors as an entry without a paper id
            raise ArxivError(_text(entry, "atom:summary") or "arXiv rejected the query.")
        primary = entry.find("arxiv:primary_category", NS)
        categories = [c.get("term") for c in entry.findall("atom:category", NS)]
        papers.append(Paper(
            arxiv_id=m.group(1),
            version=m.group(2) or "",
            title=_text(entry, "atom:title"),
            authors=[_text(a, "atom:name") for a in entry.findall("atom:author", NS)],
            abstract=_text(entry, "atom:summary"),
            published=_text(entry, "atom:published")[:10],
            updated=_text(entry, "atom:updated")[:10],
            primary_category=primary.get("term") if primary is not None else (categories[0] if categories else ""),
            categories=categories,
            comment=_text(entry, "arxiv:comment"),
            journal_ref=_text(entry, "arxiv:journal_ref"),
            doi=_text(entry, "arxiv:doi"),
        ))
    return papers, total


def _get(params: dict) -> str:
    """One API call, spaced at least ARXIV_MIN_INTERVAL after the previous one."""
    global _last_call
    with _lock:
        wait = config.ARXIV_MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(config.ARXIV_API, params=params, timeout=config.HTTP_TIMEOUT,
                                headers={"User-Agent": config.USER_AGENT})
        except requests.RequestException as e:
            raise ArxivError(f"Could not reach arXiv: {e}") from e
        finally:
            _last_call = time.monotonic()
    if resp.status_code != 200:
        raise ArxivError(f"arXiv returned HTTP {resp.status_code}.")
    return resp.text


def search(terms: str, field: str = "all", category: str | None = None, sort: str = "relevance",
           max_results: int = 20, start: int = 0) -> tuple[list[Paper], int]:
    """Search arXiv. Returns the papers on this page and the total number of matches."""
    query = build_query(terms, field, category)
    if not query:
        raise ArxivError("Enter a search term or pick a category.")
    return parse_feed(_get({"search_query": query, "start": start, "max_results": max_results,
                            "sortBy": sort, "sortOrder": "descending"}))


def fetch_by_ids(arxiv_ids: list[str]) -> list[Paper]:
    """Look up specific papers by identifier."""
    if not arxiv_ids:
        return []
    papers, _ = parse_feed(_get({"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)}))
    return papers
