"""Citation counts from the Semantic Scholar Graph API (free, no key, shared rate limit)."""

from datetime import date

import requests

from research_assistant import config
from research_assistant.models import Paper

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


class CitationsUnavailable(RuntimeError):
    pass


def update_citation_counts(papers: list[Paper]) -> int:
    """Set citation_count on each paper Semantic Scholar knows. Returns how many were found.

    Papers it has not indexed yet (usually the newest ones) keep citation_count = None.
    """
    if not papers:
        return 0
    try:
        resp = requests.post(BATCH_URL, params={"fields": "citationCount"},
                             json={"ids": [f"ARXIV:{p.arxiv_id}" for p in papers[:500]]},
                             timeout=config.HTTP_TIMEOUT, headers={"User-Agent": config.USER_AGENT})
    except requests.RequestException as e:
        raise CitationsUnavailable(f"Could not reach Semantic Scholar: {e}") from e
    if resp.status_code == 429:
        raise CitationsUnavailable("Semantic Scholar is rate-limiting anonymous requests right now. Try again later.")
    if resp.status_code != 200:
        raise CitationsUnavailable(f"Semantic Scholar returned HTTP {resp.status_code}.")
    found = 0
    today = date.today().isoformat()
    for paper, record in zip(papers, resp.json()):
        paper.citations_checked = today
        if record and record.get("citationCount") is not None:
            paper.citation_count = int(record["citationCount"])
            found += 1
    return found
