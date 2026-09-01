"""BibTeX and Markdown exports."""

import re
import unicodedata

from research_assistant.models import Paper

_TEX_SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"}


def _tex(text: str) -> str:
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in text)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def bibtex_key(paper: Paper) -> str:
    surname = re.sub(r"[^a-z]", "", _ascii(paper.authors[0].split()[-1]).lower()) if paper.authors else "anon"
    words = re.findall(r"[a-z]{4,}", _ascii(paper.title).lower())
    return f"{surname}{paper.year}{words[0] if words else ''}"


def bibtex_entry(paper: Paper) -> str:
    fields = {
        "title": "{" + _tex(paper.title) + "}",
        "author": _tex(" and ".join(paper.authors)),
        "year": str(paper.year),
        "eprint": paper.arxiv_id,
        "archivePrefix": "arXiv",
        "primaryClass": paper.primary_category,
        "url": paper.abs_url,
    }
    if paper.journal_ref:
        fields["note"] = _tex(paper.journal_ref)
    if paper.doi:
        fields["doi"] = paper.doi
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
    return f"@misc{{{bibtex_key(paper)},\n{body}\n}}"


def bibtex(papers: list[Paper]) -> str:
    return "\n\n".join(bibtex_entry(p) for p in papers) + "\n"


def digest_markdown(paper: Paper, digest: dict) -> str:
    """One paper's digest and the reader's notes as a Markdown document."""
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {i}" for i in items) or "- (none)"

    results = []
    for r in digest["key_results"]:
        mark = "quote found in the paper" if r.get("verified") else "quote not found verbatim"
        results.append(f"{r['claim']}\n  > \"{r['evidence']}\" ({mark})")
    lines = [
        f"# {paper.title}",
        f"{', '.join(paper.authors)} · {paper.published} · [arXiv:{paper.arxiv_id}]({paper.abs_url})",
        f"*Digest written by {digest.get('model', 'an LLM')} from: {digest.get('basis', 'unknown')}.*",
        "## In brief", digest["tldr"],
        "## Problem", digest["problem"],
        "## Contributions", bullets(digest["contributions"]),
        "## Method", digest["method"],
        "## Key results", bullets(results),
        "## Limitations", bullets(digest["limitations"]),
        "## Background needed", bullets(digest["prerequisites"]),
        "## Questions for discussion", bullets(digest["discussion_questions"]),
    ]
    if paper.notes.strip():
        lines += ["## My notes", paper.notes.strip()]
    return "\n\n".join(lines) + "\n"
