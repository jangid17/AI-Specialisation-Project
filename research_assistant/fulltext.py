"""Read a paper's full text from its HTML rendering with BeautifulSoup.

arXiv serves LaTeXML-generated HTML for most papers at arxiv.org/html/<id>; ar5iv.labs.arxiv.org
carries the same format for older papers. Both use the `ltx_*` class names parsed here.
"""

import re

import requests
from bs4 import BeautifulSoup, Tag

from research_assistant import config
from research_assistant.models import Paper, Section

SOURCES = ("https://arxiv.org/html/{id}", "https://ar5iv.labs.arxiv.org/html/{id}")
MIN_BODY_CHARS = 1500  # below this the page is an error stub or a failed conversion, not a paper
_HEADINGS = ["h2", "h3", "h4", "h5", "h6"]


class FullTextUnavailable(RuntimeError):
    pass


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _has_class(tag: Tag, *names: str) -> bool:
    classes = tag.get("class") or []
    return any(n in classes for n in names)


def _prepare(article: Tag) -> None:
    """Drop page furniture and make formulas readable as text."""
    for math in article.find_all("math"):
        tex = math.get("alttext") or math.get_text(" ", strip=True)
        math.replace_with(f" ${tex}$ ")
    for tag in article.select(".ltx_note, .ltx_authors, .ltx_dates, .ltx_role_footnote, nav, script, style, button, svg"):
        tag.decompose()
    for table in article.find_all("table"):
        if table.decomposed:  # nested inside a table that was already removed
            continue
        if not _has_class(table, "ltx_equation", "ltx_equationgroup", "ltx_eqn_table"):
            table.decompose()  # data tables turn into unreadable runs of numbers; their captions are kept


def _section_text(section: Tag) -> str:
    """Body text of one top-level section, with subsection titles and captions kept in order."""
    blocks = []
    for el in section.find_all(_HEADINGS[1:] + ["p", "figcaption", "table"]):
        if el.name == "p":
            if not _has_class(el, "ltx_p") or el.find_parent("figcaption"):
                continue
            text = _clean(el.get_text(" "))
        elif el.name == "figcaption":
            text = _clean(el.get_text(" "))
            text = f"[{text}]" if text else ""
        elif el.name == "table":  # a display equation; grouped equations nest, so take the outer one only
            text = "" if el.find_parent("table") else _clean(el.get_text(" "))
        else:
            text = _clean(el.get_text(" "))
            text = f"## {text}" if text else ""
        if text:
            blocks.append(text)
    return "\n\n".join(blocks)


def parse_html(html: str) -> tuple[list[Section], int | None]:
    """Extract sections and the number of bibliography entries from LaTeXML HTML."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_="ltx_document") or soup.find("article")
    if article is None:
        raise FullTextUnavailable("The page has no paper body.")

    bib = article.find(class_="ltx_bibliography")
    reference_count = len(bib.find_all(class_="ltx_bibitem")) if bib else None
    if bib:
        bib.decompose()
    _prepare(article)

    sections = []
    for sec in article.find_all("section"):
        if not _has_class(sec, "ltx_section", "ltx_appendix") or sec.find_parent("section"):
            continue
        heading = sec.find(_HEADINGS)
        title = _clean(heading.get_text(" ")) if heading else "Untitled section"
        if heading:
            heading.decompose()
        text = _section_text(sec)
        if text:
            sections.append(Section(title=title, text=text))

    if not sections:  # short papers and notes are sometimes one unsectioned body
        body = "\n\n".join(_clean(p.get_text(" ")) for p in article.find_all("p", class_="ltx_p")
                           if not p.find_parent(class_="ltx_abstract"))
        if body:
            sections = [Section(title="Body", text=body)]

    if sum(len(s.text) for s in sections) < MIN_BODY_CHARS:
        raise FullTextUnavailable("The HTML version has no usable text.")
    return sections, reference_count


def fetch_full_text(paper: Paper) -> Paper:
    """Fill paper.sections from the first HTML source that has the paper. Raises FullTextUnavailable."""
    errors = []
    for template in SOURCES:
        url = template.format(id=paper.arxiv_id)
        try:
            resp = requests.get(url, timeout=config.HTTP_TIMEOUT, headers={"User-Agent": config.USER_AGENT})
        except requests.RequestException as e:
            errors.append(f"{url}: {e}")
            continue
        if resp.status_code != 200:
            errors.append(f"{url}: HTTP {resp.status_code}")
            continue
        resp.encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else "utf-8"
        try:
            paper.sections, paper.reference_count = parse_html(resp.text)
        except FullTextUnavailable as e:
            errors.append(f"{url}: {e}")
            continue
        paper.fulltext_source = resp.url
        return paper
    raise FullTextUnavailable("No HTML version of this paper could be read (" + "; ".join(errors) + ").")
