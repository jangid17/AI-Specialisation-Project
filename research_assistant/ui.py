"""Streamlit helpers shared by the pages."""

import numpy as np
import streamlit as st

from research_assistant import config, fulltext, llm, retrieval, store
from research_assistant.models import Paper

READER_PAGE = "app_pages/reader.py"
DISCOVER_PAGE = "app_pages/discover.py"


def _stamp() -> tuple:
    if not config.LIBRARY_DIR.exists():
        return ()
    return tuple(sorted((f.name, f.stat().st_mtime_ns) for f in config.LIBRARY_DIR.glob("*.json")))


@st.cache_data(show_spinner=False, max_entries=2)
def _load_library(stamp: tuple) -> list[dict]:
    return [p.to_dict() for p in store.load_all()]


def library() -> list[Paper]:
    """The saved papers, newest first. Re-read from disk only when a file changed."""
    return [Paper.from_dict(d) for d in _load_library(_stamp())]


def add_to_library(paper: Paper) -> str:
    """Fetch the full text, save and index the paper. Returns a note when only the abstract was saved."""
    note = ""
    try:
        fulltext.fetch_full_text(paper)
    except fulltext.FullTextUnavailable:
        note = "No HTML version of this paper exists yet, so only the abstract was saved."
    store.save(paper)
    retrieval.build_index(paper)
    return note


def open_in_reader(arxiv_id: str) -> None:
    st.session_state["paper"] = arxiv_id
    st.switch_page(READER_PAGE)


def notify_wait(seconds: float) -> None:
    """Passed to the LLM calls so a rate-limit pause is visible instead of looking like a hang."""
    st.toast(f"Groq's free-tier token limit was reached. Continuing in {seconds:.0f} s.",
             icon=":material/hourglass_top:", duration="long")


def llm_ready() -> bool:
    if llm.is_configured():
        return True
    st.warning("Add `GROQ_API_KEY=...` to the `.env` file to enable digests, answers and comparisons. "
               "A free key is available at console.groq.com/keys.",
               title="No Groq API key found", icon=":material/key:")
    return False


@st.cache_data(show_spinner=False, max_entries=64)
def interest_scores(texts: tuple[str, ...], interests: str) -> list[float]:
    """Cosine similarity between each abstract and the reader's interests statement."""
    return retrieval.similarity_to(interests, list(texts))


@st.cache_data(show_spinner="Embedding abstracts", max_entries=4)
def abstract_vectors(texts: tuple[str, ...]) -> np.ndarray:
    return retrieval.embed(list(texts))


def paper_label(paper: Paper) -> str:
    return f"{paper.title} — {paper.short_authors} ({paper.year})"


def as_list(value) -> list:
    """Model output that should be a list, whatever it actually is."""
    if isinstance(value, list):
        return value
    return [value] if value else []


def empty_library() -> None:
    st.info("Search arXiv and save a few papers to get started.", title="Your library is empty",
            icon=":material/library_books:")
    st.page_link(DISCOVER_PAGE, label="Find papers", icon=":material/travel_explore:")
