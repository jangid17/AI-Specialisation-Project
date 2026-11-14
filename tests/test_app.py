"""Every page renders without an exception, with an empty library and with papers in it.

Uses Streamlit's AppTest, so no browser, network or model is involved.
"""

import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from research_assistant import analysis, config, retrieval, store
from research_assistant.models import Paper, Section

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["discover", "library", "reader", "synthesize", "insights"]
TOPICS = ["dense passage retrieval for question answering", "protein structure prediction with transformers",
          "graph neural networks for molecules", "speech recognition with self-supervision"]


@pytest.fixture
def empty_library():
    for path in config.LIBRARY_DIR.glob("*.json") if config.LIBRARY_DIR.exists() else []:
        store.delete(path.stem)


@pytest.fixture
def library(empty_library, monkeypatch):
    # The pages only need vectors of the right shape; a real model download has no place in a unit test.
    import numpy as np

    def fake_embed(texts, query=False, model=None):
        rng = np.random.default_rng(len(texts))
        vectors = rng.normal(size=(len(texts), 16)).astype("float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    monkeypatch.setattr(retrieval, "embed", fake_embed)
    papers = []
    for i in range(8):
        paper = Paper(arxiv_id=f"2401.1000{i}", title=f"A study of {TOPICS[i % 4]} ({i})",
                      authors=["Ada Lovelace", f"Author {i}"], abstract=f"We study {TOPICS[i % 4]} and report results.",
                      published=f"2024-0{i + 1}-10", updated=f"2024-0{i + 1}-11", primary_category=["cs.CL", "cs.LG"][i % 2],
                      sections=[Section("1 Introduction", "Background and motivation. " * 20),
                                Section("2 Results", "## 2.1 Main\n\nThe method reaches $95\\%$ accuracy. " * 5)],
                      fulltext_source="https://arxiv.org/html/x", citation_count=[None, 3, 40000][i % 3],
                      citations_checked="2026-11-14",
                      digests={"Focused|Specialist": digest(verified=i % 2 == 0)} if i < 2 else {})
        store.save(paper)
        retrieval.build_index(paper)
        papers.append(paper)
    return papers


def digest(verified: bool) -> dict:
    # Deliberately sloppy model output: a bare string for a list, a null, a missing field.
    d = analysis.normalise_digest({
        "tldr": "Short summary.", "problem": "P", "method": "M", "contributions": ["c1"],
        "key_results": [{"claim": "Good", "evidence": "reaches 95%"}, "a result without a quote"],
        "limitations": "one limitation as a bare string", "prerequisites": None})
    d["key_results"][0]["verified"] = verified
    return {**d, "basis": "abstract only", "model": "test"}


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()
    if page != "discover":
        at.switch_page(str(ROOT / "app_pages" / f"{page}.py")).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_an_empty_library(empty_library, page):
    at = run_page(page)
    assert at.title[0].value.lower().endswith(page)


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_papers(library, page):
    run_page(page)


def test_library_counts_and_filter(library):
    at = run_page("library")
    assert [m.value for m in at.metric[:5]] == ["8", "8", "0", "0", "8"]
    at.text_input(key="lib_filter").set_value("protein").run()
    assert not at.exception and len(at.dataframe[0].value) == 2


def test_reader_shows_digest_with_quote_check_and_every_view(library):
    at = run_page("reader")
    at.selectbox(key="paper").set_value("2401.10000").run()
    assert not at.exception
    assert at.header[0].value == library[0].title
    rendered = " ".join(m.value for m in at.markdown)  # st.badge is rendered as markdown
    assert "Quote found in the paper" in rendered and "No supporting quote was given" in rendered
    for view in ["Ask", "Full text", "Notes"]:
        at.segmented_control(key="reader_view").set_value(view).run()
        assert not at.exception, view
    at.text_area(key="notes_2401.10000").set_value("check the ablation").run()
    assert store.load("2401.10000").notes == "check the ablation"


def test_reader_status_change_is_saved(library):
    at = run_page("reader")
    at.selectbox(key="paper").set_value("2401.10003").run()
    at.segmented_control(key="status_2401.10003").set_value("Read").run()
    assert not at.exception and store.load("2401.10003").status == "Read"


def test_insights_has_all_charts(library):
    at = run_page("insights")
    assert at.metric[0].value == "8" and at.metric[1].value == "9"  # 8 papers; Ada Lovelace + 8 others
    assert len(at.get("vega_lite_chart")) == 5  # timeline, categories, terms, topic map, citations
    assert len(at.get("echarts_chart")) == 1  # co-author network
