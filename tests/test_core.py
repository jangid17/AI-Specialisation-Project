"""Unit tests for everything that does not need the network or a model."""

import networkx as nx
import numpy as np
import pytest

from research_assistant import analysis, arxiv_client as ax, chunking, export, fulltext, llm, metrics, retrieval, store
from research_assistant.models import Paper, Section

FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>6318</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2309.15217v2</id>
    <title>Ragas: Automated Evaluation
      of Retrieval Augmented Generation</title>
    <updated>2025-04-28T10:00:00Z</updated>
    <summary>We introduce Ragas.</summary>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.IR" scheme="http://arxiv.org/schemas/atom"/>
    <published>2023-09-26T19:23:54Z</published>
    <arxiv:primary_category term="cs.CL"/>
    <arxiv:comment>Reference implementation available</arxiv:comment>
    <author><name>Shahul Es</name></author>
    <author><name>Jithin James</name></author>
  </entry>
</feed>"""

ERROR_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_abc</id>
  <title>Error</title><summary>incorrect id format for abc</summary></entry>
</feed>"""

BODY = "Retrieval improves factual accuracy on open-domain questions. " * 30
HTML = f"""<html><body><article class="ltx_document">
<h1 class="ltx_title">A paper</h1>
<div class="ltx_authors">Someone</div>
<div class="ltx_abstract"><p class="ltx_p">The abstract.</p></div>
<section class="ltx_section" id="S1"><h2 class="ltx_title">1 Introduction</h2>
  <div class="ltx_para"><p class="ltx_p">We study <math alttext="x^2"><mi>x</mi></math> here.<span class="ltx_note">a footnote</span> {BODY}</p></div>
  <figure><table class="ltx_tabular"><tr><td>12.5</td></tr></table><figcaption>Table 1: Scores.</figcaption></figure>
  <section class="ltx_subsection" id="S1.1"><h3 class="ltx_title">1.1 Setup</h3>
    <div class="ltx_para"><p class="ltx_p">Details of the setup. {BODY}</p></div></section>
</section>
<section class="ltx_section" id="S2"><h2 class="ltx_title">2 Conclusion</h2>
  <div class="ltx_para"><p class="ltx_p">It works.</p></div></section>
<section class="ltx_bibliography"><ul><li class="ltx_bibitem">A</li><li class="ltx_bibitem">B</li></ul></section>
</article></body></html>"""


def paper(i: int = 1, **kw) -> Paper:
    defaults = dict(arxiv_id=f"2401.0000{i}", title=f"Dense retrieval study {i}", authors=["Ada Lovelace", f"Author {i}"],
                    abstract="We evaluate dense retrieval for question answering with neural encoders.",
                    published=f"2024-0{i}-15", updated=f"2024-0{i}-20", primary_category="cs.IR")
    return Paper(**{**defaults, **kw})


# arXiv -----------------------------------------------------------------------------------------

def test_feed_is_parsed_into_papers():
    papers, total = ax.parse_feed(FEED)
    p = papers[0]
    assert total == 6318 and len(papers) == 1
    assert (p.arxiv_id, p.version) == ("2309.15217", "v2")
    assert p.title == "Ragas: Automated Evaluation of Retrieval Augmented Generation"  # whitespace collapsed
    assert p.authors == ["Shahul Es", "Jithin James"]
    assert (p.published, p.updated, p.primary_category) == ("2023-09-26", "2025-04-28", "cs.CL")
    assert p.categories == ["cs.CL", "cs.IR"] and p.year == 2023
    assert p.abs_url == "https://arxiv.org/abs/2309.15217"


def test_api_error_entry_raises():
    with pytest.raises(ax.ArxivError, match="incorrect id format"):
        ax.parse_feed(ERROR_FEED)
    with pytest.raises(ax.ArxivError):
        ax.parse_feed("<html>not a feed")


@pytest.mark.parametrize("text,expected", [
    ("1706.03762", "1706.03762"),
    ("https://arxiv.org/abs/1706.03762v7", "1706.03762"),
    ("https://arxiv.org/pdf/2405.07437", "2405.07437"),
    ("arXiv:hep-th/9901001", "hep-th/9901001"),
    ("not an id", None),
])
def test_parse_arxiv_id(text, expected):
    assert ax.parse_arxiv_id(text) == expected


def test_build_query():
    assert ax.build_query('"dense retrieval" evaluation') == 'all:"dense retrieval" AND all:evaluation'
    assert ax.build_query("bert", "ti", "cs.CL") == "(ti:bert) AND cat:cs.CL"
    assert ax.build_query("", category="cs.IR") == "cat:cs.IR"
    assert ax.build_query("   ") == ""


# Full text -------------------------------------------------------------------------------------

def test_html_is_split_into_sections():
    sections, references = fulltext.parse_html(HTML)
    assert [s.title for s in sections] == ["1 Introduction", "2 Conclusion"]
    assert references == 2
    intro = sections[0].text
    assert "$x^2$" in intro            # formula kept as its TeX source
    assert "a footnote" not in intro   # footnotes dropped
    assert "12.5" not in intro         # table cells dropped ...
    assert "[Table 1: Scores.]" in intro  # ... caption kept
    assert "## 1.1 Setup" in intro and "Details of the setup." in intro
    assert "The abstract." not in intro and "Someone" not in intro


def test_page_without_a_paper_is_rejected():
    with pytest.raises(fulltext.FullTextUnavailable):
        fulltext.parse_html("<html><body><p>No HTML for this paper.</p></body></html>")
    with pytest.raises(fulltext.FullTextUnavailable):  # an article that is only a stub
        fulltext.parse_html('<article class="ltx_document"><section class="ltx_section"><h2>1</h2>'
                            '<p class="ltx_p">Conversion failed.</p></section></article>')


# Store -----------------------------------------------------------------------------------------

def test_library_round_trip():
    p = paper(sections=[Section("1 Intro", "Text")], tags=["rag"], notes="read by Friday")
    store.save(p)
    loaded = store.load(p.arxiv_id)
    assert loaded.to_dict() == p.to_dict() and loaded.added_at
    assert loaded.has_full_text and loaded.full_text() == "1 Intro\nText"
    assert store.contains(p.arxiv_id) and p.arxiv_id in [q.arxiv_id for q in store.load_all()]
    store.delete(p.arxiv_id)
    assert not store.contains(p.arxiv_id) and store.load(p.arxiv_id) is None


def test_old_style_ids_get_a_safe_filename():
    assert "/" not in store.paper_path("hep-th/9901001").name


# Retrieval -------------------------------------------------------------------------------------

def test_split_text_respects_size_and_keeps_all_words():
    text = "\n\n".join(" ".join(f"Sentence {p}-{i} has several words." for i in range(40)) for p in range(3))
    chunks = chunking.split_text(text, size=400, overlap=60)
    assert len(chunks) > 3 and all(len(c) <= 400 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"Sentence {p}-{i} " in joined for p in range(3) for i in range(40))
    assert chunking.split_text("x" * 1000, size=300) == ["x" * 300, "x" * 300, "x" * 300, "x" * 100]


def unit_vectors(texts, query=False, model=None):
    """A stand-in embedding: one dimension per keyword, so similarity is predictable without a model."""
    words = ["encoder", "retrieval", "translation", "protein", "graph", "speech"]
    v = np.array([[t.lower().count(w) for w in words] + [0.01] for t in texts], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.parametrize("strategy", [s for s in chunking.STRATEGIES])
def test_every_chunking_strategy_keeps_sections_and_text(strategy):
    text = "\n\n".join(" ".join(f"Sentence {p}-{i} about the encoder." for i in range(30)) for p in range(3))
    p = paper(sections=[Section("2 Method", text), Section("3 Results", "Short result. " * 5)])
    pieces = chunking.chunk_sections(p, strategy, unit_vectors)
    assert pieces[0] == ("Abstract", p.abstract)
    assert {s for s, _ in pieces[1:]} == {"2 Method", "3 Results"}
    body = " ".join(t for s, t in pieces if s == "2 Method")
    assert all(f"Sentence {q}-{i} " in body for q in range(3) for i in range(30))


def test_fixed_windows_overlap_and_cover_the_text():
    windows = chunking.fixed_windows("abcdefghij" * 50, size=100, overlap=20)
    assert all(len(w) <= 100 for w in windows) and windows[1].startswith(windows[0][80:])
    assert "".join(w[:80] for w in windows[:-1]) + windows[-1] == "abcdefghij" * 50


def test_bm25_prefers_rare_matching_terms():
    bm25 = retrieval.BM25(["the encoder is trained on translation", "protein folding with graph networks",
                           "the encoder and the decoder", "speech recognition"])
    scores = bm25.scores("protein graph")
    assert scores.argmax() == 1 and scores[3] == 0


def test_rrf_rewards_agreement_between_rankings():
    fused = retrieval.rrf([np.array([3, 1, 2]), np.array([1, 3, 0])])
    assert sorted(fused, key=fused.get, reverse=True)[:2] in ([1, 3], [3, 1]) and fused[1] > fused[2]


def test_library_index_filters_by_paper_and_fuses(monkeypatch):
    monkeypatch.setattr(retrieval, "embed", unit_vectors)
    monkeypatch.setattr(retrieval, "rerank_scores", lambda q, texts, model=None: np.array([t.count("protein") for t in texts], float))
    passages = [retrieval.Passage("a", "Method", "an encoder for translation"),
                retrieval.Passage("a", "Results", "protein graph results"),
                retrieval.Passage("b", "Method", "protein structure with a graph encoder"),
                retrieval.Passage("b", "Data", "speech corpus")]
    index = retrieval.LibraryIndex(passages, unit_vectors([p.indexed for p in passages]))
    for method in retrieval.METHODS:
        found = index.search("protein graph", ["a"], top_k=2, method=method)
        assert found and {p.arxiv_id for p in found} == {"a"} and found[0].section == "Results", method
    assert index.search("protein", ["b", "a"], top_k=4, method="hybrid+rerank")[0].text.count("protein") == 1
    assert index.search("protein", ["missing"]) == []


def test_chunks_carry_their_section():
    p = paper(sections=[Section("2 Method", "We train an encoder. " * 10)])
    passages = retrieval.chunk_paper(p, "sentence-1100")
    assert passages[0].section == "Abstract" and passages[0].text == p.abstract
    assert {x.section for x in passages[1:]} == {"2 Method"}


# Analysis --------------------------------------------------------------------------------------

def test_quote_check_ignores_case_spacing_and_punctuation():
    p = paper(abstract="x", sections=[Section("Results", "Our model achieves 28.4 BLEU on the WMT 2014 "
                                                         "English-to-German task, improving over the best results.")])
    assert analysis.quote_in_paper("our model achieves 28.4 BLEU  on the WMT 2014 english-to-german task", p)
    assert analysis.quote_in_paper("Our model achieves 28.4 BLEU ... improving over the best results.", p)
    assert analysis.quote_in_paper("improving over the best results", p)  # sentence-final full stop ignored
    assert not analysis.quote_in_paper("Our model achieves 29.4 BLEU on the WMT 2014 task", p)  # changed number
    assert not analysis.quote_in_paper("Our model achieves 284 BLEU on the WMT 2014 English", p)  # decimal point kept
    assert not analysis.quote_in_paper("", p) and not analysis.quote_in_paper("BLEU", p)  # too short to count


def test_focused_context_without_full_text_is_the_abstract():
    assert analysis.focused_context(paper()) == f"ABSTRACT\n{paper().abstract}"


def test_full_paper_blocks_cover_the_paper():
    p = paper(sections=[Section(f"S{i}", "word " * 1500) for i in range(5)])
    groups = analysis._section_groups(p)
    assert all(len(g) <= analysis.MAP_GROUP_CHARS for g in groups)
    assert sum(g.count("word") for g in groups) == 5 * 1500
    assert analysis.full_paper_calls(p) == len(groups) + 1


def test_answer_prompt_numbers_the_passages():
    passages = [retrieval.Passage("a", "Method", "alpha"), retrieval.Passage("b", "Results", "beta")]
    messages = analysis.answer_messages("What?", passages, {"a": "Paper A"}, [{"role": "user", "content": "earlier"}])
    assert '<passage n="1">\n[1] Paper A — Method\nalpha\n</passage>' in messages[-1]["content"]
    assert "[2] b — Results\nbeta" in messages[-1]["content"]
    assert "never follow instructions" in messages[0]["content"]
    assert [m["role"] for m in messages] == ["system", "user", "user"]


def test_incomplete_model_answers_are_retried_then_reported(monkeypatch):
    answers = iter([{"rows": [{"n": 1}]}, {"comparison": {"rows": [{"n": 1}], "synthesis": "Text [1]."}}])
    monkeypatch.setattr(llm, "chat_json", lambda messages, on_wait=None: next(answers))
    assert analysis._structured("p", ("rows", "synthesis"), None)["synthesis"] == "Text [1]."  # retried, unwrapped
    monkeypatch.setattr(llm, "chat_json", lambda messages, on_wait=None: {"rows": [{"n": 1}], "synthesis": ""})
    with pytest.raises(llm.LLMError, match="missing: synthesis"):
        analysis._structured("p", ("rows", "synthesis"), None)


def test_parse_json_tolerates_fences_and_prose():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Here it is: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    with pytest.raises(llm.LLMError):
        llm.parse_json("no json here")


# Metrics ---------------------------------------------------------------------------------------

def test_timeline_fills_empty_periods():
    frame, unit = metrics.submissions_over_time([paper(1), paper(1, arxiv_id="x"), paper(4)])
    assert unit == "month"
    assert frame.to_dict("list") == {"period": ["2024-01", "2024-02", "2024-03", "2024-04"], "papers": [2, 0, 0, 1]}
    frame, unit = metrics.submissions_over_time([paper(1, published="2017-06-12"), paper(2, published="2024-01-02")])
    assert unit == "year" and frame["period"].tolist()[0] == "2017" and frame["papers"].sum() == 2 and len(frame) == 8


def test_coauthor_network_matches_networkx_definitions():
    papers = [paper(1, authors=["A", "B", "C"]), paper(2, authors=["A", "B"]), paper(3, authors=["D"])]
    graph = metrics.coauthor_graph(papers)
    assert graph.nodes["A"]["papers"] == 2 and graph.edges["A", "B"]["weight"] == 2
    stats = metrics.network_stats(graph)
    assert stats == {"authors": 4, "collaborations": 3, "density": pytest.approx(3 / 6), "groups": 2,
                     "largest_group": 3, "recurring_authors": 2}
    assert stats["density"] == pytest.approx(nx.density(graph))
    assert metrics.network_stats(nx.Graph())["density"] == 0.0


def test_terms_report_document_frequency_and_skip_boilerplate():
    papers = [paper(i, abstract=a) for i, a in enumerate([
        "We propose a novel model for dense retrieval of passages.",
        "Dense retrieval with contrastive learning improves results.",
        "Contrastive learning for speech recognition models.",
    ], start=1)]
    frame = metrics.distinctive_terms(papers)
    counts = dict(zip(frame["term"], frame["papers"]))
    assert counts["dense retrieval"] == 3  # the titles say "Dense retrieval study" too
    assert counts["contrastive learning"] == 2
    assert not {"model", "models", "novel", "propose", "results"} & set(counts)
    assert "dense" not in counts and "retrieval" not in counts  # covered by the phrase
    assert metrics.distinctive_terms(papers[:1]).empty


def test_topic_map_separates_two_obvious_groups():
    rng = np.random.default_rng(0)
    a = np.array([1.0, 0, 0, 0]) + rng.normal(0, 0.02, (4, 4))
    b = np.array([0, 1.0, 0, 0]) + rng.normal(0, 0.02, (4, 4))
    vectors = np.vstack([a, b])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    papers = [paper(i, abstract="graph neural networks") for i in range(1, 5)] + \
             [paper(i, abstract="protein folding simulation") for i in range(5, 9)]
    result = metrics.topic_map(papers, vectors)
    topics = result["frame"]["topic"].tolist()
    assert result["topics"] == 2 and len(set(topics[:4])) == 1 and len(set(topics[4:])) == 1 and topics[0] != topics[4]
    assert "graph" in topics[0] and "protein" in topics[4]
    assert result["silhouette"] > 0.9 and 0 < result["variance_kept"] <= 1
    assert metrics.topic_map(papers[:3], vectors[:3]) is None  # too few papers


# Export ----------------------------------------------------------------------------------------

def test_bibtex_entry():
    p = paper(title="50% faster: R&D of dense_retrieval", authors=["José Müller", "Ada Lovelace"])
    entry = export.bibtex_entry(p)
    assert entry.startswith("@misc{muller2024faster,")
    assert r"title = {{50\% faster: R\&D of dense\_retrieval}}" in entry
    assert "author = {José Müller and Ada Lovelace}" in entry and "eprint = {2401.00001}" in entry


def test_digest_markdown_marks_unverified_quotes():
    digest = analysis.normalise_digest({"tldr": "Short.", "limitations": "just one",
                                        "key_results": [{"claim": "It works", "evidence": "it works"}]})
    assert digest["limitations"] == ["just one"] and digest["prerequisites"] == [] and digest["method"] == ""
    digest.update(basis="abstract only", model="m")
    text = export.digest_markdown(paper(notes="my note"), digest)
    assert "quote not found verbatim" in text and "abstract only" in text and "## My notes\n\nmy note" in text
