# AI research assistant

*Kunal Jangid (B23MT1026) · CSD4020: AI Specialisation Project*

A reading desk for research papers. It finds papers on arXiv, reads their full text, and helps a
researcher decide what to read, understand it faster, and write about it.

- **Real papers.** Search uses the arXiv API. Full text is read from arXiv's HTML version of each paper
  with BeautifulSoup. Citation counts come from Semantic Scholar.
- **A real model.** Digests, answers and comparisons are written by an LLM on Groq's free tier
  (`openai/gpt-oss-120b` by default).
- **Checkable output.** Every key result in a digest carries a quote, and the app checks whether that
  quote appears in the paper. Answers cite the passages they were written from and show them.
- **Hybrid retrieval.** Questions are answered from passages found by BM25 and dense embeddings (FAISS),
  fused with Reciprocal Rank Fusion and re-ranked by a cross-encoder, all on the CPU.
- **Guardrails.** Prompt-attack and safety screening of questions, a relevance gate that refuses rather than
  guesses, passage screening for planted instructions, and checks on every answer's citations and numbers.
- **Measured.** An ablation study of chunking, embedding models, re-rankers and vector stores, a guardrail
  evaluation and a Ragas evaluation are in [`eval/`](eval/README.md).
- **Computed charts.** Nothing on the Insights page is estimated by the model.

A written account of the design, with measurements, is in [`report/report.pdf`](report/report.pdf).

## Setup

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt    # or: python -m venv .venv && pip install -r requirements.txt
cp env.example .env        # then put your Groq key in it (free: https://console.groq.com/keys)
source .venv/bin/activate
streamlit run streamlit_app.py
```

The first time a paper is saved, an embedding model of about 65 MB is downloaded once into `data/models`. It runs on the CPU.

## What it does

| Page | Use it to |
|---|---|
| **Discover** | Search arXiv by words, field, category and sort order. Describe your research interests once and results are ordered by how close each abstract is to them. Follow a search on the **watchlist**, check for papers submitted in the last 7, 14 or 30 days, and ask for a briefing that says which of them deserve your time. Add papers by arXiv ID or URL. |
| **Library** | See everything you saved with reading status, tags and citation counts. Filter, re-label, remove, export BibTeX. |
| **Reader** | Read one paper. **Digest**: problem, method, contributions, key results with checked quotes, limitations, background needed and discussion questions, written for a specialist, a neighbouring field or a graduate student. **Ask**: questions answered from the paper (or the whole library) with the passages shown. **Full text**: the extracted sections. **Notes**: your own, included in the Markdown export. |
| **Synthesize** | Compare two to five papers side by side, see where they agree and differ, and get a related-work paragraph with numbered citations and the BibTeX to go with it. |
| **Insights** | Papers over time, arXiv categories, characteristic terms, a topic map, the co-author network and citation counts. |

### Two digest depths

Groq's free tier allows 8,000 tokens a minute, and a whole paper does not fit in one request.

- **Focused** (one call, a few seconds): the abstract, the introduction, the concluding sections, and the
  passages closest to "method", "results" and "limitations".
- **Full paper** (one call per ~12,000 characters, then one more): notes are taken on every section first,
  and the digest is written from the notes. When the limit is hit the app waits and says so.

Each digest records which of the two it was written from.

## How it works

```
streamlit_app.py            entry point and navigation
app_pages/                  discover, library, reader, synthesize, insights
research_assistant/
  arxiv_client.py           arXiv Atom API: query building, parsing, 3-second spacing between calls
  fulltext.py               BeautifulSoup extraction from arxiv.org/html, with ar5iv as fallback
  chunking.py               six chunking strategies (sentence, fixed, paragraph, semantic, ...)
  retrieval.py              BM25 + dense (fastembed, FAISS) search, rank fusion, cross-encoder re-ranking
  guardrails.py             input, context and output checks
  llm.py                    Groq through its OpenAI-compatible endpoint; rate-limit waits; JSON parsing
  analysis.py               prompts: digest, answer, compare, briefing; quote verification
  metrics.py                timeline, categories, terms (TF-IDF), topic map (k-means + PCA), co-author graph
  citations.py              Semantic Scholar citation counts
  export.py                 BibTeX and Markdown
  store.py, models.py       one JSON file per paper under data/library
  ui.py                     helpers shared by the pages
tests/                      unit tests and page tests (no network or model needed)
eval/                       ablation study, vector-store benchmark, guardrail and Ragas evaluations
report/                     LaTeX report, figures, and the script that produces its numbers
```

Your library, index and profile live in `data/`, which is not committed.

## Tests

```bash
python -m pytest -q
```

64 tests: arXiv feed parsing, full-text extraction, every chunking strategy, BM25, rank fusion, the FAISS
index, every guardrail, quote checking, retries for incomplete model answers, each metric, exports, and
every page rendered with an empty and a populated library.

## Limits worth knowing

- **Full text needs an HTML version.** arXiv and ar5iv cover most papers; where neither has one, the
  paper is saved with its abstract only and marked as such. PDFs are not parsed.
- **Tables and figures are not read**, only their captions. A result that appears only in a table can be
  missed, and a quote taken from a table shows as "not found word for word".
- **A matched quote is not a verified claim.** The check confirms the quote is in the paper, not that
  the model's one-sentence claim follows from it.
- **Similarity scores order results; they are not percentages.**
- **Guardrails reduce risk; they do not remove it.** Prompt Guard caught 12 of 20 test attacks; the output
  checks and the prompt are the second line of defence (see `eval/README.md`).
- **Authors are matched by name string**, so namesakes merge and spelling variants split.
- **Citation counts** depend on Semantic Scholar having indexed the paper, and its anonymous API is
  rate-limited.

## License

MIT, see [LICENSE](LICENSE).
