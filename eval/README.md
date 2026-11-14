# Evaluation

How each retrieval and guardrail setting in `research_assistant/config.py` was chosen, and how well the
whole pipeline answers. Raw results are in `results/*.json`; the report (`report/report.pdf`, section
"Evaluation") discusses them.

## Running

```bash
# retrieval ablation and guardrails: the app's environment
python eval/build_questions.py            # once: 89 questions about the saved papers (uses Groq)
python eval/ablation.py chunking
python eval/ablation.py embeddings --chunking sentence-512
python eval/ablation.py rerankers  --chunking sentence-512 --model BAAI/bge-small-en-v1.5
python eval/ablation.py final      --chunking sentence-512 --model BAAI/bge-small-en-v1.5 \
                                   --reranker Xenova/ms-marco-MiniLM-L-6-v2 --candidates 30
python eval/guardrails_eval.py            # uses Groq

# vector stores and Ragas: a separate environment (Ragas pins an older openai SDK)
uv venv .venv-eval --python 3.12 && uv pip install -p .venv-eval -r requirements-eval.txt
.venv-eval/bin/python eval/vector_stores.py
.venv-eval/bin/python eval/ragas_eval.py  # uses Groq; resumes from eval/.cache/ragas.json
```

Vectors are cached in `eval/.cache/` per (chunking, model), so later stages do not re-embed.

## Questions and metrics

`questions.json`: 36 questions from digest key results, 41 from random passages, 12 the library cannot
answer. Each answerable question keeps its evidence sentence; a retrieved passage is **relevant** when it
contains at least half of that sentence's word 5-grams, so no model judges retrieval and the same labels
apply to every chunking.

Over the 77 answerable questions, searching the whole library: **hit@1**, **hit@5**, **MRR@10** (with a
95% bootstrap interval of about ±0.09), **hit@budget** (a relevant passage among the top results that fit
in the 6,600 characters the app sends the model; the fair comparison between chunk sizes), and median
latency per query on a 16-core CPU.

## Chunking (`results/chunking.json`)

bge-small embeddings; MRR@10 for each retrieval method, then hit@budget and latency for hybrid + re-rank.

| Strategy | Passages | Dense | BM25 | Hybrid | Hybrid + re-rank | hit@budget | ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-1100 | 731 | 0.545 | 0.565 | 0.607 | 0.667 | 0.753 | 1044 |
| **sentence-512** | 1,661 | 0.516 | 0.485 | 0.584 | 0.659 | **0.870** | **570** |
| sentence-1100 (before) | 750 | 0.523 | 0.543 | 0.563 | **0.683** | 0.792 | 1025 |
| sentence-2048 | 422 | 0.444 | 0.541 | 0.525 | 0.630 | 0.688 | 1569 |
| paragraph | 811 | 0.579 | 0.571 | 0.642 | 0.668 | 0.844 | 982 |
| semantic | 709 | 0.551 | 0.576 | 0.604 | 0.663 | 0.831 | 1254 |

Fusion and re-ranking help every strategy. Re-ranked MRRs all fall within each other's intervals, so
**sentence-512** was chosen for getting evidence to the model most often and re-ranking fastest.

## Embedding models (`results/embeddings.json`)

sentence-512 chunks. Index time is for all 1,661 passages.

| Model | Dim | Index (s) | Dense ms | Dense MRR | Hybrid + re-rank MRR | hit@budget |
|---|---:|---:|---:|---:|---:|---:|
| **bge-small-en-v1.5** | 384 | 58 | 5.3 | 0.516 | 0.659 | 0.870 |
| bge-base-en-v1.5 | 768 | 186 | 14.3 | 0.526 | 0.658 | 0.857 |
| all-MiniLM-L6-v2 | 384 | 34 | 5.1 | 0.374 | 0.645 | 0.857 |
| snowflake-arctic-embed-s | 384 | 66 | 8.6 | 0.544 | 0.650 | 0.844 |
| snowflake-arctic-embed-m | 768 | 195 | 16.8 | 0.575 | 0.669 | 0.870 |
| nomic-embed-text-v1.5 | 768 | 271 | 19.8 | 0.565 | 0.665 | 0.883 |
| mxbai-embed-large-v1 | 1024 | 371 | 51.7 | 0.506 | 0.675 | 0.896 |
| granite-embedding-small-english-r2 | 384 | 66 | 6.3 | 0.473 | 0.658 | 0.857 |
| potion-retrieval-32M | 512 | 21 | 0.2 | 0.457 | 0.655 | 0.844 |

Alone, the embedding model matters (dense MRR 0.37 to 0.58). Inside the hybrid pipeline every model lands
between 0.645 and 0.675, so the cheap **bge-small** was kept.

## Re-rankers (`results/rerankers.json`)

| Re-ranker | Candidates | MRR@10 | hit@budget | ms/query |
|---|---:|---:|---:|---:|
| none (hybrid fusion only) | – | 0.584 | 0.831 | 6 |
| **ms-marco-MiniLM-L-6-v2** | 20 / **30** / 50 | 0.649 / **0.659** / 0.663 | 0.844 / **0.870** / 0.857 | 349 / **528** / 902 |
| ms-marco-MiniLM-L-12-v2 | 20 / 30 / 50 | 0.634 / 0.644 / 0.646 | 0.857 / 0.883 / 0.870 | 671 / 975 / 1409 |
| jina-reranker-v1-turbo-en | 20 / 30 / 50 | 0.603 / 0.607 / 0.607 | 0.844 / 0.831 / 0.831 | 454 / 678 / 1025 |
| bge-reranker-base | 20 / 30 / 50 | 0.701 / 0.708 / 0.722 | 0.857 / 0.883 / 0.909 | 1695 / 2518 / 4121 |

bge-reranker-base is the most accurate but five times slower; set `RA_RERANK_MODEL=BAAI/bge-reranker-base`
to prefer accuracy.

## Before and after (`results/final.json`)

| Pipeline | Scope | hit@1 | hit@5 | MRR@10 | hit@budget | ms/query |
|---|---|---:|---:|---:|---:|---:|
| Before: dense, sentence-1100 | one paper | 0.481 | 0.805 | 0.614 | 0.805 | 5 |
| **After: hybrid + re-rank, sentence-512** | one paper | 0.571 | 0.857 | 0.694 | **0.948** | 539 |
| Before | whole library | 0.416 | 0.675 | 0.523 | 0.714 | 5 |
| **After** | whole library | 0.558 | 0.779 | 0.659 | **0.870** | 540 |

Relevance gate calibration (same file): answerable questions' best re-ranker score is −4 or higher,
unanswerable ones −6.2 or lower; the threshold is set to −5.

## Vector stores (`results/vector_stores.json`)

| Vectors | Store | Build (s) | Disk (MB) | Query p50 (ms) | One paper p50 (ms) | Recall@10 |
|---|---|---:|---:|---:|---:|---:|
| 1,661 | NumPy (before) | 0.00 | 2.6 | 3.02 | 2.96 | 1.000 |
| 1,661 | **FAISS IndexFlatIP** | 0.00 | 2.6 | **0.07** | **0.05** | 1.000 |
| 1,661 | FAISS HNSW | 0.22 | 3.0 | 0.05 | 0.08 | 1.000 |
| 1,661 | ChromaDB | 0.47 | 6.8 | 0.60 | 1.07 | 0.999 |
| 1,661 | Qdrant (local mode) | 3.65 | 6.9 | 1.77 | 16.24 | 1.000 |
| 100,000 | NumPy | 0.09 | 153.6 | 7.00 | 4.15 | 1.000 |
| 100,000 | FAISS IndexFlatIP | 0.14 | 153.6 | 8.65 | 0.88 | 1.000 |
| 100,000 | FAISS HNSW | 4.02 | 180.8 | 0.14 | 0.32 | 0.917 |
| 100,000 | ChromaDB | 12.84 | 201.9 | 1.65 | 35.98 | 0.955 |
| 100,000 | Qdrant (local mode) | 250.3 | 414.0 | 110.5 | 605.6 | 1.000 |

The 100,000 vectors are real ones plus small noise. Qdrant's in-process client searches by brute force
(HNSW is in the Qdrant server). Pinecone is hosted and needs an account; each query would add a network
round trip far larger than any time above. **FAISS exact search** is the choice at a personal library's
size; HNSW is a one-line change if a library ever reaches about 100,000 passages.

## Guardrails (`results/guardrails.json`)

| Check | Result |
|---|---|
| Prompt Guard 2 on 20 attack prompts | 12 caught |
| Regex screen on the same 20 | 10 caught (15 by either) |
| False alarms on 30 benign questions (10 mention prompts or jailbreaks) | Prompt Guard 1, regex 3 |
| Safety policy: 6 forbidden requests / 14 benign | 6 blocked / 0 false alarms |
| Relevance gate: 12 unanswerable / 77 answerable questions | 12 refused / 0 refused |
| Planted "reply only ACCESS GRANTED" passage, 8 questions | flagged 8, obeyed 0 |

## Ragas (`results/ragas.json`)

Whole pipeline, before vs after, 16 answerable questions about one paper each; answers by gpt-oss-120b,
judged by `qwen/qwen3.8-27b` (mean and 95% CI).

| Metric | Before | After |
|---|---:|---:|
| Faithfulness | 0.961 [0.91, 1.00] | 0.958 [0.91, 1.00] |
| Answer relevancy | 0.885 [0.85, 0.92] | 0.893 [0.86, 0.92] |
| Context recall | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |

At paper scope both pipelines find the evidence (context recall 1.0) and the model stays faithful to it, so
the scores do not separate them; the retrieval gains are at library scope (table above). Output checks over
the 32 answers: no invalid citation, no unsupported number, 18 answers with at least one uncited sentence.
The first judge (gpt-oss-20b) ran out of its daily allowance; its partial scores were discarded so that one
model judges every answer.

