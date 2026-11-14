"""Retrieval ablation study over the saved library. Run each stage in order; each fixes the one before.

    python eval/ablation.py chunking
    python eval/ablation.py embeddings --chunking <best>
    python eval/ablation.py rerankers  --chunking <best> --model <best>
    python eval/ablation.py final      --chunking <best> --model <best> --reranker <best> --candidates <n>

No LLM is involved: a retrieved passage counts as relevant when it holds the question's evidence sentence
(common.relevant). Results go to eval/results/<stage>.json.

Metrics, over the answerable questions, searching the whole library unless stated:
  hit@1, hit@5   share of questions with a relevant passage at rank 1 / in the top 5
  mrr@10         mean reciprocal rank of the first relevant passage (0 when outside the top 10)
  hit@budget     a relevant passage among the top results that fit in 6,600 characters, the amount of
                 text the app sends the model; this is the fair comparison between chunk sizes
"""

import argparse
import time

import numpy as np
from common import APP_CONTEXT_CHARS, EVAL_DIR, bootstrap_ci, load_questions, mean, relevant, save_result

from research_assistant import chunking, config, retrieval, store

CACHE = EVAL_DIR / ".cache"
DEPTH = 20  # results retrieved per question
EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2",
    "snowflake/snowflake-arctic-embed-s", "snowflake/snowflake-arctic-embed-m", "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1", "ibm-granite/granite-embedding-small-english-r2",
    "minishlab/potion-retrieval-32M",
]
RERANKERS = ["Xenova/ms-marco-MiniLM-L-6-v2", "Xenova/ms-marco-MiniLM-L-12-v2", "jinaai/jina-reranker-v1-turbo-en",
             "BAAI/bge-reranker-base"]


def papers():
    return sorted((p for p in store.load_all() if p.has_full_text), key=lambda p: p.arxiv_id)


def build(strategy: str, model: str) -> tuple[retrieval.LibraryIndex, dict]:
    """Chunk and embed the whole library; vectors are cached on disk per (strategy, model)."""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"{strategy}__{model.replace('/', '_')}.npz"
    passages = []
    start = time.perf_counter()
    for paper in papers():
        passages += retrieval.chunk_paper(paper, strategy, model)
    chunk_seconds = time.perf_counter() - start
    texts = [p.indexed for p in passages]
    if cache.exists() and np.load(cache)["n"] == len(texts):
        data = np.load(cache)
        vectors, embed_seconds = data["vectors"], float(data["seconds"])
    else:
        start = time.perf_counter()
        # 24-layer models (mxbai-large) run out of memory at the default batch of 32 on a 16 GB machine.
        vectors = retrieval.embed(texts, model=model, batch_size=4 if "large" in model else 32)
        embed_seconds = time.perf_counter() - start
        np.savez(cache, vectors=vectors, seconds=embed_seconds, n=len(texts))
    lengths = [len(p.text) for p in passages]
    stats = {"passages": len(passages), "mean_chars": round(mean(lengths)), "chunk_seconds": round(chunk_seconds, 2),
             "embed_seconds": round(embed_seconds, 1), "dim": int(vectors.shape[1])}
    return retrieval.LibraryIndex(passages, vectors, model), stats


def score(index: retrieval.LibraryIndex, questions: list[dict], method: str, scope: str = "library",
          reranker: str | None = None, candidates: int = 30) -> dict:
    hit1, hit5, rr, budget, latency = [], [], [], [], []
    for q in questions:
        ids = None if scope == "library" else [q["arxiv_id"]]
        start = time.perf_counter()
        found = index.search(q["question"], ids, top_k=DEPTH, method=method, candidates=candidates, reranker=reranker)
        latency.append((time.perf_counter() - start) * 1000)
        rel = [relevant(p.text, q["evidence"]) for p in found]
        first = rel.index(True) + 1 if True in rel else None
        hit1.append(float(first == 1))
        hit5.append(float(first is not None and first <= 5))
        rr.append(1 / first if first and first <= 10 else 0.0)
        used, ok = 0, False
        for p, r in zip(found, rel):
            used += len(p.text)
            if used > APP_CONTEXT_CHARS:
                break
            ok = ok or r
        budget.append(float(ok))
    return {"method": method, "scope": scope, "hit@1": mean(hit1), "hit@5": mean(hit5), "mrr@10": mean(rr),
            "mrr@10_ci95": bootstrap_ci(rr), "hit@budget": mean(budget),
            "latency_ms_median": float(np.median(latency)), "latency_ms_p95": float(np.percentile(latency, 95)),
            "per_question_rr": rr}


def answerable() -> list[dict]:
    return [q for q in load_questions() if q["kind"] != "unanswerable"]


def show(rows: list[dict], keys: list[str]) -> None:
    for r in rows:
        print("  ".join(f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys), flush=True)


def stage_chunking(args) -> dict:
    qs, rows = answerable(), []
    for strategy in chunking.STRATEGIES:
        index, stats = build(strategy, config.EMBEDDING_MODEL)
        for method in retrieval.METHODS:
            rows.append({"strategy": strategy, **stats, **score(index, qs, method)})
            show(rows[-1:], ["strategy", "method", "passages", "hit@1", "hit@5", "mrr@10", "hit@budget", "latency_ms_median"])
    return {"embedding_model": config.EMBEDDING_MODEL, "reranker": config.RERANK_MODEL, "questions": len(qs),
            "descriptions": {k: v[0] for k, v in chunking.STRATEGIES.items()}, "rows": rows}


def stage_embeddings(args) -> dict:
    qs, rows = answerable(), []
    for model in EMBEDDING_MODELS:
        index, stats = build(args.chunking, model)
        for method in ("dense", "hybrid+rerank"):
            rows.append({"model": model, **stats, **score(index, qs, method)})
            show(rows[-1:], ["model", "method", "dim", "embed_seconds", "hit@1", "hit@5", "mrr@10", "hit@budget", "latency_ms_median"])
    return {"chunking": args.chunking, "reranker": config.RERANK_MODEL, "questions": len(qs), "rows": rows}


def stage_rerankers(args) -> dict:
    qs, rows = answerable(), []
    index, stats = build(args.chunking, args.model)
    rows.append({"reranker": None, "candidates": 0, **score(index, qs, "hybrid")})
    for reranker in RERANKERS:
        for candidates in (20, 30, 50):  # at least DEPTH: search() never re-ranks fewer than it returns
            rows.append({"reranker": reranker, "candidates": candidates,
                         **score(index, qs, "hybrid+rerank", reranker=reranker, candidates=candidates)})
            show(rows[-1:], ["reranker", "candidates", "hit@1", "hit@5", "mrr@10", "hit@budget", "latency_ms_median"])
    return {"chunking": args.chunking, "embedding_model": args.model, "questions": len(qs), "rows": rows}


def stage_final(args) -> dict:
    """The app before and after, in both scopes, and the calibration of the relevance gate."""
    qs = answerable()
    before, _ = build("sentence-1100", "BAAI/bge-small-en-v1.5")
    after, stats = build(args.chunking, args.model)
    rows = []
    for name, index, method, reranker in (("before: dense, sentence-1100, bge-small", before, "dense", None),
                                          ("after", after, "hybrid+rerank", args.reranker)):
        for scope in ("paper", "library"):
            rows.append({"pipeline": name, **score(index, qs, method, scope, reranker, args.candidates)})
            show(rows[-1:], ["pipeline", "scope", "hit@1", "hit@5", "mrr@10", "hit@budget", "latency_ms_median"])
    # Relevance gate: the best re-ranker score for questions the library can and cannot answer.
    unanswerable = [q for q in load_questions() if q["kind"] == "unanswerable"]
    top = lambda q, scope: after.search(q["question"], None if scope == "library" else [q["arxiv_id"]], top_k=1,
                                        method="hybrid+rerank", candidates=args.candidates, reranker=args.reranker)[0].score
    pos = [top(q, "paper") for q in qs]
    neg = [top(q, "library") for q in unanswerable]
    thresholds = np.unique(np.round(np.concatenate([pos, neg]), 2))
    def balanced(t):
        return (np.mean(np.array(pos) >= t) + np.mean(np.array(neg) < t)) / 2
    best = max(thresholds, key=lambda t: (balanced(t), -t))
    gate = {"answerable_scores": pos, "unanswerable_scores": neg, "threshold": float(best),
            "answerable_kept": float(np.mean(np.array(pos) >= best)), "unanswerable_refused": float(np.mean(np.array(neg) < best))}
    print("gate", {k: v for k, v in gate.items() if not k.endswith("scores")})
    return {"chunking": args.chunking, "embedding_model": args.model, "reranker": args.reranker,
            "candidates": args.candidates, "index": stats, "questions": len(qs), "rows": rows, "gate": gate}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["chunking", "embeddings", "rerankers", "final"])
    ap.add_argument("--chunking", default=config.CHUNKING)
    ap.add_argument("--model", default=config.EMBEDDING_MODEL)
    ap.add_argument("--reranker", default=config.RERANK_MODEL)
    ap.add_argument("--candidates", type=int, default=config.RERANK_CANDIDATES)
    a = ap.parse_args()
    result = {"chunking": stage_chunking, "embeddings": stage_embeddings, "rerankers": stage_rerankers,
              "final": stage_final}[a.stage](a)
    print("saved", save_result(a.stage, result))
