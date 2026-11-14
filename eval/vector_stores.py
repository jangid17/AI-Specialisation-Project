"""Benchmark vector stores on the library's real vectors and on a larger synthetic set.

    .venv-eval/bin/python eval/vector_stores.py [--chunking sentence-1100] [--model BAAI/bge-small-en-v1.5]

Stores: exact NumPy (the previous app), FAISS exact (IndexFlatIP) and approximate (HNSW), ChromaDB (embedded,
on disk, cosine HNSW) and Qdrant's local mode (embedded, on disk; it searches by brute force, since HNSW is
part of the Qdrant server, not the in-process client). Pinecone is a hosted service that needs an account and API key, so it
is not run here; its cost is network round trips, which the README discusses.

Measured per store: time to build, size on disk, median and 95th-percentile query latency over the
evaluation questions (whole library, and filtered to one paper), and recall@10 against exact search.
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from common import load_questions, save_result

from research_assistant import config, retrieval, store

K = 10


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


class Numpy:
    name = "NumPy exact (before)"

    def build(self, vectors, papers, tmp):
        self.v, self.papers = vectors, np.array(papers)
        np.save(tmp / "v.npy", vectors)

    def search(self, q, paper=None):
        scores = self.v @ q
        if paper is not None:
            scores = np.where(self.papers == paper, scores, -np.inf)
        return list(np.argsort(-scores)[:K])


class FaissFlat:
    name = "FAISS IndexFlatIP (exact)"

    def _index(self, d):
        import faiss
        return faiss.IndexFlatIP(d)

    def build(self, vectors, papers, tmp):
        import faiss
        self.faiss = faiss
        self.index = self._index(vectors.shape[1])
        self.index.add(vectors)
        self.papers = np.array(papers)
        faiss.write_index(self.index, str(tmp / "index.faiss"))

    def search(self, q, paper=None):
        params = None
        if paper is not None:
            ids = np.flatnonzero(self.papers == paper).astype(np.int64)
            params = self._params(ids)
        _, ids = self.index.search(q[None, :], K, params=params)
        return [int(i) for i in ids[0] if i >= 0]

    def _params(self, ids):
        return self.faiss.SearchParameters(sel=self.faiss.IDSelectorBatch(ids))


class FaissHNSW(FaissFlat):
    name = "FAISS HNSW (approximate)"

    def _index(self, d):
        import faiss
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
        return index

    def _params(self, ids):
        return self.faiss.SearchParametersHNSW(sel=self.faiss.IDSelectorBatch(ids), efSearch=64)


class Chroma:
    name = "ChromaDB (embedded, HNSW)"

    def build(self, vectors, papers, tmp):
        import chromadb
        client = chromadb.PersistentClient(path=str(tmp))
        self.col = client.create_collection("passages", metadata={"hnsw:space": "cosine"})
        for start in range(0, len(vectors), 5000):
            end = min(start + 5000, len(vectors))
            self.col.add(ids=[str(i) for i in range(start, end)], embeddings=vectors[start:end].tolist(),
                         metadatas=[{"paper": papers[i]} for i in range(start, end)])

    def search(self, q, paper=None):
        r = self.col.query(query_embeddings=[q.tolist()], n_results=K, where={"paper": paper} if paper else None)
        return [int(i) for i in r["ids"][0]]


class Qdrant:
    name = "Qdrant (local mode, no HNSW)"  # the in-process client is brute force; HNSW needs the server

    def build(self, vectors, papers, tmp):
        from qdrant_client import QdrantClient, models
        self.m = models
        self.client = QdrantClient(path=str(tmp))
        self.client.create_collection("passages", vectors_config=models.VectorParams(size=vectors.shape[1],
                                                                              distance=models.Distance.COSINE))
        for start in range(0, len(vectors), 2000):
            end = min(start + 2000, len(vectors))
            self.client.upsert("passages", points=models.Batch(ids=list(range(start, end)), vectors=vectors[start:end].tolist(),
                                                       payloads=[{"paper": papers[i]} for i in range(start, end)]))

    def search(self, q, paper=None):
        flt = None
        if paper is not None:
            flt = self.m.Filter(must=[self.m.FieldCondition(key="paper", match=self.m.MatchValue(value=paper))])
        r = self.client.query_points("passages", query=q.tolist(), limit=K, query_filter=flt)
        return [int(p.id) for p in r.points]


STORES = [Numpy, FaissFlat, FaissHNSW, Chroma, Qdrant]


def run(vectors: np.ndarray, papers: list[str], queries: np.ndarray, query_papers: list[str]) -> list[dict]:
    exact = Numpy()
    exact.build(vectors, papers, Path(tempfile.mkdtemp()))
    truth = [set(exact.search(q)) for q in queries]
    rows = []
    for cls in STORES:
        s, tmp = cls(), Path(tempfile.mkdtemp())
        start = time.perf_counter()
        s.build(vectors, papers, tmp)
        build = time.perf_counter() - start
        s.search(queries[0])  # warm-up
        lat, flat, recall = [], [], []
        for q, paper, t in zip(queries, query_papers, truth):
            t0 = time.perf_counter()
            got = s.search(q)
            lat.append((time.perf_counter() - t0) * 1000)
            recall.append(len(set(got) & t) / K)
            t0 = time.perf_counter()
            s.search(q, paper)
            flat.append((time.perf_counter() - t0) * 1000)
        rows.append({"store": cls.name, "vectors": len(vectors), "build_s": round(build, 3),
                     "disk_mb": round(dir_size(tmp) / 1e6, 2), "query_ms_p50": float(np.median(lat)),
                     "query_ms_p95": float(np.percentile(lat, 95)), "filtered_ms_p50": float(np.median(flat)),
                     "recall@10": float(np.mean(recall))})
        print(rows[-1], flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunking", default=config.CHUNKING)
    ap.add_argument("--model", default=config.EMBEDDING_MODEL)
    ap.add_argument("--synthetic", type=int, default=100_000)
    a = ap.parse_args()
    passages = [x for p in store.load_all() if p.has_full_text for x in retrieval.chunk_paper(p, a.chunking, a.model)]
    vectors = retrieval.embed([p.indexed for p in passages], model=a.model)
    papers = [p.arxiv_id for p in passages]
    qs = [q for q in load_questions() if q["kind"] != "unanswerable"]
    queries = retrieval.embed([q["question"] for q in qs], query=True, model=a.model)
    real = run(vectors, papers, queries, [q["arxiv_id"] for q in qs])
    # A library over a hundred times bigger: the real vectors plus small noise, so neighbourhoods stay realistic.
    rng = np.random.default_rng(0)
    big = vectors[rng.integers(0, len(vectors), a.synthetic)] + rng.normal(0, 0.03, (a.synthetic, vectors.shape[1]))
    big = (big / np.linalg.norm(big, axis=1, keepdims=True)).astype(np.float32)
    big_papers = [f"p{i % 500}" for i in range(a.synthetic)]
    synthetic = run(big, big_papers, queries, [f"p{i % 500}" for i in range(len(queries))])
    print("saved", save_result("vector_stores", {"chunking": a.chunking, "embedding_model": a.model, "k": K,
                                                 "real": real, "synthetic": synthetic}))


if __name__ == "__main__":
    main()
