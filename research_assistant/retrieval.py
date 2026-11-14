"""Passage retrieval: BM25 and dense embeddings, fused, then re-ranked by a cross-encoder.

Pipeline for a query (method "hybrid+rerank", the default):
  1. dense: the query embedding against every passage vector in a FAISS inner-product index;
  2. sparse: BM25 over the same passages;
  3. Reciprocal Rank Fusion of the two rankings, keeping the best `candidates`;
  4. a cross-encoder scores each (query, passage) pair and the best `top_k` are returned.

Each paper's passages and vectors are saved in one .npz file, which is the source of truth; the library-
wide FAISS and BM25 indexes are assembled from those files in memory and rebuilt when any of them changes.
eval/ablation.py measures each stage, the chunking strategies and the embedding models.
"""

import json
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from research_assistant import chunking, config, store
from research_assistant.models import Paper

METHODS = ["dense", "bm25", "hybrid", "hybrid+rerank"]
RRF_K = 60  # the constant from Cormack et al. (2009); larger values flatten the fusion

# Instruction prefixes each embedding model was trained with: (query side, passage side).
EMBEDDING_PREFIXES = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-s": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-m": ("Represent this sentence for searching relevant passages: ", ""),
    "mixedbread-ai/mxbai-embed-large-v1": ("Represent this sentence for searching relevant passages: ", ""),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
}


@dataclass
class Passage:
    arxiv_id: str
    section: str
    text: str
    score: float = 0.0

    @property
    def indexed(self) -> str:
        """What the embedding model, BM25 and the re-ranker see: the text with its section title."""
        return f"{self.section}. {self.text}"


# Models -----------------------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _embedder(model: str):
    from fastembed import TextEmbedding  # imported lazily: loading onnxruntime takes a moment

    return TextEmbedding(model_name=model, cache_dir=str(config.MODEL_CACHE_DIR))


def embed(texts: list[str], query: bool = False, model: str | None = None, batch_size: int = 32) -> np.ndarray:
    """L2-normalised embeddings, one row per text."""
    model = model or config.EMBEDDING_MODEL
    prefix = EMBEDDING_PREFIXES.get(model, ("", ""))[0 if query else 1]
    # Small batches: at 512 tokens a passage, fastembed's default of 256 needs gigabytes for attention alone.
    vectors = np.array(list(_embedder(model).embed([prefix + t for t in texts], batch_size=batch_size)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@lru_cache(maxsize=2)
def _reranker(model: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model, cache_dir=str(config.MODEL_CACHE_DIR))


def rerank_scores(query: str, texts: list[str], model: str | None = None) -> np.ndarray:
    """Cross-encoder relevance of each text to the query (logits: higher is more relevant)."""
    return np.array(list(_reranker(model or config.RERANK_MODEL).rerank(query, texts, batch_size=16)), dtype=np.float32)


# Sparse retrieval ---------------------------------------------------------------------------------

class BM25:
    """Okapi BM25 with the term weights precomputed, so scoring a query is one sparse column sum."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.vectorizer = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w[\w-]*\w\b|\b\d\b")
        tf = self.vectorizer.fit_transform(texts).tocoo().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        length = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * length / max(length.mean(), 1))
        weights = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.weights = sparse.csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        terms = self.vectorizer.transform([query]).indices
        return np.asarray(self.weights[:, terms].sum(axis=1)).ravel()


def rrf(rankings: list[np.ndarray], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: each list contributes 1 / (k + rank) for every item it ranks."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[int(item)] = fused.get(int(item), 0.0) + 1.0 / (k + rank)
    return fused


# The library index ----------------------------------------------------------------------------

class LibraryIndex:
    """Dense (FAISS), sparse (BM25) and fused search over a fixed list of passages."""

    def __init__(self, passages: list[Passage], vectors: np.ndarray, model: str | None = None):
        self.passages = passages
        self.model = model or config.EMBEDDING_MODEL
        self.faiss = faiss.IndexFlatIP(vectors.shape[1])  # exact search; vectors are unit length, so IP = cosine
        self.faiss.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.bm25 = BM25([p.indexed for p in passages])
        self.paper_of = np.array([p.arxiv_id for p in passages])

    def _allowed(self, arxiv_ids: list[str] | None) -> np.ndarray | None:
        if arxiv_ids is None:
            return None
        return np.flatnonzero(np.isin(self.paper_of, arxiv_ids))

    def dense(self, query: str, allowed: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray]:
        q = embed([query], query=True, model=self.model)
        n = min(n, self.faiss.ntotal if allowed is None else len(allowed))
        params = None if allowed is None else faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed.astype(np.int64)))
        scores, ids = self.faiss.search(q, n, params=params)
        keep = ids[0] >= 0
        return ids[0][keep], scores[0][keep]

    def sparse(self, query: str, allowed: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.bm25.scores(query)
        if allowed is not None:
            masked = np.full_like(scores, -np.inf)
            masked[allowed] = scores[allowed]
            scores = masked
        order = np.argsort(-scores)[:n]
        order = order[np.isfinite(scores[order]) & (scores[order] > 0)]
        return order, scores[order]

    def search(self, query: str, arxiv_ids: list[str] | None = None, top_k: int = 6, method: str | None = None,
               candidates: int | None = None, reranker: str | None = None) -> list[Passage]:
        method = method or config.RETRIEVAL_METHOD
        candidates = max(candidates or config.RERANK_CANDIDATES, top_k)
        allowed = self._allowed(arxiv_ids)
        if allowed is not None and not len(allowed):
            return []
        if method == "dense":
            ids, scores = self.dense(query, allowed, top_k)
        elif method == "bm25":
            ids, scores = self.sparse(query, allowed, top_k)
        else:
            fused = rrf([self.dense(query, allowed, candidates)[0], self.sparse(query, allowed, candidates)[0]])
            best = sorted(fused, key=fused.get, reverse=True)[:candidates]
            ids, scores = np.array(best, dtype=int), np.array([fused[i] for i in best])
            if method == "hybrid+rerank" and len(ids):
                scores = rerank_scores(query, [self.passages[i].indexed for i in ids], reranker)
                order = np.argsort(-scores)
                ids, scores = ids[order], scores[order]
        return [Passage(self.passages[i].arxiv_id, self.passages[i].section, self.passages[i].text, float(s))
                for i, s in zip(ids[:top_k], scores[:top_k])]


# Building and loading -----------------------------------------------------------------------

def chunk_paper(paper: Paper, strategy: str | None = None, model: str | None = None) -> list[Passage]:
    embed_sentences = lambda texts: embed(texts, model=model)  # only the semantic strategy uses it
    return [Passage(paper.arxiv_id, section, text)
            for section, text in chunking.chunk_sections(paper, strategy or config.CHUNKING, embed_sentences)]


def build_index(paper: Paper) -> int:
    """Chunk and embed the paper and save its passages and vectors. Returns the number of passages."""
    passages = chunk_paper(paper)
    vectors = embed([p.indexed for p in passages])
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(store.index_path(paper.arxiv_id), vectors=vectors,
                        passages=json.dumps([[p.section, p.text] for p in passages]),
                        settings=json.dumps(_settings()))
    return len(passages)


def _settings() -> dict:
    return {"chunking": config.CHUNKING, "embedding_model": config.EMBEDDING_MODEL}


def _stale(path) -> bool:
    with np.load(path) as data:
        return "settings" not in data or json.loads(str(data["settings"])) != _settings()


def reindex_stale() -> int:
    """Re-index every saved paper whose index was built with other chunking or embedding settings."""
    rebuilt = 0
    for path in sorted(config.INDEX_DIR.glob("*.npz")) if config.INDEX_DIR.exists() else []:
        if _stale(path):
            paper = store.load(store.arxiv_id_from_index(path))
            if paper is None:
                path.unlink()
            else:
                build_index(paper)
            rebuilt += 1
    return rebuilt


def _stamp() -> tuple:
    files = sorted(config.INDEX_DIR.glob("*.npz")) if config.INDEX_DIR.exists() else []
    return tuple((f.name, f.stat().st_mtime_ns) for f in files) + tuple(_settings().values())


@lru_cache(maxsize=1)
def _load_library(stamp: tuple) -> LibraryIndex | None:
    passages, vectors = [], []
    for path in sorted(config.INDEX_DIR.glob("*.npz")):
        arxiv_id = store.arxiv_id_from_index(path)
        with np.load(path) as data:
            passages += [Passage(arxiv_id, s, t) for s, t in json.loads(str(data["passages"]))]
            vectors.append(data["vectors"])
    return LibraryIndex(passages, np.vstack(vectors)) if passages else None


def library_index() -> LibraryIndex | None:
    if reindex_stale():
        _load_library.cache_clear()
    return _load_library(_stamp())


def search(query: str, arxiv_ids: list[str], top_k: int = 6, method: str | None = None) -> list[Passage]:
    """The passages most relevant to the query within the given papers, best first."""
    index = library_index()
    return index.search(query, arxiv_ids, top_k=top_k, method=method) if index else []


def similarity_to(statement: str, texts: list[str]) -> list[float]:
    """Cosine similarity of each text to a statement, e.g. abstracts against a reader's interests."""
    if not statement.strip() or not texts:
        return [0.0] * len(texts)
    return (embed(texts) @ embed([statement], query=True)[0]).tolist()
