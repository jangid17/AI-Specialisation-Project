"""Passage retrieval: chunk each paper, embed with a local ONNX model, search by cosine similarity.

The index is a NumPy file per paper. A reading library holds hundreds of papers, so an exact
matrix product is fast enough and no vector database is needed.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from research_assistant import config, store
from research_assistant.models import Paper

CHUNK_CHARS = 1100
CHUNK_OVERLAP = 150
# bge models are trained with this instruction on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Passage:
    arxiv_id: str
    section: str
    text: str
    score: float = 0.0


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding  # imported lazily: loading onnxruntime takes a moment

    return TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    """L2-normalised embeddings, one row per text."""
    if query:
        texts = [QUERY_PREFIX + t for t in texts]
    vectors = np.array(list(_model().embed(texts)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph and sentence boundaries into pieces of at most `size` characters."""
    units = [u for para in text.split("\n\n") for u in re.split(r"(?<=[.!?])\s+", para) if u.strip()]
    chunks, current = [], ""
    for unit in units:
        while len(unit) > size:  # a single run-on sentence, e.g. a long formula
            chunks_piece, unit = unit[:size], unit[size:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(chunks_piece)
        if current and len(current) + len(unit) + 1 > size:
            chunks.append(current)
            current = current[-overlap:].split(" ", 1)[-1] if overlap else ""
        current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_paper(paper: Paper) -> list[Passage]:
    passages = [Passage(paper.arxiv_id, "Abstract", paper.abstract)]
    for section in paper.sections:
        passages += [Passage(paper.arxiv_id, section.title, c) for c in split_text(section.text)]
    return passages


def build_index(paper: Paper) -> int:
    """Embed the paper's passages and write them to disk. Returns the number of passages."""
    passages = chunk_paper(paper)
    vectors = embed([f"{p.section}. {p.text}" for p in passages])
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(store.index_path(paper.arxiv_id), vectors=vectors,
                        passages=json.dumps([[p.section, p.text] for p in passages]))
    return len(passages)


def _load_index(arxiv_id: str) -> tuple[np.ndarray, list[Passage]] | None:
    path = store.index_path(arxiv_id)
    if not path.exists():
        return None
    with np.load(path) as data:
        passages = [Passage(arxiv_id, s, t) for s, t in json.loads(str(data["passages"]))]
        return data["vectors"], passages


def search(query: str, arxiv_ids: list[str], top_k: int = 6) -> list[Passage]:
    """The passages closest to the query across the given papers, best first."""
    vectors, passages = [], []
    for arxiv_id in arxiv_ids:
        loaded = _load_index(arxiv_id)
        if loaded:
            vectors.append(loaded[0])
            passages += loaded[1]
    if not passages:
        return []
    scores = np.vstack(vectors) @ embed([query], query=True)[0]
    best = np.argsort(-scores)[:top_k]
    return [Passage(passages[i].arxiv_id, passages[i].section, passages[i].text, float(scores[i])) for i in best]


def similarity_to(statement: str, texts: list[str]) -> list[float]:
    """Cosine similarity of each text to a statement, e.g. abstracts against a reader's interests."""
    if not statement.strip() or not texts:
        return [0.0] * len(texts)
    return (embed(texts) @ embed([statement], query=True)[0]).tolist()
