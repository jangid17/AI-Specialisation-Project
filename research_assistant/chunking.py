"""Ways to split a paper into passages for retrieval. eval/ablation.py compares them; config picks one.

Every strategy keeps the abstract as one passage and never lets a passage cross a section boundary,
so each passage can say which section it came from.
"""

import re
from collections.abc import Callable

import numpy as np

from research_assistant.models import Paper

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, size: int = 1100, overlap: int = 150) -> list[str]:
    """Split on paragraph and sentence boundaries into pieces of at most `size` characters."""
    units = [u for para in text.split("\n\n") for u in _SENTENCE_END.split(para) if u.strip()]
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


def fixed_windows(text: str, size: int = 1100, overlap: int = 150) -> list[str]:
    """Windows of exactly `size` characters, cut wherever they fall: the naive baseline."""
    text = " ".join(text.split())
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def paragraphs(text: str, size: int = 1100) -> list[str]:
    """The paper's own paragraphs; short ones are merged up to `size`, long ones split by sentence."""
    chunks, current = [], ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks += split_text(para, size, overlap=0)
        elif current and len(current) + len(para) + 1 > size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def semantic(text: str, embed: Callable[[list[str]], np.ndarray], max_size: int = 1600, min_size: int = 300,
             percentile: float = 80) -> list[str]:
    """Start a new passage where consecutive sentences are least alike (the top 20% of distances).

    Passages are kept between `min_size` and `max_size` characters, so a run of similar sentences is still
    cut and a lone sentence is joined to its neighbour.
    """
    sentences = [s for para in text.split("\n\n") for s in _SENTENCE_END.split(para) if s.strip()]
    if len(sentences) < 3:
        return [" ".join(sentences)] if sentences else []
    vectors = embed(sentences)
    distance = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)  # vectors are unit length
    cut_after = distance > np.percentile(distance, percentile)
    chunks, current = [], sentences[0]
    for i, sentence in enumerate(sentences[1:]):
        if (cut_after[i] and len(current) >= min_size) or len(current) + len(sentence) + 1 > max_size:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}"
    chunks.append(current)
    return chunks


# name -> (description, splitter). A splitter takes a section's text and an embedding function.
STRATEGIES: dict[str, tuple[str, Callable]] = {
    "fixed-1100": ("1,100-character windows, 150 overlap, cut anywhere", lambda t, e: fixed_windows(t, 1100, 150)),
    "sentence-512": ("up to 512 characters on sentence boundaries, 64 overlap", lambda t, e: split_text(t, 512, 64)),
    "sentence-1100": ("up to 1,100 characters on sentence boundaries, 150 overlap", lambda t, e: split_text(t, 1100, 150)),
    "sentence-2048": ("up to 2,048 characters on sentence boundaries, 256 overlap", lambda t, e: split_text(t, 2048, 256)),
    "paragraph": ("the paper's paragraphs, merged or split to about 1,100 characters", lambda t, e: paragraphs(t, 1100)),
    "semantic": ("cut where adjacent sentences' embeddings differ most, 300-1,600 characters", lambda t, e: semantic(t, e)),
}


def chunk_sections(paper: Paper, strategy: str, embed: Callable[[list[str]], np.ndarray]) -> list[tuple[str, str]]:
    """(section title, passage text) pairs, the abstract first."""
    split = STRATEGIES[strategy][1]
    pieces = [("Abstract", paper.abstract)]
    for section in paper.sections:
        pieces += [(section.title, chunk) for chunk in split(section.text, embed) if chunk.strip()]
    return pieces
