"""Shared by the evaluation scripts: paths, the question set, and the relevance judgement."""

import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_assistant import analysis  # noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS = EVAL_DIR / "results"
QUESTIONS = EVAL_DIR / "questions.json"
APP_CONTEXT_CHARS = 6 * 1100  # what the app sends the model per question: six passages of the default size


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS.read_text())["questions"]


def save_result(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=1))
    return path


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def relevant(passage: str, evidence: str, threshold: float = 0.5) -> bool:
    """True when the passage contains at least half of the evidence sentence.

    Measured as the share of the evidence's word 5-grams that occur in the passage, after the same
    normalisation as the quote check. A sentence split across two passages counts in whichever holds
    most of it, and paraphrase never counts.
    """
    ev = analysis._normalise(evidence).split()
    n = min(5, len(ev))
    grams = _ngrams(ev, n)
    return bool(grams) and len(grams & _ngrams(analysis._normalise(passage).split(), n)) / len(grams) >= threshold


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = rng.choice(data, size=(n, len(data)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0
