"""Library statistics. Everything here is computed from the stored papers; nothing is estimated by the LLM."""

from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

from research_assistant.models import Paper

# Words that appear in almost any abstract and say nothing about its subject.
ACADEMIC_WORDS = {
    "paper", "propose", "proposed", "proposes", "present", "presents", "introduce", "introduces", "approach",
    "approaches", "method", "methods", "model", "models", "task", "tasks", "result", "results", "show", "shows",
    "demonstrate", "demonstrates", "work", "works", "based", "using", "used", "use", "new", "novel", "large",
    "state-of-the-art", "performance", "existing", "recent", "recently", "study", "studies", "problem",
    "problems", "different", "various", "achieve", "achieves", "significantly", "experiments", "framework",
    "provide", "provides", "improve", "improves", "improved", "high", "better", "best", "art", "state",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | ACADEMIC_WORDS)

MAX_TOPICS = 4  # the chart tells topics apart by colour and shape; more than four stops being readable
MIN_PAPERS_FOR_TOPICS = 6


def submissions_over_time(papers: list[Paper]) -> tuple[pd.DataFrame, str]:
    """Papers per month of first submission, or per year when the library spans more than three years.

    Periods with no papers are included as zeros so gaps are visible.
    """
    dates = pd.to_datetime([p.published for p in papers])
    by_year = (dates.max() - dates.min()).days > 3 * 365
    freq, unit = ("Y", "year") if by_year else ("M", "month")
    periods = dates.to_period(freq)
    counts = periods.value_counts().reindex(pd.period_range(periods.min(), periods.max(), freq=freq), fill_value=0)
    labels = counts.index.strftime("%Y" if by_year else "%Y-%m")
    return pd.DataFrame({"period": labels, "papers": counts.to_numpy()}), unit


def category_counts(papers: list[Paper]) -> pd.DataFrame:
    counts = Counter(p.primary_category for p in papers)
    return pd.DataFrame(counts.most_common(), columns=["category", "papers"])


def coauthor_graph(papers: list[Paper]) -> nx.Graph:
    """Authors as nodes (attribute `papers`), joint papers as edges (attribute `weight`).

    Authors are matched by the exact name string arXiv gives, so two people with the same name merge
    and one person with two spellings splits.
    """
    graph = nx.Graph()
    for paper in papers:
        authors = list(dict.fromkeys(paper.authors))
        for author in authors:
            if author not in graph:
                graph.add_node(author, papers=0)
            graph.nodes[author]["papers"] += 1
        for a, b in combinations(authors, 2):
            weight = graph.edges[a, b]["weight"] + 1 if graph.has_edge(a, b) else 1
            graph.add_edge(a, b, weight=weight)
    return graph


def network_stats(graph: nx.Graph) -> dict:
    n = graph.number_of_nodes()
    components = list(nx.connected_components(graph)) if n else []
    return {
        "authors": n,
        "collaborations": graph.number_of_edges(),
        "density": nx.density(graph) if n > 1 else 0.0,
        "groups": len(components),
        "largest_group": max((len(c) for c in components), default=0),
        "recurring_authors": sum(1 for _, d in graph.nodes(data=True) if d["papers"] > 1),
    }


def _documents(papers: list[Paper]) -> list[str]:
    return [f"{p.title}. {p.abstract}" for p in papers]


def _stem(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _covered(term: str, chosen: list[str]) -> bool:
    """True when a chosen term already says this: a word inside a chosen phrase (or the reverse),
    or a plural of a chosen word. Keeps "language" out after "language models"."""
    words = {_stem(w) for w in term.split()}
    return any(words <= {_stem(w) for w in c.split()} or {_stem(w) for w in c.split()} <= words for c in chosen)


def distinctive_terms(papers: list[Paper], top: int = 15) -> pd.DataFrame:
    """Terms that characterise the library: chosen by mean TF-IDF weight over titles and abstracts,
    reported with the number of papers that contain each one."""
    if len(papers) < 2:
        return pd.DataFrame(columns=["term", "papers"])
    # Terms in more than 85% of the papers describe the whole library and tell its papers apart no better
    # than a stop word, so they are left out once the library is big enough for that to mean something.
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 max_df=0.85 if len(papers) >= 8 else 1.0,
                                 token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    try:
        matrix = vectorizer.fit_transform(_documents(papers))
    except ValueError:  # no term appears in two papers
        return pd.DataFrame(columns=["term", "papers"])
    terms = vectorizer.get_feature_names_out()
    weight = np.asarray(matrix.mean(axis=0)).ravel()
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
    chosen = []
    for i in np.argsort(-weight):
        if _covered(terms[i], chosen):
            continue
        chosen.append(terms[i])
        if len(chosen) == top:
            break
    index = {t: i for i, t in enumerate(terms)}
    frame = pd.DataFrame({"term": chosen, "papers": [int(doc_freq[index[t]]) for t in chosen]})
    return frame.sort_values("papers", ascending=False, ignore_index=True)


def topic_map(papers: list[Paper], vectors: np.ndarray) -> dict | None:
    """Group papers by abstract embedding and project them to two dimensions.

    The number of groups (2 to MAX_TOPICS) is the one with the best silhouette score. Returns the
    plotted frame, the share of variance the two axes keep, and the silhouette score, so the chart can
    say how much to trust it. None when the library is too small.
    """
    n = len(papers)
    if n < MIN_PAPERS_FOR_TOPICS:
        return None
    best = None
    for k in range(2, min(MAX_TOPICS, n // 3) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vectors)
        score = silhouette_score(vectors, labels, metric="cosine")
        if best is None or score > best[0]:
            best = (score, labels)
    score, labels = best

    tfidf = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    matrix = tfidf.fit_transform(_documents(papers)).toarray()
    terms = tfidf.get_feature_names_out()
    names = {}
    for cluster in sorted(set(labels)):
        inside = labels == cluster
        contrast = matrix[inside].mean(axis=0) - matrix[~inside].mean(axis=0)
        picked = []
        for i in np.argsort(-contrast):
            if not _covered(terms[i], picked):
                picked.append(terms[i])
            if len(picked) == 3:
                break
        names[cluster] = ", ".join(picked)

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(vectors)
    frame = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "title": [p.title for p in papers],
        "authors": [p.short_authors for p in papers],
        "year": [p.year for p in papers],
        "topic": [names[c] for c in labels],
    })
    return {"frame": frame, "variance_kept": float(pca.explained_variance_ratio_.sum()),
            "silhouette": float(score), "topics": len(names)}


def reading_progress(papers: list[Paper]) -> dict:
    return dict(Counter(p.status for p in papers))
