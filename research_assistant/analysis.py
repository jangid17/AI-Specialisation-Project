"""What the language model is asked to do: digest a paper, answer questions, compare, brief."""

import re
from collections.abc import Callable, Iterator

from research_assistant import config, llm, retrieval
from research_assistant.models import Paper
from research_assistant.retrieval import Passage

AUDIENCES = {
    "Specialist": "a researcher working in this exact area; be precise and skip textbook background",
    "Adjacent field": "a professor from a neighbouring field; define field-specific terms briefly",
    "Graduate student": "a first-year graduate student; explain the key ideas the paper assumes",
}
DEPTHS = ["Focused", "Full paper"]
MAP_GROUP_CHARS = 12000

SYSTEM = ("You help a professor read research papers. Use only the paper text you are given. "
          "Never invent results, numbers, datasets or citations. If the text does not say something, say so.")

DIGEST_SCHEMA = """Return one JSON object with exactly these keys:
"tldr": two sentences on what the paper does and finds.
"problem": the problem and why it matters, 2-3 sentences.
"contributions": list of 2-5 short strings.
"method": how it works, 3-5 sentences.
"key_results": list of 2-5 objects {"claim": the result in one sentence, "evidence": a quote of at most 30 words copied character-for-character from the text that supports it}.
"limitations": list of 2-4 strings; include limitations the authors state and weaknesses you can see in the text, marking the latter with "(reader's note)".
"prerequisites": list of 2-5 concepts a reader needs to know first.
"discussion_questions": list of 3 questions suitable for a reading group or a referee report.
"who_should_read": one sentence."""

_FOCUS_QUERIES = ["proposed method, model architecture and approach",
                  "main experimental results, numbers and comparison with baselines",
                  "limitations, failure cases and future work"]
_CLOSING = re.compile(r"conclusion|discussion|limitation|future work|summary", re.IGNORECASE)


def _structured(prompt: str, required: tuple[str, ...], on_wait) -> dict:
    """Ask for a JSON object and insist on the keys the page needs.

    A model sometimes wraps the object in another one or stops after the first few keys. The wrapper is
    removed; an incomplete answer is asked for once more and then reported, never shown as if empty.
    """
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    for _ in range(2):
        data = llm.chat_json(messages, on_wait=on_wait)
        if len(data) == 1 and isinstance(next(iter(data.values())), dict):
            data = next(iter(data.values()))
        if all(data.get(key) for key in required):
            return data
    missing = ", ".join(key for key in required if not data.get(key))
    raise llm.LLMError(f"The model's answer was incomplete (missing: {missing}). Try again.")


def digest_key(depth: str, audience: str) -> str:
    return f"{depth}|{audience}"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def focused_context(paper: Paper) -> str:
    """Abstract, introduction, closing sections, then the passages nearest to method/results/limits."""
    parts = [f"ABSTRACT\n{paper.abstract}"]
    if not paper.has_full_text:
        return parts[0]
    intro = paper.sections[0]
    parts.append(f"{intro.title.upper()}\n{_clip(intro.text, 3000)}")
    closing = [s for s in paper.sections[1:] if _CLOSING.search(s.title)]
    for s in closing[-2:]:
        parts.append(f"{s.title.upper()}\n{_clip(s.text, 2000)}")
    used = sum(len(p) for p in parts)
    seen = set()
    for query in _FOCUS_QUERIES:
        for passage in retrieval.search(query, [paper.arxiv_id], top_k=4):
            if passage.text in seen or passage.section in ("Abstract", intro.title):
                continue
            if used + len(passage.text) > config.CONTEXT_CHAR_BUDGET:
                break
            seen.add(passage.text)
            used += len(passage.text)
            parts.append(f"EXCERPT FROM {passage.section.upper()}\n{passage.text}")
    return "\n\n".join(parts)


def _section_groups(paper: Paper) -> list[str]:
    """The whole paper as consecutive blocks of at most MAP_GROUP_CHARS characters."""
    groups, current = [], ""
    for s in paper.sections:
        block = f"{s.title.upper()}\n{s.text}"
        while len(block) > MAP_GROUP_CHARS:
            if current:
                groups.append(current)
                current = ""
            groups.append(block[:MAP_GROUP_CHARS])
            block = block[MAP_GROUP_CHARS:]
        if current and len(current) + len(block) > MAP_GROUP_CHARS:
            groups.append(current)
            current = ""
        current = f"{current}\n\n{block}".strip()
    if current:
        groups.append(current)
    return groups


def full_paper_calls(paper: Paper) -> int:
    """How many model calls a full-paper digest needs (one per block, plus the final digest)."""
    return len(_section_groups(paper)) + 1


def _notes_for(block: str, on_wait) -> str:
    prompt = ("Take reading notes on this part of a paper, in at most 220 words: the points made, how the "
              "method works, every quantitative result with its number, and any stated limitation. "
              "Copy up to three short sentences that state key results exactly as written, in double quotes.\n\n"
              + block)
    return llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    max_tokens=900, on_wait=on_wait)


def make_digest(paper: Paper, depth: str = "Focused", audience: str = "Specialist",
                on_progress: Callable[[int, int], None] | None = None,
                on_wait: Callable[[float], None] | None = None) -> dict:
    """A structured digest. `basis` records what text it was written from."""
    if depth == "Full paper" and paper.has_full_text:
        groups = _section_groups(paper)
        notes = []
        for i, block in enumerate(groups):
            if on_progress:
                on_progress(i, len(groups) + 1)
            notes.append(_notes_for(block, on_wait))
        if on_progress:
            on_progress(len(groups), len(groups) + 1)
        context = f"ABSTRACT\n{paper.abstract}\n\nREADING NOTES ON THE WHOLE PAPER\n" + "\n\n".join(notes)
        basis = "full paper"
    else:
        context = focused_context(paper)
        basis = "abstract, introduction, conclusion and retrieved excerpts" if paper.has_full_text else "abstract only"
    prompt = (f"Paper: {paper.title}\nWrite for {AUDIENCES[audience]}.\n\n{DIGEST_SCHEMA}\n\n"
              f"--- PAPER TEXT ---\n{context}")
    digest = normalise_digest(_structured(prompt, ("tldr", "problem", "method", "key_results"), on_wait))
    for result in digest["key_results"]:
        result["verified"] = quote_in_paper(result["evidence"], paper)
    digest["basis"] = basis
    digest["model"] = config.CHAT_MODEL
    return digest


_TEXT_FIELDS = ("tldr", "problem", "method", "who_should_read")
_LIST_FIELDS = ("contributions", "limitations", "prerequisites", "discussion_questions")


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    return [value] if value else []


def normalise_digest(raw: dict) -> dict:
    """Coerce model output to the digest shape, so everything downstream can rely on it.

    Models occasionally return null for a field, a bare string where a list was asked for, or a
    result without its quote.
    """
    digest = {k: str(raw.get(k) or "") for k in _TEXT_FIELDS}
    digest.update({k: [str(item) for item in _as_list(raw.get(k))] for k in _LIST_FIELDS})
    digest["key_results"] = [
        {"claim": str(r.get("claim") or ""), "evidence": str(r.get("evidence") or "")}
        if isinstance(r, dict) else {"claim": str(r), "evidence": ""}
        for r in _as_list(raw.get("key_results"))]
    return digest


def _normalise(text: str) -> str:
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text.lower())  # drop full stops, keep decimal points
    return re.sub(r"[^a-z0-9%.]+", " ", text).strip()


def quote_in_paper(quote: str, paper: Paper) -> bool:
    """True when the quote appears in the paper's text, ignoring case, spacing and punctuation.

    A quote with an ellipsis passes when every fragment of 15+ characters appears.
    """
    haystack = _normalise(paper.abstract + " " + paper.full_text())
    fragments = [_normalise(f) for f in re.split(r"\.\.\.|…", quote)]
    fragments = [f for f in fragments if len(f) >= 15]
    return bool(fragments) and all(f in haystack for f in fragments)


def answer_messages(question: str, passages: list[Passage], titles: dict[str, str],
                    history: list[dict]) -> list[dict]:
    """Prompt for a grounded answer that cites passages as [1], [2], ..."""
    sources = "\n\n".join(f"[{i}] {titles.get(p.arxiv_id, p.arxiv_id)} — {p.section}\n{p.text}"
                          for i, p in enumerate(passages, start=1))
    instructions = ("Answer the question from the numbered passages below. Cite the passages you rely on "
                    "as [1], [2] after each claim. If the passages do not contain the answer, say that "
                    "plainly and do not answer from general knowledge. Be concise.")
    messages = [{"role": "system", "content": f"{SYSTEM} {instructions}"}]
    messages += [{"role": m["role"], "content": _clip(m["content"], 600)} for m in history[-4:]]
    messages.append({"role": "user", "content": f"PASSAGES\n{sources}\n\nQUESTION: {question}"})
    return messages


def answer_stream(question: str, passages: list[Passage], titles: dict[str, str], history: list[dict],
                  on_wait: Callable[[float], None] | None = None) -> Iterator[str]:
    return llm.stream(answer_messages(question, passages, titles, history), max_tokens=1200, on_wait=on_wait)


def _paper_brief(paper: Paper, limit: int) -> str:
    """The best short description on hand: the saved digest if there is one, else the abstract."""
    digest = next(iter(paper.digests.values()), None)
    if digest:
        results = "; ".join(r["claim"] for r in digest["key_results"])
        text = (f"Problem: {digest['problem']} Method: {digest['method']} "
                f"Results: {results} Limitations: {'; '.join(digest['limitations'])}")
    else:
        text = paper.abstract
    return _clip(text, limit)


def compare(papers: list[Paper], on_wait: Callable[[float], None] | None = None) -> dict:
    """A side-by-side comparison and a short synthesis that cites the papers as [n]."""
    limit = min(2400, config.CONTEXT_CHAR_BUDGET // len(papers))
    listing = "\n\n".join(f"[{i}] {p.title} ({p.short_authors}, {p.year})\n{_paper_brief(p, limit)}"
                          for i, p in enumerate(papers, start=1))
    prompt = (f"""Compare these papers for a professor preparing a literature review.

Return one JSON object with exactly these keys:
"rows": one object per paper, in order: {{"n": its number, "problem": ..., "approach": ..., "evidence": the data or experiments used, "key_result": ..., "limitation": ...}}; each value at most 25 words, "not stated" when the text does not say.
"agreements": list of 1-3 points where the papers agree or build on each other, citing them as [n].
"differences": list of 1-3 points where they differ or conflict, citing them as [n].
"open_questions": list of 2-3 questions none of them settles.
"synthesis": one paragraph of about 130 words, written as related-work prose, citing each paper as [n].

PAPERS
{listing}""")
    return _structured(prompt, ("rows", "agreements", "differences", "open_questions", "synthesis"), on_wait)


def brief_new_papers(papers: list[Paper], interests: str,
                     on_wait: Callable[[float], None] | None = None) -> dict:
    """Triage a batch of new papers against the reader's interests, from their abstracts."""
    limit = min(900, config.CONTEXT_CHAR_BUDGET // len(papers))
    listing = "\n\n".join(f"[{i}] {p.title}\n{_clip(p.abstract, limit)}" for i, p in enumerate(papers, start=1))
    prompt = (f"""A professor describes their research interests as: "{interests or 'not given'}".

For the new papers below, return one JSON object with exactly these keys:
"overview": two or three sentences on what this batch is about as a whole.
"items": one object per paper, in order: {{"n": its number, "priority": "high", "medium" or "low" for this reader, "why": one sentence on what is new in it and why it does or does not matter to them}}.

Judge only from the abstracts.

PAPERS
{listing}""")
    return _structured(prompt, ("overview", "items"), on_wait)
