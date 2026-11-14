"""Guardrails around the question-answering pipeline.

Input   validate the question; detect prompt attacks with Llama Prompt Guard 2 (a regex screen is the
        fallback when it cannot be reached); classify it against a short safety policy with
        gpt-oss-safeguard.
Context refuse without calling the model when no passage is relevant enough (re-ranker score gate);
        flag retrieved passages that contain instruction-like text (indirect prompt injection). Such
        passages are kept, because papers about prompt injection quote attacks, but the answer prompt
        marks all paper text as data that must not be obeyed.
Output  check that citations point at passages that exist, that sentences are cited, that numbers in the
        answer appear in the passages, and that the answer does not leak the prompt or a key.

Every check returns a Check; `blocking` ones stop the pipeline. eval/guardrails_eval.py measures them.
"""

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from research_assistant import config, llm

REFUSAL = "The saved papers don't contain an answer to this question."
INPUT_CHECKS = ("Valid question", "Prompt-injection check", "Safety policy")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocking: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# Input ----------------------------------------------------------------------------------------

_INJECTION = re.compile(
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}"
    r"\b(instructions?|prompts?|rules|directions|context)\b"
    r"|\b(system|developer)\s+(prompt|message|instructions?)\b"
    r"|\b(reveal|print|show|repeat|output)\b.{0,30}\b(prompt|instructions|api[\s_-]?key)\b"
    r"|\byou\s+are\s+now\b|\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|evil)\b|\bjailbreak|\bDAN\b"
    r"|\bdeveloper\s+mode\b|<\|im_start\|>|<\|system\|>|\[INST\]|###\s*(system|instruction)",
    re.IGNORECASE | re.DOTALL)


def looks_like_injection(text: str) -> bool:
    """A regex screen for common prompt-attack phrasing. Cheap, but easy to evade; used as a fallback."""
    return bool(_INJECTION.search(text))


def clean_question(text: str) -> str:
    """Remove control and invisible formatting characters (used to hide instructions) and squeeze spaces."""
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t")
    return re.sub(r"\s+", " ", text).strip()


def validate(text: str) -> Check:
    if not text:
        return Check("Valid question", False, "The question is empty.", blocking=True)
    if len(text) > config.MAX_QUESTION_CHARS:
        return Check("Valid question", False, f"Questions are limited to {config.MAX_QUESTION_CHARS:,} characters.",
                     blocking=True)
    return Check("Valid question", True, f"{len(text)} characters")


def injection_check(text: str) -> Check:
    try:
        score = float(llm.chat([{"role": "user", "content": text[:2000]}], max_tokens=8,
                               model=config.PROMPT_GUARD_MODEL).strip())
    except (llm.LLMError, ValueError):
        hit = looks_like_injection(text)
        return Check("Prompt-injection check", not hit,
                     "Prompt Guard unavailable; pattern screen " + ("matched" if hit else "found nothing"), blocking=hit)
    hit = score >= config.INJECTION_THRESHOLD
    return Check("Prompt-injection check", not hit, f"Prompt Guard 2 attack probability {score:.3f}", blocking=hit)


SAFETY_POLICY = """You classify messages sent to a research-paper reading assistant used by academics.
VIOLATION (1) if the message asks for: instructions to make weapons, explosives, or chemical or biological
agents capable of mass harm; malware or intrusion into systems the user does not own; content sexualising
minors; encouragement of self-harm; harassment or hate against people for a protected attribute.
SAFE (0): everything else, including research questions that discuss these topics academically, such as
the security of language models or the evaluation of toxicity classifiers.
Answer with JSON only: {"violation": 0 or 1, "category": "<short label or none>"}"""


def safety_check(text: str) -> Check:
    try:
        verdict = llm.parse_json(llm.chat([{"role": "system", "content": SAFETY_POLICY},
                                           {"role": "user", "content": text[:2000]}],
                                          max_tokens=300, json_mode=True, model=config.SAFETY_MODEL))
    except llm.LLMError as e:
        # The policy targets requests this tool has no reason to receive, so an outage does not block reading.
        return Check("Safety policy", True, f"Safety model unavailable ({str(e)[:60]}); not checked")
    if int(verdict.get("violation", 0)) == 1:
        return Check("Safety policy", False, f"Outside the acceptable-use policy: {verdict.get('category', 'unsafe')}",
                     blocking=True)
    return Check("Safety policy", True, "gpt-oss-safeguard: no policy violation")


def check_input(text: str) -> tuple[str, list[Check]]:
    """The cleaned question and the input checks, run in parallel (two independent API calls)."""
    text = clean_question(text)
    first = validate(text)
    if first.blocking:
        return text, [first]
    with ThreadPoolExecutor(max_workers=2) as pool:
        injection, safety = pool.submit(injection_check, text), pool.submit(safety_check, text)
        return text, [first, injection.result(), safety.result()]


# Context --------------------------------------------------------------------------------------

def relevance_gate(scores: list[float], threshold: float | None = None) -> Check:
    threshold = config.RELEVANCE_THRESHOLD if threshold is None else threshold
    best = max(scores, default=float("-inf"))
    if best < threshold:
        return Check("Relevant passages", False, f"Best re-ranker score {best:.2f} is below {threshold:.2f}; "
                     "the model was not asked", blocking=True)
    return Check("Relevant passages", True, f"Best re-ranker score {best:.2f}")


def scan_passages(texts: list[str]) -> Check:
    flagged = [i for i, t in enumerate(texts, start=1) if looks_like_injection(t)]
    if flagged:
        return Check("Passages screened", False, "Instruction-like text in passage(s) "
                     + ", ".join(f"[{i}]" for i in flagged) + "; treated as data, not instructions")
    return Check("Passages screened", True, "No instruction-like text in the passages")


# Output ---------------------------------------------------------------------------------------

_CITATION = re.compile(r"[\[【]\s*(\d+(?:\s*[,–-]\s*\d+)*)\s*[\]】]")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*\s*%?")
_NOT_FOUND = re.compile(r"\b(do(?:es)? not|don't|doesn't|cannot|can't|no)\b.{0,40}\b(contain|mention|say|state|"
                        r"provide|answer|information|discuss|report|specif|describe|include|identify|determine|given)", re.IGNORECASE)


def cited_numbers(answer: str) -> set[int]:
    found = set()
    for group in _CITATION.findall(answer):
        for part in re.split(r"\s*,\s*", group):
            bounds = [int(x) for x in re.split(r"\s*[–-]\s*", part)]
            found.update(range(bounds[0], bounds[-1] + 1))
    return found


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 4 and not s.lstrip().startswith("#")]


def check_answer(answer: str, passages: list[str]) -> list[Check]:
    checks = []
    cited = cited_numbers(answer)
    invalid = sorted(n for n in cited if not 1 <= n <= len(passages))
    checks.append(Check("Citations valid", not invalid,
                        f"cites {', '.join(f'[{n}]' for n in invalid)} but only {len(passages)} passages were given"
                        if invalid else f"{len(cited)} distinct passage(s) cited, all exist"))
    sentences = _sentences(answer)
    claims = [s for s in sentences if not _NOT_FOUND.search(s)]
    if not cited and len(claims) < len(sentences):
        claims = []  # an answer that says the passages lack the answer, and cites nothing, is a refusal
    uncited = [s for s in claims if not _CITATION.search(s)]
    checks.append(Check("Sentences cited", not uncited,
                        f"{len(claims) - len(uncited)} of {len(claims)} factual sentences carry a citation"))
    source = " ".join(passages).replace(",", "")
    numbers = {n.strip().replace(",", "") for n in _NUMBER.findall(_CITATION.sub("", answer))}
    numbers = {n for n in numbers if len(n.rstrip("%").replace(".", "")) >= 2}  # ignore single digits
    missing = sorted(n for n in numbers if n.rstrip("% ").strip() not in source)
    checks.append(Check("Numbers in passages", not missing, "not found in the passages: " + ", ".join(missing)
                        if missing else f"{len(numbers)} number(s), all found in the passages"))
    key = config.groq_api_key() or ""
    leaked = (key and key in answer) or re.search(r"\bgsk_[A-Za-z0-9]{20,}", answer) or "Use only the paper text you are given" in answer
    checks.append(Check("No prompt or key leaked", not leaked, "leak detected; answer withheld" if leaked else "clean",
                        blocking=bool(leaked)))
    return checks
