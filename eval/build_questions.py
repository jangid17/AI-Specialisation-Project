"""Build the evaluation set from the saved library: python eval/build_questions.py

Three kinds of question, written to eval/questions.json:
- "digest": one per key result whose quote was verified in the paper; the quote is the evidence and the
  digest's claim is the reference answer;
- "passage": the model picks a factual sentence from sampled body passages and writes a question it
  answers; kept only when that sentence is found in the paper word for word;
- "unanswerable": questions the library cannot answer, for testing the relevance gate.

Questions are asked to paraphrase rather than copy, so that keyword overlap does not decide retrieval.
"""

import json
import random

from common import QUESTIONS
from research_assistant import analysis, chunking, config, llm, store

SEED = 7
PASSAGES_PER_PAPER = 4
RULES = ("Write questions a researcher who has not read the paper would ask. Paraphrase: never copy more than "
         "three consecutive words from the evidence, and do not mention the paper's title.")

# Questions no saved paper answers, written by hand so that their answer is known to be absent.
UNANSWERABLE = [
    "What is the recommended dosage of ibuprofen for adults?",
    "Which team won the 2018 FIFA World Cup?",
    "How do I bake sourdough bread at home?",
    "What is the boiling point of ethanol at sea level?",
    "Who painted the ceiling of the Sistine Chapel?",
    "What were the main causes of the French Revolution?",
    "How does photosynthesis convert light into chemical energy?",
    "What is the capital city of Australia?",
    "Which telescope first imaged a black hole's shadow?",
    "How many moons does Jupiter have?",
    "What GPU price did the authors pay per hour on their cloud provider?",
    "Which hospital approved the clinical trial described in the study?",
]


def digest_questions(paper) -> list[dict]:
    digest = paper.digests.get("Focused|Specialist")
    results = [r for r in (digest or {}).get("key_results", []) if analysis.quote_in_paper(r["evidence"], paper)]
    if not results:
        return []
    listing = "\n".join(f'[{i}] claim: {r["claim"]}\n    evidence: "{r["evidence"]}"' for i, r in enumerate(results, 1))
    data = llm.chat_json([{"role": "user", "content":
        f"{RULES}\n\nFor each numbered item, write one question whose answer is the claim and which the evidence "
        f'sentence answers.\nReturn JSON: {{"items": [{{"n": number, "question": "..."}}]}}\n\n{listing}'}])
    out = []
    for item in data.get("items", []):
        n = int(item.get("n", 0))
        if 1 <= n <= len(results) and item.get("question"):
            r = results[n - 1]
            out.append({"kind": "digest", "arxiv_id": paper.arxiv_id, "question": item["question"].strip(),
                        "reference": r["claim"], "evidence": r["evidence"]})
    return out


def passage_questions(paper, rng: random.Random) -> list[dict]:
    body = [(s, t) for s, t in chunking.chunk_sections(paper, "sentence-1100", lambda x: None)[1:]
            if len(t) > 500 and not s.lower().startswith(("reference", "acknowledg"))]
    if not body:
        return []
    picks = sorted(rng.sample(range(len(body)), min(PASSAGES_PER_PAPER, len(body))))
    excerpts = "\n\n".join(f"[{i}] ({body[j][0]})\n{body[j][1]}" for i, j in enumerate(picks, 1))
    data = llm.chat_json([{"role": "user", "content":
        f"{RULES}\n\nFor each numbered excerpt, choose ONE sentence that states a specific fact (a number, a "
        "dataset, a design choice or a finding). Copy that sentence exactly as \"evidence\", write a question it "
        "answers, and a one-sentence answer.\nReturn JSON: "
        f'{{"items": [{{"n": number, "evidence": "...", "question": "...", "answer": "..."}}]}}\n\n{excerpts}'}])
    out = []
    for item in data.get("items", []):
        ev = str(item.get("evidence", "")).strip()
        if item.get("question") and analysis.quote_in_paper(ev, paper) and len(ev.split()) >= 6:
            out.append({"kind": "passage", "arxiv_id": paper.arxiv_id, "question": item["question"].strip(),
                        "reference": str(item.get("answer", "")).strip(), "evidence": ev})
    return out


def main() -> None:
    rng = random.Random(SEED)
    questions = []
    for paper in sorted(store.load_all(), key=lambda p: p.arxiv_id):
        if not paper.has_full_text:
            continue
        found = digest_questions(paper) + passage_questions(paper, rng)
        print(paper.arxiv_id, len(found), flush=True)
        questions += found
    questions += [{"kind": "unanswerable", "arxiv_id": None, "question": q, "reference": "", "evidence": ""}
                  for q in UNANSWERABLE]
    for i, q in enumerate(questions, 1):
        q["id"] = f"q{i:03d}"
    QUESTIONS.write_text(json.dumps({"model": config.CHAT_MODEL, "seed": SEED, "questions": questions}, indent=1))
    counts = {k: sum(q["kind"] == k for q in questions) for k in ("digest", "passage", "unanswerable")}
    print("wrote", len(questions), counts)


if __name__ == "__main__":
    main()
