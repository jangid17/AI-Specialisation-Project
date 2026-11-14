"""End-to-end answer quality with Ragas: the app's pipeline before and after the retrieval changes.

    .venv-eval/bin/python eval/ragas_eval.py [--n 16] [--judge qwen/qwen3.8-27b]

For a fixed sample of answerable questions (half from digests, half from passages), each pipeline retrieves
passages from the paper the question is about and gpt-oss-120b answers with the app's prompt. A second
model then judges with three Ragas metrics:

  faithfulness       share of the answer's claims supported by the retrieved passages
  answer_relevancy   how well the answer addresses the question (questions regenerated from the answer,
                     compared with the original by embedding similarity)
  context_recall     share of the reference answer's claims supported by the retrieved passages

Groq's free tier caps tokens per minute and per day, so every answer and score is cached in
eval/.cache/ragas.json and a rerun picks up where the last one stopped.
"""

import argparse
import asyncio
import json
import random

import numpy as np
from ablation import build
from common import EVAL_DIR, bootstrap_ci, load_questions, mean, save_result
from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

from research_assistant import analysis, config, guardrails, llm, retrieval, store

CACHE = EVAL_DIR / ".cache" / "ragas.json"
PIPELINES = {  # name: (chunking, embedding model, method, passages sent to the model); both ~6,000 characters
    "before": ("sentence-1100", "BAAI/bge-small-en-v1.5", "dense", 6),
    "after": (config.CHUNKING, config.EMBEDDING_MODEL, config.RETRIEVAL_METHOD, config.ANSWER_PASSAGES),
}


class FastEmbed(BaseRagasEmbedding):
    """The app's own embedding model, for answer relevancy."""

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return retrieval.embed([text])[0].tolist()

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)


def sample(n: int) -> list[dict]:
    qs = load_questions()
    rng = random.Random(7)
    pick = lambda kind: rng.sample([q for q in qs if q["kind"] == kind], n // 2)
    return pick("digest") + pick("passage")


def answer(questions: list[dict], cache: dict) -> None:
    titles = {p.arxiv_id: p.title for p in store.load_all()}
    # Input guardrails are measured in guardrails_eval.py; here they would only spend rate limit.
    guardrails.check_input = lambda q: (guardrails.clean_question(q), [guardrails.validate(q)])
    for name, (strategy, model, method, top_k) in PIPELINES.items():
        index, _ = build(strategy, model)
        retrieval.search = lambda q, ids, top_k=6, method=None, index=index: index.search(q, ids, top_k, method)
        for q in questions:
            key = f"{name}|{q['id']}"
            if key in cache:
                continue
            got = analysis.answer_question(q["question"], [q["arxiv_id"]], titles, top_k=top_k,
                                           method=method, on_wait=lambda s: print(f"  waiting {s:.0f}s", flush=True))
            cache[key] = {"answer": got["answer"], "contexts": [p.text for p in got["passages"]],
                          "refused": got["refused"], "checks": [c.to_dict() for c in got["checks"]]}
            CACHE.write_text(json.dumps(cache))
            print("answered", key, flush=True)


async def judge(questions: list[dict], cache: dict, model: str) -> None:
    client = AsyncOpenAI(api_key=config.groq_api_key(), base_url=config.GROQ_BASE_URL, max_retries=8)
    judge_llm = llm_factory(model, provider="openai", client=client, max_tokens=4000)
    metrics = {"faithfulness": Faithfulness(llm=judge_llm),
               "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=FastEmbed(), strictness=2),
               "context_recall": ContextRecall(llm=judge_llm)}
    for name in PIPELINES:
        for q in questions:
            row = cache[f"{name}|{q['id']}"]
            for metric, m in metrics.items():
                if metric in row:
                    continue
                args = {"user_input": q["question"]}
                if metric != "context_recall":
                    args["response"] = row["answer"]
                if metric != "answer_relevancy":
                    args["retrieved_contexts"] = row["contexts"] or ["(no passages)"]
                if metric == "context_recall":
                    args["reference"] = q["reference"]
                for attempt in range(3):
                    try:
                        row[metric] = float((await m.ascore(**args)).value)
                        break
                    except Exception as e:  # rate limits and malformed judge output; retried, then left unscored
                        print(f"  {name}|{q['id']} {metric}: {type(e).__name__} {str(e)[:120]}", flush=True)
                        await asyncio.sleep(20 * (attempt + 1))
                CACHE.write_text(json.dumps(cache))
            print("judged", name, q["id"], {k: row.get(k) for k in metrics}, flush=True)


def summarise(questions: list[dict], cache: dict, judge_model: str) -> dict:
    out = {"answer_model": config.CHAT_MODEL, "judge_model": judge_model, "questions": len(questions),
           "pipelines": {k: dict(zip(("chunking", "embedding_model", "method", "passages"), v)) for k, v in PIPELINES.items()},
           "results": {}}
    for name in PIPELINES:
        rows = [cache[f"{name}|{q['id']}"] for q in questions]
        res = {}
        for metric in ("faithfulness", "answer_relevancy", "context_recall"):
            values = [r[metric] for r in rows if r.get(metric) is not None and not np.isnan(r[metric])]
            res[metric] = {"mean": mean(values), "ci95": bootstrap_ci(values), "scored": len(values)}
        res["refused"] = sum(r["refused"] for r in rows)
        res["output_checks_flagged"] = {c: sum(any(x["name"] == c and not x["passed"] for x in r["checks"]) for r in rows)
                                        for c in ("Citations valid", "Sentences cited", "Numbers in passages")}
        out["results"][name] = res
        print(name, {k: v["mean"] if isinstance(v, dict) and "mean" in v else v for k, v in res.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--judge", default="qwen/qwen3.8-27b")  # its own daily budget, apart from the answering model
    a = ap.parse_args()
    if not llm.is_configured():
        raise SystemExit("Set GROQ_API_KEY in .env first.")
    CACHE.parent.mkdir(exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    questions = sample(a.n)
    answer(questions, cache)
    asyncio.run(judge(questions, cache, a.judge))
    print("saved", save_result("ragas", summarise(questions, cache, a.judge)))


if __name__ == "__main__":
    main()
