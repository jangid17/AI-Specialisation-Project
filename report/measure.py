"""Produce the numbers quoted in report.tex from the papers currently in the library.

    python report/measure.py            # measure, then write the LaTeX
    python report/measure.py --tex      # only rewrite measurements.tex from measurements.json

Writes report/measurements.json and report/measurements.tex (macros and table rows that report.tex
inputs). It makes one Groq call per paper that has no focused digest yet, so on the free tier it can
take several minutes.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import analysis, config, metrics, retrieval, store  # noqa: E402

KEY = analysis.digest_key("Focused", "Specialist")
JSON_FILE = Path(__file__).with_name("measurements.json")


def main() -> None:
    papers = sorted(store.load_all(), key=lambda p: p.arxiv_id)
    out_file = JSON_FILE
    # Timings exist only for digests generated in that run, so earlier ones are carried over.
    previous = json.loads(out_file.read_text()) if out_file.exists() else {"papers": []}
    timings = {r["arxiv_id"]: r["digest_seconds"] for r in previous["papers"]}
    rows, waits = [], []
    for paper in papers:
        seconds = None
        if KEY not in paper.digests:
            start = time.time()
            paper.digests[KEY] = analysis.make_digest(paper, on_wait=waits.append)
            seconds = round(time.time() - start, 1)
            store.save(paper)
        results = paper.digests[KEY]["key_results"]
        for r in results:  # the check is deterministic, so re-run it rather than trust the stored flag
            r["verified"] = analysis.quote_in_paper(r["evidence"], paper)
        store.save(paper)
        rows.append({
            "arxiv_id": paper.arxiv_id, "title": paper.title, "year": paper.year,
            "sections": len(paper.sections), "words": len(paper.full_text().split()),
            "passages": len(retrieval.chunk_paper(paper)), "references": paper.reference_count,
            "citations": paper.citation_count, "quotes": len(results),
            "quotes_verified": sum(bool(r.get("verified")) for r in results),
            "digest_seconds": seconds if seconds is not None else timings.get(paper.arxiv_id),
        })
        print(rows[-1], flush=True)
    graph = metrics.coauthor_graph(papers)
    out = {
        "model": config.CHAT_MODEL, "embedding_model": config.EMBEDDING_MODEL, "papers": rows,
        "rate_limit_waits": len(waits) or previous.get("rate_limit_waits", 0),
        "rate_limit_wait_seconds": round(sum(waits), 1) or previous.get("rate_limit_wait_seconds", 0),
        "network": metrics.network_stats(graph),
        "totals": {k: sum(r[k] for r in rows) for k in ("sections", "words", "passages", "quotes", "quotes_verified")},
    }
    out_file.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["totals"]), "waits:", out["rate_limit_waits"], out["rate_limit_wait_seconds"])
    write_tex(out)


def write_tex(out: dict) -> None:
    """Macros (\\mPapers, ...) and the per-paper table body, so the PDF cannot drift from the JSON."""
    rows, totals, net = out["papers"], out["totals"], out["network"]
    times = sorted(r["digest_seconds"] for r in rows if r["digest_seconds"] is not None)
    median = (times[len(times) // 2] + times[(len(times) - 1) // 2]) / 2
    cited = [r for r in rows if r["citations"] is not None]
    macros = {
        "Papers": len(rows), "Sections": totals["sections"], "Words": f"{totals['words']:,}",
        "Passages": totals["passages"], "Quotes": totals["quotes"], "QuotesVerified": totals["quotes_verified"],
        "QuotesPct": round(100 * totals["quotes_verified"] / totals["quotes"]),
        "Timed": len(times), "TimeMin": times[0], "TimeMax": times[-1], "TimeMedian": f"{median:.1f}",
        "TimeMean": f"{sum(times) / len(times):.1f}", "Waits": out["rate_limit_waits"],
        "WaitSeconds": round(out["rate_limit_wait_seconds"]), "Cited": len(cited),
        "Authors": net["authors"], "Pairs": net["collaborations"], "Density": f"{net['density']:.3f}",
        "Groups": net["groups"], "LargestGroup": net["largest_group"], "Recurring": net["recurring_authors"],
    }
    lines = [f"\\newcommand{{\\m{k}}}{{{v}}}" for k, v in macros.items()]
    lines.append("\\newcommand{\\mPaperRows}{%")
    for r in rows:
        title = r["title"] if len(r["title"]) <= 46 else r["title"][:44].rsplit(" ", 1)[0] + "\\,\\ldots"
        title = title.replace("&", "\\&").replace("%", "\\%")
        cells = [r["arxiv_id"], title, r["year"], r["sections"], f"{r['words']:,}", r["passages"],
                 r["references"], f"{r['citations']:,}" if r["citations"] is not None else "--",
                 f"{r['quotes_verified']}/{r['quotes']}",
                 r["digest_seconds"] if r["digest_seconds"] is not None else "--"]
        lines.append(" & ".join(str(c) for c in cells) + " \\\\")
    lines.append("}")
    JSON_FILE.with_suffix(".tex").write_text("% Written by report/measure.py. Do not edit.\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    if "--tex" in sys.argv:
        write_tex(json.loads(JSON_FILE.read_text()))
    else:
        main()
