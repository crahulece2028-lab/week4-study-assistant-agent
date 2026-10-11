"""Task 4.3: run the evaluation set and write the results + trace artifacts.

    python eval/run_eval.py

LangSmith env vars are set *before* `graph` (and therefore langchain) is imported, so that when a
key is present the same run is also traced remotely. Either way the local per-question span records
are written and a handful are rendered to `docs/traces/` for the report.

Metrics are defined in `eval/README.md`. Everything here is measured from the run; nothing is typed
in by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

import config

TRACED_REMOTELY = config.enable_langsmith()

import graph  # noqa: E402
import local_tracer  # noqa: E402

REFUSAL_MARKERS = ("could not find", "not find this", "will not guess", "don't know",
                   "do not know", "not in the provided", "unable to find",
                   "does not contain", "doesn't contain", "not contain", "no information",
                   "cannot answer", "can't answer", "cannot give", "not provided",
                   "unable to answer", "cannot be answered")

# Answers can carry typographic dashes/quotes from the source PDFs or the chat model
# ("quarter-wave" with a non-breaking hyphen); normalise them before substring checks.
_DASHES = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212\u1806"}
_DASHES.update({ord("\u2019"): "'", ord("\u2018"): "'", ord("\u201c"): '"', ord("\u201d"): '"'})

# The chat model often answers with LaTeX ("K_{p}", "g_m"); drop the markup punctuation so a
# keyword like "kp" or "gm" still matches the concept the answer actually contains.
_LATEX = str.maketrans("", "", "\\_{}$")


def normalise(text: str) -> str:
    return text.translate(_DASHES).translate(_LATEX)


def load_testset() -> dict:
    return json.loads((HERE / "testset.json").read_text(encoding="utf-8"))


def evaluate(suffix: str = "") -> dict:
    testset = load_testset()
    # Warm up once so the reported latency is steady-state, not model-load. The cost is real and
    # is reported separately as `first_run_warmup`.
    graph.answer("warm up the embedding model and compile the graph", thread_id="warmup")
    rows = []
    wall = []
    for i, q in enumerate(testset["questions"]):
        t0 = time.perf_counter()
        final = graph.answer(q["question"], thread_id=f"eval-{q['id']}")
        elapsed = (time.perf_counter() - t0) * 1000
        wall.append(elapsed)

        answer = final.get("answer", "") or ""
        sources = final.get("sources", []) or []
        low = normalise(answer).lower()
        route_ok = final.get("route") == q["route"]

        if q.get("in_scope", True) is False:
            behaviour_ok = any(m in low for m in REFUSAL_MARKERS)
            hit = None
            kw = None
        elif q["route"] == "retrieve":
            hit = q["source"] in sources
            kw = any(normalise(k).lower() in low for k in q.get("keywords", []))
            behaviour_ok = hit and kw
        elif q["route"] == "tool":
            hit = None
            kw = any(normalise(k).lower() in low for k in q.get("keywords", []))
            behaviour_ok = kw and final.get("tool_ok", False)
        else:  # direct
            hit = None
            kw = None
            behaviour_ok = final.get("confidence") == "direct"

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "expected_route": q["route"],
            "route": final.get("route"),
            "route_ok": route_ok,
            "confidence": final.get("confidence"),
            "expected_source": q.get("source"),
            "sources": sources,
            "retrieval_hit": hit,
            "keywords": q.get("keywords", []),
            "keyword_hit": kw,
            "in_scope": q.get("in_scope", True),
            "behaviour_ok": behaviour_ok,
            "answer": answer,
            "n_spans": len(final.get("trace", [])),
            "wall_ms": round(elapsed, 1),
            "trace": final.get("trace", []),
        })

    steady = wall[1:] if len(wall) > 1 else wall

    def rate(sel, field):
        vals = [r[field] for r in rows if sel(r) and r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    in_scope_retrieve = lambda r: r["in_scope"] is True and r["expected_route"] == "retrieve"  # noqa: E731

    metrics = {
        "n_questions": len(rows),
        "routing_accuracy": rate(lambda r: True, "route_ok"),
        "retrieval_hit_rate": rate(in_scope_retrieve, "retrieval_hit"),
        "answer_keyword_rate": rate(in_scope_retrieve, "keyword_hit"),
        "tool_success_rate": rate(lambda r: r["expected_route"] == "tool", "behaviour_ok"),
        "direct_ok_rate": rate(lambda r: r["expected_route"] == "direct", "behaviour_ok"),
        "out_of_scope_refusal_rate": rate(lambda r: r["in_scope"] is False, "behaviour_ok"),
        "grounded_confidence_rate": round(
            sum(r["confidence"] == "grounded" for r in rows if in_scope_retrieve(r))
            / max(1, sum(in_scope_retrieve(r) for r in rows)), 4),
        "latency_ms": {
            "first_run_warmup": round(wall[0], 1),
            "mean": round(stats.mean(steady), 1),
            "median": round(stats.median(steady), 1),
            "p95": round(sorted(steady)[max(0, int(0.95 * len(steady)) - 1)], 1),
        },
        "langsmith_traced_remotely": TRACED_REMOTELY,
        "llm_mode": config.llm_mode(),
        "embedding_model": config.EMBED_MODEL,
    }

    (HERE / f"results{suffix}.json").write_text(
        json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")

    with (HERE / f"results{suffix}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "route_ok", "expected_route", "route", "retrieval_hit",
                    "keyword_hit", "behaviour_ok", "n_spans", "wall_ms", "question"])
        for r in rows:
            w.writerow([r["id"], r["route_ok"], r["expected_route"], r["route"],
                        r["retrieval_hit"], r["keyword_hit"], r["behaviour_ok"],
                        r["n_spans"], r["wall_ms"], r["question"]])

    (HERE / f"summary{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "rows": rows}


def render_representative_traces(rows: list[dict], suffix: str = "") -> list[dict]:
    """Four deterministic traces: a grounded answer, a second grounded answer from another doc,
    a tool call, and an honest refusal. The retry-loop trace is produced by
    `improvement_experiment.py`, which runs the same question with the floor disabled."""
    by_id = {r["id"]: r for r in rows}
    picks = [("q01", "eval-q01-grounded"), ("q08", "eval-q08-grounded"),
             ("q12", "eval-q12-tool"), ("q17", "eval-q17-refusal")]
    out = []
    for qid, tag in picks:
        row = by_id.get(qid)
        if row:
            out.append(local_tracer.render(row["trace"], row["question"], tag=tag + suffix))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 4.3 evaluation set.")
    parser.add_argument("--suffix", default="",
                        help="suffix for output files, e.g. '_llm' -> summary_llm.json")
    args = parser.parse_args()
    result = evaluate(args.suffix)
    traces = render_representative_traces(result["rows"], args.suffix)
    m = result["metrics"]
    print(f"questions           {m['n_questions']}")
    print(f"routing accuracy    {m['routing_accuracy']:.0%}")
    print(f"retrieval hit rate  {m['retrieval_hit_rate']:.0%}")
    print(f"answer keyword rate {m['answer_keyword_rate']:.0%}")
    print(f"tool success        {m['tool_success_rate']:.0%}")
    print(f"OOS refusal rate    {m['out_of_scope_refusal_rate']:.0%}")
    print(f"latency (median)    {m['latency_ms']['median']} ms  "
          f"(first run {m['latency_ms']['first_run_warmup']} ms, warm)")
    print(f"langsmith remote    {m['langsmith_traced_remotely']}  (llm: {m['llm_mode']})")
    print("traces:")
    for t in traces:
        print(f"  {t['name']:24s} {t['total_ms']:8.1f} ms -> {t['png']}")


if __name__ == "__main__":
    main()
