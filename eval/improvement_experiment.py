"""Task 4.3, second half: two controlled experiments behind the "improvements" section.

Experiment A - chunking. Re-index the same corpus at three (size, overlap) settings and measure
retrieval hit@k on the in-scope questions. Reports chunk count too, because the trade is
"more, smaller chunks" vs "fewer, larger ones", not a free win.

Experiment B - retry policy. The out-of-scope questions are run with the low-coverage floor on and
off. With the floor off, the grader fails, the query is rewritten, retrieval runs again and fails
the same way - twice - before the agent gives up. The floor detects "no lexical overlap at all"
and skips straight to a low-confidence answer. This is the failure that came out of the first
evaluation run; the experiment puts a number on the fix.

    python eval/improvement_experiment.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_community.vectorstores.utils import DistanceStrategy  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
import ingest  # noqa: E402
import local_tracer  # noqa: E402
from embeddings import get_embeddings  # noqa: E402

RETRIEVE = [q for q in json.loads((HERE / "testset.json").read_text(encoding="utf-8"))["questions"]
            if q.get("in_scope") and q["route"] == "retrieve"]
OOS = [q["question"] for q in
       json.loads((HERE / "testset.json").read_text(encoding="utf-8"))["questions"]
       if q.get("in_scope") is False]
REFUSAL = ("could not find", "not find this", "will not guess", "unable to find")


def experiment_chunking() -> dict:
    docs = ingest.load_documents()
    configs = [(400, 100), (800, 150), (1200, 240)]
    results = []
    for size, overlap in configs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                                  separators=["\n\n", "\n", ". ", " ", ""])
        chunks = splitter.split_documents(docs)
        store = FAISS.from_documents(chunks, get_embeddings(),
                                     distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                                     normalize_L2=True)
        hits = 0
        top1 = []
        for q in RETRIEVE:
            found = store.similarity_search_with_score(q["question"], k=config.TOP_K)
            sources = [d.metadata.get("source") for d, _ in found]
            hits += q["source"] in sources
            if found:
                top1.append(found[0][1])
        results.append({
            "chunk_size": size, "chunk_overlap": overlap, "n_chunks": len(chunks),
            "hit_rate": round(hits / len(RETRIEVE), 4),
            "mean_top1_cosine": round(float(sum(top1) / len(top1)), 4) if top1 else None,
        })
        print(f"  chunk {size:>4}/{overlap:<4} -> {len(chunks):>3} chunks  "
              f"hit@{config.TOP_K}={results[-1]['hit_rate']:.0%}  "
              f"top1={results[-1]['mean_top1_cosine']}")
    return {"top_k": config.TOP_K, "n_questions": len(RETRIEVE), "configs": results}


def experiment_retry() -> dict:
    import graph as graph_mod

    def run(floor: float) -> dict:
        os.environ["RAG_LOW_COVER_FLOOR"] = str(floor)
        importlib.reload(graph_mod)          # re-reads the floor from the environment
        graph_mod.answer("warm up", thread_id=f"warm-{floor}")
        spans, lat = [], []
        runs = []                            # (question, trace) for every OOS question
        refused = 0
        for i, q in enumerate(OOS):
            t0 = time.perf_counter()
            final = graph_mod.answer(q, thread_id=f"floor{floor}-{i}")
            lat.append((time.perf_counter() - t0) * 1000)
            spans.append(len(final["trace"]))
            if any(m in (final.get("answer") or "").lower() for m in REFUSAL):
                refused += 1
            runs.append((q, final["trace"]))
        return {
            "floor": floor,
            "mean_spans": round(sum(spans) / len(spans), 2),
            "mean_ms": round(sum(lat) / len(lat), 1),
            "refusal_rate": round(refused / len(OOS), 4),
            "runs": runs,
        }

    print("  retry policy:")
    without = run(0.0)
    print(f"    floor off -> {without['mean_spans']} nodes/q, {without['mean_ms']} ms/q")
    with_floor = run(0.15)
    print(f"    floor on  -> {with_floor['mean_spans']} nodes/q, {with_floor['mean_ms']} ms/q")

    os.environ.pop("RAG_LOW_COVER_FLOOR", None)
    importlib.reload(graph_mod)
    # The floor-off run is the one worth looking at: pick the question whose trace actually contains
    # the rewrite -> retrieve cycle. Not every unanswerable question loops - one with partial lexical
    # overlap is graded relevant and answers on the first pass - so make no assumption about ordering.
    loop_q, loop_trace = next(
        ((q, t) for q, t in without["runs"] if any(s["node"] == "rewrite" for s in t)),
        (OOS[0], without["runs"][0][1]),
    )
    traced = local_tracer.render(loop_trace, loop_q, tag="eval-q17-retry-loop")
    without.pop("runs")
    with_floor.pop("runs")
    without.update({"trace_png": traced["png"]})
    return {"with_retry_loop": without, "with_floor": with_floor, "trace_png": traced["png"]}


def chart(chunk: dict, retry: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = [f"{c['chunk_size']}/{c['chunk_overlap']}\n({c['n_chunks']} chunks)"
              for c in chunk["configs"]]
    ax1.bar(labels, [c["hit_rate"] for c in chunk["configs"]], color="#1a73e8")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"retrieval hit@{chunk['top_k']} vs chunking")
    ax1.set_ylabel("hit rate")
    for i, c in enumerate(chunk["configs"]):
        ax1.text(i, c["hit_rate"] + 0.02, f"{c['hit_rate']:.0%}", ha="center")

    names = ["retry loop\n(floor off)", "no retry\n(floor on)"]
    x = range(2)
    spans = [retry["with_retry_loop"]["mean_spans"], retry["with_floor"]["mean_spans"]]
    ax2.bar(x, spans, color="#d93025")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names)
    ax2.set_title("nodes per unanswerable question")
    ax2.set_ylabel("mean node executions")
    for i, s in enumerate(spans):
        ax2.text(i, s + 0.05, f"{s}", ha="center")

    fig.suptitle("Study Assistant - ablated design choices", fontsize=13)
    fig.tight_layout()
    out = HERE / "improvement.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    chunk = experiment_chunking()
    retry = experiment_retry()
    (HERE / "improvement_chunks.json").write_text(json.dumps(chunk, indent=2), encoding="utf-8")
    (HERE / "improvement_retry.json").write_text(json.dumps(retry, indent=2), encoding="utf-8")
    png = chart(chunk, retry)
    print(f"chart -> {png}")
    print(f"retry-loop trace -> {retry['trace_png']}")


if __name__ == "__main__":
    main()
