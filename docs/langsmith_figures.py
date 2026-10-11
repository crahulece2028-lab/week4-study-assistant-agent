"""Render trace and latency figures from the *live LangSmith project*.

Unlike `eval/local_tracer.py` (which draws the in-process span list), this pulls the runs back from
the LangSmith API and renders their real start/end timestamps, so the figures are evidence that the
traces really landed in LangSmith - each figure is titled with its LangSmith trace id.

    python docs/langsmith_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

config.enable_langsmith()
from langsmith import Client  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NODES = {"router", "retrieve", "grade", "rewrite", "tool", "generate", "direct"}
COLORS = {
    "router": "#1a73e8", "retrieve": "#188038", "grade": "#e37400",
    "rewrite": "#a142f4", "tool": "#d93025", "generate": "#0b8043",
    "direct": "#5f6368", "LangGraph": "#3c4043",
}
OUT = config.DOCS / "traces"
OUT.mkdir(parents=True, exist_ok=True)


def fetch_all() -> list:
    """Return (root, [all runs in its trace]) for every LangGraph root in the project."""
    c = Client(api_key=config.LANGSMITH_API_KEY)
    try:
        roots = list(c.list_runs(project_name=config.LANGSMITH_PROJECT, is_root=True, limit=100))
    except TypeError:
        roots = [r for r in c.list_runs(project_name=config.LANGSMITH_PROJECT, limit=100)
                 if r.parent_run_id is None]

    traces = []
    for root in roots:
        if root.name != "LangGraph":
            continue
        q = ""
        try:
            q = (root.inputs or {}).get("question", "")
        except Exception:
            pass
        if "warm up" in q:
            continue
        children = list(c.list_runs(project_name=config.LANGSMITH_PROJECT, trace_id=root.id))
        traces.append((root.trace_id, root, children or [root], q))
    return traces


def draw_waterfall(tid, root, rs, question, tag) -> dict:
    t0 = root.start_time
    spans = []
    for r in rs:
        if r.name not in NODES:
            continue
        start = (r.start_time - t0).total_seconds() * 1000
        dur = ((r.end_time - r.start_time).total_seconds() * 1000) if r.end_time else 0.0
        spans.append({"node": r.name, "start": max(start, 0.0), "dur": dur})
    spans.sort(key=lambda s: s["start"])
    total = ((root.end_time - root.start_time).total_seconds() * 1000) if root.end_time else 1.0

    fig, ax = plt.subplots(figsize=(10, 0.62 * max(len(spans), 2) + 1.8))
    for i, sp in enumerate(spans):
        y = len(spans) - i - 1
        ax.barh(y, max(sp["dur"], total * 0.004), left=sp["start"], height=0.6,
                color=COLORS.get(sp["node"], "#5f6368"), alpha=0.92, edgecolor="white")
        ax.text(sp["start"] + max(sp["dur"], total * 0.004) + total * 0.008, y,
                f"{sp['node']}  {sp['dur']:.1f} ms", va="center", fontsize=8, color="#202124")
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels([s["node"] for s in spans][::-1], fontsize=9)
    ax.set_xlabel("elapsed since trace start (ms)")
    head = question if len(question) <= 64 else question[:61] + "..."
    ax.set_title(f"LangSmith trace {tid}\n{head}\nroot span {total:.0f} ms - {len(spans)} nodes",
                 fontsize=10)
    ax.set_xlim(0, total * 1.05 if total else 1)
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    png = OUT / f"langsmith-{tag}.png"
    fig.savefig(png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"tag": tag, "trace_id": str(tid), "question": question,
            "total_ms": round(total, 1), "n_nodes": len(spans), "png": png.name}


def draw_latency(rows):
    rows = sorted(rows, key=lambda r: r["total_ms"])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    labels = [r["tag"] for r in rows]
    vals = [r["total_ms"] for r in rows]
    ax.bar(range(len(rows)), vals, color="#1a73e8", alpha=0.85)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("trace latency (ms)")
    ax.set_title(f"LangSmith: per-question trace latency ({len(rows)} runs, offline config)\n"
                 f"median {sorted(vals)[len(vals)//2]:.0f} ms", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    png = OUT / "langsmith-latency.png"
    fig.savefig(png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png.name


def main() -> None:
    traces = fetch_all()
    offline = []
    for tid, root, rs, q in traces:
        if not root.end_time:
            continue
        dur = (root.end_time - root.start_time).total_seconds()
        if dur < 1.5 and str(root.status) == "success":
            offline.append((tid, root, rs, q))

    by_q = {}
    for tid, root, rs, q in offline:
        by_q.setdefault(q, (tid, root, rs, q))
    picks = {
        "q01-grounded": "What is the Nyquist rate and why does it matter?",
        "q12-tool": "calculate 2*pi*50*0.001",
        "q17-refusal": "Who won the 2018 FIFA World Cup?",
        "q20-friis-miss": "What is the formula for the Friis transmission equation?",
        "q22-fm-miss": "How is the bandwidth of an FM signal estimated from its message and deviation?",
    }
    figs = []
    for tag, question in picks.items():
        hit = by_q.get(question)
        if hit:
            tid, root, rs, q = hit
            figs.append(draw_waterfall(tid, root, rs, q, tag))
            print("rendered", tag, "->", figs[-1]["png"])

    rows = []
    for q, (tid, root, rs, _q) in by_q.items():
        dur = (root.end_time - root.start_time).total_seconds() * 1000
        rows.append({"tag": q[:22], "total_ms": dur})
    if rows:
        draw_latency(rows)
        print("rendered latency over", len(rows), "runs")

    (OUT / "langsmith_traces.json").write_text(
        json.dumps({"project": config.LANGSMITH_PROJECT, "n_traces_in_project": len(traces),
                    "figures": figs, "latency_runs": len(rows)}, indent=2), encoding="utf-8")
    print("total traces in project:", len(traces), "| offline traces:", len(offline))


if __name__ == "__main__":
    main()
