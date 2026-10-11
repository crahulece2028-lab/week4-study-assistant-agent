"""A local stand-in for the LangSmith trace view.

When `LANGSMITH_API_KEY` is set, `graph.answer()` also emits a real trace to LangSmith. Without a
key there is still a complete span record available: every node appends `{node, detail,
elapsed_ms}` to `state["trace"]`. This module turns that list into

  - `eval/traces/<slug>.json`  - the raw span list (what a hosted tracer would store), and
  - `docs/traces/<slug>.png`   - a waterfall chart, the visual analogue of the LangSmith run view.

This is what makes task 4.3 reproducible on a machine with no LangSmith account, without pretending
the local chart *is* LangSmith.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "src"
import sys  # noqa: E402
sys.path.insert(0, str(SRC))
import config  # noqa: E402

COLORS = {
    "router": "#1a73e8", "retrieve": "#188038", "grade": "#e37400",
    "rewrite": "#a142f4", "tool": "#d93025", "generate": "#0b8043",
    "direct": "#5f6368",
}


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:50] or "question").strip("-")


def render(trace: list[dict], question: str, tag: str | None = None) -> dict:
    name = tag or slug(question)
    traces_dir = config.EVAL / "traces"
    png_dir = config.DOCS / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    spans = []
    cursor = 0.0
    for step in trace:
        dur = step.get("elapsed_ms", 0.0)
        spans.append({**step, "start_ms": round(cursor, 1), "end_ms": round(cursor + dur, 1)})
        cursor += dur
    total = cursor

    record = {"question": question, "total_ms": round(total, 1), "spans": spans}
    json_path = traces_dir / f"{name}.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 0.62 * max(len(spans), 2) + 1.6))
    for i, sp in enumerate(spans):
        y = len(spans) - i - 1
        ax.barh(y, sp["elapsed_ms"], left=sp["start_ms"], height=0.6,
                color=COLORS.get(sp["node"], "#5f6368"), alpha=0.9, edgecolor="white")
        ax.text(sp["start_ms"] + max(sp["elapsed_ms"], total * 0.01) + total * 0.006, y,
                f"{sp['detail'][:58]}", va="center", fontsize=8, color="#202124")
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels([sp["node"] for sp in spans][::-1], fontsize=9)
    ax.set_xlabel("elapsed since start (ms)")
    head = question if len(question) <= 70 else question[:67] + "..."
    ax.set_title(f"trace: {head}\ntotal {total:.0f} ms", fontsize=11)
    ax.set_xlim(0, total * 1.02 if total else 1)
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    png_path = png_dir / f"{name}.png"
    fig.savefig(png_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"name": name, "json": str(json_path), "png": str(png_path),
            "total_ms": round(total, 1)}


if __name__ == "__main__":
    import graph as graph_mod
    q = "What is the Nyquist rate and why does it matter?"
    final = graph_mod.answer(q)
    print(json.dumps(render(final["trace"], q), indent=2))
