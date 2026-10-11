"""Render the 4.2 workflow diagram.

    python docs/make_diagram.py

Always writes `docs/4.2_workflow.mmd` (Mermaid source; no network needed) and tries to write
`docs/4.2_workflow.png`. `draw_mermaid_png()` posts the definition to mermaid.ink, so it fails on
an offline machine; when it does, the same topology is drawn with matplotlib instead so the
artifact exists either way and the two cannot disagree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
import graph as graph_mod  # noqa: E402


def write_mermaid() -> Path:
    path = config.DOCS / "4.2_workflow.mmd"
    path.write_text(graph_mod.mermaid(), encoding="utf-8")
    return path


def try_native_png() -> bool:
    try:
        png = graph_mod.get_graph().get_graph().draw_mermaid_png()
        (config.DOCS / "4.2_workflow.png").write_bytes(png)
        return True
    except Exception as exc:
        print(f"  draw_mermaid_png unavailable ({type(exc).__name__}: {exc}); using fallback")
        return False


BOX = {
    "START": (0, 5), "router": (0, 4),
    "retrieve": (-2.2, 3), "tool": (0, 3), "direct": (2.2, 3),
    "grade": (-2.2, 2), "generate": (-2.2, 1), "rewrite": (0.6, 2),
    "END": (-2.2, 0),
}
EDGES = [
    ("START", "router", ""),
    ("router", "retrieve", "retrieve"),
    ("router", "tool", "tool"),
    ("router", "direct", "direct"),
    ("retrieve", "grade", ""),
    ("grade", "generate", "relevant"),
    ("grade", "rewrite", "not relevant"),
    ("rewrite", "retrieve", "retry"),
    ("tool", "generate", ""),
    ("direct", "END", ""),
    ("generate", "END", ""),
]


def fallback_png() -> Path:
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1, 6)
    ax.axis("off")

    for name, (x, y) in BOX.items():
        fc = "#e8f0fe"
        ec = "#1a73e8"
        if name in ("START", "END"):
            fc, ec, shape = "#e6f4ea", "#137333", "round,pad=0.3,rounding_size=0.8"
        elif name == "router":
            shape = "round,pad=0.35,rounding_size=0.2"
        else:
            shape = "round,pad=0.35,rounding_size=0.2"
        box = mpatches.FancyBboxPatch((x - 1.05, y - 0.35), 2.1, 0.7,
                                      boxstyle=shape, linewidth=2, fc=fc, ec=ec)
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=12, fontweight="bold",
                color="#202124")

    for src, dst, label in EDGES:
        (x1, y1), (x2, y2) = BOX[src], BOX[dst]
        style = "arc3,rad=0.0"
        if (src, dst) == ("rewrite", "retrieve"):
            style = "arc3,rad=-0.45"
        arrow = mpatches.FancyArrowPatch((x1, y1 - 0.35), (x2, y2 + 0.35),
                                         connectionstyle=style, arrowstyle="-|>",
                                         mutation_scale=16, linewidth=1.6, color="#5f6368",
                                         shrinkA=2, shrinkB=4)
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.12, my, label, fontsize=8.5, color="#5f6368",
                    ha="left", va="center",
                    bbox=dict(fc="white", ec="none", pad=1.0))

    ax.set_title("Study Assistant Agent - LangGraph topology", fontsize=14, pad=12)
    out = config.DOCS / "4.2_workflow.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    mmd = write_mermaid()
    print(f"mermaid -> {mmd}")
    if try_native_png():
        print(f"png     -> {config.DOCS / '4.2_workflow.png'} (via mermaid.ink)")
    else:
        out = fallback_png()
        print(f"png     -> {out} (matplotlib fallback)")


if __name__ == "__main__":
    main()
