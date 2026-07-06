"""Render a baseline-vs-ours search comparison slide (PNG).

Reads artifacts/evaluation_full.json (p23_search) and draws a three-column
side-by-side (keyword / TF-IDF / semantic) for two showcase queries:

* "young people not in work"     -> synonym/acronym recall (NEET)
* "air quality and pollution"    -> robustness to long-text noise (COVID)

Output: artifacts/baseline_comparison.png
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

QUERIES = [
    ("young people not in work", "Synonym / acronym recall  (NEET)"),
    ("air quality and pollution", "Robustness to long-text noise  (COVID)"),
]

# (json key, header title, sub-label)
METHODS = [
    ("keyword", "Keyword", "Today's Library \u00b7 word-count"),
    ("tfidf", "TF-IDF", "Fair lexical baseline"),
    ("semantic", "Semantic", "Ours \u00b7 MiniLM embeddings"),
]

# relevance palette (face, edge)
REL_STYLE = {
    "hit": ("#d7f0d1", "#2e8b57"),
    "partial": ("#fdf0c8", "#d0a020"),
    "miss": ("#f7d9dc", "#c0392b"),
}
REL_MARK = {"hit": "\u2713", "partial": "~", "miss": "\u2717"}

HEADER_COLORS = {
    "keyword": ("#e9ecef", "#868e96"),
    "tfidf": ("#dbe9fb", "#4a7fc0"),
    "semantic": ("#d3f0da", "#2e8b57"),
}

# layout
LEFT, RIGHT, GAP = 0.035, 0.035, 0.018
COL_W = (1 - LEFT - RIGHT - 2 * GAP) / 3


def col_x(i: int) -> float:
    return LEFT + i * (COL_W + GAP)


def relevance(query_key: str, title: str) -> str:
    t = title.lower()
    if "young people not in work" in query_key:
        if "not in education" in t or "neet" in t:
            return "hit"
        if "young people" in t or "labour market" in t or (
            "employment" in t and "coronavirus" not in t
        ):
            return "partial"
        return "miss"
    if "air quality" in query_key:
        if any(
            k in t
            for k in (
                "air pollution",
                "air quality",
                "atmospheric",
                "emission",
                "pm2.5",
                "particulate",
                "pollution",
            )
        ):
            return "hit"
        if "covid" in t or "coronavirus" in t:
            return "miss"
        return "partial"
    return "partial"


def wrap2(text: str, width: int = 40) -> str:
    lines = textwrap.wrap(text, width=width)
    if len(lines) <= 2:
        return "\n".join(lines)
    out = lines[:2]
    out[1] = out[1].rstrip() + "\u2026"
    return "\n".join(out)


def draw_block(ax, y_top: float, y_bot: float, qkey: str, subtitle: str, data: dict) -> None:
    bar_h = 0.05
    ax.add_patch(
        Rectangle((LEFT, y_top - bar_h), 1 - LEFT - RIGHT, bar_h,
                  facecolor="#2b2b3a", edgecolor="none", zorder=1)
    )
    ax.text(LEFT + 0.012, y_top - bar_h / 2, f'Query:  "{qkey}"',
            color="white", fontsize=13.5, fontweight="bold", va="center", ha="left", zorder=2)
    ax.text(1 - RIGHT - 0.012, y_top - bar_h / 2, subtitle,
            color="#c9c9d4", fontsize=10.5, fontstyle="italic", va="center", ha="right", zorder=2)

    header_top = y_top - bar_h - 0.012
    header_h = 0.055
    rows_top = header_top - header_h - 0.008
    n = 5
    rgap = 0.007
    row_h = (rows_top - y_bot - (n - 1) * rgap) / n

    for ci, (mkey, mname, mtag) in enumerate(METHODS):
        x = col_x(ci)
        hc, he = HEADER_COLORS[mkey]
        ax.add_patch(
            Rectangle((x, header_top - header_h), COL_W, header_h,
                      facecolor=hc, edgecolor=he, linewidth=1.4, zorder=1)
        )
        ax.text(x + COL_W / 2, header_top - header_h * 0.37, mname,
                fontsize=13, fontweight="bold", va="center", ha="center", color="#1a1a1a", zorder=2)
        ax.text(x + COL_W / 2, header_top - header_h * 0.76, mtag,
                fontsize=8.3, va="center", ha="center", color="#555", zorder=2)

        for ri, title in enumerate(data[qkey][mkey][:n]):
            ry = rows_top - (ri + 1) * row_h - ri * rgap
            rel = relevance(qkey, title)
            face, edge = REL_STYLE[rel]
            ax.add_patch(
                Rectangle((x, ry), COL_W, row_h, facecolor=face, edgecolor=edge,
                          linewidth=1.0, zorder=1)
            )
            ax.text(x + 0.008, ry + row_h - 0.008, str(ri + 1),
                    fontsize=7.5, fontweight="bold", color="#777", va="top", ha="left", zorder=2)
            ax.text(x + COL_W - 0.008, ry + row_h - 0.008, REL_MARK[rel],
                    fontsize=10, fontweight="bold", color=edge, va="top", ha="right", zorder=3)
            ax.text(x + 0.024, ry + row_h / 2, wrap2(title, 40),
                    fontsize=8, va="center", ha="left", color="#1a1a1a", zorder=2)


def main() -> None:
    data = json.loads((ART / "evaluation_full.json").read_text())["p23_search"]

    fig = plt.figure(figsize=(15, 11.5))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.972, "Search quality: two baselines vs. our method",
            ha="center", fontsize=21, fontweight="bold", color="#1a1a2e")
    ax.text(0.5, 0.938,
            "Same query \u00b7 same corpus (title + notes) \u00b7 Top-5 results \u00b7 20,685 London datasets",
            ha="center", fontsize=11.5, color="#555")

    draw_block(ax, 0.885, 0.505, *QUERIES[0], data=data)
    draw_block(ax, 0.435, 0.055, *QUERIES[1], data=data)

    leg = [("Relevant", "hit"), ("Partial match", "partial"), ("Off-topic", "miss")]
    lx = 0.285
    for name, rel in leg:
        fc, ec = REL_STYLE[rel]
        ax.add_patch(Rectangle((lx, 0.4585), 0.02, 0.02, facecolor=fc, edgecolor=ec, linewidth=1.2))
        ax.text(lx + 0.027, 0.4685, f"{REL_MARK[rel]} {name}", va="center", ha="left",
                fontsize=10, color="#333")
        lx += 0.155

    ax.text(0.5, 0.02,
            "Keyword mimics today's Library  \u00b7  TF-IDF is a fair lexical baseline  \u00b7  "
            "embeddings recover synonyms (NEET) and resist long-text noise (COVID)",
            ha="center", fontsize=10.5, fontstyle="italic", color="#333")

    out = ART / "baseline_comparison.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
