"""Render the semantic-search evaluation (human-judged) as slide-ready PNGs.

Reads artifacts/search_eval_results.json (produced by
scripts/eval_semantic_search.py score) and draws:

  search_metrics_summary.png   headline: P@5 / P@10 / MRR / nDCG@10, 3 methods
  search_precision_by_query.png per-query Precision@5, keyword vs TF-IDF vs ours
  search_eval_dashboard.png     both panels tiled for a single slide

Unlike the keyword-overlap proxy in visualize_evaluation.py, every number here
comes from hand-labelled relevance (0/1/2) on the pooled top-10 -> defensible.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DATA = json.loads((ART / "search_eval_results.json").read_text())

# grey -> orange -> green reads as "worse -> ok -> best"
STYLE = {
    "keyword": ("#9c9c9c", "#6f6f6f", "Keyword\n(today's Library)"),
    "tfidf": ("#f0a35e", "#c77d34", "TF-IDF\n(fair baseline)"),
    "semantic": ("#2e8b57", "#1e6b40", "Semantic\n(ours)"),
}
ORDER = ["keyword", "tfidf", "semantic"]
METRICS = [("p5", "Precision@5"), ("p10", "Precision@10"),
           ("mrr", "MRR"), ("ndcg10", "nDCG@10")]

plt.rcParams.update({
    "font.size": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------- headline summary
def plot_summary(ax) -> None:
    summ = DATA["summary"]
    x = np.arange(len(METRICS))
    w = 0.26
    for i, m in enumerate(ORDER):
        face, edge, _ = STYLE[m]
        vals = [summ[m][key] for key, _ in METRICS]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, color=face,
                      edgecolor=edge, linewidth=1.2,
                      label=STYLE[m][2].replace("\n", " "))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                    color=edge)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in METRICS], fontsize=12.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score  (higher = better)")
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title("Semantic search wins on every metric", fontsize=16, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, ncol=3,
              bbox_to_anchor=(0.0, -0.10))


# --------------------------------------------------- per-query Precision@5
def plot_precision_by_query(ax) -> None:
    pq = DATA["per_query"]
    queries = list(pq.keys())[::-1]  # top-down reading order
    y = np.arange(len(queries))
    h = 0.26
    for i, m in enumerate(ORDER):
        face, edge, _ = STYLE[m]
        vals = [pq[q][m]["p5"] for q in queries]
        ax.barh(y + (i - 1) * h, vals, height=h, color=face, edgecolor=edge,
                linewidth=1.0, label=STYLE[m][2].replace("\n", " "))
        for yi, v in zip(y + (i - 1) * h, vals):
            ax.text(v + 0.012, yi, f"{v:.1f}", va="center", ha="left",
                    fontsize=8.5, color=edge)
    ax.set_yticks(y)
    ax.set_yticklabels(queries, fontsize=10.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Precision@5  (share of top-5 that is relevant)")
    ax.set_title("Precision@5 holds across every query", fontsize=16, pad=12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=10.5)


def _save(fig, name) -> None:
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {name}")


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    plot_summary(ax)
    fig.tight_layout()
    _save(fig, "search_metrics_summary.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_precision_by_query(ax)
    fig.tight_layout()
    _save(fig, "search_precision_by_query.png")

    fig = plt.figure(figsize=(19, 6.5))
    gs = fig.add_gridspec(1, 2, wspace=0.22)
    plot_summary(fig.add_subplot(gs[0, 0]))
    plot_precision_by_query(fig.add_subplot(gs[0, 1]))
    fig.suptitle(
        "Semantic search evaluation \u2014 8 natural-language policy queries, "
        "human-judged relevance (0/1/2), pooled top-10",
        fontsize=15, fontweight="bold", y=1.03)
    _save(fig, "search_eval_dashboard.png")
    print(f"\nAll figures written to {FIG}")


if __name__ == "__main__":
    main()
