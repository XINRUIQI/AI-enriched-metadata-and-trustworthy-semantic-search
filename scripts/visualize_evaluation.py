"""Render the evaluate_report.py results (artifacts/evaluation_full.json) as PNGs.

Produces one combined dashboard plus individual charts under
artifacts/figures/, covering the honest evaluation points:
  P1  data quality             -> quality_bars.png
  P5  theme sizes + org bias   -> theme_sizes.png
  P5c cluster stability (ARI)  -> cluster_stability.png
  P5  silhouette vs k          -> k_silhouette.png
  P6  near-duplicate counts    -> duplicate_counts.png
  P2/3 search method reach     -> search_reach.png
All are also tiled into evaluation_dashboard.png.
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

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

DATA = json.loads((ART / "evaluation_full.json").read_text())


# ------------------------------------------------------------------ P1 quality
def plot_quality(ax) -> None:
    q = DATA["p1_quality"]
    metrics = {
        "No tags": q["pct_no_tags"],
        "No theme": q["pct_no_theme"],
        "Short desc": q["pct_short_desc"],
        f"Top org\n({q['top_org']})": q["top_org_share_pct"],
    }
    names = list(metrics)
    vals = list(metrics.values())
    colors = ["#d95f5f" if v >= 60 else "#f0a35e" if v >= 20 else "#6aa66a" for v in vals]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of catalogue")
    ax.set_title(f"P1  Data quality  (n={q['n_datasets']:,} datasets, {q['n_orgs']} orgs)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", va="bottom",
                fontsize=9)


# ------------------------------------------------------------- P5 theme sizes
def plot_theme_sizes(ax) -> None:
    themes = DATA["p5_themes"]
    items = sorted(themes.values(), key=lambda d: d["size"])
    labels = [f"T{list(themes.keys())[list(themes.values()).index(it)]}: {it['name'][:34]}"
              for it in items]
    sizes = [it["size"] for it in items]
    shares = [it["top_org_share_pct"] for it in items]
    cmap = plt.cm.RdYlGn_r
    colors = [cmap(s / 100) for s in shares]
    y = np.arange(len(items))
    ax.barh(y, sizes, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("datasets in theme")
    ax.set_title("P5  Theme sizes (colour = single-org dominance %)")
    for yi, (s, sh) in enumerate(zip(sizes, shares)):
        ax.text(s + max(sizes) * 0.01, yi, f"{s:,} · {sh:.0f}%", va="center", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("top-org share %", fontsize=8)


# ------------------------------------------------------- P5c cluster stability
def plot_stability(ax) -> None:
    ari = DATA["p5c_ari_vs_seed0"]
    mean = DATA["p5c_ari_mean"]
    seeds = list(ari.keys())
    vals = list(ari.values())
    bars = ax.bar([f"seed {s}" for s in seeds], vals, color="#4c78a8")
    ax.axhline(mean, color="#e45756", ls="--", lw=1.6, label=f"mean ARI = {mean}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Adjusted Rand Index vs seed 0")
    ax.set_title("P5c  Clustering stability across seeds  (1=identical, 0=random)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v}", ha="center", fontsize=9)
    ax.legend(loc="lower right")


# ------------------------------------------------------- P5 silhouette vs k
def plot_k_silhouette(ax) -> None:
    ks = DATA["p5_k_silhouette"]
    x = [int(k) for k in ks]
    y = list(ks.values())
    ax.plot(x, y, "-o", color="#54a24b", lw=2)
    chosen = 15
    if chosen in x:
        ci = x.index(chosen)
        ax.scatter([chosen], [y[ci]], s=160, facecolors="none", edgecolors="#e45756",
                   linewidths=2, zorder=5, label="chosen k=15")
        ax.legend(loc="upper left")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("silhouette")
    ax.set_title("P5  Silhouette vs k  (all values low → weak separation)")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.002, f"{yi}", ha="center", fontsize=8)


# ------------------------------------------------------- P6 duplicate counts
def plot_duplicates(ax) -> None:
    dc = DATA["p6_dup_counts"]
    thr = list(dc.keys())
    vals = list(dc.values())
    bars = ax.bar([f"≥ {t}" for t in thr], vals, color="#b279a2")
    ax.set_yscale("log")
    ax.set_ylabel("dataset pairs (log scale)")
    ax.set_xlabel("cosine similarity threshold")
    ax.set_title("P6  Near-duplicate pairs by threshold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:,}", ha="center", fontsize=8.5)


# ------------------------------------------------- P2/3 search method reach
def plot_search_reach(ax) -> None:
    """For each query, how many of top-5 hits are on-topic per method.

    On-topic = title shares a meaningful (>3 char) keyword with the query.
    Rough proxy, but it exposes keyword search's failure mode clearly."""
    stop = {"and", "not", "the", "for", "of", "in", "at", "against", "homes", "people"}

    def relevance(query, titles):
        terms = {w for w in query.lower().replace("-", " ").split()
                 if len(w) > 3 and w not in stop}
        hits = 0
        for t in titles:
            tl = t.lower()
            if any(term[:5] in tl for term in terms):
                hits += 1
        return hits

    search = DATA["p23_search"]
    queries = list(search.keys())
    methods = ["keyword", "tfidf", "semantic"]
    mcolor = {"keyword": "#9c9c9c", "tfidf": "#f0a35e", "semantic": "#4c78a8"}
    scores = {m: [relevance(q, search[q][m]) for q in queries] for m in methods}
    y = np.arange(len(queries))
    h = 0.26
    for i, m in enumerate(methods):
        ax.barh(y + (i - 1) * h, scores[m], height=h, label=m, color=mcolor[m])
    ax.set_yticks(y)
    ax.set_yticklabels([q if len(q) <= 34 else q[:32] + "…" for q in queries], fontsize=7.5)
    ax.set_xlabel("on-topic hits in top-5  (keyword-overlap proxy)")
    ax.set_xlim(0, 5.4)
    ax.set_title("P2/3  Search relevance: keyword vs TF-IDF vs semantic")
    ax.legend(loc="lower right", fontsize=8)


# ------------------------------------------------------------------ individual
_SPECS = [
    ("quality_bars.png", plot_quality, (7, 4.5)),
    ("theme_sizes.png", plot_theme_sizes, (9, 6)),
    ("cluster_stability.png", plot_stability, (7, 4.5)),
    ("k_silhouette.png", plot_k_silhouette, (7, 4.5)),
    ("duplicate_counts.png", plot_duplicates, (7, 4.5)),
    ("search_reach.png", plot_search_reach, (9, 5.5)),
]


def main() -> None:
    for fname, fn, size in _SPECS:
        fig, ax = plt.subplots(figsize=size)
        fn(ax)
        fig.tight_layout()
        fig.savefig(FIG / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {fname}")

    # combined dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.25)
    plot_quality(fig.add_subplot(gs[0, 0]))
    plot_stability(fig.add_subplot(gs[0, 1]))
    plot_theme_sizes(fig.add_subplot(gs[1, 0]))
    plot_search_reach(fig.add_subplot(gs[1, 1]))
    plot_k_silhouette(fig.add_subplot(gs[2, 0]))
    plot_duplicates(fig.add_subplot(gs[2, 1]))
    fig.suptitle("London Data Week — Enrichment Pipeline Evaluation", fontsize=16,
                 fontweight="bold", y=0.995)
    fig.savefig(FIG / "evaluation_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig] evaluation_dashboard.png")
    print(f"\nAll figures written to {FIG}")


if __name__ == "__main__":
    main()
