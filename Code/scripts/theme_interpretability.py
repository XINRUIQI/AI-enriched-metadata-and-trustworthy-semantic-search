"""Human-facing evaluation of the theme clustering (checks A / B / C).

Automatic theming without gold labels is *clustering*, not classification, so
accuracy alone is not enough. This script produces the three interpretability
checks that convince a human the clusters mean something:

  A. Cluster interpretability  -> each theme's top TF-IDF words + a Clear /
     Mixed / Unclear rating (rule-assisted, meant to be human-reviewed).
  B. Sample consistency        -> 5 random dataset TITLES per theme; if they
     read as one topic the cluster is trustworthy.
  C. Stability                 -> re-run HDBSCAN at different ``min_cluster_size``
     and measure Adjusted Rand Index vs the base run. Stable themes survive
     parameter changes.

Everything targets the method we actually ship (HDBSCAN, min_cluster_size=60),
NOT the abandoned KMeans-15. Outputs:
  artifacts/theme_interpretability.md      (A + B, all themes)
  artifacts/figures/interpretability_cards.png  (curated themes, pitch-ready)
  artifacts/figures/hdbscan_stability.png       (C, pitch-ready)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_catalogue
from src.enrich import _stopwords, name_themes

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
STOP = set(_stopwords())

# base HDBSCAN params = what we ship
BASE = dict(min_cluster_size=60, min_samples=10)


def get_embeddings():
    for cache in (ART / "doc_emb.npy", ART / "embeddings.npy"):
        if cache.exists():
            print(f"[emb] loading {cache.name}")
            return np.load(cache)
    raise FileNotFoundError("no cached embeddings; run analysis.py first")


def rate_theme(theme_words: list[str], sample_titles: list[str]) -> str:
    """Rule-assisted Clear / Mixed / Unclear rating (a human should confirm).

    Heuristic: what fraction of the sampled titles contain at least one of the
    theme's top words? High overlap -> the label describes its members (Clear).
    """
    stems = [w[:5] for w in theme_words if len(w) > 2]
    if not stems:
        return "Unclear"
    hit = 0
    for t in sample_titles:
        tl = t.lower()
        if any(s in tl for s in stems):
            hit += 1
    frac = hit / max(len(sample_titles), 1)
    if frac >= 0.6:
        return "Clear"
    if frac >= 0.3:
        return "Mixed"
    return "Unclear"


def main() -> None:
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    emb = get_embeddings()
    assert emb.shape[0] == len(df), (emb.shape, len(df))
    texts = df["search_text"].tolist()

    # -------------------------------------------------- base clustering (A/B)
    print("[cluster] base HDBSCAN", BASE)
    labels = HDBSCAN(**BASE).fit_predict(emb)
    names = name_themes(texts, labels)
    theme_ids = sorted(c for c in set(labels) if c != -1)
    n_noise = int((labels == -1).sum())
    print(f"[cluster] {len(theme_ids)} themes, noise={n_noise} "
          f"({100 * n_noise / len(df):.1f}%)")

    # -------------------------------------------------- A + B per theme
    rows = []  # (id, size, words, rating, samples)
    for c in theme_ids:
        idx = np.where(labels == c)[0]
        words = [w.strip() for w in names[c].split(",")]
        pick = RNG.choice(idx, size=min(5, len(idx)), replace=False)
        samples = [df.iloc[i]["title"] for i in pick]
        rating = rate_theme(words, samples)
        rows.append((int(c), len(idx), words, rating, samples))

    rows.sort(key=lambda r: r[1], reverse=True)
    rating_counts = Counter(r[3] for r in rows)
    print("[rating]", dict(rating_counts))

    # -------------------------------------------------- write markdown (A+B)
    md = ["# Theme interpretability & sample consistency",
          "",
          "> Method: HDBSCAN (min_cluster_size=60, min_samples=10) — the shipped run.",
          f"> {len(theme_ids)} themes, {n_noise} datasets unclustered "
          f"({100 * n_noise / len(df):.1f}%).",
          f"> Auto-rating: **Clear {rating_counts['Clear']} · "
          f"Mixed {rating_counts['Mixed']} · Unclear {rating_counts['Unclear']}** "
          "(rule-assisted, review by eye).",
          ""]
    for cid, size, words, rating, samples in rows:
        md.append(f"## Theme {cid} — `{', '.join(words)}`  ·  {size} datasets  ·  **{rating}**")
        md.append("")
        md.append("Random sample (B — sample consistency):")
        for s in samples:
            md.append(f"- {s}")
        md.append("")
    (ART / "theme_interpretability.md").write_text("\n".join(md))
    print("[write] artifacts/theme_interpretability.md")

    # -------------------------------------------------- curated pitch cards
    # pick recognisable, human-nameable themes for the slide
    wanted = {
        "diabetes": "Diabetes / health",
        "traffic": "Road traffic & accidents",
        "homelessness": "Homelessness & housing",
        "emissions": "Emissions / energy",
        "sexual": "Sexual health",
        "apprentic": "Apprenticeships / skills",
        "smoking": "Smoking prevalence",
        "obesity": "Childhood obesity",
        "journey": "Transport / journeys",
    }
    cards = []
    for key, human in wanted.items():
        for cid, size, words, rating, samples in rows:
            if any(key in w for w in words):
                cards.append((human, words, rating, samples, size))
                break
    _plot_cards(cards)

    # -------------------------------------------------- C stability (HDBSCAN)
    _plot_stability(emb, labels)


def _plot_cards(cards) -> None:
    n = len(cards)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow))
    axes = np.array(axes).ravel()
    color = {"Clear": "#2e8b57", "Mixed": "#e08b1f", "Unclear": "#c0392b"}
    for ax, (human, words, rating, samples, size) in zip(axes, cards):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   fill=False, ec=color[rating], lw=2.5))
        ax.text(0.04, 0.92, human, fontsize=13, fontweight="bold", va="top")
        ax.text(0.96, 0.92, rating, fontsize=11, fontweight="bold", va="top",
                ha="right", color=color[rating])
        ax.text(0.04, 0.78, f"top words: {', '.join(words)}", fontsize=8.5,
                va="top", color="#444", style="italic")
        ax.text(0.04, 0.66, f"{size} datasets · 5 random samples:", fontsize=8,
                va="top", color="#666")
        y = 0.55
        for s in samples:
            ax.text(0.06, y, "• " + (s[:52] + "…" if len(s) > 52 else s),
                    fontsize=8, va="top")
            y -= 0.12
    for ax in axes[len(cards):]:
        ax.axis("off")
    fig.suptitle("Theme interpretability (A) + sample consistency (B) — HDBSCAN themes",
                 fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(FIG / "interpretability_cards.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("[fig] figures/interpretability_cards.png")


def _plot_stability(emb, base_labels) -> None:
    grid = [30, 40, 50, 60, 80, 100]
    aris, nthemes, noises = [], [], []
    for mcs in grid:
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=10).fit_predict(emb)
        aris.append(round(adjusted_rand_score(base_labels, lab), 3))
        nthemes.append(len(set(lab) - {-1}))
        noises.append(round(100 * (lab == -1).mean(), 1))
        print(f"  mcs={mcs:3d}  themes={nthemes[-1]:3d}  "
              f"noise={noises[-1]:4.1f}%  ARI_vs_base={aris[-1]}")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = [str(g) for g in grid]
    bars = ax.bar(x, aris, color="#4c78a8")
    base_i = grid.index(BASE["min_cluster_size"])
    bars[base_i].set_color("#e45756")
    mean = round(float(np.mean(aris)), 3)
    ax.axhline(mean, ls="--", color="#888", lw=1.4, label=f"mean ARI = {mean}")
    ax.set_ylim(0, 1.28)
    ax.set_xlabel("HDBSCAN min_cluster_size  (red = shipped run = 60)")
    ax.set_ylabel("Adjusted Rand Index vs shipped run")
    ax.set_title("C  Theme stability under parameter change (HDBSCAN)\n"
                 "1 = identical partition, 0 = random", fontweight="bold", pad=14)
    for b, a, nt in zip(bars, aris, nthemes):
        # keep labels below the top of tall bars so they never hit the title
        yt = a - 0.13 if a >= 0.95 else a + 0.02
        vc = "white" if a >= 0.95 else "black"
        ax.text(b.get_x() + b.get_width() / 2, yt, f"{a}\n{nt} themes",
                ha="center", va="bottom", fontsize=8.5, color=vc)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "hdbscan_stability.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("[fig] figures/hdbscan_stability.png")


if __name__ == "__main__":
    main()
