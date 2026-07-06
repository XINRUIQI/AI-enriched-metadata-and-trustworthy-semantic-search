"""2-D semantic map of the whole catalogue (the unsupervised-clustering "wow").

Projects the 20,685 MiniLM document embeddings to 2-D with UMAP and colours
each point by its discovered HDBSCAN theme. The largest themes get a distinct
colour + a label at their centroid; smaller themes and unclustered/noise points
are drawn faint grey, so the audience literally sees the data organise itself.

Output: artifacts/figures/semantic_map.png
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

N_LABELLED = 14      # how many top themes get their own colour + label
SEED = 42


def short_name(theme: str, n: int = 26) -> str:
    theme = theme.replace(" (unclustered / noise)", "noise")
    return theme if len(theme) <= n else theme[:n] + "…"


def main() -> None:
    emb = np.load(ART / "doc_emb.npy")
    enriched = json.loads((ART / "enriched_catalogue.json").read_text())
    themes = [r["theme"] for r in enriched]
    assert len(themes) == emb.shape[0], "embedding / catalogue length mismatch"

    print(f"[map] UMAP on {emb.shape[0]:,} x {emb.shape[1]} embeddings…")
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=SEED)
    xy = reducer.fit_transform(emb)

    is_noise = np.array(["unclustered" in t or "noise" in t for t in themes])
    # rank non-noise themes by size; the biggest get a colour + label
    sizes = Counter(t for t, n in zip(themes, is_noise) if not n)
    top_themes = [t for t, _ in sizes.most_common(N_LABELLED)]
    top_set = set(top_themes)

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor("white")

    # 1) noise -> faint grey background
    ax.scatter(xy[is_noise, 0], xy[is_noise, 1], s=3, c="#d9d9d9", alpha=0.35,
               linewidths=0, label=None, rasterized=True)

    # 2) smaller (unlabelled) themes -> slightly darker grey
    other = np.array([(not n) and (t not in top_set)
                      for t, n in zip(themes, is_noise)])
    ax.scatter(xy[other, 0], xy[other, 1], s=4, c="#a9a9a9", alpha=0.4,
               linewidths=0, rasterized=True)

    # 3) top themes -> distinct colours
    cmap = plt.cm.tab20
    theme_arr = np.array(themes)
    for k, theme in enumerate(top_themes):
        mask = theme_arr == theme
        color = cmap(k % 20)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=7, color=color, alpha=0.8,
                   linewidths=0, label=f"{short_name(theme)}  ({mask.sum()})",
                   rasterized=True)
        cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
        ax.text(cx, cy, short_name(theme, 18), fontsize=8, fontweight="bold",
                ha="center", va="center", color="black",
                path_effects=None,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.65))

    ax.set_title(
        "London's 20,685 datasets, organised by meaning\n"
        f"MiniLM embeddings → UMAP 2-D · coloured by top {N_LABELLED} of 76 "
        "discovered themes (grey = smaller themes / unclustered)",
        fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
              frameon=False, markerscale=2.2, title="Largest themes",
              title_fontsize=9)
    fig.tight_layout()
    out = FIG / "semantic_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[map] wrote {out}")


if __name__ == "__main__":
    main()
