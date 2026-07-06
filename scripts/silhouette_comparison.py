"""KMeans-vs-HDBSCAN silhouette comparison (the "we didn't just pick k" figure).

The k-sweep chart already shows KMeans silhouette stays low (0.07-0.11) with no
elbow. This adds the punchline the pitch makes verbally: switching to density
clustering (HDBSCAN) lifts silhouette to 0.35 — a 3-4x jump — while also
choosing the number of themes on its own.

Reads:
  * artifacts/evaluation_full.json -> p5_k_silhouette (KMeans sweep)
  * artifacts/evaluation.json      -> silhouette (HDBSCAN headline)

Output: artifacts/figures/silhouette_kmeans_vs_hdbscan.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    full = json.loads((ART / "evaluation_full.json").read_text())
    headline = json.loads((ART / "evaluation.json").read_text())

    k_sil = full["p5_k_silhouette"]            # {"8": 0.069, ...}
    hdb_sil = float(headline["silhouette"])    # 0.354
    n_themes = int(headline.get("n_themes", 76))

    labels = [f"KMeans\nk={k}" for k in k_sil] + [f"HDBSCAN\n({n_themes} themes)"]
    vals = list(k_sil.values()) + [hdb_sil]
    colors = ["#bcbcbc"] * len(k_sil) + ["#2e8b57"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold" if v == hdb_sil else "normal")

    best_km = max(k_sil.values())
    factor = hdb_sil / best_km if best_km else 0
    ax.annotate(
        f"~{factor:.1f}x better separation\n(and picks #themes automatically)",
        xy=(len(k_sil), hdb_sil), xytext=(len(k_sil) - 2.4, hdb_sil + 0.02),
        fontsize=10, fontweight="bold", color="#2e8b57",
        arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.6))

    ax.axhspan(0, 0.12, color="#f2c9c9", alpha=0.35, zorder=0)
    ax.text(0.2, 0.11, "KMeans stuck in the weak-separation band",
            fontsize=8.5, color="#a33", va="top")

    ax.set_ylabel("silhouette score  (higher = better-separated themes)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Why HDBSCAN: density clustering beats every KMeans k",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIG / "silhouette_kmeans_vs_hdbscan.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[sil] KMeans best={best_km:.3f}  HDBSCAN={hdb_sil:.3f}  "
          f"({factor:.1f}x)")
    print(f"[sil] wrote {out}")


if __name__ == "__main__":
    main()
