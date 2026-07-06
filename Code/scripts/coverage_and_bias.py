"""Before/after metadata coverage + source-bias figures.

* coverage_before_after.png — the problem→solution slide: how much of the
  catalogue had a theme / tags before vs after enrichment. We keep the honest
  framing: 35% of datasets are deliberately left "unclustered / needs manual
  metadata" instead of force-fitting them into a theme.

* org_bias.png — the responsible-AI slide: one borough (Croydon) supplies most
  of the catalogue, which can pull themes toward its topics. We show the skew
  rather than hide it.

Reads artifacts/enriched_catalogue.json (+ evaluation_full.json for the
before-state numbers). Writes into artifacts/figures/.
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

RED, GREEN, GREY = "#d95f5f", "#2e8b57", "#c9c9c9"


def _load():
    enriched = json.loads((ART / "enriched_catalogue.json").read_text())
    full = json.loads((ART / "evaluation_full.json").read_text())
    return enriched, full


def coverage_figure(enriched, full) -> None:
    n = len(enriched)
    is_noise = [("unclustered" in r["theme"] or "noise" in r["theme"])
                for r in enriched]
    theme_after = 100 * (n - sum(is_noise)) / n
    tags_after = 100 * sum(1 for r in enriched if r.get("auto_tags")) / n

    q = full["p1_quality"]
    theme_before = 100 - q["pct_no_theme"]     # 0.0
    tags_before = 100 - q["pct_no_tags"]       # 7.5

    groups = ["Has a theme", "Has tags"]
    before = [theme_before, tags_before]
    after = [theme_after, tags_after]

    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    b1 = ax.bar(x - w / 2, before, w, label="Before (today's Library)", color=RED)
    b2 = ax.bar(x + w / 2, after, w, label="After enrichment", color=GREEN)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.0f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_ylim(0, 112)
    ax.set_ylabel("% of the 20,685 datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_title("Metadata coverage: before vs after enrichment",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    noise_pct = 100 - theme_after
    ax.text(0.5, -0.16,
            f"Honest note: {noise_pct:.0f}% are left 'unclustered / needs manual "
            "metadata' on purpose — we don't force-fit unrelated datasets.",
            transform=ax.transAxes, ha="center", fontsize=8.5,
            fontstyle="italic", color="#555")
    fig.tight_layout()
    out = FIG / "coverage_before_after.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[cov] theme {theme_before:.0f}%->{theme_after:.0f}%, "
          f"tags {tags_before:.0f}%->{tags_after:.0f}%  -> {out.name}")


def bias_figure(enriched, full) -> None:
    n = len(enriched)
    counts = Counter(r["organization"] for r in enriched)
    top = counts.most_common(10)
    names = [k if len(k) <= 34 else k[:32] + "…" for k, _ in top][::-1]
    shares = [100 * v / n for _, v in top][::-1]
    colors = [RED if s >= 50 else "#f0a35e" if s >= 10 else "#6a9fd8"
              for s in shares]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor("white")
    y = np.arange(len(names))
    bars = ax.barh(y, shares, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    for b, s in zip(bars, shares):
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2,
                f"{s:.1f}%", va="center", fontsize=8.5)
    ax.set_xlabel("% of all datasets")
    ax.set_xlim(0, max(shares) * 1.18)
    ax.set_title("Source bias: one borough dominates the catalogue",
                 fontsize=13, fontweight="bold")

    org_driven = full.get("p5_org_driven_count")
    if org_driven is not None:
        ax.text(0.98, 0.06,
                f"→ {org_driven}/15 KMeans themes were single-org-dominated "
                "(≥60%);\n   a reason we moved to HDBSCAN and report per-theme "
                "sources.",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                fontstyle="italic", color="#a33",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdeded", ec="#e0b4b4"))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIG / "org_bias.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[bias] top org {top[0][0]} = {100*top[0][1]/n:.1f}%  -> {out.name}")


def main() -> None:
    enriched, full = _load()
    coverage_figure(enriched, full)
    bias_figure(enriched, full)


if __name__ == "__main__":
    main()
