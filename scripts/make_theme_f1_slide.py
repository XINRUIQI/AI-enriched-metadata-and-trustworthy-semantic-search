"""Slide-friendly replacement for the unreadable 76x76 confusion matrix.

Produces artifacts/figures/theme_f1_slide.png:

  * LEFT  : per-theme F1 (held-out 25%) as sorted horizontal bars, coloured
            green->red, with macro / median reference lines. Shows the *spread*
            honestly instead of an illegible 76-class grid.
  * RIGHT : a "skore methodological audit" text panel -- the real warnings that
            skore.train_test_split emitted, the headline metrics, the weakest
            themes to hand-review, and the honesty caveat (F1 = reproducibility,
            NOT business correctness).

Reuses the cached MiniLM embeddings in artifacts/doc_emb.npy and reproduces the
main pipeline's HDBSCAN themes (min_cluster_size=60, deterministic).
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_catalogue  # noqa: E402
from src.enrich import cluster_themes, name_themes  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Short, human-readable summary for each skore warning class we might see.
SKORE_WARNING_SUMMARY = {
    "HighClassImbalanceTooFewExamplesWarning":
        "some themes have <100 datasets in the test set -> scores are noisy",
    "HighClassImbalanceWarning":
        "class sizes are very uneven (theme sizes range widely)",
    "ShuffleTrueWarning":
        "shuffle=True can inflate scores if data is time-ordered",
    "StratifyIsSetWarning": "stratify was set -> check leakage assumptions",
    "RandomStateUnsetWarning": "random_state unset -> results not reproducible",
    "TimeBasedColumnWarning": "a time-like column was detected",
}


def skore_split_and_metrics(X, y, test_size=0.25, seed=0):
    """Run skore.train_test_split (captures its methodological warnings), then
    fit a held-out classifier and return per-class F1 + headline metrics.
    """
    import skore

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        split = skore.train_test_split(
            X=X, y=y, test_size=test_size, random_state=seed, as_dict=True
        )
    captured = buf.getvalue()
    # skore prints rich panels titled with the warning class name.
    fired = [w for w in SKORE_WARNING_SUMMARY if w in captured]

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(split["X_train"], split["y_train"])
    y_pred = clf.predict(split["X_test"])
    y_test = split["y_test"]

    classes = np.unique(y)
    f1_per = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))
    per_class = dict(zip(classes.tolist(), f1_per.tolist()))
    return per_class, macro, acc, len(y_test), fired


def main() -> None:
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    texts = df["search_text"].tolist()
    doc_emb = np.load(ART / "doc_emb.npy")
    assert doc_emb.shape[0] == len(df), "embeddings/rows misaligned"

    # Reproduce the main pipeline's themes (HDBSCAN, deterministic).
    labels, _ = cluster_themes(doc_emb, method="hdbscan", min_cluster_size=60)
    labels = np.asarray(labels)
    names = name_themes(texts, labels)

    keep = labels != -1  # drop unclustered noise from the supervised recovery
    X, y = doc_emb[keep], labels[keep]
    n_themes = len(set(y.tolist()))
    print(f"[themes] {n_themes} themes; {int((~keep).sum())} noise rows dropped")

    per_class, macro, acc, n_test, fired = skore_split_and_metrics(X, y)
    print(f"[metrics] macro-F1={macro:.3f}  accuracy={acc:.3f}  n_test={n_test}")
    print(f"[skore] warnings fired: {fired}")

    # Sort themes by F1 (ascending -> weakest at the bottom of the bar chart).
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    f1_vals = [v for _, v in items]
    median = float(np.median(f1_vals))
    weakest = items[:5]  # lowest-F1 themes -> hand-review first

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.08)

    # LEFT: sorted per-theme F1 bars
    axL = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in f1_vals]
    ys = np.arange(len(f1_vals))
    axL.barh(ys, f1_vals, color=colors, height=0.85)
    axL.axvline(macro, color="#1f4e79", ls="--", lw=2,
                label=f"macro-F1 = {macro:.3f}")
    axL.axvline(median, color="#7a7a7a", ls=":", lw=1.6,
                label=f"median = {median:.3f}")
    axL.set_xlim(0, 1.02)
    axL.set_ylim(-1, len(f1_vals))
    axL.set_yticks([])
    axL.set_xlabel("held-out F1 score")
    axL.set_ylabel(f"{n_themes} themes  (sorted: weakest at bottom)")
    axL.set_title("Per-theme recovery F1 (held-out 25%)\n"
                  "replaces the illegible 76x76 confusion matrix",
                  fontweight="bold", fontsize=12)
    axL.legend(loc="lower right", fontsize=10)
    axL.grid(axis="x", alpha=0.3)

    # RIGHT: skore methodological audit text panel
    axR = fig.add_subplot(gs[0, 1])
    axR.axis("off")

    def wrap(txt, n=52):
        out, line = [], ""
        for w in txt.split():
            if len(line) + len(w) + 1 > n:
                out.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            out.append(line)
        return out

    lines: list[tuple[str, dict]] = []
    big = {"fontsize": 13, "fontweight": "bold"}
    hdr = {"fontsize": 11.5, "fontweight": "bold", "color": "#1f4e79"}
    norm = {"fontsize": 10.5}
    small = {"fontsize": 9.5, "color": "#444"}

    lines.append(("skore-style methodological audit", big))
    lines.append(("", norm))
    lines.append((f"{n_themes} themes  |  macro-F1 = {macro:.3f}  |  "
                  f"accuracy = {acc:.3f}", {"fontsize": 11, "fontweight": "bold"}))
    lines.append((f"held-out test set: {n_test:,} datasets (25%)", small))
    lines.append(("", norm))

    lines.append(("skore.train_test_split flagged:", hdr))
    if fired:
        for w in fired:
            for i, seg in enumerate(wrap("- " + SKORE_WARNING_SUMMARY[w])):
                lines.append((seg if i == 0 else "  " + seg, norm))
    else:
        lines.append(("(no warnings fired)", norm))
    lines.append(("", norm))

    lines.append(("Weakest themes -> hand-review first:", hdr))
    for cid, v in weakest:
        nm = names.get(cid, str(cid))
        nm = (nm[:34] + "...") if len(nm) > 34 else nm
        lines.append((f"- F1 {v:.2f}  T{cid}: {nm}", norm))
    lines.append(("", norm))

    lines.append(("Honest caveat", hdr))
    for seg in wrap("F1 proves themes are LEARNABLE / REPRODUCIBLE from the "
                    "embeddings -- NOT that a theme/tag is the RIGHT one for a "
                    "dataset. That is a human call: see human_review_template.csv."):
        lines.append((seg, {"fontsize": 10, "style": "italic", "color": "#7a3b00"}))

    y = 0.99
    for txt, style in lines:
        step = 0.052 if style.get("fontsize", 10) >= 12 else 0.041
        if txt:
            axR.text(0.0, y, txt, va="top", ha="left", transform=axR.transAxes,
                     **style)
        y -= step

    # subtle border around the right panel
    axR.add_patch(plt.Rectangle((-0.02, -0.01), 1.02, 1.01, transform=axR.transAxes,
                                fill=False, edgecolor="#c9c9c9", lw=1.2,
                                clip_on=False))

    fig.suptitle("London Data Week - Theme quality & skore methodological check",
                 fontsize=15, fontweight="bold", y=1.0)
    out = FIG / "theme_f1_slide.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
