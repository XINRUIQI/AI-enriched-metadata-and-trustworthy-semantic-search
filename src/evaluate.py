"""Evaluate the enrichment with skore -- 'can you vouch for it?'.

Two complementary angles:

1. Clustering quality (unsupervised, honest): silhouette score on the
   embeddings tells us whether the themes are geometrically well separated.

2. Theme coherence (supervised recovery via skore): we train a classifier to
   predict the auto-assigned theme from the text embedding on a held-out test
   set. If a held-out model recovers the themes with high F1, the themes are
   coherent and learnable -- i.e. we can *vouch* for them. skore produces the
   metrics, methodological checks and visualisations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


def clustering_quality(embeddings, labels, sample: int = 5000, seed: int = 0):
    """Silhouette score on a sample (full matrix is O(n^2)).

    Noise points (label -1 from HDBSCAN) are excluded -- silhouette is only
    defined over assigned clusters.
    """
    emb = embeddings.toarray() if hasattr(embeddings, "toarray") else np.asarray(embeddings)
    labels = np.asarray(labels)
    keep = labels != -1
    emb, labels = emb[keep], labels[keep]
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    return float(silhouette_score(emb[idx], labels[idx]))


def theme_report(embeddings, labels, seed: int = 0):
    """Build a skore EstimatorReport for theme recovery.

    Returns (report, X_test, y_test) so the caller can pull metrics/displays.
    """
    import skore

    X = embeddings.toarray() if hasattr(embeddings, "toarray") else np.asarray(embeddings)
    y = np.asarray(labels)
    keep = y != -1  # exclude HDBSCAN noise from the supervised recovery
    X, y = X[keep], y[keep]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    report = skore.EstimatorReport(
        clf, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te
    )
    return report, X_te, y_te


def save_confusion_matrix(report, out_path: str | Path):
    """Persist the skore confusion-matrix display for the pitch deck."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp = report.metrics.confusion_matrix()
    disp.plot()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return out_path
