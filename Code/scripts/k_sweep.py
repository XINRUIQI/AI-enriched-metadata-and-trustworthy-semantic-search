"""Sweep the number of themes k and report silhouette to justify our choice.

Embeddings are computed once, then reused across all k (KMeans is cheap).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_catalogue
from src.enrich import embed_texts

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    emb, backend = embed_texts(df["search_text"].tolist())
    emb = np.asarray(emb, dtype="float32")

    rng = np.random.default_rng(0)
    idx = rng.choice(emb.shape[0], size=min(5000, emb.shape[0]), replace=False)
    sample = emb[idx]

    print(f"backend={backend}  n={emb.shape[0]}\n")
    print(f"{'k':>4} | {'silhouette':>11} | {'inertia':>12}")
    print("-" * 34)
    for k in [6, 8, 10, 12, 15, 18, 20, 25, 30]:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(emb)
        sil = silhouette_score(sample, labels[idx])
        print(f"{k:>4} | {sil:>11.4f} | {km.inertia_:>12.1f}")


if __name__ == "__main__":
    main()
