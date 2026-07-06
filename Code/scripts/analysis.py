"""Compare clustering methods to justify our choices (for the pitch).

Embeds once (cached to artifacts/embeddings.npy), then:
  1. KMeans k-sweep -> silhouette per k (justifies "why not just pick k").
  2. HDBSCAN       -> auto #themes, noise %, silhouette, theme-recovery F1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_catalogue
from src.enrich import embed_texts

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artifacts" / "embeddings.npy"


def get_embeddings():
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    if CACHE.exists():
        emb = np.load(CACHE)
        if emb.shape[0] == len(df):
            print(f"[cache] loaded embeddings {emb.shape}")
            return df, emb
    emb, _ = embed_texts(df["search_text"].tolist())
    emb = np.asarray(emb, dtype="float32")
    np.save(CACHE, emb)
    return df, emb


def recovery_scores(emb, labels):
    keep = np.asarray(labels) != -1
    X, y = emb[keep], np.asarray(labels)[keep]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")


def main() -> None:
    df, emb = get_embeddings()
    rng = np.random.default_rng(0)
    idx = rng.choice(emb.shape[0], size=min(5000, emb.shape[0]), replace=False)

    print("\n=== KMeans k-sweep (silhouette) ===")
    print(f"{'k':>4} | {'silhouette':>11}")
    for k in [8, 10, 12, 15, 20, 25, 30]:
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(emb)
        sil = silhouette_score(emb[idx], labels[idx])
        print(f"{k:>4} | {sil:>11.4f}")

    print("\n=== HDBSCAN (auto) ===")
    labels = HDBSCAN(min_cluster_size=60, min_samples=10).fit_predict(emb)
    n_themes = len(set(labels) - {-1})
    noise = (labels == -1).mean()
    keep = labels != -1
    emb_k, lab_k = emb[keep], labels[keep]
    s_idx = rng.choice(emb_k.shape[0], size=min(5000, emb_k.shape[0]), replace=False)
    sil = silhouette_score(emb_k[s_idx], lab_k[s_idx])
    acc, f1 = recovery_scores(emb, labels)
    print(f"themes={n_themes}  noise={noise:.1%}  silhouette={sil:.4f}")
    print(f"theme-recovery: accuracy={acc:.4f}  macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
