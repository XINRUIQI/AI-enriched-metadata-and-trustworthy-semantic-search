"""Enrich the catalogue: embed -> cluster into themes -> auto-tag.

This is the core "metadata enrichment" step. 100% of datasets have no theme
and 92% have no tags, so we derive both from the free text:

* ``embed_texts``   -> dense semantic vectors (sentence-transformers) with a
                       TF-IDF fallback so the pipeline always runs.
* ``cluster_themes``-> KMeans over the embeddings, each cluster = a theme.
* ``name_themes``   -> label each theme with its top TF-IDF keywords.
* ``auto_tags``     -> per-dataset keywords to fill the missing ``tags``.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# A compact set of English stop-words plus catalogue-specific noise terms that
# otherwise dominate every cluster (they describe *all* London datasets).
EXTRA_STOP = {
    "data", "dataset", "datasets", "london", "borough", "council",
    "information", "author", "date", "source", "creation", "coverage",
    "temporal", "spatial", "resolution", "url", "harvest", "n", "geometry",
}


def _stopwords() -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return sorted(ENGLISH_STOP_WORDS | EXTRA_STOP)


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """Return (embeddings, backend_name).

    Uses sentence-transformers when available (much better semantics), and
    falls back to TF-IDF so the demo never hard-fails.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        emb = model.encode(
            texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
        )
        return np.asarray(emb, dtype="float32"), f"sentence-transformers/{model_name}"
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[embed] sentence-transformers unavailable ({exc}); using TF-IDF")
        vec = TfidfVectorizer(
            stop_words=_stopwords(), max_features=4096, ngram_range=(1, 2)
        )
        emb = vec.fit_transform(texts).astype("float32")
        return emb, "tfidf"


def cluster_themes(embeddings, k: int = 15, method: str = "kmeans", seed: int = 0,
                   min_cluster_size: int = 60):
    """Cluster datasets into themes.

    method="kmeans"  -> partitions every dataset into exactly ``k`` themes.
    method="hdbscan" -> density-based; discovers the number of themes on its
                        own and labels sparse outliers as noise (label -1),
                        so we don't force unrelated datasets into a theme.

    Embeddings are L2-normalised, so Euclidean distance is monotonic with
    cosine distance -- HDBSCAN's default metric matches our semantic space.
    Returns (labels, fitted_model).
    """
    if method == "hdbscan":
        hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10)
        labels = hdb.fit_predict(embeddings)
        return labels, hdb
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km


def name_themes(texts: list[str], labels, top_n: int = 4) -> dict[int, str]:
    """Give each cluster a human-readable name from its top TF-IDF terms."""
    vec = TfidfVectorizer(stop_words=_stopwords(), max_features=6000)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    names: dict[int, str] = {}
    for c in sorted(set(labels)):
        if c == -1:  # HDBSCAN noise / unclustered
            names[c] = "(unclustered / noise)"
            continue
        mask = labels == c
        centroid = np.asarray(X[mask].mean(axis=0)).ravel()
        top = vocab[centroid.argsort()[::-1][:top_n]]
        names[c] = ", ".join(top)
    return names


def find_near_duplicates(embeddings, threshold: float = 0.97, max_pairs: int = 200):
    """Find near-duplicate datasets via cosine similarity of embeddings.

    Returns a list of (i, j, similarity) with i < j, sorted most-similar first.
    Uses a blocked matmul so a 20k x 20k matrix is never materialised.
    """
    X = embeddings.toarray() if hasattr(embeddings, "toarray") else np.asarray(embeddings)
    X = X.astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    n = X.shape[0]
    pairs: list[tuple[int, int, float]] = []
    block = 1024
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = X[start:end] @ X.T  # (block, n)
        for local_i, global_i in enumerate(range(start, end)):
            row = sims[local_i]
            cand = np.where(row >= threshold)[0]
            for j in cand:
                if j > global_i:  # keep i < j, skip self
                    pairs.append((int(global_i), int(j), float(row[j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:max_pairs]


def auto_tags(texts: list[str], top_n: int = 5) -> list[list[str]]:
    """Per-document keyword tags via TF-IDF (fills the 92% with no tags)."""
    vec = TfidfVectorizer(
        stop_words=_stopwords(), max_features=8000, ngram_range=(1, 2)
    )
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    tags: list[list[str]] = []
    X = X.tocsr()
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            tags.append([])
            continue
        idx = row.indices[row.data.argsort()[::-1][:top_n]]
        tags.append(list(vocab[idx]))
    return tags
