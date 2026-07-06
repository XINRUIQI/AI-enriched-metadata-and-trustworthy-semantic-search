"""Baseline keyword search vs. embedding-based semantic search.

The current Data for London Library only supports keyword search. We show a
side-by-side: the same query run through (a) simple substring/keyword matching
and (b) cosine similarity over the enriched embeddings.
"""
from __future__ import annotations

import re

import numpy as np


def keyword_search(df, query: str, top_k: int = 5):
    """Naive keyword search: count query-word hits in the search text.

    This mimics the current 'keyword only' library behaviour and is the
    baseline our semantic search is compared against.
    """
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    text = df["search_text"].str.lower()
    scores = np.zeros(len(df), dtype=int)
    for w in words:
        scores += text.str.count(re.escape(w)).to_numpy()
    order = np.argsort(scores)[::-1][:top_k]
    return df.iloc[order].assign(score=scores[order])


class SemanticSearch:
    """Cosine-similarity search over dataset embeddings."""

    def __init__(self, df, embeddings, embed_fn):
        self.df = df
        self.emb = _l2_normalize(embeddings)
        self._embed_fn = embed_fn  # maps [text] -> (vectors, backend)

    def query(self, text: str, top_k: int = 5):
        q, _ = self._embed_fn([text])
        q = _l2_normalize(_dense(q))
        sims = (self.emb @ q.T).ravel()
        order = np.argsort(sims)[::-1][:top_k]
        return self.df.iloc[order].assign(score=np.round(sims[order], 3))


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _l2_normalize(x):
    x = _dense(x).astype("float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
