"""Honest, reproducible evaluation of the enrichment pipeline.

Follows the 8-point evaluation plan. Unlike run.py's headline metrics, this
script actively stress-tests the claims:

* a *fair* TF-IDF/BM25-style lexical baseline (not just naive word-count)
* Precision@5 on natural-language policy queries incl. a synonym/acronym test
* clustering *stability* via Adjusted Rand Index across seeds (honest, not the
  near-tautological "theme recovery" F1)
* quantified source (Croydon) contamination per theme
* near-duplicate precision at several thresholds + exact-title cross-check

Writes artifacts/evaluation_full.json and prints a readable report.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.data import load_catalogue, quality_report
from src.enrich import EXTRA_STOP, auto_tags, cluster_themes, name_themes
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)
K = 15


def sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- embeddings
def get_embeddings(texts: list[str]):
    cache = ART / "doc_emb.npy"
    if cache.exists():
        print(f"[emb] loading cached {cache.name}")
        return np.load(cache)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")
    np.save(cache, emb)
    return emb


def encode_query(q: str):
    from sentence_transformers import SentenceTransformer

    global _QMODEL
    try:
        _QMODEL
    except NameError:
        _QMODEL = SentenceTransformer("all-MiniLM-L6-v2")
    v = _QMODEL.encode([q], normalize_embeddings=True).astype("float32")
    return v[0]


# ---------------------------------------------------------------- searches
def kw_search(df, texts_lower, query, top_k=5):
    """Naive raw word-count keyword search (the pipeline's current baseline)."""
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    scores = np.zeros(len(df), dtype=int)
    for w in words:
        scores += np.array([t.count(w) for t in texts_lower])
    order = np.argsort(scores)[::-1][:top_k]
    return [df.iloc[i]["title"] for i in order]


def build_tfidf(texts):
    stop = sorted(ENGLISH_STOP_WORDS | EXTRA_STOP)
    vec = TfidfVectorizer(stop_words=stop, max_features=20000, ngram_range=(1, 2),
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    return vec, X


def tfidf_search(df, vec, X, query, top_k=5):
    """Fair lexical baseline: TF-IDF cosine (length-normalised, IDF-weighted)."""
    q = vec.transform([query])
    sims = (X @ q.T).toarray().ravel()
    order = np.argsort(sims)[::-1][:top_k]
    return [df.iloc[i]["title"] for i in order]


def semantic_search(df, doc_emb, query, top_k=5):
    q = encode_query(query)
    q = q / (np.linalg.norm(q) or 1.0)
    sims = doc_emb @ q
    order = np.argsort(sims)[::-1][:top_k]
    return [df.iloc[i]["title"] for i in order]


# ---------------------------------------------------------------- duplicates
def count_pairs_over(doc_emb, thresholds, block=1024):
    n = doc_emb.shape[0]
    counts = {t: 0 for t in thresholds}
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = doc_emb[start:end] @ doc_emb.T
        for li, gi in enumerate(range(start, end)):
            row = sims[li]
            row[: gi + 1] = -1  # keep i < j only
            for t in thresholds:
                counts[t] += int(np.count_nonzero(row >= t))
    return counts


def main() -> None:
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    texts = df["search_text"].tolist()
    texts_lower = [t.lower() for t in texts]

    # ---------------------------------------------------------------- P1
    sep("P1  DATA QUALITY")
    qr = quality_report(df)
    org_counts = df["organization"].value_counts()
    croydon_share = round(100 * org_counts.iloc[0] / len(df), 1)
    norm_titles = df["title"].str.strip().str.lower()
    tvc = norm_titles.value_counts()
    dup_groups = int((tvc > 1).sum())
    dup_records = int(tvc[tvc > 1].sum())
    p1 = {
        **qr,
        "top_org": org_counts.index[0],
        "top_org_share_pct": croydon_share,
        "exact_dup_title_groups": dup_groups,
        "exact_dup_title_records": dup_records,
    }
    print(json.dumps(p1, indent=2, default=str))

    # ---------------------------------------------------------------- embed
    sep("EMBEDDING")
    doc_emb = get_embeddings(texts)
    print("shape:", doc_emb.shape)

    # ---------------------------------------------------------------- P5 cluster
    sep("P5  CLUSTERING  (reproduce seed=0,k=15 themes)")
    labels0, _ = cluster_themes(doc_emb, k=K, seed=0)
    names0 = name_themes(texts, labels0)
    theme_org = {}
    org_driven = 0
    for c in sorted(names0):
        sub = df[labels0 == c]
        vc = sub["organization"].value_counts()
        top_org, top_n = vc.index[0], int(vc.iloc[0])
        share = round(100 * top_n / len(sub), 1)
        if share >= 60:
            org_driven += 1
        theme_org[int(c)] = {
            "name": names0[c], "size": int(len(sub)),
            "top_org": top_org, "top_org_share_pct": share,
        }
        print(f"  T{c:2d} [{len(sub):5d}] {names0[c]:<40} top={top_org} ({share}%)")
    print(f"\n  themes with a single org >=60% (org-driven, not topic): {org_driven}/{K}")

    idx = RNG.choice(len(doc_emb), 5000, replace=False)
    sil = float(silhouette_score(doc_emb[idx], labels0[idx]))
    print(f"  silhouette(k=15) = {sil:.3f}")

    sep("P5c  CLUSTER STABILITY  (Adjusted Rand Index across seeds)")
    seeds = [1, 2, 3, 42]
    labels_by_seed = {0: labels0}
    for s in seeds:
        labels_by_seed[s], _ = cluster_themes(doc_emb, k=K, seed=s)
    aris = [round(adjusted_rand_score(labels0, labels_by_seed[s]), 3) for s in seeds]
    print(f"  ARI(seed0 vs {seeds}) = {aris}")
    print(f"  mean ARI = {round(float(np.mean(aris)), 3)}  (1.0=identical, 0=random)")

    sep("P5  K-SENSITIVITY  (silhouette vs k)")
    k_sil = {}
    for k in [8, 10, 12, 15, 20]:
        lab, _ = cluster_themes(doc_emb, k=k, seed=0)
        s = float(silhouette_score(doc_emb[idx], lab[idx]))
        k_sil[k] = round(s, 3)
        print(f"  k={k:2d}  silhouette={s:.3f}")

    # ---------------------------------------------------------------- P2/P3 search
    sep("P2/P3  SEARCH: keyword vs TF-IDF vs semantic")
    vec, X = build_tfidf(texts)
    queries = [
        "young people not in work",           # NEET synonym/acronym test
        "energy efficiency retrofit of homes",
        "air quality and pollution",
        "affordable housing and homelessness",
        "childhood obesity",
        "domestic abuse against women",
        "cycling and walking infrastructure",
        "household recycling and waste",
    ]
    search_out = {}
    for q in queries:
        kw = kw_search(df, texts_lower, q)
        tf = tfidf_search(df, vec, X, q)
        se = semantic_search(df, doc_emb, q)
        search_out[q] = {"keyword": kw, "tfidf": tf, "semantic": se}
        print(f"\nQUERY: {q!r}")
        print("  [keyword] ", kw)
        print("  [tfidf]   ", tf)
        print("  [semantic]", se)

    # NEET availability check
    neet_mask = df["title"].str.contains("NEET", case=False, na=False) | \
        df["search_text"].str.contains("not in education", case=False, na=False)
    print(f"\n  datasets mentioning NEET / 'not in education': {int(neet_mask.sum())}")
    if neet_mask.any():
        print("   examples:", df[neet_mask]["title"].head(5).tolist())

    # ---------------------------------------------------------------- P6 dupes
    sep("P6  NEAR-DUPLICATE DETECTION")
    counts = count_pairs_over(doc_emb, [0.99, 0.97, 0.95, 0.90])
    print("  pairs with cosine >=", counts)
    # cross-check: of the very top pairs, how many are exact-title matches?
    from src.enrich import find_near_duplicates
    top = find_near_duplicates(doc_emb, threshold=0.95, max_pairs=50)
    exact = sum(1 for i, j, s in top
                if norm_titles.iloc[i] == norm_titles.iloc[j])
    nonexact = [(df.iloc[i]["title"], df.iloc[j]["title"], round(s, 3))
                for i, j, s in top if norm_titles.iloc[i] != norm_titles.iloc[j]][:5]
    print(f"  of top 50 pairs (>=0.95): {exact} are identical titles, "
          f"{50 - exact} are non-identical near-dupes")
    print("  non-identical near-dupe examples:")
    for a, b, s in nonexact:
        print(f"    {s}  {a[:45]!r}  ~  {b[:45]!r}")

    # ---------------------------------------------------------------- P4 tags
    sep("P4  AUTO-TAG SAMPLE  (eyeball relevance)")
    tags = auto_tags(texts)
    sample_idx = RNG.choice(len(df), 10, replace=False)
    tag_sample = []
    for i in sample_idx:
        rec = {"title": df.iloc[i]["title"][:70], "tags": tags[i]}
        tag_sample.append(rec)
        print(f"  - {rec['title']}\n      -> {rec['tags']}")

    # ---------------------------------------------------------------- P7 skore
    sep("P7  SKORE METHODOLOGICAL CHECKS")
    skore_checks = _run_skore_checks(doc_emb, labels0)

    # ---------------------------------------------------------------- dump
    out = {
        "p1_quality": p1,
        "p5_themes": theme_org,
        "p5_org_driven_count": org_driven,
        "p5_silhouette_k15": round(sil, 3),
        "p5c_ari_vs_seed0": dict(zip(map(str, seeds), aris)),
        "p5c_ari_mean": round(float(np.mean(aris)), 3),
        "p5_k_silhouette": k_sil,
        "p23_search": search_out,
        "p6_dup_counts": {str(k): v for k, v in counts.items()},
        "p6_top50_identical_titles": exact,
        "p4_tag_sample": tag_sample,
        "p7_skore_checks": skore_checks,
    }
    (ART / "evaluation_full.json").write_text(json.dumps(out, indent=2, default=str,
                                                         ensure_ascii=False))
    sep("DONE -> artifacts/evaluation_full.json")


def _run_skore_checks(doc_emb, labels):
    try:
        import skore
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        Xtr, Xte, ytr, yte = train_test_split(
            doc_emb, labels, test_size=0.25, random_state=0, stratify=labels)
        rep = skore.EstimatorReport(
            LogisticRegression(max_iter=1000), X_train=Xtr, y_train=ytr,
            X_test=Xte, y_test=yte)
        summary = None
        try:
            summary = str(rep.checks.summarize())
        except Exception as e:  # API may differ across skore versions
            summary = f"checks.summarize() unavailable: {e}"
        print(summary)
        return {"available": True, "summary": summary}
    except Exception as e:
        print("skore unavailable:", e)
        return {"available": False, "error": str(e)}


if __name__ == "__main__":
    main()
