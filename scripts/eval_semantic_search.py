"""Evaluate semantic search vs keyword baselines with human relevance judgments.

The point of this script is NOT to let the model grade itself. It builds a
*pooled scorecard* from real natural-language policy queries, a human labels
each pooled result 0/1/2, and only then do we compute the standard IR metrics
(Precision@5, Precision@10, MRR, nDCG@10). That is what makes the claim
"semantic search is more relevant" defensible rather than self-asserted.

Workflow (three steps)::

    # 1. retrieve top-K for each method and pool the results into a scorecard
    python scripts/eval_semantic_search.py build

    # 2. judge relevance. Either hand-edit the `relevance` column in
    #    artifacts/relevance_scorecard.csv, OR write the judgments in
    #    artifacts/relevance_judgments.json and apply them:
    python scripts/eval_semantic_search.py apply

    # 3. compute the metrics table
    python scripts/eval_semantic_search.py score

Relevance scale (manual): 0 = not relevant, 1 = somewhat, 2 = highly relevant.
A result counts as "relevant" for Precision@k / MRR when its grade >= 1.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_catalogue  # noqa: E402
from src.enrich import EXTRA_STOP  # noqa: E402
from src.search import keyword_search  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # noqa: E402

ART = ROOT / "artifacts"
EMB_CACHE = ART / "doc_emb.npy"
SCORECARD_CSV = ART / "relevance_scorecard.csv"
SCORECARD_MD = ART / "search_scorecard.md"
JUDGMENTS_JSON = ART / "relevance_judgments.json"
RESULTS_MD = ART / "search_eval_results.md"
RESULTS_JSON = ART / "search_eval_results.json"

# GLA-policy-relevant natural-language queries (not keywords).
QUERIES = [
    "young people not in work",
    "air pollution near schools",
    "housing affordability by borough",
    "cycling accidents in London",
    "energy inefficient homes",
    "population change in outer London",
    "crime around transport stations",
    "access to green space",
]

TOP_K = 10
MODEL_NAME = "all-MiniLM-L6-v2"
METHODS = [("keyword", "rank_keyword"), ("tfidf", "rank_tfidf"), ("semantic", "rank_semantic")]


# --------------------------------------------------------------------- retrieval
def _build_tfidf(texts):
    stop = sorted(ENGLISH_STOP_WORDS | EXTRA_STOP)
    vec = TfidfVectorizer(stop_words=stop, max_features=20000, ngram_range=(1, 2),
                          sublinear_tf=True)
    return vec, vec.fit_transform(texts)


def _rank_keyword(df, query, k):
    return list(keyword_search(df, query, top_k=k).index)


def _rank_tfidf(vec, X, query, k):
    sims = (X @ vec.transform([query]).T).toarray().ravel()
    return list(np.argsort(sims)[::-1][:k])


def _rank_semantic(doc_emb, model, query, k):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")[0]
    sims = doc_emb @ qv
    return list(np.argsort(sims)[::-1][:k])


# --------------------------------------------------------------------- build
def build() -> None:
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    texts = df["search_text"].tolist()
    doc_emb = np.load(EMB_CACHE)
    assert len(df) == doc_emb.shape[0], "catalogue / embedding row mismatch"

    vec, X = _build_tfidf(texts)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    rows = []
    md = [
        "# Search scorecard — keyword vs TF-IDF vs semantic\n",
        "Top-10 per method for each query. To judge relevance, fill the "
        "`relevance` column in `relevance_scorecard.csv` (or edit "
        "`relevance_judgments.json`): **0 = not relevant, 1 = somewhat, "
        "2 = highly relevant**.\n",
    ]

    for q in QUERIES:
        kw = _rank_keyword(df, q, TOP_K)
        tf = _rank_tfidf(vec, X, q, TOP_K)
        se = _rank_semantic(doc_emb, model, q, TOP_K)
        rank = {
            "keyword": {idx: r + 1 for r, idx in enumerate(kw)},
            "tfidf": {idx: r + 1 for r, idx in enumerate(tf)},
            "semantic": {idx: r + 1 for r, idx in enumerate(se)},
        }
        pool = sorted(
            set(kw) | set(tf) | set(se),
            key=lambda i: min(rank["keyword"].get(i, 99),
                              rank["tfidf"].get(i, 99),
                              rank["semantic"].get(i, 99)),
        )
        for i in pool:
            rows.append({
                "query": q,
                "rank_keyword": rank["keyword"].get(i, ""),
                "rank_tfidf": rank["tfidf"].get(i, ""),
                "rank_semantic": rank["semantic"].get(i, ""),
                "title": df.iloc[i]["title"],
                "organization": df.iloc[i]["organization"],
                "description": " ".join(df.iloc[i]["notes"][:200].split()),
                "relevance": "",
                "doc_id": df.iloc[i]["id"],
            })

        md.append(f"\n## {q}\n")
        md.append("| Rank | Keyword | TF-IDF | Semantic |")
        md.append("|---:|---|---|---|")
        for r in range(TOP_K):
            cell = lambda lst: (df.iloc[lst[r]]["title"] if r < len(lst) else "")
            md.append(f"| {r + 1} | {cell(kw)} | {cell(tf)} | {cell(se)} |")

    pd.DataFrame(rows).to_csv(SCORECARD_CSV, index=False)
    SCORECARD_MD.write_text("\n".join(md) + "\n")
    n_pool = len(rows)
    print(f"[build] {len(QUERIES)} queries, {n_pool} pooled results to judge")
    print(f"[build] wrote {SCORECARD_CSV.relative_to(ROOT)}")
    print(f"[build] wrote {SCORECARD_MD.relative_to(ROOT)}")


# --------------------------------------------------------------------- apply
def _norm(s: str) -> str:
    """Normalise a title for robust matching (case, whitespace, quotes, dashes)."""
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s))
    for a, b in [("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'),
                 ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-"),
                 ("\u2212", "-")]:
        s = s.replace(a, b)
    return " ".join(s.split()).casefold()


def apply() -> None:
    """Fill the CSV `relevance` column from relevance_judgments.json.

    JSON shape: {query: {dataset_title: grade, ...}, ...}. Titles are matched
    after normalisation (case/whitespace/quotes/dashes), so you can type plain
    ASCII. Any pooled result not listed defaults to 0 (not relevant).
    """
    if not JUDGMENTS_JSON.exists():
        raise SystemExit(f"missing {JUDGMENTS_JSON}; create it first (see --help)")
    judgments = json.loads(JUDGMENTS_JSON.read_text())
    df = pd.read_csv(SCORECARD_CSV)
    df["relevance"] = 0
    df["_qn"] = df["query"].map(_norm)
    df["_tn"] = df["title"].map(_norm)

    matched = unmatched = 0
    misses = []
    for q, title_grades in judgments.items():
        qn = _norm(q)
        for title, grade in title_grades.items():
            mask = (df["_qn"] == qn) & (df["_tn"] == _norm(title))
            hits = int(mask.sum())
            if hits:
                df.loc[mask, "relevance"] = grade
                matched += hits
            else:
                unmatched += 1
                misses.append((q, title))
    df = df.drop(columns=["_qn", "_tn"])
    df.to_csv(SCORECARD_CSV, index=False)
    print(f"[apply] set relevance on {matched} rows; {unmatched} judgment "
          f"entries matched no pooled result")
    for q, t in misses[:10]:
        print(f"        unmatched: [{q}] {t!r}")
    graded = int((df["relevance"] >= 1).sum())
    print(f"[apply] pool={len(df)}  relevant(>=1)={graded}  "
          f"not-relevant={len(df) - graded}")


# --------------------------------------------------------------------- score
def _metrics(ranked_grades, pool_grades):
    rel = [1 if g >= 1 else 0 for g in ranked_grades]
    p5 = sum(rel[:5]) / 5.0
    p10 = sum(rel[:10]) / 10.0
    mrr = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ranked_grades[:10]))
    idcg = sum(g / math.log2(i + 2)
               for i, g in enumerate(sorted(pool_grades, reverse=True)[:10]))
    ndcg = dcg / idcg if idcg else 0.0
    return {"p5": p5, "p10": p10, "mrr": mrr, "ndcg10": ndcg,
            "rel5": sum(rel[:5]), "rel10": sum(rel[:10])}


def score() -> None:
    df = pd.read_csv(SCORECARD_CSV)
    df["grade"] = pd.to_numeric(df["relevance"], errors="coerce")
    n_blank = int(df["grade"].isna().sum())
    df["grade"] = df["grade"].fillna(0).astype(int)
    if n_blank:
        print(f"[score] warning: {n_blank} rows had no relevance label "
              f"(treated as 0)")

    per_query, agg = {}, {m: [] for m, _ in METHODS}
    for q, g in df.groupby("query", sort=False):
        pool_grades = g["grade"].tolist()
        per_query[q] = {}
        for m, col in METHODS:
            sub = g[pd.to_numeric(g[col], errors="coerce").notna()].copy()
            sub["_r"] = pd.to_numeric(sub[col])
            ranked = sub.sort_values("_r")["grade"].tolist()
            mt = _metrics(ranked, pool_grades)
            per_query[q][m] = mt
            agg[m].append(mt)

    summary = {
        m: {k: round(float(np.mean([a[k] for a in agg[m]])), 3)
            for k in ["p5", "p10", "mrr", "ndcg10"]}
        for m, _ in METHODS
    }

    RESULTS_JSON.write_text(json.dumps(
        {"queries": QUERIES, "per_query": per_query, "summary": summary},
        indent=2, ensure_ascii=False))
    _write_results_md(per_query, summary)
    print(f"[score] wrote {RESULTS_MD.relative_to(ROOT)}")
    print(f"[score] wrote {RESULTS_JSON.relative_to(ROOT)}")
    print("\n=== AVERAGE OVER {} QUERIES ===".format(len(QUERIES)))
    print(f"{'method':<10}{'P@5':>7}{'P@10':>7}{'MRR':>7}{'nDCG@10':>9}")
    for m, _ in METHODS:
        s = summary[m]
        print(f"{m:<10}{s['p5']:>7.2f}{s['p10']:>7.2f}{s['mrr']:>7.2f}"
              f"{s['ndcg10']:>9.2f}")


def _write_results_md(per_query, summary) -> None:
    lines = [
        "# Semantic search evaluation — results\n",
        "Relevance judged by hand on the pooled top-10 of each method "
        "(0=not, 1=somewhat, 2=highly). A result is *relevant* when grade >= 1. "
        "Metrics are the IR standards: Precision@5, Precision@10, MRR, nDCG@10.\n",
        "## Headline: averaged over all queries\n",
        "| Method | Precision@5 | Precision@10 | MRR | nDCG@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    label = {"keyword": "Keyword (current library)",
             "tfidf": "TF-IDF (fair lexical baseline)",
             "semantic": "Semantic (ours)"}
    for m, _ in METHODS:
        s = summary[m]
        lines.append(f"| {label[m]} | {s['p5']:.2f} | {s['p10']:.2f} | "
                     f"{s['mrr']:.2f} | {s['ndcg10']:.2f} |")

    lines += ["\n## Keyword baseline vs our semantic search (per query)\n",
              "| Query | Method | Top-5 relevant | Precision@5 | Precision@10 | MRR |",
              "|---|---|---:|---:|---:|---:|"]
    for q in per_query:
        for m in ("keyword", "semantic"):
            mt = per_query[q][m]
            name = "Keyword" if m == "keyword" else "Our method"
            lines.append(f"| {q} | {name} | {mt['rel5']}/5 | {mt['p5']:.2f} | "
                         f"{mt['p10']:.2f} | {mt['mrr']:.2f} |")

    lines += ["\n## Full per-query metrics (all three methods)\n",
              "| Query | Method | P@5 | P@10 | MRR | nDCG@10 |",
              "|---|---|---:|---:|---:|---:|"]
    for q in per_query:
        for m, _ in METHODS:
            mt = per_query[q][m]
            lines.append(f"| {q} | {m} | {mt['p5']:.2f} | {mt['p10']:.2f} | "
                         f"{mt['mrr']:.2f} | {mt['ndcg10']:.2f} |")
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["build", "apply", "score"])
    args = ap.parse_args()
    {"build": build, "apply": apply, "score": score}[args.mode]()


if __name__ == "__main__":
    main()
