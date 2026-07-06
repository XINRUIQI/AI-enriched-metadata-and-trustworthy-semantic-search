"""End-to-end enrichment + evaluation pipeline for the Data for London Library.

    python run.py --limit 4000 --k 15

Steps: load & clean -> embed -> cluster into themes -> auto-tag ->
evaluate with skore (silhouette + theme-recovery report) -> keyword vs
semantic search comparison. Writes artefacts to ./artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_catalogue, quality_report
from src.enrich import (
    auto_tags,
    cluster_themes,
    embed_texts,
    find_near_duplicates,
    name_themes,
)
from src.evaluate import clustering_quality, save_confusion_matrix, theme_report
from src.search import SemanticSearch, keyword_search

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

DEMO_QUERIES = [
    "energy efficiency retrofit of homes",
    "young people unemployment and training",
    "air quality and pollution",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=str(ROOT / "ckan_catalogue.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all datasets")
    ap.add_argument("--k", type=int, default=15, help="number of themes (kmeans)")
    ap.add_argument("--method", choices=["kmeans", "hdbscan"], default="hdbscan",
                    help="clustering method")
    ap.add_argument("--min-cluster-size", type=int, default=60,
                    help="hdbscan min cluster size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ART.mkdir(exist_ok=True)

    print("== 1. Load & clean ==")
    df = load_catalogue(args.catalogue)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    qr = quality_report(df)
    print("   metadata gaps (the problem):", qr)

    print("== 2. Embed ==")
    texts = df["search_text"].tolist()
    embeddings, backend = embed_texts(texts)
    print(f"   backend={backend}  shape={getattr(embeddings, 'shape', None)}")

    print(f"== 3. Cluster themes (method={args.method}) ==")
    labels, _ = cluster_themes(
        embeddings, k=args.k, method=args.method, seed=args.seed,
        min_cluster_size=args.min_cluster_size,
    )
    labels = np.asarray(labels)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"   -> {n_clusters} themes; {n_noise} unclustered ({n_noise / len(df):.1%})")
    theme_names = name_themes(texts, labels)
    df["theme_id"] = labels
    df["theme"] = df["theme_id"].map(theme_names)

    print("== 4. Auto-tag ==")
    df["auto_tags"] = auto_tags(texts)

    print("== 5. Evaluate with skore (can you vouch for it?) ==")
    sil = clustering_quality(embeddings, labels, seed=args.seed)
    report, _, _ = theme_report(embeddings, labels, seed=args.seed)
    metrics = report.metrics.report_metrics()
    cm_path = save_confusion_matrix(report, ART / "theme_confusion_matrix.png")
    print(f"   silhouette={sil:.3f}")
    print("   theme-recovery metrics:\n", metrics)
    print(f"   confusion matrix -> {cm_path}")

    print("== 5b. Near-duplicate detection ==")
    dup_pairs = find_near_duplicates(embeddings, threshold=0.97)
    dup_records = [
        {
            "similarity": round(s, 4),
            "a": df.iloc[i]["title"],
            "b": df.iloc[j]["title"],
            "org_a": df.iloc[i]["organization"],
            "org_b": df.iloc[j]["organization"],
        }
        for i, j, s in dup_pairs
    ]
    print(f"   found {len(dup_records)} near-duplicate pairs (cosine >= 0.97)")
    for d in dup_records[:3]:
        print(f"     {d['similarity']}  {d['a'][:45]!r} == {d['b'][:45]!r}")
    (ART / "near_duplicates.json").write_text(
        json.dumps(dup_records, indent=2, ensure_ascii=False)
    )

    print("== 6. Keyword vs semantic search ==")
    semantic = SemanticSearch(df, embeddings, embed_texts)
    search_dump = {}
    for q in DEMO_QUERIES:
        kw = keyword_search(df, q)[["title", "score"]].to_dict("records")
        sem = semantic.query(q)[["title", "theme", "score"]].to_dict("records")
        search_dump[q] = {"keyword": kw, "semantic": sem}
        print(f"\n   QUERY: {q!r}")
        print("   [keyword]  ", [r["title"][:60] for r in kw[:3]])
        print("   [semantic] ", [r["title"][:60] for r in sem[:3]])

    print("\n== 7. Save artefacts ==")
    _write_theme_summary(df, theme_names)
    _write_enriched_catalogue(df)
    (ART / "search_examples.json").write_text(json.dumps(search_dump, indent=2, ensure_ascii=False))
    (ART / "evaluation.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "method": args.method,
                "n_datasets": int(len(df)),
                "n_themes": int(n_clusters),
                "n_unclustered": n_noise,
                "silhouette": round(sil, 3),
                "n_near_duplicate_pairs": len(dup_records),
                "quality_before": qr,
            },
            indent=2,
        )
    )
    print(f"   artefacts written to {ART}")


def _write_theme_summary(df, theme_names) -> None:
    lines = ["# Discovered themes\n"]
    for c in sorted(theme_names):
        sub = df[df["theme_id"] == c]
        top_org = sub["organization"].value_counts().idxmax()
        lines.append(
            f"- **Theme {c}** ({len(sub)} datasets): {theme_names[c]} "
            f"— top source: {top_org}"
        )
    (ART / "themes.md").write_text("\n".join(lines))


def _write_enriched_catalogue(df) -> None:
    cols = ["id", "name", "title", "organization", "theme", "auto_tags"]
    df[cols].to_json(ART / "enriched_catalogue.json", orient="records", indent=2, force_ascii=False)


if __name__ == "__main__":
    main()
