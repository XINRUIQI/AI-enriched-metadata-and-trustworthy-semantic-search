# Discovering London's Data — metadata enrichment + semantic search

**London Data Week 2026 Hackathon · Challenge 3 — *Discovering London's Data***
*How might we improve the Data for London Library search by enriching the metadata in its catalogue?*

We take the Data for London catalogue — **20,685 datasets**, of which **100% have no theme and 92% have no tags** — and use NLP to automatically enrich the metadata, then add semantic search on top. Crucially, we don't just build it; we **evaluate it honestly** and show where it can and cannot be trusted (*"can you vouch for it?"*).

**YUYUE XIA & XINRUI QI**

---

## What it does (one vector, four jobs)

Every dataset is encoded once with a MiniLM sentence embedding. That single set of vectors powers:

1. **Theme discovery** — HDBSCAN density clustering finds **76 themes** automatically, each named with TF-IDF keywords.
2. **Auto-tagging** — fills in the 92% of datasets that had no tags (shipped as *suggestions*).
3. **Near-duplicate detection** — flags highly similar datasets for human review.
4. **Semantic search** — natural-language search that understands synonyms (e.g. "young people not in work" → NEET).

## Headline results

| Evaluation | Result |
|---|---|
| Search Precision@5 (hand-judged, 8 queries) | keyword **0.42** · TF-IDF **0.75** · **semantic 0.97** |
| Auto-tag quality (150 tags, manual review) | **78%** useful · 11% broad · 11% wrong → *suggestions only* |
| Theme separation (silhouette) | KMeans ~0.08 → **HDBSCAN 0.354** (~3.4× better) |
| Theme stability across seeds (ARI) | **0.72** (reproducible, not random) |
| Duplicate review (top-50, non-identical band) | 0% off-topic; ~12% true duplicates → *review queue* |

Full, honest write-up (including limitations) is in [`artifacts/EVALUATION.md`](artifacts/EVALUATION.md).

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# End-to-end enrichment + evaluation (all 20,685 datasets, HDBSCAN)
python run.py

# Faster smoke test on a subset
python run.py --limit 4000
```

`run.py` writes all outputs to `./artifacts/`. The MiniLM model downloads once, then runs fully offline — no API keys needed (important for government data). If `sentence-transformers` is unavailable, the pipeline falls back to TF-IDF so it always runs.

### Reproducing the evaluation figures

The evaluation numbers and slide figures are produced by the scripts in `scripts/`, e.g.:

```bash
python scripts/evaluate_final.py            # quantitative metrics -> evaluation_quant.json
python scripts/eval_semantic_search.py      # search relevance (P@5/P@10/MRR/nDCG)
python scripts/silhouette_comparison.py     # KMeans k-sweep vs HDBSCAN
python scripts/make_theme_f1_slide.py       # per-theme F1 + skore audit figure
python scripts/semantic_map.py              # UMAP 2-D map of all datasets
```

---

## Repository layout

```
run.py                     End-to-end pipeline (load → embed → cluster → tag → evaluate → search)
evaluate_report.py         Detailed evaluation report generator
requirements.txt           Pinned dependencies

src/
  data.py                  Load + clean catalogue, data-quality report
  enrich.py                Embeddings, HDBSCAN clustering, theme naming, auto-tags, near-dupes
  search.py                Keyword search + semantic (vector) search
  evaluate.py              skore evaluation (silhouette + theme-recovery + confusion matrix)

scripts/                   Reproducible evaluation + figure scripts (see above)

artifacts/                 All generated outputs (see below)
```

## Key deliverables (in `artifacts/`)

| File | What it is |
|---|---|
| `enriched_catalogue.json` | **Main deliverable** — catalogue with `theme` + `auto_tags`, ready to load back into CKAN |
| `EVALUATION.md` | Honest 8-step evaluation (what we can and cannot trust) |
| `themes.md` | The 76 discovered themes + top source org per theme |
| `SCRIPT.md` / `SCRIPT_READALOUD.md` | 5-minute pitch scripts |
| `figures/` | Slide-ready charts (search scores, theme quality, source bias, duplicate review, semantic map, …) |
| `evaluation_quant.json` / `evaluation_full.json` | Raw metrics behind every number above |

---

## Tech stack

Python 3.9 · scikit-learn (HDBSCAN, TF-IDF) · sentence-transformers (`all-MiniLM-L6-v2`) · umap-learn · skore · pandas · numpy · matplotlib

## Notes & honest limitations

- **Auto-tags are suggestions**, not official metadata — ~8–16% are numeric/noisy and need filtering.
- **Themes are exploratory, not ground truth** — the catalogue is source-skewed (Croydon 73%), so some clusters reflect the publisher more than the topic; we report per-theme sources rather than hiding this.
- **Duplicate recall is unknown** — we only measured precision on top pairs; the total number of true duplicates was never labelled.
- **The high theme-recovery score is a stability signal, not accuracy** — labels are recovered from the same embeddings that produced them (circular), so we treat it as reproducibility evidence only.
- `ckan_catalogue.json` (the input, ~98 MB) and the `*.npy` embedding caches are large; the embeddings regenerate from `run.py`.
