# 5-Minute Pitch Script · Discovering London's Data

> English spoken script · ~700 words · ≈ 4:55 at normal pace (leaves buffer, will not overrun).
> Numbers verified against `evaluation_quant.json`, `EVALUATION.md`, `tag_review_summary.md`, `search_eval_results.md`.
> Figures = the latest slide-ready set (generated 11:44–11:53).

---

## Slide 1 — Data quality first ｜ ~0:25
**Figure:** `figures/quality_bars.png` (backup: `figures/org_bias.png`)

"Before trusting any output, we check the input. London's library has **twenty thousand six hundred datasets** — but **one hundred percent have no theme, ninety-two percent have no tags**, and one publisher, Croydon, is seventy-three percent of everything. The structured metadata is almost empty — but the descriptions are rich, so NLP has real material to work with."

---

## Slide 2 — Solution ｜ ~0:25
**Figure:** `figures/semantic_map.png` (backup: `figures/coverage_before_after.png`)

"One set of MiniLM embeddings, four jobs: cluster the whole catalogue into **seventy-six themes** with HDBSCAN, auto-tag, flag near-duplicates, and power semantic search. This map is all twenty thousand datasets, organised by meaning — themes emerge on their own."

---

## Slide 3 — Baselines + Demo ｜ ~0:50
**Figure:** `baseline_comparison.png`

"Any improvement needs a control, so we built three: keyword — today's library; TF-IDF — a fair lexical baseline; and our semantic search. Same query, same corpus, same top-five.

Query: 'young people not in work.' Keyword drifts into commute tables. TF-IDF scores **zero out of five** — 'NEET' shares no words with the query. Semantic gets **five out of five** — it recovered the concept behind the acronym. That one case proves both: semantic adds real value, and the lexical fallback would throw it away."

---

## Slide 4 — Evaluate search (the headline) ｜ ~0:35
**Figure:** `figures/search_eval_dashboard.png`

"We hand-judged the results and ran the standard IR metrics. Precision-at-five: keyword **zero-four-two**, TF-IDF **zero-seven-five**, ours **zero-nine-seven**. It wins on Precision-at-ten, MRR and nDCG too — and it holds **query by query**, not just on average."

---

## Slide 5 — Evaluate auto-tags ｜ ~0:30
**Figure:** (spoken; optional `tag_review.csv` screenshot)

"For tags, we did a **manual human review** — thirty datasets, a hundred and fifty tags. **Seventy-eight percent** were directly useful, eleven percent broad but relevant, and eleven percent misleading — usually bare numbers. So we ship auto-tags as **suggestions**, not as replacements for official metadata."

---

## Slide 6 — Evaluate themes: interpretability, not just a score ｜ ~0:35
**Figure:** `figures/interpretability_cards.png` (backup: `figures/org_bias.png`)

"For themes we don't just quote a clustering score — we read them. Diabetes, road accidents, homelessness, sexual health: each theme holds datasets that genuinely belong together. But we're honest about the catch — the biggest clusters are dominated by a single publisher. Some 'themes' are really just Croydon, and we report that per theme instead of hiding it."

---

## Slide 7 — Evaluate duplicates: check the top pairs ｜ ~0:30
**Figure:** `figures/dup_review.png`

"Near-duplicates: at ninety-nine percent similarity we found forty-four thousand pairs. We hand-reviewed fifty in the harder, non-identical band: **zero were off-topic**, but only about **twelve percent were true duplicates** — the rest were merely related. So this is a **human-review queue, not auto-delete** — and recall is unknown."

---

## Slide 8 — Methodology check with skore ｜ ~0:30
**Figure:** `figures/theme_f1_slide.png` (backup: `figures/silhouette_kmeans_vs_hdbscan.png`)

"On method: we didn't guess the cluster count — a silhouette sweep had no elbow, so HDBSCAN chose it and separated themes **three-and-a-half times** better. A held-out classifier recovers the themes at **macro-F1 zero-nine-nine-nine** — but we call that **stability, not accuracy**, because the labels came from the same vectors; it's circular. We ran it through **skore**, which flagged a real warning — some themes have too few test examples. Deliberately, we don't use skore to bless that circular classifier: skore validates the pipeline, not whether the themes are true."

---

## Slide 9 — What we trust, and what we don't ｜ ~0:35
**Figure:** `figures/coverage_before_after.png`

"So — can we vouch for it? For what matters, yes. Coverage went from zero themes and eight percent tags to **sixty-five percent themed and one hundred percent tagged** — and semantic search is a real, measured gain: Precision-at-five **one-point-oh versus zero-four-seven**. What we're honest about: tags need filtering, themes are skewed by Croydon, duplicate recall is unknown, and that ninety-nine-point-nine is stability, not truth. We left thirty-five percent unclustered on purpose. We know exactly where to trust this — and where not to. Thank you."

---

## Timing & delivery notes
- **Total ≈ 4:55.** Slow down and pause on Slide 3 (the demo) — it's the emotional peak.
- If running long, cut the "usually bare numbers" clause in Slide 5 (~5s).
- Read all decimals as spoken digits (e.g. "zero-nine-seven"), already written that way.
- Every claim on screen has a figure behind it — point at the figure as you say the number.

## Final figure checklist (in slide order)
1. `figures/quality_bars.png`
2. `figures/semantic_map.png`
3. `baseline_comparison.png`
4. `figures/search_eval_dashboard.png`
5. — (spoken / `tag_review.csv`)
6. `figures/interpretability_cards.png`
7. `figures/dup_review.png`
8. `figures/theme_f1_slide.png`
9. `figures/coverage_before_after.png`
