# Slide Deck Outline · Discovering London's Data

> Paste one block per slide into PowerPoint / Google Slides / Keynote.
> Each slide = Title + on-slide bullets + the figure to drop in + the one line to say.
> 10 slides · ≈ 5 minutes. Figures live in `artifacts/` and `artifacts/figures/`.

---

## Slide 0 — Cover
**Title:** Discovering London's Data
**Subtitle:** Team By2 · London Data Week 2026
**On slide:**
- We built a tool to enrich & search London's data catalogue
- …then tested it hard, to see if we can trust it
**Image:** (team name / event logo, or `figures/semantic_map.png` faded as background)
**Say:** "We built a tool — then tested it hard, to see if we could trust it."

---

## Slide 1 — The problem: is the input trustworthy?
**Title:** 20,685 datasets — but you can't search them
**On slide:**
- 92% have no tags
- 100% have no theme
- 1 publisher (Croydon) = 73% of everything
- BUT the descriptions are rich → NLP has material
**Image:** `figures/quality_bars.png`
**Say:** "The labels are almost empty — but the text inside is good, and that's what we work with."

---

## Slide 2 — Our solution: one vector, four jobs
**Title:** One embedding, four jobs
**On slide:**
- Encode every dataset once (MiniLM)
- → Themes · Tags · Duplicates · Semantic search
- 76 themes emerge on their own
**Image:** `figures/semantic_map.png`
**Say:** "This map is all 20,685 datasets, organised by meaning — themes form on their own."

---

## Slide 3 — Live demo: search that understands
**Title:** "young people not in work"
**On slide:**
- Keyword search: fails (off-topic tables)
- Ours: finds NEET datasets — 5/5
- Understands the idea, not just the words
**Image:** `baseline_comparison.png`
**Say:** "It even found 'NEET', a word we never typed."

---

## Slide 4 — Did we prove search is better?
**Title:** Semantic search wins on every metric
**On slide:**
- 8 real questions, every result judged by hand
- Precision@5: keyword 0.42 → **ours 0.97**
- Wins query-by-query, not just on average
**Image:** `figures/search_eval_dashboard.png`
**Say:** "We didn't trust our own eyes — we scored it properly."

---

## Slide 5 — Are the auto-tags any good?
**Title:** Tags: checked 150 by hand
**On slide:**
- 78% genuinely useful
- 11% too broad · 11% wrong (usually a stray number)
- → shipped as suggestions, not final answers
**Image:** (text slide, or screenshot of `tag_review.csv`)
**Say:** "We treat these tags as suggestions, not as final answers."

---

## Slide 6 — Do the themes make sense to a human?
**Title:** Themes: interpretable, but honest about bias
**On slide:**
- Diabetes, housing, road accidents — data that belongs together
- Catch: some big "themes" are just one place (Croydon)
- We report it, not hide it
**Image:** `figures/interpretability_cards.png` (backup: `figures/org_bias.png`)
**Say:** "Some big themes aren't topics at all — they're just Croydon. We say so."

---

## Slide 7 — Duplicate detection
**Title:** Duplicates: flag for review, don't delete
**On slide:**
- 44,000 near-duplicate pairs at highest similarity
- Hand-checked 50 hard cases → only ~12% true duplicates
- → human-review queue; recall unknown
**Image:** `figures/dup_review.png`
**Say:** "This flags things for a person to check — it doesn't delete anything on its own."

---

## Slide 8 — How we chose the method (skore check)
**Title:** We didn't guess — we checked
**On slide:**
- Tried many cluster sizes → none worked well
- Switched to HDBSCAN → ~3.4× better separation
- Ran skore → surfaced a real pipeline warning (we report it)
**Image:** `figures/theme_f1_slide.png` (backup: `figures/silhouette_kmeans_vs_hdbscan.png`)
**Say:** "That theme score proves the themes are consistent — not that they're the 'right' ones. That's still a human call."

---

## Slide 9 — So, can you trust it?
**Title:** What we trust — and what we don't
**On slide:**
- Coverage: 0% → 65% themed · 8% → 100% tagged
- Search: a real, measured win (P@5 1.0 vs 0.47)
- Tags & duplicates: not fully — still need a human
- 35% left unclustered on purpose
**Image:** `figures/coverage_before_after.png`
**Say:** "We know exactly where this tool is strong — and where it still needs help. Thank you."

---

## Figure checklist (drop-in order)
0. (cover) — optional `figures/semantic_map.png`
1. `figures/quality_bars.png`
2. `figures/semantic_map.png`
3. `baseline_comparison.png`
4. `figures/search_eval_dashboard.png`
5. — (text / `tag_review.csv`)
6. `figures/interpretability_cards.png`
7. `figures/dup_review.png`
8. `figures/theme_f1_slide.png`
9. `figures/coverage_before_after.png`

## Design tips
- One figure per slide, large. Keep bullets to ≤4, ≤6 words each.
- Consistent colour: keyword = grey, TF-IDF = orange, semantic/ours = green (matches the figures).
- Slides 5–9 are the "vouch for it" half — the judges reward this; don't rush them.
