# Read-Aloud Script (stress + pauses) · Discovering London's Data

> Delivery version of `SCRIPT.md`. ≈ 5:05.
> **Bold** = stress this word. `/` = short pause (one beat). `//` = longer pause (breathe / change slide).
> Read decimals as digits: "zero-nine-seven".

---

## Slide 1 — Data quality first ｜ ~0:25
`figures/quality_bars.png`

Before we trust any output / we check the **input**. //
London's library has **twenty thousand** datasets / — but **one hundred percent** have **no theme** / **ninety-two percent** have **no tags** / and one publisher / Croydon / is **seventy-three percent** of everything. //
The metadata is almost **empty** / — but the descriptions are **rich** / so NLP has real material to work with. //

---

## Slide 2 — Solution ｜ ~0:25
`figures/semantic_map.png`

**One** set of embeddings / **four** jobs: //
cluster the whole catalogue into **seventy-six themes** / auto-tag / flag duplicates / and power **semantic search**. //
This map is **all** twenty thousand datasets / organised by **meaning** / — the themes emerge on their **own**. //

---

## Slide 3 — Baselines + Demo ｜ ~0:50
`baseline_comparison.png`

Any improvement needs a **control** / so we built **three**: //
**keyword** / — today's library. / **TF-IDF** / — a fair lexical baseline. / And our **semantic** search. //
Same query / same corpus / same top-five. //
Query: / "young people **not in work**." //
Keyword drifts into **commute** tables. / TF-IDF scores **zero** out of five / — "NEET" shares **no words** with the query. //
Semantic gets **five** out of five / — it recovered the **concept** behind the acronym. //
That one case proves **both**: / semantic adds **real** value / and the lexical fallback would **throw it away**. //

---

## Slide 4 — Evaluate search ｜ ~0:35
`figures/search_eval_dashboard.png`

We **hand-judged** the results / and ran the standard IR metrics. //
Precision-at-five: / keyword **zero-four-two** / TF-IDF **zero-seven-five** / ours **zero-nine-seven**. //
It wins on Precision-at-ten / MRR / and nDCG **too** / — and it holds **query by query** / not just on average. //

---

## Slide 5 — Evaluate auto-tags ｜ ~0:30
(spoken / optional `tag_review.csv`)

For tags / we did a **manual** human review / — thirty datasets / a hundred and fifty tags. //
**Seventy-eight percent** were directly useful / eleven percent broad but relevant / and eleven percent **misleading** / — usually bare numbers. //
So we ship auto-tags as **suggestions** / **not** as replacements for official metadata. //

---

## Slide 6 — Evaluate themes: interpretability + stability ｜ ~0:45
`figures/interpretability_cards.png` // then `figures/hdbscan_stability.png`

Themes have **no** gold labels / — so this is **clustering** / not classification. / We can't just quote **accuracy**. //
So we check them the way a **human** would / — three ways. //
**One** / interpretability: / each theme's top words name a **real** topic / — diabetes / road accidents / homelessness / sexual health. / **Seventy-one** of seventy-six rate **Clear**. //
**Two** / sample consistency: / pull **five** datasets from a theme at **random** / — "road accidents" gives pedestrian / motorcyclist / traffic casualties. / They genuinely **belong together**. //
**Three** / stability: / change the clustering parameter and re-run / — the themes barely move. / Agreement stays around **zero-eight-four** on average. / It's **not** luck. //
And we stay **honest**: / the biggest clusters lean on a **single** publisher / — some "themes" are really just **Croydon** / — we **report** it / not hide it. //

---

## Slide 7 — Evaluate duplicates ｜ ~0:30
`figures/dup_review.png`

Near-duplicates: / at ninety-nine percent similarity / we found **forty-four thousand** pairs. //
We hand-reviewed **fifty** in the harder / non-identical band: / **zero** were off-topic / but only about **twelve percent** were **true** duplicates / — the rest were merely **related**. //
So this is a **human-review queue** / **not** auto-delete / — and recall is **unknown**. //

---

## Slide 8 — Methodology check with skore ｜ ~0:30
`figures/theme_f1_slide.png`

On method: / we didn't **guess** the cluster count / — a silhouette sweep had **no elbow** / so HDBSCAN chose it / and separated themes **three-and-a-half times** better. //
A held-out classifier recovers the themes at macro-F1 **zero-nine-nine-nine** / — but we call that **stability** / not **accuracy** / because the labels came from the **same** vectors. / It's **circular**. //
We ran it through **skore** / which flagged a **real** warning / — some themes have too few test examples. //
Deliberately / we **don't** use skore to bless that circular classifier: / skore validates the **pipeline** / not whether the themes are **true**. //

---

## Slide 9 — What we trust, and what we don't ｜ ~0:35
`figures/coverage_before_after.png`

So / — can we **vouch** for it? / For what matters / **yes**. //
Coverage went from **zero** themes and eight percent tags / to **sixty-five percent** themed / and **one hundred percent** tagged. / And semantic search is a **real** / measured gain: / Precision-at-five **one-point-oh** / versus **zero-four-seven**. //
What we're **honest** about: / tags need filtering / themes are skewed by Croydon / duplicate recall is unknown / and that ninety-nine-point-nine is **stability** / not **truth**. //
We left thirty-five percent unclustered **on purpose**. //
We know **exactly** where to trust this / — and where **not** to. //
**Thank you.** //

---

## Delivery cheatsheet
- Peak = **Slide 3**. Land the "**five out of five**" and pause hard (`//`).
- Breathe on every `//` — that's also your slide-change beat.
- The judges reward honesty: don't rush Slides 8–9 (the "what we can't trust" half).
- Slide 6 shows **two** figures: cards first (points 1–2), then `hdbscan_stability.png` on "**Three / stability**".
- Running long? Drop "usually bare numbers" (Slide 5), or the Slide-6 **stability** beat (points 1–2 already carry it) — each saves ~5–8s.
