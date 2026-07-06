# Auto-tags: what to trust, and what not to (responsible AI)

We generate tags automatically (TF-IDF top keywords per dataset) to fill the
**92.5%** of the catalogue that has none. They are genuinely useful for search,
but they are **suggestions, not ground truth**. Here is where they can go wrong
and how we handle it — this is the honesty a reviewer asks for.

## Known failure modes

- **Short descriptions get mislabelled.** A dataset whose `notes` are a few
  words (or just `"Total"`) gives TF-IDF almost nothing to work with, so the
  top keywords can latch onto noise. Datasets with rich text are far more
  reliable.
- **TF-IDF favours frequent surface words.** It rewards terms that are common
  in a document, not necessarily the *meaningful* ones — so tags skew toward
  high-frequency tokens (years, numbers, boilerplate) rather than concepts.
- **Fragments and numbers slip through.** Because tags are extracted tokens /
  n-grams, you occasionally get things like `"32"`, `"jan 2020"`, or half a
  phrase — relevant-ish but not a good search term.
- **Embeddings are semantically related but not interpretable.** For the
  *theme* layer, two datasets can sit near each other in vector space for
  reasons a human can't read off directly — great for recall, weaker for
  explainability.
- **Source skew can tilt tags.** Croydon supplies ~73% of the catalogue, so
  the vocabulary (and therefore the extracted tags/themes) leans toward the
  topics and phrasing that borough happens to publish.

## How we keep it responsible

- **Manual sampling review.** We rate a random sample of tags Good / Okay / Bad
  and report the share (`tag_review_sheet.md` + `tag_review_summary.md`), rather
  than claiming they're all correct.
- **Suggestions, not replacements.** Auto-tags are surfaced as recommended
  metadata for a human/data-owner to accept or edit — they never silently
  overwrite official metadata.
- **We flag, we don't force.** ~35% of datasets are deliberately left
  "unclustered / needs manual metadata" instead of being pushed into a theme
  that doesn't fit — that list is itself a useful deliverable.
- **We expose the bias.** Per-theme source breakdowns and the Croydon share are
  reported openly (`org_bias.png`, `theme_sizes.png`), not hidden.
