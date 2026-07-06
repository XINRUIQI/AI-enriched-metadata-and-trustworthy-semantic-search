# Data Quality Check

> Step 1: confirm the input is trustworthy before doing anything else. If the metadata itself is poor, the downstream NLP results can't be fully trusted either.
> Source: `p1_quality` in `artifacts/evaluation_full.json` (computed by `evaluate_report.py` over the full `ckan_catalogue.json`, 20,685 datasets).

| Check | Result | Meaning |
| --- | --- | --- |
| Missing tags | 92.5% (19,128 / 20,685 have no tags) | Tags enrichment is necessary |
| Missing themes | 100% (none have groups/theme) | Theme generation is necessary |
| Short / missing notes | 1.8% (notes shorter than 40 chars) | Text field is usable for NLP |
| Duplicate titles | 378 records (178 groups of identical titles) | Duplicate detection is useful |
| Source imbalance | Croydon dominates at 72.9% (144 source orgs total) | Results may be biased by source distribution |

## Notes on definitions

- **Missing tags / themes**: based on whether the CKAN `tags` / `groups` fields are empty.
- **Short / missing notes**: not a strict "missing" count, but the share of records whose cleaned `notes` (description) is shorter than 40 characters — a proxy for how much text NLP has to work with. Only 1.8%, so the vast majority have enough text.
- **Duplicate titles**: titles stripped and lowercased, then grouped by exact match — 178 groups covering 378 records (the earlier PLAN figure of 372 was an early estimate, now unified to 378).
- **Source imbalance**: the most frequent `organization` is Croydon at 72.9%. This must be checked during theme evaluation to see whether any cluster groups by source org rather than by topic.
