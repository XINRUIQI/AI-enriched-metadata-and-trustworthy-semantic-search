"""Sample near-duplicate pairs that actually need a human decision.

The very top of the cosine-similarity ranking is dominated by pairs whose
titles are *byte-for-byte identical* -- those are trivial and don't test the
model's judgement. To build a realistic "human review queue" we do two things:

1. Drop pairs whose normalised titles are identical (a string match, not a model
   win).
2. STRATIFY across similarity bands (1.00, 0.98, 0.96, ... down to 0.86) and
   sample a handful of *distinct* pairs from each band. This shows exactly where
   the model's precision starts to break as the threshold loosens.

Output: a reviewable JSON (title + notes + org + similarity) so a human can
label each pair as: duplicate / near_duplicate / related / wrong
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_catalogue  # noqa: E402

ART = ROOT / "artifacts"
COLLECT_THRESHOLD = 0.86
# (low, high, how many to sample) -- roughly 50 total across the gradient.
BANDS = [
    (0.995, 1.0001, 8),
    (0.98, 0.995, 8),
    (0.96, 0.98, 8),
    (0.94, 0.96, 8),
    (0.92, 0.94, 6),
    (0.90, 0.92, 6),
    (0.86, 0.90, 6),
]
SEED = 0


def norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def main() -> None:
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    doc_emb = np.load(ART / "doc_emb.npy").astype("float32")
    assert doc_emb.shape[0] == len(df), "embeddings/rows misaligned"

    norms = np.linalg.norm(doc_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = doc_emb / norms

    titles = df["title"].fillna("").tolist()
    ntitles = [norm_title(t) for t in titles]

    n = X.shape[0]
    block = 1024
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = X[start:end] @ X.T
        for li, gi in enumerate(range(start, end)):
            row = sims[li]
            cand = np.where(row >= COLLECT_THRESHOLD)[0]
            for j in cand:
                if j > gi and ntitles[gi] != ntitles[j]:  # non-identical only
                    pairs.append((gi, int(j), float(row[j])))

    rng = np.random.default_rng(SEED)
    selected: list[tuple[int, int, float]] = []
    seen_titlepairs: set[tuple[str, str]] = set()
    for lo, hi, want in BANDS:
        band = [p for p in pairs if lo <= p[2] < hi]
        rng.shuffle(band)
        picked = 0
        for i, j, s in band:
            key = tuple(sorted((ntitles[i], ntitles[j])))
            if key in seen_titlepairs:
                continue
            seen_titlepairs.add(key)
            selected.append((i, j, s))
            picked += 1
            if picked >= want:
                break

    selected.sort(key=lambda t: t[2], reverse=True)

    def trunc(s: str, m: int = 300) -> str:
        s = str(s or "").strip()
        return s[:m] + ("..." if len(s) > m else "")

    records = []
    for rank, (i, j, s) in enumerate(selected, 1):
        records.append({
            "rank": rank,
            "similarity": round(s, 4),
            "title_a": titles[i],
            "title_b": titles[j],
            "org_a": df.iloc[i]["organization"],
            "org_b": df.iloc[j]["organization"],
            "notes_a": trunc(df.iloc[i]["notes"]),
            "notes_b": trunc(df.iloc[j]["notes"]),
            "label": "",  # to be filled by human review
        })

    out = ART / "dup_review_sample.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"collected {len(pairs)} non-identical pairs >= {COLLECT_THRESHOLD}")
    print(f"wrote {len(records)} stratified pairs to {out}")


if __name__ == "__main__":
    main()
