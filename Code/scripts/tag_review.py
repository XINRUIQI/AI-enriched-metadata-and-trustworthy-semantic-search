"""Auto-tag manual review (responsible-AI sampling).

Auto-generated tags are the most error-prone artifact, so we sample datasets
and rate every suggested tag Good / Okay / Bad -- exactly the manual check a
judge expects for "can you vouch for it?".

Two modes:

  (default)  Draw a reproducible random sample and write:
             * artifacts/tag_review.csv          - one row per (dataset, tag),
                                                    with a blank ``human`` column
                                                    for a person to fill in.
             * artifacts/tag_review_sheet.md      - readable sheet (title + notes
                                                    + tags) for eyeballing.
             * artifacts/tag_review_summary.md    - Good/Okay/Bad table computed
                                                    from a transparent HEURISTIC
                                                    proxy (placeholder numbers).

  --summarize  Re-read tag_review.csv after a human has filled the ``human``
               column and regenerate tag_review_summary.md from the *real*
               human ratings (this is the number you quote on stage).

Rating rubric (same for human and heuristic):
    Good = accurate, specific, helps search
    Okay = relevant but too broad / generic
    Bad  = irrelevant, misleading, or useless (pure numbers, fragments)
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.data import load_catalogue
from src.enrich import auto_tags

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
CSV_PATH = ART / "tag_review.csv"
SHEET_PATH = ART / "tag_review_sheet.md"
SUMMARY_PATH = ART / "tag_review_summary.md"

SAMPLE_SIZE = 30          # datasets to review
SEED = 7                  # reproducible sample
RATINGS = ["Good", "Okay", "Bad"]

_WORD = re.compile(r"[a-z]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def heuristic_rating(tag: str, title: str, notes: str) -> str:
    """Transparent proxy for the human judgement (placeholder only).

    Good = grounded in the title, or a specific multi-word phrase in the notes.
    Okay = single generic word that appears in the text.
    Bad  = pure number / 1-2 char fragment, or not found in title+notes at all.
    """
    tag_norm = tag.strip().lower()
    # useless: no letters (pure numbers) or too short to be a search term
    if not _WORD.search(tag_norm) or len(tag_norm.replace(" ", "")) <= 2:
        return "Bad"

    tag_words = _words(tag_norm)
    title_words = _words(title)
    notes_words = _words(notes)
    multiword = len(tag_words) >= 2

    # grounded in the (high-signal) title -> specific & useful
    if tag_words and tag_words <= title_words:
        return "Good"
    # specific multi-word phrase present verbatim in the description
    if multiword and tag_norm in notes.lower():
        return "Good"
    # words appear in the text but only loosely / generically
    if tag_words & (title_words | notes_words):
        return "Okay"
    # nothing matched -> likely a TF-IDF artifact
    return "Bad"


def _notes_snippet(notes: str, n: int = 240) -> str:
    notes = notes.strip() or "(no description)"
    return notes[:n] + ("…" if len(notes) > n else "")


def generate() -> None:
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    texts = df["search_text"].tolist()
    tags = auto_tags(texts)

    rng = np.random.default_rng(SEED)
    # only sample datasets that actually received tags
    have = [i for i in range(len(df)) if tags[i]]
    sample = sorted(rng.choice(have, size=min(SAMPLE_SIZE, len(have)),
                               replace=False).tolist())

    rows: list[dict] = []
    for i in sample:
        title = df.iloc[i]["title"]
        notes = df.iloc[i]["notes"]
        for t in tags[i]:
            rows.append({
                "dataset_idx": i,
                "title": title,
                "notes_snippet": _notes_snippet(notes, 160),
                "tag": t,
                "heuristic": heuristic_rating(t, title, notes),
                "human": "",  # <- fill in Good / Okay / Bad
            })

    # ---- CSV (the thing a human fills in) ----
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- readable review sheet ----
    lines = [
        "# Auto-tag manual review sheet",
        "",
        f"Reproducible random sample of **{len(sample)} datasets** "
        f"(seed={SEED}), **{len(rows)} tags** total.",
        "",
        "Rate each tag: **Good** (accurate, specific, helps search) · "
        "**Okay** (relevant but too broad) · **Bad** (irrelevant / misleading).",
        "",
        "> The `heuristic` column is an automatic *proxy* to speed you up — "
        "**overwrite it with your own judgement** in `tag_review.csv`, then run "
        "`python scripts/tag_review.py --summarize`.",
        "",
    ]
    for i in sample:
        title = df.iloc[i]["title"]
        notes = df.iloc[i]["notes"]
        lines.append(f"### [{i}] {title}")
        lines.append("")
        lines.append(f"*Notes:* {_notes_snippet(notes)}")
        lines.append("")
        lines.append("| AI tag | heuristic | your rating |")
        lines.append("|---|---|---|")
        for t in tags[i]:
            lines.append(f"| `{t}` | {heuristic_rating(t, title, notes)} |  |")
        lines.append("")
    SHEET_PATH.write_text("\n".join(lines), encoding="utf-8")

    _write_summary([r["heuristic"] for r in rows], source="heuristic proxy",
                   n_datasets=len(sample))

    print(f"[tag-review] sampled {len(sample)} datasets, {len(rows)} tags")
    print(f"[tag-review] wrote {CSV_PATH.name}, {SHEET_PATH.name}, "
          f"{SUMMARY_PATH.name}")
    print("[tag-review] NEXT: fill the 'human' column in tag_review.csv, then "
          "run:  python scripts/tag_review.py --summarize")


def summarize_from_csv() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found — run without --summarize first.")
    with CSV_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    human = [r["human"].strip().capitalize() for r in rows if r["human"].strip()]
    if not human:
        raise SystemExit("No human ratings found in tag_review.csv 'human' column.")
    bad = [h for h in human if h not in RATINGS]
    if bad:
        raise SystemExit(f"Unexpected ratings {set(bad)}; use only {RATINGS}.")
    n_datasets = len({r["dataset_idx"] for r in rows if r["human"].strip()})
    _write_summary(human, source="human review", n_datasets=n_datasets)
    print(f"[tag-review] summarized {len(human)} human-rated tags -> "
          f"{SUMMARY_PATH.name}")


def _write_summary(ratings: list[str], source: str, n_datasets: int) -> None:
    counts = Counter(ratings)
    total = sum(counts.values())
    rows = []
    for r in RATINGS:
        c = counts.get(r, 0)
        rows.append((r, c, 100 * c / total if total else 0))

    good_pct = 100 * counts.get("Good", 0) / total if total else 0
    okay_pct = 100 * counts.get("Okay", 0) / total if total else 0
    bad_pct = 100 * counts.get("Bad", 0) / total if total else 0

    is_proxy = source == "heuristic proxy"
    header_note = (
        "> ⚠️ **Placeholder numbers from an automatic heuristic**, not a human. "
        "Fill `tag_review.csv` and re-run with `--summarize` for the real figure."
        if is_proxy else
        "> ✅ Numbers below come from **manual human review**."
    )

    lines = [
        "# Auto-tag quality — manual review result",
        "",
        header_note,
        "",
        f"Source: **{source}** · {n_datasets} datasets · {total} tags reviewed",
        "",
        "| Tag quality | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    for r, c, p in rows:
        lines.append(f"| {r} | {c} | {p:.0f}% |")
    lines += [
        "",
        "## One-liner for the pitch",
        "",
        f"> In our manual review, **{good_pct:.0f}%** of suggested tags were "
        f"directly useful, **{okay_pct:.0f}%** were broad but relevant, and "
        f"**{bad_pct:.0f}%** were misleading — so we ship auto-tags as "
        f"*suggestions*, not as replacements for official metadata.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summarize", action="store_true",
                    help="recompute summary from human ratings in tag_review.csv")
    args = ap.parse_args()
    if args.summarize:
        summarize_from_csv()
    else:
        generate()


if __name__ == "__main__":
    main()
