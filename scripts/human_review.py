"""Human relevance review -- the part skore CANNOT do for you.

skore audits the pipeline/metrics/methodology; only a person can vouch that an
auto-assigned theme/tag is actually RIGHT for a dataset. This script gives you a
ready-to-fill sample and then scores it into a headline hit-rate.

Usage
-----
  # 1) generate a 20-row template (diverse across themes)
  python scripts/human_review.py template

  # 2) a teammate opens artifacts/human_review_template.csv in Excel/Sheets and
  #    fills the last three columns:
  #       theme_ok    -> 1 if the assigned theme fits the dataset, else 0
  #       tags_ok     -> how many of the 5 auto-tags are relevant (0-5)
  #       notes       -> free text (optional)

  # 3) score the filled file
  python scripts/human_review.py score
  # or: python scripts/human_review.py score --file path/to/filled.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
TEMPLATE = ART / "human_review_template.csv"

HEADER = [
    "row", "id", "title", "organization", "assigned_theme", "auto_tags",
    "theme_ok(1/0)", "tags_ok(0-5)", "notes",
]
N_SAMPLE = 20
SEED = 7


def _load_enriched() -> list[dict]:
    data = json.loads((ART / "enriched_catalogue.json").read_text())
    return [r for r in data if r.get("theme") and "unclustered" not in r["theme"]]


def make_template() -> None:
    """Sample N_SAMPLE datasets spread across DISTINCT themes for coverage."""
    import random

    records = _load_enriched()
    by_theme: dict[str, list[dict]] = {}
    for r in records:
        by_theme.setdefault(r["theme"], []).append(r)

    rng = random.Random(SEED)
    themes = list(by_theme)
    rng.shuffle(themes)

    picked: list[dict] = []
    # round-robin one dataset per distinct theme until we hit N_SAMPLE
    ti = 0
    used_ids: set[str] = set()
    while len(picked) < min(N_SAMPLE, len(records)) and themes:
        theme = themes[ti % len(themes)]
        pool = [r for r in by_theme[theme] if r["id"] not in used_ids]
        if pool:
            rec = rng.choice(pool)
            picked.append(rec)
            used_ids.add(rec["id"])
        ti += 1
        if ti > len(themes) * 5:  # safety
            break

    with TEMPLATE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, rec in enumerate(picked, 1):
            title = (rec["title"] or "").replace("\n", " ").strip()
            tags = ", ".join(rec.get("auto_tags", []))
            w.writerow([i, rec["id"], title[:120], rec.get("organization", ""),
                        rec["theme"], tags, "", "", ""])

    print(f"[template] wrote {TEMPLATE}  ({len(picked)} rows across "
          f"{len({r['theme'] for r in picked})} themes)")
    print("Fill columns: theme_ok(1/0), tags_ok(0-5), notes -> then run:")
    print("  python scripts/human_review.py score")


def score(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[score] file not found: {path}\n"
                         f"Run 'python scripts/human_review.py template' first.")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    judged = [r for r in rows if (r.get("theme_ok(1/0)") or "").strip() != ""]
    if not judged:
        raise SystemExit("[score] no rows filled yet (theme_ok column is empty).")

    n = len(judged)
    theme_hits = sum(int(float(r["theme_ok(1/0)"])) for r in judged)
    tag_ok_counts = [float(r["tags_ok(0-5)"]) for r in judged
                     if (r.get("tags_ok(0-5)") or "").strip() != ""]

    theme_rate = 100 * theme_hits / n
    print("=" * 60)
    print(f"HUMAN RELEVANCE REVIEW  ({n} datasets judged)")
    print("=" * 60)
    print(f"  Theme relevance hit-rate : {theme_hits}/{n} = {theme_rate:.0f}%")
    if tag_ok_counts:
        mean_tag = sum(tag_ok_counts) / len(tag_ok_counts)
        tag_precision = 100 * mean_tag / 5.0
        print(f"  Auto-tag relevance       : {mean_tag:.1f}/5 relevant "
              f"per dataset  ({tag_precision:.0f}% tag precision)")
    print("=" * 60)
    print("Pitch line:")
    if tag_ok_counts:
        print(f'  "A human reviewer judged {theme_hits}/{n} auto-themes as a '
              f'good fit and ~{100*mean_tag/5:.0f}% of auto-tags relevant -- '
              f'skore vouches for the pipeline, humans vouch for the meaning."')
    else:
        print(f'  "A human reviewer judged {theme_hits}/{n} auto-themes '
              f'as a good fit for their dataset."')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template", help="generate the 20-row review CSV")
    sp = sub.add_parser("score", help="score a filled review CSV")
    sp.add_argument("--file", default=str(TEMPLATE), help="path to filled CSV")
    args = ap.parse_args()

    if args.cmd == "template":
        make_template()
    else:
        score(Path(args.file))


if __name__ == "__main__":
    main()
