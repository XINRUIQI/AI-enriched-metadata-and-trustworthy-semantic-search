"""Load and clean the Data for London CKAN catalogue.

The raw catalogue is a JSON list of dataset records exported from CKAN.
Most records have a good free-text ``notes`` field but almost no structured
metadata: ~92% have no tags and ~100% have no themes/groups. This module
turns the raw records into a tidy DataFrame with a single clean text field
we can vectorise for clustering and semantic search.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Strip HTML, unescape entities and collapse whitespace."""
    if not text:
        return ""
    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def load_catalogue(path: str | Path) -> pd.DataFrame:
    """Load the raw CKAN JSON into a tidy DataFrame.

    Returns one row per dataset with the columns we care about plus a
    ``search_text`` field (title + description) used everywhere downstream.
    """
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    rows = []
    for rec in records:
        org = rec.get("organization") or {}
        title = clean_text(rec.get("title"))
        notes = clean_text(rec.get("notes"))
        rows.append(
            {
                "id": rec.get("id"),
                "name": rec.get("name"),
                "title": title,
                "notes": notes,
                "organization": org.get("title") or "(unknown)",
                "num_tags": rec.get("num_tags", 0),
                "num_resources": rec.get("num_resources", 0),
                "has_tags": bool(rec.get("tags")),
                "has_theme": bool(rec.get("groups")),
                # Title carries the most signal, so weight it by repeating it.
                "search_text": f"{title}. {title}. {notes}".strip(),
            }
        )

    df = pd.DataFrame(rows)
    # Drop records with no usable text at all (can't cluster or search these).
    df = df[df["search_text"].str.len() > 0].reset_index(drop=True)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Quantify the metadata gaps that motivate this project."""
    n = len(df)
    return {
        "n_datasets": n,
        "pct_no_tags": round(100 * (~df["has_tags"]).mean(), 1),
        "pct_no_theme": round(100 * (~df["has_theme"]).mean(), 1),
        "pct_short_desc": round(100 * (df["notes"].str.len() < 40).mean(), 1),
        "n_orgs": df["organization"].nunique(),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    df = load_catalogue(root / "ckan_catalogue.json")
    print(df.head())
    print(quality_report(df))
