"""Enriched item text from aggregated Yelp reviews."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..config import RAW_DIR, settings

REVIEW_FILE = "yelp_academic_dataset_review.json"


def _iter_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _base_text(row: pd.Series) -> str:
    parts = []
    for col in ("name", "categories"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]).strip())
    if parts:
        return ". ".join(parts)
    return str(row.get(settings.item_col, ""))


def build_enriched_item_text(
    items: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
    max_reviews_per_item: int = 5,
    max_chars_per_review: int = 200,
    max_total_chars: int = 2000,
) -> pd.DataFrame:
    """Return a copy of *items* with an enriched ``text`` column.

    Streams ``yelp_academic_dataset_review.json``, keeps the top reviews per
    business (by ``useful``, then ``stars``), and appends snippets after the
    existing name + categories string.
    """
    raw_dir = Path(raw_dir)
    review_path = raw_dir / REVIEW_FILE
    if not review_path.exists():
        raise FileNotFoundError(
            f"Missing {review_path}. Download the Yelp Open Dataset into {raw_dir}."
        )

    i = settings.item_col
    out = items.copy()
    keep_items = set(out[i].astype(str))
    base_text = {str(r[i]): _base_text(r) for _, r in out.iterrows()}

    snippets: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for rev in _iter_json(review_path):
        bid = str(rev.get("business_id", ""))
        if bid not in keep_items:
            continue
        text = (rev.get("text") or "").strip().replace("\n", " ")
        if not text:
            continue
        useful = int(rev.get("useful") or 0)
        stars = float(rev.get("stars") or 0)
        snippets[bid].append((useful, stars, text))

    enriched: dict[str, str] = {}
    for bid, base in base_text.items():
        rows = snippets.get(bid, [])
        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        parts = [base] if base else []
        for useful, _stars, text in rows[:max_reviews_per_item]:
            chunk = text[:max_chars_per_review].strip()
            if chunk:
                parts.append(chunk)
        full = ". ".join(parts)
        if len(full) > max_total_chars:
            full = full[: max_total_chars - 3].rsplit(". ", 1)[0] + "..."
        enriched[bid] = full if full else base

    out["text"] = out[i].astype(str).map(lambda x: enriched.get(x, base_text.get(x, x)))
    n_with_reviews = sum(1 for bid in keep_items if snippets.get(bid))
    print(
        f"Enriched item text: {n_with_reviews:,}/{len(keep_items):,} items "
        f"have review snippets (max {max_reviews_per_item}/item)."
    )
    return out


def enrich_processed_items(
    processed_dir: Path,
    raw_dir: Path = RAW_DIR,
    **kwargs,
) -> None:
    """Update ``items.parquet`` in a processed directory with enriched text."""
    processed_dir = Path(processed_dir)
    items_path = processed_dir / "items.parquet"
    if not items_path.exists():
        raise FileNotFoundError(f"No items.parquet at {processed_dir}")
    items = pd.read_parquet(items_path)
    items = build_enriched_item_text(items, raw_dir=raw_dir, **kwargs)
    items.to_parquet(items_path, index=False)
    print(f"Wrote enriched items to {items_path}")
