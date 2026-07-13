"""Enriched item text from aggregated Yelp reviews  [SCAFFOLD — Tier 1].

Cold-item recall stays near zero when item text is only ``name + categories``.
The highest-ROI *data* upgrade before adding more models: attach real review
snippets per business so content-based retrieval and the LLM reranker have
discriminative signal.

Suggested pipeline:
  1. Stream ``yelp_academic_dataset_review.json`` (same raw dir as subset.py).
  2. Group by ``business_id``; keep top-N reviews by ``useful`` or recency.
  3. Concatenate: ``name + categories + " Reviews: " + snippet1 + …``.
  4. Write back into ``items.parquet`` ``text`` column (or a new ``text_enriched``).
  5. Re-fit ``ContentBasedRecommender`` / unified ``MultiRetriever``; re-run
     ``scripts/benchmark.py`` and compare **cold_item_r@20** specifically.

Interview point: same architecture, better features — did cold-item move?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import RAW_DIR, settings

REVIEW_FILE = "yelp_academic_dataset_review.json"


def build_enriched_item_text(
    items: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
    max_reviews_per_item: int = 5,
    max_chars_per_review: int = 200,
    max_total_chars: int = 2000,
) -> pd.DataFrame:
    """Return a copy of *items* with an enriched ``text`` column.

    Parameters
    ----------
    items:
        Must include ``item_id`` (business_id), ``name``, ``categories``.
    max_reviews_per_item:
        How many review snippets to attach per business.
    max_chars_per_review:
        Truncate each review body (Yelp reviews can be long).
    max_total_chars:
        Cap total enriched text length (embedding model limits).

    Returns
    -------
    DataFrame with same rows as *items* and updated ``text`` field.
    """
    raise NotImplementedError(
        "Review-text enrichment is a Tier-1 exercise. Stream review JSON, "
        "aggregate top reviews per business_id, and concatenate with name + "
        "categories. See this module's docstring and docs/techniques.md "
        "(Senior MLE analysis: cold-item lift attribution)."
    )
