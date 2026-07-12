"""Carve a small, dense slice out of the full Yelp Open Dataset.

The full dataset is several GB of line-delimited JSON. This reads it in a
streaming fashion, keeps one city and only sufficiently active users/items,
then writes tidy parquet with the SAME schema the loaders expect. That gives
MovieLens-like iteration speed while preserving real social edges + text.

Usage:
    python scripts/make_subset.py --city "Santa Barbara" \
        --min-user-reviews 10 --min-item-reviews 10
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RAW_DIR, settings

REVIEW_FILE = "yelp_academic_dataset_review.json"
BUSINESS_FILE = "yelp_academic_dataset_business.json"
USER_FILE = "yelp_academic_dataset_user.json"


def _iter_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_subset(
    city: str | None = None,
    min_user_reviews: int = 10,
    min_item_reviews: int = 10,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    raw_dir = Path(raw_dir)
    business_path = raw_dir / BUSINESS_FILE
    review_path = raw_dir / REVIEW_FILE
    user_path = raw_dir / USER_FILE

    for p in (business_path, review_path, user_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Download the Yelp Open Dataset into {raw_dir} "
                f"(see data/README.md)."
            )

    # 1. Businesses (optionally filter by city) -> item universe
    biz_rows = []
    for b in _iter_json(business_path):
        if city and (b.get("city") or "").strip().lower() != city.strip().lower():
            continue
        biz_rows.append(
            {
                settings.item_col: b["business_id"],
                "name": b.get("name", ""),
                "categories": b.get("categories") or "",
                "city": b.get("city", ""),
                "text": b.get("name", "") + ". " + (b.get("categories") or ""),
            }
        )
    items = pd.DataFrame(biz_rows)
    if items.empty:
        raise ValueError(f"No businesses found for city={city!r}.")
    keep_items = set(items[settings.item_col])

    # 2. Reviews limited to those businesses -> interactions
    rev_rows = []
    for r in _iter_json(review_path):
        if r["business_id"] not in keep_items:
            continue
        rev_rows.append(
            {
                settings.user_col: r["user_id"],
                settings.item_col: r["business_id"],
                settings.rating_col: float(r.get("stars", 0)),
                settings.time_col: r.get("date"),
            }
        )
    interactions = pd.DataFrame(rev_rows)
    interactions[settings.time_col] = pd.to_datetime(interactions[settings.time_col])

    # 3. Iteratively prune to dense core (users & items with enough activity)
    interactions = _densify(interactions, min_user_reviews, min_item_reviews)
    keep_users = set(interactions[settings.user_col])
    keep_items = set(interactions[settings.item_col])
    items = items[items[settings.item_col].isin(keep_items)].reset_index(drop=True)

    # 4. Users + social graph (only edges within the kept user set)
    user_rows = []
    social_rows = []
    for u in _iter_json(user_path):
        uid = u["user_id"]
        if uid not in keep_users:
            continue
        user_rows.append({settings.user_col: uid, "review_count": u.get("review_count", 0)})
        friends = (u.get("friends") or "").split(", ")
        for fid in friends:
            fid = fid.strip()
            if fid and fid in keep_users and fid != uid:
                a, b = sorted((uid, fid))
                social_rows.append((a, b))
    users = pd.DataFrame(user_rows)
    social = pd.DataFrame(
        sorted(set(social_rows)), columns=[settings.user_col, "friend_id"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    social.to_parquet(out_dir / "social.parquet", index=False)

    density = len(interactions) / (len(users) * len(items) or 1)
    print(
        f"Wrote Yelp subset to {out_dir}:\n"
        f"  users={len(users):,} items={len(items):,} "
        f"interactions={len(interactions):,} density={density:.4%} "
        f"social_edges={len(social):,}"
    )
    print("Note: photos/image vectors are not built here — see models/multimodal.py.")


def downsample_processed(
    max_users: int = 2500,
    max_per_user: int | None = None,
    min_user_reviews: int = 10,
    min_item_reviews: int = 10,
    processed_dir: Path = PROCESSED_DIR,
    seed: int | None = None,
) -> None:
    """Shrink an already-built processed subset in place (fast, no raw rescan).

    Keeps the ``max_users`` most active users, optionally caps each user to their
    ``max_per_user`` most recent interactions (the biggest run-time lever, since
    a few "power users" carry most of the interactions), re-densifies, and
    filters items, users, and the social graph to match. Use this to trade
    dataset size for run time without re-parsing the multi-GB raw Yelp files.
    """
    from ..config import settings

    processed_dir = Path(processed_dir)
    inter = pd.read_parquet(processed_dir / "interactions.parquet")
    u, i = settings.user_col, settings.item_col

    # Keep the most active users (stable, deterministic).
    top_users = inter[u].value_counts().head(max_users).index
    inter = inter[inter[u].isin(top_users)]

    # Cap per-user history to the most recent interactions. This is what
    # actually shrinks the interaction count when a few users review hundreds
    # of items, and it keeps the temporal-split evaluation meaningful.
    if max_per_user is not None and settings.time_col in inter.columns:
        inter = (
            inter.sort_values(settings.time_col)
            .groupby(u, group_keys=False)
            .tail(max_per_user)
        )

    # Re-densify so items/users still meet the interaction thresholds.
    inter = _densify(inter, min_user_reviews, min_item_reviews)
    keep_users = set(inter[u])
    keep_items = set(inter[i])

    items = pd.read_parquet(processed_dir / "items.parquet")
    items = items[items[i].isin(keep_items)].reset_index(drop=True)
    users = pd.read_parquet(processed_dir / "users.parquet")
    users = users[users[u].isin(keep_users)].reset_index(drop=True)

    social_path = processed_dir / "social.parquet"
    if social_path.exists():
        social = pd.read_parquet(social_path)
        social = social[
            social[u].isin(keep_users) & social["friend_id"].isin(keep_users)
        ].reset_index(drop=True)
        social.to_parquet(social_path, index=False)
        n_social = len(social)
    else:
        n_social = 0

    inter.to_parquet(processed_dir / "interactions.parquet", index=False)
    items.to_parquet(processed_dir / "items.parquet", index=False)
    users.to_parquet(processed_dir / "users.parquet", index=False)

    density = len(inter) / (len(users) * len(items) or 1)
    print(
        f"Downsampled processed subset in {processed_dir}:\n"
        f"  users={len(users):,} items={len(items):,} "
        f"interactions={len(inter):,} density={density:.4%} "
        f"social_edges={n_social:,}"
    )


def _densify(interactions: pd.DataFrame, min_u: int, min_i: int) -> pd.DataFrame:
    """Iteratively drop users/items below the interaction thresholds."""
    prev = -1
    while len(interactions) != prev:
        prev = len(interactions)
        uc = interactions[settings.user_col].value_counts()
        interactions = interactions[
            interactions[settings.user_col].isin(uc[uc >= min_u].index)
        ]
        ic = interactions[settings.item_col].value_counts()
        interactions = interactions[
            interactions[settings.item_col].isin(ic[ic >= min_i].index)
        ]
    return interactions.reset_index(drop=True)
