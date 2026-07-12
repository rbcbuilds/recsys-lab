#!/usr/bin/env python
"""Build the cross-regime Philadelphia slice from data/raw/.

Keeps a dense warm core and injects a real low-activity tail of items/users
first-seen after a global time cutoff, so that a single global-time split
produces cold items and cold users naturally (no simulation). Written to
data/processed_philly_xreg/ by default; activate by copying its parquet files
into data/processed/.

Example:
    python scripts/make_crossregime.py --city Philadelphia \
        --max-core-users 2000 --cutoff-quantile 0.9 \
        --max-cold-items 500 --max-cold-users 500
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recsys.data.subset import build_crossregime_subset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default="Philadelphia")
    p.add_argument("--min-user-reviews", type=int, default=10)
    p.add_argument("--min-item-reviews", type=int, default=10)
    p.add_argument("--max-core-users", type=int, default=2000)
    p.add_argument("--cutoff-quantile", type=float, default=0.9)
    p.add_argument("--max-cold-items", type=int, default=500)
    p.add_argument("--max-cold-users", type=int, default=500)
    p.add_argument("--tail-review-min", type=int, default=1)
    p.add_argument("--tail-review-max", type=int, default=8)
    p.add_argument("--out-dir", default=None, help="Output dir (default data/processed_philly_xreg).")
    args = p.parse_args()

    build_crossregime_subset(
        city=args.city,
        min_user_reviews=args.min_user_reviews,
        min_item_reviews=args.min_item_reviews,
        max_core_users=args.max_core_users,
        cutoff_quantile=args.cutoff_quantile,
        max_cold_items=args.max_cold_items,
        max_cold_users=args.max_cold_users,
        tail_review_min=args.tail_review_min,
        tail_review_max=args.tail_review_max,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
