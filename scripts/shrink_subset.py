#!/usr/bin/env python
"""Shrink the already-built processed subset (fast; no raw Yelp rescan).

Trades dataset size for run time. Keeps the most active users, re-densifies, and
filters items + the social graph to match.

    python scripts/shrink_subset.py --max-users 2500
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recsys.data.subset import downsample_processed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-users", type=int, default=2500)
    p.add_argument(
        "--max-per-user",
        type=int,
        default=None,
        help="Cap each user to their N most recent interactions (biggest speed lever).",
    )
    p.add_argument("--min-user-reviews", type=int, default=10)
    p.add_argument("--min-item-reviews", type=int, default=10)
    args = p.parse_args()
    downsample_processed(
        max_users=args.max_users,
        max_per_user=args.max_per_user,
        min_user_reviews=args.min_user_reviews,
        min_item_reviews=args.min_item_reviews,
    )


if __name__ == "__main__":
    main()
