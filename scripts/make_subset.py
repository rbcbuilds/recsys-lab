#!/usr/bin/env python
"""Build a small, dense Yelp subset from data/raw/ into data/processed/.

Example:
    python scripts/make_subset.py --city "Santa Barbara" \
        --min-user-reviews 10 --min-item-reviews 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recsys.data.subset import build_subset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default=None, help="Restrict to a single city (dense slice).")
    p.add_argument("--min-user-reviews", type=int, default=10)
    p.add_argument("--min-item-reviews", type=int, default=10)
    args = p.parse_args()

    build_subset(
        city=args.city,
        min_user_reviews=args.min_user_reviews,
        min_item_reviews=args.min_item_reviews,
    )


if __name__ == "__main__":
    main()
