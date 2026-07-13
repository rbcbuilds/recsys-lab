#!/usr/bin/env python
"""Attach aggregated Yelp review snippets to items.parquet text.

    python scripts/enrich_item_text.py
    python scripts/enrich_item_text.py --processed-dir data/processed_philly_xreg_fast
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recsys.config import PROCESSED_DIR, RAW_DIR
from src.recsys.data.review_text import enrich_processed_items


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--raw-dir", default=str(RAW_DIR))
    p.add_argument("--max-reviews-per-item", type=int, default=5)
    args = p.parse_args()
    enrich_processed_items(
        Path(args.processed_dir),
        raw_dir=Path(args.raw_dir),
        max_reviews_per_item=args.max_reviews_per_item,
    )


if __name__ == "__main__":
    main()
