#!/usr/bin/env python
"""Shrink the processed subset for faster iteration (no raw Yelp rescan).

Dense subset (drops cold tail if re-densified):

    python scripts/shrink_subset.py --max-users 2000 --max-per-user 40

Cross-regime (keeps cold users/items; caps warm pre-cutoff history only):

    python scripts/shrink_subset.py --cross-regime
    cp data/processed_philly_xreg_fast/*.parquet data/processed/

Fast iteration slice (cap=10, top-1000 warm by social+sequence, cold users pinned):

    python scripts/shrink_subset.py --cross-regime --max-per-user 10 --max-warm-users 1000 --user-select social_seq
    cp data/processed_philly_xreg_fast/*.parquet data/processed/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recsys.config import PROCESSED_DIR, RAW_DIR
from src.recsys.data.subset import downsample_crossregime_processed, downsample_processed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cross-regime",
        action="store_true",
        help="Cap warm pre-cutoff history only; preserve cold tail (cross-regime eval).",
    )
    p.add_argument(
        "--source-dir",
        default=None,
        help="Input parquet dir (default: data/processed_philly_xreg if --cross-regime, else data/processed).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output dir (default: data/processed_philly_xreg_fast for cross-regime).",
    )
    p.add_argument("--max-users", type=int, default=2500, help="Dense mode only.")
    p.add_argument(
        "--max-warm-users",
        type=int,
        default=None,
        help="Cross-regime: keep this many warm users (+ all cold users).",
    )
    p.add_argument(
        "--user-select",
        choices=["activity", "social_seq"],
        default="social_seq",
        help="How to rank warm users when --max-warm-users is set.",
    )
    p.add_argument(
        "--max-per-user",
        type=int,
        default=None,
        help="Cap pre-cutoff warm history to N most recent interactions.",
    )
    p.add_argument("--cutoff-quantile", type=float, default=0.9)
    p.add_argument("--min-user-reviews", type=int, default=10)
    p.add_argument("--min-item-reviews", type=int, default=10)
    args = p.parse_args()

    if args.cross_regime:
        source = Path(args.source_dir or RAW_DIR.parent / "processed_philly_xreg")
        out = Path(args.out_dir or RAW_DIR.parent / "processed_philly_xreg_fast")
        downsample_crossregime_processed(
            max_per_user=args.max_per_user if args.max_per_user is not None else 10,
            max_warm_users=args.max_warm_users,
            user_select=args.user_select,
            cutoff_quantile=args.cutoff_quantile,
            processed_dir=source,
            out_dir=out,
        )
        return

    downsample_processed(
        max_users=args.max_users,
        max_per_user=args.max_per_user,
        min_user_reviews=args.min_user_reviews,
        min_item_reviews=args.min_item_reviews,
        processed_dir=Path(args.source_dir or PROCESSED_DIR),
    )


if __name__ == "__main__":
    main()
