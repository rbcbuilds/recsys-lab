#!/usr/bin/env python
"""Cold-start experiment: can a model recommend items with no interactions?

Simulates cold items by stripping a fraction of items entirely from the training
set (``cold_item_holdout``), then compares collaborative retrievers (which cannot
represent unseen items) against content-aware ones (which embed items from text).

    python scripts/cold_start.py --cold-fraction 0.15

Headline you should see: collaborative models (ALS, two-tower) score ~0 recall on
cold items, while content_based and content_two_tower score > 0 — the concrete
answer to "how do you handle new items?". Warm-item recall is printed alongside so
the trade-off (content is weaker on warm users) stays honest.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.recsys.config import settings
from src.recsys.data import load_dataset
from src.recsys.eval import (
    cold_item_holdout,
    evaluate,
    evaluate_cold_items,
    temporal_split,
)
from src.recsys.models import (
    ALSRecommender,
    ContentBasedRecommender,
    ContentTwoTowerRecommender,
    TwoTowerRecommender,
)


def build_models(ds):
    return {
        "als": ALSRecommender(factors=64, iterations=15),
        "two_tower": TwoTowerRecommender(dim=64, epochs=10),
        "content_based": ContentBasedRecommender(items=ds.items),
        "content_two_tower": ContentTwoTowerRecommender(items=ds.items, dim=64, epochs=10),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold-fraction", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=max(settings.eval_ks))
    args = ap.parse_args()

    ds = load_dataset()
    print("DATA:", ds.summary(), flush=True)
    train, test_pos = temporal_split(ds.interactions)
    train_cold, cold_items = cold_item_holdout(
        train, test_pos, cold_fraction=args.cold_fraction
    )
    users = list(test_pos.keys())
    print(
        f"held out {len(cold_items):,} cold items; "
        f"train {len(train):,} -> {len(train_cold):,} rows\n",
        flush=True,
    )

    rows = []
    for name, model in build_models(ds).items():
        t = time.time()
        model.fit(train_cold)
        recs = model.recommend(users, k=args.k)
        dt = time.time() - t
        warm = evaluate(recs, test_pos, ks=(args.k,))
        cold = evaluate_cold_items(recs, test_pos, cold_items, ks=(args.k,))
        rows.append(
            {
                "model": name,
                f"cold_recall@{args.k}": cold.get(f"recall@{args.k}", 0.0),
                f"cold_ndcg@{args.k}": cold.get(f"ndcg@{args.k}", 0.0),
                f"warm_recall@{args.k}": warm[f"recall@{args.k}"],
                "cold_users": cold.get("n_users", 0),
                "time_s": round(dt, 1),
            }
        )
        print(
            f"{name:20s} cold_recall@{args.k}={rows[-1][f'cold_recall@{args.k}']:.4f} "
            f"warm_recall@{args.k}={rows[-1][f'warm_recall@{args.k}']:.4f} "
            f"({dt:.1f}s)",
            flush=True,
        )

    df = pd.DataFrame(rows).set_index("model")
    print("\n=== COLD-START EXPERIMENT ===")
    print(df.to_string())
    print(
        "\nRead: cold_recall is recall restricted to held-out cold items. "
        "Collaborative models can't represent them (≈0); content models can."
    )


if __name__ == "__main__":
    main()
