#!/usr/bin/env python
"""End-to-end Phase-1 demo: load data -> temporal split -> baselines -> compare.

Runs on synthetic data out of the box (no download). If you've built a real Yelp
subset into data/processed/, it uses that instead automatically.

    python scripts/demo.py
"""

import os
import sys
from pathlib import Path

# Single-thread BLAS: avoids the OpenBLAS threadpool warning from implicit and
# gives more stable timing on small lab-sized problems.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recsys.config import settings
from src.recsys.data import load_dataset
from src.recsys.eval import evaluate, temporal_split
from src.recsys.models import (
    ALSRecommender,
    ItemCFRecommender,
    PopularityRecommender,
    TwoTowerRecommender,
)


def main() -> None:
    print("Loading dataset (auto-generates synthetic if none present)...")
    ds = load_dataset()
    print(" ", ds.summary())

    train, test_positives = temporal_split(ds.interactions)
    test_users = list(test_positives.keys())
    print(
        f"  temporal split: {len(train):,} train interactions, "
        f"{len(test_users):,} test users with held-out positives\n"
    )

    models = [
        PopularityRecommender(),
        PopularityRecommender(recency_weighted=True),
        ItemCFRecommender(top_n_neighbors=50),
    ]
    # ALS needs the optional 'implicit' package; skip gracefully if missing.
    try:
        import implicit  # noqa: F401

        models.append(ALSRecommender(factors=64, iterations=15))
    except ImportError:
        print("(implicit not installed -> skipping ALS. `pip install implicit` to enable.)\n")

    # Two-tower needs torch; skip gracefully if missing.
    try:
        import torch  # noqa: F401

        models.append(TwoTowerRecommender(dim=64, epochs=15))
    except ImportError:
        print("(torch not installed -> skipping two-tower. `pip install torch` to enable.)\n")

    # Generate enough recommendations to score the largest K we evaluate at.
    max_k = max(settings.eval_ks)

    rows = []
    for model in models:
        label = model.name + ("_recency" if getattr(model, "recency_weighted", False) else "")
        model.fit(train)
        recs = model.recommend(test_users, k=max_k)
        metrics = evaluate(recs, test_positives, ks=settings.eval_ks, n_items=ds.n_items)
        rows.append((label, metrics))

    _print_table(rows)


def _print_table(rows) -> None:
    metric_keys = list(rows[0][1].keys())
    header = f"{'model':<20}" + "".join(f"{m:>14}" for m in metric_keys)
    print(header)
    print("-" * len(header))
    for label, metrics in rows:
        line = f"{label:<20}" + "".join(f"{metrics[m]:>14.4f}" for m in metric_keys)
        print(line)
    print("\nReminder: beat these baselines before trusting any fancy model.")


if __name__ == "__main__":
    main()
