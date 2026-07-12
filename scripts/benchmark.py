#!/usr/bin/env python
"""Cross-regime benchmark: one dataset, one split, all models, four views.

The active dataset should be the cross-regime Philadelphia slice (dense core +
real cold tail). Every model trains once on a global-time split, then the same
recommendations are scored four ways:

    overall    all test positives
    warm       user and item both seen in train
    cold-item  target item unseen in train
    cold-user  user unseen in train

Build the dataset:

    python scripts/make_crossregime.py
    cp data/processed_philly_xreg/*.parquet data/processed/

Run:

    python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.recsys.config import settings
from src.recsys.data import load_dataset
from src.recsys.eval import evaluate, global_temporal_split
from src.recsys.models import (
    ALSRecommender,
    BPRRecommender,
    ContentBasedRecommender,
    MultiRetriever,
    PopularityRecommender,
    SASRecRecommender,
    SocialRecommender,
    TwoStageRecommender,
    TwoTowerRecommender,
)

REPORT_PATH = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"


def build_unified_two_stage(ds) -> TwoStageRecommender:
    """Multi-retriever pipeline meant to span warm, cold-item, and cold-user."""
    retriever = MultiRetriever(
        [
            TwoTowerRecommender(dim=64, epochs=10),
            ContentBasedRecommender(items=ds.items),
            SocialRecommender(social=ds.social),
            PopularityRecommender(),
        ]
    )
    return TwoStageRecommender(
        retriever, candidate_n=200, use_social=True, social=ds.social
    )


def build_models(ds):
    """Popularity, ALS, BPR, SASRec, two-stage baseline, unified two-stage."""
    return {
        "popularity": PopularityRecommender(),
        "als": ALSRecommender(factors=64, iterations=15),
        "bpr": BPRRecommender(factors=64, iterations=80),
        "sasrec": SASRecRecommender(dim=64, epochs=15, max_len=50),
        "two_stage": TwoStageRecommender(
            TwoTowerRecommender(dim=64, epochs=10), candidate_n=200, use_social=False
        ),
        "two_stage_unified": build_unified_two_stage(ds),
    }


def build_slices(ds, cutoff_quantile: float):
    """Global-time split; cold sets are test entities not seen in train."""
    u, i = settings.user_col, settings.item_col
    train, test_pos = global_temporal_split(ds.interactions, cutoff_quantile=cutoff_quantile)
    train_users = set(train[u].astype(str))
    train_items = set(train[i].astype(str))

    test_items, test_users = set(), set()
    for user_id, items in test_pos.items():
        test_users.add(str(user_id))
        test_items |= {str(x) for x in items}
    cold_items = test_items - train_items
    cold_users = test_users - train_users
    return train, test_pos, cold_items, cold_users


def slice_ground_truth(test_pos, cold_items, cold_users):
    """Partition test positives into warm / cold-item / cold-user (overlap dropped)."""
    cold_items = {str(x) for x in cold_items}
    cold_users = {str(x) for x in cold_users}
    warm, ci, cu = {}, {}, {}
    for user_id, items in test_pos.items():
        u = str(user_id)
        u_cold = u in cold_users
        for it in items:
            it = str(it)
            i_cold = it in cold_items
            if u_cold and i_cold:
                continue
            if u_cold:
                cu.setdefault(u, set()).add(it)
            elif i_cold:
                ci.setdefault(u, set()).add(it)
            else:
                warm.setdefault(u, set()).add(it)
    return warm, ci, cu


def run_benchmark(ds, cutoff_quantile: float, k: int):
    train, test_pos, cold_items, cold_users = build_slices(ds, cutoff_quantile)
    warm_gt, ci_gt, cu_gt = slice_ground_truth(test_pos, cold_items, cold_users)
    users = sorted(set(warm_gt) | set(ci_gt) | set(cu_gt))
    split_desc = f"global-time split at q={cutoff_quantile}"

    counts = dict(
        n_cold_items=len(cold_items),
        n_cold_users=len(cold_users),
        warm_u=len(warm_gt),
        ci_u=len(ci_gt),
        cu_u=len(cu_gt),
    )

    print(f"split: {split_desc}", flush=True)
    print(
        f"cold items={counts['n_cold_items']:,} cold users={counts['n_cold_users']:,}; "
        f"train {len(train):,} rows",
        flush=True,
    )
    print(
        f"slice users: warm={counts['warm_u']:,} cold_item={counts['ci_u']:,} "
        f"cold_user={counts['cu_u']:,}\n",
        flush=True,
    )

    rows = []
    t_all = time.time()
    for name, model in build_models(ds).items():
        t = time.time()
        model.fit(train)
        recs = model.recommend(users, k=k)
        dt = time.time() - t

        overall = evaluate(recs, test_pos, ks=(k, 10))
        warm = evaluate(recs, warm_gt, ks=(k,)) if warm_gt else {}
        ci = evaluate(recs, ci_gt, ks=(k,)) if ci_gt else {}
        cu = evaluate(recs, cu_gt, ks=(k,)) if cu_gt else {}

        row = {
            "model": name,
            f"overall_r@{k}": overall.get(f"recall@{k}", 0.0),
            f"warm_r@{k}": warm.get(f"recall@{k}", 0.0),
            f"cold_item_r@{k}": ci.get(f"recall@{k}", 0.0),
            f"cold_user_r@{k}": cu.get(f"recall@{k}", 0.0),
            "ndcg@10": overall.get("ndcg@10", 0.0),
            "time_s": round(dt, 1),
        }
        rows.append(row)
        print(
            f"{name:20s} overall={row[f'overall_r@{k}']:.4f} "
            f"warm={row[f'warm_r@{k}']:.4f} "
            f"cold_item={row[f'cold_item_r@{k}']:.4f} "
            f"cold_user={row[f'cold_user_r@{k}']:.4f} ({dt:.1f}s)",
            flush=True,
        )

    total = time.time() - t_all
    df = pd.DataFrame(rows).set_index("model")
    return df, split_desc, counts, total


def write_report(df, ds, k: int, split_desc: str, counts: dict, total: float) -> None:
    cols = [
        f"overall_r@{k}",
        f"warm_r@{k}",
        f"cold_item_r@{k}",
        f"cold_user_r@{k}",
        "ndcg@10",
    ]
    header = "| model | " + " | ".join(cols) + " | fit+rec (s) |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [header, sep]
    for name in df.index:
        cells = " | ".join(f"{df.loc[name, c]:.4f}" for c in cols)
        lines.append(f"| {name} | {cells} | {df.loc[name, 'time_s']:.1f} |")
    table = "\n".join(lines)

    content = f"""# Benchmark results

Auto-generated by `scripts/benchmark.py`. One realistic dataset (dense core +
cold tail), one global-time split, every model scored on **overall**, **warm**,
**cold-item**, and **cold-user**. Reproduce with:

```bash
python scripts/benchmark.py
```

## Cross-regime Philadelphia

One model, one training set, four views. Cold entities arise naturally under a
single wall-clock cutoff — not simulated. See `docs/techniques.md`.

- **Dataset:** {ds.summary()}
- **Split:** {split_desc}
- **Cold sets:** {counts['n_cold_items']:,} cold items, {counts['n_cold_users']:,} cold users
- **Slice users:** warm={counts['warm_u']:,}, cold-item={counts['ci_u']:,}, cold-user={counts['cu_u']:,}
- **Run date:** {date.today().isoformat()}
- **Total run time:** {total:.0f}s (single-thread BLAS, laptop CPU)

{table}

## How to read this

- **overall_r@{k}** — recall across all test positives (the blended score).
- **warm_r@{k}** — dense regime: user and item both seen in train (collaborative signal).
- **cold_item_r@{k}** — target item never seen in train (content signal).
- **cold_user_r@{k}** — user never seen in train (social / popularity signal).
- **ndcg@10** — ranking quality on the full test set.

Collaborative models (als, bpr, sasrec, two_stage) should be strong on warm and
near-zero on both cold slices. Popularity lifts cold-user. The unified two-stage
(`MultiRetriever` → ranker) is the architecture meant to stay non-zero across all
three regimes. Cold-item recall stays low on Yelp because item text is only
name + categories — see `docs/techniques.md`.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff-quantile", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=max(settings.eval_ks))
    ap.add_argument("--no-write", action="store_true", help="Print only; skip benchmarks.md.")
    args = ap.parse_args()

    ds = load_dataset()
    print("DATA:", ds.summary(), flush=True)

    df, split_desc, counts, total = run_benchmark(ds, args.cutoff_quantile, args.k)

    print(f"\n=== CROSS-REGIME BENCHMARK (recall@{args.k}) ===")
    print(df.to_string())

    processed_dir = settings.paths["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "benchmark_results.csv")

    if not args.no_write:
        write_report(df, ds, args.k, split_desc, counts, total)
        print(f"\nTOTAL {total:.1f}s — written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
