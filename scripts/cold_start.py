#!/usr/bin/env python
"""Cold-start experiment: one model, one training set, three test slices.

The claim we test is "a single pipeline handles warm users/items, cold items,
and cold users." So we train **once** on a hard training set that has both
carve-outs, then score the *same* recommendations against three slices of the
test ground truth (see docs/techniques.md):

    warm       user not cold AND item not cold   → collaborative signal
    cold-item  target item held out of train     → content (text) signal
    cold-user  user held out of train (keep=0)    → social / popularity signal

Construction (a pure function of dataset + seed + fractions; nothing is saved):

    temporal_split → cold_item_holdout → cold_user_holdout → train_hard → fit once

Overlap (cold user AND cold item) is dropped from all slices for clean attribution.

    python scripts/cold_start.py --cold-item-fraction 0.15 --cold-user-fraction 0.15
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
    cold_user_holdout,
    evaluate,
    temporal_split,
)
from src.recsys.models import (
    ALSRecommender,
    ContentTwoTowerRecommender,
    MultiRetriever,
    PopularityRecommender,
    SocialRecommender,
    TwoStageRecommender,
    TwoTowerRecommender,
)


def build_unified_two_stage(ds) -> TwoStageRecommender:
    """One strong two-stage that covers every regime.

    Retrieval unions complementary sources so *something* fires in each regime:
    two-tower (warm), content-two-tower (cold items), social (cold users),
    popularity (universal fallback). The ranker then re-orders using retrieval
    rank + user/item activity + a social feature, learning to lean on whichever
    signal exists for a given (user, item).
    """
    retriever = MultiRetriever(
        [
            TwoTowerRecommender(dim=64, epochs=10),
            ContentTwoTowerRecommender(items=ds.items, dim=64, epochs=10),
            SocialRecommender(social=ds.social),
            PopularityRecommender(),
        ]
    )
    return TwoStageRecommender(
        retriever, candidate_n=200, use_social=True, social=ds.social
    )


def build_models(ds):
    """Contrasting retrievers + the unified pipeline that should span all slices."""
    return {
        "popularity": PopularityRecommender(),
        "als": ALSRecommender(factors=64, iterations=15),
        "two_tower": TwoTowerRecommender(dim=64, epochs=10),
        "content_two_tower": ContentTwoTowerRecommender(items=ds.items, dim=64, epochs=10),
        "social": SocialRecommender(social=ds.social),
        "two_stage_unified": build_unified_two_stage(ds),
    }


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
                continue  # overlap: ambiguous attribution, drop from all slices
            if u_cold:
                cu.setdefault(u, set()).add(it)
            elif i_cold:
                ci.setdefault(u, set()).add(it)
            else:
                warm.setdefault(u, set()).add(it)
    return warm, ci, cu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold-item-fraction", type=float, default=0.15)
    ap.add_argument("--cold-user-fraction", type=float, default=0.15)
    ap.add_argument("--keep", type=int, default=0, help="Interactions kept per cold user.")
    ap.add_argument("--k", type=int, default=max(settings.eval_ks))
    args = ap.parse_args()

    ds = load_dataset()
    print("DATA:", ds.summary(), flush=True)

    train, test_pos = temporal_split(ds.interactions)
    train_i, cold_items = cold_item_holdout(
        train, test_pos, cold_fraction=args.cold_item_fraction, seed=settings.seed
    )
    train_hard, cold_users = cold_user_holdout(
        train_i,
        test_pos,
        cold_fraction=args.cold_user_fraction,
        keep=args.keep,
        seed=settings.seed + 1,
    )
    warm_gt, ci_gt, cu_gt = slice_ground_truth(test_pos, cold_items, cold_users)
    users = sorted(set(warm_gt) | set(ci_gt) | set(cu_gt))

    print(
        f"held out {len(cold_items):,} cold items, {len(cold_users):,} cold users "
        f"(keep={args.keep}); train {len(train):,} -> {len(train_hard):,} rows",
        flush=True,
    )
    print(
        f"slice users: warm={len(warm_gt):,} cold_item={len(ci_gt):,} "
        f"cold_user={len(cu_gt):,}\n",
        flush=True,
    )

    kk = args.k
    rows = []
    for name, model in build_models(ds).items():
        t = time.time()
        model.fit(train_hard)
        recs = model.recommend(users, k=kk)
        dt = time.time() - t
        overall = evaluate(recs, test_pos, ks=(kk,))
        warm = evaluate(recs, warm_gt, ks=(kk,)) if warm_gt else {}
        ci = evaluate(recs, ci_gt, ks=(kk,)) if ci_gt else {}
        cu = evaluate(recs, cu_gt, ks=(kk,)) if cu_gt else {}
        rows.append(
            {
                "model": name,
                f"overall_r@{kk}": overall.get(f"recall@{kk}", 0.0),
                f"warm_r@{kk}": warm.get(f"recall@{kk}", 0.0),
                f"cold_item_r@{kk}": ci.get(f"recall@{kk}", 0.0),
                f"cold_user_r@{kk}": cu.get(f"recall@{kk}", 0.0),
                "time_s": round(dt, 1),
            }
        )
        print(
            f"{name:20s} warm={rows[-1][f'warm_r@{kk}']:.4f} "
            f"cold_item={rows[-1][f'cold_item_r@{kk}']:.4f} "
            f"cold_user={rows[-1][f'cold_user_r@{kk}']:.4f} ({dt:.1f}s)",
            flush=True,
        )

    df = pd.DataFrame(rows).set_index("model")
    print("\n=== COLD-START EXPERIMENT (recall@%d, one model per row) ===" % kk)
    print(df.to_string())
    print(
        "\nRead: same ranked list scored three ways. Collaborative models die on "
        "cold_item AND cold_user; content lifts cold_item; social/popularity lift "
        "cold_user; the unified two-stage should stay non-zero across all three."
    )


if __name__ == "__main__":
    main()
