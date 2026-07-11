"""Ranking metrics for top-K recommendation.

All functions operate on ``recommendations``: ``{user_id: [item_id, ...]}`` in
ranked order, and ``ground_truth``: ``{user_id: {relevant_item_id, ...}}``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Set

from ..config import settings


def recall_at_k(recs: List[str], truth: Set[str], k: int) -> float:
    if not truth:
        return 0.0
    hits = len(set(recs[:k]) & truth)
    return hits / len(truth)


def precision_at_k(recs: List[str], truth: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(recs[:k]) & truth) / k


def hit_rate_at_k(recs: List[str], truth: Set[str], k: int) -> float:
    return 1.0 if set(recs[:k]) & truth else 0.0


def ndcg_at_k(recs: List[str], truth: Set[str], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recs[:k]):
        if item in truth:
            dcg += 1.0 / math.log2(idx + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / ideal if ideal > 0 else 0.0


def coverage(recommendations: Dict[str, List[str]], n_items: int, k: int) -> float:
    """Fraction of the catalog that appears in anyone's top-K (diversity signal)."""
    if n_items == 0:
        return 0.0
    shown: Set[str] = set()
    for recs in recommendations.values():
        shown.update(recs[:k])
    return len(shown) / n_items


def evaluate(
    recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    ks: Iterable[int] | None = None,
    n_items: int | None = None,
) -> Dict[str, float]:
    """Average metrics across all users present in ``ground_truth``.

    Returns a flat dict like ``{"recall@10": .., "ndcg@10": .., "hitrate@10": ..,
    "coverage@10": ..}`` for each K.
    """
    ks = list(settings.eval_ks if ks is None else ks)
    users = list(ground_truth.keys())
    results: Dict[str, float] = {}

    for k in ks:
        recalls, ndcgs, hits, precs = [], [], [], []
        for user_id in users:
            truth = ground_truth[user_id]
            recs = recommendations.get(user_id, [])
            recalls.append(recall_at_k(recs, truth, k))
            ndcgs.append(ndcg_at_k(recs, truth, k))
            hits.append(hit_rate_at_k(recs, truth, k))
            precs.append(precision_at_k(recs, truth, k))
        n = max(len(users), 1)
        results[f"recall@{k}"] = sum(recalls) / n
        results[f"ndcg@{k}"] = sum(ndcgs) / n
        results[f"hitrate@{k}"] = sum(hits) / n
        results[f"precision@{k}"] = sum(precs) / n
        if n_items:
            results[f"coverage@{k}"] = coverage(recommendations, n_items, k)

    return results
