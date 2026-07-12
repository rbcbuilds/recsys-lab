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


def evaluate_by_activity(
    recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    train_counts: Dict[str, int],
    bins: Iterable[int] = (0, 5, 20, 10_000),
    k: int | None = None,
) -> "Dict[str, Dict[str, float]]":
    """Slice metrics by how much *training* history each user had.

    This is the key view for social recsys: social signal typically helps
    **sparse** users (few interactions) most. ``train_counts`` maps user_id ->
    number of training interactions. ``bins`` are right-open edges, so
    (0, 5, 20, inf) yields buckets "0-4", "5-19", "20+".

    Returns ``{bucket_label: metrics_dict}`` (metrics via :func:`evaluate`).
    """
    k = settings.top_k if k is None else k
    edges = list(bins)
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo}-{hi - 1}" if hi < 10_000 else f"{lo}+")

    buckets: Dict[str, Dict[str, Set[str]]] = {lab: {} for lab in labels}
    for user_id, truth in ground_truth.items():
        c = train_counts.get(user_id, 0)
        for (lo, hi), lab in zip(zip(edges[:-1], edges[1:]), labels):
            if lo <= c < hi:
                buckets[lab][user_id] = truth
                break

    out: Dict[str, Dict[str, float]] = {}
    for lab, gt in buckets.items():
        if not gt:
            out[lab] = {"n_users": 0}
            continue
        m = evaluate(recommendations, gt, ks=(k,))
        m["n_users"] = len(gt)
        out[lab] = m
    return out


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
