"""Debiased / causal-style evaluation (IPS / SNIPS).

Down-weights easy popular hits so offline metrics are less confounded by
exposure bias.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..config import settings
from .metrics import recall_at_k


def item_propensity(
    train: pd.DataFrame,
    smoothing: float = 1.0,
) -> Dict[str, float]:
    """Estimate P(item appears in a random user's history) from train counts."""
    i = settings.item_col
    counts = train.groupby(i).size()
    n_items = len(counts)
    total = float(counts.sum()) + smoothing * n_items
    prop: Dict[str, float] = {}
    for iid, c in counts.items():
        prop[str(iid)] = (float(c) + smoothing) / total
    return prop


def evaluate_ips(
    recommendations: Dict[str, list],
    ground_truth: Dict[str, set],
    propensity: Dict[str, float],
    k: int = 10,
    clip_max: float = 100.0,
    snips: bool = False,
) -> Dict[str, float]:
    """IPS-weighted recall@K; set ``snips=True`` for self-normalized weights."""
    num = 0.0
    den = 0.0
    per_user_num: list[float] = []
    per_user_den: list[float] = []

    for user_id, truth in ground_truth.items():
        if not truth:
            continue
        recs = recommendations.get(user_id, [])[:k]
        u_num, u_den = 0.0, 0.0
        for item in truth:
            p = max(propensity.get(str(item), 1e-6), 1e-6)
            w = min(1.0 / p, clip_max)
            u_den += w
            if str(item) in recs:
                u_num += w
        if u_den > 0:
            per_user_num.append(u_num)
            per_user_den.append(u_den)
            num += u_num
            den += u_den

    if snips and per_user_den:
        snips_val = float(
            sum(n / d for n, d in zip(per_user_num, per_user_den) if d > 0)
            / len(per_user_den)
        )
    else:
        snips_val = 0.0

    ips = num / den if den > 0 else 0.0

    raw = 0.0
    n_users = 0
    for user_id, truth in ground_truth.items():
        if not truth:
            continue
        raw += recall_at_k(recommendations.get(user_id, []), truth, k)
        n_users += 1
    raw /= n_users if n_users else 1.0

    out = {
        f"ips_recall@{k}": ips,
        f"raw_recall@{k}": raw,
    }
    if snips:
        out[f"snips_recall@{k}"] = snips_val
    return out


def evaluate_ips_slices(
    recommendations: Dict[str, list],
    ground_truth_slices: Dict[str, Dict[str, set]],
    propensity: Dict[str, float],
    k: int = 10,
    clip_max: float = 100.0,
) -> Dict[str, Dict[str, float]]:
    """IPS metrics per named slice (e.g. warm / cold-user)."""
    return {
        name: evaluate_ips(recommendations, ground_truth, propensity, k, clip_max)
        for name, ground_truth in ground_truth_slices.items()
    }
