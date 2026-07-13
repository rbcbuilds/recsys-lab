"""Debiased / causal-style evaluation  [SCAFFOLD — Tier 3].

Average Recall@K on logged data confounds **model quality** with **exposure
bias** (popular items are seen more → easier to hit in test). Senior interviews
ask how you'd report metrics honestly.

Techniques to scaffold:

| Method | Idea | When to use |
|---|---|---|
| **IPS** | Weight each test event by 1 / P(expose item) | Known logging policy |
| **SNIPS** | Self-normalized IPS (more stable) | Same, small samples |
| **Stratified slices** | Report warm / cold / activity buckets separately | Always (you do this) |
| **Counterfactual ranking** | Train on propensity-scored pairs | Research / offline eval |

This repo's cross-regime slices are already a form of stratified analysis.
Extend with propensity weights from item popularity::

    P(expose i) ∝ pop(i)^alpha

and report IPS-weighted recall alongside raw recall.

Interview point: "Popularity wins cold-user partly because it's the exposure
baseline — here's IPS-weighted lift for the unified model."

References: Schnabel et al. (IPS for recsys), 2016; Yang et al. (unbiased LTR).
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def item_propensity(
    train: pd.DataFrame,
    smoothing: float = 1.0,
) -> Dict[str, float]:
    """Estimate P(item appears in a random user's history) from train counts."""
    raise NotImplementedError(
        "Compute normalized item counts from train; use as IPS denominators. "
        "See this module's docstring."
    )


def evaluate_ips(
    recommendations: Dict[str, list],
    ground_truth: Dict[str, set],
    propensity: Dict[str, float],
    k: int = 10,
    clip_max: float = 100.0,
) -> Dict[str, float]:
    """IPS-weighted recall@K (and optionally SNIPS variant)."""
    raise NotImplementedError
