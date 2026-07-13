"""Contrastive / hard-negative two-tower retrieval  [SCAFFOLD — Tier 3].

Standard two-tower uses in-batch negatives (random other items in the minibatch).
Contrastive extensions (SimGCL, CLRec, etc.) add:
  * **Hard negatives**: items popular but not clicked by this user.
  * **Augmentation**: dropout on embeddings for consistency regularization.
  * **Temperature / margin** tuning for better separation on warm users.

Drop-in replacement for ``TwoTowerRecommender`` training loop — same inference
(dot product + ANN), different loss. Compare warm_r@20 vs baseline two-tower on
the cross-regime slice; expect little cold-user/cold-item lift (still needs
history).

Suggested first step: add ``HardNegativeTwoTowerRecommender`` that samples
negatives from global popularity excluding user's positives (cheap hard neg).

References: Yu et al. (SimGCL), 2022; CLRec, 2021.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .two_tower import TwoTowerRecommender


class ContrastiveTwoTowerRecommender(TwoTowerRecommender):
    """Two-tower with hard-negative / augmentation training (scaffold)."""

    name = "contrastive_two_tower"

    def __init__(
        self,
        hard_negative_k: int = 5,
        augmentation_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hard_negative_k = hard_negative_k
        self.augmentation_dropout = augmentation_dropout

    def fit(self, train: pd.DataFrame) -> "ContrastiveTwoTowerRecommender":
        raise NotImplementedError(
            "Override the training loop to sample hard negatives per user and "
            "optionally apply embedding dropout. See this module's docstring."
        )

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError
