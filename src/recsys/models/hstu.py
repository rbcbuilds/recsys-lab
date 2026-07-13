"""HSTU — hierarchical sequential transduction units  [SCAFFOLD — Tier 3].

Meta's industrial successor to SASRec/BERT4Rec for long, noisy action sequences
(ads, feed ranking). Uses attention with relative time biases and larger-scale
training — same *role* as SASRec in your stack: sequential warm-user signal.

Suggested integration (lowest effort first):
  1. **Ranker feature** (like ``use_sasrec``): fit HSTU on train sequences,
     ``score_candidates`` on the fused pool — no 6th retriever.
  2. **Standalone retriever**: compare warm_r@20 vs SASRec on cross-regime slice.
  3. Full implementation: see Zhai et al. 2024 (HSTU); start from ``sasrec.py``
     and add relative-time attention + pointwise aggregated attention.

References: Zhai et al., "Actions Speak Louder than Words", 2024.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import IndexedRecommender


class HSTURecommender(IndexedRecommender):
    name = "hstu"

    def __init__(
        self,
        dim: int = 64,
        max_len: int = 100,
        n_layers: int = 2,
        epochs: int = 15,
    ):
        self.dim = dim
        self.max_len = max_len
        self.n_layers = n_layers
        self.epochs = epochs

    def fit(self, train: pd.DataFrame) -> "HSTURecommender":
        raise NotImplementedError(
            "HSTU is a Tier-3 exercise. Start by cloning sasrec.py and adding "
            "relative time embeddings; or expose score_candidates for the ranker. "
            "See this module's docstring."
        )

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError

    def score_candidates(
        self, candidates: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError
