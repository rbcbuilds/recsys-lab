"""Semantic / hierarchical item IDs for generative retrieval  [SCAFFOLD — Tier 3].

Research direction (TIGER, LC-Rec, etc.): map each item to a short token sequence
derived from attributes (category, geo, cluster id) so a generative LM can emit
*compositional* item codes instead of flat business_ids.

Contrast with ``item_token_lm.py``:
  * **Item-token LM here**: one token = one arbitrary item id (flat vocabulary).
  * **Semantic IDs**: token sequence = ``[category_token, cluster_token, leaf]``;
    new items share prefixes with similar catalog entries → better cold transfer.

Suggested exercise:
  1. Cluster item text embeddings (k-means on sentence-transformers).
  2. Assign each item a 2–3 token code: ``<cat>_<cluster>_<local_id>``.
  3. Train causal LM on semantic token sequences; decode for retrieval.
  4. Evaluate cold-item slice vs flat ``ItemTokenLMRecommender``.

References: Rajput et al. (TIGER), 2023; LC-Rec, 2024.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .base import Recommender


class SemanticIDRecommender(Recommender):
    name = "semantic_id_lm"

    def __init__(
        self,
        n_clusters: int = 32,
        tokens_per_item: int = 3,
        dim: int = 64,
        epochs: int = 12,
    ):
        self.n_clusters = n_clusters
        self.tokens_per_item = tokens_per_item
        self.dim = dim
        self.epochs = epochs
        self._item_to_tokens: Dict[str, List[str]] = {}

    def fit(
        self, train: pd.DataFrame, items: Optional[pd.DataFrame] = None
    ) -> "SemanticIDRecommender":
        raise NotImplementedError(
            "Build hierarchical semantic ids from item text clusters, then "
            "train a token LM on those sequences. See this module's docstring."
        )

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError
