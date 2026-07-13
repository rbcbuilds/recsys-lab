"""Graph recommendation (LightGCN / Node2Vec)  [SCAFFOLD — Tier 2].

Treat the user-item interactions as a bipartite graph and learn embeddings by
propagating information along edges. LightGCN is the deployed favorite: it
strips GCNs down to just neighborhood aggregation (no feature transforms or
nonlinearities) and consistently beats plain matrix factorization on implicit
feedback.

Two learnable exercises:
  * Node2Vec / DeepWalk: random walks on the graph -> word2vec-style embeddings.
    Simple, framework-light (networkx + gensim), great for intuition.
  * LightGCN: K layers of normalized neighbor aggregation over the user-item
    adjacency; final embedding = mean of layer embeddings. Train with BPR loss.

Note vs. social: this operates on the *interaction* graph and needs NO social
data — it's the mainstream graph-recsys technique. The social variant
(models/social.py) adds user-user edges on top.

References: He et al. 2020 (LightGCN); Grover & Leskovec 2016 (node2vec).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Recommender


class LightGCNRecommender(Recommender):
    name = "lightgcn"

    def __init__(self, dim: int = 64, n_layers: int = 3, epochs: int = 20):
        self.dim = dim
        self.n_layers = n_layers
        self.epochs = epochs

    def fit(self, train: pd.DataFrame) -> "LightGCNRecommender":
        raise NotImplementedError(
            "Graph recsys is a Phase-3 exercise. Start with Node2Vec on the "
            "user-item graph for intuition, then implement LightGCN (K-layer "
            "neighbor aggregation + BPR loss). See this module's docstring."
        )

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError
