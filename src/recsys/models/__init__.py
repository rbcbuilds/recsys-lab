"""Models: working Phase-1 baselines + scaffolds for Phase 2-3.

Working now:
    PopularityRecommender      - popularity / trending baseline
    ItemCFRecommender          - item-based collaborative filtering
    ALSRecommender             - matrix factorization (implicit ALS)
    TwoTowerRecommender        - neural retrieval (in-batch negatives + ANN)

Scaffolds (documented interfaces + guidance, raise NotImplementedError):
    ranker.LightGBMRanker
    text_embeddings.TextEmbeddingRecommender
    graph.LightGCNRecommender
    social.SocialRecommender
    multimodal.ImageEmbeddingRecommender
"""

from .base import Recommender
from .item_cf import ItemCFRecommender
from .matrix_factorization import ALSRecommender
from .popularity import PopularityRecommender
from .two_tower import TwoTowerRecommender

__all__ = [
    "Recommender",
    "PopularityRecommender",
    "ItemCFRecommender",
    "ALSRecommender",
    "TwoTowerRecommender",
]
