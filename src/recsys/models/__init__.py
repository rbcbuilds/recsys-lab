"""Models: working Phase-1/2 recommenders + scaffolds for Phase 3.

Working now:
    PopularityRecommender      - popularity / trending baseline
    ItemCFRecommender          - item-based collaborative filtering
    ALSRecommender             - matrix factorization (implicit ALS)
    TwoTowerRecommender        - neural retrieval (in-batch negatives + ANN)
    LightGBMRanker             - learning-to-rank re-ranker (candidates only)
    TwoStageRecommender        - retrieve (e.g. two-tower) → LightGBM re-rank

Scaffolds (documented interfaces + guidance, raise NotImplementedError):
    text_embeddings.TextEmbeddingRecommender
    graph.LightGCNRecommender
    social.SocialRecommender
    multimodal.ImageEmbeddingRecommender
"""

from .base import Recommender
from .item_cf import ItemCFRecommender
from .matrix_factorization import ALSRecommender
from .popularity import PopularityRecommender
from .ranker import LightGBMRanker
from .two_stage import TwoStageRecommender
from .two_tower import TwoTowerRecommender

__all__ = [
    "Recommender",
    "PopularityRecommender",
    "ItemCFRecommender",
    "ALSRecommender",
    "TwoTowerRecommender",
    "LightGBMRanker",
    "TwoStageRecommender",
]
