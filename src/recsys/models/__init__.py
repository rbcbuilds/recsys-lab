"""Models: working recommenders + scaffolds for remaining Phase-3 modules.

Working now:
    PopularityRecommender         - popularity / trending baseline
    ItemCFRecommender             - item-based collaborative filtering
    ALSRecommender                - matrix factorization (implicit ALS, pointwise)
    BPRRecommender                - same MF family, pairwise ranking loss
    TwoTowerRecommender           - neural retrieval (in-batch negatives + ANN)
    LightGBMRanker                - learning-to-rank re-ranker (candidates only)
    TwoStageRecommender           - retrieve → LightGBM re-rank
    SocialRecommender             - social-neighbor CF from the friend graph
    SASRecRecommender             - self-attentive sequential recommendation
    ContentBasedRecommender       - pure content CF from item text (cold-start)
    ContentTwoTowerRecommender    - two-tower with text in the item tower (cold-start-aware)
    MultiRetriever                - union + reciprocal-rank-fuse several retrievers

Scaffolds (documented interfaces + guidance, raise NotImplementedError):
    graph.LightGCNRecommender
    multimodal.ImageEmbeddingRecommender
"""

from .base import Recommender
from .bpr import BPRRecommender
from .item_cf import ItemCFRecommender
from .matrix_factorization import ALSRecommender
from .multi_retriever import MultiRetriever
from .popularity import PopularityRecommender
from .ranker import LightGBMRanker
from .sasrec import SASRecRecommender
from .social import SocialRecommender
from .text_embeddings import ContentBasedRecommender, ContentTwoTowerRecommender
from .two_stage import TwoStageRecommender
from .two_tower import TwoTowerRecommender

__all__ = [
    "Recommender",
    "PopularityRecommender",
    "ItemCFRecommender",
    "ALSRecommender",
    "BPRRecommender",
    "TwoTowerRecommender",
    "LightGBMRanker",
    "TwoStageRecommender",
    "SocialRecommender",
    "SASRecRecommender",
    "ContentBasedRecommender",
    "ContentTwoTowerRecommender",
    "MultiRetriever",
]
