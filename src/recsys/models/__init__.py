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
    ItemTokenLMRecommender        - generative causal LM over item-token sequences
    LLMReranker / LLMTwoStage     - language-model scoring re-ranks retrieved candidates
    LightGCNRecommender           - interaction-graph propagation (BPR + LightGCN)
    ImageEmbeddingRecommender     - CLIP image embeddings for cold-item retrieval
    HSTURecommender               - time-aware sequential model (Tier 3)
    SemanticIDRecommender         - compositional semantic item tokens (Tier 3)
    ContrastiveTwoTowerRecommender - hard-negative two-tower training (Tier 3)

Tier 3 eval utilities (``recsys.eval``):
    slate.diversify_slate         - MMR / DPP post-ranking diversity
    debias.evaluate_ips           - IPS-weighted offline metrics
"""

from .base import Recommender
from .bpr import BPRRecommender
from .contrastive import ContrastiveTwoTowerRecommender
from .graph import LightGCNRecommender
from .hstu import HSTURecommender
from .item_cf import ItemCFRecommender
from .item_token_lm import ItemTokenLMRecommender
from .llm_reranker import LLMReranker, LLMTwoStageRecommender
from .matrix_factorization import ALSRecommender
from .multi_retriever import MultiRetriever
from .multimodal import ImageEmbeddingRecommender
from .popularity import PopularityRecommender
from .ranker import LightGBMRanker
from .ranker_features import ExtendedRankerFeatures
from .sasrec import SASRecRecommender
from .semantic_ids import SemanticIDRecommender
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
    "ItemTokenLMRecommender",
    "LLMReranker",
    "LLMTwoStageRecommender",
    "LightGCNRecommender",
    "ImageEmbeddingRecommender",
    "ExtendedRankerFeatures",
    "HSTURecommender",
    "SemanticIDRecommender",
    "ContrastiveTwoTowerRecommender",
]
