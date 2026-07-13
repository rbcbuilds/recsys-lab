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

Scaffolds (Tier 1–3 — documented interfaces, raise NotImplementedError):
    review_text.build_enriched_item_text   - aggregate Yelp reviews into item text
    ranker_features.ExtendedRankerFeatures - content_score, retriever source, activity
    graph.LightGCNRecommender              - interaction-graph propagation (Tier 2)
    multimodal.ImageEmbeddingRecommender   - CLIP image embeddings (Tier 2)
    hstu.HSTURecommender                   - industrial sequential model (Tier 3)
    semantic_ids.SemanticIDRecommender     - TIGER-style semantic item tokens (Tier 3)
    contrastive.ContrastiveTwoTowerRecommender - hard-negative two-tower (Tier 3)
    eval.slate.diversify_slate             - MMR / DPP post-ranking diversity (Tier 3)
    eval.debias.evaluate_ips               - propensity-weighted metrics (Tier 3)
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
