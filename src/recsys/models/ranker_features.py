"""Extended ranker features for cross-regime routing."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..config import settings
from .base import Recommender

# Maps MultiRetriever child ``name`` → ranker feature column.
SOURCE_NAME_TO_FEATURE = {
    "two_tower": "retriever_source_two_tower",
    "content_based": "retriever_source_content",
    "social": "retriever_source_social",
    "popularity": "retriever_source_popularity",
    "image_embedding": "retriever_source_image",
}


class ExtendedRankerFeatures:
    """Collector for optional ranker feature dicts (not a standalone recommender)."""

    FEATURE_NAMES = (
        "content_score",
        "image_score",
        "retriever_source_two_tower",
        "retriever_source_content",
        "retriever_source_social",
        "retriever_source_popularity",
        "retriever_source_image",
        "user_activity_bucket",
        "item_age_days",
    )

    def __init__(self, train: pd.DataFrame):
        u, i, t = settings.user_col, settings.item_col, settings.time_col
        uc = train.groupby(u).size()
        self._user_bucket: Dict[str, float] = {}
        for uid, n in uc.items():
            if n <= 1:
                b = 0.0
            elif n <= 5:
                b = 1.0
            elif n <= 20:
                b = 2.0
            else:
                b = 3.0
            self._user_bucket[str(uid)] = b

        ts = pd.to_datetime(train[t])
        self._ref_time = ts.max()
        first = train.groupby(i)[t].min()
        self._item_age_days: Dict[str, float] = {}
        for iid, t0 in first.items():
            days = (self._ref_time - pd.to_datetime(t0)).total_seconds() / 86400.0
            self._item_age_days[str(iid)] = float(max(days, 0.0))

    def build(
        self,
        candidates: Dict[str, List[str]],
        retrieval_sources: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        content_scores: Optional[Dict[str, Dict[str, float]]] = None,
        image_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Per-user, per-item feature values for ranker training/inference."""
        retrieval_sources = retrieval_sources or {}
        content_scores = content_scores or {}
        image_scores = image_scores or {}
        out: Dict[str, Dict[str, Dict[str, float]]] = {}

        for user_id, items in candidates.items():
            u = str(user_id)
            user_feats: Dict[str, Dict[str, float]] = {}
            src_map = retrieval_sources.get(u, {})
            cscores = content_scores.get(u, {})
            iscores = image_scores.get(u, {})
            for item_id in items:
                it = str(item_id)
                src = src_map.get(it, {})
                row = {
                    "content_score": float(cscores.get(it, 0.0)),
                    "image_score": float(iscores.get(it, 0.0)),
                    "retriever_source_two_tower": float(src.get("two_tower", 0.0)),
                    "retriever_source_content": float(src.get("content_based", 0.0)),
                    "retriever_source_social": float(src.get("social", 0.0)),
                    "retriever_source_popularity": float(src.get("popularity", 0.0)),
                    "retriever_source_image": float(src.get("image_embedding", 0.0)),
                    "user_activity_bucket": self._user_bucket.get(u, 0.0),
                    "item_age_days": self._item_age_days.get(it, 365.0),
                }
                user_feats[it] = row
            out[u] = user_feats
        return out

    @staticmethod
    def child_retriever_from_multi(retriever: Recommender, name: str) -> Optional[Recommender]:
        """Return a named child inside a ``MultiRetriever``, if present."""
        if getattr(retriever, "name", "") != "multi_retriever":
            return None
        for child in getattr(retriever, "retrievers", []):
            if getattr(child, "name", "") == name:
                return child
        return None

    @staticmethod
    def content_retriever_from_multi(retriever: Recommender) -> Optional[Recommender]:
        return ExtendedRankerFeatures.child_retriever_from_multi(retriever, "content_based")

    @staticmethod
    def image_retriever_from_multi(retriever: Recommender) -> Optional[Recommender]:
        return ExtendedRankerFeatures.child_retriever_from_multi(retriever, "image_embedding")
