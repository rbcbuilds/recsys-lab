"""Extended ranker features for cross-regime routing  [SCAFFOLD — Tier 1].

The unified two-stage already fuses complementary retrievers. The next step is
teaching the ranker *which signal to trust* per (user, item) without adding more
retrieval arms.

Candidate features to add to ``LightGBMRanker._feature_row``:

| Feature | Source | Regime it helps |
|---|---|---|
| ``content_score`` | ``ContentBasedRecommender.score_candidates`` | cold-item |
| ``sasrec_score`` | already wired via ``use_sasrec`` | warm |
| ``social_score`` | already wired via ``use_social`` | cold-user |
| ``retriever_source`` | one-hot / count: which MultiRetriever arm surfaced item | routing |
| ``user_activity_bucket`` | log(train interactions) binned: tail vs dense | all |
| ``item_age_days`` | days since first interaction (global-time coldness) | cold-item |

Implementation sketch:
  1. Extend ``TwoStageRecommender._score_candidates`` to collect optional scores.
  2. Pass ``content_scores``, ``source_flags`` into ``ranker.fit`` / ``rerank``.
  3. Compare unified with/without on warm / cold-item / cold-user slices.

Interview point: feature ablation table beats adding a 6th retriever.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .base import Recommender


class ExtendedRankerFeatures:
    """Collector for optional ranker feature dicts (not a standalone recommender)."""

    FEATURE_NAMES = (
        "content_score",
        "retriever_source_two_tower",
        "retriever_source_content",
        "retriever_source_social",
        "retriever_source_popularity",
        "user_activity_bucket",
        "item_age_days",
    )

    def build(
        self,
        train: pd.DataFrame,
        candidates: Dict[str, List[str]],
        retrieval_sources: Optional[Dict[str, Dict[str, List[str]]]] = None,
        content_model: Optional[Recommender] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Per-user, per-item feature values for ranker training/inference.

        Returns ``{user_id: {item_id: {feature_name: value}}}``.
        """
        raise NotImplementedError(
            "Wire content_score and MultiRetriever source flags into the "
            "LightGBM ranker. See this module's docstring."
        )
