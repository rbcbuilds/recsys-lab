"""Social recommendation  [SCAFFOLD — Phase 3]  ← your priority experiment.

Uses the **real** Yelp friend graph (user-user edges) to improve recs. The core
hypothesis: your friends' tastes are informative about yours, especially when
your own history is thin (cold-ish users).

Why the real graph matters (and why we don't simulate it):
  If you *built* friendships from taste similarity, "social helps" would be true
  by construction — a data leak. Yelp's friendships are formed independently of
  the ratings you're predicting, so measuring lift is honest. This is exactly
  why Path A (Yelp) was chosen over bolting a fake social graph onto MovieLens.

Approaches to try, simplest first:
  1. Social-neighbor CF (baseline social model):
       score(u, i) = sum over friends f of  sim(u, f) * liked(f, i)
     A trust-weighted vote from friends. Easy, interpretable, a real baseline.
  2. Social regularization on matrix factorization:
       add a loss term pulling friends' user vectors together (SoReg / SocialMF).
  3. GraphRec / social-augmented LightGCN:
       propagate over BOTH the interaction graph and the social graph.

The honest evaluation, and the real lesson:
  * Compare each social model against ALS (no social) on the SAME temporal split.
  * Report overall Recall@K/NDCG@K, AND slice by user activity — social usually
    helps sparse users most. If social can't beat a tuned ALS, that's a finding,
    not a failure.

The social graph is available as ``Dataset.social`` (columns: user_id, friend_id).

References: Ma et al. 2011 (SoReg); Jamali & Ester 2010 (SocialMF); Fan et al. 2019 (GraphRec).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Recommender


class SocialRecommender(Recommender):
    name = "social"

    def __init__(self, social: pd.DataFrame | None = None, trust_weight: float = 1.0):
        """``social`` is the friend-edge DataFrame from ``Dataset.social``."""
        self.social = social
        self.trust_weight = trust_weight

    def fit(self, train: pd.DataFrame) -> "SocialRecommender":
        raise NotImplementedError(
            "Social recsys is YOUR priority Phase-3 exercise. Start with "
            "social-neighbor CF using Dataset.social, then compare against ALS on "
            "the same temporal split, sliced by user activity. See the docstring."
        )

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError
