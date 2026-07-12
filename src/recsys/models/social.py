"""Social recommendation  [Phase 3 — implemented]  ← the priority experiment.

Uses the friend graph (user-user edges) to recommend. Core hypothesis: your
friends' tastes are informative about yours — most useful when your own history
is thin (cold-ish users).

Model 1 — Social-neighbor CF (implemented here):

    score(u, i) = (1 - alpha) * social(u, i) + alpha * popularity(i)

    social(u, i) = sum over friends f of  trust(u, f) * liked(f, i)

where ``liked(f, i)`` is f's (optionally rating-weighted) interaction with i,
``trust(u, f)`` defaults to 1 (uniform) and can later be Jaccard/overlap based,
and the popularity term is a small back-off so users with few/no friends (or
friends with no signal) still get sensible recommendations.

Honest-evaluation notes (why the dataset matters):
  * On the SYNTHETIC dataset the friend graph is partly built from latent taste,
    so a "social helps" result here only validates the *mechanics*, not the
    scientific claim. Use the real Yelp friend graph for the honest lift test.
  * Compare against ALS / item-CF on the SAME temporal split, and slice by user
    activity (see ``recsys.eval.evaluate_by_activity``): social usually helps
    sparse users most.

References: Ma et al. 2011 (SoReg); Jamali & Ester 2010 (SocialMF); Fan et al. 2019 (GraphRec).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import settings
from .base import IndexedRecommender


class SocialRecommender(IndexedRecommender):
    name = "social"

    def __init__(
        self,
        social: Optional[pd.DataFrame] = None,
        alpha: float = 0.15,
        use_ratings: bool = True,
        trust: str = "uniform",
    ):
        """
        Parameters
        ----------
        social:
            Friend-edge DataFrame with columns [user_id, friend_id] (undirected;
            each edge may appear once). Typically ``Dataset.social``.
        alpha:
            Weight on the popularity back-off term in [0, 1]. 0 = pure social.
        use_ratings:
            Weight a friend's interaction by their rating (else binary).
        trust:
            ``"uniform"`` (trust=1 per friend) or ``"jaccard"`` (overlap of
            interacted items between u and f). Jaccard needs u to have history.
        """
        self.social = social
        self.alpha = alpha
        self.use_ratings = use_ratings
        self.trust = trust
        self._friends: Dict[str, List[str]] = {}
        self._item_pop = None  # (n_items,) popularity back-off scores

    def fit(self, train: pd.DataFrame) -> "SocialRecommender":
        if self.social is None:
            raise ValueError(
                "SocialRecommender needs a social graph. Pass social=Dataset.social."
            )
        self._build_index(train)
        u, i, r = settings.user_col, settings.item_col, settings.rating_col

        # Per-user item->weight map (a friend's contribution to a candidate item).
        weight = train[r].astype(float) if self.use_ratings else pd.Series(
            np.ones(len(train)), index=train.index
        )
        tmp = train.assign(_w=weight.values)
        self._user_item_weight: Dict[str, Dict[str, float]] = {
            uid: dict(zip(grp[i], grp["_w"]))
            for uid, grp in tmp.groupby(u)
        }

        # Undirected adjacency limited to users seen in training.
        self._build_friends()

        # Popularity back-off (interaction count per item), aligned to item index.
        pop = train.groupby(i).size()
        self._item_pop = np.zeros(len(self.item_ids), dtype=float)
        for item_id, c in pop.items():
            idx = self.item_to_idx.get(item_id)
            if idx is not None:
                self._item_pop[idx] = c
        if self._item_pop.max() > 0:
            self._item_pop = self._item_pop / self._item_pop.max()

        return self

    def _build_friends(self) -> None:
        u_col, f_col = settings.user_col, "friend_id"
        known = set(self.user_to_idx)
        friends: Dict[str, set] = {}
        for a, b in zip(self.social[u_col], self.social[f_col]):
            if a in known and b in known and a != b:
                friends.setdefault(a, set()).add(b)
                friends.setdefault(b, set()).add(a)  # ensure undirected
        self._friends = {u: list(fs) for u, fs in friends.items()}

    def _trust(self, user_id: str, friend_id: str) -> float:
        if self.trust == "uniform":
            return 1.0
        # Jaccard similarity of interacted item sets.
        ui = set(self._user_item_weight.get(user_id, {}))
        fi = set(self._user_item_weight.get(friend_id, {}))
        if not ui or not fi:
            return 0.0
        inter = len(ui & fi)
        union = len(ui | fi)
        return inter / union if union else 0.0

    def _score_vector(self, user_id: str, blend_popularity: bool = True) -> np.ndarray:
        """Full (n_items,) social score vector for one user.

        When ``blend_popularity`` is False, returns the pure normalized social
        term (useful as a standalone *feature* for a downstream ranker, where a
        separate popularity feature already exists).
        """
        scores = np.zeros(len(self.item_ids), dtype=float)

        # Social term: trust-weighted sum of friends' interactions.
        social_mass = 0.0
        for f in self._friends.get(user_id, []):
            t = self._trust(user_id, f)
            if t <= 0:
                continue
            for item_id, w in self._user_item_weight.get(f, {}).items():
                idx = self.item_to_idx.get(item_id)
                if idx is not None:
                    scores[idx] += t * w
                    social_mass += t * w

        # Normalize social term so the alpha blend is comparable across users.
        if social_mass > 0:
            scores = scores / social_mass

        if not blend_popularity:
            return scores
        # Popularity back-off (helps cold / friendless users).
        return (1.0 - self.alpha) * scores + self.alpha * self._item_pop

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for user_id in users:
            scores = self._score_vector(user_id, blend_popularity=True)
            out[user_id] = self._top_k_from_scores(user_id, scores, k, exclude_seen)
        return out

    def score_candidates(
        self, candidates: Dict[str, List[str]], blend_popularity: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Per-(user, item) social scores for a candidate set.

        Used to inject a ``social_score`` feature into a downstream ranker.
        Defaults to the *pure* social term (no popularity blend) since the ranker
        already has its own popularity features.
        """
        out: Dict[str, Dict[str, float]] = {}
        for user_id, items in candidates.items():
            if user_id not in self.user_to_idx and user_id not in self._friends:
                out[user_id] = {}
                continue
            vec = self._score_vector(user_id, blend_popularity=blend_popularity)
            scores: Dict[str, float] = {}
            for item_id in items:
                idx = self.item_to_idx.get(item_id)
                if idx is not None:
                    scores[item_id] = float(vec[idx])
            out[user_id] = scores
        return out
