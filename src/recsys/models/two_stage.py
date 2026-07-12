"""Two-stage recommender: retrieval → LightGBM re-ranking  [Phase 2].

Architecture::

    request → retriever (e.g. two-tower) → top-N candidates
           → LightGBM ranker → top-K

Keeps a standalone retrieval model for comparison: if two-stage does not beat
retrieval-alone on NDCG@K, that is a useful finding (the ranker features may
not yet be rich enough).

Training protocol (to avoid label leakage):
  1. Temporally split ``train`` into ``ret_train`` + ``rank_labels``.
  2. Fit the retriever on ``ret_train`` only.
  3. Retrieve candidates (excluding ``ret_train`` history); label with
     ``rank_labels`` positives.
  4. Train the LightGBM ranker on those (candidate, label) groups.
  5. Re-fit the retriever on the *full* train for strong inference-time retrieval.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..config import settings
from ..eval.split import temporal_split
from .base import Recommender
from .ranker import LightGBMRanker
from .social import SocialRecommender


class TwoStageRecommender(Recommender):
    name = "two_stage"

    def __init__(
        self,
        retriever: Recommender,
        candidate_n: int = 100,
        rank_label_fraction: float = 0.2,
        ranker_kwargs: Optional[dict] = None,
        use_social: bool = False,
        social: Optional[pd.DataFrame] = None,
        social_kwargs: Optional[dict] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        retriever:
            Any model with ``fit`` / ``recommend`` (typically TwoTowerRecommender).
            Stored and reused; do not share one instance across parallel demos
            that need independent fits.
        candidate_n:
            How many items the retriever returns for the ranker to re-order.
        rank_label_fraction:
            Fraction of each user's most recent *training* interactions held out
            as ranker labels (inner temporal split).
        use_social:
            If True, add a ``social_score`` feature to the ranker (requires
            ``social=``). This is the with/without-social toggle: build two
            TwoStageRecommenders differing only in this flag to measure lift.
        social:
            Friend-edge DataFrame (Dataset.social). Required when use_social.
        """
        self.retriever = retriever
        self.candidate_n = candidate_n
        self.rank_label_fraction = rank_label_fraction
        self.ranker_kwargs = ranker_kwargs or {}
        self.use_social = use_social
        self.social = social
        self.social_kwargs = social_kwargs or {}
        self.verbose = verbose
        self.ranker = LightGBMRanker(verbose=verbose, **self.ranker_kwargs)
        self._social_model: Optional[SocialRecommender] = None
        self._train: Optional[pd.DataFrame] = None

        if self.use_social and self.social is None:
            raise ValueError("use_social=True requires social=Dataset.social.")

    def fit(self, train: pd.DataFrame) -> "TwoStageRecommender":
        self._train = train
        # Inner temporal split: past → retriever, recent → ranker labels.
        ret_train, rank_label_positives = temporal_split(
            train,
            test_fraction=self.rank_label_fraction,
            positive_threshold=settings.positive_threshold,
            min_train=1,
        )
        if self.verbose:
            print(
                f"  two-stage inner split: ret_train={len(ret_train):,}, "
                f"rank_label_users={len(rank_label_positives):,}"
            )

        if self.verbose:
            print(f"  fitting retriever ({self.retriever.name}) on ret_train...")
        self.retriever.fit(ret_train)

        # Social model (optional) fit on the same ret_train for consistent features.
        train_social_scores = None
        if self.use_social:
            self._social_model = SocialRecommender(social=self.social, **self.social_kwargs)
            self._social_model.fit(ret_train)

        label_users = list(rank_label_positives.keys())
        candidates = self.retriever.recommend(
            label_users, k=self.candidate_n, exclude_seen=True
        )
        retrieval_scores = self._score_candidates(label_users, candidates)
        if self.use_social:
            train_social_scores = self._social_model.score_candidates(candidates)
        labels = {
            u: {item: 1.0 for item in items}
            for u, items in rank_label_positives.items()
        }

        if self.verbose:
            print(f"  training ranker (use_social={self.use_social})...")
        # Features from full train aggregates (user/item stats); labels from holdout.
        self.ranker.fit(
            train,
            candidates=candidates,
            retrieval_scores=retrieval_scores,
            labels=labels,
            social_scores=train_social_scores,
        )

        # Re-fit retriever (and social model) on all train for inference-time recall.
        if self.verbose:
            print(f"  re-fitting retriever ({self.retriever.name}) on full train...")
        self.retriever.fit(train)
        if self.use_social:
            self._social_model.fit(train)
        return self

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        if self._train is None:
            raise RuntimeError("Call fit() before recommend().")

        # Retrieve a wider pool, then re-rank down to k.
        n = max(self.candidate_n, k)
        candidates = self.retriever.recommend(users, k=n, exclude_seen=exclude_seen)
        retrieval_scores = self._score_candidates(users, candidates)
        social_scores = (
            self._social_model.score_candidates(candidates) if self.use_social else None
        )
        return self.ranker.rerank(
            candidates, retrieval_scores, k=k, social_scores=social_scores
        )

    # ------------------------------------------------------------- helpers
    def _score_candidates(
        self, users: List[str], candidates: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Best-effort retrieval scores for the ranker feature.

        Prefer exact scores from two-tower / ALS when available; otherwise fall
        back to reciprocal rank from the candidate list order.
        """
        # Two-tower: exact dot product
        if hasattr(self.retriever, "_item_matrix") and hasattr(
            self.retriever, "user_to_idx"
        ):
            return self._scores_two_tower(users, candidates)

        # ALS: user/item factors
        if hasattr(self.retriever, "_model") and hasattr(self.retriever, "ui"):
            model = getattr(self.retriever, "_model", None)
            if model is not None and hasattr(model, "user_factors"):
                return self._scores_als(users, candidates)

        # Fallback: reciprocal rank from retrieval order
        out: Dict[str, Dict[str, float]] = {}
        for user_id in users:
            items = candidates.get(user_id, [])
            out[user_id] = {item: 1.0 / (rank + 1) for rank, item in enumerate(items)}
        return out

    def _scores_two_tower(
        self, users: List[str], candidates: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        import torch

        ret = self.retriever
        device = next(ret._model.parameters()).device
        out: Dict[str, Dict[str, float]] = {}
        known = [u for u in users if u in ret.user_to_idx]
        if not known:
            return {u: {} for u in users}

        idxs = torch.as_tensor(
            [ret.user_to_idx[u] for u in known], dtype=torch.long, device=device
        )
        with torch.no_grad():
            user_vecs = ret._model.user(idxs).cpu().numpy()

        item_mat = ret._item_matrix
        for row, user_id in enumerate(known):
            items = candidates.get(user_id, [])
            scores: Dict[str, float] = {}
            for item_id in items:
                j = ret.item_to_idx.get(item_id)
                if j is None:
                    continue
                scores[item_id] = float(user_vecs[row] @ item_mat[j])
            out[user_id] = scores
        for u in users:
            out.setdefault(u, {})
        return out

    def _scores_als(
        self, users: List[str], candidates: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        ret = self.retriever
        uf = ret._model.user_factors
        itf = ret._model.item_factors
        out: Dict[str, Dict[str, float]] = {}
        for user_id in users:
            uidx = ret.user_to_idx.get(user_id)
            items = candidates.get(user_id, [])
            if uidx is None:
                out[user_id] = {}
                continue
            uvec = uf[uidx]
            scores: Dict[str, float] = {}
            for item_id in items:
                j = ret.item_to_idx.get(item_id)
                if j is None:
                    continue
                scores[item_id] = float(uvec @ itf[j])
            out[user_id] = scores
        return out
