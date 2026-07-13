"""Learning-to-rank re-ranker  [Phase 2 — implemented].

Retrieval (two-tower / ALS / item-CF) returns a few hundred candidates optimized
for *recall*. This ranker re-orders them for *precision* using per-(user, item)
features: the retrieval score plus user/item aggregates.

Backends (tried in order):
  1. **LightGBM** ``lambdarank`` (preferred — true listwise LTR). On macOS this
     needs OpenMP: ``brew install libomp``.
  2. **sklearn HistGradientBoostingRegressor** (pointwise fallback — no extra
     system libs). Still a real re-ranker; just not listwise.

At serving time it only scores candidates handed to it — use
:class:`~recsys.models.two_stage.TwoStageRecommender` for the full pipeline.

References: Burges 2010 (LambdaMART); the RecSys "retrieve then rank" pattern.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import settings
from .base import Recommender
from .ranker_features import ExtendedRankerFeatures


class LightGBMRanker(Recommender):
    """Re-ranks a *candidate list* per user; does not retrieve from the full catalog.

    Name kept as LightGBMRanker for the roadmap; may use sklearn if LightGBM's
    native lib is unavailable.
    """

    name = "lgbm_ranker"

    BASE_FEATURE_NAMES = (
        "retrieval_score",
        "retrieval_rank",
        "user_n",
        "user_avg_rating",
        "item_n",
        "item_avg_rating",
        "rating_gap",
    )

    def __init__(
        self,
        num_leaves: int = 31,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        min_data_in_leaf: int = 20,
        seed: int | None = None,
        verbose: bool = False,
        prefer_lightgbm: bool = True,
    ):
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_data_in_leaf = min_data_in_leaf
        self.seed = settings.seed if seed is None else seed
        self.verbose = verbose
        self.prefer_lightgbm = prefer_lightgbm
        self._backend: str | None = None
        self._model = None
        self._use_social = False  # set at fit time if social_scores provided
        self._use_sasrec = False  # set at fit time if sasrec_scores provided
        self._use_hstu = False
        self._use_lightgcn = False
        self._use_extended = False
        self._user_stats: Dict[str, Dict[str, float]] = {}
        self._item_stats: Dict[str, Dict[str, float]] = {}
        self._global_avg = 3.0

    @property
    def feature_names(self) -> tuple:
        """Active feature set (adds social_score when social features are used)."""
        names = list(self.BASE_FEATURE_NAMES)
        if self._use_social:
            names.append("social_score")
        if self._use_sasrec:
            names.append("sasrec_score")
        if self._use_hstu:
            names.append("hstu_score")
        if self._use_lightgcn:
            names.append("lightgcn_score")
        if self._use_extended:
            names.extend(ExtendedRankerFeatures.FEATURE_NAMES)
        return tuple(names)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train: pd.DataFrame,
        candidates: Optional[Dict[str, List[str]]] = None,
        retrieval_scores: Optional[Dict[str, Dict[str, float]]] = None,
        labels: Optional[Dict[str, Dict[str, float]]] = None,
        social_scores: Optional[Dict[str, Dict[str, float]]] = None,
        sasrec_scores: Optional[Dict[str, Dict[str, float]]] = None,
        hstu_scores: Optional[Dict[str, Dict[str, float]]] = None,
        lightgcn_scores: Optional[Dict[str, Dict[str, float]]] = None,
        extended_features: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> "LightGBMRanker":
        if candidates is None or labels is None:
            raise ValueError(
                "LightGBMRanker.fit requires candidates= and labels=. "
                "Use TwoStageRecommender to build them from a retriever."
            )

        self._build_stats(train)
        retrieval_scores = retrieval_scores or {}
        self._use_social = social_scores is not None
        self._use_sasrec = sasrec_scores is not None
        self._use_hstu = hstu_scores is not None
        self._use_lightgcn = lightgcn_scores is not None
        self._use_extended = extended_features is not None
        social_scores = social_scores or {}
        sasrec_scores = sasrec_scores or {}
        hstu_scores = hstu_scores or {}
        lightgcn_scores = lightgcn_scores or {}
        extended_features = extended_features or {}

        X_rows: List[List[float]] = []
        y_rows: List[float] = []
        groups: List[int] = []

        for user_id, items in candidates.items():
            if not items:
                continue
            rel = labels.get(user_id, {})
            scores = retrieval_scores.get(user_id, {})
            soc = social_scores.get(user_id, {})
            sas = sasrec_scores.get(user_id, {})
            hst = hstu_scores.get(user_id, {})
            lgc = lightgcn_scores.get(user_id, {})
            ext = extended_features.get(user_id, {})
            if not any(rel.get(it, 0) > 0 for it in items):
                continue
            for rank, item_id in enumerate(items):
                X_rows.append(
                    self._feature_row(
                        user_id, item_id, scores.get(item_id, 0.0), rank,
                        social_score=soc.get(item_id, 0.0),
                        sasrec_score=sas.get(item_id, 0.0),
                        hstu_score=hst.get(item_id, 0.0),
                        lightgcn_score=lgc.get(item_id, 0.0),
                        extended=ext.get(str(item_id), {}),
                    )
                )
                y_rows.append(float(rel.get(item_id, 0.0)))
            groups.append(len(items))

        if not X_rows:
            raise ValueError(
                "No ranking training rows built. Check that candidates overlap "
                "with labeled positives for at least some users."
            )

        X = np.asarray(X_rows, dtype=np.float32)
        y = np.asarray(y_rows, dtype=np.float32)

        if self.prefer_lightgbm and self._try_fit_lightgbm(X, y, groups):
            return self
        self._fit_sklearn(X, y)
        return self

    def feature_importance(self) -> Dict[str, float]:
        """Feature importances keyed by name (backend-appropriate)."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        names = self.feature_names
        if self._backend == "lightgbm":
            imp = self._model.feature_importance(importance_type="gain")
        else:  # sklearn permutation-free proxy not available; use split-based
            imp = getattr(self._model, "feature_importances_", None)
            if imp is None:
                return {n: float("nan") for n in names}
        return {n: float(v) for n, v in zip(names, imp)}

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError(
            "LightGBMRanker only re-ranks candidates. Call "
            "rerank(...) or use TwoStageRecommender."
        )

    # ---------------------------------------------------------------- rerank
    def rerank(
        self,
        candidates: Dict[str, List[str]],
        retrieval_scores: Optional[Dict[str, Dict[str, float]]] = None,
        k: int = 10,
        social_scores: Optional[Dict[str, Dict[str, float]]] = None,
        sasrec_scores: Optional[Dict[str, Dict[str, float]]] = None,
        hstu_scores: Optional[Dict[str, Dict[str, float]]] = None,
        lightgcn_scores: Optional[Dict[str, Dict[str, float]]] = None,
        extended_features: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> Dict[str, List[str]]:
        if self._model is None:
            raise RuntimeError("Call fit() before rerank().")

        retrieval_scores = retrieval_scores or {}
        social_scores = social_scores or {}
        sasrec_scores = sasrec_scores or {}
        hstu_scores = hstu_scores or {}
        lightgcn_scores = lightgcn_scores or {}
        extended_features = extended_features or {}
        out: Dict[str, List[str]] = {}
        for user_id, items in candidates.items():
            if not items:
                out[user_id] = []
                continue
            scores = retrieval_scores.get(user_id, {})
            soc = social_scores.get(user_id, {})
            sas = sasrec_scores.get(user_id, {})
            hst = hstu_scores.get(user_id, {})
            lgc = lightgcn_scores.get(user_id, {})
            ext = extended_features.get(user_id, {})
            X = np.asarray(
                [
                    self._feature_row(
                        user_id, item_id, scores.get(item_id, 0.0), rank,
                        social_score=soc.get(item_id, 0.0),
                        sasrec_score=sas.get(item_id, 0.0),
                        hstu_score=hst.get(item_id, 0.0),
                        lightgcn_score=lgc.get(item_id, 0.0),
                        extended=ext.get(str(item_id), {}),
                    )
                    for rank, item_id in enumerate(items)
                ],
                dtype=np.float32,
            )
            if self._backend == "lightgbm":
                pred = self._model.predict(X)
            else:
                pred = self._model.predict(X)
            order = np.argsort(-pred)[:k]
            out[user_id] = [items[i] for i in order]
        return out

    # ------------------------------------------------------------- backends
    def _try_fit_lightgbm(self, X: np.ndarray, y: np.ndarray, groups: List[int]) -> bool:
        try:
            import lightgbm as lgb

            # Force-load the native lib here so we catch macOS libomp errors.
            _ = lgb.__version__
            dataset = lgb.Dataset(
                X, label=y, group=groups, feature_name=list(self.feature_names)
            )
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_at": [settings.top_k],
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "min_data_in_leaf": self.min_data_in_leaf,
                "feature_fraction": 0.9,
                "verbosity": 1 if self.verbose else -1,
                "seed": self.seed,
            }
            self._model = lgb.train(params, dataset, num_boost_round=self.n_estimators)
            self._backend = "lightgbm"
            if self.verbose:
                print(
                    f"  ranker backend=lightgbm (lambdarank) "
                    f"rows={len(y):,} groups={len(groups):,}"
                )
            return True
        except Exception as exc:
            if self.verbose:
                print(f"  LightGBM unavailable ({exc.__class__.__name__}: {exc})")
                print("  falling back to sklearn HistGradientBoosting (pointwise)")
            return False

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._model = HistGradientBoostingRegressor(
            max_depth=6,
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_data_in_leaf,
            random_state=self.seed,
        )
        self._model.fit(X, y)
        self._backend = "sklearn"
        if self.verbose:
            print(f"  ranker backend=sklearn (pointwise) rows={len(y):,}")

    # ------------------------------------------------------------- features
    def _build_stats(self, train: pd.DataFrame) -> None:
        u, i, r = settings.user_col, settings.item_col, settings.rating_col
        self._global_avg = float(train[r].mean()) if len(train) else 3.0

        user_g = train.groupby(u)[r]
        self._user_stats = {
            uid: {"n": float(n), "avg": float(avg)}
            for uid, n, avg in zip(
                user_g.size().index, user_g.size().values, user_g.mean().values
            )
        }
        item_g = train.groupby(i)[r]
        self._item_stats = {
            iid: {"n": float(n), "avg": float(avg)}
            for iid, n, avg in zip(
                item_g.size().index, item_g.size().values, item_g.mean().values
            )
        }

    def _feature_row(
        self,
        user_id: str,
        item_id: str,
        retrieval_score: float,
        rank: int,
        social_score: float = 0.0,
        sasrec_score: float = 0.0,
        hstu_score: float = 0.0,
        lightgcn_score: float = 0.0,
        extended: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        us = self._user_stats.get(user_id, {"n": 0.0, "avg": self._global_avg})
        its = self._item_stats.get(item_id, {"n": 0.0, "avg": self._global_avg})
        row = [
            float(retrieval_score),
            float(rank),
            us["n"],
            us["avg"],
            its["n"],
            its["avg"],
            us["avg"] - its["avg"],
        ]
        if self._use_social:
            row.append(float(social_score))
        if self._use_sasrec:
            row.append(float(sasrec_score))
        if self._use_hstu:
            row.append(float(hstu_score))
        if self._use_lightgcn:
            row.append(float(lightgcn_score))
        if self._use_extended:
            ext = extended or {}
            for name in ExtendedRankerFeatures.FEATURE_NAMES:
                row.append(float(ext.get(name, 0.0)))
        return row
