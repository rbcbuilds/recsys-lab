"""LLM-style re-ranker + two-stage wrapper  [generative / language scoring].

**LLMReranker** scores (user history, candidate item) pairs with a language
model-style relevance function — preferred: a sentence-transformers *cross-encoder*
(pretrained MS MARCO). Fallback: bi-encoder cosine over a short natural-language
prompt built from item names/categories.

This is still **discriminative at the pipeline level** (score candidates, then
sort), but the scoring function is language-native — the pattern used when you
want LLM judgment without generating item tokens from scratch.

**LLMTwoStageRecommender** = retriever → ``LLMReranker``. No gradient-boosted
ranker training; the cross-encoder is zero-shot (or prompt-scored via embeddings).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import settings
from ..eval.split import temporal_split
from .base import Recommender
from .text_embeddings import _item_text_series


def _history_text(
    user_id: str,
    train: pd.DataFrame,
    item_text: Dict[str, str],
    max_items: int = 8,
) -> str:
    """Compact natural-language summary of a user's recent positives."""
    u, i, r, t = (
        settings.user_col,
        settings.item_col,
        settings.rating_col,
        settings.time_col,
    )
    grp = train[train[u] == user_id]
    if r in grp.columns:
        grp = grp[grp[r] >= settings.positive_threshold]
    if t in grp.columns:
        grp = grp.sort_values(t)
    names = []
    for iid in grp[i].astype(str).tolist()[-max_items:]:
        text = item_text.get(iid, iid)
        name = text.split(".")[0].strip() if text else iid
        if name and name not in names:
            names.append(name)
    if not names:
        return "User with no prior preferences."
    return "User recently enjoyed: " + ", ".join(names) + "."


def _candidate_text(item_id: str, item_text: Dict[str, str]) -> str:
    text = item_text.get(item_id, item_id)
    return f"Recommend: {text}"


class LLMReranker(Recommender):
    """Re-rank a candidate list using cross-encoder or prompt embedding similarity."""

    name = "llm_ranker"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        bi_encoder_name: str = "all-MiniLM-L6-v2",
        max_history_items: int = 8,
        batch_size: int = 64,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.bi_encoder_name = bi_encoder_name
        self.max_history_items = max_history_items
        self.batch_size = batch_size
        self.verbose = verbose
        self.backend: str = ""
        self._cross_encoder = None
        self._bi_encoder = None
        self._train: Optional[pd.DataFrame] = None
        self._item_text: Dict[str, str] = {}

    def fit(
        self,
        train: pd.DataFrame,
        items: Optional[pd.DataFrame] = None,
        **_,
    ) -> "LLMReranker":
        self._train = train
        if items is not None:
            i = settings.item_col
            texts = _item_text_series(items)
            self._item_text = {
                str(items.iloc[n][i]): str(texts.iloc[n])
                for n in range(len(items))
            }
        self._load_backend()
        return self

    def _load_backend(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
            self.backend = f"cross-encoder:{self.model_name}"
            if self.verbose:
                print(f"  llm ranker backend={self.backend}")
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer

                self._bi_encoder = SentenceTransformer(self.bi_encoder_name)
                self.backend = f"bi-encoder:{self.bi_encoder_name}"
                if self.verbose:
                    print(f"  llm ranker backend={self.backend} (cross-encoder unavailable)")
            except ImportError:
                self.backend = "tfidf_prompt_fallback"
                if self.verbose:
                    print("  llm ranker backend=tfidf_prompt_fallback")

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError(
            "LLMReranker only re-ranks candidates. Use LLMTwoStageRecommender "
            "or call rerank(...) directly."
        )

    def rerank(
        self,
        candidates: Dict[str, List[str]],
        k: int = 10,
    ) -> Dict[str, List[str]]:
        if self._train is None:
            raise RuntimeError("Call fit() before rerank().")

        out: Dict[str, List[str]] = {}
        for user_id, items in candidates.items():
            if not items:
                out[user_id] = []
                continue
            hist = _history_text(
                user_id, self._train, self._item_text, self.max_history_items
            )
            scores = self._score_pairs(hist, items)
            order = sorted(range(len(items)), key=lambda j: scores[j], reverse=True)
            out[user_id] = [items[j] for j in order[:k]]
        return out

    def _score_pairs(self, history: str, items: List[str]) -> List[float]:
        if self._cross_encoder is not None:
            pairs = [(history, _candidate_text(it, self._item_text)) for it in items]
            raw = self._cross_encoder.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )
            return [float(x) for x in raw]

        if self._bi_encoder is not None:
            cand_texts = [_candidate_text(it, self._item_text) for it in items]
            embs = self._bi_encoder.encode(
                [history] + cand_texts, show_progress_bar=False, convert_to_numpy=True
            )
            h = embs[0]
            h = h / (np.linalg.norm(h) or 1.0)
            c = embs[1:]
            c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
            return (c @ h).tolist()

        # Minimal fallback: token overlap between history and candidate text.
        hist_tokens = set(history.lower().split())
        scores = []
        for it in items:
            ct = _candidate_text(it, self._item_text).lower().split()
            overlap = len(hist_tokens & set(ct))
            scores.append(float(overlap))
        return scores


class LLMTwoStageRecommender(Recommender):
    """Retrieve candidates, then re-rank with ``LLMReranker`` (language scoring)."""

    name = "two_stage_llm"

    def __init__(
        self,
        retriever: Recommender,
        items: pd.DataFrame,
        candidate_n: int = 100,
        rank_label_fraction: float = 0.2,
        llm_kwargs: Optional[dict] = None,
        verbose: bool = False,
    ):
        self.retriever = retriever
        self.items = items
        self.candidate_n = candidate_n
        self.rank_label_fraction = rank_label_fraction
        self.llm_kwargs = llm_kwargs or {}
        self.verbose = verbose
        self.ranker = LLMReranker(verbose=verbose, **self.llm_kwargs)
        self._train: Optional[pd.DataFrame] = None

    def fit(self, train: pd.DataFrame) -> "LLMTwoStageRecommender":
        self._train = train
        ret_train, _ = temporal_split(
            train,
            test_fraction=self.rank_label_fraction,
            positive_threshold=settings.positive_threshold,
            min_train=1,
        )
        if self.verbose:
            print(f"  llm two-stage: fitting retriever on inner train ({len(ret_train):,} rows)")
        self.retriever.fit(ret_train)
        if self.verbose:
            print("  llm two-stage: re-fitting retriever on full train")
        self.retriever.fit(train)
        self.ranker.fit(train, items=self.items)
        return self

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        if self._train is None:
            raise RuntimeError("Call fit() before recommend().")
        n = max(self.candidate_n, k)
        candidates = self.retriever.recommend(users, k=n, exclude_seen=exclude_seen)
        return self.ranker.rerank(candidates, k=k)
