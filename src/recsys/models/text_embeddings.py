"""Text / LLM item embeddings  [SCAFFOLD — Phase 3].

Encode each item's text (Yelp: name + categories + review text) into a dense
vector with a sentence embedding model, then recommend by semantic similarity.
This is the honest "does an LLM/text signal beat a tuned baseline?" experiment.

Where it helps most:
  * cold-start items (no interactions yet, but they have text)
  * semantic "more like this" beyond co-occurrence

Suggested implementation (see requirements-extra.txt):
  1. text = items['name'] + ' ' + items['categories'] (+ concatenated reviews)
  2. emb = SentenceTransformer('all-MiniLM-L6-v2').encode(text)  -> (n_items, d)
  3. user vector = mean of embeddings of items they liked
  4. score = cosine(user_vec, item_emb); top-K
  5. Compare Recall@K / NDCG@K against ALS. Also measure cold-start Recall.

Bonus: feed these embeddings into the two-tower item tower instead of using
them standalone — that's how text usually enters a real system (as features,
not a replacement).

References: Reimers & Gurevych 2019 (Sentence-BERT).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Recommender


class TextEmbeddingRecommender(Recommender):
    name = "text_embeddings"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def fit(self, train: pd.DataFrame, items: pd.DataFrame | None = None) -> "TextEmbeddingRecommender":
        raise NotImplementedError(
            "Text/LLM embeddings is a Phase-3 exercise. Encode items['text'] with "
            "sentence-transformers, build user vectors from liked items, recommend "
            "by cosine similarity, and compare against ALS. See the docstring."
        )

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError
