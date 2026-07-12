"""Text / LLM item embeddings + content-aware two-tower  [Phase 3].

Two complementary ways to use item text (name + categories + description):

1. ``TextEmbeddingRecommender`` — content-based CF.
   Embed every item from its text. User vector = mean of liked-item embeddings.
   Score = cosine. Works for **cold-start items** (text but no interactions)
   and **cold-ish users** (few interactions but those items have text).

2. ``ContentTwoTowerRecommender`` — production pattern.
   Same in-batch-negative two-tower training as ``TwoTowerRecommender``, but the
   item tower is ``MLP(id_emb ⊕ projected_text)``. Text enters retrieval as a
   feature, not a replacement for collaborative signal.

Encoder backends:
  * Preferred: ``sentence-transformers`` (``all-MiniLM-L6-v2``).
  * Fallback: TF-IDF → TruncatedSVD (no extra install; weaker but runnable).

Cold-start evaluation helper: ``cold_item_ids()`` returns items present in the
items table (with text) that never appear in training interactions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import settings
from .base import IndexedRecommender
from .two_tower import TwoTowerRecommender


def _item_text_series(items: pd.DataFrame) -> pd.Series:
    """Build a text string per item from available columns."""
    i = settings.item_col
    if "text" in items.columns:
        text = items["text"].fillna("").astype(str)
    else:
        parts = []
        for col in ("name", "categories"):
            if col in items.columns:
                parts.append(items[col].fillna("").astype(str))
        text = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=". ")
        if not parts:
            text = items[i].astype(str)
    return text


def encode_item_text(
    items: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    svd_dim: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, List[str], str]:
    """Encode items → (n_items, d) L2-normalized matrix.

    Returns ``(embeddings, item_ids, backend_name)``.
    """
    item_ids = items[settings.item_col].astype(str).tolist()
    texts = _item_text_series(items).tolist()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        emb = emb.astype(np.float32)
        backend = f"sentence-transformers:{model_name}"
    except ImportError:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(texts)
        d = min(svd_dim, max(2, X.shape[1] - 1), X.shape[0] - 1)
        svd = TruncatedSVD(n_components=d, random_state=seed)
        emb = svd.fit_transform(X).astype(np.float32)
        backend = f"tfidf+svd(d={d})"

    # L2 normalize for cosine = dot product.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emb = emb / norms
    return emb, item_ids, backend


class TextEmbeddingRecommender(IndexedRecommender):
    """Content-based recommender from item text embeddings."""

    name = "text_embeddings"

    def __init__(
        self,
        items: Optional[pd.DataFrame] = None,
        model_name: str = "all-MiniLM-L6-v2",
        svd_dim: int = 64,
        positive_threshold: float | None = None,
        seed: int | None = None,
    ):
        self.items = items
        self.model_name = model_name
        self.svd_dim = svd_dim
        self.positive_threshold = (
            settings.positive_threshold if positive_threshold is None else positive_threshold
        )
        self.seed = settings.seed if seed is None else seed
        self._item_emb: Optional[np.ndarray] = None
        self._user_emb: Dict[str, np.ndarray] = {}
        self.backend: str = ""
        self._all_item_ids: List[str] = []

    def fit(
        self, train: pd.DataFrame, items: pd.DataFrame | None = None
    ) -> "TextEmbeddingRecommender":
        items = items if items is not None else self.items
        if items is None:
            raise ValueError(
                "TextEmbeddingRecommender needs items=Dataset.items "
                "(must include a text/name/categories column)."
            )
        self.items = items
        self._build_index(train)

        emb, item_ids, backend = encode_item_text(
            items, model_name=self.model_name, svd_dim=self.svd_dim, seed=self.seed
        )
        self.backend = backend
        self._all_item_ids = item_ids
        self._item_id_to_row = {iid: n for n, iid in enumerate(item_ids)}
        self._item_emb = emb
        self._train_item_ids = set(train[settings.item_col].unique())

        # Align IndexedRecommender item space to *all* items with text (incl. cold).
        # Collaborative models only index train items; content can score cold ones.
        self.item_ids = item_ids
        self.item_to_idx = dict(self._item_id_to_row)
        self.idx_to_item = np.array(item_ids)

        u, i, r = settings.user_col, settings.item_col, settings.rating_col
        pos = train[train[r] >= self.positive_threshold] if r in train.columns else train
        self._user_emb = {}
        for uid, grp in pos.groupby(u):
            rows = [
                self._item_id_to_row[iid]
                for iid in grp[i].tolist()
                if iid in self._item_id_to_row
            ]
            if not rows:
                continue
            vec = emb[rows].mean(axis=0)
            nrm = np.linalg.norm(vec)
            self._user_emb[uid] = vec / nrm if nrm > 0 else vec

        # Popularity fallback for users with no positive text-backed history.
        pop = train.groupby(i).size()
        self._pop_order = pop.sort_values(ascending=False).index.tolist()
        return self

    def cold_item_ids(self) -> List[str]:
        """Items with text that never appear in training interactions."""
        train_items = getattr(self, "_train_item_ids", set())
        return [iid for iid in self._all_item_ids if iid not in train_items]

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        if self._item_emb is None:
            raise RuntimeError("Call fit() before recommend().")
        out: Dict[str, List[str]] = {}
        for user_id in users:
            uvec = self._user_emb.get(user_id)
            if uvec is None:
                # Cold / empty user: popularity over items that have text.
                seen = self.seen.get(user_id, set()) if exclude_seen else set()
                recs = [i for i in self._pop_order if i not in seen][:k]
                out[user_id] = recs
                continue
            scores = self._item_emb @ uvec  # cosine (both L2-normalized)
            out[user_id] = self._top_k_from_scores(user_id, scores.copy(), k, exclude_seen)
        return out


class ContentTwoTowerRecommender(TwoTowerRecommender):
    """Two-tower retrieval with text features in the item tower.

    Item representation = MLP([id_embedding ; text_embedding]). Collaborative
    id signal + content signal in one retriever — the usual production pattern
    for cold-start-aware retrieval.
    """

    name = "content_two_tower"

    def __init__(
        self,
        items: Optional[pd.DataFrame] = None,
        model_name: str = "all-MiniLM-L6-v2",
        svd_dim: int = 64,
        dim: int = 64,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-2,
        temperature: float = 0.1,
        weight_decay: float = 1e-5,
        normalize: bool = True,
        use_faiss: bool = False,
        device: str | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            dim=dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            temperature=temperature,
            weight_decay=weight_decay,
            normalize=normalize,
            use_faiss=use_faiss,
            device=device,
            seed=seed,
            verbose=verbose,
        )
        self.items = items
        self.model_name = model_name
        self.svd_dim = svd_dim
        self.backend: str = ""
        self._text_dim: int = 0
        self._text_by_train_idx: Optional[np.ndarray] = None

    def fit(
        self, train: pd.DataFrame, items: pd.DataFrame | None = None
    ) -> "ContentTwoTowerRecommender":
        import torch

        items = items if items is not None else self.items
        if items is None:
            raise ValueError("ContentTwoTowerRecommender needs items=Dataset.items.")
        self.items = items

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Build collaborative index first, then align text rows to train item order.
        self._build_index(train)
        emb, item_ids, backend = encode_item_text(
            items, model_name=self.model_name, svd_dim=self.svd_dim, seed=self.seed
        )
        self.backend = backend
        id_to_row = {iid: n for n, iid in enumerate(item_ids)}
        text_dim = emb.shape[1]
        self._text_dim = text_dim
        aligned = np.zeros((len(self.item_ids), text_dim), dtype=np.float32)
        for j, iid in enumerate(self.item_ids):
            r = id_to_row.get(iid)
            if r is not None:
                aligned[j] = emb[r]
        self._text_by_train_idx = aligned

        device = self._resolve_device(torch)
        u, i = settings.user_col, settings.item_col
        user_idx = torch.as_tensor(
            train[u].map(self.user_to_idx).to_numpy(), dtype=torch.long
        )
        item_idx = torch.as_tensor(
            train[i].map(self.item_to_idx).to_numpy(), dtype=torch.long
        )
        text_t = torch.as_tensor(aligned, dtype=torch.float32)

        self._model = _ContentTwoTowerNet(
            n_users=len(self.user_ids),
            n_items=len(self.item_ids),
            dim=self.dim,
            text_dim=text_dim,
            normalize=self.normalize,
        ).to(device)
        text_t = text_t.to(device)
        self._model.set_text(text_t)

        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        n = user_idx.shape[0]

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                if b.numel() < 2:
                    continue
                bu = user_idx[b].to(device)
                bi = item_idx[b].to(device)
                u_vec = self._model.user(bu)
                i_vec = self._model.item(bi)
                logits = (u_vec @ i_vec.t()) / self.temperature
                labels = torch.arange(bu.shape[0], device=device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * bu.shape[0]
            if self.verbose:
                print(f"  epoch {epoch + 1:>2}/{self.epochs}  loss={total / n:.4f}")

        with torch.no_grad():
            self._item_matrix = self._model.all_item_vectors().cpu().numpy()
        self._maybe_build_faiss()
        return self


try:
    import torch
    import torch.nn as _nn

    class _ContentTwoTowerNet(_nn.Module):
        def __init__(
            self, n_users: int, n_items: int, dim: int, text_dim: int, normalize: bool
        ):
            super().__init__()
            self.user_emb = _nn.Embedding(n_users, dim)
            self.item_id_emb = _nn.Embedding(n_items, dim)
            self.item_proj = _nn.Sequential(
                _nn.Linear(dim + text_dim, dim),
                _nn.ReLU(),
                _nn.Linear(dim, dim),
            )
            self.normalize = normalize
            self.register_buffer(
                "text_emb", torch.zeros(n_items, text_dim), persistent=False
            )
            _nn.init.normal_(self.user_emb.weight, std=0.1)
            _nn.init.normal_(self.item_id_emb.weight, std=0.1)

        def set_text(self, text: torch.Tensor) -> None:
            self.text_emb = text

        def _maybe_norm(self, x):
            import torch.nn.functional as F

            return F.normalize(x, dim=-1) if self.normalize else x

        def user(self, idx):
            return self._maybe_norm(self.user_emb(idx))

        def item(self, idx):
            id_v = self.item_id_emb(idx)
            tx = self.text_emb[idx]
            return self._maybe_norm(self.item_proj(torch.cat([id_v, tx], dim=-1)))

        def all_item_vectors(self):
            idx = torch.arange(
                self.item_id_emb.num_embeddings, device=self.item_id_emb.weight.device
            )
            return self.item(idx)

except ImportError:  # pragma: no cover
    _ContentTwoTowerNet = None
