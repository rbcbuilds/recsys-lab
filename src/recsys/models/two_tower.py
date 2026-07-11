"""Two-tower retrieval model  [Phase 2 — implemented].

A user tower and an item tower each map ids into a shared embedding space; the
score is their dot product. We train with **in-batch negatives** (sampled
softmax): within a mini-batch of (user, positive-item) pairs, every *other*
item in the batch acts as a negative. This is the workhorse of modern retrieval
(YouTube DNN / Yi et al. 2019) and scales to millions of items because, at
serving time, you embed items once and do a fast nearest-neighbor lookup.

This implementation keeps the towers deliberately simple — id embedding tables —
so the *training mechanics* are the lesson. The towers are the natural place to
later inject text/image/graph features (see the other Phase-3 modules): swap the
embedding lookup for an encoder over item features and nothing else changes.

Retrieval:
* Default: exact dot-product over the item matrix (fine for lab-sized catalogs).
* Optional: if ``faiss`` is installed and ``use_faiss=True``, an inner-product
  index is used — the same code path you'd deploy at scale.

References: Covington et al. 2016; Yi et al. 2019 (sampled-softmax two-tower).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import settings
from .base import IndexedRecommender


class TwoTowerRecommender(IndexedRecommender):
    name = "two_tower"

    def __init__(
        self,
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
        self.dim = dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.normalize = normalize
        self.use_faiss = use_faiss
        self.device = device
        self.seed = settings.seed if seed is None else seed
        self.verbose = verbose
        self._model = None
        self._item_matrix = None  # cached (n_items, dim) numpy for retrieval

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "TwoTowerRecommender":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_index(train)
        device = self._resolve_device(torch)
        u, i = settings.user_col, settings.item_col

        user_idx = torch.as_tensor(
            train[u].map(self.user_to_idx).to_numpy(), dtype=torch.long
        )
        item_idx = torch.as_tensor(
            train[i].map(self.item_to_idx).to_numpy(), dtype=torch.long
        )

        self._model = _TwoTowerNet(
            n_users=len(self.user_ids),
            n_items=len(self.item_ids),
            dim=self.dim,
            normalize=self.normalize,
        ).to(device)

        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        n = user_idx.shape[0]

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                if b.numel() < 2:  # in-batch negatives need >=2 examples
                    continue
                bu = user_idx[b].to(device)
                bi = item_idx[b].to(device)

                u_vec = self._model.user(bu)          # (B, d)
                i_vec = self._model.item(bi)          # (B, d)
                logits = (u_vec @ i_vec.t()) / self.temperature  # (B, B)
                labels = torch.arange(bu.shape[0], device=device)
                loss = torch.nn.functional.cross_entropy(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * bu.shape[0]

            if self.verbose:
                print(f"  epoch {epoch + 1:>2}/{self.epochs}  loss={total / n:.4f}")

        # Cache item embedding matrix for retrieval.
        with torch.no_grad():
            self._item_matrix = self._model.all_item_vectors().cpu().numpy()
        self._maybe_build_faiss()
        return self

    # ------------------------------------------------------------ recommend
    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        import torch

        if self._model is None:
            raise RuntimeError("Call fit() before recommend().")

        known = [u for u in users if u in self.user_to_idx]
        out: Dict[str, List[str]] = {u: [] for u in users}
        if not known:
            return out

        device = next(self._model.parameters()).device
        idxs = torch.as_tensor(
            [self.user_to_idx[u] for u in known], dtype=torch.long, device=device
        )
        with torch.no_grad():
            user_vecs = self._model.user(idxs).cpu().numpy()  # (m, d)

        if self._faiss_index is not None:
            self._recommend_faiss(known, user_vecs, k, exclude_seen, out)
        else:
            scores_all = user_vecs @ self._item_matrix.T  # (m, n_items)
            for row, user_id in enumerate(known):
                out[user_id] = self._top_k_from_scores(
                    user_id, scores_all[row].copy(), k, exclude_seen
                )
        return out

    # ------------------------------------------------------------- helpers
    def _resolve_device(self, torch):
        if self.device:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _maybe_build_faiss(self) -> None:
        self._faiss_index = None
        if not self.use_faiss:
            return
        try:
            import faiss
        except ImportError:
            if self.verbose:
                print("  faiss not installed; falling back to exact dot-product.")
            return
        mat = np.ascontiguousarray(self._item_matrix.astype(np.float32))
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        self._faiss_index = index

    def _recommend_faiss(self, known, user_vecs, k, exclude_seen, out) -> None:
        # Over-fetch so we can drop seen items and still return k.
        max_seen = max((len(self.seen.get(u, ())) for u in known), default=0)
        fetch = k + max_seen + 1
        q = np.ascontiguousarray(user_vecs.astype(np.float32))
        _scores, idxs = self._faiss_index.search(q, min(fetch, len(self.item_ids)))
        for row, user_id in enumerate(known):
            seen = self.seen.get(user_id, set()) if exclude_seen else set()
            recs = []
            for j in idxs[row]:
                item = self.idx_to_item[j]
                if item in seen:
                    continue
                recs.append(item)
                if len(recs) == k:
                    break
            out[user_id] = recs


try:  # Define the nn.Module only if torch is importable.
    import torch.nn as _nn

    class _TwoTowerNet(_nn.Module):
        """Two id-embedding towers sharing an output space."""

        def __init__(self, n_users: int, n_items: int, dim: int, normalize: bool):
            super().__init__()
            self.user_emb = _nn.Embedding(n_users, dim)
            self.item_emb = _nn.Embedding(n_items, dim)
            self.normalize = normalize
            _nn.init.normal_(self.user_emb.weight, std=0.1)
            _nn.init.normal_(self.item_emb.weight, std=0.1)

        def _maybe_norm(self, x):
            import torch.nn.functional as F

            return F.normalize(x, dim=-1) if self.normalize else x

        def user(self, idx):
            return self._maybe_norm(self.user_emb(idx))

        def item(self, idx):
            return self._maybe_norm(self.item_emb(idx))

        def all_item_vectors(self):
            return self._maybe_norm(self.item_emb.weight)

except ImportError:  # pragma: no cover - torch not installed
    _TwoTowerNet = None
