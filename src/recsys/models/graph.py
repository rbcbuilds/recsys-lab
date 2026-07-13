"""Graph recommendation (LightGCN) on the user-item interaction bipartite graph."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import settings
from .base import IndexedRecommender


def _normalized_bipartite_adj(
    user_item: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Symmetric normalized adjacency for the user-item bipartite graph."""
    n_users, n_items = user_item.shape
    n_nodes = n_users + n_items
    rows, cols = user_item.nonzero()
    data = np.ones(len(rows), dtype=np.float32)
    ui = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
    )
    upper = sparse.hstack(
        [
            sparse.csr_matrix((n_users, n_users), dtype=np.float32),
            ui,
        ]
    )
    lower = sparse.hstack(
        [
            ui.T,
            sparse.csr_matrix((n_items, n_items), dtype=np.float32),
        ]
    )
    adj = sparse.vstack([upper, lower]).tocsr()
    deg = np.asarray(adj.sum(axis=1)).flatten()
    inv_sqrt = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    d = sparse.diags(inv_sqrt)
    return (d @ adj @ d).tocsr()


class LightGCNRecommender(IndexedRecommender):
    """LightGCN: K-layer neighborhood aggregation + BPR training (He et al. 2020)."""

    name = "lightgcn"

    def __init__(
        self,
        dim: int = 64,
        n_layers: int = 2,
        epochs: int = 15,
        batch_size: int = 2048,
        lr: float = 1e-3,
        reg: float = 1e-4,
        n_neg: int = 1,
        device: str | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reg = reg
        self.n_neg = n_neg
        self.device = device
        self.seed = settings.seed if seed is None else seed
        self.verbose = verbose
        self._embeddings: Optional[np.ndarray] = None
        self._norm_adj = None
        self._pos_items: Dict[int, np.ndarray] = {}

    def fit(self, train: pd.DataFrame) -> "LightGCNRecommender":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_index(train)
        u, i = settings.user_col, settings.item_col
        n_users, n_items = len(self.user_ids), len(self.item_ids)

        rows = train[u].map(self.user_to_idx).astype(np.int64).values
        cols = train[i].map(self.item_to_idx).astype(np.int64).values
        vals = np.ones(len(train), dtype=np.float32)
        ui = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        self._norm_adj = _normalized_bipartite_adj(ui)

        for uidx, grp in train.groupby(u):
            self._pos_items[self.user_to_idx[uidx]] = (
                grp[i].map(self.item_to_idx).astype(np.int64).unique()
            )

        device = self._resolve_device(torch)
        n_nodes = n_users + n_items
        adj = self._sparse_to_torch(self._norm_adj, device)

        emb = nn.Embedding(n_nodes, self.dim)
        nn.init.xavier_uniform_(emb.weight)
        opt = torch.optim.Adam(emb.parameters(), lr=self.lr)

        all_items = np.arange(n_items, dtype=np.int64)
        users_with_pos = np.asarray(sorted(self._pos_items.keys()), dtype=np.int64)
        if users_with_pos.size == 0:
            raise ValueError("LightGCN needs at least one user-item interaction.")

        def propagate() -> torch.Tensor:
            x = emb.weight
            layers = [x]
            for _ in range(self.n_layers):
                x = torch.sparse.mm(adj, x)
                layers.append(x)
            return torch.stack(layers, dim=0).mean(dim=0)

        for epoch in range(self.epochs):
            perm = np.random.permutation(users_with_pos.size)
            total_loss = 0.0
            n_steps = 0
            for start in range(0, users_with_pos.size, self.batch_size):
                batch_u = users_with_pos[perm[start : start + self.batch_size]]
                u_t = torch.as_tensor(batch_u, device=device, dtype=torch.long)
                pos = [int(np.random.choice(self._pos_items[int(u)])) for u in batch_u]
                pos_t = torch.as_tensor(pos, device=device, dtype=torch.long)
                neg = [
                    int(np.random.choice(all_items))
                    for _ in batch_u
                ]
                # Resample negatives that collide with a positive.
                for j, u in enumerate(batch_u):
                    pos_set = set(self._pos_items[int(u)].tolist())
                    while neg[j] in pos_set:
                        neg[j] = int(np.random.choice(all_items))
                neg_t = torch.as_tensor(neg, device=device, dtype=torch.long)

                all_emb = propagate()
                u_emb = all_emb[u_t]
                pos_emb = all_emb[pos_t + n_users]
                neg_emb = all_emb[neg_t + n_users]
                pos_score = (u_emb * pos_emb).sum(dim=1)
                neg_score = (u_emb * neg_emb).sum(dim=1)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
                loss = loss + self.reg * emb.weight.norm(2).pow(2) / n_nodes

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss)
                n_steps += 1

            if self.verbose and n_steps:
                print(f"  epoch {epoch + 1:>2}/{self.epochs}  loss={total_loss / n_steps:.4f}")

        with torch.no_grad():
            self._embeddings = propagate().cpu().numpy()
        return self

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        if self._embeddings is None:
            raise RuntimeError("Call fit() before recommend().")
        n_users = len(self.user_ids)
        out: Dict[str, List[str]] = {}
        for user_id in users:
            uidx = self.user_to_idx.get(user_id)
            if uidx is None:
                out[user_id] = []
                continue
            uvec = self._embeddings[uidx]
            scores = self._embeddings[n_users:] @ uvec
            out[user_id] = self._top_k_from_scores(user_id, scores.copy(), k, exclude_seen)
        return out

    def score_candidates(
        self, candidates: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        if self._embeddings is None:
            raise RuntimeError("Call fit() before score_candidates().")
        n_users = len(self.user_ids)
        out: Dict[str, Dict[str, float]] = {}
        for user_id, items in candidates.items():
            uidx = self.user_to_idx.get(user_id)
            if uidx is None:
                out[user_id] = {str(it): 0.0 for it in items}
                continue
            uvec = self._embeddings[uidx]
            scores: Dict[str, float] = {}
            for item_id in items:
                j = self.item_to_idx.get(str(item_id))
                if j is None:
                    scores[str(item_id)] = 0.0
                else:
                    scores[str(item_id)] = float(self._embeddings[n_users + j] @ uvec)
            out[user_id] = scores
        return out

    def _sparse_to_torch(self, mat: sparse.csr_matrix, device):
        import torch

        coo = mat.tocoo()
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(coo.data.astype(np.float32))
        return torch.sparse_coo_tensor(
            indices, values, mat.shape, device=device
        ).coalesce()

    def _resolve_device(self, torch):
        if self.device:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
