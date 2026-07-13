"""Contrastive / hard-negative two-tower retrieval.

Extends :class:`~recsys.models.two_tower.TwoTowerRecommender` with popularity-sampled
hard negatives and optional embedding dropout augmentation during training.
Inference is identical (dot product + cached item matrix).

References: Yu et al. (SimGCL), 2022; CLRec, 2021.
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from ..config import settings
from .two_tower import TwoTowerRecommender


class ContrastiveTwoTowerRecommender(TwoTowerRecommender):
    """Two-tower with hard-negative contrastive training."""

    name = "contrastive_two_tower"

    def __init__(
        self,
        hard_negative_k: int = 5,
        augmentation_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hard_negative_k = hard_negative_k
        self.augmentation_dropout = augmentation_dropout
        self._hard_neg_pool: np.ndarray = np.array([], dtype=np.int64)
        self._user_positives: Dict[int, Set[int]] = {}

    def fit(self, train: pd.DataFrame) -> "ContrastiveTwoTowerRecommender":
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_index(train)
        device = self._resolve_device(torch)
        u, i = settings.user_col, settings.item_col

        user_idx = train[u].map(self.user_to_idx).astype(np.int64).to_numpy()
        item_idx = train[i].map(self.item_to_idx).astype(np.int64).to_numpy()

        pop = train.groupby(i).size().sort_values(ascending=False)
        self._hard_neg_pool = np.asarray(
            [self.item_to_idx[str(x)] for x in pop.index], dtype=np.int64
        )

        self._user_positives = {}
        for uidx, iidx in zip(user_idx, item_idx):
            self._user_positives.setdefault(int(uidx), set()).add(int(iidx))

        self._model = self._build_model().to(device)
        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        n = user_idx.shape[0]
        k = self.hard_negative_k

        for epoch in range(self.epochs):
            perm = np.random.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                batch = perm[start : start + self.batch_size]
                if batch.size < 1:
                    continue
                bu = torch.as_tensor(user_idx[batch], dtype=torch.long, device=device)
                bi = torch.as_tensor(item_idx[batch], dtype=torch.long, device=device)

                u_vec = self._model.user(bu)
                i_vec = self._model.item(bi)
                if self.augmentation_dropout > 0:
                    u_vec = F.dropout(u_vec, p=self.augmentation_dropout, training=True)
                    i_vec = F.dropout(i_vec, p=self.augmentation_dropout, training=True)

                # In-batch negatives
                in_batch = (u_vec @ i_vec.t()) / self.temperature
                labels = torch.arange(bu.shape[0], device=device)
                loss = F.cross_entropy(in_batch, labels)

                # Hard negatives: popular items the user did not interact with
                if k > 0 and self._hard_neg_pool.size > 0:
                    hard = []
                    for uidx in user_idx[batch]:
                        pos = self._user_positives[int(uidx)]
                        cands = [j for j in self._hard_neg_pool if j not in pos]
                        if not cands:
                            cands = self._hard_neg_pool.tolist()
                        pick = np.random.choice(
                            cands, size=min(k, len(cands)), replace=len(cands) < k
                        )
                        hard.append(pick)
                    hard_t = torch.as_tensor(np.asarray(hard), dtype=torch.long, device=device)
                    hard_emb = self._model.item(hard_t.view(-1)).view(bu.shape[0], -1, self.dim)
                    hard_logits = (u_vec.unsqueeze(1) * hard_emb).sum(dim=-1) / self.temperature
                    pos_logits = (u_vec * i_vec).sum(dim=-1, keepdim=True) / self.temperature
                    logits = torch.cat([pos_logits, hard_logits], dim=1)
                    hard_labels = torch.zeros(bu.shape[0], dtype=torch.long, device=device)
                    loss = loss + F.cross_entropy(logits, hard_labels)

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

    def _build_model(self):
        from .two_tower import _TwoTowerNet

        return _TwoTowerNet(
            n_users=len(self.user_ids),
            n_items=len(self.item_ids),
            dim=self.dim,
            normalize=self.normalize,
        )
