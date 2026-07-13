"""HSTU — hierarchical sequential transduction units (simplified lab version).

Industrial successor to SASRec: same causal sequential role, plus **relative time
bias** so recent gaps between actions matter (session breaks vs dense browsing).

This implementation keeps the SASRec stack and adds bucketed time-delta
embeddings at each position — enough to compare warm_r@20 vs SASRec and to
expose ``score_candidates`` as an optional ranker feature.

References: Zhai et al., "Actions Speak Louder than Words", 2024.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import settings
from .base import IndexedRecommender

_PAD = 0
_TIME_BUCKETS = 32


def _time_bucket(seconds: float) -> int:
    """Log-spaced bucket for inter-event gaps (seconds)."""
    if seconds <= 0:
        return 0
    # ~1 min .. ~1 year in 32 buckets
    log = np.log1p(seconds)
    return int(min(_TIME_BUCKETS - 1, log / np.log1p(365 * 86400) * (_TIME_BUCKETS - 1)))


class HSTURecommender(IndexedRecommender):
    name = "hstu"

    def __init__(
        self,
        dim: int = 64,
        max_len: int = 50,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
        epochs: int = 15,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.dim = dim
        self.max_len = max_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed = settings.seed if seed is None else seed
        self.verbose = verbose
        self._model = None
        self._sequences: Dict[str, List[int]] = {}
        self._time_gaps: Dict[str, List[int]] = {}

    def fit(self, train: pd.DataFrame) -> "HSTURecommender":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_index(train)
        self._n_item_emb = len(self.item_ids) + 1
        device = self._resolve_device(torch)

        u, i, t = settings.user_col, settings.item_col, settings.time_col
        ordered = train.sort_values([u, t])
        self._sequences = {}
        self._time_gaps = {}
        for uid, grp in ordered.groupby(u, sort=False):
            idxs = [self.item_to_idx[x] + 1 for x in grp[i].tolist()]
            ts = pd.to_datetime(grp[t]).astype("int64") // 10**9
            gaps = [0]
            for a, b in zip(ts.iloc[:-1], ts.iloc[1:]):
                gaps.append(_time_bucket(float(b - a)))
            self._sequences[uid] = idxs
            self._time_gaps[uid] = gaps

        seq_ins, targets, gap_ins = [], [], []
        for uid, idxs in self._sequences.items():
            if len(idxs) < 2:
                continue
            gaps = self._time_gaps[uid]
            tail_i = idxs[-(self.max_len + 1) :]
            tail_g = gaps[-(self.max_len + 1) :]
            seq, tgt = tail_i[:-1], tail_i[1:]
            gseq = tail_g[:-1]
            pad = self.max_len - len(seq)
            seq_ins.append([_PAD] * pad + seq)
            targets.append([_PAD] * pad + tgt)
            gap_ins.append([0] * pad + gseq)

        if not seq_ins:
            raise ValueError("HSTU needs users with at least 2 interactions.")

        X = torch.as_tensor(np.asarray(seq_ins, dtype=np.int64))
        Y = torch.as_tensor(np.asarray(targets, dtype=np.int64))
        G = torch.as_tensor(np.asarray(gap_ins, dtype=np.int64))

        self._model = _HSTUNet(
            n_items=self._n_item_emb,
            dim=self.dim,
            max_len=self.max_len,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            n_time_buckets=_TIME_BUCKETS,
        ).to(device)

        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        n = X.shape[0]
        self._model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total, n_tok = 0.0, 0
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                xb = X[b].to(device)
                yb = Y[b].to(device)
                gb = G[b].to(device)
                logits = self._model(xb, gb)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self._n_item_emb),
                    yb.view(-1),
                    ignore_index=_PAD,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                valid = int((yb != _PAD).sum())
                total += float(loss) * valid
                n_tok += valid
            if self.verbose and n_tok:
                print(f"  epoch {epoch + 1:>2}/{self.epochs}  loss={total / n_tok:.4f}")

        self._model.eval()
        return self

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        import torch

        if self._model is None:
            raise RuntimeError("Call fit() before recommend().")

        device = next(self._model.parameters()).device
        out: Dict[str, List[str]] = {u: [] for u in users}
        known = [u for u in users if u in self._sequences]
        if not known:
            return out

        seqs, gaps = [], []
        for uid in known:
            idxs = self._sequences[uid][-self.max_len :]
            g = self._time_gaps[uid][-self.max_len :]
            pad = self.max_len - len(idxs)
            seqs.append([_PAD] * pad + idxs)
            gaps.append([0] * pad + g)
        xb = torch.as_tensor(np.asarray(seqs, dtype=np.int64), device=device)
        gb = torch.as_tensor(np.asarray(gaps, dtype=np.int64), device=device)

        with torch.no_grad():
            hidden = self._model.encode(xb, gb)
            last = hidden[:, -1, :]
            item_emb = self._model.item_emb.weight
            scores = (last @ item_emb.t()).cpu().numpy()

        for row, user_id in enumerate(known):
            s = scores[row, 1:].copy()
            out[user_id] = self._top_k_from_scores(user_id, s, k, exclude_seen)
        return out

    def score_candidates(
        self, candidates: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        import torch

        if self._model is None:
            raise RuntimeError("Call fit() before score_candidates().")

        device = next(self._model.parameters()).device
        out: Dict[str, Dict[str, float]] = {u: {} for u in candidates}
        known = [u for u in candidates if u in self._sequences]
        if not known:
            return out

        seqs, gaps = [], []
        for uid in known:
            idxs = self._sequences[uid][-self.max_len :]
            g = self._time_gaps[uid][-self.max_len :]
            pad = self.max_len - len(idxs)
            seqs.append([_PAD] * pad + idxs)
            gaps.append([0] * pad + g)
        xb = torch.as_tensor(np.asarray(seqs, dtype=np.int64), device=device)
        gb = torch.as_tensor(np.asarray(gaps, dtype=np.int64), device=device)

        with torch.no_grad():
            hidden = self._model.encode(xb, gb)
            last = hidden[:, -1, :]
            item_weight = self._model.item_emb.weight

        for row, user_id in enumerate(known):
            uvec = last[row]
            scores: Dict[str, float] = {}
            for item_id in candidates.get(user_id, []):
                j = self.item_to_idx.get(item_id)
                if j is None:
                    continue
                scores[item_id] = float(uvec @ item_weight[j + 1])
            out[user_id] = scores
        return out

    def _resolve_device(self, torch):
        if self.device:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


try:
    import torch.nn as _nn

    class _HSTUNet(_nn.Module):
        def __init__(
            self,
            n_items: int,
            dim: int,
            max_len: int,
            n_layers: int,
            n_heads: int,
            dropout: float,
            n_time_buckets: int,
        ):
            super().__init__()
            self.item_emb = _nn.Embedding(n_items, dim, padding_idx=_PAD)
            self.pos_emb = _nn.Embedding(max_len, dim)
            self.time_emb = _nn.Embedding(n_time_buckets, dim)
            self.dropout = _nn.Dropout(dropout)
            layer = _nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = _nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = _nn.LayerNorm(dim)
            _nn.init.normal_(self.item_emb.weight, std=0.02)
            _nn.init.normal_(self.pos_emb.weight, std=0.02)
            import torch

            with torch.no_grad():
                self.item_emb.weight[_PAD].zero_()

        def encode(self, seq, time_gaps):
            import torch

            b, L = seq.shape
            pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(b, -1)
            x = self.item_emb(seq) + self.pos_emb(pos) + self.time_emb(time_gaps)
            x = self.dropout(x)
            mask = torch.triu(
                torch.full((L, L), float("-inf"), device=seq.device), diagonal=1
            )
            x = self.encoder(x, mask=mask)
            return self.norm(x)

        def forward(self, seq, time_gaps):
            hidden = self.encode(seq, time_gaps)
            return hidden @ self.item_emb.weight.t()

except ImportError:  # pragma: no cover
    _HSTUNet = None
