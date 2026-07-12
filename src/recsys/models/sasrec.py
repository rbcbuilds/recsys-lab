"""SASRec — Self-Attentive Sequential Recommendation  [Phase 3].

Models the *order* of interactions, not just the set. A user is represented by
their recent item sequence; a causal transformer predicts the next item.

When static user embeddings break down:
  * taste drifts (seasonality, trends)
  * session / short-term intent dominates long-term taste
  * the same user needs different recs after different recent items

Architecture (Kang & McAuley 2018, simplified for this lab):
  item id → embedding (+ positional) → stacked causal self-attention blocks
  → last-position hidden state · item embedding matrix → next-item scores

Training: next-item cross-entropy on each position (causal mask so position t
only attends to 1..t). Inference: feed the user's last ``max_len`` items, take
the final hidden state, score all catalog items, top-K (exclude seen).

Reference: Kang & McAuley, \"Self-Attentive Sequential Recommendation\", ICDM 2018.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import settings
from .base import IndexedRecommender

# Padding index inside the item embedding table (index 0 is reserved).
_PAD = 0


class SASRecRecommender(IndexedRecommender):
    name = "sasrec"

    def __init__(
        self,
        dim: int = 64,
        max_len: int = 50,
        n_blocks: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.dim = dim
        self.max_len = max_len
        self.n_blocks = n_blocks
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

    def fit(self, train: pd.DataFrame) -> "SASRecRecommender":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_index(train)
        # Embedding index 0 = PAD; real items are shifted by +1.
        self._n_item_emb = len(self.item_ids) + 1
        device = self._resolve_device(torch)

        u, i, t = settings.user_col, settings.item_col, settings.time_col
        ordered = train.sort_values([u, t])
        self._sequences = {}
        for uid, grp in ordered.groupby(u, sort=False):
            idxs = [self.item_to_idx[x] + 1 for x in grp[i].tolist()]
            self._sequences[uid] = idxs

        # Training examples: for each user sequence, predict next item at each step.
        # seq_in[t] = history up to t; target[t] = item at t+1.
        seq_ins: List[List[int]] = []
        targets: List[List[int]] = []
        for idxs in self._sequences.values():
            if len(idxs) < 2:
                continue
            # Keep the most recent max_len+1 items so we can form max_len predictions.
            idxs = idxs[-(self.max_len + 1) :]
            seq = idxs[:-1]
            tgt = idxs[1:]
            # Left-pad to max_len.
            pad = self.max_len - len(seq)
            seq_ins.append([_PAD] * pad + seq)
            targets.append([_PAD] * pad + tgt)

        if not seq_ins:
            raise ValueError("SASRec needs users with at least 2 interactions.")

        X = torch.as_tensor(np.asarray(seq_ins, dtype=np.int64))
        Y = torch.as_tensor(np.asarray(targets, dtype=np.int64))

        self._model = _SASRecNet(
            n_items=self._n_item_emb,
            dim=self.dim,
            max_len=self.max_len,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(device)

        opt = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        n = X.shape[0]

        self._model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total = 0.0
            n_tok = 0
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                xb = X[b].to(device)
                yb = Y[b].to(device)
                logits = self._model(xb)  # (B, L, n_items)
                # Flatten; ignore PAD targets.
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

        # Build padded sequences (most recent max_len items).
        seqs = []
        for uid in known:
            idxs = self._sequences[uid][-self.max_len :]
            pad = self.max_len - len(idxs)
            seqs.append([_PAD] * pad + idxs)
        xb = torch.as_tensor(np.asarray(seqs, dtype=np.int64), device=device)

        with torch.no_grad():
            # Last-position hidden → score all items.
            hidden = self._model.encode(xb)  # (B, L, d)
            last = hidden[:, -1, :]  # (B, d)
            item_emb = self._model.item_emb.weight  # (n_items+1, d)
            scores = last @ item_emb.t()  # (B, n_items+1)
            scores = scores.cpu().numpy()

        for row, user_id in enumerate(known):
            # Drop PAD column (index 0); map remaining back to item ids.
            s = scores[row, 1:].copy()
            out[user_id] = self._top_k_from_scores(user_id, s, k, exclude_seen)
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

    class _SASRecNet(_nn.Module):
        def __init__(
            self,
            n_items: int,
            dim: int,
            max_len: int,
            n_blocks: int,
            n_heads: int,
            dropout: float,
        ):
            super().__init__()
            self.item_emb = _nn.Embedding(n_items, dim, padding_idx=_PAD)
            self.pos_emb = _nn.Embedding(max_len, dim)
            self.dropout = _nn.Dropout(dropout)
            self.blocks = _nn.ModuleList(
                [
                    _CausalBlock(dim, n_heads, dropout)
                    for _ in range(n_blocks)
                ]
            )
            self.norm = _nn.LayerNorm(dim)
            _nn.init.normal_(self.item_emb.weight, std=0.02)
            _nn.init.normal_(self.pos_emb.weight, std=0.02)
            import torch

            with torch.no_grad():
                self.item_emb.weight[_PAD].zero_()

        def encode(self, seq):
            """seq: (B, L) long → (B, L, d) hidden states."""
            import torch

            b, L = seq.shape
            pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(b, -1)
            x = self.item_emb(seq) + self.pos_emb(pos)
            x = self.dropout(x)
            # Causal mask only. Do NOT also key-pad-mask: a PAD query whose only
            # legal key is itself (also PAD) gets an empty softmax → NaNs that
            # poison the whole sequence. Zero PAD embeddings + ignore_index on
            # the loss are enough; inference always reads the last (real) step.
            for block in self.blocks:
                x = block(x)
            return self.norm(x)

        def forward(self, seq):
            hidden = self.encode(seq)
            # Tie output weights to item embeddings (standard SASRec).
            return hidden @ self.item_emb.weight.t()

    class _CausalBlock(_nn.Module):
        def __init__(self, dim: int, n_heads: int, dropout: float):
            super().__init__()
            self.attn = _nn.MultiheadAttention(
                dim, n_heads, dropout=dropout, batch_first=True
            )
            self.ff = _nn.Sequential(
                _nn.Linear(dim, dim * 4),
                _nn.GELU(),
                _nn.Dropout(dropout),
                _nn.Linear(dim * 4, dim),
                _nn.Dropout(dropout),
            )
            self.norm1 = _nn.LayerNorm(dim)
            self.norm2 = _nn.LayerNorm(dim)
            self.dropout = _nn.Dropout(dropout)

        def forward(self, x):
            import torch

            L = x.size(1)
            # Causal mask: position i cannot attend to j > i.
            causal = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )
            h, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
            x = self.norm1(x + self.dropout(h))
            x = self.norm2(x + self.ff(x))
            return x

except ImportError:  # pragma: no cover
    _SASRecNet = None
