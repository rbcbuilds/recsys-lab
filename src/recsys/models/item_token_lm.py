"""Item-token language model — generative sequential recommendation.

Treats each item id as a token in a causal language model (GPT-style over item
sequences). Training = next-token prediction on user histories. Inference =
**autoregressive decoding**: generate one item token at a time, append to context,
repeat — the model *produces* the list rather than scoring the full catalog.

Contrast with SASRec (discriminative): SASRec scores every item via a bi-encoder
dot product. This LM samples/emits tokens from a softmax over the vocabulary.

Cold / empty users fall back to popularity over train items.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import settings
from .base import IndexedRecommender

_PAD = 0


class ItemTokenLMRecommender(IndexedRecommender):
    name = "item_token_lm"

    def __init__(
        self,
        dim: int = 64,
        max_len: int = 50,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
        epochs: int = 12,
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
        self._pop_order: List[str] = []

    def fit(self, train: pd.DataFrame) -> "ItemTokenLMRecommender":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_index(train)
        self._n_vocab = len(self.item_ids) + 1  # 0 = PAD
        device = self._resolve_device(torch)

        u, i, t = settings.user_col, settings.item_col, settings.time_col
        ordered = train.sort_values([u, t])
        self._sequences = {}
        for uid, grp in ordered.groupby(u, sort=False):
            self._sequences[uid] = [
                self.item_to_idx[x] + 1 for x in grp[i].tolist()
            ]

        pop = train.groupby(i).size()
        self._pop_order = [
            str(x) for x in pop.sort_values(ascending=False).index.tolist()
        ]

        seq_ins, targets = [], []
        for idxs in self._sequences.values():
            if len(idxs) < 2:
                continue
            tail = idxs[-(self.max_len + 1) :]
            seq, tgt = tail[:-1], tail[1:]
            pad = self.max_len - len(seq)
            seq_ins.append([_PAD] * pad + seq)
            targets.append([_PAD] * pad + tgt)

        if not seq_ins:
            raise ValueError("ItemTokenLM needs users with at least 2 interactions.")

        X = torch.as_tensor(np.asarray(seq_ins, dtype=np.int64))
        Y = torch.as_tensor(np.asarray(targets, dtype=np.int64))

        self._model = _ItemTokenLM(
            n_vocab=self._n_vocab,
            dim=self.dim,
            max_len=self.max_len,
            n_layers=self.n_layers,
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
            total, n_tok = 0.0, 0
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                xb = X[b].to(device)
                yb = Y[b].to(device)
                logits = self._model(xb)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self._n_vocab),
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
        out: Dict[str, List[str]] = {}

        for user_id in users:
            idxs = self._sequences.get(user_id)
            if not idxs:
                seen = self.seen.get(user_id, set()) if exclude_seen else set()
                out[user_id] = [i for i in self._pop_order if i not in seen][:k]
                continue

            banned = set()
            if exclude_seen:
                banned |= {
                    self.item_to_idx[it] + 1
                    for it in self.seen.get(user_id, set())
                    if it in self.item_to_idx
                }

            context = idxs[-self.max_len :]
            recs: List[str] = []
            with torch.no_grad():
                for _ in range(k):
                    pad = self.max_len - len(context)
                    xb = torch.as_tensor(
                        [[_PAD] * pad + context], dtype=torch.long, device=device
                    )
                    logits = self._model(xb)[0, -1].clone()
                    logits[_PAD] = -np.inf
                    for tok in banned:
                        logits[tok] = -np.inf
                    tok = int(logits.argmax().item())
                    if not np.isfinite(float(logits[tok])):
                        break
                    item_id = self.idx_to_item[tok - 1]
                    recs.append(str(item_id))
                    banned.add(tok)
                    context = (context + [tok])[-self.max_len :]

            out[user_id] = recs
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
    import torch
    import torch.nn as _nn

    class _ItemTokenLM(_nn.Module):
        """Causal item-token LM with tied input/output embeddings."""

        def __init__(
            self, n_vocab: int, dim: int, max_len: int, n_layers: int, n_heads: int, dropout: float
        ):
            super().__init__()
            self.max_len = max_len
            self.tok_emb = _nn.Embedding(n_vocab, dim, padding_idx=_PAD)
            self.pos_emb = _nn.Embedding(max_len, dim)
            layer = _nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = _nn.TransformerEncoder(layer, num_layers=n_layers)
            self.dropout = _nn.Dropout(dropout)
            self.lm_head = _nn.Linear(dim, n_vocab, bias=False)
            self.lm_head.weight = self.tok_emb.weight  # weight tying
            _nn.init.normal_(self.tok_emb.weight, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, seq_len = x.shape
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, -1)
            h = self.tok_emb(x) + self.pos_emb(pos)
            h = self.dropout(h)
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
            )
            h = self.encoder(h, mask=mask)
            return self.lm_head(h)

except ImportError:  # pragma: no cover
    _ItemTokenLM = None
