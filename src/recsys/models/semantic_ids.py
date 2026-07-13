"""Semantic / hierarchical item IDs for generative retrieval.

Maps each item to a short compositional token sequence (category + cluster + leaf)
so a causal LM can emit structured codes — better cold transfer than flat item ids.

Contrast with ``item_token_lm.py``: one arbitrary id per item vs shared prefixes
for similar catalog entries.

References: Rajput et al. (TIGER), 2023; LC-Rec, 2024.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import settings
from .base import IndexedRecommender
from .text_embeddings import encode_item_text

_PAD = 0
_TOKENS_PER_ITEM = 3


def _category_token(categories: str) -> str:
    text = (categories or "").strip().lower()
    if not text:
        return "cat_unknown"
    first = re.split(r"[,;|]", text)[0].strip()
    first = re.sub(r"[^a-z0-9]+", "_", first)[:24] or "unknown"
    return f"cat_{first}"


def build_semantic_item_tokens(
    items: pd.DataFrame,
    n_clusters: int = 32,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Assign each item a 3-token semantic code.

    Returns ``(item_to_tokens, token_triplet_to_items)``.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    from .text_embeddings import _item_text_series

    i = settings.item_col
    item_ids = items[i].astype(str).tolist()
    cats = (
        items["categories"].fillna("").astype(str).tolist()
        if "categories" in items.columns
        else [""] * len(items)
    )

    texts = _item_text_series(items).tolist()
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    d = min(64, max(2, X.shape[1] - 1), X.shape[0] - 1)
    emb = TruncatedSVD(n_components=d, random_state=seed).fit_transform(X).astype(
        np.float32
    )
    n_clusters = min(n_clusters, max(2, len(item_ids) // 4))
    cluster_ids = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
        emb
    )

    cluster_leaves: Dict[int, List[str]] = defaultdict(list)
    for iid, cid in zip(item_ids, cluster_ids):
        cluster_leaves[int(cid)].append(iid)

    item_to_tokens: Dict[str, List[str]] = {}
    triplet_to_items: Dict[str, List[str]] = defaultdict(list)

    for row, iid in enumerate(item_ids):
        cid = int(cluster_ids[row])
        leaf_rank = cluster_leaves[cid].index(iid)
        tokens = [
            _category_token(cats[row]),
            f"cl_{cid}",
            f"lf_{leaf_rank}",
        ]
        item_to_tokens[iid] = tokens
        triplet_to_items["|".join(tokens)].append(iid)

    return item_to_tokens, dict(triplet_to_items)


class SemanticIDRecommender(IndexedRecommender):
    name = "semantic_id_lm"

    def __init__(
        self,
        n_clusters: int = 32,
        tokens_per_item: int = _TOKENS_PER_ITEM,
        dim: int = 64,
        max_len: int = 150,
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
        self.n_clusters = n_clusters
        self.tokens_per_item = tokens_per_item
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
        self._token_to_idx: Dict[str, int] = {}
        self._idx_to_token: List[str] = []
        self._item_to_tokens: Dict[str, List[str]] = {}
        self._triplet_to_items: Dict[str, List[str]] = {}
        self._token_sequences: Dict[str, List[int]] = {}
        self._pop_order: List[str] = []
        self._item_repr: Optional[np.ndarray] = None

    def fit(
        self, train: pd.DataFrame, items: Optional[pd.DataFrame] = None
    ) -> "SemanticIDRecommender":
        import torch

        if items is None:
            raise ValueError("SemanticIDRecommender requires items=Dataset.items.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_index(train)
        i_col = settings.item_col
        all_ids = items[i_col].astype(str).tolist()
        self.item_ids = all_ids
        self.item_to_idx = {iid: j for j, iid in enumerate(all_ids)}
        self.idx_to_item = np.array(all_ids)

        self._item_to_tokens, self._triplet_to_items = build_semantic_item_tokens(
            items, n_clusters=self.n_clusters, seed=self.seed
        )

        vocab = sorted({t for toks in self._item_to_tokens.values() for t in toks})
        self._token_to_idx = {t: j + 1 for j, t in enumerate(vocab)}
        self._idx_to_token = ["<pad>"] + vocab
        n_vocab = len(self._idx_to_token)

        u, i, t = settings.user_col, settings.item_col, settings.time_col
        ordered = train.sort_values([u, t])
        self._token_sequences = {}
        for uid, grp in ordered.groupby(u, sort=False):
            seq: List[int] = []
            for iid in grp[i].astype(str):
                for tok in self._item_to_tokens.get(iid, []):
                    seq.append(self._token_to_idx.get(tok, 0))
            if seq:
                self._token_sequences[uid] = seq

        pop = train.groupby(i).size()
        self._pop_order = [str(x) for x in pop.sort_values(ascending=False).index]

        seq_ins, targets = [], []
        for seq in self._token_sequences.values():
            if len(seq) < 2:
                continue
            tail = seq[-(self.max_len + 1) :]
            s, tgt = tail[:-1], tail[1:]
            pad = self.max_len - len(s)
            seq_ins.append([_PAD] * pad + s)
            targets.append([_PAD] * pad + tgt)

        if not seq_ins:
            raise ValueError("SemanticIDRecommender needs users with >=2 item tokens.")

        device = self._resolve_device(torch)
        X = torch.as_tensor(np.asarray(seq_ins, dtype=np.int64))
        Y = torch.as_tensor(np.asarray(targets, dtype=np.int64))

        self._model = _SemanticLM(
            n_vocab=n_vocab,
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
                    logits.view(-1, n_vocab), yb.view(-1), ignore_index=_PAD
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
        self._cache_item_representations()
        return self

    def _cache_item_representations(self) -> None:
        import torch

        device = next(self._model.parameters()).device
        tok_emb = self._model.tok_emb.weight.detach()
        rows = []
        for iid in self.item_ids:
            ids = [self._token_to_idx[t] for t in self._item_to_tokens[str(iid)]]
            vec = tok_emb[torch.as_tensor(ids, device=device)].mean(dim=0)
            rows.append(vec.cpu().numpy())
        self._item_repr = np.asarray(rows, dtype=np.float32)

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        import torch

        if self._model is None:
            raise RuntimeError("Call fit() before recommend().")

        device = next(self._model.parameters()).device
        out: Dict[str, List[str]] = {}
        item_ids = [str(x) for x in self.item_ids]

        for user_id in users:
            seq = self._token_sequences.get(user_id)
            if not seq:
                seen = self.seen.get(user_id, set()) if exclude_seen else set()
                out[user_id] = [i for i in self._pop_order if i not in seen][:k]
                continue

            seen_items = self.seen.get(user_id, set()) if exclude_seen else set()
            with torch.no_grad():
                uvec = self._model.encode_context(
                    seq[-self.max_len :], self.max_len, device
                )
            scores = self._item_repr @ uvec
            ranked = sorted(
                (
                    (item_ids[i], float(scores[i]))
                    for i in range(len(item_ids))
                    if item_ids[i] not in seen_items
                ),
                key=lambda x: -x[1],
            )
            out[user_id] = [iid for iid, _ in ranked[:k]]
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

    class _SemanticLM(_nn.Module):
        def __init__(
            self, n_vocab: int, dim: int, max_len: int, n_layers: int, n_heads: int, dropout: float
        ):
            super().__init__()
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
            self.lm_head.weight = self.tok_emb.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.encode(x))

        def encode_context(
            self, seq: List[int], max_len: int, device
        ) -> np.ndarray:
            pad = max_len - len(seq)
            xb = torch.as_tensor(
                [[_PAD] * pad + seq], dtype=torch.long, device=device
            )
            h = self.encode(xb)
            return h[0, -1].cpu().numpy()

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            b, seq_len = x.shape
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, -1)
            h = self.tok_emb(x) + self.pos_emb(pos)
            h = self.dropout(h)
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
            )
            return self.encoder(h, mask=mask)

except ImportError:  # pragma: no cover
    _SemanticLM = None
