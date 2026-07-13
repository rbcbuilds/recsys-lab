"""Slate diversity optimization (MMR, DPP).

Re-orders a ranked candidate list post-hoc to balance relevance and diversity.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np


def _relevance(
    user_id: str,
    item_id: str,
    rank: int,
    relevance_scores: Optional[Dict[str, Dict[str, float]]],
) -> float:
    if relevance_scores and user_id in relevance_scores:
        return float(relevance_scores[user_id].get(item_id, 0.0))
    # Inverse rank when explicit scores are unavailable.
    return 1.0 / (rank + 1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(a @ b / (na * nb))


def _mmr_select(
    items: List[str],
    embeddings: Dict[str, np.ndarray],
    k: int,
    lambda_: float,
    user_id: str,
    relevance_scores: Optional[Dict[str, Dict[str, float]]],
) -> List[str]:
    selected: List[str] = []
    remaining = list(items)
    rank_map = {it: r for r, it in enumerate(items)}

    while remaining and len(selected) < k:
        best_item, best_score = None, -np.inf
        for item in remaining:
            rel = _relevance(user_id, item, rank_map[item], relevance_scores)
            if not selected:
                mmr = rel
            else:
                emb = embeddings.get(item)
                if emb is None:
                    sim = 0.0
                else:
                    sim = max(
                        _cosine(emb, embeddings[s])
                        for s in selected
                        if s in embeddings
                    )
                mmr = lambda_ * rel - (1.0 - lambda_) * sim
            if mmr > best_score:
                best_score, best_item = mmr, item
        if best_item is None:
            break
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


def _dpp_select(
    items: List[str],
    embeddings: Dict[str, np.ndarray],
    k: int,
    user_id: str,
    relevance_scores: Optional[Dict[str, Dict[str, float]]],
) -> List[str]:
    """Greedy MAP inference on a quality-diversity kernel (simplified DPP)."""
    rank_map = {it: r for r, it in enumerate(items)}
    selected: List[str] = []
    remaining = list(items)

    while remaining and len(selected) < k:
        best_item, best_gain = None, -np.inf
        for item in remaining:
            emb = embeddings.get(item)
            q = _relevance(user_id, item, rank_map[item], relevance_scores)
            if not selected or emb is None:
                gain = q * q
            else:
                sims = [
                    _cosine(emb, embeddings[s]) for s in selected if s in embeddings
                ]
                orth = 1.0 - max(sims) if sims else 1.0
                gain = (q * orth) ** 2
            if gain > best_gain:
                best_gain, best_item = gain, item
        if best_item is None:
            break
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


def diversify_slate(
    recommendations: Dict[str, List[str]],
    item_embeddings: Dict[str, np.ndarray],
    k: int = 10,
    method: Literal["mmr", "dpp"] = "mmr",
    lambda_: float = 0.7,
    relevance_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, List[str]]:
    """Re-rank each user's list for diversity while preserving relevance."""
    out: Dict[str, List[str]] = {}
    for user_id, items in recommendations.items():
        if not items:
            out[user_id] = []
            continue
        pool = items[: max(k * 5, k)]
        if method == "dpp":
            out[user_id] = _dpp_select(
                pool, item_embeddings, k, user_id, relevance_scores
            )
        else:
            out[user_id] = _mmr_select(
                pool, item_embeddings, k, lambda_, user_id, relevance_scores
            )
    return out
