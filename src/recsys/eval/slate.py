"""Slate diversity optimization (MMR, DPP)  [SCAFFOLD — Tier 3].

Retrieval + ranking optimize per-item relevance; production feeds also need
**diversity** (avoid 10 sushi places in a row). You already measure
``coverage@K``; this module re-orders a ranked list post-hoc.

Techniques:
  * **MMR** (Maximal Marginal Relevance): greedy pick items balancing relevance
    vs similarity to already-selected items (needs item embeddings).
  * **DPP** (Determinantal Point Process): probabilistic diverse subsets; harder
    but SOTA for slate diversity.

Integration:
  ```python
  recs = model.recommend(users, k=200)
  diverse = diversify_slate(recs, item_embeddings, k=20, method="mmr", lambda_=0.7)
  metrics = evaluate(diverse, test_pos)  # compare NDCG vs coverage tradeoff
  ```

Interview point: show relevance–diversity Pareto curve, not a single number.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np


def diversify_slate(
    recommendations: Dict[str, List[str]],
    item_embeddings: Dict[str, np.ndarray],
    k: int = 10,
    method: Literal["mmr", "dpp"] = "mmr",
    lambda_: float = 0.7,
    relevance_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, List[str]]:
    """Re-rank each user's list for diversity while preserving relevance.

    Parameters
    ----------
    recommendations:
        ``{user_id: [item_id, ...]}`` from any model (longer list than final k).
    item_embeddings:
        ``{item_id: vector}`` for similarity (text or two-tower item vectors).
    lambda_:
        MMR tradeoff: 1.0 = pure relevance, 0.0 = pure diversity.
    relevance_scores:
        Optional per-(user, item) scores; defaults to inverse rank.
    """
    raise NotImplementedError(
        "Implement MMR greedy selection first (simpler than DPP). See this "
        "module's docstring and docs/techniques.md (diversity Pareto analysis)."
    )
