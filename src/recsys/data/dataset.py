"""The unified in-memory dataset container.

Every data source (synthetic generator or real Yelp subset) is loaded into this
same shape, so models never care where the data came from. This is the schema
contract for the whole lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import settings


@dataclass
class Dataset:
    """A unified recsys dataset.

    Attributes
    ----------
    interactions:
        Long-format DataFrame with columns [user_id, item_id, rating, timestamp].
        This is the core signal every model consumes.
    users:
        One row per user. At minimum a ``user_id`` column.
    items:
        One row per item. Includes ``item_id`` and (when available) ``name``,
        ``categories``, and a ``text`` column used by the text/LLM module.
    social:
        Undirected friend edges as a DataFrame with columns [user_id, friend_id].
        Empty when the source has no social graph.
    item_image_vectors:
        Optional (n_items, dim) array of image (or image-stand-in) embeddings,
        row-aligned to ``items``. Used by the multimodal module.
    """

    interactions: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    social: pd.DataFrame
    item_image_vectors: Optional[np.ndarray] = None

    # ---- convenient views ------------------------------------------------
    @property
    def n_users(self) -> int:
        return self.users[settings.user_col].nunique()

    @property
    def n_items(self) -> int:
        return self.items[settings.item_col].nunique()

    @property
    def n_interactions(self) -> int:
        return len(self.interactions)

    @property
    def n_social_edges(self) -> int:
        return len(self.social)

    def summary(self) -> str:
        density = (
            self.n_interactions / (self.n_users * self.n_items)
            if self.n_users and self.n_items
            else 0.0
        )
        return (
            f"Dataset(users={self.n_users:,}, items={self.n_items:,}, "
            f"interactions={self.n_interactions:,}, density={density:.4%}, "
            f"social_edges={self.n_social_edges:,}, "
            f"has_images={self.item_image_vectors is not None})"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()
