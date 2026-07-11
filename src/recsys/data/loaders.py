"""Load a :class:`Dataset` from processed parquet, generating synthetic data if missing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, settings
from .dataset import Dataset


def load_dataset(auto_synth: bool = True) -> Dataset:
    """Load the processed dataset.

    Looks for parquet files in ``data/processed/``. If none exist and
    ``auto_synth`` is True, generates a synthetic dataset first so the pipeline
    always has something to run on.
    """
    interactions_path = PROCESSED_DIR / "interactions.parquet"

    if not interactions_path.exists():
        if not auto_synth:
            raise FileNotFoundError(
                f"No processed data at {PROCESSED_DIR}. Run scripts/make_subset.py "
                f"for real Yelp data, or recsys.data.synthetic.generate() for synthetic."
            )
        from . import synthetic

        synthetic.generate(write=True)

    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    users = pd.read_parquet(PROCESSED_DIR / "users.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")

    social_path = PROCESSED_DIR / "social.parquet"
    social = (
        pd.read_parquet(social_path)
        if social_path.exists()
        else pd.DataFrame(columns=[settings.user_col, "friend_id"])
    )

    img_path = PROCESSED_DIR / "item_image_vectors.npy"
    item_image_vectors = np.load(img_path) if img_path.exists() else None

    interactions[settings.time_col] = pd.to_datetime(interactions[settings.time_col])

    return Dataset(
        interactions=interactions,
        users=users,
        items=items,
        social=social,
        item_image_vectors=item_image_vectors,
    )
