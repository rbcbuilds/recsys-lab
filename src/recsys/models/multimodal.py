"""Multimodal (image) recommendation  [SCAFFOLD — Phase 3].

Encode item photos into vectors (CLIP or a ResNet) and use them like the text
embeddings — most valuable for cold-start items that have a photo but no
interactions yet. The mechanics are IDENTICAL to text embeddings; only the
encoder changes. That's the point: once you've done one modality, adding another
is the same pattern.

Data:
  * Real Yelp: the separate ``yelp_photos`` download -> data/raw/photos/.
    Map photo -> business_id via photos.json, then encode with CLIP.
  * Synthetic: ``Dataset.item_image_vectors`` already provides stand-in image
    vectors so you can build the *plumbing* before wrangling real images.

Suggested implementation:
  1. item_img_emb = CLIP.encode_image(photos)  (or use Dataset.item_image_vectors)
  2. user vector = mean of image embeddings of liked items
  3. score = cosine(user_vec, item_img_emb); top-K
  4. Compare cold-start Recall@K vs. text-only and vs. ALS.
  5. Best practice: fuse image + text + id features in the two-tower item tower
     rather than using images alone.

References: He & McAuley 2016 (VBPR); Radford et al. 2021 (CLIP).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import Recommender


class ImageEmbeddingRecommender(Recommender):
    name = "multimodal"

    def __init__(self, item_image_vectors: Optional[np.ndarray] = None):
        """``item_image_vectors`` is (n_items, dim), e.g. Dataset.item_image_vectors."""
        self.item_image_vectors = item_image_vectors

    def fit(self, train: pd.DataFrame) -> "ImageEmbeddingRecommender":
        raise NotImplementedError(
            "Multimodal is a Phase-3 exercise. Reuse the text-embedding recipe with "
            "image vectors (Dataset.item_image_vectors for synthetic, CLIP on Yelp "
            "photos for real). Focus on cold-start lift. See the docstring."
        )

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError
