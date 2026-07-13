"""Multimodal (image) recommendation from item image embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RAW_DIR, settings
from .base import IndexedRecommender


def load_item_image_vectors(
    items: pd.DataFrame,
    vectors: Optional[np.ndarray] = None,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, List[str], str]:
    """Resolve (n_items, dim) image embeddings aligned to *items* rows."""
    i = settings.item_col
    item_ids = items[i].astype(str).tolist()

    if vectors is not None:
        if len(vectors) != len(items):
            raise ValueError(
                f"item_image_vectors length {len(vectors)} != items {len(items)}"
            )
        emb = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize_rows(emb), item_ids, "provided"

    npy_path = Path(processed_dir) / "item_image_vectors.npy"
    if npy_path.exists():
        emb = np.load(npy_path).astype(np.float32)
        if len(emb) != len(items):
            raise ValueError(
                f"{npy_path} has {len(emb)} rows but items has {len(items)}"
            )
        return _l2_normalize_rows(emb), item_ids, f"file:{npy_path.name}"

    raise ValueError(
        "No image vectors available. Use synthetic data, run "
        "scripts/build_image_vectors.py on Yelp photos, or pass "
        "item_image_vectors=Dataset.item_image_vectors."
    )


def build_image_vectors_from_photos(
    items: pd.DataFrame,
    photos_dir: Path = RAW_DIR / "photos",
    photos_json: str = "photos.json",
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode Yelp business photos with CLIP; mean-pool per business_id.

    Expects ``photos_dir/photos.json`` plus image files under ``photos_dir/photos/``.
    Businesses without photos get a zero vector (handled at recommend time).
    """
    photos_dir = Path(photos_dir)
    meta_path = photos_dir / photos_json
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Download yelp_photos.tar and extract to data/raw/photos/."
        )

    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError(
            "Photo encoding needs torch, Pillow, and transformers. "
            "Install with: pip install transformers Pillow"
        ) from exc

    i = settings.item_col
    item_ids = items[i].astype(str).tolist()
    id_to_row = {iid: n for n, iid in enumerate(item_ids)}

    photo_paths: Dict[str, List[Path]] = {iid: [] for iid in item_ids}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bid = str(row.get("business_id", ""))
            if bid not in id_to_row:
                continue
            photo_id = row.get("photo_id")
            if not photo_id:
                continue
            for ext in (".jpg", ".jpeg", ".png"):
                p = photos_dir / "photos" / f"{photo_id}{ext}"
                if p.exists():
                    photo_paths[bid].append(p)
                    break

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    dim = model.config.projection_dim
    emb = np.zeros((len(item_ids), dim), dtype=np.float32)
    counts = np.zeros(len(item_ids), dtype=np.int32)

    for bid, paths in photo_paths.items():
        if not paths:
            continue
        row = id_to_row[bid]
        vecs = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            vecs.append(feats.cpu().numpy())
        mean = np.concatenate(vecs, axis=0).mean(axis=0)
        nrm = np.linalg.norm(mean)
        emb[row] = mean / nrm if nrm > 0 else mean
        counts[row] = len(paths)

    print(
        f"CLIP image vectors: {int((counts > 0).sum()):,}/{len(item_ids):,} "
        f"items have >=1 photo (dim={dim})."
    )
    return emb


def save_item_image_vectors(
    vectors: np.ndarray, processed_dir: Path = PROCESSED_DIR
) -> Path:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "item_image_vectors.npy"
    np.save(out, vectors.astype(np.float32))
    print(f"Wrote {out} shape={vectors.shape}")
    return out


def _l2_normalize_rows(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return emb / norms


class ImageEmbeddingRecommender(IndexedRecommender):
    """Content-style recommender using image (CLIP) embeddings instead of text."""

    name = "image_embedding"

    def __init__(
        self,
        item_image_vectors: Optional[np.ndarray] = None,
        items: Optional[pd.DataFrame] = None,
        positive_threshold: float | None = None,
        seed: int | None = None,
    ):
        self.item_image_vectors = item_image_vectors
        self.items = items
        self.positive_threshold = (
            settings.positive_threshold if positive_threshold is None else positive_threshold
        )
        self.seed = settings.seed if seed is None else seed
        self._item_emb: Optional[np.ndarray] = None
        self._user_emb: Dict[str, np.ndarray] = {}
        self.backend: str = ""
        self._all_item_ids: List[str] = []

    def fit(
        self,
        train: pd.DataFrame,
        items: pd.DataFrame | None = None,
        item_image_vectors: np.ndarray | None = None,
    ) -> "ImageEmbeddingRecommender":
        items = items if items is not None else self.items
        if items is None:
            raise ValueError(
                "ImageEmbeddingRecommender needs items=Dataset.items and "
                "item_image_vectors (or item_image_vectors.npy in processed/)."
            )
        self.items = items
        vectors = (
            item_image_vectors
            if item_image_vectors is not None
            else self.item_image_vectors
        )
        emb, item_ids, backend = load_item_image_vectors(items, vectors=vectors)
        self.backend = backend
        self._item_emb = emb
        self._all_item_ids = item_ids
        self._item_id_to_row = {iid: n for n, iid in enumerate(item_ids)}
        self._build_index(train)

        self.item_ids = item_ids
        self.item_to_idx = dict(self._item_id_to_row)
        self.idx_to_item = np.array(item_ids)
        self._train_item_ids = set(train[settings.item_col].astype(str).unique())

        u, i, r = settings.user_col, settings.item_col, settings.rating_col
        pos = train[train[r] >= self.positive_threshold] if r in train.columns else train
        self._user_emb = {}
        for uid, grp in pos.groupby(u):
            rows = [
                self._item_id_to_row[str(iid)]
                for iid in grp[i].tolist()
                if str(iid) in self._item_id_to_row
            ]
            if not rows:
                continue
            vec = emb[rows].mean(axis=0)
            nrm = np.linalg.norm(vec)
            self._user_emb[uid] = vec / nrm if nrm > 0 else vec

        pop = train.groupby(i).size()
        self._pop_order = [str(x) for x in pop.sort_values(ascending=False).index.tolist()]
        return self

    def cold_item_ids(self) -> List[str]:
        train_items = getattr(self, "_train_item_ids", set())
        return [iid for iid in self._all_item_ids if iid not in train_items]

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        if self._item_emb is None:
            raise RuntimeError("Call fit() before recommend().")
        out: Dict[str, List[str]] = {}
        for user_id in users:
            uvec = self._user_emb.get(user_id)
            if uvec is None:
                seen = self.seen.get(user_id, set()) if exclude_seen else set()
                out[user_id] = [i for i in self._pop_order if i not in seen][:k]
                continue
            scores = self._item_emb @ uvec
            out[user_id] = self._top_k_from_scores(user_id, scores.copy(), k, exclude_seen)
        return out

    def score_candidates(
        self, candidates: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        if self._item_emb is None:
            raise RuntimeError("Call fit() before score_candidates().")
        out: Dict[str, Dict[str, float]] = {}
        for user_id, items in candidates.items():
            uvec = self._user_emb.get(user_id)
            if uvec is None:
                out[user_id] = {str(it): 0.0 for it in items}
                continue
            scores: Dict[str, float] = {}
            for item_id in items:
                row = self._item_id_to_row.get(str(item_id))
                scores[str(item_id)] = (
                    float(self._item_emb[row] @ uvec) if row is not None else 0.0
                )
            out[user_id] = scores
        return out
