"""Synthetic Yelp-like data generator.

Produces a small dataset with the *same schema* as the real Yelp subset so the
entire pipeline runs before you download anything. It intentionally includes:

* implicit/explicit interactions with **timestamps** (for temporal split + sequential models)
* a **social graph** with homophily (friends tend to share taste) *plus noise*
* per-item **text** (for the text/LLM module)
* per-item **image-stand-in vectors** (for the multimodal module)

Caveat for learning: because we *build* the social graph partly from latent
taste here, this synthetic social signal is fine for exercising the *plumbing*
but is NOT a valid test of "does social help?" — use the real Yelp friend graph
for that honest experiment (see data/README.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, settings

_CATEGORY_WORDS = [
    "coffee", "sushi", "tacos", "ramen", "pizza", "vegan", "bakery", "bbq",
    "brunch", "cocktails", "wine", "burgers", "thai", "indian", "seafood",
    "dessert", "steakhouse", "noodles", "sandwich", "breakfast",
]


def generate(
    n_users: int = 800,
    n_items: int = 400,
    n_factors: int = 16,
    avg_interactions_per_user: int = 25,
    avg_friends_per_user: int = 6,
    image_dim: int = 32,
    seed: int | None = None,
    write: bool = True,
) -> dict:
    """Generate a synthetic dataset and (optionally) write parquet to processed/.

    Returns a dict of DataFrames/arrays; when ``write`` is True they are also
    saved to ``data/processed/`` where :func:`recsys.data.loaders.load_dataset`
    can pick them up.
    """
    rng = np.random.default_rng(settings.seed if seed is None else seed)

    # --- latent factors drive everything (taste, text, images) -----------
    user_factors = rng.normal(size=(n_users, n_factors))
    item_factors = rng.normal(size=(n_items, n_factors))

    user_ids = np.array([f"u_{i:05d}" for i in range(n_users)])
    item_ids = np.array([f"b_{i:05d}" for i in range(n_items)])

    # --- interactions: sample items each user is likely to engage with ----
    affinity = user_factors @ item_factors.T  # (n_users, n_items)
    # Softmax per user to get a sampling distribution over items.
    probs = np.exp(affinity - affinity.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    base_time = np.datetime64("2021-01-01")
    rows = []
    for u in range(n_users):
        k = max(1, int(rng.poisson(avg_interactions_per_user)))
        k = min(k, n_items)
        chosen = rng.choice(n_items, size=k, replace=False, p=probs[u])
        for it in chosen:
            # Rating correlates with affinity, clipped to 1..5 stars.
            score = affinity[u, it]
            rating = int(np.clip(round(3 + score), 1, 5))
            day = int(rng.integers(0, 900))
            ts = base_time + np.timedelta64(day, "D")
            rows.append((user_ids[u], item_ids[it], rating, ts))

    interactions = pd.DataFrame(
        rows,
        columns=[
            settings.user_col,
            settings.item_col,
            settings.rating_col,
            settings.time_col,
        ],
    )
    interactions[settings.time_col] = pd.to_datetime(interactions[settings.time_col])

    # --- social graph: homophily on latent taste + noise ------------------
    sim = user_factors @ user_factors.T
    np.fill_diagonal(sim, -np.inf)
    social_edges = set()
    for u in range(n_users):
        n_friends = max(0, int(rng.poisson(avg_friends_per_user)))
        if n_friends == 0:
            continue
        # 70% taste-similar friends, 30% random (keeps it from being a pure leak)
        n_sim = int(round(0.7 * n_friends))
        top = np.argsort(sim[u])[::-1][: max(n_sim * 3, 1)]
        sim_friends = rng.choice(top, size=min(n_sim, len(top)), replace=False)
        rand_friends = rng.integers(0, n_users, size=n_friends - n_sim)
        for v in np.concatenate([sim_friends, rand_friends]):
            if v == u:
                continue
            a, b = sorted((u, int(v)))
            social_edges.add((a, b))

    social = pd.DataFrame(
        [(user_ids[a], user_ids[b]) for a, b in social_edges],
        columns=[settings.user_col, "friend_id"],
    )

    # --- item text (pseudo review/categories) + images -------------------
    item_categories = []
    item_text = []
    for it in range(n_items):
        # Pick categories weighted by the item's dominant latent dims.
        w = np.abs(item_factors[it, : len(_CATEGORY_WORDS)]) if n_factors >= len(_CATEGORY_WORDS) else np.abs(rng.normal(size=len(_CATEGORY_WORDS)))
        w = w / w.sum()
        cats = rng.choice(_CATEGORY_WORDS, size=3, replace=False, p=w)
        item_categories.append(", ".join(cats))
        item_text.append(
            f"A popular spot known for {cats[0]} and {cats[1]}. "
            f"Guests love the {cats[2]}."
        )

    items = pd.DataFrame(
        {
            settings.item_col: item_ids,
            "name": [f"Business {i}" for i in range(n_items)],
            "categories": item_categories,
            "text": item_text,
        }
    )
    users = pd.DataFrame({settings.user_col: user_ids})

    # Image-stand-in vectors: item latent factors projected + noise.
    proj = rng.normal(size=(n_factors, image_dim))
    item_image_vectors = item_factors @ proj + 0.1 * rng.normal(size=(n_items, image_dim))
    item_image_vectors = item_image_vectors.astype(np.float32)

    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        interactions.to_parquet(PROCESSED_DIR / "interactions.parquet", index=False)
        users.to_parquet(PROCESSED_DIR / "users.parquet", index=False)
        items.to_parquet(PROCESSED_DIR / "items.parquet", index=False)
        social.to_parquet(PROCESSED_DIR / "social.parquet", index=False)
        np.save(PROCESSED_DIR / "item_image_vectors.npy", item_image_vectors)

    return {
        "interactions": interactions,
        "users": users,
        "items": items,
        "social": social,
        "item_image_vectors": item_image_vectors,
    }


if __name__ == "__main__":  # pragma: no cover
    out = generate()
    print("Wrote synthetic dataset to", PROCESSED_DIR)
    for name, obj in out.items():
        if hasattr(obj, "shape"):
            print(f"  {name}: shape={obj.shape}")
