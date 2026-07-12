# Data

## Option A — Synthetic (default, zero download)

You don't need anything here to get started. The synthetic generator
(`recsys.data.synthetic`) writes a small Yelp-like dataset into
`data/processed/` on demand, including a social graph, text, and stand-in image
vectors. Just run `python scripts/demo.py`.

## Option B — Real Yelp Open Dataset

1. Go to the Yelp Open Dataset page: https://www.yelp.com/dataset
2. Accept the terms and download the dataset (JSON format).
3. Extract the files into `data/raw/` so you have:

```
data/raw/
├── yelp_academic_dataset_user.json      # users + friends (social graph)
├── yelp_academic_dataset_business.json   # items + categories + location
├── yelp_academic_dataset_review.json     # interactions + text + timestamps
├── yelp_academic_dataset_tip.json        # (optional)
└── yelp_academic_dataset_checkin.json    # (optional)
```

4. Photos (for the multimodal module) are a **separate** download on the same
   page (`yelp_photos.tar`). Extract to `data/raw/photos/` if you want images.

5. Carve a small, dense slice so iteration is fast:

```bash
python scripts/make_subset.py \
    --city "Santa Barbara" \
    --min-user-reviews 10 \
    --min-item-reviews 10
```

This writes tidy parquet files to `data/processed/` with the **same schema** the
loaders expect, so every model works identically on synthetic or real data.

## Why subset?

The full Yelp dataset is several GB and millions of reviews. A single dense
metro area with active users keeps the interaction matrix small enough for fast
laptop iteration while preserving real social edges, text, and (optionally)
photos — the whole point of using a unified dataset.

> Both `data/raw/` and `data/processed/` are gitignored. Never commit the dataset.
