#!/usr/bin/env python
"""Build CLIP image vectors for items in data/processed/.

Requires Yelp photos extracted to data/raw/photos/ (photos.json + photos/).

    python scripts/build_image_vectors.py
    python scripts/build_image_vectors.py --processed-dir data/processed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.recsys.config import PROCESSED_DIR, RAW_DIR
from src.recsys.models.multimodal import (
    build_image_vectors_from_photos,
    save_item_image_vectors,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--photos-dir", default=str(RAW_DIR / "photos"))
    args = p.parse_args()
    processed = Path(args.processed_dir)
    items = pd.read_parquet(processed / "items.parquet")
    vectors = build_image_vectors_from_photos(
        items, photos_dir=Path(args.photos_dir)
    )
    save_item_image_vectors(vectors, processed_dir=processed)


if __name__ == "__main__":
    main()
