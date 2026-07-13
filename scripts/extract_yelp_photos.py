#!/usr/bin/env python
"""Extract Yelp photo JPEGs needed for items in data/processed/.

Reads ``photos.json`` under ``--photos-dir``, finds photo_ids for businesses
in ``items.parquet``, then pulls only those paths from ``yelp_photos.tar``.

    python scripts/extract_yelp_photos.py \\
        --tar ~/Downloads/Yelp\\ Photos/yelp_photos.tar
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.recsys.config import PROCESSED_DIR, RAW_DIR


def photo_ids_for_items(
    items: pd.DataFrame, photos_json: Path, item_col: str = "item_id"
) -> list[str]:
    item_ids = set(items[item_col].astype(str))
    photo_ids: list[str] = []
    with open(photos_json, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("business_id", "")) not in item_ids:
                continue
            pid = row.get("photo_id")
            if pid:
                photo_ids.append(str(pid))
    return list(dict.fromkeys(photo_ids))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--photos-dir", default=str(RAW_DIR / "photos"))
    p.add_argument(
        "--tar",
        default=str(RAW_DIR / "photos" / "yelp_photos.tar"),
        help="Path to yelp_photos.tar (default: data/raw/photos/yelp_photos.tar)",
    )
    args = p.parse_args()

    processed = Path(args.processed_dir)
    photos_dir = Path(args.photos_dir)
    tar_path = Path(args.tar)
    photos_json = photos_dir / "photos.json"

    if not photos_json.exists():
        raise FileNotFoundError(
            f"Missing {photos_json}. Extract metadata first:\n"
            f"  tar -xf {tar_path} -C {photos_dir} photos.json"
        )
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing {tar_path}")

    items = pd.read_parquet(processed / "items.parquet")
    photo_ids = photo_ids_for_items(items, photos_json)
    if not photo_ids:
        print("No photos matched items in the processed slice.")
        return

    paths_file = photos_dir / "_needed_paths.txt"
    paths_file.write_text(
        "\n".join(f"photos/{pid}.jpg" for pid in photo_ids) + "\n"
    )
    out_dir = photos_dir / "photos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Extracting {len(photo_ids):,} photos for {len(items):,} items "
        f"from {tar_path}..."
    )
    subprocess.run(
        [
            "tar",
            "-xf",
            str(tar_path),
            "-C",
            str(photos_dir),
            "--files-from",
            str(paths_file),
        ],
        check=True,
    )
    n_jpg = len(list(out_dir.glob("*.jpg")))
    print(f"Done: {n_jpg:,} JPEGs in {out_dir}")


if __name__ == "__main__":
    main()
