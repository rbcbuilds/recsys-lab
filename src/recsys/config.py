"""Global paths and settings for recsys-lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo root is two levels up from this file: src/recsys/config.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class Settings:
    """Tunable knobs shared across the lab."""

    # Reproducibility
    seed: int = 42

    # Evaluation
    top_k: int = 10
    eval_ks: tuple[int, ...] = (5, 10, 20)
    # Fraction of each user's most recent interactions held out for testing.
    test_fraction: float = 0.2
    # A rating >= this counts as a positive (relevant) interaction.
    positive_threshold: float = 4.0

    # Column names used by every loader / model (schema contract)
    user_col: str = "user_id"
    item_col: str = "item_id"
    rating_col: str = "rating"
    time_col: str = "timestamp"

    paths: dict = field(
        default_factory=lambda: {
            "raw": RAW_DIR,
            "processed": PROCESSED_DIR,
            "artifacts": ARTIFACTS_DIR,
        }
    )


settings = Settings()

for _p in (DATA_DIR, RAW_DIR, PROCESSED_DIR):
    _p.mkdir(parents=True, exist_ok=True)
