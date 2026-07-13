"""Per-model benchmark cache — skip expensive fit+recommend on reruns."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import ARTIFACTS_DIR, settings


def dataset_fingerprint(processed_dir: Path | None = None) -> str:
    """Stable id for the active processed parquet set (size + mtime)."""
    processed_dir = Path(processed_dir or settings.paths["processed"])
    parts = []
    for name in ("interactions", "users", "items", "social"):
        path = processed_dir / f"{name}.parquet"
        if path.exists():
            st = path.stat()
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime_ns)}")
    if not parts:
        return "empty"
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _config_hash(model_config: Any) -> str:
    blob = json.dumps(model_config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_cache_dir(
    *,
    cutoff_quantile: float,
    k: int,
    processed_dir: Path | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Shared cache bucket for one dataset + split + eval-k."""
    base_dir = Path(base_dir or ARTIFACTS_DIR / "benchmark_cache")
    ds = dataset_fingerprint(processed_dir)
    return base_dir / f"{ds}_q{cutoff_quantile}_k{k}"


def model_cache_path(
    cache_dir: Path,
    model_name: str,
    model_config: Any,
) -> Path:
    return cache_dir / f"{model_name}_{_config_hash(model_config)}.json"


def load_model_cache(
    cache_dir: Path, model_name: str, model_config: Any
) -> Optional[dict]:
    path = model_cache_path(cache_dir, model_name, model_config)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_model_cache(
    cache_dir: Path,
    model_name: str,
    model_config: Any,
    *,
    recs: Dict[str, list],
    metrics: dict,
    time_s: float,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "config_hash": _config_hash(model_config),
        "recs": {str(u): [str(i) for i in items] for u, items in recs.items()},
        "metrics": metrics,
        "time_s": time_s,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    path = model_cache_path(cache_dir, model_name, model_config)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(path)

    manifest_path = cache_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    key = f"{model_name}_{_config_hash(model_config)}"
    manifest.setdefault("models", {})[key] = {
        "time_s": time_s,
        "cached_at": payload["cached_at"],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
