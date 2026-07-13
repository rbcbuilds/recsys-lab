#!/usr/bin/env python
"""Cross-regime benchmark: one dataset, one split, all models, four views.

The active dataset should be the cross-regime Philadelphia slice (dense core +
real cold tail). Every model trains once on a global-time split, then the same
recommendations are scored four ways:

    overall    all test positives
    warm       user and item both seen in train
    cold-item  target item unseen in train
    cold-user  user unseen in train

Build the dataset:

    python scripts/make_crossregime.py
    cp data/processed_philly_xreg/*.parquet data/processed/

Run:

    python scripts/benchmark.py                    # core set (default, ~25–45 min)
    python scripts/benchmark.py --mode full        # all 13 models (~2+ hours)
    python scripts/benchmark.py --mode ab-sasrec   # unified with/without SASRec ranker feature

Cached reruns skip fit+recommend for models already in artifacts/benchmark_cache/.
Use --force to retrain everything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.recsys.config import settings
from src.recsys.data import load_dataset
from src.recsys.eval import evaluate, evaluate_ips, global_temporal_split, item_propensity
from src.recsys.eval.cache import (
    load_model_cache,
    run_cache_dir,
    save_model_cache,
)
from src.recsys.eval.slate import diversify_slate
from src.recsys.models import (
    ALSRecommender,
    BPRRecommender,
    ContentBasedRecommender,
    ContrastiveTwoTowerRecommender,
    HSTURecommender,
    ImageEmbeddingRecommender,
    ItemTokenLMRecommender,
    LightGCNRecommender,
    LLMTwoStageRecommender,
    MultiRetriever,
    PopularityRecommender,
    SASRecRecommender,
    SemanticIDRecommender,
    SocialRecommender,
    TwoStageRecommender,
    TwoTowerRecommender,
)
from src.recsys.models.text_embeddings import encode_item_text

REPORT_PATH = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"

SASREC_KWARGS = dict(dim=64, epochs=15, max_len=50)
ITEM_TOKEN_LM_KWARGS = dict(dim=64, epochs=12, max_len=50)
LLM_TWO_STAGE_CANDIDATES = 80

UNIFIED_KWARGS = dict(
    candidate_n=200,
    use_social=True,
    use_extended_features=True,
    sasrec_kwargs=SASREC_KWARGS,
)
LIGHTGCN_KWARGS = dict(dim=64, n_layers=2, epochs=10)
HSTU_KWARGS = dict(dim=64, epochs=12, max_len=50)
SEMANTIC_ID_KWARGS = dict(dim=64, epochs=10, n_clusters=32)
CONTRASTIVE_TT_KWARGS = dict(dim=64, epochs=10, hard_negative_k=5)
TWO_TOWER_KWARGS = dict(dim=64, epochs=10)


def _unified_retrievers(ds) -> list:
    retrievers = [
        TwoTowerRecommender(**TWO_TOWER_KWARGS),
        ContentBasedRecommender(items=ds.items),
        SocialRecommender(social=ds.social),
        PopularityRecommender(),
    ]
    if ds.item_image_vectors is not None:
        retrievers.insert(
            2,
            ImageEmbeddingRecommender(
                items=ds.items, item_image_vectors=ds.item_image_vectors
            ),
        )
    return retrievers


def build_unified_two_stage(ds, use_sasrec: bool = False) -> TwoStageRecommender:
    """Multi-retriever pipeline meant to span warm, cold-item, and cold-user."""
    return TwoStageRecommender(
        MultiRetriever(_unified_retrievers(ds)),
        candidate_n=UNIFIED_KWARGS["candidate_n"],
        use_social=UNIFIED_KWARGS["use_social"],
        social=ds.social,
        use_sasrec=use_sasrec,
        sasrec_kwargs=UNIFIED_KWARGS["sasrec_kwargs"] if use_sasrec else None,
        use_extended_features=UNIFIED_KWARGS["use_extended_features"],
    )


TIER3_MODELS = (
    "hstu",
    "semantic_id_lm",
    "contrastive_two_tower",
)


def _fit_model(model, train, ds) -> None:
    if getattr(model, "name", "") == "semantic_id_lm":
        model.fit(train, items=ds.items)
    else:
        model.fit(train)


def _item_embedding_dict(ds) -> dict:
    emb, item_ids, _ = encode_item_text(ds.items)
    return {iid: emb[row] for row, iid in enumerate(item_ids)}


# Curated default set — see CORE_MODELS below. Ranked on philly_xreg_fast + enriched
# text + images (Jul 2026); times are fresh-run seconds on laptop CPU.
#
#  rank  model                  time_s  overall  warm   cold_item  cold_user  notes
#  ----  ---------------------  ------  -------  -----  ---------  ---------  -----
#   1    popularity               0.1   0.061   0.053   0.000      0.090    baseline
#   2    item_token_lm           88.0   0.057   0.048   0.000      0.090    generative
#   3    two_stage_unified     333–1130 0.048–0.062 0.049–0.059 0.000 0.052–0.077  headline
#   4    image_embedding          0.5   0.047   0.034   0.000      0.090    multimodal
#   5    sasrec                  99.0   0.030   0.042   0.000      0.000    sequential
#   6    hstu                   170.0   0.031   0.044   0.000      0.000    core (Tier 3 seq)
#   7    semantic_id_lm         190.0   0.030   0.011   0.008      0.090    only cold-item lift
#   8    two_stage               12.0   0.032   0.045   0.000      0.000    unified dominates
#   9    bpr                      0.8   0.025   0.035   0.000      0.000    weak MF
#  10    als                      0.7   0.022   0.031   0.000      0.000    weak MF
#  11    contrastive_two_tower  373.0   0.017   0.025   0.000      0.000    core (Tier 3)
#  12    two_stage_llm          843.0   0.008   0.011   0.000      0.000    skip (cost)
#  13    lightgcn                10.0   0.004   0.006   0.000      0.000    core (Tier 2)

# Default benchmark: headline architecture + one model per tier/signal (~25–45 min).
CORE_MODELS = (
    "popularity",              # instant baseline; cold-user floor
    "sasrec",                  # warm sequential reference
    "hstu",                    # time-aware sequential (Tier 3)
    "item_token_lm",           # strong overall + cold-user generative
    "lightgcn",                # Tier 2 graph CF
    "contrastive_two_tower",   # Tier 3 hard-negative two-tower
    "semantic_id_lm",          # Tier 3 compositional tokens; cold-item lift
    "two_stage_unified",       # MultiRetriever → ranker (headline)
)

# Smoke test (~5 min): baseline + headline only.
FAST_MODELS = (
    "popularity",
    "two_stage_unified",
)


def _has_image_vectors() -> bool:
    return (settings.paths["processed"] / "item_image_vectors.npy").exists()


def model_configs_for_mode(mode: str) -> dict:
    if mode == "ab-sasrec":
        return {
            "two_stage_unified": {"use_sasrec": False, **UNIFIED_KWARGS},
            "two_stage_unified_sasrec": {"use_sasrec": True, **UNIFIED_KWARGS},
        }
    configs = {
        "popularity": {},
        "als": {"factors": 64, "iterations": 15},
        "bpr": {"factors": 64, "iterations": 80},
        "sasrec": SASREC_KWARGS,
        "lightgcn": LIGHTGCN_KWARGS,
        "hstu": HSTU_KWARGS,
        "semantic_id_lm": SEMANTIC_ID_KWARGS,
        "contrastive_two_tower": CONTRASTIVE_TT_KWARGS,
        "item_token_lm": ITEM_TOKEN_LM_KWARGS,
        "two_stage": {"candidate_n": 200, "use_social": False, **TWO_TOWER_KWARGS},
        "two_stage_llm": {
            "candidate_n": LLM_TWO_STAGE_CANDIDATES,
            **TWO_TOWER_KWARGS,
        },
        "two_stage_unified": {"use_sasrec": False, **UNIFIED_KWARGS},
    }
    if _has_image_vectors():
        configs["image_embedding"] = {}
    if mode == "tier3":
        return {k: configs[k] for k in TIER3_MODELS if k in configs}
    if mode in ("core", "fast"):
        keys = [k for k in (CORE_MODELS if mode == "core" else FAST_MODELS) if k in configs]
        if "image_embedding" in configs and "image_embedding" not in keys:
            if mode == "core":
                insert_at = keys.index("item_token_lm") + 1 if "item_token_lm" in keys else len(keys)
                keys.insert(insert_at, "image_embedding")
            else:
                keys.append("image_embedding")
        return {k: configs[k] for k in keys}
    return configs


def _tier3_models(ds) -> dict:
    return {
        "hstu": HSTURecommender(**HSTU_KWARGS),
        "semantic_id_lm": SemanticIDRecommender(**SEMANTIC_ID_KWARGS),
        "contrastive_two_tower": ContrastiveTwoTowerRecommender(**CONTRASTIVE_TT_KWARGS),
    }


def _tier2_models(ds) -> dict:
    models = {"lightgcn": LightGCNRecommender(**LIGHTGCN_KWARGS)}
    if ds.item_image_vectors is not None:
        models["image_embedding"] = ImageEmbeddingRecommender(
            items=ds.items, item_image_vectors=ds.item_image_vectors
        )
    return models


def build_models(ds, mode: str = "full"):
    """Model set for the requested benchmark mode."""
    if mode == "ab-sasrec":
        return {
            "two_stage_unified": build_unified_two_stage(ds, use_sasrec=False),
            "two_stage_unified_sasrec": build_unified_two_stage(ds, use_sasrec=True),
        }
    base = {
        "popularity": PopularityRecommender(),
        "als": ALSRecommender(factors=64, iterations=15),
        "bpr": BPRRecommender(factors=64, iterations=80),
        "sasrec": SASRecRecommender(**SASREC_KWARGS),
        "item_token_lm": ItemTokenLMRecommender(**ITEM_TOKEN_LM_KWARGS),
        "two_stage": TwoStageRecommender(
            TwoTowerRecommender(**TWO_TOWER_KWARGS),
            candidate_n=200,
            use_social=False,
        ),
        "two_stage_llm": LLMTwoStageRecommender(
            retriever=TwoTowerRecommender(**TWO_TOWER_KWARGS),
            items=ds.items,
            candidate_n=LLM_TWO_STAGE_CANDIDATES,
        ),
        "two_stage_unified": build_unified_two_stage(ds, use_sasrec=False),
    }
    base.update(_tier2_models(ds))
    base.update(_tier3_models(ds))
    if mode == "tier3":
        return {k: base[k] for k in TIER3_MODELS if k in base}
    if mode in ("core", "fast"):
        keys = [k for k in (CORE_MODELS if mode == "core" else FAST_MODELS) if k in base]
        if "image_embedding" in base and "image_embedding" not in keys:
            if mode == "core":
                insert_at = keys.index("item_token_lm") + 1 if "item_token_lm" in keys else len(keys)
                keys.insert(insert_at, "image_embedding")
            else:
                keys.append("image_embedding")
        return {k: base[k] for k in keys}
    return base


def build_slices(ds, cutoff_quantile: float):
    """Global-time split; cold sets are test entities not seen in train."""
    u, i = settings.user_col, settings.item_col
    train, test_pos = global_temporal_split(ds.interactions, cutoff_quantile=cutoff_quantile)
    train_users = set(train[u].astype(str))
    train_items = set(train[i].astype(str))

    test_items, test_users = set(), set()
    for user_id, items in test_pos.items():
        test_users.add(str(user_id))
        test_items |= {str(x) for x in items}
    cold_items = test_items - train_items
    cold_users = test_users - train_users
    return train, test_pos, cold_items, cold_users


def slice_ground_truth(test_pos, cold_items, cold_users):
    """Partition test positives into warm / cold-item / cold-user (overlap dropped)."""
    cold_items = {str(x) for x in cold_items}
    cold_users = {str(x) for x in cold_users}
    warm, ci, cu = {}, {}, {}
    for user_id, items in test_pos.items():
        u = str(user_id)
        u_cold = u in cold_users
        for it in items:
            it = str(it)
            i_cold = it in cold_items
            if u_cold and i_cold:
                continue
            if u_cold:
                cu.setdefault(u, set()).add(it)
            elif i_cold:
                ci.setdefault(u, set()).add(it)
            else:
                warm.setdefault(u, set()).add(it)
    return warm, ci, cu


def _score_recs(recs, test_pos, warm_gt, ci_gt, cu_gt, k: int) -> dict:
    overall = evaluate(recs, test_pos, ks=(k, 10))
    warm = evaluate(recs, warm_gt, ks=(k,)) if warm_gt else {}
    ci = evaluate(recs, ci_gt, ks=(k,)) if ci_gt else {}
    cu = evaluate(recs, cu_gt, ks=(k,)) if cu_gt else {}
    return {
        f"overall_r@{k}": overall.get(f"recall@{k}", 0.0),
        f"warm_r@{k}": warm.get(f"recall@{k}", 0.0),
        f"cold_item_r@{k}": ci.get(f"recall@{k}", 0.0),
        f"cold_user_r@{k}": cu.get(f"recall@{k}", 0.0),
        "ndcg@10": overall.get("ndcg@10", 0.0),
    }


def _cache_key(model_name: str, model_config: dict) -> str:
    """Canonical cache identity (share hits across benchmark modes when config matches)."""
    if model_config.get("use_sasrec") is True and model_name in (
        "two_stage_unified",
        "two_stage_unified_sasrec",
    ):
        return "unified_sasrec_ranker"
    if model_config.get("use_sasrec") is False and model_name == "two_stage_unified":
        return "unified_no_sasrec"
    return model_name


def run_benchmark(
    ds,
    cutoff_quantile: float,
    k: int,
    models: dict,
    *,
    mode: str,
    use_cache: bool = True,
    force: bool = False,
    only: set[str] | None = None,
    diversify: bool = False,
    diversify_lambda: float = 0.7,
    report_ips: bool = False,
):
    train, test_pos, cold_items, cold_users = build_slices(ds, cutoff_quantile)
    warm_gt, ci_gt, cu_gt = slice_ground_truth(test_pos, cold_items, cold_users)
    users = sorted(set(warm_gt) | set(ci_gt) | set(cu_gt))
    split_desc = f"global-time split at q={cutoff_quantile}"

    counts = dict(
        n_cold_items=len(cold_items),
        n_cold_users=len(cold_users),
        warm_u=len(warm_gt),
        ci_u=len(ci_gt),
        cu_u=len(cu_gt),
    )

    cache_dir = run_cache_dir(
        cutoff_quantile=cutoff_quantile,
        k=k,
    )
    configs = model_configs_for_mode(mode)
    propensity = item_propensity(train) if report_ips else None
    item_emb = _item_embedding_dict(ds) if diversify else None

    print(f"split: {split_desc}", flush=True)
    print(
        f"cold items={counts['n_cold_items']:,} cold users={counts['n_cold_users']:,}; "
        f"train {len(train):,} rows",
        flush=True,
    )
    print(
        f"slice users: warm={counts['warm_u']:,} cold_item={counts['ci_u']:,} "
        f"cold_user={counts['cu_u']:,}",
        flush=True,
    )
    if use_cache:
        print(f"cache: {cache_dir} ({'force retrain' if force else 'reuse hits'})", flush=True)
    print(flush=True)

    rows = []
    t_all = time.time()
    for name, model in models.items():
        if only is not None and name not in only:
            continue

        cfg = configs[name]
        ckey = _cache_key(name, cfg)
        cached = None if (force or not use_cache) else load_model_cache(cache_dir, ckey, cfg)
        if cached is not None:
            recs = {u: list(items) for u, items in cached["recs"].items()}
            metrics = cached["metrics"]
            dt = float(cached.get("time_s", 0.0))
            tag = "cached"
        else:
            t = time.time()
            _fit_model(model, train, ds)
            recs = model.recommend(users, k=k * 5 if diversify else k)
            if diversify and item_emb:
                recs = diversify_slate(
                    recs, item_emb, k=k, method="mmr", lambda_=diversify_lambda
                )
            dt = time.time() - t
            metrics = _score_recs(recs, test_pos, warm_gt, ci_gt, cu_gt, k)
            if report_ips and propensity:
                ips = evaluate_ips(recs, test_pos, propensity, k=k)
                metrics[f"ips_recall@{k}"] = ips[f"ips_recall@{k}"]
            if use_cache:
                save_model_cache(
                    cache_dir,
                    ckey,
                    cfg,
                    recs=recs,
                    metrics=metrics,
                    time_s=round(dt, 1),
                )
            tag = f"{dt:.1f}s"

        row = {"model": name, **metrics, "time_s": round(dt, 1)}
        rows.append(row)
        if name == "two_stage_unified" and hasattr(model, "ranker"):
            try:
                imp = model.ranker.feature_importance()
                imp_path = (
                    settings.paths["processed"] / "ranker_feature_importance.json"
                )
                imp_path.parent.mkdir(parents=True, exist_ok=True)
                imp_path.write_text(json.dumps(imp, indent=2))
                print(f"  ranker feature importance -> {imp_path}", flush=True)
            except Exception:
                pass
        print(
            f"{name:28s} overall={row[f'overall_r@{k}']:.4f} "
            f"warm={row[f'warm_r@{k}']:.4f} "
            f"cold_item={row[f'cold_item_r@{k}']:.4f} "
            f"cold_user={row[f'cold_user_r@{k}']:.4f} "
            + (
                f"ips={row.get(f'ips_recall@{k}', 0.0):.4f} "
                if report_ips and f"ips_recall@{k}" in row
                else ""
            )
            + f"({tag})",
            flush=True,
        )

    total = time.time() - t_all
    df = pd.DataFrame(rows).set_index("model")
    return df, split_desc, counts, total


def _render_table(df, k: int) -> str:
    cols = [
        f"overall_r@{k}",
        f"warm_r@{k}",
        f"cold_item_r@{k}",
        f"cold_user_r@{k}",
        "ndcg@10",
    ]
    header = "| model | " + " | ".join(cols) + " | fit+rec (s) |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [header, sep]
    for name in df.index:
        cells = " | ".join(f"{df.loc[name, c]:.4f}" for c in cols)
        lines.append(f"| {name} | {cells} | {df.loc[name, 'time_s']:.1f} |")
    return "\n".join(lines)


HOW_TO_READ = """## How to read this

- **overall_r@20** — recall across all test positives (the blended score).
- **warm_r@20** — dense regime: user and item both seen in train (collaborative signal).
- **cold_item_r@20** — target item never seen in train (content signal).
- **cold_user_r@20** — user never seen in train (social / popularity signal).
- **ndcg@10** — ranking quality on the full test set.

Collaborative models (als, bpr, sasrec, lightgcn, two_stage) should be strong on warm and
near-zero on both cold slices. Popularity lifts cold-user. Image embeddings help
cold-item when `item_image_vectors.npy` is present. The unified two-stage
(`MultiRetriever` → ranker) is the architecture meant to stay non-zero across all
three regimes. **Generative / language models:** `item_token_lm` autoregressively
generates item tokens; `two_stage_llm` re-ranks two-tower candidates with a
cross-encoder (candidate pool=80). **Tier 3:** `hstu` adds time-aware sequential
signal; `semantic_id_lm` uses compositional item codes; `contrastive_two_tower`
trains with popularity hard negatives. Pass ``--diversify`` for MMR post-ranking
and ``--ips`` for propensity-weighted recall. Cold-item recall stays
low on Yelp because item text is only name + categories — see `docs/techniques.md`.
"""

REPORT_INTRO = """# Benchmark results

Auto-generated by `scripts/benchmark.py`. One realistic dataset (dense core +
cold tail), one global-time split, every model scored on **overall**, **warm**,
**cold-item**, and **cold-user**. Reproduce with:

```bash
python scripts/benchmark.py
python scripts/benchmark.py --mode full
python scripts/benchmark.py --mode ab-sasrec
```

Cached per-model results live under `artifacts/benchmark_cache/`; pass `--force`
to retrain everything.
"""

FULL_SECTION = "Cross-regime Philadelphia"
AB_SECTION = "SASRec ranker feature A/B (unified pipeline)"


def _parse_sections(existing: str) -> dict[str, str]:
    titles = (FULL_SECTION, AB_SECTION)
    out: dict[str, str] = {}
    for title in titles:
        pattern = rf"## {re.escape(title)}\n\n(.*?)(?=\n## |\Z)"
        m = re.search(pattern, existing, flags=re.DOTALL)
        if m:
            out[title] = m.group(1).strip()
    return out


def write_report(
    df, ds, k: int, split_desc: str, counts: dict, total: float, mode: str
) -> None:
    section_title = AB_SECTION if mode == "ab-sasrec" else FULL_SECTION
    mode_note = (
        "**Comparison:** `two_stage_unified` vs `two_stage_unified_sasrec` — "
        "identical `MultiRetriever`, ranker gains `sasrec_score` in the second row.\n\n"
        if mode == "ab-sasrec"
        else ""
    )

    section_body = (
        f"One model, one training set, four views. Cold entities arise naturally under a\n"
        f"single wall-clock cutoff — not simulated. See `docs/techniques.md`.\n\n"
        f"{mode_note}"
        f"- **Dataset:** {ds.summary()}\n"
        f"- **Split:** {split_desc}\n"
        f"- **Cold sets:** {counts['n_cold_items']:,} cold items, {counts['n_cold_users']:,} cold users\n"
        f"- **Slice users:** warm={counts['warm_u']:,}, cold-item={counts['ci_u']:,}, cold-user={counts['cu_u']:,}\n"
        f"- **Run date:** {date.today().isoformat()}\n"
        f"- **Total run time:** {total:.0f}s (single-thread BLAS, laptop CPU)\n\n"
        f"{_render_table(df, k)}"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections = _parse_sections(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    sections[section_title] = section_body

    parts = [REPORT_INTRO]
    if FULL_SECTION in sections:
        parts.append(f"## {FULL_SECTION}\n\n{sections[FULL_SECTION]}")
    if AB_SECTION in sections:
        parts.append(f"## {AB_SECTION}\n\n{sections[AB_SECTION]}")
    parts.append(HOW_TO_READ)
    REPORT_PATH.write_text("\n".join(parts) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["core", "full", "fast", "tier3", "ab-sasrec"],
        default="core",
        help="core (default) = 9–10 curated models (~25–45 min); full = all 13 (~2+ h); "
        "fast = popularity + unified smoke (~5 min); tier3 = hstu + semantic_id_lm "
        "+ contrastive_two_tower; ab-sasrec = unified SASRec ranker A/B.",
    )
    ap.add_argument("--cutoff-quantile", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=max(settings.eval_ks))
    ap.add_argument("--no-write", action="store_true", help="Print only; skip benchmarks.md.")
    ap.add_argument("--no-cache", action="store_true", help="Disable per-model cache.")
    ap.add_argument("--force", action="store_true", help="Ignore cache and retrain all models.")
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated model keys to run (others skipped).",
    )
    ap.add_argument(
        "--diversify",
        action="store_true",
        help="Apply MMR slate diversification before scoring.",
    )
    ap.add_argument(
        "--diversify-lambda",
        type=float,
        default=0.7,
        help="MMR relevance/diversity tradeoff (1.0 = pure relevance).",
    )
    ap.add_argument(
        "--ips",
        action="store_true",
        help="Also report IPS-weighted recall@K.",
    )
    args = ap.parse_args()

    ds = load_dataset()
    print("DATA:", ds.summary(), flush=True)

    models = build_models(ds, mode=args.mode)
    only = set(args.only.split(",")) if args.only else None
    df, split_desc, counts, total = run_benchmark(
        ds,
        args.cutoff_quantile,
        args.k,
        models,
        mode=args.mode,
        use_cache=not args.no_cache,
        force=args.force,
        only=only,
        diversify=args.diversify,
        diversify_lambda=args.diversify_lambda,
        report_ips=args.ips,
    )

    print(f"\n=== CROSS-REGIME BENCHMARK (recall@{args.k}, mode={args.mode}) ===")
    print(df.to_string())

    processed_dir = settings.paths["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "benchmark_results_ab_sasrec.csv" if args.mode == "ab-sasrec" else "benchmark_results.csv"
    csv_path = processed_dir / csv_name
    if only and csv_path.exists():
        prev = pd.read_csv(csv_path, index_col=0)
        prev = prev.drop(index=[x for x in df.index if x in prev.index], errors="ignore")
        df = pd.concat([prev, df])
    df.to_csv(csv_path)

    if not args.no_write:
        write_report(df, ds, args.k, split_desc, counts, total, args.mode)
        print(f"\nTOTAL {total:.1f}s — written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
