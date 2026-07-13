# recsys-lab

An end-to-end modern recommendation system on **one realistic dataset** — dense
interactions plus natural cold users and cold items — evaluated under a single
global-time split. Classical baselines, two-stage retrieval/ranking, sequential
models, content/social signals, and generative approaches all plug into the same
skeleton.

## Latest results (cross-regime Philadelphia)

One dataset, one train set, four test views: **overall**, **warm**, **cold-item**,
**cold-user**. Cold entities arise naturally from a global-time cutoff — not
simulated carve-outs. Core benchmark (`python scripts/benchmark.py`, Jul 2026).

| model | overall@20 | warm@20 | cold-item@20 | cold-user@20 |
|---|---|---|---|---|
| **two_stage_unified** | **0.061** | **0.067** | 0.000 | 0.054 |
| popularity | 0.061 | 0.053 | 0.000 | **0.090** |
| item_token_lm | 0.057 | 0.048 | 0.000 | **0.090** |
| image_embedding | 0.047 | 0.034 | 0.000 | **0.090** |
| hstu | 0.031 | 0.044 | 0.000 | 0.000 |
| sasrec | 0.030 | 0.042 | 0.000 | 0.000 |
| semantic_id_lm | 0.030 | 0.011 | **0.008** | **0.090** |
| contrastive_two_tower | 0.017 | 0.024 | 0.000 | 0.000 |
| lightgcn | 0.004 | 0.006 | 0.000 | 0.000 |

**Dataset:** 1,507 users · 3,401 items · 20k interactions · 51 cold items · 78 cold users · enriched review text + CLIP images

**Headline finding:** collaborative models score ~0 on cold slices; the unified
`MultiRetriever[two-tower + content + social + popularity] → ranker` edges overall/warm.
Popularity and generative models (`item_token_lm`) still win cold-user. Cold-item
recall stays near-zero except **`semantic_id_lm` (0.008)** — review enrichment helps
slightly but the ceiling is still low.

Full table + reproduction: [`docs/benchmarks.md`](docs/benchmarks.md) ·
`python scripts/benchmark.py`

## What this builds

```
request → retrieval (multi-source) → re-ranker → top-K
                ↓
     warm / cold-item / cold-user slices (same recommendations, four views)
```

| Layer | Models |
|---|---|
| **Baselines** | popularity, item-CF, ALS, BPR |
| **Retrieval** | two-tower, SASRec, content-based, social, MultiRetriever (RRF) |
| **Ranking** | LightGBM ranker (+ social / SASRec features), LLM cross-encoder reranker |
| **Generative** | item-token LM (autoregressive decode), LLM reranker (language scoring) |
| **Production pattern** | unified two-stage: complementary retrievers → ranker |

## Dataset

**Main slice:** cross-regime Philadelphia fast subset (`scripts/make_crossregime.py`
+ `scripts/shrink_subset.py`) with review-text enrichment and optional CLIP image
vectors — dense core (≥10 interactions per user/item) plus a real low-activity tail
injected from raw Yelp so cold users and cold items appear naturally under a
global-time split.

| Signal | Yelp source |
|---|---|
| Interactions | reviews / star ratings |
| Social graph | user `friends` |
| Text | business name + categories (+ review text as extension) |
| Timestamps | review `date` → global-time split |

**No download yet?** A synthetic generator runs the full pipeline immediately.
Real Yelp drops in via the same loader.

```bash
# Build cross-regime slice from raw Yelp JSON
python scripts/make_crossregime.py

# Shrink for laptop iteration (headline benchmark dataset)
python scripts/shrink_subset.py --cross-regime --max-per-user 10 --max-warm-users 1000 --user-select social_seq
python scripts/enrich_item_text.py
python scripts/build_image_vectors.py   # optional: Yelp photos → CLIP vectors
cp data/processed_philly_xreg_fast/*.parquet data/processed/
cp data/processed_philly_xreg_fast/item_image_vectors.npy data/processed/ 2>/dev/null || true

# Run core benchmark (9 models, default; ~2h on laptop CPU — unified dominates)
python scripts/benchmark.py --ips

# All 13 models (~2+ hours beyond unified) or smoke test (~5 min)
python scripts/benchmark.py --mode full
python scripts/benchmark.py --mode fast

# SASRec ranker-feature A/B on unified pipeline only
python scripts/benchmark.py --mode ab-sasrec
```

To rebuild only the fast subset without re-running the full cross-regime build:

```bash
python scripts/shrink_subset.py --cross-regime --max-per-user 10 --max-warm-users 1000 --user-select social_seq
cp data/processed_philly_xreg_fast/*.parquet data/processed/
```

Enrich item text from Yelp reviews (raises cold-item ceiling), then re-benchmark:

```bash
python scripts/enrich_item_text.py
python scripts/benchmark.py --force --ips
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-extra.txt   # torch, sentence-transformers (neural models)

python scripts/demo.py                  # quick smoke test
python scripts/benchmark.py             # core cross-regime comparison (default)
python scripts/benchmark.py --mode full # all 13 models
```

Real Yelp: see [`data/README.md`](data/README.md).

## Evaluation

**Global-time split** (train on past, test on future) with derived cold sets:
`cold_items = test_items − train_items`, `cold_users = test_users − train_users`.

Metrics: **Recall@K**, **NDCG@K** on overall + warm + cold-item + cold-user slices.
Details: [`docs/techniques.md`](docs/techniques.md).

## Status

- [x] Cross-regime dataset (dense core + cold tail, natural coldness)
- [x] Global-time split + four-view benchmark (`scripts/benchmark.py`)
- [x] Baselines: popularity, item-CF, ALS, BPR
- [x] Two-stage: two-tower + LightGBM ranker
- [x] Unified pipeline: MultiRetriever → ranker (social + optional SASRec feature)
- [x] Sequential: SASRec (standalone + ranker feature)
- [x] Content: content-based retrieval, content two-tower
- [x] Generative: item-token LM, LLM cross-encoder reranker
- [x] Graph: LightGCN (BPR + neighborhood aggregation)
- [x] Multimodal: CLIP image embeddings (`scripts/build_image_vectors.py`)
- [x] Tier 3: HSTU, semantic ID LM, contrastive two-tower, MMR diversity, IPS eval

## Project structure

```
recsys-lab/
├── scripts/
│   ├── make_crossregime.py   # build main dataset (dense + cold tail)
│   ├── benchmark.py          # headline cross-regime comparison
│   ├── build_image_vectors.py  # CLIP vectors from Yelp photos (Tier 2)
│   └── demo.py               # quick smoke test
├── src/recsys/
│   ├── data/                 # loaders, subsetter, synthetic generator
│   ├── eval/                 # global_temporal_split, metrics
│   └── models/               # all recommenders + rankers
├── docs/
│   ├── techniques.md         # methods reference + interview concepts
│   └── benchmarks.md         # latest numbers (auto-generated)
└── notebooks/
    ├── 01_setup_and_eda.ipynb
    ├── 02_baselines.ipynb … 05_social.ipynb
    ├── 06_tier1_cold_start.ipynb   # Tier 1: review text, extended ranker
    ├── 07_tier2_graph_multimodal.ipynb  # Tier 2: LightGCN, CLIP images
    └── 08_tier3_modern.ipynb       # Tier 3: HSTU, semantic IDs, MMR, IPS
```

## License

[MIT License](LICENSE). Yelp Open Dataset not included — see [Yelp's terms](https://www.yelp.com/dataset) and `data/README.md`.
