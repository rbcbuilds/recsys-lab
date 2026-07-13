# recsys-lab

An end-to-end modern recommendation system on **one realistic dataset** — dense
interactions plus natural cold users and cold items — evaluated under a single
global-time split. Classical baselines, two-stage retrieval/ranking, sequential
models, content/social signals, and generative approaches all plug into the same
skeleton.

## Latest results (cross-regime Philadelphia)

One dataset, one train set, four test views: **overall**, **warm**, **cold-item**,
**cold-user**. Cold entities arise naturally from a global-time cutoff — not
simulated carve-outs.

| model | overall@20 | warm@20 | cold-item@20 | cold-user@20 |
|---|---|---|---|---|
| popularity | 0.046 | 0.044 | 0.000 | 0.069 |
| sasrec | 0.037 | **0.060** | 0.000 | 0.000 |
| two_stage | 0.023 | 0.039 | 0.000 | 0.000 |
| **two_stage_unified** | **0.053** | 0.052 | 0.000 | **0.087** |

**Dataset:** 2,500 users · 4,200 items · 185k interactions · 223 cold items · 360 cold users

**Headline finding:** collaborative models score ~0 on cold slices; the unified
`MultiRetriever[two-tower + content + social + popularity] → ranker` stays strong
on overall and cold-user. Cold-item recall stays low — Yelp item text is only
name + categories.

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

**Main slice:** cross-regime Philadelphia (`scripts/make_crossregime.py`) — dense
core (≥10 interactions per user/item) plus a real low-activity tail injected from
raw Yelp so cold users and cold items appear naturally under a global-time split.

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
cp data/processed_philly_xreg/*.parquet data/processed/

# Run headline benchmark (8 models, ~10 min on current slice)
python scripts/benchmark.py

# SASRec ranker-feature A/B on unified pipeline only
python scripts/benchmark.py --mode ab-sasrec
```

To shrink for faster iteration (cap=10, top-1000 warm by social+sequence, cold users pinned):

```bash
python scripts/shrink_subset.py --cross-regime --max-per-user 10 --max-warm-users 1000 --user-select social_seq
cp data/processed_philly_xreg_fast/*.parquet data/processed/
```

Enrich item text from Yelp reviews (raises cold-item ceiling), then re-benchmark:

```bash
python scripts/enrich_item_text.py
python scripts/benchmark.py --mode fast --force
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-extra.txt   # torch, sentence-transformers (neural models)

python scripts/demo.py                  # quick smoke test
python scripts/benchmark.py             # full cross-regime comparison
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
```

## License

[MIT License](LICENSE). Yelp Open Dataset not included — see [Yelp's terms](https://www.yelp.com/dataset) and `data/README.md`.
