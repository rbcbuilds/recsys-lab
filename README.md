# recsys-lab

An end-to-end modern recommendation system built on **one unified dataset** —
from classical baselines through two-stage retrieval/ranking, and extended with
LLM/text, graph, **social**, and multimodal signals. Every stage plugs into the
same skeleton, so the whole system fits together instead of living as disconnected
demos.

**What this builds:**
- Classical baselines (popularity, item-CF, ALS, BPR)
- A two-stage retrieval → ranking pipeline (two-tower + gradient-boosted ranker)
- Sequential retrieval (SASRec) and text/LLM item embeddings for cold start
- A social-graph signal wired in as both a standalone model and a ranker feature
- Honest evaluation (temporal split, Recall/NDCG, activity-sliced metrics)
- A synthetic data generator so the full pipeline runs with zero downloads

## Why this layout

Real recommenders are layered: fast **retrieval** narrows millions of items to a
few hundred candidates, then a slower **ranker** orders them. Every technique
here plugs into that same retrieval → ranking skeleton, so adding text, graph,
social, or image signals is an *upgrade to a component*, not a new paradigm.

```
request → candidate retrieval → re-ranker → business rules/diversity → results
```

## Dataset: Yelp Open Dataset (unified)

Yelp is a single public dataset that natively bundles **all** the signals this
system uses:

| Signal                    | Where it lives in Yelp                |
| ------------------------- | ------------------------------------- |
| User–item interactions    | reviews / star ratings                |
| **Social graph**          | user `friends` list                   |
| Text content              | full review + business text           |
| Item metadata             | categories, attributes, location      |
| Images                    | native business photos                |
| Sequential (timestamps)   | review `date`                         |

The full dataset is several GB, so this project includes an **aggressive subsetter**
(`recsys.data.subset`) that carves a small, dense slice (e.g. one metro area,
users/items with ≥N interactions) for fast iteration on the unified dataset.

> **No download yet? No problem.** A **synthetic Yelp-like generator**
> (`recsys.data.synthetic`) produces interactions + a social graph + text +
> stand-in image vectors, so the entire pipeline (splits, metrics, baselines)
> runs *immediately* — before any download. The real Yelp subset drops in via
> the same loader interface.

## Roadmap

Built top-to-bottom. Each phase has a module and a notebook.

**Phase 1 — Classical baselines (the bar every later model must beat)**
1. Popularity / trending — `models/popularity.py`
2. Item-based collaborative filtering — `models/item_cf.py`
3. Matrix factorization (ALS, implicit feedback) — `models/matrix_factorization.py`
4. BPR (pairwise ranking MF) — `models/bpr.py`

**Phase 2 — Retrieval + ranking**
5. Two-tower retrieval (+ ANN) — `models/two_tower.py`
6. Learning-to-rank re-ranker (LightGBM) — `models/ranker.py`
7. Two-stage (retrieve → re-rank) — `models/two_stage.py`  *(compare vs two-tower alone)*

**Phase 3 — Modern layer**
8. **Social recsys** — `models/social.py` (standalone + ranker feature)
9. **SASRec** (sequential / self-attentive) — `models/sasrec.py`
10. **Text / LLM item embeddings + content two-tower** — `models/text_embeddings.py`
11. Graph recsys (LightGCN / Node2Vec) — `models/graph.py` *(scaffold)*
12. Multimodal (image embeddings from photos) — `models/multimodal.py` *(scaffold)*

The question driving every addition: **can it beat a well-tuned baseline?**

> **Reference:** see [`docs/techniques.md`](docs/techniques.md) for a concise
> summary of every technique and key concepts (ALS, GBM, in-batch negatives,
> evaluation), and [`docs/benchmarks.md`](docs/benchmarks.md) for cross-regime
> results — overall, warm, cold-item, and cold-user on one dataset (reproduce
> with `python scripts/benchmark.py`).

## Project structure

```
recsys-lab/
├── README.md
├── requirements.txt          # core deps (light install)
├── requirements-extra.txt    # heavy/optional deps (torch, sentence-transformers, faiss, ...)
├── config.py                 # paths + global settings
├── data/
│   ├── raw/                  # real Yelp JSON goes here (gitignored)
│   ├── processed/            # subset / synthetic output (gitignored)
│   └── README.md             # how to download the real Yelp dataset
├── src/recsys/
│   ├── config.py
│   ├── data/                 # synthetic generator, subsetter, unified loaders
│   ├── eval/                 # temporal split + ranking metrics
│   └── models/               # baselines (working) + modern modules (scaffolds)
├── notebooks/                # step-by-step, calls into src/recsys
└── scripts/                  # CLI entry points (make_subset, demo)
```

## Quickstart

```bash
# 1. (recommended) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. install core dependencies
pip install -r requirements.txt

# 3. run the whole pipeline on SYNTHETIC data (no download needed)
python scripts/demo.py

# 4. or open the notebooks
jupyter lab
```

To use the **real** Yelp data instead, follow `data/README.md`, then:

```bash
python scripts/make_subset.py --city "Santa Barbara" --min-user-reviews 10 --min-item-reviews 10
```

### Dataset size vs. run time

The subsetter can still produce a slice that's slow to iterate on (a metro area
like Philadelphia is ~14k users / 419k interactions and a full model comparison
takes >15 min). Use `scripts/shrink_subset.py` to shrink the **already-built**
`data/processed/` slice in seconds — no multi-GB rescan:

```bash
python scripts/shrink_subset.py --max-users 2000 --max-per-user 40
```

Measured full 5-model comparison (popularity, ALS, social, two-stage ±social)
on a typical laptop, single-thread BLAS:

| slice | users | items | interactions | full comparison |
|---|---|---|---|---|
| Philadelphia (10/10) | 14,325 | 6,472 | 419k | >15 min (didn't finish) |
| top-2500 users | 2,500 | 4,348 | 205k | ~5–6 min |
| **top-2000, ≤40/user** | **2,000** | **2,167** | **63k** | **~2.4 min** |
| synthetic (default) | 800 | 400 | 20k | ~40 s |

The dominant cost is **two-tower training**, which is ~linear in
`interactions × epochs`; the cheap models (popularity/ALS/social) are all a few
seconds. Speed knobs, biggest lever first:

1. **`--max-per-user`** — caps each user's history. A few "power users" carry
   most interactions, so capping this shrinks the interaction count (and thus
   two-tower time) far more than dropping users.
2. **`TwoTowerRecommender(epochs=...)`** — linear; `10 → 5` roughly halves training.
3. **`candidate_n`** on the two-stage ranker — drives the rerank/social-scoring loop.
4. **`--max-users`** — mostly affects the cheap eval/social loops.

Rule of thumb: `two-tower fit ≈ interactions/1000 × epochs × ~0.09s`.

## Evaluation

Always use a **temporal split** (train on past, test on future) — never random.
Core metrics: **Recall@K**, **NDCG@K**, **HitRate@K**, plus **coverage** to watch
for filter bubbles. All in `recsys.eval`.

## Status

- [x] Project scaffold, config, data layer (synthetic + subsetter + loaders)
- [x] Evaluation (temporal split + metrics)
- [x] Phase 1 baselines (popularity, item-CF, ALS, BPR)
- [x] Phase 2 retrieval/ranking (two-tower, LightGBM ranker, two-stage)
- [x] Social signal (standalone model + ranker feature)
- [x] SASRec (sequential self-attentive retrieval)
- [x] Text embeddings + content two-tower (cold-start path)

**Extension points** (scaffolded): graph models (LightGCN/Node2Vec) and
multimodal (image) signals — each slots into the existing retrieval → ranking
skeleton.

## License

Released under the [MIT License](LICENSE). The Yelp Open Dataset is **not**
included and is subject to [Yelp's dataset terms](https://www.yelp.com/dataset)
(academic/educational use); see `data/README.md` for download steps.
