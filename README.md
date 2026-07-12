# recsys-lab

A hands-on lab for learning modern recommendation systems end to end — from
classical baselines all the way to LLM/text, graph, **social**, and multimodal
signals — all on **one unified dataset** so you can see how everything fits
together in a single modern system.

## Why this layout

Real recommenders are layered: fast **retrieval** narrows millions of items to a
few hundred candidates, then a slower **ranker** orders them. Every technique in
this lab plugs into that same retrieval → ranking skeleton, so adding text,
graph, social, or image signals is an *upgrade to a component*, not a new
paradigm.

```
request → candidate retrieval → re-ranker → business rules/diversity → results
```

## Dataset: Yelp Open Dataset (Path A — unified)

Yelp is the one common learning dataset that natively bundles **all** the signals
this lab needs:

| Signal                    | Where it lives in Yelp                |
| ------------------------- | ------------------------------------- |
| User–item interactions    | reviews / star ratings                |
| **Social graph**          | user `friends` list                   |
| Text content              | full review + business text           |
| Item metadata             | categories, attributes, location      |
| Images                    | native business photos                |
| Sequential (timestamps)   | review `date`                         |

The full dataset is several GB, so this lab includes an **aggressive subsetter**
(`recsys.data.subset`) that carves a small, dense slice (e.g. one metro area,
users/items with ≥N interactions) to give MovieLens-like iteration speed on the
unified dataset.

> **No download yet? No problem.** A **synthetic Yelp-like generator**
> (`recsys.data.synthetic`) produces interactions + a social graph + text +
> stand-in image vectors, so the entire pipeline (splits, metrics, baselines)
> runs *immediately* — before you download anything. Swap in the real Yelp
> subset when you're ready; the loaders expose the same interface.

## Learning roadmap

Build top-to-bottom. Each phase has a module and (eventually) a notebook.

**Phase 1 — Classical baselines (do first, these set your evaluation bar)**
1. Popularity / trending — `models/popularity.py`
2. Item-based collaborative filtering — `models/item_cf.py`
3. Matrix factorization (ALS, implicit feedback) — `models/matrix_factorization.py`

**Phase 2 — Retrieval + ranking**
4. Two-tower retrieval (+ ANN) — `models/two_tower.py`
5. Learning-to-rank re-ranker (LightGBM) — `models/ranker.py`
6. Two-stage (retrieve → re-rank) — `models/two_stage.py`  *(compare vs two-tower alone)*

**Phase 3 — Modern layer (upgrades to the components above)**
7. Text / LLM item embeddings — `models/text_embeddings.py` *(scaffold)*
8. Graph recsys (LightGCN / Node2Vec on the user–item graph) — `models/graph.py` *(scaffold)*
9. **Social recsys** (social-augmented recs from the real friend graph) — `models/social.py` *(scaffold)*
10. Multimodal (image embeddings from photos) — `models/multimodal.py` *(scaffold)*

The golden rule throughout: **can this beat a well-tuned baseline?** That
measurement skill is the real lesson.

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

## Evaluation

Always use a **temporal split** (train on past, test on future) — never random.
Core metrics: **Recall@K**, **NDCG@K**, **HitRate@K**, plus **coverage** to watch
for filter bubbles. All in `recsys.eval`.

## Status

- [x] Project scaffold, config, data layer (synthetic + subsetter + loaders)
- [x] Evaluation (temporal split + metrics)
- [x] Phase 1 baselines (popularity, item-CF, ALS)
- [x] Phase 2 retrieval/ranking (two-tower, LightGBM ranker, two-stage)
- [ ] Phase 3 modern layer (text/LLM, graph, social, multimodal) — scaffolded
