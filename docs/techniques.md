# Recsys Techniques — Quick Reference

A short review sheet for the methods implemented in this lab. Each maps to a file
in `src/recsys/models/` and a notebook in `notebooks/`.

---

## The core architecture: retrieval → ranking

Modern recommenders are **two stages**:

```
request → RETRIEVAL (fast, high recall) → ~100s candidates
        → RANKING (slower, high precision) → top-K shown
```

- **Retrieval**: cut millions of items to a few hundred. Optimizes *recall*.
- **Ranking**: carefully re-order that short list with rich features. Optimizes *precision* (NDCG).

Every model below is either a retriever, a ranker, or a baseline that does both in one shot.

---

## Models (in build order)

| Model | File | Type | One-line idea |
|---|---|---|---|
| Popularity / trending | `popularity.py` | Baseline | Recommend the most (recency-weighted) popular items. Cold-start fallback. |
| Item-based CF | `item_cf.py` | Baseline | "Items co-liked by the same users are similar." Recommend items similar to your history. |
| ALS (matrix factorization) | `matrix_factorization.py` | Retrieval | Learn user & item vectors so that `user · item ≈ preference`. |
| Two-tower | `two_tower.py` | Retrieval | Neural user tower + item tower in a shared space; score = dot product. Scales via ANN. |
| LightGBM ranker | `ranker.py` | Ranking | Gradient-boosted trees re-order candidates using features. |
| Two-stage | `two_stage.py` | Retrieval+Ranking | Two-tower retrieves → ranker re-orders. The production pattern. |
| Social-neighbor CF | `social.py` | Signal / baseline | Recommend what your friends liked (trust-weighted). Also a *feature* for the ranker. |

---

## Key concepts

### Implicit vs explicit feedback
- **Explicit**: ratings/stars (user states preference).
- **Implicit**: clicks/views/reviews-exist (preference inferred). Most real data. ALS uses **confidence weighting** (`alpha`): seen = confidently positive, unseen = weakly negative.

### ALS (Alternating Least Squares)
Factor the sparse user×item matrix `R ≈ U·Vᵀ`. Solving both sides at once is hard, so **alternate**: freeze items → solve users (closed-form least squares) → freeze users → solve items. Repeat. Fast, strong baseline for implicit feedback.

### Two-tower + in-batch negatives
Two embedding towers; score = dot product. Trained with **in-batch negatives** (sampled softmax): within a batch of (user, positive-item) pairs, every *other* item is a negative. At serving time embed items once, use **ANN** (e.g. FAISS) for fast nearest-neighbor retrieval. The item tower is where extra features (text/image/graph/social) get injected later.

### GBM (Gradient Boosting Machine)
Build many small decision trees **sequentially**, each correcting the previous ensemble's errors (defined by the loss **gradient**). LightGBM/HistGradientBoosting are implementations.
- **`lambdarank`** objective = *listwise* learning-to-rank: optimizes item **order** within each user's candidate list (NDCG). This is LambdaMART-style.
- Pointwise fallback (sklearn `HistGradientBoostingRegressor`) predicts a score per item independently — simpler, weaker ranking signal.

### Social recommendation
`score(u,i) = (1-α)·Σ_friends trust(u,f)·liked(f,i) + α·popularity(i)`
- **trust**: `uniform` (all friends equal) or `jaccard` (share of common items).
- Popularity back-off handles friendless/cold users.
- Best used as a **feature into the ranker** (`use_social=True`), not standalone.

---

## Evaluation (do this right or nothing else matters)

| Rule | Why |
|---|---|
| **Temporal split** (train on past, test on future) | Random splits leak the future and inflate scores. |
| **Recall@K** | Did relevant items make the top-K? (retrieval quality) |
| **NDCG@K** | Are relevant items ranked high? (ranking quality) |
| **HitRate@K** | Did *any* relevant item appear? |
| **Coverage@K** | How much of the catalog gets shown? (diversity / anti-popularity-bias) |
| **Slice by user activity** (`evaluate_by_activity`) | Social/content signals usually help **sparse** users most; averages hide this. |

**The golden question for every new model:** *does it beat a well-tuned baseline?* If not, that's a finding — not a failure.

---

## Synthetic vs real data (important caveat)

The synthetic generator builds the friend graph partly from latent taste, so a
"social helps" result on synthetic data only validates the **mechanics**. The
honest scientific test needs the **real Yelp friend graph** (friendships formed
independently of ratings). Same code — just point `load_dataset()` at the Yelp
subset (`scripts/make_subset.py`).

---

## Results so far (synthetic, recall@10 / ndcg@10)

| Model | recall@10 | ndcg@10 |
|---|---|---|
| popularity | 0.059 | 0.044 |
| als | 0.229 | 0.209 |
| two_tower | 0.288 | 0.244 |
| item_cf | 0.335 | 0.316 |
| two_stage (no social) | 0.360 | 0.325 |
| **two_stage + social** | **0.395** | **0.363** |

Takeaways: two-stage beats retrieval-alone; social adds a further lift as a ranker feature. Re-run on real Yelp for the honest numbers.

---

## Still to explore (scaffolds present)

- **Text / LLM embeddings** (`text_embeddings.py`) — semantic item vectors, cold-start.
- **Graph / LightGCN** (`graph.py`) — propagate over the user–item graph.
- **Multimodal** (`multimodal.py`) — image embeddings (CLIP) from Yelp photos.
- **Richer ranker features** — text/image similarity, geo distance, context.
