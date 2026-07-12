# Recsys Techniques — Quick Reference

A concise reference for the methods implemented in this project. Each maps to a
file in `src/recsys/models/` and a notebook in `notebooks/`.

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
| Item-based CF | `item_cf.py` | Baseline | "Items co-liked by the same users are similar." Recommend items similar to a user's history. |
| ALS (matrix factorization) | `matrix_factorization.py` | Retrieval | Learn user & item vectors so that `user · item ≈ preference` (pointwise / confidence-weighted). |
| BPR | `bpr.py` | Retrieval | Same MF factors as ALS; pairwise ranking loss (observed > unobserved). |
| Two-tower | `two_tower.py` | Retrieval | Neural user tower + item tower in a shared space; score = dot product. Scales via ANN. |
| LightGBM ranker | `ranker.py` | Ranking | Gradient-boosted trees re-order candidates using features. |
| Two-stage | `two_stage.py` | Retrieval+Ranking | Two-tower retrieves → ranker re-orders. The production pattern. |
| Social-neighbor CF | `social.py` | Signal / baseline | Recommend what a user's friends liked (trust-weighted). Also a *feature* for the ranker. |
| SASRec | `sasrec.py` | Sequential retrieval | Causal self-attention over the user's recent item sequence; predicts next item. |
| Text embeddings | `text_embeddings.py` | Content / cold-start | Encode item text; user = mean of liked-item vectors; cosine score. |
| Content two-tower | `text_embeddings.py` | Retrieval | Two-tower with text concatenated into the item tower (production cold-start pattern). |

---

## Collaborative filtering — method classes

### Memory-based CF

**User-user CF**: find users with similar rating vectors (cosine or Pearson), recommend what they liked. Doesn't scale — user space is large and the matrix is sparse.

**Item-item CF**: find items similar to what a user has interacted with. Item similarities are precomputed and stable → scales better than user-user. Amazon's original algorithm.

**Why memory-based fails at scale**: the rating matrix is extremely sparse (most users rate <0.1% of items), so similarity estimates are unreliable.

---

### Matrix factorization

**SVD / FunkSVD**: decompose `R ≈ U·Vᵀ`, minimize squared error on *observed* ratings via SGD. Classic for explicit feedback (star ratings).

**ALS (Alternating Least Squares)**: same decomposition but solves U and V alternately in closed form. Handles implicit feedback with confidence weighting: `c_ui = 1 + α·r_ui` where `r_ui = 1` if the user interacted, 0 otherwise. Fast, parallelizable, still used in production. **Alternate** means: freeze items → solve users (least squares) → freeze users → solve items → repeat.

**BPR (Bayesian Personalized Ranking)**: optimizes pairwise ranking — observed items should rank above unobserved. Loss: `Σ -log σ(x̂_ui - x̂_uj)` for (u, i, j) where i is observed, j is not. Better objective than pointwise MSE for implicit data.

---

### Neural CF

**NCF / NeuMF**: replaces the dot product in MF with an MLP, learning non-linear user×item interactions. Outperforms pure MF on dense datasets.

**Two-tower (dual encoder)**: separate neural networks for user and item → shared embedding space. Score = dot product. The standard for *retrieval at scale*: item embeddings are precomputed once, ANN (FAISS/ScaNN) retrieves top-K in milliseconds.

Trained with **in-batch negatives** (sampled softmax): within a batch of (user, positive-item) pairs, every other item is a negative. Popularity bias: popular items appear as negatives more often → their scores get suppressed. Corrected with logQ correction or hard negative mining.

The item tower is where text/image/graph/social features get injected.

---

### Sequential / session-based CF

Models the *order* of interactions, not just the set. No persistent user embedding — user is their recent history window.

**GRU4Rec**: RNN over item sequence. First neural sequential rec paper.

**SASRec**: self-attention (transformer decoder) over item sequences. Faster than RNN, captures long-range dependencies. Current strong baseline.

**BERT4Rec**: bidirectional transformer with masked item prediction (BERT-style pretraining). Slightly stronger than SASRec on dense data, slower.

Use when: recency and order matter (e-commerce, streaming, news). Less useful when taste is stable over long periods.

---

### Graph-based CF

**LightGCN**: propagates user and item embeddings over the user–item bipartite graph via weighted sum (no non-linearity). Final embedding = average across propagation layers. Strong on sparse data; captures multi-hop signals. Expensive — full graph must be in memory.

**Social CF (GraphRec, DiffNet)**: adds the social/friend graph as a second propagation path. Helps cold-start users who have friends with rich histories.

---

### Factorization machines (ranker stage)

**FM**: generalizes MF to arbitrary feature vectors. Models all pairwise feature interactions: `ŷ = w₀ + Σwᵢxᵢ + Σᵢ<ⱼ <vᵢ,vⱼ>xᵢxⱼ`. Interaction term is O(nk) via the FM trick (not O(n²)).

**DeepFM**: FM (2nd-order interactions) + DNN (higher-order) sharing the same input embeddings. Used as a ranker in production systems.

---

### When to use what

| Situation | Method |
|---|---|
| Explicit ratings, small scale | SVD / ALS |
| Implicit, millions of items | Two-tower retrieval → BPR/LambdaRank ranker |
| Sequential behavior matters | SASRec / BERT4Rec |
| Rich side features available | DeepFM / two-tower with feature inputs |
| Social/graph signal available | LightGCN or social feature in ranker |
| Cold start | Content tower (text/image) + popularity fallback |

---

## Key concepts

### Implicit vs explicit feedback
- **Explicit**: ratings/stars (user directly states preference). Clean signal, rare — most users don't rate things.
- **Implicit**: preference inferred from behavior — a purchase, click, play, or review existing regardless of score. Noisy but abundant. Most production systems run on implicit data.

| | Explicit | Implicit |
|---|---|---|
| What you observe | Actual preference score | Binary interaction (seen / not seen) |
| Unobserved = | Missing data | Weakly negative (probably not relevant) |
| Loss function | MSE on observed ratings | Confidence-weighted MSE (ALS) or pairwise ranking (BPR) |
| Metric | RMSE | Recall@K, NDCG@K |

**Why use ALS as implicit even when star ratings exist (e.g. Yelp)?**
The task is *what to recommend next*, not *what score will they give*. Using implicit:
1. Lets you model the 99.9% of unreviewed items ("didn't interact = probably not relevant") — explicit MF ignores unobserved entries entirely.
2. Is more robust to selection bias: people rate things they already chose to visit, so the rating distribution is not a random sample of preferences.
3. The *act* of reviewing is the relevance signal; the star value feeds downstream (as a ranker feature or `positive_threshold` filter), not into retrieval.

Typical production split of rating signals:
- **Retrieval** (ALS / two-tower): implicit — did they interact?
- **Ranker**: uses stars as a feature (`user_avg_rating`, `item_avg_rating`, `rating_gap`)
- **Rating prediction**: separate model only if the product needs to display a predicted score

### ALS (Alternating Least Squares)
See matrix factorization above.

### Two-tower + in-batch negatives
See neural CF above.

### GBM (Gradient Boosting Machine)
Build many small decision trees **sequentially**, each correcting the previous ensemble's errors (defined by the loss **gradient**). LightGBM/HistGradientBoosting are implementations.
- **`lambdarank`** objective = *listwise* learning-to-rank: optimizes item **order** within each user's candidate list (NDCG). This is LambdaMART-style.
- Pointwise fallback (sklearn `HistGradientBoostingRegressor`) predicts a score per item independently — simpler, weaker ranking signal.

### Social recommendation
`score(u,i) = (1-α)·Σ_friends trust(u,f)·liked(f,i) + α·popularity(i)`
- **trust**: `uniform` (all friends equal) or `jaccard` (share of common items).
- Popularity back-off handles friendless/cold users.
- Best used as a **feature into the ranker** (`use_social=True`), not standalone.

### ALS vs BPR (same factors, different loss)
- **ALS**: pointwise. Reconstructs a confidence-weighted preference matrix. Strong when interaction counts are informative.
- **BPR**: pairwise. For each (user, observed i, unobserved j), push `score(u,i) > score(u,j)`. Directly optimizes ranking.
- Run both on the same split (`ALSRecommender` vs `BPRRecommender`) to isolate the objective. Library-matched via `implicit`.

### SASRec (sequence vs set)
Static user embeddings (ALS, two-tower id table) average a user's whole history into one vector. That breaks when:
- intent is session-local (just browsed sushi → next rec should shift)
- taste drifts over time
- order of interactions matters more than the bag of items

SASRec replaces the user embedding with a **causal transformer over the recent item sequence**. Training target = next item at each position. Inference = last hidden state · item embedding matrix. Implemented in `sasrec.py`.

### Text embeddings + cold start
Collaborative models cannot score an item with zero interactions. Content can:
1. **Standalone** (`ContentBasedRecommender`): embed item text (sentence-transformers, or TF-IDF+SVD fallback) → user vector = mean of liked-item embeddings → cosine. `cold_item_ids()` lists catalog items never seen in train.
2. **In the retriever** (`ContentTwoTowerRecommender`): item tower = MLP([id_emb ; text_emb]). Cold items are still embeddable (mean id-embedding + their text), so retrieval degrades gracefully as the catalog turns over — the usual production answer to "how do you handle new items?".

**Measuring it** — `cold_item_holdout()` strips a fraction of items entirely from train (simulating cold items with real test targets), and `evaluate_cold_items()` reports recall restricted to those items. Run `scripts/cold_start.py`: collaborative retrievers score ≈0 on cold items; content-aware ones score > 0. (Natural cold items — first-seen-in-test — barely exist here because the subset densifies to items with ≥10 reviews, hence the simulated holdout.)

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

## Benchmark results (synthetic, recall@10 / ndcg@10)

| Model | recall@10 | ndcg@10 |
|---|---|---|
| popularity | 0.059 | 0.044 |
| als | 0.229 | 0.209 |
| two_tower | 0.288 | 0.244 |
| item_cf | 0.335 | 0.316 |
| two_stage (no social) | 0.360 | 0.325 |
| **two_stage + social** | **0.395** | **0.363** |

Takeaways: two-stage beats retrieval-alone; social adds a further lift as a ranker feature. Re-run on real Yelp for the honest numbers.

> Numbers are on synthetic data (mechanics check). The synthetic friend graph is
> built partly from latent taste, so treat the social lift as illustrative until
> reproduced on the real Yelp friend graph.

---

## Extension points (scaffolds present)

- **Graph / LightGCN** (`graph.py`) — propagate over the user–item graph.
- **Multimodal** (`multimodal.py`) — image embeddings (CLIP) from Yelp photos.
- **Richer ranker features** — text/image similarity, geo distance, context.
