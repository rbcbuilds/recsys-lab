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
| Content-based | `text_embeddings.py` | Content / cold-start | Encode item text; user = mean of liked-item vectors; cosine score. |
| Content two-tower | `text_embeddings.py` | Retrieval | Two-tower with text concatenated into the item tower (production cold-start pattern). |
| MultiRetriever | `multi_retriever.py` | Retrieval | Union several retrievers and fuse via reciprocal-rank fusion; usable as a two-stage retriever. |
| Item-token LM | `item_token_lm.py` | Generative retrieval | Causal LM over item ids; autoregressively *generates* the next items token-by-token. |
| LLM reranker | `llm_reranker.py` | Ranking (language) | Cross-encoder (or prompt embedding) scores (history, candidate) pairs; used in `LLMTwoStageRecommender`. |

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

SASRec replaces the user embedding with a **causal transformer over the recent item sequence**. Training target = next item at each position. Inference = last hidden state · item embedding matrix. Implemented in `sasrec.py`. In the unified two-stage, SASRec is also used as a **ranker feature** (`use_sasrec=True`): it scores candidates from the fused pool without adding a fifth retriever.

### Generative vs discriminative recommendation

| | **Discriminative** (most of this repo) | **Generative** |
|---|---|---|
| **Question** | How relevant is item *i* for user *u*? | What item(s) should we show user *u*? |
| **Output** | A score per (user, item) pair | The recommendation(s) directly |
| **Serving** | Score candidates → sort → top-K | Model *emits* items (tokens or text) |
| **Examples here** | ALS, BPR, two-tower, SASRec, LightGBM ranker | `ItemTokenLMRecommender` |

**Discriminative path (production default):** retrieve ~200 candidates → score each → re-rank. Even SASRec is discriminative at inference: it dot-products the sequence embedding against *all* item embeddings, then takes top-K.

**Generative path — item-token LM** (`item_token_lm.py`): each item id is a vocabulary token. A causal transformer is trained with next-token cross-entropy on user histories. At inference the model **autoregressively decodes** item tokens one at a time (append each pick to context, predict the next) — it generates the list instead of scanning the catalog. Cold/empty users fall back to popularity.

**Generative-adjacent — LLM reranker** (`llm_reranker.py`): still discriminative in the pipeline (candidates in, sorted list out), but the *scoring function* is language-native. `LLMReranker` builds a short text history ("User recently enjoyed: …") and scores each candidate's item text with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`, or bi-encoder / token-overlap fallback). `LLMTwoStageRecommender` = retriever → `LLMReranker`. Useful when you want LLM-style relevance judgment without training a GBM ranker — and without generating the full catalog from scratch.

**When to use which:** discriminative retrieve+rank is still the reliable production pattern. Item-token LMs are the research direction toward unified generative rec (TIGER, LC-Rec, etc.). LLM rerankers are a pragmatic middle ground: language understanding on a *small* candidate pool.

### Text embeddings + cold start
Collaborative models cannot score an item with zero interactions. Content can:
1. **Standalone** (`ContentBasedRecommender`): embed item text (sentence-transformers, or TF-IDF+SVD fallback) → user vector = mean of liked-item embeddings → cosine. `cold_item_ids()` lists catalog items never seen in train.
2. **In the retriever** (`ContentTwoTowerRecommender`): item tower = MLP([id_emb ; text_emb]). Cold items are still embeddable (mean id-embedding + their text), so retrieval degrades gracefully as the catalog turns over — the usual production answer to "how do you handle new items?".

### Cold-start, done honestly: one model, three slices (the headline result)
The claim worth testing is "**one** pipeline handles warm users/items, cold items, **and** cold users." You don't get to retrain per regime — so we train **once**, then score the *same* recommendations against four views of the test set (`scripts/benchmark.py`):

- Slices: **warm** (user & item both seen in train), **cold-item** (target item unseen in train), **cold-user** (user unseen in train). Overlap (cold user *and* cold item) is dropped for clean attribution.

**Main construction — natural coldness via a global-time split.** The headline dataset is the *cross-regime* slice (`scripts/make_crossregime.py` → `data/processed_philly_xreg/`): keep the dense Philadelphia core, then inject a real low-activity tail of items and users whose activity is entirely *after* a global cutoff `T`. Evaluate with a single wall-clock split (`global_temporal_split`, cutoff at the 90th time percentile):

```
train = interactions with t <= T        test = interactions with t > T
cold_items = test items not in train    cold_users = test users not in train
```

Because the injected tail is first-seen after `T`, it has **zero** training history — genuinely cold, no carve-out. Unlike a per-user split (which leaves every user with train history and so can never produce a cold *user*), one global cutoff makes both cold items and cold users arise naturally. Run `python scripts/benchmark.py` on the cross-regime slice.

**Results** (cross-regime Philadelphia, recall@20 — `python scripts/benchmark.py`):

| model | overall | warm | cold-item | cold-user |
|---|---|---|---|---|
| popularity | 0.046 | 0.044 | 0.000 | 0.069 |
| als / bpr | 0.017 | 0.027 | 0.000 | 0.000 |
| sasrec | 0.037 | **0.060** | 0.000 | 0.000 |
| two_stage | 0.023 | 0.039 | 0.000 | 0.000 |
| item_token_lm | — | — | — | — |
| two_stage_llm | — | — | — | — |
| **two_stage_unified** | **0.052** | 0.052 | 0.000 | **0.087** |

*`item_token_lm` and `two_stage_llm` are in `benchmark.py` full mode; re-run to fill rows.*

The unified two-stage (`MultiRetriever[two-tower + content_based + social + popularity] → ranker`, see `scripts/benchmark.py::build_unified_two_stage`) wins on **overall and cold-user**; SASRec is strongest on warm among single-signal models.

**\*The cold-item finding (a real one, unchanged by the natural slice).** Injecting *real* cold items makes the slice statistically credible; it does **not** raise the score. Cold-item recall stays low because it's the *data*, not the ranker: Yelp item text is only `name + categories` (~75 chars, e.g. `"Tuna Bar. Sushi Bars, Restaurants, Japanese"`), which clusters items by cuisine but can't pinpoint the specific next business. Content-cosine barely separates relevant cold items (0.667 vs 0.625 warm), so few cold targets even reach the candidate pool. Attempts to force it (content-score ranker feature, cold-feature dropout, round-robin fusion) didn't help and slightly hurt the other slices — you can't rank cold items the retriever never surfaces, and you can't teach a ranker cold behavior when training has zero cold positives. **Text-based cold-start works only when the text is discriminative** (product descriptions, review text, news bodies), not 3-word category lists. *Extension point:* aggregate real Yelp **review text** per business into the item text to raise the ceiling.

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
