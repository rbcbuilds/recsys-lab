"""Carve a small, dense slice out of the full Yelp Open Dataset.

The full dataset is several GB of line-delimited JSON. This reads it in a
streaming fashion, keeps one city and only sufficiently active users/items,
then writes tidy parquet with the SAME schema the loaders expect. That gives
MovieLens-like iteration speed while preserving real social edges + text.

Usage:
    python scripts/make_subset.py --city "Santa Barbara" \
        --min-user-reviews 10 --min-item-reviews 10
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, RAW_DIR, settings

REVIEW_FILE = "yelp_academic_dataset_review.json"
BUSINESS_FILE = "yelp_academic_dataset_business.json"
USER_FILE = "yelp_academic_dataset_user.json"


def _iter_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_subset(
    city: str | None = None,
    min_user_reviews: int = 10,
    min_item_reviews: int = 10,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    raw_dir = Path(raw_dir)
    business_path = raw_dir / BUSINESS_FILE
    review_path = raw_dir / REVIEW_FILE
    user_path = raw_dir / USER_FILE

    for p in (business_path, review_path, user_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Download the Yelp Open Dataset into {raw_dir} "
                f"(see data/README.md)."
            )

    # 1. Businesses (optionally filter by city) -> item universe
    biz_rows = []
    for b in _iter_json(business_path):
        if city and (b.get("city") or "").strip().lower() != city.strip().lower():
            continue
        biz_rows.append(
            {
                settings.item_col: b["business_id"],
                "name": b.get("name", ""),
                "categories": b.get("categories") or "",
                "city": b.get("city", ""),
                "text": b.get("name", "") + ". " + (b.get("categories") or ""),
            }
        )
    items = pd.DataFrame(biz_rows)
    if items.empty:
        raise ValueError(f"No businesses found for city={city!r}.")
    keep_items = set(items[settings.item_col])

    # 2. Reviews limited to those businesses -> interactions
    rev_rows = []
    for r in _iter_json(review_path):
        if r["business_id"] not in keep_items:
            continue
        rev_rows.append(
            {
                settings.user_col: r["user_id"],
                settings.item_col: r["business_id"],
                settings.rating_col: float(r.get("stars", 0)),
                settings.time_col: r.get("date"),
            }
        )
    interactions = pd.DataFrame(rev_rows)
    interactions[settings.time_col] = pd.to_datetime(interactions[settings.time_col])

    # 3. Iteratively prune to dense core (users & items with enough activity)
    interactions = _densify(interactions, min_user_reviews, min_item_reviews)
    keep_users = set(interactions[settings.user_col])
    keep_items = set(interactions[settings.item_col])
    items = items[items[settings.item_col].isin(keep_items)].reset_index(drop=True)

    # 4. Users + social graph (only edges within the kept user set)
    user_rows = []
    social_rows = []
    for u in _iter_json(user_path):
        uid = u["user_id"]
        if uid not in keep_users:
            continue
        user_rows.append({settings.user_col: uid, "review_count": u.get("review_count", 0)})
        friends = (u.get("friends") or "").split(", ")
        for fid in friends:
            fid = fid.strip()
            if fid and fid in keep_users and fid != uid:
                a, b = sorted((uid, fid))
                social_rows.append((a, b))
    users = pd.DataFrame(user_rows)
    social = pd.DataFrame(
        sorted(set(social_rows)), columns=[settings.user_col, "friend_id"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    social.to_parquet(out_dir / "social.parquet", index=False)

    density = len(interactions) / (len(users) * len(items) or 1)
    print(
        f"Wrote Yelp subset to {out_dir}:\n"
        f"  users={len(users):,} items={len(items):,} "
        f"interactions={len(interactions):,} density={density:.4%} "
        f"social_edges={len(social):,}"
    )
    print("Note: photos/image vectors are not built here — see models/multimodal.py.")


def build_crossregime_subset(
    city: str = "Philadelphia",
    min_user_reviews: int = 10,
    min_item_reviews: int = 10,
    max_core_users: int = 2000,
    cutoff_quantile: float = 0.9,
    max_cold_items: int = 500,
    max_cold_users: int = 500,
    tail_review_min: int = 1,
    tail_review_max: int = 8,
    raw_dir: Path = RAW_DIR,
    out_dir: Path | None = None,
    seed: int | None = None,
) -> None:
    """Build the *cross-regime* slice: a dense warm core + a real cold tail.

    The point of this dataset is that cold users and cold items arise
    **naturally** under a single global-time split, instead of being carved out
    by simulation. Construction (one streaming pass over the raw city reviews):

    1. Densify the city reviews to a warm core (``min_user``/``min_item``
       thresholds) and cap to the ``max_core_users`` most active users.
    2. Pick a cutoff ``T`` = ``cutoff_quantile`` of the core's timestamps.
    3. Inject a low-activity tail sampled from the *raw* city reviews:
         * cold items: first reviewed after ``T``, total city reviews in
           ``[tail_review_min, tail_review_max]``, reviewed by >= 1 core user
           (so warm users have them as future targets);
         * cold users: first active after ``T``, total reviews in the same
           window, reviewing >= 1 core item (so their targets are scorable by
           popularity/social).
    4. Merge core + tail interactions/items/users/social, same schema as
       :func:`build_subset`.

    Evaluate with :func:`recsys.eval.global_temporal_split` at the same ``T``:
    everything first-seen after ``T`` has zero training history, hence cold.
    """
    from ..config import settings as _settings

    seed = _settings.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    out_dir = (RAW_DIR.parent / "processed_philly_xreg") if out_dir is None else Path(out_dir)

    raw_dir = Path(raw_dir)
    business_path = raw_dir / BUSINESS_FILE
    review_path = raw_dir / REVIEW_FILE
    user_path = raw_dir / USER_FILE
    for p in (business_path, review_path, user_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Download the Yelp Open Dataset into {raw_dir}."
            )

    u, i, t, r = (
        _settings.user_col,
        _settings.item_col,
        _settings.time_col,
        _settings.rating_col,
    )

    # 1. City businesses -> item universe (+ text).
    biz_rows = []
    for b in _iter_json(business_path):
        if (b.get("city") or "").strip().lower() != city.strip().lower():
            continue
        biz_rows.append(
            {
                i: b["business_id"],
                "name": b.get("name", ""),
                "categories": b.get("categories") or "",
                "city": b.get("city", ""),
                "text": b.get("name", "") + ". " + (b.get("categories") or ""),
            }
        )
    items_all = pd.DataFrame(biz_rows)
    if items_all.empty:
        raise ValueError(f"No businesses found for city={city!r}.")
    city_items = set(items_all[i])

    # 2. All city reviews (unfiltered) -> we need the full tail, not just a core.
    rev_rows = []
    for rev in _iter_json(review_path):
        if rev["business_id"] not in city_items:
            continue
        rev_rows.append(
            {
                u: rev["user_id"],
                i: rev["business_id"],
                r: float(rev.get("stars", 0)),
                t: rev.get("date"),
            }
        )
    city_inter = pd.DataFrame(rev_rows)
    city_inter[t] = pd.to_datetime(city_inter[t])

    # 3. Dense warm core + cap to most active users.
    core = _densify(city_inter, min_user_reviews, min_item_reviews)
    top_users = core[u].value_counts().head(max_core_users).index
    core = core[core[u].isin(top_users)]
    core = _densify(core, min_user_reviews, min_item_reviews)
    core_users = set(core[u])
    core_items = set(core[i])

    # 4. Global cutoff from the core's time distribution.
    cutoff = core[t].quantile(cutoff_quantile)

    # 5. Per-item / per-user tail stats over ALL city reviews.
    item_first = city_inter.groupby(i)[t].min()
    item_count = city_inter.groupby(i).size()
    user_first = city_inter.groupby(u)[t].min()
    user_count = city_inter.groupby(u).size()

    in_window = lambda c: (c >= tail_review_min) & (c <= tail_review_max)

    # Cold items: born after T, low activity, reviewed by >= 1 core user.
    cand_items = item_first[(item_first > cutoff) & in_window(item_count)].index
    cand_items = [x for x in cand_items if x not in core_items]
    reviewed_by_core = set(
        city_inter[city_inter[u].isin(core_users)][i].unique()
    )
    cand_items = [x for x in cand_items if x in reviewed_by_core]
    if len(cand_items) > max_cold_items:
        cand_items = rng.choice(
            np.array(cand_items), size=max_cold_items, replace=False
        ).tolist()
    cold_items = set(cand_items)

    # Cold users: born after T, low activity, reviewing >= 1 core item.
    cand_users = user_first[(user_first > cutoff) & in_window(user_count)].index
    cand_users = [x for x in cand_users if x not in core_users]
    reviews_core_item = set(
        city_inter[city_inter[i].isin(core_items)][u].unique()
    )
    cand_users = [x for x in cand_users if x in reviews_core_item]
    if len(cand_users) > max_cold_users:
        cand_users = rng.choice(
            np.array(cand_users), size=max_cold_users, replace=False
        ).tolist()
    cold_users = set(cand_users)

    # 6. Merge interactions:
    #    core  +  (cold-item reviews by core users)  +  (cold-user reviews of core items)
    cold_item_rows = city_inter[
        city_inter[i].isin(cold_items) & city_inter[u].isin(core_users)
    ]
    cold_user_rows = city_inter[
        city_inter[u].isin(cold_users) & city_inter[i].isin(core_items)
    ]
    merged = pd.concat([core, cold_item_rows, cold_user_rows]).drop_duplicates(
        subset=[u, i, t]
    )
    merged = merged.sort_values([u, t]).reset_index(drop=True)

    keep_users = set(merged[u])
    keep_items = set(merged[i])
    items = items_all[items_all[i].isin(keep_items)].reset_index(drop=True)

    # 7. Users + social graph (edges within kept users only).
    user_rows = []
    social_rows = []
    for uu in _iter_json(user_path):
        uid = uu["user_id"]
        if uid not in keep_users:
            continue
        user_rows.append({u: uid, "review_count": uu.get("review_count", 0)})
        for fid in (uu.get("friends") or "").split(", "):
            fid = fid.strip()
            if fid and fid in keep_users and fid != uid:
                a, b = sorted((uid, fid))
                social_rows.append((a, b))
    users = pd.DataFrame(user_rows)
    social = pd.DataFrame(
        sorted(set(social_rows)), columns=[u, "friend_id"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "interactions.parquet", index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    social.to_parquet(out_dir / "social.parquet", index=False)

    # Report the cold sets *as they will appear under a global split at T*.
    train_users = set(merged[merged[t] <= cutoff][u])
    train_items = set(merged[merged[t] <= cutoff][i])
    test = merged[merged[t] > cutoff]
    nat_cold_items = set(test[i]) - train_items
    nat_cold_users = set(test[u]) - train_users
    density = len(merged) / (len(users) * len(items) or 1)
    print(
        f"Wrote cross-regime slice to {out_dir}:\n"
        f"  users={len(users):,} items={len(items):,} "
        f"interactions={len(merged):,} density={density:.4%} "
        f"social_edges={len(social):,}\n"
        f"  cutoff T={pd.Timestamp(cutoff)} (q={cutoff_quantile})\n"
        f"  injected cold items={len(cold_items):,} cold users={len(cold_users):,}\n"
        f"  under global split @T -> cold items={len(nat_cold_items):,} "
        f"cold users={len(nat_cold_users):,}"
    )


def _score_warm_users_social_seq(
    pre: pd.DataFrame,
    warm_users: set,
    social: pd.DataFrame | None,
    u: str,
    t: str,
    *,
    w_social: float = 0.35,
    w_embed: float = 0.25,
    w_pre: float = 0.25,
    w_seq: float = 0.15,
) -> pd.Series:
    """Rank warm users for retention: social degree, friend activity, sequence quality."""
    warm = sorted(warm_users)
    pre_cnt = pre.groupby(u).size().reindex(warm, fill_value=0).astype(float)

    social_deg = pd.Series(0.0, index=warm)
    embed = pd.Series(0.0, index=warm)
    if social is not None and len(social):
        edges = pd.concat(
            [
                social[[u, "friend_id"]].rename(columns={u: "a", "friend_id": "b"}),
                social[[u, "friend_id"]].rename(columns={"friend_id": "a", u: "b"}),
            ],
            ignore_index=True,
        )
        social_deg = (
            edges.groupby("a")
            .size()
            .reindex(warm, fill_value=0)
            .astype(float)
        )
        friend_lists = edges.groupby("a")["b"].apply(list)
        embed = pd.Series(
            {
                uid: sum(pre_cnt.get(f, 0) for f in friend_lists.get(uid, ()))
                for uid in warm
            },
            dtype=float,
        )

    seq_scores = pd.Series(0.0, index=warm)
    week_sec = 7 * 86400
    for uid in warm:
        times = pd.to_datetime(pre.loc[pre[u] == uid, t]).sort_values()
        n = len(times)
        if n < 2:
            continue
        span = max((times.iloc[-1] - times.iloc[0]).total_seconds(), week_sec)
        months = span / (30 * 86400)
        density = n / months
        gaps = times.diff().dt.total_seconds().dropna()
        regularity = 1.0 / (1.0 + gaps.median() / week_sec) if len(gaps) else 0.0
        seq_scores[uid] = density * regularity

    def _norm(s: pd.Series) -> pd.Series:
        hi = float(s.max())
        return s / hi if hi > 0 else s * 0.0

    score = (
        w_social * _norm(social_deg)
        + w_embed * _norm(embed)
        + w_pre * _norm(pre_cnt)
        + w_seq * _norm(seq_scores)
    )
    return score.sort_values(ascending=False)


def downsample_crossregime_processed(
    max_per_user: int = 20,
    max_warm_users: int | None = None,
    user_select: str = "activity",
    cutoff_quantile: float = 0.9,
    processed_dir: Path = PROCESSED_DIR,
    out_dir: Path | None = None,
) -> None:
    """Shrink a cross-regime slice for faster benchmarks without losing cold tail.

    Caps each user's *pre-cutoff* history to their ``max_per_user`` most recent
    interactions (where ~90% of rows live). Post-cutoff rows are kept in full —
    that is where cold users, cold items, and test positives live, so the
    global-time eval story stays intact. No re-densify step (cold entities are
    low-activity by design and would be dropped by ``min_item_reviews=10``).

    When ``max_warm_users`` is set, retain that many *warm* users (seen before
    the cutoff) and always keep every cold user. ``user_select`` chooses how
    warm users are ranked:

    * ``activity`` — most pre-cutoff reviews (default).
    * ``social_seq`` — composite of social degree, friends' activity, review
      count, and sequential density/regularity (better for social + SASRec).
    """
    from ..config import settings

    if user_select not in ("activity", "social_seq"):
        raise ValueError(f"user_select must be 'activity' or 'social_seq', got {user_select!r}")

    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir) if out_dir is not None else processed_dir
    u, i, t = settings.user_col, settings.item_col, settings.time_col

    inter = pd.read_parquet(processed_dir / "interactions.parquet")
    social_path = processed_dir / "social.parquet"
    social = pd.read_parquet(social_path) if social_path.exists() else None

    ts = pd.to_datetime(inter[t])
    cutoff = ts.quantile(cutoff_quantile)

    pre = inter[ts <= cutoff]
    post = inter[ts > cutoff]
    warm_users = set(pre[u])
    cold_users = set(post[u]) - warm_users

    if max_warm_users is not None:
        if user_select == "activity":
            ranked = pre.groupby(u).size().sort_values(ascending=False)
        else:
            if social is None:
                raise ValueError("social_seq selection requires social.parquet in processed_dir.")
            ranked = _score_warm_users_social_seq(pre, warm_users, social, u, t)
        selected_warm = set(ranked.head(max_warm_users).index.astype(str))
        keep_users = selected_warm | cold_users
        inter = inter[inter[u].isin(keep_users)]

    ts = pd.to_datetime(inter[t])
    pre = inter[ts <= cutoff]
    post = inter[ts > cutoff]
    pre_cap = pre.sort_values(t).groupby(u, group_keys=False).tail(max_per_user)
    shrunk = (
        pd.concat([pre_cap, post])
        .drop_duplicates(subset=[u, i, t])
        .sort_values([u, t])
        .reset_index(drop=True)
    )

    keep_users = set(shrunk[u])
    keep_items = set(shrunk[i])
    n_warm = len(keep_users - cold_users)
    n_cold = len(keep_users & cold_users)

    items = pd.read_parquet(processed_dir / "items.parquet")
    items = items[items[i].isin(keep_items)].reset_index(drop=True)
    users = pd.read_parquet(processed_dir / "users.parquet")
    users = users[users[u].isin(keep_users)].reset_index(drop=True)

    if social is not None:
        social = social[
            social[u].isin(keep_users) & social["friend_id"].isin(keep_users)
        ].reset_index(drop=True)
        n_social = len(social)
    else:
        n_social = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    shrunk.to_parquet(out_dir / "interactions.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    if social is not None:
        social.to_parquet(out_dir / "social.parquet", index=False)

    train_users = set(shrunk[pd.to_datetime(shrunk[t]) <= cutoff][u])
    train_items = set(shrunk[pd.to_datetime(shrunk[t]) <= cutoff][i])
    test = shrunk[pd.to_datetime(shrunk[t]) > cutoff]
    nat_cold_items = set(test[i]) - train_items
    nat_cold_users = set(test[u]) - train_users
    density = len(shrunk) / (len(users) * len(items) or 1)
    select_note = (
        f"  warm_users={n_warm:,} (select={user_select}, cap={max_warm_users}), "
        f"cold_users={n_cold:,} (pinned)\n"
        if max_warm_users is not None
        else ""
    )
    print(
        f"Cross-regime downsample {'-> ' + str(out_dir) if out_dir != processed_dir else 'in ' + str(out_dir)}:\n"
        f"  users={len(users):,} items={len(items):,} "
        f"interactions={len(shrunk):,} density={density:.4%} "
        f"social_edges={n_social:,}\n"
        f"{select_note}"
        f"  warm cap: max_per_user={max_per_user} pre-cutoff only "
        f"(cutoff q={cutoff_quantile}, T={pd.Timestamp(cutoff)})\n"
        f"  under global split @T -> cold items={len(nat_cold_items):,} "
        f"cold users={len(nat_cold_users):,}"
    )


def downsample_processed(
    max_users: int = 2500,
    max_per_user: int | None = None,
    min_user_reviews: int = 10,
    min_item_reviews: int = 10,
    processed_dir: Path = PROCESSED_DIR,
    seed: int | None = None,
) -> None:
    """Shrink an already-built processed subset in place (fast, no raw rescan).

    Keeps the ``max_users`` most active users, optionally caps each user to their
    ``max_per_user`` most recent interactions (the biggest run-time lever, since
    a few "power users" carry most of the interactions), re-densifies, and
    filters items, users, and the social graph to match. Use this to trade
    dataset size for run time without re-parsing the multi-GB raw Yelp files.
    """
    from ..config import settings

    processed_dir = Path(processed_dir)
    inter = pd.read_parquet(processed_dir / "interactions.parquet")
    u, i = settings.user_col, settings.item_col

    # Keep the most active users (stable, deterministic).
    top_users = inter[u].value_counts().head(max_users).index
    inter = inter[inter[u].isin(top_users)]

    # Cap per-user history to the most recent interactions. This is what
    # actually shrinks the interaction count when a few users review hundreds
    # of items, and it keeps the temporal-split evaluation meaningful.
    if max_per_user is not None and settings.time_col in inter.columns:
        inter = (
            inter.sort_values(settings.time_col)
            .groupby(u, group_keys=False)
            .tail(max_per_user)
        )

    # Re-densify so items/users still meet the interaction thresholds.
    inter = _densify(inter, min_user_reviews, min_item_reviews)
    keep_users = set(inter[u])
    keep_items = set(inter[i])

    items = pd.read_parquet(processed_dir / "items.parquet")
    items = items[items[i].isin(keep_items)].reset_index(drop=True)
    users = pd.read_parquet(processed_dir / "users.parquet")
    users = users[users[u].isin(keep_users)].reset_index(drop=True)

    social_path = processed_dir / "social.parquet"
    if social_path.exists():
        social = pd.read_parquet(social_path)
        social = social[
            social[u].isin(keep_users) & social["friend_id"].isin(keep_users)
        ].reset_index(drop=True)
        social.to_parquet(social_path, index=False)
        n_social = len(social)
    else:
        n_social = 0

    inter.to_parquet(processed_dir / "interactions.parquet", index=False)
    items.to_parquet(processed_dir / "items.parquet", index=False)
    users.to_parquet(processed_dir / "users.parquet", index=False)

    density = len(inter) / (len(users) * len(items) or 1)
    print(
        f"Downsampled processed subset in {processed_dir}:\n"
        f"  users={len(users):,} items={len(items):,} "
        f"interactions={len(inter):,} density={density:.4%} "
        f"social_edges={n_social:,}"
    )


def _densify(interactions: pd.DataFrame, min_u: int, min_i: int) -> pd.DataFrame:
    """Iteratively drop users/items below the interaction thresholds."""
    prev = -1
    while len(interactions) != prev:
        prev = len(interactions)
        uc = interactions[settings.user_col].value_counts()
        interactions = interactions[
            interactions[settings.user_col].isin(uc[uc >= min_u].index)
        ]
        ic = interactions[settings.item_col].value_counts()
        interactions = interactions[
            interactions[settings.item_col].isin(ic[ic >= min_i].index)
        ]
    return interactions.reset_index(drop=True)
