"""Temporal train/test split.

Golden rule of recsys evaluation: train on the past, test on the future. A
random split leaks future information and massively overstates performance.

For each user we sort interactions by time and hold out the most recent
``test_fraction`` as test positives. Only interactions at/above
``positive_threshold`` count as relevant (positive) in the test set.
"""

from __future__ import annotations

from typing import Set

import numpy as np
import pandas as pd

from ..config import settings


def cold_item_holdout(
    train: pd.DataFrame,
    test_positives: dict[str, set],
    cold_fraction: float = 0.15,
    seed: int | None = None,
) -> tuple[pd.DataFrame, Set[str]]:
    """Simulate cold-start items by stripping some items entirely from train.

    Picks ``cold_fraction`` of items that appear in *both* train and the test
    positives (so removal actually matters and there are targets to score), and
    removes all their training interactions. The returned train set has no trace
    of those items — collaborative models cannot represent them, while content
    models still can (via item text). Use with :func:`evaluate_cold_items`.

    Returns ``(train_without_cold_items, cold_item_ids)``.
    """
    i = settings.item_col
    seed = settings.seed if seed is None else seed

    train_items = set(train[i].astype(str).unique())
    test_items: Set[str] = set()
    for items in test_positives.values():
        test_items |= {str(x) for x in items}

    eligible = sorted(train_items & test_items)
    if not eligible:
        return train.copy(), set()

    rng = np.random.default_rng(seed)
    n_cold = max(1, int(round(len(eligible) * cold_fraction)))
    cold_items = set(rng.choice(eligible, size=n_cold, replace=False).tolist())

    keep = ~train[i].astype(str).isin(cold_items)
    return train[keep].reset_index(drop=True), cold_items


def cold_user_holdout(
    train: pd.DataFrame,
    test_positives: dict[str, set],
    cold_fraction: float = 0.15,
    keep: int = 0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, Set[str]]:
    """Simulate cold-start users by stripping (most of) their training history.

    Symmetric to :func:`cold_item_holdout`. Picks ``cold_fraction`` of users that
    appear in both train and the test positives, and keeps only their ``keep``
    earliest training interactions. With ``keep=0`` (default) the user is fully
    new — absent from train — so collaborative models cannot represent them and
    only social / popularity fallbacks can help. Use with
    :func:`evaluate_cold_items` style slicing (see ``scripts/benchmark.py``).

    Returns ``(train_with_cold_user_history_stripped, cold_user_ids)``.
    """
    u, t = settings.user_col, settings.time_col
    seed = settings.seed if seed is None else seed

    train_users = set(train[u].astype(str).unique())
    test_users = {str(x) for x in test_positives.keys()}
    eligible = sorted(train_users & test_users)
    if not eligible:
        return train.copy(), set()

    rng = np.random.default_rng(seed)
    n_cold = max(1, int(round(len(eligible) * cold_fraction)))
    cold_users = set(rng.choice(eligible, size=n_cold, replace=False).tolist())

    is_cold = train[u].astype(str).isin(cold_users)
    if keep <= 0:
        return train[~is_cold].reset_index(drop=True), cold_users

    # keep > 0: retain each cold user's `keep` earliest interactions (sparse user).
    kept_cold = (
        train[is_cold].sort_values([u, t]).groupby(u, sort=False).head(keep)
    )
    out = pd.concat([train[~is_cold], kept_cold]).reset_index(drop=True)
    return out, cold_users


def temporal_split(
    interactions: pd.DataFrame,
    test_fraction: float | None = None,
    positive_threshold: float | None = None,
    min_train: int = 1,
) -> tuple[pd.DataFrame, dict[str, set]]:
    """Split interactions into (train_df, test_positives).

    Parameters
    ----------
    interactions:
        Long-format interactions [user_id, item_id, rating, timestamp].
    test_fraction:
        Fraction of each user's most recent interactions to hold out.
    positive_threshold:
        Minimum rating to be considered a relevant test item.
    min_train:
        Users with fewer than this many training interactions are dropped from
        the *test* ground truth (they still contribute to training).

    Returns
    -------
    train_df:
        Interactions to train on.
    test_positives:
        ``{user_id: {item_id, ...}}`` of held-out relevant items.
    """
    test_fraction = settings.test_fraction if test_fraction is None else test_fraction
    positive_threshold = (
        settings.positive_threshold if positive_threshold is None else positive_threshold
    )

    u, i, t, r = (
        settings.user_col,
        settings.item_col,
        settings.time_col,
        settings.rating_col,
    )

    df = interactions.sort_values([u, t]).reset_index(drop=True)
    train_parts = []
    test_positives: dict[str, set] = {}

    for user_id, grp in df.groupby(u, sort=False):
        n = len(grp)
        n_test = int(round(n * test_fraction))
        if n_test == 0 or n - n_test < min_train:
            train_parts.append(grp)
            continue
        train_parts.append(grp.iloc[: n - n_test])
        test_grp = grp.iloc[n - n_test :]
        positives = set(test_grp.loc[test_grp[r] >= positive_threshold, i])
        if positives:
            test_positives[user_id] = positives

    train_df = pd.concat(train_parts).reset_index(drop=True)
    return train_df, test_positives


def global_temporal_split(
    interactions: pd.DataFrame,
    cutoff_quantile: float = 0.9,
    cutoff: "pd.Timestamp | None" = None,
    positive_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, set]]:
    """Split by a single global time cutoff: train on the past, test on the future.

    Unlike :func:`temporal_split` (which holds out each user's most-recent
    fraction and therefore always leaves every user with training history), this
    uses one wall-clock cutoff shared by everyone. That is what makes cold users
    (and cold items) arise *naturally*: any user or item whose interactions all
    fall after the cutoff never appears in training.

        train = interactions with timestamp <= T
        test  = interactions with timestamp >  T   (positives: rating >= threshold)

    Cold sets are simply derived by the caller as ``test_entities - train_entities``.

    Parameters
    ----------
    cutoff_quantile:
        If ``cutoff`` is not given, T is this quantile of the interaction
        timestamps (0.9 => newest 10% of time is the test window).
    cutoff:
        Explicit cutoff timestamp (overrides ``cutoff_quantile``).
    positive_threshold:
        Minimum rating to count as a relevant test item.

    Returns
    -------
    train_df, test_positives (``{user_id: {item_id, ...}}``).
    """
    positive_threshold = (
        settings.positive_threshold if positive_threshold is None else positive_threshold
    )
    u, i, t, r = (
        settings.user_col,
        settings.item_col,
        settings.time_col,
        settings.rating_col,
    )

    ts = pd.to_datetime(interactions[t])
    if cutoff is None:
        cutoff = ts.quantile(cutoff_quantile)

    is_train = ts <= cutoff
    train_df = interactions[is_train].reset_index(drop=True)
    test_df = interactions[~is_train]

    test_positives: dict[str, set] = {}
    pos = test_df[test_df[r] >= positive_threshold]
    for user_id, grp in pos.groupby(u, sort=False):
        items = set(grp[i])
        if items:
            test_positives[user_id] = items

    return train_df, test_positives
