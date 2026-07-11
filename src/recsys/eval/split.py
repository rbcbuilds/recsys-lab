"""Temporal train/test split.

Golden rule of recsys evaluation: train on the past, test on the future. A
random split leaks future information and massively overstates performance.

For each user we sort interactions by time and hold out the most recent
``test_fraction`` as test positives. Only interactions at/above
``positive_threshold`` count as relevant (positive) in the test set.
"""

from __future__ import annotations

import pandas as pd

from ..config import settings


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
