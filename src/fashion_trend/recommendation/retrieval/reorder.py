from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.retrieval.popularity import SOURCE_COLUMNS


def build_reorder_candidates(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    *,
    top_n: int = 12,
) -> pd.DataFrame:
    """Return user-specific reorder candidates from cutoff-bounded history."""
    if transactions.empty or target_users.empty:
        return _empty_source_frame()

    transactions = _with_string_ids(transactions)
    target_users = _with_string_ids(target_users)
    frames: list[pd.DataFrame] = []
    for window in windows.to_dict("records"):
        window_targets = _target_users_for_window(target_users, window)
        if window_targets.empty:
            continue
        window_transactions = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
        )
        if window_transactions.empty:
            continue
        ranked = _rank_user_history(window_transactions, window_targets, top_n)
        if ranked.empty:
            continue
        ranked.insert(0, "label_week", window["label_week"])
        ranked.insert(0, "cutoff_week", window["cutoff_week"])
        ranked.insert(0, "split", window["split"])
        frames.append(ranked.loc[:, SOURCE_COLUMNS])
    return _concat_source_frames(frames)


def _transactions_for_window(
    transactions: pd.DataFrame,
    cutoff_week: int,
) -> pd.DataFrame:
    week_id = pd.to_numeric(transactions["week_id"], errors="raise")
    return transactions.loc[week_id <= cutoff_week].copy()


def _rank_user_history(
    transactions: pd.DataFrame,
    window_targets: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    history = transactions.merge(window_targets, on="customer_id", how="inner")
    if history.empty:
        return pd.DataFrame(
            columns=["customer_id", "article_id", "source", "source_rank"]
        )

    ranked = (
        history.assign(week_id=pd.to_numeric(history["week_id"], errors="raise"))
        .groupby(["customer_id", "article_id"], as_index=False)
        .agg(
            last_purchase_week=("week_id", "max"),
            purchase_count=("week_id", "size"),
        )
        .sort_values(
            ["customer_id", "last_purchase_week", "purchase_count", "article_id"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
    )
    limited = ranked.groupby("customer_id", group_keys=False).head(top_n).copy()
    limited["source"] = "reorder"
    limited["source_rank"] = limited.groupby("customer_id").cumcount() + 1
    return limited.loc[:, ["customer_id", "article_id", "source", "source_rank"]]


def _target_users_for_window(
    target_users: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    mask = (
        (target_users["split"] == window["split"])
        & (target_users["cutoff_week"] == window["cutoff_week"])
        & (target_users["label_week"] == window["label_week"])
    )
    return target_users.loc[mask, ["customer_id"]].drop_duplicates().copy()


def _concat_source_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_source_frame()
    result = pd.concat(non_empty, ignore_index=True)
    return _with_string_ids(result).loc[:, SOURCE_COLUMNS]


def _empty_source_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SOURCE_COLUMNS)


def _with_string_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("article_id", "customer_id"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result
