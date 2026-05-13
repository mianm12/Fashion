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

    transactions = _prepare_transactions(transactions)
    target_users = _with_string_ids(target_users)
    frames: list[pd.DataFrame] = []
    for window in windows.to_dict("records"):
        window_targets = _target_users_for_window(target_users, window)
        if window_targets.empty:
            continue
        window_transactions = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
            customer_ids=window_targets["customer_id"],
        )
        if window_transactions.empty:
            continue
        ranked = _rank_user_history(window_transactions, top_n)
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
    customer_ids: pd.Series,
) -> pd.DataFrame:
    return transactions.loc[
        transactions["week_id"].le(cutoff_week)
        & transactions["customer_id"].isin(customer_ids),
        ["customer_id", "article_id", "week_id"],
    ]


def _rank_user_history(
    transactions: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(
            columns=["customer_id", "article_id", "source", "source_rank"]
        )

    ranked = (
        transactions.groupby(["customer_id", "article_id"], as_index=False, sort=False)
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


def _prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    result = transactions.loc[:, ["customer_id", "article_id", "week_id"]].copy()
    result["week_id"] = pd.to_numeric(result["week_id"], errors="raise").astype(int)
    return result


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
