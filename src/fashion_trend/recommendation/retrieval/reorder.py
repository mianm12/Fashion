from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fashion_trend.recommendation.retrieval.popularity import SOURCE_COLUMNS


@dataclass(frozen=True)
class PreparedTransactions:
    data: pd.DataFrame
    customer_ids: pd.Index
    article_ids: pd.Index
    customer_code_by_id: pd.Series


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

    prepared_transactions = _prepare_transactions(transactions)
    target_users = _with_string_ids(target_users)
    frames: list[pd.DataFrame] = []
    for window in windows.to_dict("records"):
        window_customer_codes = _target_customer_codes_for_window(
            target_users,
            prepared_transactions,
            window,
        )
        if window_customer_codes.empty:
            continue
        window_transactions = _transactions_for_window(
            prepared_transactions.data,
            cutoff_week=int(window["cutoff_week"]),
            customer_codes=window_customer_codes,
        )
        if window_transactions.empty:
            continue
        ranked = _rank_user_history(
            window_transactions,
            prepared_transactions,
            top_n,
        )
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
    customer_codes: pd.Series,
) -> pd.DataFrame:
    return transactions.loc[
        transactions["week_id"].le(cutoff_week)
        & transactions["customer_code"].isin(customer_codes),
        ["customer_code", "article_code", "week_id"],
    ]


def _rank_user_history(
    transactions: pd.DataFrame,
    prepared_transactions: PreparedTransactions,
    top_n: int,
) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(
            columns=["customer_id", "article_id", "source", "source_rank"]
        )

    ranked = transactions.groupby(
        ["customer_code", "article_code"],
        as_index=False,
        sort=False,
    ).agg(
        last_purchase_week=("week_id", "max"),
        purchase_count=("week_id", "size"),
    )
    ranked["article_id"] = prepared_transactions.article_ids.take(
        ranked["article_code"].to_numpy()
    )
    ranked = ranked.sort_values(
        ["customer_code", "last_purchase_week", "purchase_count", "article_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    limited = ranked.groupby("customer_code", group_keys=False).head(top_n).copy()
    limited["customer_id"] = prepared_transactions.customer_ids.take(
        limited["customer_code"].to_numpy()
    )
    limited["source"] = "reorder"
    limited["source_rank"] = limited.groupby("customer_code").cumcount() + 1
    return limited.loc[:, ["customer_id", "article_id", "source", "source_rank"]]


def _prepare_transactions(transactions: pd.DataFrame) -> PreparedTransactions:
    customer_codes, customer_ids = pd.factorize(transactions["customer_id"], sort=False)
    article_codes, article_ids = pd.factorize(transactions["article_id"], sort=False)
    customer_ids = pd.Index(customer_ids).astype(str)
    article_ids = pd.Index(article_ids).astype(str)
    data = pd.DataFrame(
        {
            "customer_code": customer_codes.astype(np.int32, copy=False),
            "article_code": article_codes.astype(np.int32, copy=False),
            "week_id": pd.to_numeric(transactions["week_id"], errors="raise").astype(
                np.int16
            ),
        }
    )
    customer_code_by_id = pd.Series(
        np.arange(len(customer_ids), dtype=np.int32),
        index=customer_ids,
    )
    return PreparedTransactions(
        data=data,
        customer_ids=customer_ids,
        article_ids=article_ids,
        customer_code_by_id=customer_code_by_id,
    )


def _target_customer_codes_for_window(
    target_users: pd.DataFrame,
    prepared_transactions: PreparedTransactions,
    window: dict[str, object],
) -> pd.Series:
    mask = (
        (target_users["split"] == window["split"])
        & (target_users["cutoff_week"] == window["cutoff_week"])
        & (target_users["label_week"] == window["label_week"])
    )
    customer_ids = target_users.loc[mask, "customer_id"].drop_duplicates()
    customer_codes = prepared_transactions.customer_code_by_id.reindex(customer_ids)
    return customer_codes.dropna().astype(np.int32)


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
