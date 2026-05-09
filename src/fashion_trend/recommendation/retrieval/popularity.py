from __future__ import annotations

import pandas as pd

SOURCE_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "article_id",
    "source",
    "source_rank",
)


def build_popularity_candidates(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return popularity candidates from history bounded by each cutoff week."""
    return _build_popularity_candidates(
        transactions=transactions,
        windows=windows,
        target_users=target_users,
        top_n=top_n,
        recent_weeks=None,
    )


def build_recent_popularity_candidates(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
    recent_weeks: int = 4,
) -> pd.DataFrame:
    """Return recent popularity candidates from cutoff-bounded history."""
    return _build_popularity_candidates(
        transactions=transactions,
        windows=windows,
        target_users=target_users,
        top_n=top_n,
        recent_weeks=recent_weeks,
    )


def _build_popularity_candidates(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
    recent_weeks: int | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    transactions = _with_string_ids(transactions)
    target_users = _with_string_ids(target_users)
    for window in windows.to_dict("records"):
        window_transactions = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
            recent_weeks=recent_weeks,
        )
        if window_transactions.empty:
            continue
        ranked_articles = _rank_articles_by_count(window_transactions, top_n)
        window_targets = _target_users_for_window(target_users, window)
        frames.append(
            _cross_join_window_targets(window, window_targets, ranked_articles)
        )
    return _concat_source_frames(frames)


def _transactions_for_window(
    transactions: pd.DataFrame,
    cutoff_week: int,
    recent_weeks: int | None,
) -> pd.DataFrame:
    week_id = pd.to_numeric(transactions["week_id"], errors="raise")
    mask = week_id <= cutoff_week
    if recent_weeks is not None:
        mask &= week_id > cutoff_week - recent_weeks
    return transactions.loc[mask].copy()


def _rank_articles_by_count(transactions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = (
        transactions.groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "purchase_count"})
        .sort_values(["purchase_count", "article_id"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked["source"] = "popularity"
    ranked["source_rank"] = ranked.index + 1
    return ranked.loc[:, ["article_id", "source", "source_rank"]]


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


def _cross_join_window_targets(
    window: dict[str, object],
    window_targets: pd.DataFrame,
    ranked_articles: pd.DataFrame,
) -> pd.DataFrame:
    if window_targets.empty or ranked_articles.empty:
        return _empty_source_frame()
    frame = window_targets.merge(ranked_articles, how="cross")
    frame.insert(0, "label_week", window["label_week"])
    frame.insert(0, "cutoff_week", window["cutoff_week"])
    frame.insert(0, "split", window["split"])
    return frame.loc[:, SOURCE_COLUMNS]


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
