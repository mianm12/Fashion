from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.retrieval.popularity import SOURCE_COLUMNS


def build_age_popularity_candidates(
    transactions: pd.DataFrame,
    customer_profile: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    *,
    pool_top_n: int = 50,
    per_user_top_n: int = 12,
    recent_weeks: int = 4,
) -> pd.DataFrame:
    """Return recent popularity candidates within each target user's age bucket."""
    if transactions.empty or customer_profile.empty or target_users.empty:
        return _empty_source_frame()

    transactions = _with_string_ids(transactions)
    profile = _profile_buckets(customer_profile)
    target_users = _with_string_ids(target_users)
    if profile.empty:
        return _empty_source_frame()

    frames: list[pd.DataFrame] = []
    for window in windows.to_dict("records"):
        window_targets = _target_users_with_bucket(target_users, profile, window)
        if window_targets.empty:
            continue
        window_transactions = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
            recent_weeks=recent_weeks,
        )
        ranked_articles = _rank_bucket_articles(
            window_transactions,
            profile,
            pool_top_n,
        )
        if ranked_articles.empty:
            continue
        ranked = _rank_targets(window_targets, ranked_articles, per_user_top_n)
        if ranked.empty:
            continue
        ranked.insert(0, "label_week", window["label_week"])
        ranked.insert(0, "cutoff_week", window["cutoff_week"])
        ranked.insert(0, "split", window["split"])
        frames.append(ranked.loc[:, SOURCE_COLUMNS])
    return _concat_source_frames(frames)


def _profile_buckets(customer_profile: pd.DataFrame) -> pd.DataFrame:
    profile = customer_profile.loc[:, ["customer_id", "age_bucket"]].copy()
    profile = profile.loc[profile["age_bucket"].notna()].copy()
    profile["customer_id"] = profile["customer_id"].astype(str)
    profile["age_bucket"] = profile["age_bucket"].astype(str)
    profile = profile.loc[profile["age_bucket"] != ""]
    return profile.drop_duplicates("customer_id").reset_index(drop=True)


def _transactions_for_window(
    transactions: pd.DataFrame,
    cutoff_week: int,
    recent_weeks: int,
) -> pd.DataFrame:
    week_id = pd.to_numeric(transactions["week_id"], errors="raise")
    mask = (week_id <= cutoff_week) & (week_id > cutoff_week - recent_weeks)
    return transactions.loc[mask].copy()


def _rank_bucket_articles(
    transactions: pd.DataFrame,
    profile: pd.DataFrame,
    pool_top_n: int,
) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["age_bucket", "article_id", "bucket_rank"])

    history = transactions.merge(profile, on="customer_id", how="inner")
    if history.empty:
        return pd.DataFrame(columns=["age_bucket", "article_id", "bucket_rank"])

    ranked = (
        history.groupby(["age_bucket", "article_id"], as_index=False)
        .size()
        .rename(columns={"size": "purchase_count"})
        .sort_values(
            ["age_bucket", "purchase_count", "article_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    limited = ranked.groupby("age_bucket", group_keys=False).head(pool_top_n).copy()
    limited["bucket_rank"] = limited.groupby("age_bucket").cumcount() + 1
    return limited.loc[:, ["age_bucket", "article_id", "bucket_rank"]]


def _target_users_with_bucket(
    target_users: pd.DataFrame,
    profile: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    mask = (
        (target_users["split"] == window["split"])
        & (target_users["cutoff_week"] == window["cutoff_week"])
        & (target_users["label_week"] == window["label_week"])
    )
    targets = target_users.loc[mask, ["customer_id"]].drop_duplicates().copy()
    return targets.merge(profile, on="customer_id", how="inner")


def _rank_targets(
    window_targets: pd.DataFrame,
    ranked_articles: pd.DataFrame,
    per_user_top_n: int,
) -> pd.DataFrame:
    candidates = window_targets.merge(ranked_articles, on="age_bucket", how="inner")
    if candidates.empty:
        return pd.DataFrame(
            columns=["customer_id", "article_id", "source", "source_rank"]
        )
    sorted_candidates = candidates.sort_values(
        ["customer_id", "bucket_rank", "article_id"],
        kind="mergesort",
    )
    limited = (
        sorted_candidates.groupby("customer_id", group_keys=False)
        .head(per_user_top_n)
        .copy()
    )
    limited["source"] = "age_popularity"
    limited["source_rank"] = limited.groupby("customer_id").cumcount() + 1
    return limited.loc[:, ["customer_id", "article_id", "source", "source_rank"]]


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
