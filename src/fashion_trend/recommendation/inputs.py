from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fashion_trend.foundation.io import write_parquet_atomic
from fashion_trend.recommendation.contracts import (
    EVALUATION_LABEL_COLUMNS,
    EVALUATION_LABEL_KEY_COLUMNS,
    RECOMMENDATION_CORE_ATTR_TYPES,
    RECOMMENDATION_PROFILE_TOP_ATTRIBUTES,
    TARGET_USER_COLUMNS,
    TARGET_USER_KEY_COLUMNS,
    TIME_WINDOW_COLUMNS,
    USER_PROFILE_COLUMNS,
    USER_PROFILE_KEY_COLUMNS,
)
from fashion_trend.recommendation.paths import (
    EVALUATION_LABELS_PATH,
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    USER_PROFILE_PATH,
)
from fashion_trend.recommendation.readers import reject_duplicate_key, validate_columns
from fashion_trend.recommendation.time_windows import build_recommendation_windows


@dataclass(frozen=True)
class RecommendationInputArtifacts:
    time_windows: pd.DataFrame
    target_users: pd.DataFrame
    evaluation_labels: pd.DataFrame
    user_profile: pd.DataFrame


def build_target_users(
    transactions: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    """Build eligible recommendation users for each label window."""
    validate_columns(windows, TIME_WINDOW_COLUMNS, "time_windows")

    frames: list[pd.DataFrame] = []
    transactions = _coerce_text_columns(transactions)
    for window in windows.itertuples(index=False):
        history = transactions.loc[transactions["week_id"] <= window.cutoff_week]
        labels = transactions.loc[transactions["week_id"] == window.label_week]
        history_counts = (
            history.groupby("customer_id").size().rename("history_purchase_count")
        )
        label_counts = (
            labels.groupby("customer_id").size().rename("label_purchase_count")
        )
        eligible = (
            pd.concat([history_counts, label_counts], axis=1).dropna().reset_index()
        )
        if eligible.empty:
            continue
        eligible["history_purchase_count"] = eligible["history_purchase_count"].astype(
            "int64"
        )
        eligible["label_purchase_count"] = eligible["label_purchase_count"].astype(
            "int64"
        )
        eligible = eligible.assign(
            split=window.split,
            cutoff_week=window.cutoff_week,
            label_week=window.label_week,
        )
        frames.append(eligible.loc[:, list(TARGET_USER_COLUMNS)])

    result = _concat_or_empty(frames, TARGET_USER_COLUMNS)
    result = _coerce_text_columns(result)
    result = result.sort_values(
        ["split", "cutoff_week", "label_week", "customer_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    reject_duplicate_key(result, TARGET_USER_KEY_COLUMNS, "target_users")
    return result


def build_evaluation_labels(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
) -> pd.DataFrame:
    """Build deduplicated label-week purchases for eligible users."""
    validate_columns(windows, TIME_WINDOW_COLUMNS, "time_windows")

    frames: list[pd.DataFrame] = []
    transactions = _coerce_text_columns(transactions)
    target_users = _coerce_text_columns(target_users)
    for window in windows.itertuples(index=False):
        labels = transactions.loc[transactions["week_id"] == window.label_week]
        eligible = _users_for_window(target_users, window)
        merged = labels.merge(eligible[["customer_id"]], on="customer_id", how="inner")
        if merged.empty:
            continue
        merged = merged.assign(
            split=window.split,
            cutoff_week=window.cutoff_week,
            label_week=window.label_week,
        )
        frames.append(merged.loc[:, list(EVALUATION_LABEL_COLUMNS)])

    result = _concat_or_empty(frames, EVALUATION_LABEL_COLUMNS)
    result = _coerce_text_columns(result)
    result = result.drop_duplicates(list(EVALUATION_LABEL_KEY_COLUMNS))
    result = result.sort_values(
        ["split", "cutoff_week", "label_week", "customer_id", "article_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    reject_duplicate_key(result, EVALUATION_LABEL_KEY_COLUMNS, "evaluation_labels")
    return result


def build_user_profile(
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
) -> pd.DataFrame:
    """Build attribute preference profiles from cutoff-week purchase history."""
    validate_columns(windows, TIME_WINDOW_COLUMNS, "time_windows")

    frames: list[pd.DataFrame] = []
    transactions = _coerce_text_columns(transactions)
    article_attributes = _coerce_text_columns(article_attributes)
    article_attributes = article_attributes.loc[
        article_attributes["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES)
    ].copy()
    target_users = _coerce_text_columns(target_users)
    for window in windows.itertuples(index=False):
        eligible = _users_for_window(target_users, window)
        history = transactions.loc[transactions["week_id"] <= window.cutoff_week]
        history = history.merge(
            eligible[["customer_id"]], on="customer_id", how="inner"
        )
        if history.empty:
            continue

        profile = history.merge(article_attributes, on="article_id", how="inner")
        if profile.empty:
            continue

        profile = profile.groupby(
            ["customer_id", "attr_id", "attr_type", "attr_value"],
            as_index=False,
            sort=True,
        ).agg(
            purchase_count=("article_id", "size"),
            last_purchase_week=("week_id", "max"),
        )
        totals = profile.groupby("customer_id")["purchase_count"].transform("sum")
        profile = profile.assign(
            split=window.split,
            cutoff_week=window.cutoff_week,
            label_week=window.label_week,
            preference_score=profile["purchase_count"] / totals,
        )
        profile = _limit_profile_attributes(profile)
        frames.append(profile.loc[:, list(USER_PROFILE_COLUMNS)])

    result = _concat_or_empty(frames, USER_PROFILE_COLUMNS)
    result = _coerce_text_columns(result)
    result = result.sort_values(
        [
            "split",
            "cutoff_week",
            "label_week",
            "customer_id",
            "attr_id",
            "attr_type",
            "attr_value",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    reject_duplicate_key(result, USER_PROFILE_KEY_COLUMNS, "user_profile")
    return result


def build_and_write_recommendation_inputs(
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    trend_predictions: pd.DataFrame,
) -> RecommendationInputArtifacts:
    """Build and write recommendation input artifacts."""
    windows = build_recommendation_windows(trend_predictions)
    target_users = build_target_users(transactions, windows)
    labels = build_evaluation_labels(transactions, windows, target_users)
    profile = build_user_profile(
        transactions, article_attributes, windows, target_users
    )

    write_parquet_atomic(windows, TIME_WINDOWS_PATH)
    write_parquet_atomic(target_users, TARGET_USERS_PATH)
    write_parquet_atomic(labels, EVALUATION_LABELS_PATH)
    write_parquet_atomic(profile, USER_PROFILE_PATH)
    return RecommendationInputArtifacts(windows, target_users, labels, profile)


def _users_for_window(target_users: pd.DataFrame, window: object) -> pd.DataFrame:
    return target_users.loc[
        (target_users["split"] == window.split)
        & (target_users["cutoff_week"] == window.cutoff_week)
        & (target_users["label_week"] == window.label_week)
    ]


def _concat_or_empty(
    frames: list[pd.DataFrame],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True).loc[:, list(columns)]


def _limit_profile_attributes(profile: pd.DataFrame) -> pd.DataFrame:
    sorted_profile = profile.sort_values(
        [
            "customer_id",
            "preference_score",
            "purchase_count",
            "last_purchase_week",
            "attr_type",
            "attr_value",
        ],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    )
    return sorted_profile.groupby("customer_id", group_keys=False).head(
        RECOMMENDATION_PROFILE_TOP_ATTRIBUTES
    )


def _coerce_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for column in ("customer_id", "article_id", "attr_type", "attr_value"):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("string")
    return dataframe
