from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic, write_parquet_atomic
from fashion_trend.recommendation.contracts import (
    ARTICLE_PRODUCT_MAP_COLUMNS,
    ARTICLE_PRODUCT_MAP_KEY_COLUMNS,
    CUSTOMER_AGE_BUCKETS,
    CUSTOMER_PROFILE_COLUMNS,
    CUSTOMER_PROFILE_KEY_COLUMNS,
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
from fashion_trend.recommendation.freshness import build_artifact_metadata
from fashion_trend.recommendation.paths import (
    ARTICLE_PRODUCT_MAP_PATH,
    CUSTOMER_PROFILE_PATH,
    EVALUATION_LABELS_PATH,
    RECOMMEND_METADATA_PATH,
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    USER_PROFILE_PATH,
)
from fashion_trend.recommendation.readers import reject_duplicate_key, validate_columns
from fashion_trend.recommendation.time_windows import build_recommendation_windows

CUSTOMER_PROFILE_SCHEMA_VERSION = 1
ARTICLE_PRODUCT_MAP_SCHEMA_VERSION = 1
CUSTOMER_AGE_BUCKET_ALGORITHM_VERSION = "customer-age-buckets-v1"


@dataclass(frozen=True)
class RecommendationInputArtifacts:
    time_windows: pd.DataFrame
    target_users: pd.DataFrame
    evaluation_labels: pd.DataFrame
    user_profile: pd.DataFrame
    customer_profile: pd.DataFrame | None = None
    article_product_map: pd.DataFrame | None = None


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


def build_customer_profile(customers: pd.DataFrame) -> pd.DataFrame:
    """Build a strict customer profile artifact without using it as a feature."""
    _validate_required_columns(
        customers,
        ("customer_id", "age", "club_member_status", "fashion_news_frequency"),
        "customers",
    )
    profile = customers.loc[
        :,
        ["customer_id", "age", "club_member_status", "fashion_news_frequency"],
    ].copy()
    profile["customer_id"] = profile["customer_id"].astype("string")
    profile["age"] = pd.to_numeric(profile["age"], errors="coerce").astype("Float64")
    profile["age_bucket"] = profile["age"].map(_bucket_customer_age).astype("string")
    profile["club_member_status"] = _fill_unknown_text(profile["club_member_status"])
    profile["fashion_news_frequency"] = _fill_unknown_text(
        profile["fashion_news_frequency"]
    )
    profile = profile.loc[:, list(CUSTOMER_PROFILE_COLUMNS)].reset_index(drop=True)
    _reject_missing_text(
        profile,
        "customer_id",
        "customer_profile",
    )
    reject_duplicate_key(profile, CUSTOMER_PROFILE_KEY_COLUMNS, "customer_profile")
    return profile


def build_article_product_map(clean_articles: pd.DataFrame) -> pd.DataFrame:
    """Build article-to-product-code mapping for product-variant retrieval."""
    _validate_required_columns(
        clean_articles,
        ARTICLE_PRODUCT_MAP_COLUMNS,
        "clean_articles",
    )
    product_map = clean_articles.loc[:, list(ARTICLE_PRODUCT_MAP_COLUMNS)].copy()
    product_map["article_id"] = product_map["article_id"].astype("string")
    product_map["product_code"] = product_map["product_code"].astype("string")
    _reject_missing_text(product_map, "article_id", "article_product_map")
    _reject_missing_text(product_map, "product_code", "article_product_map")
    reject_duplicate_key(
        product_map,
        ARTICLE_PRODUCT_MAP_KEY_COLUMNS,
        "article_product_map",
    )
    return product_map.reset_index(drop=True)


def build_and_write_recommendation_inputs(
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    trend_predictions: pd.DataFrame,
    input_paths: dict[str, str] | None = None,
    customers: pd.DataFrame | None = None,
    clean_articles: pd.DataFrame | None = None,
) -> RecommendationInputArtifacts:
    """Build and write recommendation input artifacts."""
    windows = build_recommendation_windows(trend_predictions)
    target_users = build_target_users(transactions, windows)
    labels = build_evaluation_labels(transactions, windows, target_users)
    profile = build_user_profile(
        transactions, article_attributes, windows, target_users
    )
    customer_profile = (
        build_customer_profile(customers) if customers is not None else None
    )
    article_product_map = (
        build_article_product_map(clean_articles)
        if clean_articles is not None
        else None
    )

    write_parquet_atomic(windows, TIME_WINDOWS_PATH)
    write_parquet_atomic(target_users, TARGET_USERS_PATH)
    write_parquet_atomic(labels, EVALUATION_LABELS_PATH)
    write_parquet_atomic(profile, USER_PROFILE_PATH)
    if customer_profile is not None:
        write_parquet_atomic(customer_profile, CUSTOMER_PROFILE_PATH)
    if article_product_map is not None:
        write_parquet_atomic(article_product_map, ARTICLE_PRODUCT_MAP_PATH)
    write_json_atomic(
        build_artifact_metadata(
            name="recommendation_inputs",
            input_artifacts=dict(input_paths or {}),
            output_artifacts=_recommendation_input_output_artifacts(
                include_customer_profile=customer_profile is not None,
                include_article_product_map=article_product_map is not None,
            ),
            schema_version=1,
            algorithm_version="recommendation-inputs-v1",
            config=_recommendation_input_config(
                include_customer_profile=customer_profile is not None,
                include_article_product_map=article_product_map is not None,
            ),
            row_counts=_recommendation_input_row_counts(
                windows=windows,
                target_users=target_users,
                labels=labels,
                profile=profile,
                customer_profile=customer_profile,
                article_product_map=article_product_map,
            ),
        ),
        RECOMMEND_METADATA_PATH,
    )
    return RecommendationInputArtifacts(
        windows,
        target_users,
        labels,
        profile,
        customer_profile,
        article_product_map,
    )


def _recommendation_input_row_counts(
    *,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
    profile: pd.DataFrame,
    customer_profile: pd.DataFrame | None,
    article_product_map: pd.DataFrame | None,
) -> dict[str, int]:
    row_counts = {
        "time_windows": int(len(windows)),
        "target_users": int(len(target_users)),
        "evaluation_labels": int(len(labels)),
        "user_profile": int(len(profile)),
    }
    if customer_profile is not None:
        row_counts["customer_profile"] = int(len(customer_profile))
    if article_product_map is not None:
        row_counts["article_product_map"] = int(len(article_product_map))
    return row_counts


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


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"{artifact_name} 缺少必要字段: {missing_columns}")


def _bucket_customer_age(age: object) -> str:
    if pd.isna(age):
        return "unknown"
    age_value = float(age)
    if 0 <= age_value < 20:
        return "0-19"
    if 20 <= age_value < 30:
        return "20-29"
    if 30 <= age_value < 40:
        return "30-39"
    if 40 <= age_value < 50:
        return "40-49"
    if 50 <= age_value < 60:
        return "50-59"
    if age_value >= 60:
        return "60+"
    return "unknown"


def _fill_unknown_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text == "", "unknown").fillna("unknown")


def _reject_missing_text(
    dataframe: pd.DataFrame,
    column: str,
    artifact_name: str,
) -> None:
    text = dataframe[column].astype("string").str.strip()
    missing = text.isna() | (text == "")
    if missing.any():
        sample = dataframe.loc[missing, [column]].head(3).to_dict("records")
        raise ValueError(f"{artifact_name} {column} 缺失: {sample}")


def _recommendation_input_config(
    *,
    include_customer_profile: bool,
    include_article_product_map: bool,
) -> dict[str, object]:
    config: dict[str, object] = {
        "profile_top_attributes": RECOMMENDATION_PROFILE_TOP_ATTRIBUTES,
    }
    if include_customer_profile:
        config.update(
            {
                "customer_profile_schema_version": CUSTOMER_PROFILE_SCHEMA_VERSION,
                "customer_age_bucket_algorithm_version": (
                    CUSTOMER_AGE_BUCKET_ALGORITHM_VERSION
                ),
                "customer_age_buckets": list(CUSTOMER_AGE_BUCKETS),
            }
        )
    if include_article_product_map:
        config["article_product_map_schema_version"] = (
            ARTICLE_PRODUCT_MAP_SCHEMA_VERSION
        )
    return config


def _recommendation_input_output_artifacts(
    *,
    include_customer_profile: bool = False,
    include_article_product_map: bool = False,
) -> dict[str, str]:
    output_artifacts = {
        "time_windows": str(TIME_WINDOWS_PATH),
        "target_users": str(TARGET_USERS_PATH),
        "evaluation_labels": str(EVALUATION_LABELS_PATH),
        "user_profile": str(USER_PROFILE_PATH),
    }
    if include_customer_profile:
        output_artifacts["customer_profile"] = str(CUSTOMER_PROFILE_PATH)
    if include_article_product_map:
        output_artifacts["article_product_map"] = str(ARTICLE_PRODUCT_MAP_PATH)
    return output_artifacts
