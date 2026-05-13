from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.recommendation.contracts import (
    ARTICLE_PRODUCT_MAP_COLUMNS,
    ARTICLE_PRODUCT_MAP_KEY_COLUMNS,
    CANDIDATE_ITEM_COLUMNS,
    CANDIDATE_ITEM_KEY_COLUMNS,
    CUSTOMER_AGE_BUCKETS,
    CUSTOMER_PROFILE_COLUMNS,
    CUSTOMER_PROFILE_KEY_COLUMNS,
    ENHANCED_CANDIDATE_ITEM_COLUMNS,
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
    EVALUATION_LABEL_COLUMNS,
    EVALUATION_LABEL_KEY_COLUMNS,
    RECOMMENDATION_ITEMS_COLUMNS,
    RECOMMENDATION_ITEMS_KEY_COLUMNS,
    RECOMMENDATION_TEXT_COLUMNS,
    RECOMMENDATION_TOP_K,
    RECOMMENDATIONS_COLUMNS,
    RECOMMENDATIONS_KEY_COLUMNS,
    TARGET_USER_COLUMNS,
    TARGET_USER_KEY_COLUMNS,
    TIME_WINDOW_COLUMNS,
    TIME_WINDOW_KEY_COLUMNS,
    USER_PROFILE_COLUMNS,
    USER_PROFILE_KEY_COLUMNS,
)


def validate_columns(
    dataframe: pd.DataFrame,
    expected_columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    actual_columns = tuple(dataframe.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            f"{artifact_name} 列契约不匹配: "
            f"expected={expected_columns}, actual={actual_columns}"
        )


def reject_duplicate_key(
    dataframe: pd.DataFrame,
    key_columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    duplicated = dataframe.duplicated(list(key_columns), keep=False)
    if duplicated.any():
        sample = dataframe.loc[duplicated, list(key_columns)].head(3).to_dict("records")
        raise ValueError(f"{artifact_name} 存在重复键: {sample}")


def reject_missing_key(
    dataframe: pd.DataFrame,
    key_columns: tuple[str, ...],
    artifact_name: str,
) -> None:
    missing = dataframe.loc[:, list(key_columns)].isna().any(axis=1)
    if missing.any():
        sample = dataframe.loc[missing, list(key_columns)].head(3).to_dict("records")
        raise ValueError(f"{artifact_name} 存在缺失键: {sample}")


def text_dtypes_for_columns(columns: tuple[str, ...]) -> dict[str, str]:
    return {
        column: "string" for column in columns if column in RECOMMENDATION_TEXT_COLUMNS
    }


def read_csv_artifact(
    path: Path,
    expected_columns: tuple[str, ...],
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        dtype=text_dtypes_for_columns(expected_columns),
        keep_default_na=False,
    )
    validate_columns(dataframe, expected_columns, path.name)
    return coerce_article_id_string(dataframe)


def coerce_article_id_string(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for column in RECOMMENDATION_TEXT_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("string")
    return dataframe


def validate_path_value_matches(
    dataframe: pd.DataFrame,
    column: str,
    expected: str,
    artifact_name: str,
) -> None:
    actual_values = set(dataframe[column].dropna().astype(str))
    if actual_values != {expected}:
        raise ValueError(
            f"{artifact_name} {column} 与路径不匹配: "
            f"expected={expected}, actual={sorted(actual_values)}"
        )


def read_time_windows(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    validate_columns(dataframe, TIME_WINDOW_COLUMNS, "time_windows")
    reject_duplicate_key(dataframe, TIME_WINDOW_KEY_COLUMNS, "time_windows")
    return coerce_article_id_string(dataframe)


def read_target_users(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    validate_columns(dataframe, TARGET_USER_COLUMNS, "target_users")
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(dataframe, TARGET_USER_KEY_COLUMNS, "target_users")
    return dataframe


def read_evaluation_labels(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    validate_columns(dataframe, EVALUATION_LABEL_COLUMNS, "evaluation_labels")
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(dataframe, EVALUATION_LABEL_KEY_COLUMNS, "evaluation_labels")
    return dataframe


def read_user_profile(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    validate_columns(dataframe, USER_PROFILE_COLUMNS, "user_profile")
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(dataframe, USER_PROFILE_KEY_COLUMNS, "user_profile")
    return dataframe


def read_customer_profile(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    validate_columns(dataframe, CUSTOMER_PROFILE_COLUMNS, "customer_profile")
    dataframe = coerce_article_id_string(dataframe)
    reject_missing_key(dataframe, CUSTOMER_PROFILE_KEY_COLUMNS, "customer_profile")
    reject_duplicate_key(dataframe, CUSTOMER_PROFILE_KEY_COLUMNS, "customer_profile")

    invalid_bucket = ~dataframe["age_bucket"].isin(CUSTOMER_AGE_BUCKETS)
    if invalid_bucket.any():
        sample = dataframe.loc[invalid_bucket, ["customer_id", "age_bucket"]]
        sample_records = sample.head(3).to_dict("records")
        raise ValueError(f"customer_profile age_bucket 非法: {sample_records}")
    return dataframe


def read_article_product_map(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    validate_columns(dataframe, ARTICLE_PRODUCT_MAP_COLUMNS, "article_product_map")
    dataframe = coerce_article_id_string(dataframe)
    reject_missing_key(
        dataframe,
        ARTICLE_PRODUCT_MAP_KEY_COLUMNS,
        "article_product_map",
    )
    reject_duplicate_key(
        dataframe,
        ARTICLE_PRODUCT_MAP_KEY_COLUMNS,
        "article_product_map",
    )
    missing_product_code = dataframe["product_code"].isna() | (
        dataframe["product_code"].astype("string").str.strip() == ""
    )
    if missing_product_code.any():
        sample = dataframe.loc[missing_product_code, ["article_id", "product_code"]]
        raise ValueError(
            "article_product_map product_code 缺失: "
            f"{sample.head(3).to_dict('records')}"
        )
    return dataframe


def read_candidate_items(path: Path) -> pd.DataFrame:
    expected_strategy = path.parent.name
    validate_safe_path_segment(expected_strategy, "strategy")
    dataframe = pd.read_parquet(path)
    expected_columns = candidate_item_columns_for_strategy(expected_strategy)
    validate_columns(dataframe, expected_columns, "candidate_items")
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(dataframe, CANDIDATE_ITEM_KEY_COLUMNS, "candidate_items")
    validate_path_value_matches(
        dataframe,
        "strategy",
        expected_strategy,
        "candidate_items",
    )
    return dataframe


def candidate_item_columns_for_strategy(strategy: str) -> tuple[str, ...]:
    if strategy == "enhanced_default":
        return ENHANCED_CANDIDATE_ITEM_COLUMNS
    return CANDIDATE_ITEM_COLUMNS


def read_recommendations(path: Path) -> pd.DataFrame:
    expected_method = path.parent.name
    validate_safe_path_segment(expected_method, "method")
    dataframe = read_csv_artifact(path, RECOMMENDATIONS_COLUMNS)
    reject_duplicate_key(dataframe, RECOMMENDATIONS_KEY_COLUMNS, "recommendations")
    validate_path_value_matches(
        dataframe,
        "method",
        expected_method,
        "recommendations",
    )
    return dataframe


def read_recommendation_items(path: Path) -> pd.DataFrame:
    expected_method = path.parent.name
    validate_safe_path_segment(expected_method, "method")
    dataframe = pd.read_parquet(path)
    dataframe = normalize_recommendation_items_columns(dataframe, expected_method)
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(
        dataframe,
        RECOMMENDATION_ITEMS_KEY_COLUMNS,
        "recommendation_items",
    )
    invalid_rank = ~dataframe["rank"].between(1, RECOMMENDATION_TOP_K)
    if invalid_rank.any():
        sample = dataframe.loc[invalid_rank, ["customer_id", "article_id", "rank"]]
        raise ValueError(
            "recommendation_items rank 超出 Top-K 范围: "
            f"{sample.head(3).to_dict('records')}"
        )
    if dataframe.empty:
        return dataframe
    validate_path_value_matches(
        dataframe,
        "method",
        expected_method,
        "recommendation_items",
    )
    return dataframe


def normalize_recommendation_items_columns(
    dataframe: pd.DataFrame,
    expected_method: str | None = None,
) -> pd.DataFrame:
    """Return recommendation items in the current schema.

    Older stable method artifacts predate enhanced score columns. They remain valid
    legacy inputs and are normalized by filling the enhanced-only score columns with
    zero instead of forcing callers to overwrite stable outputs.
    """
    result = dataframe.copy()
    fillable_columns = set(ENHANCED_RECOMMENDATION_SCORE_COLUMNS[4:])
    missing_columns = [
        column
        for column in RECOMMENDATION_ITEMS_COLUMNS
        if column not in result.columns
    ]
    missing_enhanced_score_columns = [
        column for column in missing_columns if column in fillable_columns
    ]
    if (
        expected_method == "enhanced_pop_similarity_trend"
        and missing_enhanced_score_columns
    ):
        raise ValueError(
            "recommendation_items enhanced score columns missing: "
            f"{missing_enhanced_score_columns}"
        )
    unfillable_columns = [
        column for column in missing_columns if column not in fillable_columns
    ]
    if unfillable_columns:
        raise ValueError(
            "recommendation_items 列契约不匹配: "
            f"expected={RECOMMENDATION_ITEMS_COLUMNS}, actual={tuple(dataframe.columns)}"
        )
    for column in missing_columns:
        result[column] = 0.0
    return result.loc[:, list(RECOMMENDATION_ITEMS_COLUMNS)]


def read_recommendation_result(result_path: Path) -> pd.DataFrame:
    """读取符合 Top-N 推荐结果契约的 CSV 表。"""
    if not result_path.exists():
        raise FileNotFoundError(f"推荐结果文件不存在: {result_path}")
    return read_recommendations(result_path)
