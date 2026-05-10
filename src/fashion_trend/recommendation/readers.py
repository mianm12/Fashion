from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.recommendation.contracts import (
    CANDIDATE_ITEM_COLUMNS,
    CANDIDATE_ITEM_KEY_COLUMNS,
    EVALUATION_LABEL_COLUMNS,
    EVALUATION_LABEL_KEY_COLUMNS,
    RECOMMENDATION_ITEMS_COLUMNS,
    RECOMMENDATION_ITEMS_KEY_COLUMNS,
    RECOMMENDATION_TEXT_COLUMNS,
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


def read_candidate_items(path: Path) -> pd.DataFrame:
    expected_strategy = path.parent.name
    validate_safe_path_segment(expected_strategy, "strategy")
    dataframe = pd.read_parquet(path)
    validate_columns(dataframe, CANDIDATE_ITEM_COLUMNS, "candidate_items")
    dataframe = coerce_article_id_string(dataframe)
    reject_duplicate_key(dataframe, CANDIDATE_ITEM_KEY_COLUMNS, "candidate_items")
    validate_path_value_matches(
        dataframe,
        "strategy",
        expected_strategy,
        "candidate_items",
    )
    return dataframe


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
    dataframe = read_csv_artifact(path, RECOMMENDATION_ITEMS_COLUMNS)
    reject_duplicate_key(
        dataframe,
        RECOMMENDATION_ITEMS_KEY_COLUMNS,
        "recommendation_items",
    )
    validate_path_value_matches(
        dataframe,
        "method",
        expected_method,
        "recommendation_items",
    )
    return dataframe


def read_recommendation_result(result_path: Path) -> pd.DataFrame:
    """读取符合 Top-N 推荐结果契约的 CSV 表。"""
    if not result_path.exists():
        raise FileNotFoundError(f"推荐结果文件不存在: {result_path}")
    return read_recommendations(result_path)
