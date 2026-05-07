from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"{source_name} 缺少必要字段: {', '.join(missing_columns)}")


def validate_no_missing_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    missing_counts = dataframe[list(columns)].isna().sum()
    invalid_columns = [
        f"{column}={int(count)}"
        for column, count in missing_counts.items()
        if int(count) > 0
    ]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在缺失值: {', '.join(invalid_columns)}")


def validate_unique_key(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str],
    source_name: str,
) -> None:
    duplicate_count = int(dataframe.duplicated(list(key_columns)).sum())
    if duplicate_count > 0:
        key_names = ", ".join(key_columns)
        raise ValueError(f"{source_name} 存在重复键 {key_names}: {duplicate_count} 行")


def validate_non_negative_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    invalid_columns = [
        column
        for column in columns
        if bool((dataframe[column] < 0).any())
    ]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在负数: {', '.join(invalid_columns)}")


def validate_positive_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    invalid_columns = [
        column
        for column in columns
        if bool((dataframe[column] <= 0).any())
    ]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在非正数: {', '.join(invalid_columns)}")
