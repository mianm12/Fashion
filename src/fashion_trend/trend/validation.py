from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_required_columns(
    actual_columns: Sequence[str],
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(actual_columns))
    if missing_columns:
        raise ValueError(f"{source_name} 缺少必要字段: " + ", ".join(missing_columns))


def validate_no_missing_values(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = [
        column for column in required_columns if int(dataframe[column].isna().sum()) > 0
    ]
    if missing_columns:
        raise ValueError(f"{source_name} 存在缺失值字段: " + ", ".join(missing_columns))


def validate_unique_key(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str],
    source_name: str,
) -> None:
    duplicate_mask = dataframe.duplicated(subset=list(key_columns), keep=False)
    if duplicate_mask.any():
        raise ValueError(f"{source_name} 存在重复字段值: " + ", ".join(key_columns))


def validate_non_negative_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    invalid_columns = [column for column in columns if (dataframe[column] < 0).any()]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在负值字段: " + ", ".join(invalid_columns))


def validate_positive_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    invalid_columns = [column for column in columns if (dataframe[column] <= 0).any()]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在非正值字段: " + ", ".join(invalid_columns))
