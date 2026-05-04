from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import pandas as pd

WEEKLY_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "customer_id",
    "price",
)

ARTICLE_WEEK_SALES_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "sales_cnt",
    "sales_user_cnt",
    "sales_amount",
)

ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS: tuple[str, ...] = (
    "article_id",
    "attr_id",
    "attr_type",
    "attr_value",
)

ATTRIBUTE_WEEK_HEAT_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "heat_cnt",
    "type_total_heat",
    "heat_share",
    "log_heat",
    "rank_in_type",
)


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


def read_weekly_transactions(weekly_transactions_path: Path) -> pd.DataFrame:
    if not weekly_transactions_path.exists():
        raise FileNotFoundError(f"周级交易表不存在: {weekly_transactions_path}")

    try:
        return pd.read_parquet(
            weekly_transactions_path,
            columns=list(WEEKLY_TRANSACTION_COLUMNS),
        )
    except ValueError as exc:
        raise ValueError(
            f"周级交易表缺少必要字段: {weekly_transactions_path}"
        ) from exc


def build_article_week_sales_frame(weekly_transactions: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        weekly_transactions.columns.tolist(),
        WEEKLY_TRANSACTION_COLUMNS,
        source_name="周级交易表",
    )
    validate_no_missing_values(
        weekly_transactions,
        WEEKLY_TRANSACTION_COLUMNS,
        source_name="周级交易表",
    )
    validate_non_negative_values(
        weekly_transactions,
        ["price"],
        source_name="周级交易表",
    )

    normalized_transactions = weekly_transactions.loc[
        :, list(WEEKLY_TRANSACTION_COLUMNS)
    ].copy()
    normalized_transactions["article_id"] = normalized_transactions["article_id"].astype(
        "string"
    )

    sales = (
        normalized_transactions.groupby(["week_id", "article_id"], as_index=False)
        .agg(
            sales_cnt=("article_id", "size"),
            sales_user_cnt=("customer_id", "nunique"),
            sales_amount=("price", "sum"),
        )
        .sort_values(["week_id", "article_id"], ignore_index=True)
    )
    sales["sales_cnt"] = sales["sales_cnt"].astype("int64")
    sales["sales_user_cnt"] = sales["sales_user_cnt"].astype("int64")

    return sales.loc[:, list(ARTICLE_WEEK_SALES_COLUMNS)]


def validate_article_week_sales(article_week_sales: pd.DataFrame) -> None:
    validate_required_columns(
        article_week_sales.columns.tolist(),
        ARTICLE_WEEK_SALES_COLUMNS,
        source_name="商品周销量表",
    )
    validate_no_missing_values(
        article_week_sales,
        ARTICLE_WEEK_SALES_COLUMNS,
        source_name="商品周销量表",
    )
    validate_unique_key(
        article_week_sales,
        ["week_id", "article_id"],
        source_name="商品周销量表",
    )
    validate_positive_values(
        article_week_sales,
        ["sales_cnt", "sales_user_cnt"],
        source_name="商品周销量表",
    )
    validate_non_negative_values(
        article_week_sales,
        ["sales_amount"],
        source_name="商品周销量表",
    )


def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_trend_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_csv(tmp_output_path, index=False, quoting=csv.QUOTE_ALL)
        tmp_output_path.replace(output_path)
    except Exception:
        remove_file_if_exists(tmp_output_path)
        raise
