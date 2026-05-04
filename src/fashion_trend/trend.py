from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
        parquet_file = pq.ParquetFile(weekly_transactions_path)
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise ValueError(f"无法读取周级交易表: {weekly_transactions_path}") from exc

    validate_required_columns(
        parquet_file.schema_arrow.names,
        WEEKLY_TRANSACTION_COLUMNS,
        source_name="周级交易表",
    )

    try:
        return pd.read_parquet(
            weekly_transactions_path,
            columns=list(WEEKLY_TRANSACTION_COLUMNS),
        )
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise ValueError(f"无法读取周级交易表: {weekly_transactions_path}") from exc


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


def validate_article_attribute_edges_for_heat(
    article_attribute_edges: pd.DataFrame,
) -> None:
    validate_required_columns(
        article_attribute_edges.columns.tolist(),
        ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS,
        source_name="商品-属性边表",
    )
    validate_no_missing_values(
        article_attribute_edges,
        ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS,
        source_name="商品-属性边表",
    )
    validate_unique_key(
        article_attribute_edges,
        ["article_id", "attr_id"],
        source_name="商品-属性边表",
    )


def validate_all_sales_articles_have_attribute_edges(
    article_week_sales: pd.DataFrame,
    article_attribute_edges: pd.DataFrame,
) -> None:
    sales_article_ids = set(article_week_sales["article_id"].astype("string"))
    edge_article_ids = set(article_attribute_edges["article_id"].astype("string"))
    missing_article_ids = sorted(sales_article_ids - edge_article_ids)
    if missing_article_ids:
        examples = ", ".join(missing_article_ids[:5])
        raise ValueError(
            f"商品周销量表存在 {len(missing_article_ids)} 个 article_id "
            f"无法映射到属性边，例如: {examples}"
        )


def build_attribute_week_heat_frame(
    article_week_sales: pd.DataFrame,
    article_attribute_edges: pd.DataFrame,
) -> pd.DataFrame:
    validate_article_week_sales(article_week_sales)
    validate_article_attribute_edges_for_heat(article_attribute_edges)
    validate_all_sales_articles_have_attribute_edges(
        article_week_sales,
        article_attribute_edges,
    )

    normalized_sales = article_week_sales.loc[
        :, ["week_id", "article_id", "sales_cnt"]
    ].copy()
    normalized_sales["article_id"] = normalized_sales["article_id"].astype("string")

    normalized_edges = article_attribute_edges.loc[
        :, list(ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS)
    ].copy()
    normalized_edges["article_id"] = normalized_edges["article_id"].astype("string")
    normalized_edges["attr_id"] = normalized_edges["attr_id"].astype("string")
    normalized_edges["attr_type"] = normalized_edges["attr_type"].astype("string")
    normalized_edges["attr_value"] = normalized_edges["attr_value"].astype("string")

    joined = normalized_sales.merge(
        normalized_edges,
        on="article_id",
        how="inner",
    )
    heat = (
        joined.groupby(["week_id", "attr_id", "attr_type", "attr_value"], as_index=False)[
            "sales_cnt"
        ]
        .sum()
        .rename(columns={"sales_cnt": "heat_cnt"})
    )
    heat["heat_cnt"] = heat["heat_cnt"].astype("int64")
    heat["type_total_heat"] = heat.groupby(["week_id", "attr_type"])[
        "heat_cnt"
    ].transform("sum")
    heat["type_total_heat"] = heat["type_total_heat"].astype("int64")
    heat["heat_share"] = heat["heat_cnt"] / heat["type_total_heat"]
    heat["log_heat"] = np.log1p(heat["heat_cnt"])

    heat = heat.sort_values(
        ["week_id", "attr_type", "heat_cnt", "attr_id"],
        ascending=[True, True, False, True],
        ignore_index=True,
    )
    heat["rank_in_type"] = (
        heat.groupby(["week_id", "attr_type"]).cumcount().add(1).astype("int64")
    )

    return heat.loc[:, list(ATTRIBUTE_WEEK_HEAT_COLUMNS)].sort_values(
        ["week_id", "attr_type", "rank_in_type", "attr_id"],
        ignore_index=True,
    )


def validate_attribute_week_heat(attribute_week_heat: pd.DataFrame) -> None:
    validate_required_columns(
        attribute_week_heat.columns.tolist(),
        ATTRIBUTE_WEEK_HEAT_COLUMNS,
        source_name="属性周热度表",
    )
    validate_no_missing_values(
        attribute_week_heat,
        ATTRIBUTE_WEEK_HEAT_COLUMNS,
        source_name="属性周热度表",
    )
    validate_unique_key(
        attribute_week_heat,
        ["week_id", "attr_id"],
        source_name="属性周热度表",
    )
    validate_positive_values(
        attribute_week_heat,
        ["heat_cnt", "type_total_heat", "heat_share", "rank_in_type"],
        source_name="属性周热度表",
    )

    if (attribute_week_heat["type_total_heat"] < attribute_week_heat["heat_cnt"]).any():
        raise ValueError("属性周热度表存在 type_total_heat 小于 heat_cnt 的记录。")
    if (attribute_week_heat["heat_share"] > 1).any():
        raise ValueError("属性周热度表存在 heat_share 大于 1 的记录。")

    share_totals = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "heat_share"
    ].sum()
    invalid_share_totals = share_totals[~np.isclose(share_totals, 1.0, atol=1e-9)]
    if not invalid_share_totals.empty:
        raise ValueError("属性周热度表存在 week_id + attr_type 占比和不等于 1 的分组。")

    rank_counts = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "rank_in_type"
    ].nunique()
    row_counts = attribute_week_heat.groupby(["week_id", "attr_type"]).size()
    if not rank_counts.equals(row_counts):
        raise ValueError("属性周热度表存在重复 rank_in_type。")
    min_ranks = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "rank_in_type"
    ].min()
    if (min_ranks != 1).any():
        raise ValueError("属性周热度表存在 rank_in_type 未从 1 开始的分组。")


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
