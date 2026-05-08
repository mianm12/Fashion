from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion_trend.foundation.dataframe import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
from fashion_trend.transactions.contracts import WEEKLY_TRANSACTION_COLUMNS
from fashion_trend.trend.schema import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ARTICLE_WEEK_SALES_DTYPES,
)


def build_article_week_sales_frame(weekly_transactions: pd.DataFrame) -> pd.DataFrame:
    """聚合周级交易表，生成商品周销量表。

    Args:
        weekly_transactions: 已按交易周展开的明细表，必须满足周级交易列契约。

    Returns:
        按 `week_id + article_id` 聚合后的销量、购买用户数和销售额表。

    Raises:
        ValueError: 当输入缺少必要列、存在缺失值或价格为负时抛出。
    """
    validate_required_columns(
        weekly_transactions,
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
    normalized_transactions["article_id"] = normalized_transactions[
        "article_id"
    ].astype("string")

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
    """校验商品周销量表的列、主键、正销量/用户数和非负销售额。"""
    validate_required_columns(
        article_week_sales,
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


def read_article_week_sales(article_week_sales_path: Path) -> pd.DataFrame:
    """读取 `article_week_sales.csv` 商品周销量表并保留契约列类型。"""
    if not article_week_sales_path.exists():
        raise FileNotFoundError(f"商品周销量表不存在: {article_week_sales_path}")

    try:
        header = pd.read_csv(article_week_sales_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取商品周销量表: {article_week_sales_path}") from exc

    missing_columns = sorted(set(ARTICLE_WEEK_SALES_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "商品周销量表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {article_week_sales_path}"
        )

    try:
        return pd.read_csv(
            article_week_sales_path,
            usecols=list(ARTICLE_WEEK_SALES_COLUMNS),
            dtype=ARTICLE_WEEK_SALES_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取商品周销量表: {article_week_sales_path}") from exc
