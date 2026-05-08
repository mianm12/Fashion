from __future__ import annotations

from fashion_trend.foundation import logging as log
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.heat.article_sales import (
    build_article_week_sales_frame,
    validate_article_week_sales,
)
from fashion_trend.trend.paths import TREND_ARTICLE_WEEK_SALES_PATH

LOG_SOURCE = "article-week-sales"


def compute_article_week_sales() -> dict[str, int]:
    """编排商品周销量构建流程。

    流程:
        1. 读取 transactions_train_weekly.parquet。
        2. 构建并校验商品-周粒度销量表。
        3. 写出 article_week_sales.csv 并返回输出摘要。

    Returns:
        dict[str, int]: 输出行数、覆盖周数和覆盖商品数。
    """
    log.info(f"输入周级交易表: {WEEKLY_TRANSACTIONS_PATH}", source=LOG_SOURCE)
    log.info(
        "业务阶段: transactions_train_weekly.parquet -> article_week_sales.csv",
        source=LOG_SOURCE,
    )
    weekly_transactions = read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH)
    article_week_sales = build_article_week_sales_frame(weekly_transactions)
    validate_article_week_sales(article_week_sales)
    write_csv_atomic(article_week_sales, TREND_ARTICLE_WEEK_SALES_PATH)

    return {
        "rows": len(article_week_sales),
        "weeks": int(article_week_sales["week_id"].nunique()),
        "articles": int(article_week_sales["article_id"].nunique()),
    }


def main() -> int:
    """商品周销量阶段入口，稳定写出 article_week_sales.csv。"""
    try:
        stats = compute_article_week_sales()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"商品周销量行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖商品数: {stats['articles']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {TREND_ARTICLE_WEEK_SALES_PATH}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
