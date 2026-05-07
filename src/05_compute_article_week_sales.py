from __future__ import annotations

from fashion_trend.foundation import logging as log
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.foundation.paths import PATH
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_weekly_transactions,
    validate_article_week_sales,
)

LOG_SOURCE = "article-week-sales"


def compute_article_week_sales() -> dict[str, int]:
    log.info(
        f"输入周级交易表: {PATH['interim_transactions_weekly']}", source=LOG_SOURCE
    )
    weekly_transactions = read_weekly_transactions(PATH["interim_transactions_weekly"])
    article_week_sales = build_article_week_sales_frame(weekly_transactions)
    validate_article_week_sales(article_week_sales)
    write_csv_atomic(article_week_sales, PATH["trend_article_week_sales"])

    return {
        "rows": len(article_week_sales),
        "weeks": int(article_week_sales["week_id"].nunique()),
        "articles": int(article_week_sales["article_id"].nunique()),
    }


def main() -> int:
    try:
        stats = compute_article_week_sales()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"商品周销量行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖商品数: {stats['articles']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {PATH['trend_article_week_sales']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
