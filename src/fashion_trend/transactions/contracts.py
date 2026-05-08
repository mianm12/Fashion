from __future__ import annotations

# 下游读取和消费 `transactions_train_weekly.parquet` 的周级交易列契约。
WEEKLY_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "customer_id",
    "price",
)
