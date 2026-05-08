from __future__ import annotations

# `transactions_train_weekly.parquet` 的稳定列契约。
WEEKLY_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "customer_id",
    "price",
)
