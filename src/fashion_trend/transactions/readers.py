from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.transactions.contracts import WEEKLY_TRANSACTION_COLUMNS


def read_weekly_transactions(weekly_transactions_path: Path) -> pd.DataFrame:
    if not weekly_transactions_path.exists():
        raise FileNotFoundError(f"周级交易表不存在: {weekly_transactions_path}")

    try:
        parquet_file = pq.ParquetFile(weekly_transactions_path)
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise ValueError(f"无法读取周级交易表: {weekly_transactions_path}") from exc

    validate_required_columns(
        pd.DataFrame(columns=parquet_file.schema_arrow.names),
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
