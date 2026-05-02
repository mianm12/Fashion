from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fashion_trend import log
from fashion_trend.config import PATH

REQUIRED_COLUMNS: tuple[str, ...] = (
    "t_dat",
    "customer_id",
    "article_id",
    "price",
    "sales_channel_id",
)
OUTPUT_COLUMNS = (
    "t_dat",
    "week_id",
    "customer_id",
    "article_id",
    "price",
    "sales_channel_id",
)
TRANSACTION_DTYPES = {
    "t_dat": "string",
    "customer_id": "string",
    "article_id": "string",
    "price": "float64",
    "sales_channel_id": "int8",
}
DEFAULT_CHUNKSIZE = 1_000_000
LOG_SOURCE = "weekly-transactions"


def read_csv_columns(csv_path: Path) -> list[str]:
    """读取交易 CSV 的表头列名。

    Args:
        csv_path (Path): 原始 `transactions_train.csv` 文件路径。

    Returns:
        list[str]: CSV 文件中实际存在的列名列表。

    Raises:
        FileNotFoundError: 当输入 CSV 文件不存在时抛出。
        ValueError: 当 CSV 文件为空或无法读取表头时抛出。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"原始交易文件不存在: {csv_path}")

    try:
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"原始交易文件为空: {csv_path}") from exc

    if not columns:
        raise ValueError(f"原始交易文件没有可用列: {csv_path}")

    return columns


def validate_required_columns(columns: list[str]) -> None:
    """校验交易表是否包含生成周级交易表所需字段。

    Args:
        columns (list[str]): 原始交易表的列名列表。

    Returns:
        None: 校验通过时不返回数据。

    Raises:
        ValueError: 当交易表缺少必要字段时抛出。
    """
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing_columns:
        raise ValueError("原始交易表缺少必要字段: " + ", ".join(missing_columns))


def parse_transaction_dates(date_values: pd.Series, context: str) -> pd.Series:
    """解析交易日期列并显式拦截无法解析的日期值。

    Args:
        date_values (pd.Series): 交易表中的 `t_dat` 日期列。
        context (str): 当前解析场景说明，用于生成可定位的错误信息。

    Returns:
        pd.Series: pandas datetime64 类型的日期序列。

    Raises:
        ValueError: 当存在无法解析的日期值时抛出。
    """
    parsed_dates = pd.to_datetime(date_values, errors="coerce")
    invalid_date_count = int(parsed_dates.isna().sum())
    if invalid_date_count > 0:
        raise ValueError(
            f"{context} 存在 {invalid_date_count} 条无法解析的 t_dat 日期。"
        )

    return parsed_dates


def read_transaction_chunks(
    csv_path: Path,
    usecols: list[str],
    dtype: dict[str, str],
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    """按指定列和类型分块读取交易 CSV。

    Args:
        csv_path (Path): 原始 `transactions_train.csv` 文件路径。
        usecols (list[str]): 每个分块需要读取的列名。
        dtype (dict[str, str]): 每个读取列对应的 pandas 数据类型。
        chunksize (int): 每次读取的行数。

    Returns:
        Iterator[pd.DataFrame]: 可迭代的 CSV 分块读取器。
    """
    pandas_dtype = cast(Mapping[Hashable, str], dtype)

    return pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=pandas_dtype,
        chunksize=chunksize,
    )


def scan_transaction_date_range(
    csv_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """扫描原始交易表，获取日期范围和交易总行数。

    Args:
        csv_path (Path): 原始 `transactions_train.csv` 文件路径。
        chunksize (int, optional): 每次读取的行数。Defaults to
            DEFAULT_CHUNKSIZE.

    Returns:
        tuple[pd.Timestamp, pd.Timestamp, int]: 最早交易日期、最晚交易日期和
            总交易行数。

    Raises:
        ValueError: 当交易表没有数据行或存在非法日期时抛出。
    """
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    total_rows = 0

    chunks = read_transaction_chunks(
        csv_path,
        usecols=["t_dat"],
        dtype={"t_dat": "string"},
        chunksize=chunksize,
    )
    for chunk_index, chunk in enumerate(chunks, start=1):
        parsed_dates = parse_transaction_dates(
            chunk["t_dat"],
            context=f"第 {chunk_index} 个日期扫描分块",
        )
        chunk_min_date = parsed_dates.min()
        chunk_max_date = parsed_dates.max()
        min_date = chunk_min_date if min_date is None else min(min_date, chunk_min_date)
        max_date = chunk_max_date if max_date is None else max(max_date, chunk_max_date)
        total_rows += len(chunk)

    if total_rows == 0 or min_date is None or max_date is None:
        raise ValueError(f"原始交易表没有数据行: {csv_path}")

    return min_date, max_date, total_rows


def add_week_id(
    transactions: pd.DataFrame,
    min_date: pd.Timestamp,
    chunk_index: int,
) -> pd.DataFrame:
    """为交易分块添加基于最早交易日期的周编号。

    Args:
        transactions (pd.DataFrame): 原始交易表的一个读取分块。
        min_date (pd.Timestamp): 全量交易表中的最早交易日期，作为第 0 周起点。
        chunk_index (int): 当前分块序号，用于错误信息定位。

    Returns:
        pd.DataFrame: 已添加 `week_id` 且列顺序固定的交易分块。

    Raises:
        ValueError: 当当前分块存在非法日期时抛出。
    """
    weekly_transactions = transactions.copy()
    parsed_dates = parse_transaction_dates(
        weekly_transactions["t_dat"],
        context=f"第 {chunk_index} 个交易处理分块",
    )
    week_ids = ((parsed_dates - min_date).dt.days // 7).astype("int16")

    weekly_transactions["t_dat"] = parsed_dates
    weekly_transactions["week_id"] = week_ids

    return weekly_transactions[list(OUTPUT_COLUMNS)]


def write_weekly_transactions(
    csv_path: Path,
    output_path: Path,
    min_date: pd.Timestamp,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> int:
    """按块读取原始交易表并写出周级交易 Parquet 文件。

    Args:
        csv_path (Path): 原始 `transactions_train.csv` 文件路径。
        output_path (Path): 周级交易表 Parquet 输出路径。
        min_date (pd.Timestamp): 全量交易表中的最早交易日期，作为第 0 周起点。
        chunksize (int, optional): 每次读取的行数。Defaults to
            DEFAULT_CHUNKSIZE.

    Returns:
        int: 实际写入 Parquet 文件的交易行数。

    Raises:
        RuntimeError: 当 Parquet 写入失败时抛出。
        ValueError: 当交易数据存在非法日期时抛出。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    writer = None
    written_rows = 0
    write_succeeded = False

    try:
        chunks = read_transaction_chunks(
            csv_path,
            usecols=list(REQUIRED_COLUMNS),
            dtype=TRANSACTION_DTYPES,
            chunksize=chunksize,
        )
        for chunk_index, chunk in enumerate(chunks, start=1):
            weekly_chunk = add_week_id(chunk, min_date, chunk_index)
            table = pa.Table.from_pandas(
                weekly_chunk,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_output_path,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)
            written_rows += len(weekly_chunk)
            log.info(f"已处理 {written_rows:,} 行交易。", source=LOG_SOURCE)

        write_succeeded = True
    finally:
        if writer is not None:
            writer.close()
        if not write_succeeded and tmp_output_path.exists():
            tmp_output_path.unlink()

    tmp_output_path.replace(output_path)
    return written_rows


def build_weekly_transactions(
    raw_transactions_path: Path,
    weekly_transactions_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> None:
    """构建 H&M 原始交易表对应的周级交易基础表。

    Args:
        raw_transactions_path (Path): 原始交易 CSV 文件路径。
        weekly_transactions_path (Path): 周级交易 Parquet 输出路径。
        chunksize (int, optional): 每次读取的行数。Defaults to
            DEFAULT_CHUNKSIZE.

    Returns:
        None: 本函数完成文件写入后不返回业务数据。

    Raises:
        FileNotFoundError: 当原始交易文件不存在时抛出。
        ValueError: 当原始交易表结构或日期数据不合法时抛出。
        RuntimeError: 当 Parquet 写入失败时抛出。
    """
    log.info(f"输入文件: {raw_transactions_path}", source=LOG_SOURCE)
    columns = read_csv_columns(raw_transactions_path)
    validate_required_columns(columns)

    min_date, max_date, total_rows = scan_transaction_date_range(
        raw_transactions_path,
        chunksize=chunksize,
    )
    max_week_id = int((max_date - min_date).days // 7)

    log.info(f"原始交易行数: {total_rows:,}", source=LOG_SOURCE)
    log.info(
        f"交易日期范围: {min_date.date()} 至 {max_date.date()}",
        source=LOG_SOURCE,
    )
    log.info(f"周编号范围: 0 至 {max_week_id}", source=LOG_SOURCE)

    written_rows = write_weekly_transactions(
        csv_path=raw_transactions_path,
        output_path=weekly_transactions_path,
        min_date=min_date,
        chunksize=chunksize,
    )
    if written_rows != total_rows:
        raise RuntimeError(
            f"写入行数与原始行数不一致: 写入 {written_rows:,} 行，"
            f"原始 {total_rows:,} 行。"
        )

    log.info(f"输出文件: {weekly_transactions_path}", source=LOG_SOURCE)
    log.info("周级交易表构建完成。", source=LOG_SOURCE)


def main() -> int:
    """脚本入口，执行周级交易表构建并返回进程退出码。

    Args:
        None: 本函数直接读取项目路径配置，不接收外部参数。

    Returns:
        int: 进程退出码；0 表示成功，1 表示遇到可定位的处理错误。
    """
    try:
        build_weekly_transactions(
            raw_transactions_path=PATH["raw_transactions"],
            weekly_transactions_path=PATH["interim_transactions_weekly"],
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
