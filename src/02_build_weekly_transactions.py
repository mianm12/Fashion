from __future__ import annotations

from fashion_trend.datasets.paths import RAW_TRANSACTIONS_PATH
from fashion_trend.foundation import logging as log
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.weekly import build_weekly_transactions

LOG_SOURCE = "weekly-transactions"


def main() -> int:
    """周级交易构建阶段入口，稳定写出 transactions_train_weekly.parquet。

    Args:
        None: 本函数直接读取项目路径配置，不接收外部参数。

    Returns:
        int: 进程退出码；0 表示成功，1 表示遇到可定位的处理错误。
    """
    try:
        log.info(f"输入文件: {RAW_TRANSACTIONS_PATH}", source=LOG_SOURCE)
        log.info(
            "业务阶段: 交易周聚合 -> transactions_train_weekly.parquet",
            source=LOG_SOURCE,
        )
        build_weekly_transactions(
            raw_transactions_path=RAW_TRANSACTIONS_PATH,
            weekly_transactions_path=WEEKLY_TRANSACTIONS_PATH,
        )
        log.info(f"输出文件: {WEEKLY_TRANSACTIONS_PATH}", source=LOG_SOURCE)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
