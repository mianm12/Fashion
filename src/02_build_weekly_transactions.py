from __future__ import annotations

from fashion_trend.foundation import logging as log
from fashion_trend.foundation.paths import PATH
from fashion_trend.transactions.weekly import build_weekly_transactions

LOG_SOURCE = "weekly-transactions"


def main() -> int:
    """脚本入口，执行周级交易表构建并返回进程退出码。

    Args:
        None: 本函数直接读取项目路径配置，不接收外部参数。

    Returns:
        int: 进程退出码；0 表示成功，1 表示遇到可定位的处理错误。
    """
    try:
        log.info(f"输入文件: {PATH['raw_transactions']}", source=LOG_SOURCE)
        build_weekly_transactions(
            raw_transactions_path=PATH["raw_transactions"],
            weekly_transactions_path=PATH["interim_transactions_weekly"],
        )
        log.info(f"输出文件: {PATH['interim_transactions_weekly']}", source=LOG_SOURCE)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
