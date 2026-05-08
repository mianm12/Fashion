from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fashion_trend.datasets.download import download_competition
from fashion_trend.datasets.paths import DEFAULT_COMPETITION
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.paths import RAW_DIR

LOG_SOURCE = "download-data"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，生成下载数据集所需的运行配置。

    Args:
        argv (Sequence[str] | None, optional): 命令行参数序列；为 None 时从
            sys.argv 读取实际命令行输入。Defaults to None.

    Returns:
        argparse.Namespace: 解析后的参数对象，包含 competition、data_dir、unzip
            和 force 等下载配置。
    """
    parser = argparse.ArgumentParser(
        description="Download a Kaggle competition dataset into the project data directory."
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug. Default: {DEFAULT_COMPETITION}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Base data directory. Default: {RAW_DIR}",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_false",
        dest="unzip",
        help="Keep downloaded zip files without extracting them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the Kaggle download even when destination files already exist.",
    )
    parser.set_defaults(unzip=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """数据下载阶段入口，负责解析参数、执行下载并返回进程退出码。

    稳定输出位置为 raw H&M 数据目录。

    Args:
        argv (Sequence[str] | None, optional): 命令行参数序列；为 None 时使用
            sys.argv 中的实际输入。Defaults to None.

    Returns:
        int: 进程退出码；0 表示执行成功，1 表示下载或解压阶段出现可处理错误。
    """
    args = parse_args(argv)
    try:
        log.info(
            f"输入数据源: Kaggle competition={args.competition}",
            source=LOG_SOURCE,
        )
        log.info(
            f"关键参数: unzip={args.unzip}, force={args.force}",
            source=LOG_SOURCE,
        )
        log.info(f"输出目录: {args.data_dir}", source=LOG_SOURCE)
        log.info("业务阶段: Kaggle 下载/解压 -> raw H&M 数据目录", source=LOG_SOURCE)
        destination = download_competition(
            competition=args.competition,
            data_dir=args.data_dir,
            unzip=args.unzip,
            force=args.force,
        )
    except RuntimeError as exc:
        log.error(str(exc), source=LOG_SOURCE)
        return 1

    log.info(f"原始数据目录已就绪: {destination}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
