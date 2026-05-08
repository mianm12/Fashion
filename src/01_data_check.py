from __future__ import annotations

from fashion_trend.datasets.paths import RAW_HM_DIR
from fashion_trend.datasets.profile import validate_raw_dataset_files
from fashion_trend.foundation import logging as log

LOG_SOURCE = "data-check"


def main() -> int:
    """原始数据检查阶段入口，稳定输出原始 CSV 文件行数日志摘要。"""
    try:
        log.info(f"检查原始数据目录: {RAW_HM_DIR}", source=LOG_SOURCE)
        log.info(
            "业务阶段: 原始 CSV 文件存在性/可读性检查 -> 行数摘要",
            source=LOG_SOURCE,
        )
        row_counts = validate_raw_dataset_files(RAW_HM_DIR)
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    for file_name, row_count in row_counts.items():
        log.info(f"{file_name}: {row_count:,} 行", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
