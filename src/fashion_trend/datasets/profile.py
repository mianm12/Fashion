from __future__ import annotations

from pathlib import Path

RAW_FILE_NAMES = (
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
)


def validate_raw_dataset_files(raw_dataset_dir: Path) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for file_name in RAW_FILE_NAMES:
        csv_path = raw_dataset_dir / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"原始数据文件不存在: {csv_path}")
        with csv_path.open("rb") as handle:
            line_count = sum(1 for _ in handle)
        row_counts[file_name] = max(line_count - 1, 0)
    return row_counts
