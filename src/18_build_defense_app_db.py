from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fashion_trend.foundation import logging as log
from fashion_trend.presentation.paths import DEFENSE_APP_DB_PATH
from fashion_trend.presentation.runner import run_defense_app_db_build

LOG_SOURCE = "defense-app-db"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=DEFENSE_APP_DB_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_defense_app_db_build(output_path=args.output_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    table_counts = payload["table_counts"]
    if not isinstance(table_counts, dict):
        raise TypeError("runner payload table_counts must be a dict")
    summary = ", ".join(
        f"{table_name}={count}" for table_name, count in sorted(table_counts.items())
    )
    log.info(
        f"答辩展示 SQLite 构建完成: database={payload['database_path']}, {summary}",
        source=LOG_SOURCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
