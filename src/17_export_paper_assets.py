from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fashion_trend.foundation import logging as log
from fashion_trend.reports.runner import (
    PaperAssetsExportConfig,
    run_paper_assets_export,
)

LOG_SOURCE = "paper-assets-export"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--trend-week", type=int, default=103)
    parser.add_argument("--figure-format", default="svg,png")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_paper_assets_export(
            PaperAssetsExportConfig(
                case_count=args.case_count,
                top_k=args.top_k,
                trend_week=args.trend_week,
                figure_formats=_parse_figure_formats(args.figure_format),
                output_dir=args.output_dir,
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(
        f"论文素材导出完成: figures={payload['figure_count']}, "
        f"tables={payload['table_count']}, cases={payload['case_count']}",
        source=LOG_SOURCE,
    )
    return 0


def _parse_figure_formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {"svg", "png"}
    if not formats or not set(formats).issubset(allowed):
        raise ValueError(f"figure-format 只支持 svg,png: {value}")
    if len(set(formats)) != len(formats):
        raise ValueError(f"figure-format 不能重复: {value}")
    return formats


if __name__ == "__main__":
    raise SystemExit(main())
