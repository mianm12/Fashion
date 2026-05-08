from __future__ import annotations

from fashion_trend.catalog.graph import build_attribute_graph_files
from fashion_trend.catalog.paths import ARTICLES_CLEAN_PATH, GRAPH_DIR
from fashion_trend.foundation import logging as log

LOG_SOURCE = "attribute-graph"


def main() -> int:
    try:
        log.info(f"输入文件: {ARTICLES_CLEAN_PATH}", source=LOG_SOURCE)
        log.info(
            "业务阶段: 商品属性抽取 -> attribute graph CSV 文件",
            source=LOG_SOURCE,
        )
        output_counts = build_attribute_graph_files(
            clean_articles_path=ARTICLES_CLEAN_PATH,
            graph_dir=GRAPH_DIR,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    for output_name, row_count in output_counts.items():
        log.info(f"{output_name}: {row_count:,} 行", source=LOG_SOURCE)
    log.info(f"属性图输出目录: {GRAPH_DIR}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
