from __future__ import annotations

from fashion_trend import log
from fashion_trend.articles import build_attribute_graph_files
from fashion_trend.config import GRAPH_DIR, PATH

LOG_SOURCE = "attribute-graph"


def main() -> int:
    try:
        output_counts = build_attribute_graph_files(
            clean_articles_path=PATH["interim_articles_clean"],
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
