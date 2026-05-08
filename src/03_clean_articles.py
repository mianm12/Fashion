from __future__ import annotations

from fashion_trend.catalog.articles import clean_articles_file
from fashion_trend.catalog.paths import ARTICLES_CLEAN_MVP_PATH, ARTICLES_CLEAN_PATH
from fashion_trend.datasets.paths import RAW_ARTICLES_PATH
from fashion_trend.foundation import logging as log

LOG_SOURCE = "clean-articles"


def main() -> int:
    """商品清洗阶段入口，稳定写出 articles_clean_mvp.csv 与 articles_clean.csv。"""
    try:
        log.info(f"输入文件: {RAW_ARTICLES_PATH}", source=LOG_SOURCE)
        log.info(
            "业务阶段: 商品字段清洗/稳妥字段裁剪 -> articles_clean.csv",
            source=LOG_SOURCE,
        )
        row_count = clean_articles_file(
            raw_articles_path=RAW_ARTICLES_PATH,
            mvp_output_path=ARTICLES_CLEAN_MVP_PATH,
            clean_output_path=ARTICLES_CLEAN_PATH,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"已写出商品中间表行数: {row_count:,}", source=LOG_SOURCE)
    log.info(f"MVP 输出文件: {ARTICLES_CLEAN_MVP_PATH}", source=LOG_SOURCE)
    log.info(f"稳妥版输出文件: {ARTICLES_CLEAN_PATH}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
