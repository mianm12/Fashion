from __future__ import annotations

from fashion_trend.catalog.articles import clean_articles_file
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.paths import PATH

LOG_SOURCE = "clean-articles"


def main() -> int:
    try:
        log.info(f"输入文件: {PATH['raw_articles']}", source=LOG_SOURCE)
        log.info(
            "业务阶段: 商品字段清洗/稳妥字段裁剪 -> articles_clean.csv",
            source=LOG_SOURCE,
        )
        row_count = clean_articles_file(
            raw_articles_path=PATH["raw_articles"],
            mvp_output_path=PATH["interim_articles_clean_mvp"],
            clean_output_path=PATH["interim_articles_clean"],
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"已写出商品中间表行数: {row_count:,}", source=LOG_SOURCE)
    log.info(f"MVP 输出文件: {PATH['interim_articles_clean_mvp']}", source=LOG_SOURCE)
    log.info(f"稳妥版输出文件: {PATH['interim_articles_clean']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
