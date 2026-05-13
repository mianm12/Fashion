from __future__ import annotations

import pandas as pd

from fashion_trend.catalog.paths import (
    ARTICLES_CLEAN_PATH,
    GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH,
)
from fashion_trend.catalog.readers import (
    read_article_attribute_edges,
    read_clean_articles,
)
from fashion_trend.datasets.paths import RAW_CUSTOMERS_PATH
from fashion_trend.foundation import logging as log
from fashion_trend.recommendation.inputs import build_and_write_recommendation_inputs
from fashion_trend.recommendation.perf import StageTimer, format_stage_log
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.readers import read_trend_model_predictions

LOG_SOURCE = "recommendation-inputs"
TREND_PREDICTIONS_PATH = OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv"


def main() -> int:
    """Build recommendation time windows and model input artifacts."""
    timer = StageTimer("input_build")
    try:
        artifacts = build_and_write_recommendation_inputs(
            transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
            article_attributes=read_article_attribute_edges(
                GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
            ),
            trend_predictions=read_trend_model_predictions(TREND_PREDICTIONS_PATH),
            input_paths={
                "weekly_transactions": str(WEEKLY_TRANSACTIONS_PATH),
                "article_attributes": str(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
                "trend_predictions": str(TREND_PREDICTIONS_PATH),
                "raw_customers": str(RAW_CUSTOMERS_PATH),
                "clean_articles": str(ARTICLES_CLEAN_PATH),
            },
            customers=pd.read_csv(
                RAW_CUSTOMERS_PATH,
                dtype={"customer_id": "string"},
            ),
            clean_articles=read_clean_articles(ARTICLES_CLEAN_PATH),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(
        f"推荐窗口: rows={len(artifacts.time_windows):,}",
        source=LOG_SOURCE,
    )
    log.info(
        f"目标用户: rows={len(artifacts.target_users):,}",
        source=LOG_SOURCE,
    )
    log.info(
        f"评价标签: rows={len(artifacts.evaluation_labels):,}",
        source=LOG_SOURCE,
    )
    log.info(
        f"用户画像: rows={len(artifacts.user_profile):,}",
        source=LOG_SOURCE,
    )
    if artifacts.customer_profile is not None:
        log.info(
            f"顾客画像: rows={len(artifacts.customer_profile):,}",
            source=LOG_SOURCE,
        )
    if artifacts.article_product_map is not None:
        log.info(
            f"商品产品映射: rows={len(artifacts.article_product_map):,}",
            source=LOG_SOURCE,
        )
    timer.rows = len(artifacts.user_profile)
    timer.details.update(
        {
            "time_windows": len(artifacts.time_windows),
            "target_users": len(artifacts.target_users),
            "evaluation_labels": len(artifacts.evaluation_labels),
            "user_profile": len(artifacts.user_profile),
            "customer_profile": (
                len(artifacts.customer_profile)
                if artifacts.customer_profile is not None
                else 0
            ),
            "article_product_map": (
                len(artifacts.article_product_map)
                if artifacts.article_product_map is not None
                else 0
            ),
        }
    )
    log.info(format_stage_log(timer.finish()), source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
