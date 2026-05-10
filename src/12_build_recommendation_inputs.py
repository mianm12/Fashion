from __future__ import annotations

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
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
            },
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
    timer.rows = len(artifacts.user_profile)
    timer.details.update(
        {
            "time_windows": len(artifacts.time_windows),
            "target_users": len(artifacts.target_users),
            "evaluation_labels": len(artifacts.evaluation_labels),
            "user_profile": len(artifacts.user_profile),
        }
    )
    log.info(format_stage_log(timer.finish()), source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
