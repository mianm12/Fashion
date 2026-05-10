from __future__ import annotations

import argparse

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.foundation import logging as log
from fashion_trend.recommendation.experiments.runner import (
    RecommendationExperimentContext,
    run_recommendation_experiment,
)
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.readers import read_trend_model_predictions

LOG_SOURCE = "recommendation-experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        trend_prediction_path = OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv"
        payload = run_recommendation_experiment(
            experiment_id=args.experiment,
            force=args.force,
            context=RecommendationExperimentContext(
                transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
                article_attributes=read_article_attribute_edges(
                    GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
                ),
                trend_predictions=read_trend_model_predictions(trend_prediction_path),
                input_paths={
                    "weekly_transactions": str(WEEKLY_TRANSACTIONS_PATH),
                    "article_attributes": str(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
                    "trend_predictions": str(trend_prediction_path),
                },
                trend_model_source=str(trend_prediction_path),
            ),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(
        f"推荐实验已写出: {payload['experiment_path']}",
        source=LOG_SOURCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
