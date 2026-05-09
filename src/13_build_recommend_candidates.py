from __future__ import annotations

import argparse

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.foundation import logging as log
from fashion_trend.recommendation.contracts import RECOMMENDATION_CANDIDATE_STRATEGIES
from fashion_trend.recommendation.paths import (
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    USER_PROFILE_PATH,
)
from fashion_trend.recommendation.readers import (
    read_target_users,
    read_time_windows,
    read_user_profile,
)
from fashion_trend.recommendation.retrieval.candidates import (
    build_and_write_candidate_items,
)
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.readers import read_trend_model_predictions

LOG_SOURCE = "recommend-candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=RECOMMENDATION_CANDIDATE_STRATEGIES,
        default="default",
    )
    return parser.parse_args()


def main() -> int:
    """Build strategy-scoped recommendation candidate items."""
    args = parse_args()
    try:
        article_attributes = None
        user_profile = None
        trend_predictions = None
        if args.strategy in {"similarity", "trend_union", "default"}:
            article_attributes = read_article_attribute_edges(
                GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
            )
        if args.strategy in {"similarity", "default"}:
            user_profile = read_user_profile(USER_PROFILE_PATH)
        if args.strategy in {"trend_union", "default"}:
            trend_predictions = read_trend_model_predictions(
                OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv"
            )
        output_path = build_and_write_candidate_items(
            strategy=args.strategy,
            transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
            article_attributes=article_attributes,
            trend_predictions=trend_predictions,
            windows=read_time_windows(TIME_WINDOWS_PATH),
            target_users=read_target_users(TARGET_USERS_PATH),
            user_profile=user_profile,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"推荐候选已写出: {output_path}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
