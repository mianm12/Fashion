from __future__ import annotations

import argparse

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.foundation import logging as log
from fashion_trend.recommendation.contracts import RECOMMENDATION_METHODS
from fashion_trend.recommendation.paths import (
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    USER_PROFILE_PATH,
    candidate_items_path,
)
from fashion_trend.recommendation.readers import (
    read_candidate_items,
    read_target_users,
    read_time_windows,
    read_user_profile,
)
from fashion_trend.recommendation.registry import get_recommendation_method
from fashion_trend.recommendation.runner import run_recommendation_method_by_window
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.readers import read_trend_model_predictions

LOG_SOURCE = "recommendation-rerank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=RECOMMENDATION_METHODS, required=True)
    parser.add_argument("--exclude-seen", action="store_true", default=True)
    parser.add_argument("--include-seen", action="store_false", dest="exclude_seen")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        method = get_recommendation_method(args.method)
        candidates = None
        candidate_path = None
        if method.default_candidate_strategy is not None:
            candidate_path = candidate_items_path(method.default_candidate_strategy)
            candidates = read_candidate_items(candidate_path)
        trend_prediction_path = None
        trend_predictions = None
        if method.name == "pop_similarity_trend":
            trend_prediction_path = OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv"
            trend_predictions = read_trend_model_predictions(trend_prediction_path)
        user_profile = None
        if "sim_score" in method.required_features and USER_PROFILE_PATH.exists():
            user_profile = read_user_profile(USER_PROFILE_PATH)
        input_paths = {
            "weekly_transactions": str(WEEKLY_TRANSACTIONS_PATH),
            "article_attributes": str(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
            "time_windows": str(TIME_WINDOWS_PATH),
            "target_users": str(TARGET_USERS_PATH),
        }
        if candidate_path is not None:
            input_paths["candidate_items"] = str(candidate_path)
        if user_profile is not None:
            input_paths["user_profile"] = str(USER_PROFILE_PATH)
        if trend_prediction_path is not None:
            input_paths["trend_predictions"] = str(trend_prediction_path)
        result = run_recommendation_method_by_window(
            method_name=args.method,
            transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
            article_attributes=read_article_attribute_edges(
                GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
            ),
            windows=read_time_windows(TIME_WINDOWS_PATH),
            target_users=read_target_users(TARGET_USERS_PATH),
            candidates=candidates,
            user_profile=user_profile,
            trend_predictions=trend_predictions,
            exclude_seen=args.exclude_seen,
            collect_result=False,
            input_paths=input_paths,
            trend_model_source=(
                str(trend_prediction_path)
                if trend_prediction_path is not None
                else None
            ),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(
        "推荐结果已写出: "
        f"method={args.method}, rows={result.metadata['recommendation_rows']}",
        source=LOG_SOURCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
