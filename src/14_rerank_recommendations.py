from __future__ import annotations

import argparse

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.foundation import logging as log
from fashion_trend.recommendation.contracts import RECOMMENDATION_METHODS
from fashion_trend.recommendation.paths import (
    ARTICLE_PRODUCT_MAP_PATH,
    CUSTOMER_PROFILE_PATH,
    FEATURE_CACHE_METADATA_PATH,
    RECOMMEND_METADATA_PATH,
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    USER_PROFILE_PATH,
    candidate_items_path,
)
from fashion_trend.recommendation.perf import StageTimer, format_stage_log
from fashion_trend.recommendation.readers import (
    read_article_product_map,
    read_candidate_items,
    read_customer_profile,
    read_target_users,
    read_time_windows,
    read_user_profile,
)
from fashion_trend.recommendation.registry import get_recommendation_method
from fashion_trend.recommendation.runner import (
    method_input_paths_for_artifacts,
    run_recommendation_method_by_window,
)
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
        timer = StageTimer("method", details={"method": args.method})
        method = get_recommendation_method(args.method)
        candidates = None
        candidate_path = None
        if method.default_candidate_strategy is not None:
            candidate_path = candidate_items_path(method.default_candidate_strategy)
            candidates = read_candidate_items(candidate_path)
        trend_prediction_path = None
        trend_predictions = None
        if "trend_score" in method.required_features:
            trend_prediction_path = OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv"
            trend_predictions = read_trend_model_predictions(trend_prediction_path)
        user_profile = None
        if "sim_score" in method.required_features and USER_PROFILE_PATH.exists():
            user_profile = read_user_profile(USER_PROFILE_PATH)
        if method.name == "enhanced_pop_similarity_trend":
            read_customer_profile(CUSTOMER_PROFILE_PATH)
            read_article_product_map(ARTICLE_PRODUCT_MAP_PATH)
            if not FEATURE_CACHE_METADATA_PATH.exists():
                raise FileNotFoundError(
                    f"增强推荐特征缓存 metadata 不存在: {FEATURE_CACHE_METADATA_PATH}"
                )
        available_input_paths = {
            "recommendation_inputs": str(RECOMMEND_METADATA_PATH),
            "weekly_transactions": str(WEEKLY_TRANSACTIONS_PATH),
            "article_attributes": str(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
            "time_windows": str(TIME_WINDOWS_PATH),
            "target_users": str(TARGET_USERS_PATH),
        }
        if USER_PROFILE_PATH.exists():
            available_input_paths["user_profile"] = str(USER_PROFILE_PATH)
        if candidate_path is not None:
            available_input_paths["candidate_items"] = str(candidate_path)
            available_input_paths["candidate_metadata"] = str(
                candidate_path.with_name("metadata.json")
            )
        if FEATURE_CACHE_METADATA_PATH.exists():
            available_input_paths["feature_cache_metadata"] = str(
                FEATURE_CACHE_METADATA_PATH
            )
        if method.name == "enhanced_pop_similarity_trend":
            available_input_paths["customer_profile"] = str(CUSTOMER_PROFILE_PATH)
            available_input_paths["article_product_map"] = str(ARTICLE_PRODUCT_MAP_PATH)
        if trend_prediction_path is not None:
            available_input_paths["trend_predictions"] = str(trend_prediction_path)
        input_paths = method_input_paths_for_artifacts(
            args.method,
            available_input_paths,
        )
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
    timer.rows = int(result.metadata["recommendation_rows"])
    log.info(format_stage_log(timer.finish()), source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
