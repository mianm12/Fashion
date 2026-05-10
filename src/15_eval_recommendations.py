from __future__ import annotations

import argparse

from fashion_trend.foundation import logging as log
from fashion_trend.recommendation.contracts import RECOMMENDATION_METHODS
from fashion_trend.recommendation.evaluation.runner import (
    build_recommendable_pool_for_windows,
    run_recommendation_evaluation,
)
from fashion_trend.recommendation.paths import (
    EVALUATION_LABELS_PATH,
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    method_output_paths,
)
from fashion_trend.recommendation.perf import StageTimer, format_stage_log
from fashion_trend.recommendation.readers import (
    read_evaluation_labels,
    read_recommendations,
    read_target_users,
    read_time_windows,
)
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions

LOG_SOURCE = "recommendation-eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=RECOMMENDATION_METHODS, required=True)
    parser.add_argument("--strict-missing-users", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        timer = StageTimer("evaluation", details={"method": args.method})
        output_paths = method_output_paths(args.method)
        input_paths = {
            "recommendations": str(output_paths.recommendations),
            "target_users": str(TARGET_USERS_PATH),
            "evaluation_labels": str(EVALUATION_LABELS_PATH),
            "time_windows": str(TIME_WINDOWS_PATH),
            "weekly_transactions": str(WEEKLY_TRANSACTIONS_PATH),
        }
        recommendations = read_recommendations(output_paths.recommendations)
        payload = run_recommendation_evaluation(
            method=args.method,
            recommendations=recommendations,
            target_users=read_target_users(TARGET_USERS_PATH),
            labels=read_evaluation_labels(EVALUATION_LABELS_PATH),
            recommendable_pool=build_recommendable_pool_for_windows(
                read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
                read_time_windows(TIME_WINDOWS_PATH),
            ),
            input_paths=input_paths,
            strict_missing_users=args.strict_missing_users,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(
        f"推荐评价完成: method={args.method}, splits={sorted(payload['metrics'])}",
        source=LOG_SOURCE,
    )
    timer.rows = len(recommendations)
    log.info(format_stage_log(timer.finish()), source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
