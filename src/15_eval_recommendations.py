from __future__ import annotations

import argparse
import json

from fashion_trend.foundation import logging as log
from fashion_trend.recommendation.contracts import RECOMMENDATION_METHODS
from fashion_trend.recommendation.evaluation.runner import run_recommendation_evaluation
from fashion_trend.recommendation.features.cache import (
    RECOMMENDABLE_POOL_MANIFEST_KEY,
    read_recommendable_pool_cache,
    recommendable_pool_cache_exists,
    recommendable_pool_cache_fresh,
)
from fashion_trend.recommendation.paths import (
    EVALUATION_LABELS_PATH,
    FEATURE_CACHE_METADATA_PATH,
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
            "feature_cache_metadata": str(FEATURE_CACHE_METADATA_PATH),
        }
        recommendations = read_recommendations(output_paths.recommendations)
        time_windows = read_time_windows(TIME_WINDOWS_PATH)
        payload = run_recommendation_evaluation(
            method=args.method,
            recommendations=recommendations,
            target_users=read_target_users(TARGET_USERS_PATH),
            labels=read_evaluation_labels(EVALUATION_LABELS_PATH),
            recommendable_pool=read_cached_recommendable_pool_for_evaluation(
                time_windows
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


def read_cached_recommendable_pool_for_evaluation(time_windows):
    if not FEATURE_CACHE_METADATA_PATH.exists():
        raise RuntimeError(_missing_recommendable_pool_cache_message())
    if not recommendable_pool_cache_exists(time_windows):
        raise RuntimeError(_missing_recommendable_pool_cache_message())

    input_artifacts = _recommendable_pool_cache_input_artifacts()
    if not recommendable_pool_cache_fresh(time_windows, input_artifacts):
        raise RuntimeError(_missing_recommendable_pool_cache_message())
    return read_recommendable_pool_cache(time_windows)


def _recommendable_pool_cache_input_artifacts() -> dict[str, str]:
    try:
        manifest = json.loads(FEATURE_CACHE_METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(_missing_recommendable_pool_cache_message()) from error
    if not isinstance(manifest, dict):
        raise RuntimeError(_missing_recommendable_pool_cache_message())
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError(_missing_recommendable_pool_cache_message())
    entry = entries.get(RECOMMENDABLE_POOL_MANIFEST_KEY)
    if not isinstance(entry, dict):
        raise RuntimeError(_missing_recommendable_pool_cache_message())
    input_artifacts = entry.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise RuntimeError(_missing_recommendable_pool_cache_message())
    return {str(key): str(value) for key, value in input_artifacts.items()}


def _missing_recommendable_pool_cache_message() -> str:
    return (
        "recommendable_pool feature cache 缺失或已过期。"
        "请先运行 `uv run python src/16_run_recommendation_experiment.py "
        "--experiment main --force-cache` 重建缓存；如果候选也已过期，"
        "请改用 `--force-candidates` 或 `--force-rebuild-all`；15 不会静默重建。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
