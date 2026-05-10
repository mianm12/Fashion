from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.foundation.paths import OUTPUT_DIR, PROCESSED_DIR

RECOMMEND_DIR = PROCESSED_DIR / "recommend"
OUTPUT_RECOMMENDATION_DIR = OUTPUT_DIR / "recommendation"

TIME_WINDOWS_PATH = RECOMMEND_DIR / "time_windows.parquet"
TARGET_USERS_PATH = RECOMMEND_DIR / "target_users.parquet"
EVALUATION_LABELS_PATH = RECOMMEND_DIR / "evaluation_labels.parquet"
USER_PROFILE_PATH = RECOMMEND_DIR / "user_profile.parquet"
CANDIDATES_DIR = RECOMMEND_DIR / "candidates"
EXPERIMENTS_DIR = OUTPUT_RECOMMENDATION_DIR / "experiments"


@dataclass(frozen=True)
class RecommendationOutputPaths:
    output_dir: Path
    recommendations: Path
    recommendation_items: Path
    recommendation_items_csv: Path
    params: Path
    metadata: Path
    metrics: Path


def candidate_items_path(strategy: str) -> Path:
    _validate_recommendation_path_segment(strategy, "strategy")
    return CANDIDATES_DIR / strategy / "candidate_items.parquet"


def method_output_paths(method: str) -> RecommendationOutputPaths:
    _validate_recommendation_path_segment(method, "method")
    output_dir = OUTPUT_RECOMMENDATION_DIR / method
    return RecommendationOutputPaths(
        output_dir=output_dir,
        recommendations=output_dir / "recommendations.csv",
        recommendation_items=output_dir / "recommendation_items.parquet",
        recommendation_items_csv=output_dir / "recommendation_items.csv",
        params=output_dir / "params.json",
        metadata=output_dir / "metadata.json",
        metrics=output_dir / "metrics.json",
    )


def experiment_dir(experiment_id: str) -> Path:
    _validate_recommendation_path_segment(experiment_id, "experiment_id")
    return EXPERIMENTS_DIR / experiment_id


def experiment_run_dir(experiment_id: str, run_id: str) -> Path:
    _validate_recommendation_path_segment(run_id, "run_id")
    return experiment_dir(experiment_id) / "runs" / run_id


def _validate_recommendation_path_segment(segment: str, source_name: str) -> None:
    try:
        validate_safe_path_segment(segment, source_name)
    except ValueError as error:
        raise ValueError(f"{source_name} 不是安全的路径片段: {segment}") from error
