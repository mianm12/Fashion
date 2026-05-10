from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic, write_parquet_atomic
from fashion_trend.recommendation.contracts import (
    CANDIDATE_ITEM_COLUMNS,
    RECOMMENDATION_CANDIDATE_STRATEGIES,
    RECOMMENDATION_CANDIDATES_PER_SOURCE,
)
from fashion_trend.recommendation.freshness import build_artifact_metadata
from fashion_trend.recommendation.paths import candidate_items_path
from fashion_trend.recommendation.retrieval.attributes import (
    build_attribute_similarity_candidates,
)
from fashion_trend.recommendation.retrieval.popularity import (
    build_recent_popularity_candidates,
)
from fashion_trend.recommendation.retrieval.trend import build_trend_candidates

SOURCE_ORDER = {"popularity": 0, "similarity": 1, "trend": 2}
SOURCE_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "article_id",
    "source",
    "source_rank",
)
CANDIDATE_INPUT_KEYS_BY_STRATEGY = {
    "popularity": ("weekly_transactions", "time_windows", "target_users"),
    "similarity": (
        "article_attributes",
        "time_windows",
        "target_users",
        "user_profile",
    ),
    "trend_union": (
        "article_attributes",
        "trend_predictions",
        "time_windows",
        "target_users",
    ),
    "default": (
        "weekly_transactions",
        "article_attributes",
        "trend_predictions",
        "time_windows",
        "target_users",
        "user_profile",
    ),
}


def validate_candidate_strategy(strategy: str) -> None:
    if strategy not in RECOMMENDATION_CANDIDATE_STRATEGIES:
        choices = ", ".join(RECOMMENDATION_CANDIDATE_STRATEGIES)
        raise ValueError(f"未知候选 strategy: {strategy}. 可用 strategy: {choices}")


def candidate_input_paths_for_strategy(
    strategy: str,
    input_paths: dict[str, str] | None,
) -> dict[str, str]:
    validate_candidate_strategy(strategy)
    available_paths = dict(input_paths or {})
    return {
        key: available_paths[key]
        for key in CANDIDATE_INPUT_KEYS_BY_STRATEGY[strategy]
        if key in available_paths
    }


def build_candidate_items(
    strategy: str,
    source_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    validate_candidate_strategy(strategy)
    non_empty_frames = [frame for frame in source_frames if not frame.empty]
    if strategy == "trend_union" and not non_empty_frames:
        raise FileNotFoundError("trend_union strategy requires trend source candidates")
    if not non_empty_frames:
        return pd.DataFrame(columns=CANDIDATE_ITEM_COLUMNS)

    sources = pd.concat(non_empty_frames, ignore_index=True)
    sources = _prepare_source_frame(sources)
    sorted_sources = sources.sort_values(
        [
            "split",
            "cutoff_week",
            "label_week",
            "customer_id",
            "article_id",
            "source_rank",
            "_source_order",
        ],
        kind="mergesort",
    )
    grouped = sorted_sources.groupby(
        ["split", "cutoff_week", "label_week", "customer_id", "article_id"],
        sort=False,
    )
    result = grouped.agg(
        candidate_sources=("source", _join_sources),
        primary_source=("source", "first"),
        best_source_rank=("source_rank", "min"),
    ).reset_index()
    result.insert(3, "strategy", strategy)
    return result.loc[:, list(CANDIDATE_ITEM_COLUMNS)]


def build_source_frames_for_frames(
    strategy: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    user_profile: pd.DataFrame | None,
) -> list[pd.DataFrame]:
    validate_candidate_strategy(strategy)
    frames: list[pd.DataFrame] = []
    if strategy in {"popularity", "default"}:
        frames.append(
            build_recent_popularity_candidates(
                transactions,
                windows,
                target_users,
                top_n=RECOMMENDATION_CANDIDATES_PER_SOURCE,
            )
        )
    if strategy in {"similarity", "default"}:
        if user_profile is None or article_attributes is None:
            raise FileNotFoundError(
                "similarity strategy requires user profile and article attributes"
            )
        frames.append(
            build_attribute_similarity_candidates(
                user_profile,
                article_attributes,
                windows,
                target_users,
                top_n=RECOMMENDATION_CANDIDATES_PER_SOURCE,
            )
        )
    if strategy in {"trend_union", "default"}:
        if trend_predictions is None or article_attributes is None:
            raise FileNotFoundError(
                "trend strategy requires trend predictions and article attributes"
            )
        frames.append(
            build_trend_candidates(
                trend_predictions,
                article_attributes,
                windows,
                target_users,
                top_n=RECOMMENDATION_CANDIDATES_PER_SOURCE,
            )
        )
    return frames


def build_and_write_candidate_items(
    strategy: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    input_paths: dict[str, str] | None = None,
) -> Path:
    candidate_input_paths = candidate_input_paths_for_strategy(strategy, input_paths)
    source_frames = build_source_frames_for_frames(
        strategy=strategy,
        transactions=transactions,
        article_attributes=article_attributes,
        trend_predictions=trend_predictions,
        windows=windows,
        target_users=target_users,
        user_profile=user_profile,
    )
    candidates = build_candidate_items(strategy, source_frames)
    output_path = candidate_items_path(strategy)
    write_parquet_atomic(candidates, output_path)
    candidate_rows = int(len(candidates))
    write_json_atomic(
        {
            **build_artifact_metadata(
                name="recommendation_candidates",
                input_artifacts=candidate_input_paths,
                output_artifacts={"candidate_items": str(output_path)},
                schema_version=1,
                algorithm_version="recommendation-candidates-v1",
                config={
                    "strategy": strategy,
                    "candidates_per_source": RECOMMENDATION_CANDIDATES_PER_SOURCE,
                },
                row_counts={"candidate_rows": candidate_rows},
            ),
            "strategy": strategy,
            "candidate_rows": candidate_rows,
        },
        output_path.with_name("metadata.json"),
    )
    return output_path


def _prepare_source_frame(source_frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(source_frame.columns))
    if missing_columns:
        raise ValueError(f"候选 source frame 缺少必要字段: {missing_columns}")
    prepared = source_frame.loc[:, SOURCE_COLUMNS].copy()
    prepared["article_id"] = prepared["article_id"].astype(str)
    prepared["customer_id"] = prepared["customer_id"].astype(str)
    prepared["source"] = prepared["source"].astype(str)
    unknown_sources = sorted(set(prepared["source"]) - set(SOURCE_ORDER))
    if unknown_sources:
        raise ValueError(f"未知候选 source: {unknown_sources}")
    prepared["source_rank"] = pd.to_numeric(
        prepared["source_rank"],
        errors="raise",
    ).astype(int)
    prepared["_source_order"] = prepared["source"].map(SOURCE_ORDER).astype(int)
    return prepared


def _join_sources(values: pd.Series) -> str:
    sources = sorted(set(values.astype(str)), key=SOURCE_ORDER.__getitem__)
    return "|".join(sources)
