from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.contracts import (
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
    RECOMMENDATION_CORE_ATTR_TYPES,
    RECOMMENDATION_TOP_K,
    RECOMMENDATION_TREND_ATTR_WEIGHTS,
)
from fashion_trend.recommendation.features.cache import (
    FEATURE_CACHE_ALGORITHM_VERSION,
    FEATURE_CACHE_SCHEMA_VERSION,
)
from fashion_trend.recommendation.fingerprints import build_input_fingerprints
from fashion_trend.recommendation.freshness import (
    assert_fresh_metadata,
    build_artifact_metadata,
)
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationMethod,
    RecommendationResult,
)
from fashion_trend.recommendation.outputs import (
    RecommendationResultChunkWriter,
    build_recommendations_csv,
    format_recommendation_items,
    write_recommendation_result,
)
from fashion_trend.recommendation.paths import (
    FEATURE_CACHE_METADATA_PATH,
    feature_cache_partition_metadata_path,
    feature_cache_partition_path,
    method_output_paths,
)
from fashion_trend.recommendation.ranking.backfill import append_backfill_items
from fashion_trend.recommendation.ranking.filters import (
    filter_seen_items_by_source_policy,
)
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items
from fashion_trend.recommendation.registry import get_recommendation_method

WINDOW_COLUMNS = ("split", "cutoff_week", "label_week")
COMMON_METHOD_INPUT_KEYS = (
    "recommendation_inputs",
    "weekly_transactions",
    "time_windows",
    "target_users",
)
SCORE_COLUMNS = ENHANCED_RECOMMENDATION_SCORE_COLUMNS
FEATURE_REQUIRED_INPUT_KEYS = {
    "candidate_seen_flags": ("weekly_transactions", "candidate_items"),
    "popularity_scores": ("weekly_transactions", "candidate_items"),
    "recent_scores": ("weekly_transactions", "candidate_items"),
    "similarity_scores": ("article_attributes", "user_profile", "candidate_items"),
    "trend_scores": ("article_attributes", "trend_predictions", "candidate_items"),
    "reorder_scores": ("weekly_transactions", "candidate_items"),
    "variant_scores": (
        "weekly_transactions",
        "article_product_map",
        "candidate_items",
    ),
    "age_popularity_scores": (
        "weekly_transactions",
        "customer_profile",
        "candidate_items",
    ),
    "preference_popularity_scores": (
        "weekly_transactions",
        "article_attributes",
        "user_profile",
        "candidate_items",
    ),
    "source_rank_scores": ("candidate_items",),
    "source_count_scores": ("candidate_items",),
}
FEATURE_JOIN_SPECS = {
    "pop_score": (
        "popularity_scores",
        ["split", "cutoff_week", "label_week", "strategy", "article_id"],
    ),
    "recent_score": (
        "recent_scores",
        ["split", "cutoff_week", "label_week", "strategy", "article_id"],
    ),
    "sim_score": (
        "similarity_scores",
        [
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
    ),
    "trend_score": (
        "trend_scores",
        ["split", "cutoff_week", "label_week", "strategy", "article_id"],
    ),
    "reorder_score": (
        "reorder_scores",
        [
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
    ),
    "variant_score": (
        "variant_scores",
        [
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
    ),
    "age_pop_score": (
        "age_popularity_scores",
        [
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
    ),
    "preference_pop_score": (
        "preference_popularity_scores",
        [
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
    ),
    "source_rank_score": (
        "source_rank_scores",
        [
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
    ),
    "source_count_score": (
        "source_count_scores",
        [
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
    ),
}
BACKFILL_MODE_BY_METHOD = {
    "global_popularity": "popularity",
    "recent_popularity": "recent",
    "attribute_similarity": "recent",
    "pop_similarity": "recent",
    "pop_similarity_trend": "recent",
}


def method_input_artifacts(
    *,
    base_input_paths: dict[str, str],
    candidate_items: str | None,
    candidate_metadata: str | None,
    feature_cache_metadata: str | None,
    feature_partitions: list[str],
) -> dict[str, str]:
    artifacts = dict(base_input_paths)
    if candidate_items is not None:
        artifacts["candidate_items"] = candidate_items
    if candidate_metadata is not None:
        artifacts["candidate_metadata"] = candidate_metadata
    if feature_cache_metadata is not None:
        artifacts["feature_cache_metadata"] = feature_cache_metadata
    for index, path in enumerate(feature_partitions):
        artifacts[f"feature_partition_{index:04d}"] = path
    return artifacts


def feature_artifact_paths_for_method_window(
    *,
    method_name: str,
    strategy: str,
    window: dict[str, object],
    include_seen: bool,
) -> list[str]:
    method = get_recommendation_method(method_name)
    feature_names = [
        FEATURE_JOIN_SPECS[score_column][0]
        for score_column in method.required_features
        if score_column in FEATURE_JOIN_SPECS
    ]
    if include_seen:
        feature_names.insert(0, "candidate_seen_flags")

    artifacts: list[str] = []
    for feature_name in feature_names:
        partition = feature_cache_partition_path(
            feature_name,
            strategy=strategy,
            split=str(window["split"]),
            cutoff_week=int(window["cutoff_week"]),
        )
        metadata = feature_cache_partition_metadata_path(
            feature_name,
            strategy=strategy,
            split=str(window["split"]),
            cutoff_week=int(window["cutoff_week"]),
        )
        artifacts.extend([str(partition), str(metadata)])
    return artifacts


def method_input_paths_for_artifacts(
    method_name: str,
    available_paths: dict[str, str],
) -> dict[str, str]:
    """Select only the artifacts a recommendation method output depends on."""
    method = get_recommendation_method(method_name)
    selected = {
        key: available_paths[key]
        for key in COMMON_METHOD_INPUT_KEYS
        if key in available_paths
    }
    if (
        "sim_score" in method.required_features
        or "trend_score" in method.required_features
        or "preference_pop_score" in method.required_features
    ) and "article_attributes" in available_paths:
        selected["article_attributes"] = available_paths["article_attributes"]
    if "sim_score" in method.required_features and "user_profile" in available_paths:
        selected["user_profile"] = available_paths["user_profile"]
    if (
        "preference_pop_score" in method.required_features
        and "user_profile" in available_paths
    ):
        selected["user_profile"] = available_paths["user_profile"]
    if (
        "variant_score" in method.required_features
        and "article_product_map" in available_paths
    ):
        selected["article_product_map"] = available_paths["article_product_map"]
    if (
        "age_pop_score" in method.required_features
        and "customer_profile" in available_paths
    ):
        selected["customer_profile"] = available_paths["customer_profile"]
    strategy = method.default_candidate_strategy
    if strategy is not None:
        selected["candidate_items"] = _candidate_input_path(strategy, available_paths)
        selected["candidate_metadata"] = _candidate_metadata_input_path(
            strategy,
            selected["candidate_items"],
            available_paths,
        )
        if "feature_cache_metadata" in available_paths:
            selected["feature_cache_metadata"] = available_paths[
                "feature_cache_metadata"
            ]
    if (
        "trend_score" in method.required_features
        and "trend_predictions" in available_paths
    ):
        selected["trend_predictions"] = available_paths["trend_predictions"]
    return selected


def filter_cached_seen_items(
    candidates: pd.DataFrame,
    *,
    strategy: str,
    window: dict[str, object],
    input_paths: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, str, str]:
    path = feature_cache_partition_path(
        "candidate_seen_flags",
        strategy=strategy,
        split=str(window["split"]),
        cutoff_week=int(window["cutoff_week"]),
    )
    metadata_path = feature_cache_partition_metadata_path(
        "candidate_seen_flags",
        strategy=strategy,
        split=str(window["split"]),
        cutoff_week=int(window["cutoff_week"]),
    )
    if input_paths is not None:
        _validate_feature_cache_partition_fresh(
            feature_name="candidate_seen_flags",
            strategy=strategy,
            window=window,
            partition_path=path,
            metadata_path=metadata_path,
            current_input_paths=input_paths,
        )
    seen = pd.read_parquet(path)
    if seen.empty:
        return candidates.copy(), str(path), str(metadata_path)

    join_columns = [
        "split",
        "cutoff_week",
        "label_week",
        "strategy",
        "customer_id",
        "article_id",
    ]
    if "is_seen" in seen.columns:
        marker = seen.loc[:, [*join_columns, "is_seen"]].drop_duplicates()
        merged = candidates.merge(marker, on=join_columns, how="left")
        merged["is_seen"] = merged["is_seen"].fillna(False).astype(bool)
        filtered = filter_seen_items_by_source_policy(merged)
        return (
            filtered.loc[:, candidates.columns].reset_index(drop=True),
            str(path),
            str(metadata_path),
        )

    marker = seen.loc[:, join_columns].assign(_seen=True)
    merged = candidates.merge(marker, on=join_columns, how="left")
    return (
        merged.loc[merged["_seen"].isna(), candidates.columns].reset_index(drop=True),
        str(path),
        str(metadata_path),
    )


def build_cached_feature_frame_for_window(
    *,
    method_name: str,
    strategy: str,
    window: dict[str, object],
    candidates: pd.DataFrame,
    required_features: Sequence[str],
    input_paths: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    feature_frame = candidates.copy()
    for score_column in SCORE_COLUMNS:
        if score_column not in feature_frame.columns:
            feature_frame[score_column] = 0.0

    if candidates.empty:
        feature_frame["method"] = method_name
        return feature_frame, []

    split = str(window["split"])
    cutoff_week = int(window["cutoff_week"])
    used_artifacts: list[str] = []
    for score_column in required_features:
        if score_column not in FEATURE_JOIN_SPECS:
            raise ValueError(f"unknown cached score feature: {score_column}")
        feature_name, join_columns = FEATURE_JOIN_SPECS[score_column]
        path = feature_cache_partition_path(
            feature_name,
            strategy=strategy,
            split=split,
            cutoff_week=cutoff_week,
        )
        metadata_path = feature_cache_partition_metadata_path(
            feature_name,
            strategy=strategy,
            split=split,
            cutoff_week=cutoff_week,
        )
        if input_paths is not None:
            _validate_feature_cache_partition_fresh(
                feature_name=feature_name,
                strategy=strategy,
                window=window,
                partition_path=path,
                metadata_path=metadata_path,
                current_input_paths=input_paths,
            )
        scores = pd.read_parquet(path).loc[:, [*join_columns, score_column]]
        feature_frame = feature_frame.merge(
            scores.drop_duplicates(join_columns),
            on=join_columns,
            how="left",
            suffixes=("", "_cached"),
        )
        cached_column = f"{score_column}_cached"
        if cached_column in feature_frame.columns:
            feature_frame[score_column] = feature_frame[cached_column].fillna(
                feature_frame[score_column]
            )
            feature_frame = feature_frame.drop(columns=[cached_column])
        feature_frame[score_column] = feature_frame[score_column].fillna(0.0)
        used_artifacts.extend([str(path), str(metadata_path)])

    feature_frame["method"] = method_name
    return feature_frame, used_artifacts


def build_cached_recommendation_result_for_window(
    *,
    method: RecommendationMethod,
    method_name: str,
    strategy: str,
    window: dict[str, object],
    target_users: pd.DataFrame,
    candidates: pd.DataFrame,
    context: RecommendationContext,
    weights: dict[str, float],
    backfill_mode: str | None,
) -> RecommendationResult:
    _validate_cached_method_context(method, method_name, context)
    used_feature_artifacts: list[str] = []
    filtered_candidates = candidates
    if context.exclude_seen and not candidates.empty:
        filtered_candidates, seen_partition, seen_metadata = filter_cached_seen_items(
            candidates,
            strategy=strategy,
            window=window,
            input_paths=context.input_paths,
        )
        used_feature_artifacts.extend([seen_partition, seen_metadata])

    feature_frame, score_artifacts = build_cached_feature_frame_for_window(
        method_name=method_name,
        strategy=strategy,
        window=window,
        candidates=filtered_candidates,
        required_features=method.required_features,
        input_paths=context.input_paths,
    )
    used_feature_artifacts.extend(score_artifacts)
    ranked = rank_candidate_items(
        feature_frame,
        weights=weights,
        top_k=context.top_k,
        required_features=method.required_features,
    )
    underfilled_before = _underfilled_user_count(
        context.target_users,
        ranked,
        context.top_k,
    )
    counts_before = _recommendation_counts_by_target(context.target_users, ranked)
    ranked = append_backfill_items(
        context,
        candidates,
        ranked,
        weights,
        backfill_mode,
    )
    counts_after = _recommendation_counts_by_target(context.target_users, ranked)
    still_underfilled = int((counts_after["_count"] < context.top_k).sum())
    backfilled_user_count = int(
        (counts_after["_count"] > counts_before["_count"]).sum()
    )
    recommendation_items = format_recommendation_items(ranked)
    recommendations = build_recommendations_csv(recommendation_items, context.top_k)
    return RecommendationResult(
        recommendations=recommendations,
        recommendation_items=recommendation_items,
        params={
            "method": method_name,
            "method_type": method.method_type,
            "top_k": context.top_k,
            "exclude_seen": context.exclude_seen,
            "weights": dict(weights),
            "candidate_strategy": strategy,
            "score_features": list(method.required_features),
        },
        metadata={
            "method": method_name,
            "target_user_rows": int(len(target_users)),
            "candidate_rows": int(len(candidates)),
            "recommendation_rows": int(len(recommendations)),
            "recommendation_item_rows": int(len(recommendation_items)),
            "used_feature_artifacts": used_feature_artifacts,
            "backfill_mode": backfill_mode,
            "underfilled_user_count": underfilled_before,
            "backfilled_user_count": backfilled_user_count,
            "still_underfilled_user_count": still_underfilled,
            "candidate_strategy": strategy,
            **_source_level_seen_policy_metadata(strategy),
        },
    )


def run_recommendation_method(
    method_name: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    user_profile: pd.DataFrame | None = None,
    trend_predictions: pd.DataFrame | None = None,
    exclude_seen: bool = True,
    weights: dict[str, float] | None = None,
    input_paths: dict[str, str] | None = None,
    trend_model_source: str | None = None,
) -> RecommendationResult:
    method = get_recommendation_method(method_name)
    context = RecommendationContext(
        method=method_name,
        top_k=RECOMMENDATION_TOP_K,
        exclude_seen=exclude_seen,
        transactions=transactions,
        article_attributes=article_attributes,
        windows=windows,
        target_users=target_users,
        candidates=candidates,
        user_profile=user_profile,
        trend_predictions=trend_predictions,
        weights=weights,
        input_paths=input_paths,
        trend_model_source=trend_model_source,
    )
    result = method.build_recommendations(context)
    result.params.update(_params_for_method(method, method_name, exclude_seen, weights))
    result.metadata.update(
        _base_metadata(
            method_name,
            method.required_features,
            method.default_candidate_strategy,
            exclude_seen,
            weights,
            windows,
            input_paths,
            trend_model_source,
        )
    )
    write_recommendation_result(result)
    return result


def run_recommendation_method_by_window(
    method_name: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    user_profile: pd.DataFrame | None = None,
    trend_predictions: pd.DataFrame | None = None,
    exclude_seen: bool = True,
    weights: dict[str, float] | None = None,
    collect_result: bool = True,
    input_paths: dict[str, str] | None = None,
    trend_model_source: str | None = None,
) -> RecommendationResult:
    """Run a recommendation method window-by-window and stream large CSV outputs."""
    method = get_recommendation_method(method_name)
    params = _params_for_method(method, method_name, exclude_seen, weights)
    metadata: dict[str, object] = {
        **_base_metadata(
            method_name,
            method.required_features,
            method.default_candidate_strategy,
            exclude_seen,
            weights,
            windows,
            input_paths,
            trend_model_source,
        ),
        "candidate_rows": 0,
        "recommendation_rows": 0,
        "recommendation_item_rows": 0,
        "window_count": 0,
        "window_summaries": [],
    }
    recommendation_chunks: list[pd.DataFrame] = []
    item_chunks: list[pd.DataFrame] = []

    with RecommendationResultChunkWriter(method_name) as writer:
        for window in windows.loc[:, WINDOW_COLUMNS].to_dict("records"):
            window_target_users = _frame_for_window(target_users, window)
            window_candidates = _optional_frame_for_window(candidates, window)
            context = RecommendationContext(
                method=method_name,
                top_k=RECOMMENDATION_TOP_K,
                exclude_seen=exclude_seen,
                transactions=transactions,
                article_attributes=article_attributes,
                windows=pd.DataFrame([window], columns=list(WINDOW_COLUMNS)),
                target_users=window_target_users,
                candidates=window_candidates,
                user_profile=_optional_frame_for_window(user_profile, window),
                trend_predictions=trend_predictions,
                weights=weights,
                input_paths=input_paths,
                trend_model_source=trend_model_source,
            )
            if method.default_candidate_strategy is None:
                result = method.build_recommendations(context)
            else:
                if window_candidates is None:
                    raise FileNotFoundError(
                        f"{method_name} requires {method.default_candidate_strategy} "
                        "candidates"
                    )
                result = build_cached_recommendation_result_for_window(
                    method=method,
                    method_name=method_name,
                    strategy=method.default_candidate_strategy,
                    window=window,
                    target_users=window_target_users,
                    candidates=window_candidates,
                    context=context,
                    weights=dict(
                        weights if weights is not None else method.default_weights
                    ),
                    backfill_mode=BACKFILL_MODE_BY_METHOD.get(method_name),
                )
            writer.write_chunk(result)
            if collect_result:
                recommendation_chunks.append(result.recommendations)
                item_chunks.append(result.recommendation_items)
            _merge_window_metadata(metadata, window, result, context)
            params = result.params
        writer.publish()
        _refresh_metadata_inputs(metadata, input_paths)
        _refresh_metadata_row_counts(metadata)
        write_json_atomic(params, writer.output_paths.params)
        write_json_atomic(metadata, writer.output_paths.metadata)

    return RecommendationResult(
        recommendations=_concat_chunks(recommendation_chunks),
        recommendation_items=_concat_chunks(item_chunks),
        params=params,
        metadata=metadata,
    )


def _params_for_method(
    method,
    method_name: str,
    exclude_seen: bool,
    weights: dict[str, float] | None,
) -> dict[str, object]:
    return {
        "method": method_name,
        "method_type": method.method_type,
        "top_k": RECOMMENDATION_TOP_K,
        "exclude_seen": exclude_seen,
        "weights": dict(weights if weights is not None else method.default_weights),
        "candidate_strategy": method.default_candidate_strategy,
        "score_features": list(method.required_features),
    }


def _base_metadata(
    method_name: str,
    required_features,
    candidate_strategy: str | None,
    exclude_seen: bool,
    weights: dict[str, float] | None,
    windows: pd.DataFrame,
    input_paths: dict[str, str] | None,
    trend_model_source: str | None,
) -> dict[str, object]:
    method = get_recommendation_method(method_name)
    effective_weights = dict(weights if weights is not None else method.default_weights)
    metadata: dict[str, object] = build_artifact_metadata(
        name=f"recommendation_method:{method_name}",
        input_artifacts=dict(input_paths or {}),
        output_artifacts=_method_output_artifacts(method_name),
        schema_version=1,
        algorithm_version="recommendation-method-v1",
        config={
            "method": method_name,
            "top_k": RECOMMENDATION_TOP_K,
            "candidate_strategy": candidate_strategy,
            "exclude_seen": exclude_seen,
            "weights": effective_weights,
        },
        row_counts={},
    )
    metadata.update(
        {
            "method": method_name,
            "candidate_strategy": candidate_strategy,
            "backfill_mode": BACKFILL_MODE_BY_METHOD.get(method_name),
            "generated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "window_config": _window_config(windows),
            **_source_level_seen_policy_metadata(candidate_strategy),
        }
    )
    metadata.update(
        {
            "used_feature_artifacts": [],
        }
    )
    if "trend_score" in required_features:
        metadata["trend_score_config"] = {
            "stable_trend_model_source": trend_model_source,
            "core_attr_types": list(RECOMMENDATION_CORE_ATTR_TYPES),
            "attr_weights": dict(RECOMMENDATION_TREND_ATTR_WEIGHTS),
        }
    return metadata


def _method_output_artifacts(method_name: str) -> dict[str, str]:
    output_paths = method_output_paths(method_name)
    return {
        "recommendations": str(output_paths.recommendations),
        "recommendation_items": str(output_paths.recommendation_items),
        "params": str(output_paths.params),
        "metadata": str(output_paths.metadata),
    }


def _candidate_input_path(strategy: str, available_paths: dict[str, str]) -> str:
    strategy_key = f"{strategy}_candidates"
    if strategy_key in available_paths:
        return available_paths[strategy_key]
    return available_paths["candidate_items"]


def _candidate_metadata_input_path(
    strategy: str,
    candidate_items: str,
    available_paths: dict[str, str],
) -> str:
    strategy_key = f"{strategy}_candidate_metadata"
    if strategy_key in available_paths:
        return available_paths[strategy_key]
    if "candidate_metadata" in available_paths:
        return available_paths["candidate_metadata"]
    return str(Path(candidate_items).with_name("metadata.json"))


def _validate_cached_method_context(
    method: RecommendationMethod,
    method_name: str,
    context: RecommendationContext,
) -> None:
    if "trend_score" in method.required_features and (
        context.trend_predictions is None
        or context.trend_predictions.empty
        or not context.input_paths
        or "trend_predictions" not in context.input_paths
    ):
        raise FileNotFoundError(
            f"{method_name} requires trend predictions for cached trend scores"
        )


def _validate_feature_cache_partition_fresh(
    *,
    feature_name: str,
    strategy: str,
    window: dict[str, object],
    partition_path: Path,
    metadata_path: Path,
    current_input_paths: dict[str, str],
) -> None:
    metadata = _read_feature_cache_metadata(metadata_path, feature_name)
    input_artifacts = _feature_cache_input_artifacts(
        metadata,
        feature_name,
        current_input_paths,
    )
    assert_fresh_metadata(
        metadata_path=metadata_path,
        expected_input_artifacts=input_artifacts,
        expected_output_artifacts={
            "partition": str(partition_path),
            "partition_metadata": str(metadata_path),
        },
        expected_schema_version=FEATURE_CACHE_SCHEMA_VERSION,
        expected_algorithm_version=FEATURE_CACHE_ALGORITHM_VERSION,
        expected_config={
            "feature_name": feature_name,
            "strategy": strategy,
            "split": str(window["split"]),
            "cutoff_week": int(window["cutoff_week"]),
            "label_week": int(window["label_week"]),
        },
        stale_message=lambda reason: (
            f"{feature_name} feature cache partition is stale: {reason}. "
            "Rebuild feature cache before reranking."
        ),
    )


def _read_feature_cache_metadata(
    metadata_path: Path,
    feature_name: str,
) -> dict[str, object]:
    if not metadata_path.exists():
        raise RuntimeError(
            f"{feature_name} feature cache partition is stale: metadata.json is "
            "missing. Rebuild feature cache before reranking."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{feature_name} feature cache partition is stale: metadata is invalid. "
            "Rebuild feature cache before reranking."
        ) from error
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"{feature_name} feature cache partition is stale: metadata is invalid. "
            "Rebuild feature cache before reranking."
        )
    return metadata


def _feature_cache_input_artifacts(
    metadata: dict[str, object],
    feature_name: str,
    current_input_paths: dict[str, str],
) -> dict[str, str]:
    stored = metadata.get("input_artifacts")
    if not isinstance(stored, dict):
        raise RuntimeError(
            f"{feature_name} feature cache partition is stale: input_artifacts is "
            "missing. Rebuild feature cache before reranking."
        )
    input_artifacts = {str(key): str(value) for key, value in stored.items()}
    _require_feature_cache_input_keys(
        feature_name,
        input_artifacts,
        current_input_paths,
    )
    for key, current_value in current_input_paths.items():
        if key in input_artifacts and input_artifacts[key] != current_value:
            raise RuntimeError(
                f"{feature_name} feature cache partition is stale: "
                f"input_artifacts changed: {key}. "
                "Rebuild feature cache before reranking."
            )
    if feature_name == "trend_scores" and input_artifacts.get(
        "trend_predictions"
    ) != current_input_paths.get("trend_predictions"):
        raise RuntimeError(
            "trend_scores feature cache partition is stale: trend_predictions "
            "changed. Rebuild feature cache before reranking."
        )
    return input_artifacts


def _require_feature_cache_input_keys(
    feature_name: str,
    input_artifacts: dict[str, str],
    current_input_paths: dict[str, str],
) -> None:
    required_keys = FEATURE_REQUIRED_INPUT_KEYS.get(feature_name, ())
    missing = [
        key
        for key in required_keys
        if key in current_input_paths and key not in input_artifacts
    ]
    if missing:
        raise RuntimeError(
            f"{feature_name} feature cache partition is stale: "
            f"input_artifacts missing: {missing}. "
            "Rebuild feature cache before reranking."
        )


def _window_config(windows: pd.DataFrame) -> dict[str, object]:
    if windows.empty:
        return {
            "window_count": 0,
            "splits": [],
            "min_cutoff_week": None,
            "max_cutoff_week": None,
            "min_label_week": None,
            "max_label_week": None,
        }
    return {
        "window_count": int(len(windows)),
        "splits": sorted(windows["split"].astype(str).unique().tolist()),
        "min_cutoff_week": int(windows["cutoff_week"].min()),
        "max_cutoff_week": int(windows["cutoff_week"].max()),
        "min_label_week": int(windows["label_week"].min()),
        "max_label_week": int(windows["label_week"].max()),
    }


def _merge_window_metadata(
    metadata: dict[str, object],
    window: dict[str, object],
    result: RecommendationResult,
    context: RecommendationContext,
) -> None:
    candidate_rows = int(result.metadata.get("candidate_rows", 0))
    summary = {
        **window,
        "target_user_rows": int(len(context.target_users)),
        "candidate_rows": candidate_rows,
        "recommendation_rows": int(len(result.recommendations)),
        "recommendation_item_rows": int(len(result.recommendation_items)),
    }
    used_feature_artifacts = _dedupe_strings(
        result.metadata.get("used_feature_artifacts", [])
    )
    if used_feature_artifacts:
        summary["used_feature_artifacts"] = used_feature_artifacts
        metadata["used_feature_artifacts"] = _dedupe_strings(
            [
                *metadata.get("used_feature_artifacts", []),
                *used_feature_artifacts,
            ]
        )
    for key in (
        "underfilled_user_count",
        "backfilled_user_count",
        "still_underfilled_user_count",
    ):
        if key in result.metadata:
            summary[key] = int(result.metadata[key])
            metadata[key] = int(metadata.get(key, 0)) + int(result.metadata[key])
    for key in ("backfill_mode", "candidate_strategy", "source_level_seen_policy"):
        if key in result.metadata:
            summary[key] = result.metadata[key]
            metadata[key] = result.metadata[key]
    metadata["candidate_rows"] = int(metadata["candidate_rows"]) + candidate_rows
    metadata["recommendation_rows"] = int(metadata["recommendation_rows"]) + int(
        len(result.recommendations)
    )
    metadata["recommendation_item_rows"] = int(
        metadata["recommendation_item_rows"]
    ) + int(len(result.recommendation_items))
    metadata["window_count"] = int(metadata["window_count"]) + 1
    metadata["window_summaries"].append(summary)


def _refresh_metadata_inputs(
    metadata: dict[str, object],
    input_paths: dict[str, str] | None,
) -> None:
    base_input_paths = dict(input_paths or {})
    used_feature_artifacts = _dedupe_strings(metadata.get("used_feature_artifacts", []))
    metadata["used_feature_artifacts"] = used_feature_artifacts
    metadata["input_artifacts"] = method_input_artifacts(
        base_input_paths={
            key: value
            for key, value in base_input_paths.items()
            if key
            not in {
                "candidate_items",
                "candidate_metadata",
                "feature_cache_metadata",
            }
        },
        candidate_items=base_input_paths.get("candidate_items"),
        candidate_metadata=base_input_paths.get("candidate_metadata"),
        feature_cache_metadata=_feature_cache_metadata_input(
            base_input_paths,
            include_default=bool(used_feature_artifacts),
        ),
        feature_partitions=used_feature_artifacts,
    )
    metadata["input_fingerprints"] = build_input_fingerprints(
        metadata["input_artifacts"]
    )


def _refresh_metadata_row_counts(metadata: dict[str, object]) -> None:
    metadata["row_counts"] = {
        "candidate_rows": int(metadata["candidate_rows"]),
        "recommendation_rows": int(metadata["recommendation_rows"]),
        "recommendation_item_rows": int(metadata["recommendation_item_rows"]),
        "window_count": int(metadata["window_count"]),
    }


def _feature_cache_metadata_input(
    base_input_paths: dict[str, str],
    *,
    include_default: bool,
) -> str | None:
    if "feature_cache_metadata" in base_input_paths:
        return base_input_paths["feature_cache_metadata"]
    if include_default and FEATURE_CACHE_METADATA_PATH.exists():
        return str(FEATURE_CACHE_METADATA_PATH)
    return None


def _dedupe_strings(values: object) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return result
    for value in values:
        item = str(value)
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _source_level_seen_policy_metadata(
    candidate_strategy: str | None,
) -> dict[str, str]:
    if candidate_strategy == "enhanced_default":
        return {"source_level_seen_policy": "reorder_only"}
    return {}


def _underfilled_user_count(
    target_users: pd.DataFrame,
    items: pd.DataFrame,
    top_k: int,
) -> int:
    counts = _recommendation_counts_by_target(target_users, items)
    return int((counts["_count"] < top_k).sum())


def _recommendation_counts_by_target(
    target_users: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["split", "cutoff_week", "label_week", "customer_id"]
    targets = target_users.loc[:, key_columns].drop_duplicates().copy()
    if items.empty:
        targets["_count"] = 0
        return targets
    counts = (
        items.groupby(key_columns, as_index=False)
        .size()
        .rename(columns={"size": "_count"})
    )
    result = targets.merge(counts, on=key_columns, how="left")
    result["_count"] = result["_count"].fillna(0).astype(int)
    return result


def _frame_for_window(frame: pd.DataFrame, window: dict[str, object]) -> pd.DataFrame:
    mask = (
        (frame["split"] == window["split"])
        & (frame["cutoff_week"] == window["cutoff_week"])
        & (frame["label_week"] == window["label_week"])
    )
    return frame.loc[mask].reset_index(drop=True)


def _optional_frame_for_window(
    frame: pd.DataFrame | None,
    window: dict[str, object],
) -> pd.DataFrame | None:
    if frame is None:
        return None
    return _frame_for_window(frame, window)


def _concat_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_chunks = [chunk for chunk in chunks if not chunk.empty]
    if not non_empty_chunks:
        return pd.DataFrame()
    return pd.concat(non_empty_chunks, ignore_index=True)
