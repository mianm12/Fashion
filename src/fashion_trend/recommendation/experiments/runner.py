from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.contracts import (
    RECOMMENDATION_CANDIDATES_PER_SOURCE,
    RECOMMENDATION_PROFILE_TOP_ATTRIBUTES,
    RECOMMENDATION_TOP_K,
)
from fashion_trend.recommendation.evaluation.metrics import evaluate_recommendations
from fashion_trend.recommendation.evaluation.runner import (
    build_recommendable_pool_for_windows,
    run_recommendation_evaluation,
)
from fashion_trend.recommendation.experiments.ablation import build_ablation_summary
from fashion_trend.recommendation.experiments.grid_search import (
    iter_weight_grid,
    select_best_weights,
)
from fashion_trend.recommendation.features.cache import (
    build_and_write_feature_cache_for_strategy,
)
from fashion_trend.recommendation.freshness import (
    assert_fresh_metadata,
)
from fashion_trend.recommendation.inputs import (
    RecommendationInputArtifacts,
    build_and_write_recommendation_inputs,
)
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.paths import (
    EVALUATION_LABELS_PATH,
    FEATURE_CACHE_METADATA_PATH,
    RECOMMEND_METADATA_PATH,
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    USER_PROFILE_PATH,
    candidate_items_path,
    experiment_dir,
    feature_cache_partition_metadata_path,
    feature_cache_partition_path,
    method_output_paths,
)
from fashion_trend.recommendation.readers import (
    read_candidate_items,
    read_evaluation_labels,
    read_recommendations,
    read_target_users,
    read_time_windows,
    read_user_profile,
)
from fashion_trend.recommendation.registry import get_recommendation_method
from fashion_trend.recommendation.retrieval.candidates import (
    build_and_write_candidate_items,
    candidate_input_paths_for_strategy,
)
from fashion_trend.recommendation.runner import (
    build_cached_recommendation_result_for_window,
    feature_artifact_paths_for_method_window,
    method_input_paths_for_artifacts,
    run_recommendation_method_by_window,
)

BASELINE_METHODS = (
    "global_popularity",
    "recent_popularity",
    "attribute_similarity",
    "pop_similarity",
)
TREND_METHOD = "pop_similarity_trend"


@dataclass(frozen=True)
class RecommendationExperimentContext:
    transactions: pd.DataFrame
    article_attributes: pd.DataFrame
    trend_predictions: pd.DataFrame
    input_paths: dict[str, str] | None = None
    trend_model_source: str | None = None


def generate_experiment_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{secrets.token_hex(4)}"
    validate_safe_path_segment(run_id, "experiment run_id")
    return run_id


def candidate_strategy_for_method(method_name: str) -> str | None:
    return get_recommendation_method(method_name).default_candidate_strategy


def ensure_or_build_recommendation_inputs(
    context: RecommendationExperimentContext,
    force: bool = False,
) -> RecommendationInputArtifacts:
    if not force and _recommendation_inputs_are_fresh(context):
        return RecommendationInputArtifacts(
            time_windows=read_time_windows(TIME_WINDOWS_PATH),
            target_users=read_target_users(TARGET_USERS_PATH),
            evaluation_labels=read_evaluation_labels(EVALUATION_LABELS_PATH),
            user_profile=read_user_profile(USER_PROFILE_PATH),
        )

    return build_and_write_recommendation_inputs(
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        trend_predictions=context.trend_predictions,
        input_paths=_upstream_input_paths(context),
    )


def ensure_or_build_candidate_items(
    strategy: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
) -> pd.DataFrame:
    path = candidate_items_path(strategy)
    input_paths = candidate_input_paths_for_strategy(
        strategy,
        _experiment_input_paths(context),
    )
    if not force and path.exists():
        _validate_candidate_items_fresh(strategy, input_paths)
        return read_candidate_items(path)

    build_and_write_candidate_items(
        strategy=strategy,
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        trend_predictions=context.trend_predictions,
        windows=inputs.time_windows,
        target_users=inputs.target_users,
        user_profile=inputs.user_profile,
        input_paths=input_paths,
    )
    return read_candidate_items(path)


def ensure_or_build_candidates_for_method(
    method_name: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
) -> pd.DataFrame | None:
    strategy = candidate_strategy_for_method(method_name)
    if strategy is None:
        return None
    return ensure_or_build_candidate_items(strategy, context, inputs, force=force)


def ensure_or_build_feature_cache_for_strategy(
    strategy: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
) -> None:
    input_paths = _feature_cache_input_paths(strategy, context)
    if not force and _feature_cache_partitions_exist(strategy, candidates):
        return
    if not _feature_cache_manifest_can_merge():
        write_json_atomic({"entries": {}}, FEATURE_CACHE_METADATA_PATH)
    build_and_write_feature_cache_for_strategy(
        strategy=strategy,
        candidates=candidates,
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        user_profile=inputs.user_profile,
        trend_predictions=context.trend_predictions,
        input_paths=input_paths,
    )


def run_baseline_methods(
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    available_paths = _experiment_input_paths(context)
    for method_name in BASELINE_METHODS:
        input_paths = _method_input_paths(method_name, available_paths)
        method = get_recommendation_method(method_name)
        if not force and method_output_paths(method_name).recommendations.exists():
            _validate_method_output_fresh(method_name, input_paths)
            payloads.append(
                evaluate_method_output_for_experiment(method_name, context, inputs)
            )
            continue
        candidates = ensure_or_build_candidates_for_method(
            method_name,
            context,
            inputs,
            force=force,
        )
        if candidates is not None:
            ensure_or_build_feature_cache_for_strategy(
                str(method.default_candidate_strategy),
                context,
                inputs,
                candidates,
                force=force,
            )
        run_recommendation_method_by_window(
            method_name=method_name,
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=inputs.time_windows,
            target_users=inputs.target_users,
            candidates=candidates,
            user_profile=(
                inputs.user_profile if "sim_score" in method.required_features else None
            ),
            trend_predictions=None,
            collect_result=False,
            input_paths=input_paths,
        )
        payloads.append(
            evaluate_method_output_for_experiment(method_name, context, inputs)
        )
    return payloads


def evaluate_weight_grid_on_valid(
    weight_grid: list[dict[str, float]],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> list[dict[str, Any]]:
    ensure_or_build_feature_cache_for_strategy(
        "default",
        context,
        inputs,
        candidates,
        force=False,
    )
    return [
        evaluate_one_weight_run_on_valid(
            grid_index=grid_index,
            weights=weights,
            context=context,
            inputs=inputs,
            candidates=candidates,
        )
        for grid_index, weights in enumerate(weight_grid)
    ]


def evaluate_one_weight_run_on_valid(
    grid_index: int,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    result = build_recommendation_result_in_memory(
        method_name=TREND_METHOD,
        weights=weights,
        split_filter="valid",
        context=context,
        inputs=inputs,
        candidates=candidates,
    )
    target_users = _filter_split(inputs.target_users, "valid")
    labels = _filter_split(inputs.evaluation_labels, "valid")
    recommendable_pool = _filter_split(
        build_recommendable_pool_for_windows(
            context.transactions,
            inputs.time_windows,
        ),
        "valid",
    )
    metrics = evaluate_recommendations(
        result.recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=RECOMMENDATION_TOP_K,
        strict_missing_users=False,
    )
    return {
        "run_id": generate_experiment_run_id(),
        "grid_index": grid_index,
        "weights": dict(weights),
        "valid_metrics": metrics["valid"],
    }


def build_recommendation_result_in_memory(
    method_name: str,
    weights: dict[str, float],
    split_filter: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> RecommendationResult:
    method = get_recommendation_method(method_name)
    windows = _filter_split(inputs.time_windows, split_filter)
    target_users = _filter_split(inputs.target_users, split_filter)
    filtered_candidates = _filter_split(candidates, split_filter)
    user_profile = _filter_split(inputs.user_profile, split_filter)
    input_paths = _method_input_paths(method_name, _experiment_input_paths(context))
    recommendation_chunks: list[pd.DataFrame] = []
    item_chunks: list[pd.DataFrame] = []
    metadata: dict[str, object] = {}
    for window in windows.loc[:, ["split", "cutoff_week", "label_week"]].to_dict(
        "records"
    ):
        window_candidates = _frame_for_window(filtered_candidates, window)
        window_context = RecommendationContext(
            method=method_name,
            top_k=RECOMMENDATION_TOP_K,
            exclude_seen=True,
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=pd.DataFrame(
                [window], columns=["split", "cutoff_week", "label_week"]
            ),
            target_users=_frame_for_window(target_users, window),
            candidates=window_candidates,
            user_profile=_frame_for_window(user_profile, window),
            trend_predictions=context.trend_predictions,
            weights=weights,
            input_paths=input_paths,
            trend_model_source=context.trend_model_source,
        )
        result = build_cached_recommendation_result_for_window(
            method=method,
            method_name=method_name,
            strategy=str(method.default_candidate_strategy),
            window=window,
            target_users=window_context.target_users,
            candidates=window_candidates,
            context=window_context,
            weights=dict(weights),
            backfill_mode="recent",
        )
        recommendation_chunks.append(result.recommendations)
        item_chunks.append(result.recommendation_items)
        metadata = result.metadata
    return RecommendationResult(
        recommendations=_concat_chunks(recommendation_chunks),
        recommendation_items=_concat_chunks(item_chunks),
        params={
            "method": method_name,
            "method_type": method.method_type,
            "top_k": RECOMMENDATION_TOP_K,
            "exclude_seen": True,
            "weights": dict(weights),
            "candidate_strategy": method.default_candidate_strategy,
            "score_features": list(method.required_features),
        },
        metadata=metadata,
    )


def publish_trend_method_with_weights(
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    ensure_or_build_feature_cache_for_strategy(
        "default",
        context,
        inputs,
        candidates,
        force=False,
    )
    input_paths = _method_input_paths(TREND_METHOD, _experiment_input_paths(context))
    run_recommendation_method_by_window(
        method_name=TREND_METHOD,
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        windows=inputs.time_windows,
        target_users=inputs.target_users,
        candidates=candidates,
        user_profile=inputs.user_profile,
        trend_predictions=context.trend_predictions,
        weights=weights,
        collect_result=False,
        input_paths=input_paths,
        trend_model_source=context.trend_model_source,
    )
    return evaluate_method_output_for_experiment(TREND_METHOD, context, inputs)


def evaluate_result_for_experiment(
    method: str,
    result: RecommendationResult,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
) -> dict[str, Any]:
    return run_recommendation_evaluation(
        method=method,
        recommendations=result.recommendations,
        target_users=inputs.target_users,
        labels=inputs.evaluation_labels,
        recommendable_pool=build_recommendable_pool_for_windows(
            context.transactions,
            inputs.time_windows,
        ),
        input_paths={"experiment": "in_memory"},
        strict_missing_users=False,
    )


def evaluate_method_output_for_experiment(
    method: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
) -> dict[str, Any]:
    return run_recommendation_evaluation(
        method=method,
        recommendations=read_recommendations(
            method_output_paths(method).recommendations
        ),
        target_users=inputs.target_users,
        labels=inputs.evaluation_labels,
        recommendable_pool=build_recommendable_pool_for_windows(
            context.transactions,
            inputs.time_windows,
        ),
        input_paths={"experiment": "in_memory"},
        strict_missing_users=False,
    )


def build_experiment_payload(
    experiment_id: str,
    baseline_payloads: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    trend_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "experiment_path": str(experiment_dir(experiment_id) / "experiment.json"),
        "best_weights": select_best_weights(search_results),
        "search_results": search_results,
        "ablation": build_ablation_summary([*baseline_payloads, trend_payload]),
    }


def run_recommendation_experiment(
    context: RecommendationExperimentContext,
    experiment_id: str = "main",
    force: bool = False,
) -> dict[str, Any]:
    validate_safe_path_segment(experiment_id, "experiment_id")
    inputs = ensure_or_build_recommendation_inputs(context, force=force)
    baseline_payloads = run_baseline_methods(context, inputs, force=force)
    default_candidates = ensure_or_build_candidates_for_method(
        TREND_METHOD,
        context,
        inputs,
        force=force,
    )
    if default_candidates is None:
        raise ValueError(f"{TREND_METHOD} requires a candidate strategy")
    ensure_or_build_feature_cache_for_strategy(
        "default",
        context,
        inputs,
        default_candidates,
        force=force,
    )

    search_results = evaluate_weight_grid_on_valid(
        iter_weight_grid(),
        context,
        inputs,
        default_candidates,
    )
    best_weights = select_best_weights(search_results)
    trend_payload = publish_trend_method_with_weights(
        best_weights,
        context,
        inputs,
        default_candidates,
    )
    payload = build_experiment_payload(
        experiment_id,
        baseline_payloads,
        search_results,
        trend_payload,
    )
    write_json_atomic(payload, experiment_dir(experiment_id) / "experiment.json")
    return payload


def _filter_split(dataframe: pd.DataFrame, split: str) -> pd.DataFrame:
    return dataframe.loc[dataframe["split"].astype(str) == split].reset_index(drop=True)


def _feature_cache_input_paths(
    strategy: str,
    context: RecommendationExperimentContext,
) -> dict[str, str]:
    available_paths = _experiment_input_paths(context)
    return {
        **candidate_input_paths_for_strategy(strategy, available_paths),
        "candidate_items": str(candidate_items_path(strategy)),
        "candidate_metadata": str(
            candidate_items_path(strategy).with_name("metadata.json")
        ),
    }


def _feature_cache_partitions_exist(
    strategy: str,
    candidates: pd.DataFrame,
) -> bool:
    if not _feature_cache_manifest_has_strategy(strategy):
        return False
    if candidates.empty:
        return True
    windows = candidates.loc[
        :, ["split", "cutoff_week", "label_week"]
    ].drop_duplicates()
    for window in windows.to_dict("records"):
        for feature_name in (
            "candidate_seen_flags",
            "popularity_scores",
            "recent_scores",
            "similarity_scores",
            "trend_scores",
        ):
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
            if not partition.exists() or not metadata.exists():
                return False
    return True


def _feature_cache_manifest_has_strategy(strategy: str) -> bool:
    manifest = _read_feature_cache_manifest()
    if manifest is None:
        return False
    entries = manifest.get("entries")
    return isinstance(entries, dict) and f"strategy:{strategy}" in entries


def _feature_cache_manifest_can_merge() -> bool:
    manifest = _read_feature_cache_manifest()
    return isinstance(manifest, dict) and isinstance(manifest.get("entries"), dict)


def _read_feature_cache_manifest() -> dict[str, object] | None:
    if not FEATURE_CACHE_METADATA_PATH.exists():
        return {"entries": {}}
    try:
        manifest = json.loads(FEATURE_CACHE_METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(manifest, dict):
        return None
    return manifest


def _frame_for_window(frame: pd.DataFrame, window: dict[str, object]) -> pd.DataFrame:
    mask = (
        (frame["split"] == window["split"])
        & (frame["cutoff_week"] == window["cutoff_week"])
        & (frame["label_week"] == window["label_week"])
    )
    return frame.loc[mask].reset_index(drop=True)


def _concat_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_chunks = [chunk for chunk in chunks if not chunk.empty]
    if not non_empty_chunks:
        return pd.DataFrame()
    return pd.concat(non_empty_chunks, ignore_index=True)


def _validate_method_output_fresh(
    method_name: str,
    input_paths: dict[str, str],
) -> None:
    output_paths = method_output_paths(method_name)
    for path in (
        output_paths.recommendations,
        output_paths.recommendation_items,
        output_paths.params,
        output_paths.metadata,
    ):
        if not path.exists():
            raise RuntimeError(
                _stale_output_message(method_name, f"{path.name} is missing")
            )
    assert_fresh_metadata(
        metadata_path=output_paths.metadata,
        expected_input_artifacts=_expected_method_input_artifacts(
            method_name,
            input_paths,
            output_paths.metadata,
        ),
        expected_output_artifacts=_method_output_artifacts(method_name),
        expected_schema_version=1,
        expected_algorithm_version="recommendation-method-v1",
        expected_config=_method_freshness_config(method_name, exclude_seen=True),
        stale_message=lambda reason: _stale_output_message(method_name, reason),
    )


def _recommendation_inputs_are_fresh(
    context: RecommendationExperimentContext,
) -> bool:
    if not all(
        path.exists()
        for path in (
            TIME_WINDOWS_PATH,
            TARGET_USERS_PATH,
            EVALUATION_LABELS_PATH,
            USER_PROFILE_PATH,
            RECOMMEND_METADATA_PATH,
        )
    ):
        return False
    try:
        assert_fresh_metadata(
            metadata_path=RECOMMEND_METADATA_PATH,
            expected_input_artifacts=_upstream_input_paths(context),
            expected_output_artifacts=_recommendation_input_output_artifacts(),
            expected_schema_version=1,
            expected_algorithm_version="recommendation-inputs-v1",
            expected_config={
                "profile_top_attributes": RECOMMENDATION_PROFILE_TOP_ATTRIBUTES,
            },
            stale_message=lambda reason: f"recommendation inputs are stale: {reason}",
        )
    except RuntimeError:
        return False
    return True


def _validate_candidate_items_fresh(
    strategy: str,
    input_paths: dict[str, str],
) -> None:
    candidate_path = candidate_items_path(strategy)
    metadata_path = candidate_path.with_name("metadata.json")
    assert_fresh_metadata(
        metadata_path=metadata_path,
        expected_input_artifacts=dict(input_paths),
        expected_output_artifacts={"candidate_items": str(candidate_path)},
        expected_schema_version=1,
        expected_algorithm_version="recommendation-candidates-v1",
        expected_config={
            "strategy": strategy,
            "candidates_per_source": RECOMMENDATION_CANDIDATES_PER_SOURCE,
        },
        stale_message=lambda reason: _stale_candidate_message(strategy, reason),
    )


def _stale_output_message(method_name: str, reason: str) -> str:
    return (
        f"{method_name} recommendation output is stale: {reason}. "
        "Run src/16_run_recommendation_experiment.py with --force to rebuild."
    )


def _stale_candidate_message(strategy: str, reason: str) -> str:
    return (
        f"{strategy} candidate_items output is stale: {reason}. "
        "Run src/16_run_recommendation_experiment.py with --force to rebuild."
    )


def _experiment_input_paths(
    context: RecommendationExperimentContext,
) -> dict[str, str]:
    return {
        **dict(context.input_paths or {}),
        "time_windows": str(TIME_WINDOWS_PATH),
        "target_users": str(TARGET_USERS_PATH),
        "evaluation_labels": str(EVALUATION_LABELS_PATH),
        "user_profile": str(USER_PROFILE_PATH),
        "recommendation_inputs": str(RECOMMEND_METADATA_PATH),
        "default_candidates": str(candidate_items_path("default")),
        "default_candidate_metadata": str(
            candidate_items_path("default").with_name("metadata.json")
        ),
        "similarity_candidates": str(candidate_items_path("similarity")),
        "similarity_candidate_metadata": str(
            candidate_items_path("similarity").with_name("metadata.json")
        ),
        "feature_cache_metadata": str(FEATURE_CACHE_METADATA_PATH),
    }


def _method_input_paths(
    method_name: str,
    available_paths: dict[str, str],
) -> dict[str, str]:
    return method_input_paths_for_artifacts(method_name, available_paths)


def _expected_method_input_artifacts(
    method_name: str,
    current_input_paths: dict[str, str],
    metadata_path,
) -> dict[str, str]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            _stale_output_message(method_name, "metadata is invalid")
        ) from error
    if not isinstance(metadata, dict):
        raise RuntimeError(_stale_output_message(method_name, "metadata is invalid"))
    stored_input_artifacts = metadata.get("input_artifacts")
    if not isinstance(stored_input_artifacts, dict):
        return dict(current_input_paths)

    stored = {str(key): str(value) for key, value in stored_input_artifacts.items()}
    for key, value in current_input_paths.items():
        if stored.get(key) != value:
            raise RuntimeError(
                _stale_output_message(method_name, f"input_artifacts changed: {key}")
            )

    method = get_recommendation_method(method_name)
    if method.default_candidate_strategy is not None:
        _require_cached_method_input_artifacts(method_name, stored, metadata)
    return stored


def _require_cached_method_input_artifacts(
    method_name: str,
    input_artifacts: dict[str, str],
    metadata: dict[str, object],
) -> None:
    required_keys = ("candidate_items", "candidate_metadata", "feature_cache_metadata")
    for key in required_keys:
        if key not in input_artifacts:
            raise RuntimeError(
                _stale_output_message(method_name, f"input_artifacts missing: {key}")
            )
    required_feature_artifacts = _required_feature_artifacts_for_method_metadata(
        method_name,
        metadata,
    )
    stored_paths = set(input_artifacts.values())
    missing_paths = [
        path for path in required_feature_artifacts if path not in stored_paths
    ]
    if missing_paths:
        raise RuntimeError(
            _stale_output_message(
                method_name,
                f"input_artifacts missing feature partitions: {missing_paths[:3]}",
            )
        )


def _required_feature_artifacts_for_method_metadata(
    method_name: str,
    metadata: dict[str, object],
) -> list[str]:
    method = get_recommendation_method(method_name)
    strategy = method.default_candidate_strategy
    if strategy is None:
        return []
    config = metadata.get("config")
    exclude_seen = True
    if isinstance(config, dict):
        exclude_seen = bool(config.get("exclude_seen", True))
    artifacts: list[str] = []
    for window in _metadata_windows(method_name, metadata):
        if int(window.get("candidate_rows", 0)) <= 0:
            continue
        artifacts.extend(
            feature_artifact_paths_for_method_window(
                method_name=method_name,
                strategy=str(strategy),
                window=window,
                include_seen=exclude_seen,
            )
        )
    return artifacts


def _metadata_windows(
    method_name: str,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    summaries = metadata.get("window_summaries")
    if not isinstance(summaries, list):
        raise RuntimeError(
            _stale_output_message(method_name, "window_summaries missing")
        )
    windows: list[dict[str, object]] = []
    required_keys = {
        "split",
        "cutoff_week",
        "label_week",
        "candidate_rows",
    }
    for index, summary in enumerate(summaries):
        if not isinstance(summary, dict):
            raise RuntimeError(
                _stale_output_message(
                    method_name,
                    f"window_summaries[{index}] is invalid",
                )
            )
        missing = sorted(required_keys - set(summary))
        if missing:
            raise RuntimeError(
                _stale_output_message(
                    method_name,
                    f"window_summaries[{index}] missing: {missing}",
                )
            )
        windows.append(summary)
    return windows


def _upstream_input_paths(context: RecommendationExperimentContext) -> dict[str, str]:
    return {
        key: value
        for key, value in dict(context.input_paths or {}).items()
        if key in {"weekly_transactions", "article_attributes", "trend_predictions"}
    }


def _recommendation_input_output_artifacts() -> dict[str, str]:
    return {
        "time_windows": str(TIME_WINDOWS_PATH),
        "target_users": str(TARGET_USERS_PATH),
        "evaluation_labels": str(EVALUATION_LABELS_PATH),
        "user_profile": str(USER_PROFILE_PATH),
    }


def _method_output_artifacts(method_name: str) -> dict[str, str]:
    output_paths = method_output_paths(method_name)
    return {
        "recommendations": str(output_paths.recommendations),
        "recommendation_items": str(output_paths.recommendation_items),
        "params": str(output_paths.params),
        "metadata": str(output_paths.metadata),
    }


def _method_freshness_config(
    method_name: str,
    exclude_seen: bool,
    weights: dict[str, float] | None = None,
) -> dict[str, object]:
    method = get_recommendation_method(method_name)
    return {
        "method": method_name,
        "top_k": RECOMMENDATION_TOP_K,
        "candidate_strategy": method.default_candidate_strategy,
        "exclude_seen": exclude_seen,
        "weights": dict(weights if weights is not None else method.default_weights),
    }
