from __future__ import annotations

import json
import math
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.contracts import (
    CUSTOMER_AGE_BUCKETS,
    ENHANCED_CANDIDATE_SOURCE_CAPS,
    RECOMMENDATION_CANDIDATES_PER_SOURCE,
    RECOMMENDATION_PROFILE_TOP_ATTRIBUTES,
    RECOMMENDATION_TOP_K,
    SOURCE_ORDER,
)
from fashion_trend.recommendation.evaluation.metrics import evaluate_recommendations
from fashion_trend.recommendation.evaluation.runner import (
    run_recommendation_evaluation,
)
from fashion_trend.recommendation.experiments.ablation import (
    SCORE_FEATURES,
    STRICT_VARIANTS,
    build_ablation_summary,
    build_named_ablation_rows,
    derive_strict_ablation_weights,
    select_trend_bucket_representatives,
)
from fashion_trend.recommendation.experiments.grid_search import (
    iter_weight_grid,
    select_best_weights,
)
from fashion_trend.recommendation.features.cache import (
    build_and_write_feature_cache_for_strategy,
    read_recommendable_pool_cache,
    recommendable_pool_cache_fresh,
    write_recommendable_pool_cache,
)
from fashion_trend.recommendation.fingerprints import build_input_fingerprints
from fashion_trend.recommendation.freshness import (
    assert_fresh_metadata,
)
from fashion_trend.recommendation.inputs import (
    ARTICLE_PRODUCT_MAP_SCHEMA_VERSION,
    CUSTOMER_AGE_BUCKET_ALGORITHM_VERSION,
    CUSTOMER_PROFILE_SCHEMA_VERSION,
    RecommendationInputArtifacts,
    build_and_write_recommendation_inputs,
)
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.outputs import (
    build_recommendations_csv,
    format_recommendation_items,
)
from fashion_trend.recommendation.paths import (
    ARTICLE_PRODUCT_MAP_PATH,
    CUSTOMER_PROFILE_PATH,
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
from fashion_trend.recommendation.perf import StageTimer
from fashion_trend.recommendation.ranking.backfill import append_backfill_items
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items
from fashion_trend.recommendation.readers import (
    read_article_product_map,
    read_candidate_items,
    read_customer_profile,
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
    BACKFILL_MODE_BY_METHOD,
    build_cached_feature_frame_for_window,
    build_cached_recommendation_result_for_window,
    feature_artifact_paths_for_method_window,
    filter_cached_seen_items,
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
SEARCH_CACHE_SCHEMA_VERSION = 1
SEARCH_CACHE_ALGORITHM_VERSION = "recommendation-weight-search-v1"


@dataclass(frozen=True)
class RecommendationExperimentContext:
    transactions: pd.DataFrame
    article_attributes: pd.DataFrame
    trend_predictions: pd.DataFrame
    input_paths: dict[str, str] | None = None
    trend_model_source: str | None = None
    customers: pd.DataFrame | None = None
    clean_articles: pd.DataFrame | None = None


@dataclass(frozen=True)
class RebuildDecision:
    rebuild: bool
    reason: str


@dataclass(frozen=True)
class _PreparedRecommendationWindow:
    window: dict[str, object]
    target_users: pd.DataFrame
    candidates: pd.DataFrame
    feature_frame: pd.DataFrame


def should_rebuild_method(
    *,
    method_name: str,
    stale_reason: str | None,
    force_methods: Sequence[str],
    force_cache: bool,
    force_candidates: bool,
    force_rebuild_all: bool,
) -> RebuildDecision:
    if force_rebuild_all:
        return RebuildDecision(True, "force-rebuild-all")
    if method_name in force_methods:
        return RebuildDecision(True, "force-method")
    if force_candidates:
        return RebuildDecision(True, "force-candidates")
    if force_cache:
        return RebuildDecision(True, "force-cache")
    if stale_reason is not None:
        return RebuildDecision(True, stale_reason)
    return RebuildDecision(False, "fresh")


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
            customer_profile=(
                read_customer_profile(CUSTOMER_PROFILE_PATH)
                if CUSTOMER_PROFILE_PATH.exists()
                else None
            ),
            article_product_map=(
                read_article_product_map(ARTICLE_PRODUCT_MAP_PATH)
                if ARTICLE_PRODUCT_MAP_PATH.exists()
                else None
            ),
        )

    _require_optional_recommendation_input_frames(context)
    return build_and_write_recommendation_inputs(
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        trend_predictions=context.trend_predictions,
        input_paths=_upstream_input_paths(context),
        customers=context.customers,
        clean_articles=context.clean_articles,
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
        customer_profile=inputs.customer_profile,
        article_product_map=inputs.article_product_map,
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
        customer_profile=inputs.customer_profile,
        article_product_map=inputs.article_product_map,
        input_paths=input_paths,
    )


def ensure_or_build_recommendable_pool_cache(
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    *,
    force: bool,
) -> pd.DataFrame:
    input_artifacts = _recommendable_pool_input_artifacts(context)
    if force or not recommendable_pool_cache_fresh(
        inputs.time_windows,
        input_artifacts,
    ):
        if not _feature_cache_manifest_can_merge():
            write_json_atomic({"entries": {}}, FEATURE_CACHE_METADATA_PATH)
        write_recommendable_pool_cache(
            transactions=context.transactions,
            windows=inputs.time_windows,
            input_artifacts=input_artifacts,
        )
    return read_recommendable_pool_cache(inputs.time_windows)


def _recommendable_pool_input_artifacts(
    context: RecommendationExperimentContext,
) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            **_experiment_input_paths(context),
            "recommendation_inputs": str(RECOMMEND_METADATA_PATH),
        }.items()
        if key != "feature_cache_metadata"
    }


def run_baseline_methods(
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
    force_methods: Sequence[str] = (),
    force_cache: bool = False,
    force_candidates: bool = False,
    force_rebuild_all: bool = False,
    rebuild_stale_outputs: bool = False,
    stage_status: list[dict[str, Any]] | None = None,
    timings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    available_paths = _experiment_input_paths(context)
    for method_name in BASELINE_METHODS:
        timer = StageTimer("method", details={"method": method_name})
        input_paths = _method_input_paths(method_name, available_paths)
        method = get_recommendation_method(method_name)
        legacy_force = force or force_rebuild_all
        stale_reason = None
        output_exists = method_output_paths(method_name).recommendations.exists()
        if not legacy_force:
            stale_reason = _method_output_stale_reason(method_name, input_paths)
            if (
                stale_reason is not None
                and output_exists
                and not rebuild_stale_outputs
                and not (
                    force_cache or force_candidates or method_name in force_methods
                )
            ):
                raise RuntimeError(stale_reason)
        decision = should_rebuild_method(
            method_name=method_name,
            stale_reason=stale_reason,
            force_methods=force_methods,
            force_cache=force_cache,
            force_candidates=force_candidates,
            force_rebuild_all=legacy_force,
        )
        if not decision.rebuild:
            payloads.append(
                evaluate_method_output_for_experiment(
                    method_name,
                    context,
                    inputs,
                    force=force_cache or force_rebuild_all,
                )
            )
            _record_stage_status(
                stage_status,
                {
                    "stage": "method",
                    "method": method_name,
                    "status": "reused",
                    "reason": decision.reason,
                },
            )
            _record_stage_status(
                stage_status,
                {
                    "stage": "metrics",
                    "method": method_name,
                    "status": "rebuilt",
                    "reason": decision.reason,
                },
            )
            _record_timing(timings, timer.finish())
            continue
        strategy = method.default_candidate_strategy
        candidates = None
        if strategy is None:
            _record_stage_status(
                stage_status,
                {
                    "stage": "candidates",
                    "method": method_name,
                    "status": "skipped",
                    "reason": "no-candidate-strategy",
                },
            )
            _record_stage_status(
                stage_status,
                {
                    "stage": "cache",
                    "method": method_name,
                    "status": "skipped",
                    "reason": "no-candidate-strategy",
                },
            )
        else:
            strategy_name = str(strategy)
            candidate_force = legacy_force or force_candidates
            candidate_rebuild = _candidate_items_rebuild_required(
                strategy_name,
                context,
                force=candidate_force,
            )
            candidates = ensure_or_build_candidates_for_method(
                method_name,
                context,
                inputs,
                force=candidate_rebuild,
            )
            if candidates is None:
                raise ValueError(f"{method_name} requires a candidate strategy")
            _record_stage_status(
                stage_status,
                {
                    "stage": "candidates",
                    "method": method_name,
                    "strategy": strategy_name,
                    "status": "rebuilt" if candidate_rebuild else "reused",
                    "reason": _artifact_stage_reason(
                        rebuilt=candidate_rebuild,
                        forced=candidate_force,
                        force_reason=decision.reason,
                    ),
                },
            )
            cache_force = legacy_force or force_cache or force_candidates
            cache_rebuild = cache_force or not _feature_cache_partitions_exist(
                strategy_name,
                candidates,
            )
            ensure_or_build_feature_cache_for_strategy(
                strategy_name,
                context,
                inputs,
                candidates,
                force=cache_rebuild,
            )
            _record_stage_status(
                stage_status,
                {
                    "stage": "cache",
                    "method": method_name,
                    "strategy": strategy_name,
                    "status": "rebuilt" if cache_rebuild else "reused",
                    "reason": _artifact_stage_reason(
                        rebuilt=cache_rebuild,
                        forced=cache_force,
                        force_reason=decision.reason,
                    ),
                },
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
            evaluate_method_output_for_experiment(
                method_name,
                context,
                inputs,
                force=force_cache or force_rebuild_all,
            )
        )
        _record_stage_status(
            stage_status,
            {
                "stage": "metrics",
                "method": method_name,
                "status": "rebuilt",
                "reason": decision.reason,
            },
        )
        _record_stage_status(
            stage_status,
            {
                "stage": "method",
                "method": method_name,
                "status": "rebuilt",
                "reason": decision.reason,
            },
        )
        _record_timing(timings, timer.finish())
    return payloads


def evaluate_weight_grid_on_valid(
    weight_grid: list[dict[str, float]],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
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
            force=force,
        )
        for grid_index, weights in enumerate(weight_grid)
    ]


def evaluate_one_weight_run_on_valid(
    grid_index: int,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
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
        ensure_or_build_recommendable_pool_cache(context, inputs, force=force),
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
            backfill_mode=BACKFILL_MODE_BY_METHOD.get(method_name),
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


def prepare_weight_variant_windows(
    method_name: str,
    split_filter: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> list[_PreparedRecommendationWindow]:
    method = get_recommendation_method(method_name)
    strategy = method.default_candidate_strategy
    if strategy is None:
        raise ValueError(f"{method_name} requires a candidate strategy")
    windows = _filter_split(inputs.time_windows, split_filter)
    target_users = _filter_split(inputs.target_users, split_filter)
    filtered_candidates = _filter_split(candidates, split_filter)
    input_paths = _method_input_paths(method_name, _experiment_input_paths(context))

    prepared: list[_PreparedRecommendationWindow] = []
    for window in windows.loc[:, ["split", "cutoff_week", "label_week"]].to_dict(
        "records"
    ):
        window_candidates = _frame_for_window(filtered_candidates, window)
        feature_candidates = window_candidates
        if not window_candidates.empty:
            feature_candidates, _seen_partition, _seen_metadata = (
                filter_cached_seen_items(
                    window_candidates,
                    strategy=str(strategy),
                    window=window,
                    input_paths=input_paths,
                )
            )
        feature_frame, _score_artifacts = build_cached_feature_frame_for_window(
            method_name=method_name,
            strategy=str(strategy),
            window=window,
            candidates=feature_candidates,
            required_features=method.required_features,
            input_paths=input_paths,
        )
        prepared.append(
            _PreparedRecommendationWindow(
                window=window,
                target_users=_frame_for_window(target_users, window),
                candidates=window_candidates,
                feature_frame=feature_frame,
            )
        )
    return prepared


def build_recommendation_result_from_prepared_windows(
    *,
    method_name: str,
    weights: dict[str, float],
    prepared_windows: list[_PreparedRecommendationWindow],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> RecommendationResult:
    method = get_recommendation_method(method_name)
    input_paths = _method_input_paths(method_name, _experiment_input_paths(context))
    recommendation_chunks: list[pd.DataFrame] = []
    item_chunks: list[pd.DataFrame] = []
    for prepared in prepared_windows:
        window_context = RecommendationContext(
            method=method_name,
            top_k=RECOMMENDATION_TOP_K,
            exclude_seen=True,
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=pd.DataFrame(
                [prepared.window],
                columns=["split", "cutoff_week", "label_week"],
            ),
            target_users=prepared.target_users,
            candidates=prepared.candidates,
            user_profile=pd.DataFrame(),
            trend_predictions=context.trend_predictions,
            weights=weights,
            input_paths=input_paths,
            trend_model_source=context.trend_model_source,
        )
        ranked = rank_candidate_items(
            prepared.feature_frame,
            weights=weights,
            top_k=RECOMMENDATION_TOP_K,
            required_features=method.required_features,
        )
        backfill_mode = BACKFILL_MODE_BY_METHOD.get(method_name)
        if backfill_mode is not None:
            ranked = append_backfill_items(
                window_context,
                prepared.candidates,
                ranked,
                weights,
                backfill_mode,
            )
        recommendation_items = format_recommendation_items(ranked)
        recommendation_chunks.append(
            build_recommendations_csv(recommendation_items, RECOMMENDATION_TOP_K)
        )
        item_chunks.append(recommendation_items)

    recommendations = _concat_chunks(recommendation_chunks)
    recommendation_items = _concat_chunks(item_chunks)
    return RecommendationResult(
        recommendations=recommendations,
        recommendation_items=recommendation_items,
        params={
            "method": method_name,
            "method_type": method.method_type,
            "top_k": RECOMMENDATION_TOP_K,
            "exclude_seen": True,
            "weights": dict(weights),
            "candidate_strategy": method.default_candidate_strategy,
            "score_features": list(method.required_features),
        },
        metadata={
            "method": method_name,
            "candidate_rows": int(len(candidates)),
            "recommendation_rows": int(len(recommendations)),
            "recommendation_item_rows": int(len(recommendation_items)),
        },
    )


def publish_trend_method_with_weights(
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
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
    return evaluate_method_output_for_experiment(
        TREND_METHOD, context, inputs, force=force
    )


def _publish_or_reuse_trend_method(
    *,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force_methods: Sequence[str],
    force_cache: bool,
    force_candidates: bool,
    force_rebuild_all: bool,
    stage_status: list[dict[str, Any]],
    timings: list[dict[str, Any]],
) -> dict[str, Any]:
    timer = StageTimer("method", details={"method": TREND_METHOD})
    input_paths = _method_input_paths(TREND_METHOD, _experiment_input_paths(context))
    stale_reason = _method_output_stale_reason(
        TREND_METHOD,
        input_paths,
        weights=weights,
    )
    decision = should_rebuild_method(
        method_name=TREND_METHOD,
        stale_reason=stale_reason,
        force_methods=force_methods,
        force_cache=force_cache,
        force_candidates=force_candidates,
        force_rebuild_all=force_rebuild_all,
    )
    if decision.rebuild:
        payload = publish_trend_method_with_weights(
            weights,
            context,
            inputs,
            candidates,
            force=force_cache or force_rebuild_all,
        )
        status = "rebuilt"
    else:
        payload = evaluate_method_output_for_experiment(
            TREND_METHOD,
            context,
            inputs,
            force=force_cache or force_rebuild_all,
        )
        status = "reused"
    _record_stage_status(
        stage_status,
        {
            "stage": "method",
            "method": TREND_METHOD,
            "status": status,
            "reason": decision.reason,
        },
    )
    _record_stage_status(
        stage_status,
        {
            "stage": "metrics",
            "method": TREND_METHOD,
            "status": "rebuilt",
            "reason": decision.reason,
        },
    )
    _record_timing(timings, timer.finish())
    return payload


def evaluate_result_for_experiment(
    method: str,
    result: RecommendationResult,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
) -> dict[str, Any]:
    return run_recommendation_evaluation(
        method=method,
        recommendations=result.recommendations,
        target_users=inputs.target_users,
        labels=inputs.evaluation_labels,
        recommendable_pool=ensure_or_build_recommendable_pool_cache(
            context,
            inputs,
            force=force,
        ),
        input_paths={"experiment": "in_memory"},
        strict_missing_users=False,
    )


def evaluate_weight_variant_by_split(
    *,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
    prepared_windows_by_split: (
        dict[str, list[_PreparedRecommendationWindow]] | None
    ) = None,
) -> dict[str, dict[str, float]]:
    metrics_by_split: dict[str, dict[str, float]] = {}
    recommendable_pool = ensure_or_build_recommendable_pool_cache(
        context,
        inputs,
        force=force,
    )
    for split in ("valid", "test"):
        if prepared_windows_by_split is None:
            result = build_recommendation_result_in_memory(
                method_name=TREND_METHOD,
                weights=weights,
                split_filter=split,
                context=context,
                inputs=inputs,
                candidates=candidates,
            )
        else:
            result = build_recommendation_result_from_prepared_windows(
                method_name=TREND_METHOD,
                weights=weights,
                prepared_windows=prepared_windows_by_split[split],
                context=context,
                inputs=inputs,
                candidates=candidates,
            )
        metrics = evaluate_recommendations(
            result.recommendations,
            _filter_split(inputs.target_users, split),
            _filter_split(inputs.evaluation_labels, split),
            _filter_split(recommendable_pool, split),
            top_k=RECOMMENDATION_TOP_K,
            strict_missing_users=False,
        )
        metrics_by_split[split] = dict(metrics[split])
    return metrics_by_split


def evaluate_method_output_for_experiment(
    method: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
) -> dict[str, Any]:
    return run_recommendation_evaluation(
        method=method,
        recommendations=read_recommendations(
            method_output_paths(method).recommendations
        ),
        target_users=inputs.target_users,
        labels=inputs.evaluation_labels,
        recommendable_pool=ensure_or_build_recommendable_pool_cache(
            context,
            inputs,
            force=force,
        ),
        input_paths={"experiment": "in_memory"},
        strict_missing_users=False,
    )


def build_experiment_payload(
    experiment_id: str,
    baseline_payloads: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    trend_payload: dict[str, Any],
    stage_status: list[dict[str, Any]] | None = None,
    force: dict[str, object] | None = None,
    timings: list[dict[str, Any]] | None = None,
    named_ablation: list[dict[str, Any]] | None = None,
    trend_bucket_best_by_valid: list[dict[str, Any]] | None = None,
    search_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "experiment_path": str(experiment_dir(experiment_id) / "experiment.json"),
        "search_cache": dict(search_cache or {}),
        "best_weights": select_best_weights(search_results),
        "search_results": search_results,
        "ablation": build_ablation_summary([*baseline_payloads, trend_payload]),
        "named_ablation": list(named_ablation or []),
        "trend_bucket_best_by_valid": list(trend_bucket_best_by_valid or []),
        "stage_status": list(stage_status or []),
        "force": dict(force or {}),
        "timings": list(timings or []),
    }


def run_recommendation_experiment(
    context: RecommendationExperimentContext,
    experiment_id: str = "main",
    force_experiment: bool = False,
    force_methods: Sequence[str] = (),
    force_cache: bool = False,
    force_candidates: bool = False,
    force_rebuild_all: bool = False,
) -> dict[str, Any]:
    validate_safe_path_segment(experiment_id, "experiment_id")
    force_methods = tuple(force_methods)
    if experiment_id == "recommendation_enhanced":
        return run_recommendation_enhanced_experiment(
            context=context,
            force_experiment=force_experiment,
            force_methods=force_methods,
            force_cache=force_cache,
            force_candidates=force_candidates,
            force_rebuild_all=force_rebuild_all,
        )
    _validate_force_methods(force_methods)
    stage_status: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    force_payload = {
        "force_experiment": force_experiment,
        "force_methods": list(force_methods),
        "force_cache": force_cache,
        "force_candidates": force_candidates,
        "force_rebuild_all": force_rebuild_all,
    }

    timer = StageTimer("inputs")
    inputs_fresh = _recommendation_inputs_are_fresh(context)
    inputs = ensure_or_build_recommendation_inputs(
        context,
        force=force_rebuild_all,
    )
    _record_stage_status(
        stage_status,
        {
            "stage": "inputs",
            "status": "rebuilt" if force_rebuild_all or not inputs_fresh else "reused",
            "reason": (
                "force-rebuild-all"
                if force_rebuild_all
                else ("fresh" if inputs_fresh else "stale-or-missing")
            ),
        },
    )
    _record_timing(timings, timer.finish())

    baseline_payloads = run_baseline_methods(
        context,
        inputs,
        force_methods=force_methods,
        force_cache=force_cache,
        force_candidates=force_candidates,
        force_rebuild_all=force_rebuild_all,
        rebuild_stale_outputs=True,
        stage_status=stage_status,
        timings=timings,
    )
    timer = StageTimer("candidates", details={"method": TREND_METHOD})
    trend_strategy = candidate_strategy_for_method(TREND_METHOD)
    if trend_strategy is None:
        raise ValueError(f"{TREND_METHOD} requires a candidate strategy")
    candidate_force = force_rebuild_all or force_candidates
    candidate_rebuild = _candidate_items_rebuild_required(
        str(trend_strategy),
        context,
        force=candidate_force,
    )
    default_candidates = ensure_or_build_candidates_for_method(
        TREND_METHOD,
        context,
        inputs,
        force=candidate_rebuild,
    )
    if default_candidates is None:
        raise ValueError(f"{TREND_METHOD} requires a candidate strategy")
    _record_stage_status(
        stage_status,
        {
            "stage": "candidates",
            "method": TREND_METHOD,
            "status": "rebuilt" if candidate_rebuild else "reused",
            "reason": _artifact_stage_reason(
                rebuilt=candidate_rebuild,
                forced=candidate_force,
                force_reason=(
                    "force-rebuild-all" if force_rebuild_all else "force-candidates"
                ),
            ),
        },
    )
    _record_timing(timings, timer.finish())

    timer = StageTimer("cache", details={"strategy": "default"})
    cache_force = force_rebuild_all or force_cache or force_candidates
    cache_rebuild = cache_force or not _feature_cache_partitions_exist(
        "default",
        default_candidates,
    )
    ensure_or_build_feature_cache_for_strategy(
        "default",
        context,
        inputs,
        default_candidates,
        force=cache_rebuild,
    )
    _record_stage_status(
        stage_status,
        {
            "stage": "cache",
            "strategy": "default",
            "status": "rebuilt" if cache_rebuild else "reused",
            "reason": _artifact_stage_reason(
                rebuilt=cache_rebuild,
                forced=cache_force,
                force_reason=(
                    "force-rebuild-all"
                    if force_rebuild_all
                    else ("force-candidates" if force_candidates else "force-cache")
                ),
            ),
        },
    )
    _record_timing(timings, timer.finish())

    timer = StageTimer("weight_search")
    weight_grid = iter_weight_grid()
    search_cache = _build_search_cache_metadata(
        weight_grid,
        _search_cache_input_artifacts(context, inputs, default_candidates),
    )
    search_results = None
    if force_experiment and not (force_cache or force_candidates or force_rebuild_all):
        search_results = _cached_search_results_for_current_grid(
            experiment_id,
            weight_grid,
            expected_search_cache=search_cache,
        )
    search_reused = search_results is not None
    if search_results is None:
        search_results = evaluate_weight_grid_on_valid(
            weight_grid,
            context,
            inputs,
            default_candidates,
            force=force_cache or force_rebuild_all,
        )
    _record_stage_status(
        stage_status,
        {
            "stage": "experiment",
            "status": "reused" if search_reused else "rebuilt",
            "reason": (
                "cached-search-results"
                if search_reused
                else ("force-experiment" if force_experiment else "payload-write")
            ),
        },
    )
    _record_timing(timings, timer.finish())

    best_weights = select_best_weights(search_results)
    trend_payload = _publish_or_reuse_trend_method(
        weights=best_weights,
        context=context,
        inputs=inputs,
        candidates=default_candidates,
        force_methods=force_methods,
        force_cache=force_cache,
        force_candidates=force_candidates,
        force_rebuild_all=force_rebuild_all,
        stage_status=stage_status,
        timings=timings,
    )
    prepared_windows_by_split = {
        split: prepare_weight_variant_windows(
            TREND_METHOD,
            split,
            context,
            inputs,
            default_candidates,
        )
        for split in ("valid", "test")
    }
    strict_metrics = {
        variant_id: evaluate_weight_variant_by_split(
            weights=derive_strict_ablation_weights(best_weights, dropped_feature),
            context=context,
            inputs=inputs,
            candidates=default_candidates,
            force=force_cache or force_rebuild_all,
            prepared_windows_by_split=prepared_windows_by_split,
        )
        for variant_id, _display_name, dropped_feature in STRICT_VARIANTS
    }
    named_ablation = build_named_ablation_rows(
        best_weights=best_weights,
        strict_metrics=strict_metrics,
        full_model_metrics=dict(trend_payload["metrics"]),
        stable_baseline_metrics=_baseline_metrics_for_named_ablation(baseline_payloads),
    )
    trend_bucket_best_by_valid: list[dict[str, Any]] = []
    for row in select_trend_bucket_representatives(search_results):
        metrics = evaluate_weight_variant_by_split(
            weights=dict(row["weights"]),
            context=context,
            inputs=inputs,
            candidates=default_candidates,
            force=force_cache or force_rebuild_all,
            prepared_windows_by_split=prepared_windows_by_split,
        )
        trend_bucket_best_by_valid.append({**row, "metrics": metrics})
    payload = build_experiment_payload(
        experiment_id,
        baseline_payloads,
        search_results,
        trend_payload,
        stage_status=stage_status,
        force=force_payload,
        timings=timings,
        named_ablation=named_ablation,
        trend_bucket_best_by_valid=trend_bucket_best_by_valid,
        search_cache=search_cache,
    )
    write_json_atomic(payload, experiment_dir(experiment_id) / "experiment.json")
    return payload


def run_recommendation_enhanced_experiment(*args, **kwargs) -> dict[str, Any]:
    from fashion_trend.recommendation.experiments.enhanced_runner import (
        run_recommendation_enhanced_experiment as _run_enhanced_experiment,
    )

    return _run_enhanced_experiment(*args, **kwargs)


def _filter_split(dataframe: pd.DataFrame, split: str) -> pd.DataFrame:
    return dataframe.loc[dataframe["split"].astype(str) == split].reset_index(drop=True)


def _feature_cache_input_paths(
    strategy: str,
    context: RecommendationExperimentContext,
) -> dict[str, str]:
    available_paths = _experiment_input_paths(context)
    feature_input_paths = {
        key: available_paths[key]
        for key in (
            "recommendation_inputs",
            "weekly_transactions",
            "article_attributes",
            "trend_predictions",
            "time_windows",
            "target_users",
            "user_profile",
        )
        if key in available_paths
    }
    return {
        **feature_input_paths,
        **candidate_input_paths_for_strategy(strategy, available_paths),
        "candidate_items": str(candidate_items_path(strategy)),
        "candidate_metadata": str(
            candidate_items_path(strategy).with_name("metadata.json")
        ),
    }


def _candidate_items_rebuild_required(
    strategy: str,
    context: RecommendationExperimentContext,
    *,
    force: bool,
) -> bool:
    if force:
        return True
    if not candidate_items_path(strategy).exists():
        return True
    _validate_candidate_items_fresh(
        strategy,
        candidate_input_paths_for_strategy(
            strategy,
            _experiment_input_paths(context),
        ),
    )
    return False


def _artifact_stage_reason(
    *,
    rebuilt: bool,
    forced: bool,
    force_reason: str,
) -> str:
    if forced:
        return force_reason
    return "stale-or-missing" if rebuilt else "fresh"


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
    weights: dict[str, float] | None = None,
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
        expected_config=_method_freshness_config(
            method_name,
            exclude_seen=True,
            weights=weights,
        ),
        stale_message=lambda reason: _stale_output_message(method_name, reason),
    )


def _method_output_stale_reason(
    method_name: str,
    input_paths: dict[str, str],
    weights: dict[str, float] | None = None,
) -> str | None:
    output_paths = method_output_paths(method_name)
    if not output_paths.recommendations.exists():
        return "recommendations.csv is missing"
    try:
        _validate_method_output_fresh(method_name, input_paths, weights=weights)
    except RuntimeError as error:
        return str(error)
    return None


def _record_stage_status(
    stage_status: list[dict[str, Any]] | None,
    payload: dict[str, Any],
) -> None:
    if stage_status is not None:
        stage_status.append(payload)


def _record_timing(
    timings: list[dict[str, Any]] | None,
    payload: dict[str, Any],
) -> None:
    if timings is not None:
        timings.append(payload)


def _validate_force_methods(force_methods: Sequence[str]) -> None:
    allowed_methods = {*BASELINE_METHODS, TREND_METHOD}
    unknown = sorted(
        {method for method in force_methods if method not in allowed_methods}
    )
    if unknown:
        raise ValueError(f"unknown force methods: {unknown}")


def _baseline_metrics_for_named_ablation(
    baseline_payloads: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_method = {str(payload["method"]): payload for payload in baseline_payloads}
    return {
        "recent_only_baseline": {
            "method": "recent_popularity",
            "display_name": "Recent Only",
            "metrics": dict(by_method["recent_popularity"]["metrics"]),
        },
        "pop_similarity_baseline": {
            "method": "pop_similarity",
            "display_name": "Pop + Similarity baseline",
            "metrics": dict(by_method["pop_similarity"]["metrics"]),
        },
    }


def _build_search_cache_metadata(
    weight_grid: list[dict[str, float]],
    input_artifacts: dict[str, str],
) -> dict[str, Any]:
    artifacts = {str(key): str(value) for key, value in input_artifacts.items()}
    return {
        "schema_version": SEARCH_CACHE_SCHEMA_VERSION,
        "algorithm_version": SEARCH_CACHE_ALGORITHM_VERSION,
        "selection_split": "valid",
        "selection_metric": "ndcg_at_12",
        "weight_grid": [
            {feature: float(weights[feature]) for feature in SCORE_FEATURES}
            for weights in weight_grid
        ],
        "input_artifacts": artifacts,
        "input_fingerprints": build_input_fingerprints(artifacts),
    }


def _search_cache_input_artifacts(
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> dict[str, str]:
    artifacts = dict(
        _method_input_paths(TREND_METHOD, _experiment_input_paths(context))
    )
    artifacts["evaluation_labels"] = str(EVALUATION_LABELS_PATH)

    required_columns = {"split", "cutoff_week", "label_week"}
    missing_time_window_columns = sorted(
        required_columns - set(inputs.time_windows.columns)
    )
    missing_candidate_columns = sorted(required_columns - set(candidates.columns))
    if missing_time_window_columns or missing_candidate_columns:
        details = []
        if missing_time_window_columns:
            details.append(f"time_windows missing {missing_time_window_columns}")
        if missing_candidate_columns:
            details.append(f"candidates missing {missing_candidate_columns}")
        raise ValueError(
            "search cache input artifacts require window columns: " + "; ".join(details)
        )

    feature_artifacts: list[str] = []
    valid_windows = _filter_split(inputs.time_windows, "valid")
    valid_candidates = _filter_split(candidates, "valid")
    for window in valid_windows.loc[:, ["split", "cutoff_week", "label_week"]].to_dict(
        "records"
    ):
        if _frame_for_window(valid_candidates, window).empty:
            continue
        feature_artifacts.extend(
            feature_artifact_paths_for_method_window(
                method_name=TREND_METHOD,
                strategy="default",
                window=window,
                include_seen=True,
            )
        )

    for index, path in enumerate(feature_artifacts):
        artifacts[f"feature_artifact_{index:04d}"] = path
    return artifacts


def _cached_search_results_for_current_grid(
    experiment_id: str,
    weight_grid: list[dict[str, float]],
    *,
    expected_search_cache: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    path = experiment_dir(experiment_id) / "experiment.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if expected_search_cache is not None:
        stored_search_cache = payload.get("search_cache")
        if stored_search_cache != expected_search_cache:
            return None
    values = payload.get("search_results")
    if not isinstance(values, list) or len(values) != len(weight_grid):
        return None

    results: list[dict[str, Any]] = []
    for index, (expected_weights, row) in enumerate(zip(weight_grid, values)):
        if not isinstance(row, dict):
            return None
        if int(row.get("grid_index", index)) != index:
            return None
        weights = row.get("weights")
        valid_metrics = row.get("valid_metrics")
        if not isinstance(weights, dict) or not isinstance(valid_metrics, dict):
            return None
        if not _weights_match_current_grid(weights, expected_weights):
            return None
        results.append(
            {
                **row,
                "grid_index": index,
                "weights": {
                    feature: float(weights[feature]) for feature in SCORE_FEATURES
                },
                "valid_metrics": dict(valid_metrics),
            }
        )
    return results


def _weights_match_current_grid(
    actual: dict[str, object],
    expected: dict[str, float],
) -> bool:
    for feature in SCORE_FEATURES:
        if feature not in actual:
            return False
        try:
            actual_value = float(actual[feature])
        except (TypeError, ValueError):
            return False
        if not math.isclose(
            actual_value,
            float(expected[feature]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
    return True


def _recommendation_inputs_are_fresh(
    context: RecommendationExperimentContext,
) -> bool:
    include_customer_profile = _context_includes_customer_profile(context)
    include_article_product_map = _context_includes_article_product_map(context)
    required_paths = [
        TIME_WINDOWS_PATH,
        TARGET_USERS_PATH,
        EVALUATION_LABELS_PATH,
        USER_PROFILE_PATH,
        RECOMMEND_METADATA_PATH,
    ]
    if include_customer_profile:
        required_paths.append(CUSTOMER_PROFILE_PATH)
    if include_article_product_map:
        required_paths.append(ARTICLE_PRODUCT_MAP_PATH)
    if not all(path.exists() for path in required_paths):
        return False
    try:
        assert_fresh_metadata(
            metadata_path=RECOMMEND_METADATA_PATH,
            expected_input_artifacts=_upstream_input_paths(context),
            expected_output_artifacts=_recommendation_input_output_artifacts(
                include_customer_profile=include_customer_profile,
                include_article_product_map=include_article_product_map,
            ),
            expected_schema_version=1,
            expected_algorithm_version="recommendation-inputs-v1",
            expected_config=_recommendation_input_config(
                include_customer_profile=include_customer_profile,
                include_article_product_map=include_article_product_map,
            ),
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
        expected_config=_candidate_config(strategy),
        stale_message=lambda reason: _stale_candidate_message(strategy, reason),
    )


def _candidate_config(strategy: str) -> dict[str, object]:
    config: dict[str, object] = {
        "strategy": strategy,
        "candidates_per_source": RECOMMENDATION_CANDIDATES_PER_SOURCE,
    }
    if strategy == "enhanced_default":
        config.update(
            {
                "source_caps": ENHANCED_CANDIDATE_SOURCE_CAPS,
                "seen_policy": "source_level_reorder_only",
                "include_seen_for_reorder": True,
                "source_order": SOURCE_ORDER,
            }
        )
    return config


def _stale_output_message(method_name: str, reason: str) -> str:
    return (
        f"{method_name} recommendation output is stale: {reason}. "
        "Run src/16_run_recommendation_experiment.py with "
        f"--force-method {method_name} or --force-rebuild-all to rebuild."
    )


def _stale_candidate_message(strategy: str, reason: str) -> str:
    return (
        f"{strategy} candidate_items output is stale: {reason}. "
        "Run src/16_run_recommendation_experiment.py with "
        "--force-candidates or --force-rebuild-all to rebuild."
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
        "customer_profile": str(CUSTOMER_PROFILE_PATH),
        "article_product_map": str(ARTICLE_PRODUCT_MAP_PATH),
        "recommendation_inputs": str(RECOMMEND_METADATA_PATH),
        "default_candidates": str(candidate_items_path("default")),
        "default_candidate_metadata": str(
            candidate_items_path("default").with_name("metadata.json")
        ),
        "enhanced_default_candidates": str(candidate_items_path("enhanced_default")),
        "enhanced_default_candidate_metadata": str(
            candidate_items_path("enhanced_default").with_name("metadata.json")
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
        if key
        in {
            "weekly_transactions",
            "article_attributes",
            "trend_predictions",
            "raw_customers",
            "clean_articles",
        }
    }


def _context_includes_customer_profile(
    context: RecommendationExperimentContext,
) -> bool:
    return "raw_customers" in dict(context.input_paths or {})


def _context_includes_article_product_map(
    context: RecommendationExperimentContext,
) -> bool:
    return "clean_articles" in dict(context.input_paths or {})


def _require_optional_recommendation_input_frames(
    context: RecommendationExperimentContext,
) -> None:
    missing = []
    if _context_includes_customer_profile(context) and context.customers is None:
        missing.append("customers")
    if (
        _context_includes_article_product_map(context)
        and context.clean_articles is None
    ):
        missing.append("clean_articles")
    if missing:
        raise FileNotFoundError(
            "recommendation input rebuild requires loaded optional frames: "
            f"{', '.join(missing)}"
        )


def _recommendation_input_output_artifacts(
    *,
    include_customer_profile: bool,
    include_article_product_map: bool,
) -> dict[str, str]:
    artifacts = {
        "time_windows": str(TIME_WINDOWS_PATH),
        "target_users": str(TARGET_USERS_PATH),
        "evaluation_labels": str(EVALUATION_LABELS_PATH),
        "user_profile": str(USER_PROFILE_PATH),
    }
    if include_customer_profile:
        artifacts["customer_profile"] = str(CUSTOMER_PROFILE_PATH)
    if include_article_product_map:
        artifacts["article_product_map"] = str(ARTICLE_PRODUCT_MAP_PATH)
    return artifacts


def _recommendation_input_config(
    *,
    include_customer_profile: bool,
    include_article_product_map: bool,
) -> dict[str, object]:
    config: dict[str, object] = {
        "profile_top_attributes": RECOMMENDATION_PROFILE_TOP_ATTRIBUTES,
    }
    if include_customer_profile:
        config.update(
            {
                "customer_profile_schema_version": CUSTOMER_PROFILE_SCHEMA_VERSION,
                "customer_age_bucket_algorithm_version": (
                    CUSTOMER_AGE_BUCKET_ALGORITHM_VERSION
                ),
                "customer_age_buckets": list(CUSTOMER_AGE_BUCKETS),
            }
        )
    if include_article_product_map:
        config["article_product_map_schema_version"] = (
            ARTICLE_PRODUCT_MAP_SCHEMA_VERSION
        )
    return config


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
