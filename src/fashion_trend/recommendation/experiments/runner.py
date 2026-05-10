from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.contracts import RECOMMENDATION_TOP_K
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
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    USER_PROFILE_PATH,
    candidate_items_path,
    experiment_dir,
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
)
from fashion_trend.recommendation.runner import run_recommendation_method_by_window

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
    if not force and all(
        path.exists()
        for path in (
            TIME_WINDOWS_PATH,
            TARGET_USERS_PATH,
            EVALUATION_LABELS_PATH,
            USER_PROFILE_PATH,
        )
    ):
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
    )


def ensure_or_build_candidate_items(
    strategy: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
) -> pd.DataFrame:
    path = candidate_items_path(strategy)
    if not force and path.exists():
        return read_candidate_items(path)

    build_and_write_candidate_items(
        strategy=strategy,
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        trend_predictions=context.trend_predictions,
        windows=inputs.time_windows,
        target_users=inputs.target_users,
        user_profile=inputs.user_profile,
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


def run_baseline_methods(
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for method_name in BASELINE_METHODS:
        if not force and method_output_paths(method_name).recommendations.exists():
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
        run_recommendation_method_by_window(
            method_name=method_name,
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=inputs.time_windows,
            target_users=inputs.target_users,
            candidates=candidates,
            user_profile=inputs.user_profile,
            trend_predictions=None,
            collect_result=False,
            input_paths=_experiment_input_paths(context),
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
    result = method.build_recommendations(
        RecommendationContext(
            method=method_name,
            top_k=RECOMMENDATION_TOP_K,
            exclude_seen=True,
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=_filter_split(inputs.time_windows, split_filter),
            target_users=_filter_split(inputs.target_users, split_filter),
            candidates=_filter_split(candidates, split_filter),
            user_profile=_filter_split(inputs.user_profile, split_filter),
            trend_predictions=context.trend_predictions,
            weights=weights,
        )
    )
    return RecommendationResult(
        recommendations=_filter_split(result.recommendations, split_filter),
        recommendation_items=_filter_split(
            result.recommendation_items,
            split_filter,
        ),
        params=result.params,
        metadata=result.metadata,
    )


def publish_trend_method_with_weights(
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
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
        input_paths=_experiment_input_paths(context),
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


def _experiment_input_paths(
    context: RecommendationExperimentContext,
) -> dict[str, str]:
    return {
        **dict(context.input_paths or {}),
        "time_windows": str(TIME_WINDOWS_PATH),
        "target_users": str(TARGET_USERS_PATH),
        "evaluation_labels": str(EVALUATION_LABELS_PATH),
        "user_profile": str(USER_PROFILE_PATH),
        "default_candidates": str(candidate_items_path("default")),
        "similarity_candidates": str(candidate_items_path("similarity")),
    }
