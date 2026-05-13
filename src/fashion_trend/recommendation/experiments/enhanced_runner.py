from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.contracts import RECOMMENDATION_TOP_K
from fashion_trend.recommendation.evaluation.metrics import evaluate_recommendations
from fashion_trend.recommendation.experiments.enhanced_grid_search import (
    iter_enhanced_weight_grid,
    select_best_enhanced_weights,
)
from fashion_trend.recommendation.experiments.runner import (
    RecommendationExperimentContext,
    build_recommendation_result_in_memory,
    ensure_or_build_candidate_items,
    ensure_or_build_feature_cache_for_strategy,
    ensure_or_build_recommendable_pool_cache,
    ensure_or_build_recommendation_inputs,
    feature_artifact_paths_for_method_window,
    generate_experiment_run_id,
)
from fashion_trend.recommendation.inputs import RecommendationInputArtifacts
from fashion_trend.recommendation.paths import experiment_dir, method_output_paths
from fashion_trend.recommendation.readers import read_recommendations

ENHANCED_EXPERIMENT_ID = "recommendation_enhanced"
ENHANCED_METHOD = "enhanced_pop_similarity_trend"
ENHANCED_STRATEGY = "enhanced_default"
ENHANCED_SELECTION_METRIC = "map_at_12"
ENHANCED_TIE_BREAK = "ndcg_at_12"
COMPARISON_METHODS = (
    "recent_popularity",
    "pop_similarity",
    "pop_similarity_trend",
)


def run_recommendation_enhanced_experiment(
    context: RecommendationExperimentContext,
    force_experiment: bool = False,
    force_methods: Sequence[str] = (),
    force_cache: bool = False,
    force_candidates: bool = False,
    force_rebuild_all: bool = False,
) -> dict[str, Any]:
    _validate_enhanced_force_methods(force_methods)

    inputs = ensure_or_build_recommendation_inputs(
        context,
        force=force_rebuild_all,
    )
    candidate_force = force_rebuild_all or force_candidates
    candidates = ensure_or_build_candidate_items(
        ENHANCED_STRATEGY,
        context,
        inputs,
        force=candidate_force,
    )
    cache_force = force_rebuild_all or force_cache or force_candidates
    cache_rebuild = cache_force or not enhanced_feature_cache_partitions_exist(
        candidates
    )
    ensure_or_build_feature_cache_for_strategy(
        ENHANCED_STRATEGY,
        context,
        inputs,
        candidates,
        force=cache_rebuild,
    )

    comparison_payloads = evaluate_comparison_methods(
        context,
        inputs,
        force=force_cache or force_rebuild_all,
    )
    weight_grid = iter_enhanced_weight_grid()
    search_results = evaluate_enhanced_weight_grid_on_valid(
        weight_grid,
        context,
        inputs,
        candidates,
        force=force_cache or force_rebuild_all or force_experiment,
    )
    best_weights = select_best_enhanced_weights(search_results)
    enhanced_metrics = evaluate_enhanced_weights_by_split(
        weights=best_weights,
        context=context,
        inputs=inputs,
        candidates=candidates,
        force=force_cache or force_rebuild_all,
    )

    payload = build_enhanced_experiment_payload(
        comparison_payloads=comparison_payloads,
        search_results=search_results,
        best_weights=best_weights,
        enhanced_metrics=enhanced_metrics,
        force={
            "force_experiment": force_experiment,
            "force_methods": list(force_methods),
            "force_cache": force_cache,
            "force_candidates": force_candidates,
            "force_rebuild_all": force_rebuild_all,
        },
    )
    write_json_atomic(
        payload,
        experiment_dir(ENHANCED_EXPERIMENT_ID) / "experiment.json",
    )
    return payload


def evaluate_comparison_methods(
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    force: bool = False,
) -> list[dict[str, Any]]:
    recommendable_pool = ensure_or_build_recommendable_pool_cache(
        context,
        inputs,
        force=force,
    )
    payloads: list[dict[str, Any]] = []
    for method in COMPARISON_METHODS:
        recommendations = read_recommendations(
            method_output_paths(method).recommendations
        )
        metrics = evaluate_recommendations(
            recommendations,
            inputs.target_users,
            inputs.evaluation_labels,
            recommendable_pool,
            top_k=RECOMMENDATION_TOP_K,
            strict_missing_users=False,
        )
        payloads.append({"method": method, "metrics": metrics})
    return payloads


def evaluate_enhanced_weight_grid_on_valid(
    weight_grid: list[dict[str, float]],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
) -> list[dict[str, Any]]:
    recommendable_pool = _recommendable_pool_for_split(
        context,
        inputs,
        split="valid",
        force=force,
    )
    target_users = _filter_split(inputs.target_users, "valid")
    labels = _filter_split(inputs.evaluation_labels, "valid")
    return [
        _evaluate_one_enhanced_weight_run_on_valid(
            grid_index=grid_index,
            weights=weights,
            context=context,
            inputs=inputs,
            candidates=candidates,
            recommendable_pool=recommendable_pool,
            target_users=target_users,
            labels=labels,
        )
        for grid_index, weights in enumerate(weight_grid)
    ]


def evaluate_enhanced_weights_by_split(
    *,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
) -> dict[str, dict[str, float]]:
    metrics_by_split: dict[str, dict[str, float]] = {}
    recommendable_pool = ensure_or_build_recommendable_pool_cache(
        context,
        inputs,
        force=force,
    )
    for split in ("valid", "test"):
        result = build_recommendation_result_in_memory(
            method_name=ENHANCED_METHOD,
            weights=weights,
            split_filter=split,
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


def enhanced_feature_cache_partitions_exist(candidates: pd.DataFrame) -> bool:
    if candidates.empty:
        return True
    required_columns = {"split", "cutoff_week", "label_week"}
    missing = sorted(required_columns - set(candidates.columns))
    if missing:
        raise ValueError(f"enhanced candidates missing window columns: {missing}")

    windows = candidates.loc[
        :, ["split", "cutoff_week", "label_week"]
    ].drop_duplicates()
    for window in windows.to_dict("records"):
        paths = feature_artifact_paths_for_method_window(
            method_name=ENHANCED_METHOD,
            strategy=ENHANCED_STRATEGY,
            window=window,
            include_seen=True,
        )
        if not all(Path(path).exists() for path in paths):
            return False
    return True


def build_enhanced_experiment_payload(
    *,
    comparison_payloads: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    best_weights: dict[str, float],
    enhanced_metrics: dict[str, dict[str, float]],
    force: dict[str, object] | None = None,
) -> dict[str, Any]:
    metrics = {
        str(payload["method"]): dict(payload["metrics"])
        for payload in comparison_payloads
    }
    metrics[ENHANCED_METHOD] = dict(enhanced_metrics)
    return {
        "experiment_id": ENHANCED_EXPERIMENT_ID,
        "experiment_path": str(
            experiment_dir(ENHANCED_EXPERIMENT_ID) / "experiment.json"
        ),
        "selection_split": "valid",
        "selection_metric": ENHANCED_SELECTION_METRIC,
        "tie_break": ENHANCED_TIE_BREAK,
        "best_weights": dict(best_weights),
        "search_results": search_results,
        "comparison_methods": list(COMPARISON_METHODS),
        "metrics": metrics,
        "force": dict(force or {}),
    }


def _evaluate_one_enhanced_weight_run_on_valid(
    *,
    grid_index: int,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    recommendable_pool: pd.DataFrame,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    result = build_recommendation_result_in_memory(
        method_name=ENHANCED_METHOD,
        weights=weights,
        split_filter="valid",
        context=context,
        inputs=inputs,
        candidates=candidates,
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


def _recommendable_pool_for_split(
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    *,
    split: str,
    force: bool,
) -> pd.DataFrame:
    recommendable_pool = ensure_or_build_recommendable_pool_cache(
        context,
        inputs,
        force=force,
    )
    return _filter_split(recommendable_pool, split)


def _filter_split(dataframe: pd.DataFrame, split: str) -> pd.DataFrame:
    return dataframe.loc[dataframe["split"].astype(str) == split].reset_index(drop=True)


def _validate_enhanced_force_methods(force_methods: Sequence[str]) -> None:
    allowed = {*COMPARISON_METHODS, ENHANCED_METHOD}
    unknown = sorted({method for method in force_methods if method not in allowed})
    if unknown:
        raise ValueError(f"unknown enhanced force methods: {unknown}")
