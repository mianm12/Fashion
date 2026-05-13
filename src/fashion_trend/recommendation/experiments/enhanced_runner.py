from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.contracts import (
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
    RECOMMENDATION_TOP_K,
)
from fashion_trend.recommendation.evaluation.metrics import evaluate_recommendations
from fashion_trend.recommendation.experiments.enhanced_diagnostics import (
    build_candidate_diagnostics_payload,
    filter_candidate_sources_for_ablation,
    filter_seen_candidates_for_diagnostics,
)
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
from fashion_trend.recommendation.outputs import (
    build_recommendations_csv,
    format_recommendation_items,
)
from fashion_trend.recommendation.paths import experiment_dir, method_output_paths
from fashion_trend.recommendation.ranking.features import build_ranking_features
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items
from fashion_trend.recommendation.readers import read_recommendations
from fashion_trend.recommendation.registry import get_recommendation_method

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
    post_seen_candidates = filter_seen_candidates_for_diagnostics(
        candidates,
        context.transactions,
    )
    diagnostics = build_candidate_diagnostics_payload(
        candidates=candidates,
        post_seen_candidates=post_seen_candidates,
        target_users=inputs.target_users,
        labels=inputs.evaluation_labels,
    )
    named_ablation = build_enhanced_source_level_ablation_rows(
        best_weights=best_weights,
        full_model_metrics=enhanced_metrics,
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
        diagnostics=diagnostics,
        named_ablation=named_ablation,
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
    diagnostics: dict[str, object] | None = None,
    named_ablation: list[dict[str, Any]] | None = None,
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
        **dict(diagnostics or {}),
        "named_ablation": list(named_ablation or []),
        "force": dict(force or {}),
    }


def build_enhanced_source_level_ablation_rows(
    *,
    best_weights: dict[str, float],
    full_model_metrics: dict[str, dict[str, float]],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
) -> list[dict[str, Any]]:
    variants = [
        {
            "variant_id": "full_model",
            "display_name": "Full Model",
            "dropped_sources": set(),
            "dropped_weights": set(),
            "allow_all_seen": False,
            "metrics": full_model_metrics,
        },
        {
            "variant_id": "without_trend_score",
            "display_name": "enhanced_w/o Trend Score",
            "dropped_sources": set(),
            "dropped_weights": {"trend_score"},
            "allow_all_seen": False,
        },
        {
            "variant_id": "without_trend_source_score",
            "display_name": "enhanced_w/o Trend Source+Score",
            "dropped_sources": {"trend"},
            "dropped_weights": {"trend_score"},
            "allow_all_seen": False,
        },
        {
            "variant_id": "without_reorder_variant",
            "display_name": "enhanced_w/o Reorder/Variant",
            "dropped_sources": {"reorder", "product_variant"},
            "dropped_weights": {"reorder_score", "variant_score"},
            "allow_all_seen": False,
        },
        {
            "variant_id": "without_customer_segment",
            "display_name": "enhanced_w/o Customer Segment",
            "dropped_sources": {"age_popularity"},
            "dropped_weights": {"age_pop_score"},
            "allow_all_seen": False,
        },
        {
            "variant_id": "enhanced_seen_filtered",
            "display_name": "enhanced_seen_filtered",
            "dropped_sources": set(),
            "dropped_weights": set(),
            "allow_all_seen": True,
        },
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        dropped_sources = set(variant["dropped_sources"])
        dropped_weights = set(variant["dropped_weights"])
        allow_all_seen = bool(variant["allow_all_seen"])
        weights = _drop_and_renormalize_enhanced_weights(
            best_weights,
            dropped_weights,
        )
        if dropped_sources or allow_all_seen:
            variant_candidates = filter_candidate_sources_for_ablation(
                candidates,
                dropped_sources=dropped_sources,
                strategy=ENHANCED_STRATEGY,
                allow_all_seen=allow_all_seen,
            )
        else:
            variant_candidates = candidates

        metrics = dict(variant.get("metrics") or {})
        if not metrics:
            metrics = evaluate_enhanced_ablation_by_split(
                weights=weights,
                context=context,
                inputs=inputs,
                candidates=variant_candidates,
                force=force,
            )
        rows.append(
            {
                "variant_id": str(variant["variant_id"]),
                "display_name": str(variant["display_name"]),
                "method": ENHANCED_METHOD,
                "base_method": ENHANCED_METHOD,
                "candidate_strategy": ENHANCED_STRATEGY,
                "source_filter": {
                    "dropped_sources": sorted(dropped_sources),
                    "allow_all_seen": allow_all_seen,
                },
                "weight_policy": (
                    "drop_and_renormalize" if dropped_weights else "unchanged"
                ),
                "weights": weights,
                "metrics": metrics,
                "candidate_rows": int(len(variant_candidates)),
                "lineage": {
                    "base_candidate_strategy": ENHANCED_STRATEGY,
                    "evaluation_mode": "in_memory",
                    "writes_candidate_artifact": False,
                },
            }
        )
    return rows


def evaluate_enhanced_ablation_by_split(
    *,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
) -> dict[str, dict[str, float]]:
    recommendable_pool = ensure_or_build_recommendable_pool_cache(
        context,
        inputs,
        force=force,
    )
    metrics_by_split: dict[str, dict[str, float]] = {}
    for split in ("valid", "test"):
        recommendations = _build_uncached_enhanced_recommendations(
            weights=weights,
            split=split,
            context=context,
            inputs=inputs,
            candidates=candidates,
        )
        metrics = evaluate_recommendations(
            recommendations,
            _filter_split(inputs.target_users, split),
            _filter_split(inputs.evaluation_labels, split),
            _filter_split(recommendable_pool, split),
            top_k=RECOMMENDATION_TOP_K,
            strict_missing_users=False,
        )
        metrics_by_split[split] = dict(metrics[split])
    return metrics_by_split


def _build_uncached_enhanced_recommendations(
    *,
    weights: dict[str, float],
    split: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    method = get_recommendation_method(ENHANCED_METHOD)
    windows = _filter_split(inputs.time_windows, split)
    split_candidates = _filter_split(candidates, split)
    split_user_profile = _filter_split(inputs.user_profile, split)
    recommendation_chunks: list[pd.DataFrame] = []

    for window in windows.loc[:, ["split", "cutoff_week", "label_week"]].to_dict(
        "records"
    ):
        window_candidates = _frame_for_window(split_candidates, window)
        if window_candidates.empty:
            continue
        feature_frame = build_ranking_features(
            window_candidates,
            context.transactions,
            context.article_attributes,
            _frame_for_window(split_user_profile, window),
            context.trend_predictions,
            customer_profile=inputs.customer_profile,
            article_product_map=inputs.article_product_map,
        )
        feature_frame["method"] = ENHANCED_METHOD
        feature_frame = filter_seen_candidates_for_diagnostics(
            feature_frame,
            context.transactions,
        )
        ranked = rank_candidate_items(
            feature_frame,
            weights=weights,
            top_k=RECOMMENDATION_TOP_K,
            required_features=method.required_features,
        )
        recommendation_items = format_recommendation_items(ranked)
        recommendation_chunks.append(
            build_recommendations_csv(recommendation_items, RECOMMENDATION_TOP_K)
        )

    if not recommendation_chunks:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "split",
                "cutoff_week",
                "label_week",
                "method",
                "prediction",
            ]
        )
    return pd.concat(recommendation_chunks, ignore_index=True)


def _drop_and_renormalize_enhanced_weights(
    weights: dict[str, float],
    dropped_weights: set[str],
) -> dict[str, float]:
    unknown = sorted(dropped_weights - set(ENHANCED_RECOMMENDATION_SCORE_COLUMNS))
    if unknown:
        raise ValueError(f"unknown enhanced ablation weights: {unknown}")
    selected = {
        feature: float(weights.get(feature, 0.0))
        for feature in ENHANCED_RECOMMENDATION_SCORE_COLUMNS
    }
    for feature in dropped_weights:
        selected[feature] = 0.0
    remaining = sum(selected.values())
    if remaining <= 0.0:
        raise ValueError("enhanced ablation weights cannot be normalized")
    return {feature: value / remaining for feature, value in selected.items()}


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


def _frame_for_window(
    dataframe: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    mask = (
        (dataframe["split"] == window["split"])
        & (dataframe["cutoff_week"] == window["cutoff_week"])
        & (dataframe["label_week"] == window["label_week"])
    )
    return dataframe.loc[mask].copy()


def _validate_enhanced_force_methods(force_methods: Sequence[str]) -> None:
    allowed = {*COMPARISON_METHODS, ENHANCED_METHOD}
    unknown = sorted({method for method in force_methods if method not in allowed})
    if unknown:
        raise ValueError(f"unknown enhanced force methods: {unknown}")
