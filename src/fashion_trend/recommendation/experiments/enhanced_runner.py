from __future__ import annotations

import json
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
    compute_candidate_recall,
    filter_candidate_sources_for_ablation,
    filter_seen_candidates_for_diagnostics,
)
from fashion_trend.recommendation.experiments.enhanced_grid_search import (
    iter_enhanced_weight_grid,
    select_best_enhanced_result,
    select_best_enhanced_weights,
)
from fashion_trend.recommendation.experiments.runner import (
    RecommendationExperimentContext,
    _experiment_input_paths,
    _feature_cache_input_paths,
    build_recommendation_result_in_memory,
    ensure_or_build_candidate_items,
    ensure_or_build_feature_cache_for_strategy,
    ensure_or_build_recommendable_pool_cache,
    ensure_or_build_recommendation_inputs,
    feature_artifact_paths_for_method_window,
    generate_experiment_run_id,
)
from fashion_trend.recommendation.features.cache import (
    assert_feature_cache_partitions_fresh,
)
from fashion_trend.recommendation.fingerprints import build_input_fingerprints
from fashion_trend.recommendation.inputs import RecommendationInputArtifacts
from fashion_trend.recommendation.outputs import (
    build_recommendations_csv,
    format_recommendation_items,
)
from fashion_trend.recommendation.paths import (
    FEATURE_CACHE_METADATA_PATH,
    candidate_items_path,
    experiment_dir,
    method_output_paths,
)
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items
from fashion_trend.recommendation.readers import read_recommendations
from fashion_trend.recommendation.registry import get_recommendation_method
from fashion_trend.recommendation.runner import (
    build_cached_feature_frame_for_window,
    filter_cached_seen_items,
    method_input_paths_for_artifacts,
)

ENHANCED_EXPERIMENT_ID = "recommendation_enhanced"
ENHANCED_METHOD = "enhanced_pop_similarity_trend"
ENHANCED_STRATEGY = "enhanced_default"
ENHANCED_SELECTION_METRIC = "map_at_12"
ENHANCED_TIE_BREAK = "ndcg_at_12"
ENHANCED_ABLATION_SPLITS = ("valid",)
SOURCE_DERIVED_SCORE_COLUMNS = {"source_rank_score", "source_count_score"}
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
    cache_input_paths = _feature_cache_input_paths(ENHANCED_STRATEGY, context)
    cache_rebuild = cache_force or not enhanced_feature_cache_partitions_exist(
        candidates,
        input_paths=cache_input_paths,
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
    valid_feature_windows = _prepare_cached_enhanced_feature_windows(
        split="valid",
        context=context,
        inputs=inputs,
        candidates=candidates,
    )
    search_results = evaluate_enhanced_weight_grid_on_valid(
        weight_grid,
        context,
        inputs,
        candidates,
        force=force_cache or force_rebuild_all or force_experiment,
        feature_windows=valid_feature_windows,
    )
    best_weights = select_best_enhanced_weights(search_results)
    best_search_result = select_best_enhanced_result(search_results)
    named_ablation = build_enhanced_source_level_ablation_rows(
        best_weights=best_weights,
        full_model_metrics={"valid": dict(best_search_result["valid_metrics"])},
        context=context,
        inputs=inputs,
        candidates=candidates,
        force=force_cache or force_rebuild_all,
        valid_feature_windows=valid_feature_windows,
    )
    del valid_feature_windows

    enhanced_metrics = evaluate_enhanced_weights_by_split(
        weights=best_weights,
        context=context,
        inputs=inputs,
        candidates=candidates,
        force=force_cache or force_rebuild_all,
        valid_metrics=dict(best_search_result["valid_metrics"]),
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
        freshness_artifacts=enhanced_payload_freshness_artifacts(
            context,
            candidates,
        ),
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
    payloads: list[dict[str, Any]] = []
    recommendable_pool: pd.DataFrame | None = None
    for method in COMPARISON_METHODS:
        output_paths = method_output_paths(method)
        if output_paths.metrics.exists():
            payloads.append(
                {
                    "method": method,
                    "metrics": _read_comparison_metrics(output_paths.metrics, method),
                    "metrics_source": str(output_paths.metrics),
                }
            )
            continue

        if recommendable_pool is None:
            recommendable_pool = ensure_or_build_recommendable_pool_cache(
                context,
                inputs,
                force=force,
            )
        recommendations = read_recommendations(output_paths.recommendations)
        metrics = evaluate_recommendations(
            recommendations,
            inputs.target_users,
            inputs.evaluation_labels,
            recommendable_pool,
            top_k=RECOMMENDATION_TOP_K,
            strict_missing_users=False,
        )
        payloads.append(
            {
                "method": method,
                "metrics": metrics,
                "metrics_source": "computed_from_recommendations",
            }
        )
    return payloads


def _read_comparison_metrics(path: Path, method: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") != method:
        raise RuntimeError(
            f"{method} metrics payload method mismatch: {payload.get('method')}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"{method} metrics payload missing metrics")
    missing_splits = sorted({"valid", "test"} - set(metrics))
    if missing_splits:
        raise RuntimeError(f"{method} metrics payload missing splits: {missing_splits}")
    return metrics


def evaluate_enhanced_weight_grid_on_valid(
    weight_grid: list[dict[str, float]],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
    feature_windows: list[tuple[dict[str, object], pd.DataFrame]] | None = None,
) -> list[dict[str, Any]]:
    recommendable_pool = _recommendable_pool_for_split(
        context,
        inputs,
        split="valid",
        force=force,
    )
    target_users = _filter_split(inputs.target_users, "valid")
    labels = _filter_split(inputs.evaluation_labels, "valid")
    if feature_windows is None:
        feature_windows = _prepare_cached_enhanced_feature_windows(
            split="valid",
            context=context,
            inputs=inputs,
            candidates=candidates,
        )
    return [
        _evaluate_one_enhanced_weight_run_on_valid(
            grid_index=grid_index,
            weights=weights,
            feature_windows=feature_windows,
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
    valid_metrics: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    metrics_by_split: dict[str, dict[str, float]] = {}
    for split in ("valid", "test"):
        if split == "valid" and valid_metrics is not None:
            metrics_by_split[split] = dict(valid_metrics)
            continue

        recommendable_pool = _recommendable_pool_for_split(
            context=context,
            inputs=inputs,
            split=split,
            force=force,
        )
        feature_windows = _prepare_cached_enhanced_feature_windows(
            split=split,
            context=context,
            inputs=inputs,
            candidates=candidates,
        )
        recommendations = _rank_cached_feature_windows(
            feature_windows,
            weights=weights,
        )
        metrics = evaluate_recommendations(
            recommendations,
            _filter_split(inputs.target_users, split),
            _filter_split(inputs.evaluation_labels, split),
            recommendable_pool,
            top_k=RECOMMENDATION_TOP_K,
            strict_missing_users=False,
        )
        metrics_by_split[split] = dict(metrics[split])
    return metrics_by_split


def enhanced_feature_cache_partitions_exist(
    candidates: pd.DataFrame,
    input_paths: dict[str, str] | None = None,
) -> bool:
    if not _feature_cache_manifest_has_strategy(ENHANCED_STRATEGY):
        return False
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
    if input_paths is not None:
        assert_feature_cache_partitions_fresh(
            strategy=ENHANCED_STRATEGY,
            candidates=candidates,
            input_paths=input_paths,
        )
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
    freshness_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    metrics = {
        str(payload["method"]): dict(payload["metrics"])
        for payload in comparison_payloads
    }
    metrics[ENHANCED_METHOD] = dict(enhanced_metrics)
    freshness_artifacts = dict(freshness_artifacts or {})
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
        "freshness_artifacts": freshness_artifacts,
        "freshness_fingerprints": build_input_fingerprints(freshness_artifacts),
    }


def enhanced_payload_freshness_artifacts(
    context: RecommendationExperimentContext,
    candidates: pd.DataFrame,
) -> dict[str, str]:
    available_paths = _experiment_input_paths(context)
    artifacts = {
        key: available_paths[key]
        for key in (
            "weekly_transactions",
            "article_attributes",
            "trend_predictions",
            "time_windows",
            "target_users",
            "user_profile",
            "customer_profile",
            "article_product_map",
            "recommendation_inputs",
        )
        if key in available_paths
    }
    artifacts["enhanced_default_candidates"] = str(
        candidate_items_path(ENHANCED_STRATEGY)
    )
    artifacts["enhanced_default_candidate_metadata"] = str(
        candidate_items_path(ENHANCED_STRATEGY).with_name("metadata.json")
    )
    artifacts["feature_cache_metadata"] = str(FEATURE_CACHE_METADATA_PATH)

    feature_artifacts: list[str] = []
    if not candidates.empty:
        windows = candidates.loc[
            :, ["split", "cutoff_week", "label_week"]
        ].drop_duplicates()
        for window in windows.to_dict("records"):
            feature_artifacts.extend(
                feature_artifact_paths_for_method_window(
                    method_name=ENHANCED_METHOD,
                    strategy=ENHANCED_STRATEGY,
                    window=window,
                    include_seen=True,
                )
            )
    for index in range(0, len(feature_artifacts), 2):
        partition_index = index // 2
        artifacts[f"feature_partition_{partition_index:04d}"] = feature_artifacts[index]
        if index + 1 < len(feature_artifacts):
            artifacts[f"feature_partition_metadata_{partition_index:04d}"] = (
                feature_artifacts[index + 1]
            )
    return artifacts


def build_enhanced_source_level_ablation_rows(
    *,
    best_weights: dict[str, float],
    full_model_metrics: dict[str, dict[str, float]],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
    valid_feature_windows: list[tuple[dict[str, object], pd.DataFrame]] | None = None,
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
    candidate_scope = _filter_splits(candidates, ENHANCED_ABLATION_SPLITS)
    for variant in variants:
        dropped_sources = set(variant["dropped_sources"])
        dropped_weights = set(variant["dropped_weights"])
        allow_all_seen = bool(variant["allow_all_seen"])
        weights = _drop_and_renormalize_enhanced_weights(
            best_weights,
            dropped_weights,
        )
        variant_candidates: pd.DataFrame | None = None
        metrics = dict(variant.get("metrics") or {})
        candidate_rows = int(len(candidate_scope))
        evaluation_mode = "in_memory"
        if not metrics:
            if valid_feature_windows and _can_rank_prepared_ablation(
                dropped_sources,
                allow_all_seen=allow_all_seen,
            ):
                metrics, candidate_rows = (
                    evaluate_enhanced_ablation_from_feature_windows(
                        weights=weights,
                        feature_windows=valid_feature_windows,
                        dropped_sources=dropped_sources,
                        allow_all_seen=allow_all_seen,
                        target_users=_filter_split(inputs.target_users, "valid"),
                        labels=_filter_split(inputs.evaluation_labels, "valid"),
                        recommendable_pool=_recommendable_pool_for_split(
                            context=context,
                            inputs=inputs,
                            split="valid",
                            force=force,
                        ),
                    )
                )
                evaluation_mode = "prepared_valid_feature_windows"
            elif valid_feature_windows:
                variant_candidates = _filter_candidates_for_ablation_variant(
                    candidate_scope,
                    dropped_sources=dropped_sources,
                    allow_all_seen=allow_all_seen,
                )
                candidate_rows = int(len(variant_candidates))
                metrics = _candidate_only_ablation_metrics(
                    variant_candidates,
                    inputs,
                )
                evaluation_mode = "candidate_diagnostic_only"
            else:
                variant_candidates = _filter_candidates_for_ablation_variant(
                    candidate_scope,
                    dropped_sources=dropped_sources,
                    allow_all_seen=allow_all_seen,
                )
                candidate_rows = int(len(variant_candidates))
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
                "candidate_rows": candidate_rows,
                "lineage": {
                    "base_candidate_strategy": ENHANCED_STRATEGY,
                    "evaluation_mode": evaluation_mode,
                    "evaluation_splits": sorted(metrics),
                    "writes_candidate_artifact": False,
                },
            }
        )
    return rows


def _filter_candidates_for_ablation_variant(
    candidates: pd.DataFrame,
    *,
    dropped_sources: set[str],
    allow_all_seen: bool,
) -> pd.DataFrame:
    if dropped_sources or allow_all_seen:
        return filter_candidate_sources_for_ablation(
            candidates,
            dropped_sources=dropped_sources,
            strategy=ENHANCED_STRATEGY,
            allow_all_seen=allow_all_seen,
        )
    return candidates


def _can_rank_prepared_ablation(
    dropped_sources: set[str],
    *,
    allow_all_seen: bool,
) -> bool:
    if allow_all_seen:
        return False
    seen_sensitive_sources = {"reorder", "product_variant"}
    return not bool(dropped_sources & seen_sensitive_sources)


def evaluate_enhanced_ablation_from_feature_windows(
    *,
    weights: dict[str, float],
    feature_windows: list[tuple[dict[str, object], pd.DataFrame]],
    dropped_sources: set[str],
    allow_all_seen: bool,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
    recommendable_pool: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], int]:
    feature_windows_for_variant: list[tuple[dict[str, object], pd.DataFrame]] = []
    candidate_rows = 0
    for window, feature_frame in feature_windows:
        if dropped_sources or allow_all_seen:
            feature_frame = filter_candidate_sources_for_ablation(
                feature_frame,
                dropped_sources=dropped_sources,
                strategy=ENHANCED_STRATEGY,
                allow_all_seen=allow_all_seen,
            )
        candidate_rows += int(len(feature_frame))
        feature_windows_for_variant.append((window, feature_frame))

    recommendations = _rank_cached_feature_windows(
        feature_windows_for_variant,
        weights=weights,
    )
    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=RECOMMENDATION_TOP_K,
        strict_missing_users=False,
    )
    return {"valid": dict(metrics["valid"])}, candidate_rows


def _candidate_only_ablation_metrics(
    candidates: pd.DataFrame,
    inputs: RecommendationInputArtifacts,
) -> dict[str, dict[str, float]]:
    recall = compute_candidate_recall(
        candidates,
        inputs.target_users,
        inputs.evaluation_labels,
        split="valid",
    )
    return {
        "valid": {
            "candidate_recall": float(recall["candidate_recall"]),
            "hit_label_item_count": float(recall["hit_label_item_count"]),
            "label_item_count": float(recall["label_item_count"]),
        }
    }


def evaluate_enhanced_ablation_by_split(
    *,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
) -> dict[str, dict[str, float]]:
    metrics_by_split: dict[str, dict[str, float]] = {}
    for split in ENHANCED_ABLATION_SPLITS:
        recommendable_pool = _recommendable_pool_for_split(
            context=context,
            inputs=inputs,
            split=split,
            force=force,
        )
        recommendations = _build_cached_enhanced_ablation_recommendations(
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
            recommendable_pool,
            top_k=RECOMMENDATION_TOP_K,
            strict_missing_users=False,
        )
        metrics_by_split[split] = dict(metrics[split])
    return metrics_by_split


def _build_cached_enhanced_ablation_recommendations(
    *,
    weights: dict[str, float],
    split: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    method = get_recommendation_method(ENHANCED_METHOD)
    input_paths = method_input_paths_for_artifacts(
        ENHANCED_METHOD,
        _experiment_input_paths(context),
    )
    cached_required_features = [
        feature
        for feature in method.required_features
        if feature not in SOURCE_DERIVED_SCORE_COLUMNS
    ]
    windows = _filter_split(inputs.time_windows, split)
    split_candidates = _filter_split(candidates, split)
    recommendation_chunks: list[pd.DataFrame] = []

    for window in windows.loc[:, ["split", "cutoff_week", "label_week"]].to_dict(
        "records"
    ):
        window_candidates = _frame_for_window(split_candidates, window)
        if window_candidates.empty:
            continue
        filtered_candidates, _, _ = filter_cached_seen_items(
            window_candidates,
            strategy=ENHANCED_STRATEGY,
            window=window,
            input_paths=input_paths,
        )
        if filtered_candidates.empty:
            continue
        feature_frame, _ = build_cached_feature_frame_for_window(
            method_name=ENHANCED_METHOD,
            strategy=ENHANCED_STRATEGY,
            window=window,
            candidates=filtered_candidates,
            required_features=cached_required_features,
            input_paths=input_paths,
        )
        ranked = rank_candidate_items(
            feature_frame,
            weights=weights,
            top_k=RECOMMENDATION_TOP_K,
            required_features=method.required_features,
            include_candidate_sources=False,
        )
        recommendation_chunks.append(
            build_recommendations_csv(ranked, RECOMMENDATION_TOP_K)
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
    feature_windows: list[tuple[dict[str, object], pd.DataFrame]],
    recommendable_pool: pd.DataFrame,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    recommendations = _rank_cached_feature_windows(
        feature_windows,
        weights=weights,
    )
    metrics = evaluate_recommendations(
        recommendations,
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


def _prepare_cached_enhanced_feature_windows(
    *,
    split: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> list[tuple[dict[str, object], pd.DataFrame]]:
    method = get_recommendation_method(ENHANCED_METHOD)
    input_paths = method_input_paths_for_artifacts(
        ENHANCED_METHOD,
        _experiment_input_paths(context),
    )
    windows = _filter_split(inputs.time_windows, split)
    split_candidates = _filter_split(candidates, split)
    feature_windows: list[tuple[dict[str, object], pd.DataFrame]] = []
    for window in windows.loc[:, ["split", "cutoff_week", "label_week"]].to_dict(
        "records"
    ):
        window_candidates = _frame_for_window(split_candidates, window)
        if window_candidates.empty:
            continue
        filtered_candidates, _, _ = filter_cached_seen_items(
            window_candidates,
            strategy=ENHANCED_STRATEGY,
            window=window,
            input_paths=input_paths,
        )
        feature_frame, _ = build_cached_feature_frame_for_window(
            method_name=ENHANCED_METHOD,
            strategy=ENHANCED_STRATEGY,
            window=window,
            candidates=filtered_candidates,
            required_features=method.required_features,
            input_paths=input_paths,
        )
        feature_windows.append((window, feature_frame))
    return feature_windows


def _rank_cached_feature_windows(
    feature_windows: list[tuple[dict[str, object], pd.DataFrame]],
    *,
    weights: dict[str, float],
) -> pd.DataFrame:
    method = get_recommendation_method(ENHANCED_METHOD)
    recommendation_chunks: list[pd.DataFrame] = []
    for _window, feature_frame in feature_windows:
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


def _filter_splits(dataframe: pd.DataFrame, splits: Sequence[str]) -> pd.DataFrame:
    if dataframe.empty or "split" not in dataframe.columns:
        return dataframe.iloc[0:0].copy()
    split_values = {str(split) for split in splits}
    return dataframe.loc[dataframe["split"].astype(str).isin(split_values)].reset_index(
        drop=True
    )


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


def _feature_cache_manifest_has_strategy(strategy: str) -> bool:
    if not FEATURE_CACHE_METADATA_PATH.exists():
        return False
    try:
        manifest = json.loads(FEATURE_CACHE_METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(manifest, dict):
        return False
    entries = manifest.get("entries")
    return isinstance(entries, dict) and f"strategy:{strategy}" in entries
