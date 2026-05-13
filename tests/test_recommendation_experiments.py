from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fashion_trend.recommendation.contracts import (
    CANDIDATE_ITEM_COLUMNS,
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
)
from fashion_trend.recommendation.experiments import enhanced_runner
from fashion_trend.recommendation.experiments import runner as experiment_runner
from fashion_trend.recommendation.experiments.ablation import build_ablation_summary
from fashion_trend.recommendation.experiments.enhanced_diagnostics import (
    filter_candidate_sources_for_ablation,
)
from fashion_trend.recommendation.experiments.enhanced_grid_search import (
    iter_enhanced_weight_grid,
    select_best_enhanced_weights,
)
from fashion_trend.recommendation.experiments.grid_search import (
    iter_weight_grid,
    select_best_weights,
)
from fashion_trend.recommendation.experiments.runner import (
    RecommendationExperimentContext,
    candidate_strategy_for_method,
    ensure_or_build_candidate_items,
    ensure_or_build_feature_cache_for_strategy,
    generate_experiment_run_id,
    run_baseline_methods,
    run_recommendation_experiment,
)
from fashion_trend.recommendation.fingerprints import build_input_fingerprints
from fashion_trend.recommendation.freshness import build_artifact_metadata
from fashion_trend.recommendation.inputs import RecommendationInputArtifacts
from fashion_trend.recommendation.paths import experiment_run_dir
from fashion_trend.recommendation.perf import StageTimer, format_stage_log


def test_stage_timer_records_elapsed_and_rows() -> None:
    timer = StageTimer("feature_cache", rows=123, details={"name": "sim_score"})

    payload = timer.finish()

    assert payload["stage"] == "feature_cache"
    assert payload["rows"] == 123
    assert payload["name"] == "sim_score"
    assert payload["elapsed_seconds"] >= 0.0


def test_format_stage_log_keeps_stage_first_and_payload_order() -> None:
    payload = {
        "stage": "evaluation",
        "elapsed_seconds": 1.25,
        "method": "m",
        "rows": 10,
    }

    assert (
        format_stage_log(payload)
        == "stage=evaluation elapsed_seconds=1.25 method=m rows=10"
    )


def test_weight_grid_contains_only_valid_normalized_weights() -> None:
    weights = list(iter_weight_grid())

    assert weights
    assert len(weights) <= 30
    assert all(abs(sum(item.values()) - 1.0) <= 1e-9 for item in weights)
    assert all(all(value >= 0.0 for value in item.values()) for item in weights)
    assert {
        "pop_score": 0.4,
        "sim_score": 0.3,
        "trend_score": 0.2,
        "recent_score": 0.1,
    } in weights
    assert {
        "pop_score": 0.4,
        "sim_score": 0.1,
        "trend_score": 0.0,
        "recent_score": 0.5,
    } in weights
    assert {
        "pop_score": 0.3,
        "sim_score": 0.1,
        "trend_score": 0.2,
        "recent_score": 0.4,
    } in weights


def test_select_best_weights_defaults_to_ndcg_at_12() -> None:
    results = [
        {
            "grid_index": 0,
            "weights": {
                "pop_score": 0.4,
                "sim_score": 0.3,
                "trend_score": 0.2,
                "recent_score": 0.1,
            },
            "valid_metrics": {"map_at_12": 0.30, "ndcg_at_12": 0.10},
        },
        {
            "grid_index": 1,
            "weights": {
                "pop_score": 0.3,
                "sim_score": 0.1,
                "trend_score": 0.2,
                "recent_score": 0.4,
            },
            "valid_metrics": {"map_at_12": 0.20, "ndcg_at_12": 0.20},
        },
    ]

    assert select_best_weights(results) == {
        "pop_score": 0.3,
        "sim_score": 0.1,
        "trend_score": 0.2,
        "recent_score": 0.4,
    }


def test_select_best_weights_uses_stable_grid_order_for_ties() -> None:
    results = [
        {
            "grid_index": 1,
            "weights": {
                "pop_score": 0.3,
                "sim_score": 0.4,
                "trend_score": 0.2,
                "recent_score": 0.1,
            },
            "valid_metrics": {"map_at_12": 0.25},
        },
        {
            "grid_index": 0,
            "weights": {
                "pop_score": 0.4,
                "sim_score": 0.3,
                "trend_score": 0.2,
                "recent_score": 0.1,
            },
            "valid_metrics": {"map_at_12": 0.25},
        },
    ]

    assert select_best_weights(results, metric_name="map_at_12") == {
        "pop_score": 0.4,
        "sim_score": 0.3,
        "trend_score": 0.2,
        "recent_score": 0.1,
    }


def test_select_best_weights_rejects_empty_grid_results() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_best_weights([])


def test_enhanced_weight_grid_has_at_most_32_normalized_rows() -> None:
    weights = list(iter_enhanced_weight_grid())

    assert weights
    assert len(weights) <= 32
    assert all(tuple(item) == ENHANCED_RECOMMENDATION_SCORE_COLUMNS for item in weights)
    assert all(abs(sum(item.values()) - 1.0) <= 1e-9 for item in weights)
    assert all(all(value >= 0.0 for value in item.values()) for item in weights)


def test_enhanced_selects_best_weights_by_valid_map_then_ndcg() -> None:
    low_map = {
        "grid_index": 0,
        "weights": dict(iter_enhanced_weight_grid()[0]),
        "valid_metrics": {"map_at_12": 0.10, "ndcg_at_12": 0.90},
    }
    best_tie_break = {
        "grid_index": 2,
        "weights": dict(iter_enhanced_weight_grid()[2]),
        "valid_metrics": {"map_at_12": 0.20, "ndcg_at_12": 0.70},
    }
    earlier_grid = {
        "grid_index": 1,
        "weights": dict(iter_enhanced_weight_grid()[1]),
        "valid_metrics": {"map_at_12": 0.20, "ndcg_at_12": 0.60},
    }

    assert select_best_enhanced_weights(
        [low_map, best_tie_break, earlier_grid]
    ) == dict(iter_enhanced_weight_grid()[2])
    grid_order_tie = {
        "grid_index": 1,
        "weights": dict(iter_enhanced_weight_grid()[1]),
        "valid_metrics": {"map_at_12": 0.20, "ndcg_at_12": 0.70},
    }
    assert select_best_enhanced_weights([best_tie_break, grid_order_tie]) == dict(
        iter_enhanced_weight_grid()[1]
    )


def test_recommendation_enhanced_dispatch_does_not_call_main_runner(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        experiment_runner,
        "run_recommendation_enhanced_experiment",
        lambda **kwargs: calls.append(("enhanced", kwargs))
        or {"experiment_id": "recommendation_enhanced"},
    )
    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_recommendation_inputs",
        lambda *args, **kwargs: pytest.fail("main runner path should not run"),
    )

    payload = run_recommendation_experiment(
        RecommendationExperimentContext(
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
        ),
        experiment_id="recommendation_enhanced",
        force_experiment=True,
        force_cache=True,
    )

    assert payload == {"experiment_id": "recommendation_enhanced"}
    assert calls[0][0] == "enhanced"
    assert calls[0][1]["force_experiment"] is True
    assert calls[0][1]["force_cache"] is True


def test_recommendation_enhanced_payload_records_valid_test_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(
            [
                {"split": "valid", "cutoff_week": 10, "label_week": 11},
                {"split": "test", "cutoff_week": 11, "label_week": 12},
            ]
        ),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "c1",
                "article_id": "0000000001",
                "source": "reorder",
                "source_rank": 1,
            }
        ]
    )
    writes: list[tuple[dict[str, object], object]] = []

    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_recommendation_inputs",
        lambda context, force=False: inputs,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_candidate_items",
        lambda strategy, context, inputs, force=False: candidates,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_feature_cache_for_strategy",
        lambda strategy, context, inputs, candidates, force=False: None,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "enhanced_feature_cache_partitions_exist",
        lambda candidates, input_paths=None: True,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_comparison_methods",
        lambda context, inputs, force=False: [
            {"method": "recent_popularity", "metrics": {"valid": {}, "test": {}}},
            {"method": "pop_similarity", "metrics": {"valid": {}, "test": {}}},
            {"method": "pop_similarity_trend", "metrics": {"valid": {}, "test": {}}},
        ],
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_enhanced_weight_grid_on_valid",
        lambda weight_grid, context, inputs, candidates, force=False: [
            {
                "grid_index": 0,
                "weights": dict(weight_grid[0]),
                "valid_metrics": {"map_at_12": 0.20, "ndcg_at_12": 0.30},
            }
        ],
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_enhanced_weights_by_split",
        lambda **kwargs: {
            "valid": {"map_at_12": 0.20, "ndcg_at_12": 0.30},
            "test": {"map_at_12": 0.25, "ndcg_at_12": 0.35},
        },
    )
    monkeypatch.setattr(
        enhanced_runner,
        "build_enhanced_source_level_ablation_rows",
        lambda **kwargs: [
            {
                "variant_id": "full_model",
                "display_name": "Full Model",
                "source_filter": {},
                "weights": {},
                "metrics": {},
                "candidate_rows": 1,
                "lineage": {},
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "experiment_dir",
        lambda experiment_id: tmp_path / "experiments" / experiment_id,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "write_json_atomic",
        lambda payload, path: writes.append((payload, path)),
    )

    payload = enhanced_runner.run_recommendation_enhanced_experiment(
        RecommendationExperimentContext(
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
        )
    )

    assert payload["experiment_id"] == "recommendation_enhanced"
    assert payload["selection_metric"] == "map_at_12"
    assert payload["tie_break"] == "ndcg_at_12"
    assert "valid" in payload["metrics"]["enhanced_pop_similarity_trend"]
    assert "test" in payload["metrics"]["enhanced_pop_similarity_trend"]
    assert payload["named_ablation"][0]["display_name"] == "Full Model"
    assert payload["experiment_path"] == str(
        tmp_path / "experiments" / "recommendation_enhanced" / "experiment.json"
    )
    assert writes == [
        (
            payload,
            tmp_path / "experiments" / "recommendation_enhanced" / "experiment.json",
        )
    ]


def test_recommendation_enhanced_does_not_publish_pop_similarity_trend_stable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        enhanced_runner,
        "publish_trend_method_with_weights",
        lambda *args, **kwargs: pytest.fail(
            "enhanced experiment must not publish stable"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "run_recommendation_method_by_window",
        lambda *args, **kwargs: pytest.fail(
            "enhanced experiment must not write stable"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_recommendation_inputs",
        lambda context, force=False: RecommendationInputArtifacts(
            time_windows=pd.DataFrame(),
            target_users=pd.DataFrame(),
            evaluation_labels=pd.DataFrame(),
            user_profile=pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_candidate_items",
        lambda strategy, context, inputs, force=False: pd.DataFrame(),
    )
    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_feature_cache_for_strategy",
        lambda strategy, context, inputs, candidates, force=False: None,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "enhanced_feature_cache_partitions_exist",
        lambda candidates, input_paths=None: True,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_comparison_methods",
        lambda context, inputs, force=False: [],
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_enhanced_weight_grid_on_valid",
        lambda weight_grid, context, inputs, candidates, force=False: [
            {
                "grid_index": 0,
                "weights": dict(weight_grid[0]),
                "valid_metrics": {"map_at_12": 0.20, "ndcg_at_12": 0.30},
            }
        ],
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_enhanced_weights_by_split",
        lambda **kwargs: {"valid": {}, "test": {}},
    )
    monkeypatch.setattr(
        enhanced_runner,
        "build_enhanced_source_level_ablation_rows",
        lambda **kwargs: [],
        raising=False,
    )
    monkeypatch.setattr(enhanced_runner, "write_json_atomic", lambda *args: None)

    payload = enhanced_runner.run_recommendation_enhanced_experiment(
        RecommendationExperimentContext(
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
        )
    )

    assert payload["metrics"]["enhanced_pop_similarity_trend"] == {
        "valid": {},
        "test": {},
    }


def test_source_level_ablation_recomputes_source_fields_after_filter() -> None:
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "u1",
                "article_id": "a1",
                "candidate_sources": "trend|reorder|age_popularity",
                "primary_source": "trend",
                "best_source_rank": 1,
                "has_reorder_source": True,
                "allow_seen": True,
                "source_rank_score": 999.0,
                "source_count_score": 999.0,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "u1",
                "article_id": "a2",
                "candidate_sources": "trend",
                "primary_source": "trend",
                "best_source_rank": 1,
                "has_reorder_source": False,
                "allow_seen": False,
                "source_rank_score": 999.0,
                "source_count_score": 999.0,
            },
        ]
    )

    filtered = filter_candidate_sources_for_ablation(
        candidates,
        dropped_sources={"trend"},
        strategy="enhanced_default",
    )

    assert filtered["article_id"].tolist() == ["a1"]
    assert filtered.loc[0, "candidate_sources"] == "reorder|age_popularity"
    assert filtered.loc[0, "primary_source"] == "reorder"
    assert filtered.loc[0, "has_reorder_source"] is True
    assert filtered.loc[0, "allow_seen"] is True
    assert filtered.loc[0, "source_rank_score"] != 999.0
    assert filtered.loc[0, "source_count_score"] != 999.0


def test_enhanced_seen_filtered_filters_all_seen_items() -> None:
    from fashion_trend.recommendation.features.cache import build_candidate_seen_flags
    from fashion_trend.recommendation.ranking.filters import (
        filter_seen_items_by_source_policy,
    )

    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "u1",
                "article_id": "a1",
                "candidate_sources": "reorder",
                "primary_source": "reorder",
                "best_source_rank": 1,
                "has_reorder_source": True,
                "allow_seen": True,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "u1",
                "article_id": "a2",
                "candidate_sources": "trend",
                "primary_source": "trend",
                "best_source_rank": 2,
                "has_reorder_source": False,
                "allow_seen": False,
            },
        ]
    )
    transactions = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "a1", "week_id": 10},
            {"customer_id": "u1", "article_id": "a2", "week_id": 9},
        ]
    )

    ablated = filter_candidate_sources_for_ablation(
        candidates,
        dropped_sources=set(),
        strategy="enhanced_default",
        allow_all_seen=True,
    )
    seen_flags = build_candidate_seen_flags(ablated, transactions)
    merged = ablated.merge(
        seen_flags.loc[
            :,
            [
                "split",
                "cutoff_week",
                "label_week",
                "strategy",
                "customer_id",
                "article_id",
                "is_seen",
            ],
        ],
        on=[
            "split",
            "cutoff_week",
            "label_week",
            "strategy",
            "customer_id",
            "article_id",
        ],
        how="left",
    )
    merged["is_seen"] = merged["is_seen"].fillna(False).astype(bool)

    assert ablated["allow_seen"].tolist() == [False, False]
    assert filter_seen_items_by_source_policy(merged).empty


def test_enhanced_payload_contains_required_ablation_rows(
    tmp_path,
    monkeypatch,
) -> None:
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(
            [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
        ),
        target_users=pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "customer_id": "u1",
                }
            ]
        ),
        evaluation_labels=pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "customer_id": "u1",
                    "article_id": "a1",
                }
            ]
        ),
        user_profile=pd.DataFrame(),
    )
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "u1",
                "article_id": "a1",
                "candidate_sources": "trend|reorder",
                "primary_source": "trend",
                "best_source_rank": 1,
                "has_reorder_source": True,
                "allow_seen": True,
            }
        ]
    )
    required_rows = [
        "Full Model",
        "enhanced_w/o Trend Score",
        "enhanced_w/o Trend Source+Score",
        "enhanced_w/o Reorder/Variant",
        "enhanced_w/o Customer Segment",
        "enhanced_seen_filtered",
    ]

    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_recommendation_inputs",
        lambda context, force=False: inputs,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_candidate_items",
        lambda strategy, context, inputs, force=False: candidates,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "ensure_or_build_feature_cache_for_strategy",
        lambda strategy, context, inputs, candidates, force=False: None,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "enhanced_feature_cache_partitions_exist",
        lambda candidates, input_paths=None: True,
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_comparison_methods",
        lambda context, inputs, force=False: [],
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_enhanced_weight_grid_on_valid",
        lambda weight_grid, context, inputs, candidates, force=False: [
            {
                "grid_index": 0,
                "weights": dict(weight_grid[0]),
                "valid_metrics": {"map_at_12": 0.20, "ndcg_at_12": 0.30},
            }
        ],
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_enhanced_weights_by_split",
        lambda **kwargs: {"valid": {"map_at_12": 0.20}},
    )
    monkeypatch.setattr(
        enhanced_runner,
        "evaluate_enhanced_ablation_by_split",
        lambda **kwargs: {"valid": {"map_at_12": 0.10}},
    )
    monkeypatch.setattr(
        enhanced_runner,
        "experiment_dir",
        lambda experiment_id: tmp_path / "experiments" / experiment_id,
    )
    monkeypatch.setattr(enhanced_runner, "write_json_atomic", lambda *args: None)

    payload = enhanced_runner.run_recommendation_enhanced_experiment(
        RecommendationExperimentContext(
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
        )
    )

    rows = payload["named_ablation"]
    by_name = {row["display_name"]: row for row in rows}
    assert set(by_name) == set(required_rows)
    assert all(
        {"source_filter", "weights", "metrics", "candidate_rows", "lineage"} <= set(row)
        for row in rows
    )
    assert by_name["enhanced_w/o Trend Score"]["source_filter"]["dropped_sources"] == []
    assert by_name["enhanced_w/o Trend Score"]["weights"]["trend_score"] == 0.0
    assert by_name["enhanced_w/o Trend Source+Score"]["source_filter"][
        "dropped_sources"
    ] == ["trend"]
    assert by_name["enhanced_w/o Reorder/Variant"]["source_filter"][
        "dropped_sources"
    ] == ["product_variant", "reorder"]
    assert by_name["enhanced_w/o Customer Segment"]["source_filter"][
        "dropped_sources"
    ] == ["age_popularity"]
    assert by_name["enhanced_seen_filtered"]["source_filter"]["allow_all_seen"] is True
    assert "candidate_recall_pre_seen" in payload
    assert "candidate_recall_post_seen" in payload
    assert "source_hit_contribution_pre_seen" in payload
    assert "source_hit_contribution_post_seen" in payload
    assert "avg_candidates_per_user" in payload
    assert "source_coverage" in payload


def test_source_level_ablation_does_not_write_enhanced_default_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    artifact_path = tmp_path / "candidate_items.parquet"
    artifact_path.write_text("existing artifact", encoding="utf-8")
    monkeypatch.setattr(
        "fashion_trend.recommendation.retrieval.candidates.candidate_items_path",
        lambda strategy: artifact_path,
    )
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "u1",
                "article_id": "a1",
                "candidate_sources": "trend|reorder",
                "primary_source": "trend",
                "best_source_rank": 1,
                "has_reorder_source": True,
                "allow_seen": True,
            }
        ]
    )

    filter_candidate_sources_for_ablation(
        candidates,
        dropped_sources={"trend"},
        strategy="enhanced_default",
    )

    assert artifact_path.read_text(encoding="utf-8") == "existing artifact"


def test_enhanced_candidate_build_receives_customer_and_product_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    inputs = _enhanced_input_artifacts()

    monkeypatch.setattr(
        experiment_runner,
        "candidate_items_path",
        lambda strategy: tmp_path / "candidates" / strategy / "candidate_items.parquet",
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_and_write_candidate_items",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        experiment_runner,
        "read_candidate_items",
        lambda path: pd.DataFrame(),
    )

    ensure_or_build_candidate_items(
        "enhanced_default",
        _enhanced_artifact_context(),
        inputs,
        force=True,
    )

    assert captured["customer_profile"] is inputs.customer_profile
    assert captured["article_product_map"] is inputs.article_product_map


def test_enhanced_feature_cache_build_receives_customer_and_product_artifacts(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    inputs = _enhanced_input_artifacts()

    monkeypatch.setattr(
        experiment_runner,
        "_feature_cache_manifest_can_merge",
        lambda: True,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_and_write_feature_cache_for_strategy",
        lambda **kwargs: captured.update(kwargs),
    )

    ensure_or_build_feature_cache_for_strategy(
        "enhanced_default",
        _enhanced_artifact_context(),
        inputs,
        pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "strategy": "enhanced_default",
                    "customer_id": "u1",
                    "article_id": "0000000001",
                }
            ]
        ),
        force=True,
    )

    assert captured["customer_profile"] is inputs.customer_profile
    assert captured["article_product_map"] is inputs.article_product_map


def test_enhanced_in_memory_weight_run_disables_method_backfill(
    monkeypatch,
) -> None:
    captured_backfill_modes: list[object] = []
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(
            [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
        ),
        target_users=pd.DataFrame(
            [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
        ),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(
            [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
        ),
    )
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "enhanced_default",
                "customer_id": "u1",
                "article_id": "0000000001",
            }
        ]
    )

    def fake_build_cached_result(**kwargs):
        captured_backfill_modes.append(kwargs["backfill_mode"])
        return SimpleNamespace(
            recommendations=pd.DataFrame(),
            recommendation_items=pd.DataFrame(),
            metadata={},
        )

    monkeypatch.setattr(
        experiment_runner,
        "build_cached_recommendation_result_for_window",
        fake_build_cached_result,
    )

    experiment_runner.build_recommendation_result_in_memory(
        method_name="enhanced_pop_similarity_trend",
        weights=dict(iter_enhanced_weight_grid()[0]),
        split_filter="valid",
        context=_enhanced_artifact_context(),
        inputs=inputs,
        candidates=candidates,
    )

    assert captured_backfill_modes == [None]


def test_recommendation_inputs_fresh_requires_optional_enhanced_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    path_names = {
        "TIME_WINDOWS_PATH": "time_windows.parquet",
        "TARGET_USERS_PATH": "target_users.parquet",
        "EVALUATION_LABELS_PATH": "evaluation_labels.parquet",
        "USER_PROFILE_PATH": "user_profile.parquet",
        "RECOMMEND_METADATA_PATH": "metadata.json",
        "CUSTOMER_PROFILE_PATH": "customer_profile.parquet",
        "ARTICLE_PRODUCT_MAP_PATH": "article_product_map.parquet",
    }
    for attr, name in path_names.items():
        monkeypatch.setattr(experiment_runner, attr, tmp_path / name)
    for attr in (
        "TIME_WINDOWS_PATH",
        "TARGET_USERS_PATH",
        "EVALUATION_LABELS_PATH",
        "USER_PROFILE_PATH",
        "RECOMMEND_METADATA_PATH",
    ):
        getattr(experiment_runner, attr).write_text("stub", encoding="utf-8")

    assert (
        experiment_runner._recommendation_inputs_are_fresh(_enhanced_artifact_context())
        is False
    )


def test_generated_run_id_is_safe_path_segment() -> None:
    run_id = generate_experiment_run_id()

    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", run_id)
    assert (
        experiment_run_dir("main", run_id)
        .as_posix()
        .endswith(f"/experiments/main/runs/{run_id}")
    )


def test_experiment_uses_method_default_candidate_strategy() -> None:
    assert candidate_strategy_for_method("global_popularity") is None
    assert candidate_strategy_for_method("recent_popularity") is None
    assert candidate_strategy_for_method("attribute_similarity") == "similarity"
    assert candidate_strategy_for_method("pop_similarity") == "default"
    assert candidate_strategy_for_method("pop_similarity_trend") == "default"


def test_experiment_runner_exposes_explicit_force_switches() -> None:
    signature = inspect.signature(run_recommendation_experiment)

    assert "force" not in signature.parameters
    assert "force_experiment" in signature.parameters
    assert "force_methods" in signature.parameters
    assert "force_cache" in signature.parameters
    assert "force_candidates" in signature.parameters
    assert "force_rebuild_all" in signature.parameters


def test_force_cache_marks_method_outputs_stale() -> None:
    from fashion_trend.recommendation.experiments.runner import should_rebuild_method

    decision = should_rebuild_method(
        method_name="pop_similarity",
        stale_reason=None,
        force_methods=(),
        force_cache=True,
        force_candidates=False,
        force_rebuild_all=False,
    )

    assert decision.rebuild is True
    assert decision.reason == "force-cache"


def test_force_candidates_marks_method_outputs_stale() -> None:
    from fashion_trend.recommendation.experiments.runner import should_rebuild_method

    decision = should_rebuild_method(
        method_name="pop_similarity",
        stale_reason=None,
        force_methods=(),
        force_cache=False,
        force_candidates=True,
        force_rebuild_all=False,
    )

    assert decision.rebuild is True
    assert decision.reason == "force-candidates"


def test_experiment_cli_parses_explicit_force_options() -> None:
    module = _load_experiment_cli_module()

    args = module.parse_args(
        [
            "--experiment",
            "main",
            "--force-experiment",
            "--force-method",
            "pop_similarity",
            "--force-method",
            "recent_popularity",
            "--force-cache",
            "--force-candidates",
            "--force-rebuild-all",
            "--force",
        ]
    )

    assert args.experiment == "main"
    assert args.force_experiment is True
    assert args.force_method == ["pop_similarity", "recent_popularity"]
    assert args.force_cache is True
    assert args.force_candidates is True
    assert args.force_rebuild_all is True
    assert args.force is True


def test_experiment_payload_records_force_status_and_timings() -> None:
    payload = experiment_runner.build_experiment_payload(
        "main",
        baseline_payloads=[{"method": "global_popularity", "metrics": {}}],
        search_results=[
            {
                "grid_index": 0,
                "weights": {
                    "pop_score": 0.4,
                    "sim_score": 0.3,
                    "trend_score": 0.2,
                    "recent_score": 0.1,
                },
                "valid_metrics": {"map_at_12": 0.1, "ndcg_at_12": 0.1},
            }
        ],
        trend_payload={"method": "pop_similarity_trend", "metrics": {}},
        stage_status=[
            {
                "stage": "method",
                "method": "pop_similarity",
                "status": "rebuilt",
                "reason": "force-cache",
            }
        ],
        force={
            "force_experiment": False,
            "force_methods": ["pop_similarity"],
            "force_cache": True,
            "force_candidates": False,
            "force_rebuild_all": False,
        },
        timings=[{"stage": "method", "elapsed_seconds": 0.01}],
    )

    assert payload["stage_status"] == [
        {
            "stage": "method",
            "method": "pop_similarity",
            "status": "rebuilt",
            "reason": "force-cache",
        }
    ]
    assert payload["force"]["force_cache"] is True
    assert payload["force"]["force_methods"] == ["pop_similarity"]
    assert payload["timings"] == [{"stage": "method", "elapsed_seconds": 0.01}]


def test_build_experiment_payload_includes_named_ablation_and_trend_buckets() -> None:
    payload = experiment_runner.build_experiment_payload(
        "main",
        baseline_payloads=[
            {
                "method": "recent_popularity",
                "metrics": {
                    "valid": {"ndcg_at_12": 0.1},
                    "test": {"ndcg_at_12": 0.2},
                },
            },
            {
                "method": "pop_similarity",
                "metrics": {
                    "valid": {"ndcg_at_12": 0.3},
                    "test": {"ndcg_at_12": 0.4},
                },
            },
        ],
        search_results=[
            {
                "grid_index": 0,
                "weights": {
                    "pop_score": 0.2,
                    "sim_score": 0.2,
                    "trend_score": 0.1,
                    "recent_score": 0.5,
                },
                "valid_metrics": {"ndcg_at_12": 0.7},
            }
        ],
        trend_payload={
            "method": "pop_similarity_trend",
            "metrics": {
                "valid": {"ndcg_at_12": 0.8},
                "test": {"ndcg_at_12": 0.9},
            },
        },
        named_ablation=[{"variant_id": "full_model"}],
        trend_bucket_best_by_valid=[{"variant_id": "trend_bucket_0_1"}],
    )

    assert payload["named_ablation"] == [{"variant_id": "full_model"}]
    assert payload["trend_bucket_best_by_valid"] == [{"variant_id": "trend_bucket_0_1"}]


def test_cached_search_results_require_current_grid(
    tmp_path,
    monkeypatch,
) -> None:
    experiment_root = tmp_path / "experiments" / "main"
    experiment_root.mkdir(parents=True)
    monkeypatch.setattr(
        experiment_runner,
        "experiment_dir",
        lambda experiment_id: experiment_root,
    )
    weight_grid = [
        {
            "pop_score": 0.2,
            "sim_score": 0.2,
            "trend_score": 0.1,
            "recent_score": 0.5,
        }
    ]
    payload = {
        "search_results": [
            {
                "grid_index": 0,
                "weights": dict(weight_grid[0]),
                "valid_metrics": {"ndcg_at_12": 0.7},
            }
        ]
    }
    (experiment_root / "experiment.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    cached = experiment_runner._cached_search_results_for_current_grid(
        "main",
        weight_grid,
    )

    assert cached == payload["search_results"]

    payload["search_results"][0]["weights"]["trend_score"] = 0.2
    (experiment_root / "experiment.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert (
        experiment_runner._cached_search_results_for_current_grid("main", weight_grid)
        is None
    )


def test_cached_search_results_require_search_cache_fingerprints(
    tmp_path,
    monkeypatch,
) -> None:
    experiment_root = tmp_path / "experiments" / "main"
    experiment_root.mkdir(parents=True)
    input_path = tmp_path / "trend_predictions.csv"
    input_path.write_text("current input", encoding="utf-8")
    monkeypatch.setattr(
        experiment_runner,
        "experiment_dir",
        lambda experiment_id: experiment_root,
    )
    weight_grid = [
        {
            "pop_score": 0.2,
            "sim_score": 0.2,
            "trend_score": 0.1,
            "recent_score": 0.5,
        }
    ]
    search_cache = experiment_runner._build_search_cache_metadata(
        weight_grid,
        {"trend_predictions": str(input_path)},
    )
    payload = {
        "search_cache": search_cache,
        "search_results": [
            {
                "grid_index": 0,
                "weights": dict(weight_grid[0]),
                "valid_metrics": {"ndcg_at_12": 0.7},
            }
        ],
    }
    (experiment_root / "experiment.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert (
        experiment_runner._cached_search_results_for_current_grid(
            "main",
            weight_grid,
            expected_search_cache=search_cache,
        )
        == payload["search_results"]
    )

    input_path.write_text("changed input contents", encoding="utf-8")
    changed_search_cache = experiment_runner._build_search_cache_metadata(
        weight_grid,
        {"trend_predictions": str(input_path)},
    )

    assert (
        experiment_runner._cached_search_results_for_current_grid(
            "main",
            weight_grid,
            expected_search_cache=changed_search_cache,
        )
        is None
    )


def test_search_cache_input_artifacts_include_feature_cache_paths(
    tmp_path,
    monkeypatch,
) -> None:
    feature_paths = [
        str(tmp_path / "features" / "trend_scores" / "part.parquet"),
        str(tmp_path / "features" / "trend_scores" / "metadata.json"),
    ]
    monkeypatch.setattr(
        experiment_runner,
        "feature_artifact_paths_for_method_window",
        lambda **kwargs: feature_paths,
    )
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(
            [
                {"split": "valid", "cutoff_week": 10, "label_week": 11},
                {"split": "test", "cutoff_week": 11, "label_week": 12},
            ]
        ),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "article_id": "0000000001",
            }
        ]
    )
    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        input_paths={
            "weekly_transactions": str(tmp_path / "weekly.parquet"),
            "article_attributes": str(tmp_path / "articles.csv"),
            "trend_predictions": str(tmp_path / "trend_predictions.csv"),
        },
    )

    artifacts = experiment_runner._search_cache_input_artifacts(
        context,
        inputs,
        candidates,
    )

    assert artifacts["trend_predictions"].endswith("trend_predictions.csv")
    assert artifacts["evaluation_labels"].endswith("evaluation_labels.parquet")
    assert artifacts["feature_artifact_0000"] == feature_paths[0]
    assert artifacts["feature_artifact_0001"] == feature_paths[1]


def test_search_cache_input_artifacts_reject_missing_window_columns(tmp_path) -> None:
    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        input_paths={
            "weekly_transactions": str(tmp_path / "weekly.parquet"),
            "article_attributes": str(tmp_path / "articles.csv"),
            "trend_predictions": str(tmp_path / "trend_predictions.csv"),
        },
    )
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame([{"split": "valid", "cutoff_week": 10}]),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    candidates = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "article_id": "0000000001"}]
    )

    with pytest.raises(ValueError, match="label_week"):
        experiment_runner._search_cache_input_artifacts(context, inputs, candidates)


@pytest.mark.parametrize(
    (
        "force_kwargs",
        "expected_reason",
        "expected_candidate_force",
        "expected_cache_force",
    ),
    [
        ({"force_methods": ("pop_similarity",)}, "force-method", False, False),
        ({"force_cache": True}, "force-cache", False, True),
        ({"force_candidates": True}, "force-candidates", True, True),
    ],
)
def test_baseline_force_rebuilds_fresh_cached_method_output(
    tmp_path,
    monkeypatch,
    force_kwargs: dict[str, object],
    expected_reason: str,
    expected_candidate_force: bool,
    expected_cache_force: bool,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    calls: list[tuple[str, object]] = []
    stage_status: list[dict[str, object]] = []
    candidates = _default_candidates_for_cache_test()

    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: _cached_method_current_inputs(paths),
    )
    monkeypatch.setattr(
        experiment_runner,
        "feature_artifact_paths_for_method_window",
        lambda **kwargs: paths.feature_artifacts,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_candidate_items_rebuild_required",
        lambda strategy, context, *, force: calls.append(("candidate_required", force))
        or bool(force),
    )
    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_candidate_items",
        lambda strategy, context, inputs, force=False: calls.append(
            ("candidates", force)
        )
        or candidates,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_feature_cache_partitions_exist",
        lambda strategy, candidates: True,
    )
    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_feature_cache_for_strategy",
        lambda strategy, context, inputs, candidates, force=False: calls.append(
            ("cache", force)
        ),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_recommendation_method_by_window",
        lambda **kwargs: calls.append(("method", kwargs["method_name"])),
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_method_output_for_experiment",
        lambda method, context, inputs, force=False: calls.append(("evaluate", force))
        or {"method": method, "metrics": {}},
    )

    payloads = run_baseline_methods(
        RecommendationExperimentContext(
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
            input_paths={},
        ),
        RecommendationInputArtifacts(
            time_windows=pd.DataFrame(),
            target_users=pd.DataFrame(),
            evaluation_labels=pd.DataFrame(),
            user_profile=pd.DataFrame(),
        ),
        stage_status=stage_status,
        **force_kwargs,
    )

    assert payloads == [{"method": "pop_similarity", "metrics": {}}]
    assert ("method", "pop_similarity") in calls
    assert ("candidates", expected_candidate_force) in calls
    assert ("cache", expected_cache_force) in calls
    assert {
        "stage": "method",
        "method": "pop_similarity",
        "status": "rebuilt",
        "reason": expected_reason,
    } in stage_status


def test_force_candidates_rebuilds_candidates_cache_and_methods(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(columns=["split", "cutoff_week", "label_week"]),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    candidates = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])

    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_recommendation_inputs",
        lambda context, force=False: calls.append(("inputs", force)) or inputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_baseline_methods",
        lambda context, inputs, **kwargs: calls.append(("baselines", kwargs))
        or _minimal_baseline_payloads(),
    )
    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_candidates_for_method",
        lambda method, context, inputs, force=False: calls.append(("candidates", force))
        or candidates,
    )
    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_feature_cache_for_strategy",
        lambda strategy, context, inputs, candidates, force=False: calls.append(
            ("cache", force)
        ),
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_weight_grid_on_valid",
        lambda weight_grid, context, inputs, candidates, **kwargs: calls.append(
            ("search", kwargs)
        )
        or [
            {
                "grid_index": 0,
                "weights": {
                    "pop_score": 0.4,
                    "sim_score": 0.3,
                    "trend_score": 0.2,
                    "recent_score": 0.1,
                },
                "valid_metrics": {"map_at_12": 0.1, "ndcg_at_12": 0.1},
            }
        ],
    )
    monkeypatch.setattr(
        experiment_runner,
        "publish_trend_method_with_weights",
        lambda weights, context, inputs, candidates, **kwargs: calls.append(
            ("trend_method", kwargs)
        )
        or _minimal_trend_payload(),
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_weight_variant_by_split",
        lambda **kwargs: _minimal_split_metrics(0.5),
    )
    monkeypatch.setattr(
        experiment_runner,
        "prepare_weight_variant_windows",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        experiment_runner,
        "select_trend_bucket_representatives",
        lambda search_results: [],
    )
    monkeypatch.setattr(
        experiment_runner,
        "write_json_atomic",
        lambda payload, path: calls.append(("payload", payload)),
    )

    payload = run_recommendation_experiment(
        RecommendationExperimentContext(
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
        ),
        force_candidates=True,
    )

    assert ("inputs", False) in calls
    assert ("candidates", True) in calls
    assert ("cache", True) in calls
    baseline_call = next(payload for stage, payload in calls if stage == "baselines")
    assert baseline_call["force_candidates"] is True
    assert baseline_call["rebuild_stale_outputs"] is True
    assert ("trend_method", {"force": False}) in calls
    assert payload["force"]["force_candidates"] is True
    assert [row["variant_id"] for row in payload["named_ablation"]] == [
        "full_model",
        "without_trend_in_rec",
        "without_similarity",
        "without_recent",
        "recent_only_baseline",
        "pop_similarity_baseline",
    ]
    assert {
        "stage": "candidates",
        "method": "pop_similarity_trend",
        "status": "rebuilt",
        "reason": "force-candidates",
    } in payload["stage_status"]


def test_experiment_records_missing_candidate_and_cache_as_rebuilt(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(columns=["split", "cutoff_week", "label_week"]),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    candidates = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])

    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_recommendation_inputs",
        lambda context, force=False: inputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_baseline_methods",
        lambda context, inputs, **kwargs: _minimal_baseline_payloads(),
    )
    monkeypatch.setattr(
        experiment_runner,
        "_candidate_items_rebuild_required",
        lambda strategy, context, *, force: True,
    )
    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_candidates_for_method",
        lambda method, context, inputs, force=False: calls.append(("candidates", force))
        or candidates,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_feature_cache_partitions_exist",
        lambda strategy, candidates: False,
    )
    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_feature_cache_for_strategy",
        lambda strategy, context, inputs, candidates, force=False: calls.append(
            ("cache", force)
        ),
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_weight_grid_on_valid",
        lambda *args, **kwargs: [
            {
                "grid_index": 0,
                "weights": {
                    "pop_score": 0.4,
                    "sim_score": 0.3,
                    "trend_score": 0.2,
                    "recent_score": 0.1,
                },
                "valid_metrics": {"map_at_12": 0.1, "ndcg_at_12": 0.1},
            }
        ],
    )
    monkeypatch.setattr(
        experiment_runner,
        "publish_trend_method_with_weights",
        lambda *args, **kwargs: _minimal_trend_payload(),
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_weight_variant_by_split",
        lambda **kwargs: _minimal_split_metrics(0.5),
    )
    monkeypatch.setattr(
        experiment_runner,
        "prepare_weight_variant_windows",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        experiment_runner,
        "select_trend_bucket_representatives",
        lambda search_results: [],
    )
    monkeypatch.setattr(
        experiment_runner, "write_json_atomic", lambda payload, path: None
    )

    payload = run_recommendation_experiment(
        RecommendationExperimentContext(
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
        )
    )

    assert ("candidates", True) in calls
    assert ("cache", True) in calls
    assert {
        "stage": "candidates",
        "method": "pop_similarity_trend",
        "status": "rebuilt",
        "reason": "stale-or-missing",
    } in payload["stage_status"]
    assert {
        "stage": "cache",
        "strategy": "default",
        "status": "rebuilt",
        "reason": "stale-or-missing",
    } in payload["stage_status"]


def _minimal_split_metrics(value: float) -> dict[str, dict[str, float]]:
    return {
        "valid": {"ndcg_at_12": value},
        "test": {"ndcg_at_12": value + 0.1},
    }


def _minimal_baseline_payloads() -> list[dict[str, object]]:
    return [
        {
            "method": "recent_popularity",
            "metrics": _minimal_split_metrics(0.1),
        },
        {
            "method": "pop_similarity",
            "metrics": _minimal_split_metrics(0.3),
        },
    ]


def _minimal_trend_payload() -> dict[str, object]:
    return {
        "method": "pop_similarity_trend",
        "metrics": _minimal_split_metrics(0.7),
    }


def test_experiment_rejects_unknown_force_method() -> None:
    with pytest.raises(ValueError, match="unknown force methods"):
        run_recommendation_experiment(
            RecommendationExperimentContext(
                transactions=pd.DataFrame(),
                article_attributes=pd.DataFrame(),
                trend_predictions=pd.DataFrame(),
            ),
            force_methods=("typo",),
        )


def test_experiment_rejects_stale_existing_method_output(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "weekly_transactions.parquet"
    input_path.write_text("current input", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "global_popularity"
    output_dir.mkdir(parents=True)
    recommendations_path = output_dir / "recommendations.csv"
    metadata_path = output_dir / "metadata.json"
    recommendations_path.write_text("existing output", encoding="utf-8")
    input_paths = {"weekly_transactions": str(input_path)}
    metadata_path.write_text(
        json.dumps(
            {
                "input_artifacts": input_paths,
                "input_fingerprints": {
                    "weekly_transactions": {
                        "path": str(input_path),
                        "exists": True,
                        "size_bytes": 0,
                        "mtime_ns": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("global_popularity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: SimpleNamespace(
            recommendations=recommendations_path,
            recommendation_items=output_dir / "recommendation_items.parquet",
            params=output_dir / "params.json",
            metadata=metadata_path,
        ),
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: input_paths,
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_method_output_for_experiment",
        lambda method, context, inputs, force=False: {"method": method, "metrics": {}},
    )

    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        input_paths=input_paths,
    )
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="recommendation_items.parquet is missing"):
        run_baseline_methods(context, inputs, force=False)


def test_experiment_rejects_method_output_missing_params(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "weekly_transactions.parquet"
    input_path.write_text("current input", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "global_popularity"
    output_dir.mkdir(parents=True)
    recommendations_path = output_dir / "recommendations.csv"
    items_path = output_dir / "recommendation_items.parquet"
    params_path = output_dir / "params.json"
    metadata_path = output_dir / "metadata.json"
    recommendations_path.write_text("existing output", encoding="utf-8")
    pd.DataFrame({"customer_id": ["u1"]}).to_parquet(items_path, index=False)
    metadata_path.write_text(json.dumps({"metadata": "old"}), encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("global_popularity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: SimpleNamespace(
            recommendations=recommendations_path,
            recommendation_items=items_path,
            params=params_path,
            metadata=metadata_path,
        ),
    )

    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        input_paths={"weekly_transactions": str(input_path)},
    )
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="params.json is missing"):
        run_baseline_methods(context, inputs, force=False)


def test_experiment_rejects_old_method_metadata_schema(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "weekly_transactions.parquet"
    input_path.write_text("current input", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "global_popularity"
    output_dir.mkdir(parents=True)
    recommendations_path = output_dir / "recommendations.csv"
    items_path = output_dir / "recommendation_items.parquet"
    params_path = output_dir / "params.json"
    metadata_path = output_dir / "metadata.json"
    recommendations_path.write_text("existing output", encoding="utf-8")
    pd.DataFrame({"customer_id": ["u1"]}).to_parquet(items_path, index=False)
    params_path.write_text(
        json.dumps({"method": "global_popularity"}), encoding="utf-8"
    )
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("global_popularity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: SimpleNamespace(
            recommendations=recommendations_path,
            recommendation_items=items_path,
            params=params_path,
            metadata=metadata_path,
        ),
    )

    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        input_paths={"weekly_transactions": str(input_path)},
    )
    method_input_paths = experiment_runner._method_input_paths(
        "global_popularity",
        experiment_runner._experiment_input_paths(context),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "input_artifacts": method_input_paths,
                "input_fingerprints": build_input_fingerprints(method_input_paths),
            }
        ),
        encoding="utf-8",
    )
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    with pytest.raises(
        RuntimeError,
        match="output_artifacts changed|schema_version changed|config changed",
    ):
        run_baseline_methods(context, inputs, force=False)


def test_experiment_rejects_method_output_with_include_seen_config(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "weekly_transactions.parquet"
    input_path.write_text("current input", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "global_popularity"
    output_dir.mkdir(parents=True)
    recommendations_path = output_dir / "recommendations.csv"
    items_path = output_dir / "recommendation_items.parquet"
    params_path = output_dir / "params.json"
    metadata_path = output_dir / "metadata.json"
    recommendations_path.write_text("existing output", encoding="utf-8")
    pd.DataFrame({"customer_id": ["u1"]}).to_parquet(items_path, index=False)
    params_path.write_text(
        json.dumps({"method": "global_popularity"}), encoding="utf-8"
    )
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("global_popularity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: SimpleNamespace(
            recommendations=recommendations_path,
            recommendation_items=items_path,
            params=params_path,
            metadata=metadata_path,
        ),
    )
    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        input_paths={"weekly_transactions": str(input_path)},
    )
    method_input_paths = experiment_runner._method_input_paths(
        "global_popularity",
        experiment_runner._experiment_input_paths(context),
    )
    metadata_path.write_text(
        json.dumps(
            build_artifact_metadata(
                name="recommendation_method",
                input_artifacts=method_input_paths,
                output_artifacts=experiment_runner._method_output_artifacts(
                    "global_popularity"
                ),
                schema_version=1,
                algorithm_version="recommendation-method-v1",
                config={
                    "method": "global_popularity",
                    "top_k": 12,
                    "candidate_strategy": None,
                    "exclude_seen": False,
                    "weights": {"pop_score": 1.0},
                },
                row_counts={
                    "recommendation_rows": 1,
                    "recommendation_item_rows": 1,
                },
            )
        ),
        encoding="utf-8",
    )
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="config changed"):
        run_baseline_methods(context, inputs, force=False)


def test_non_similarity_method_freshness_excludes_unrelated_inputs(tmp_path) -> None:
    available_paths = {
        "weekly_transactions": str(tmp_path / "weekly.parquet"),
        "article_attributes": str(tmp_path / "articles.csv"),
        "time_windows": str(tmp_path / "time_windows.parquet"),
        "target_users": str(tmp_path / "target_users.parquet"),
        "user_profile": str(tmp_path / "user_profile.parquet"),
        "evaluation_labels": str(tmp_path / "labels.parquet"),
        "default_candidates": str(tmp_path / "default.parquet"),
        "similarity_candidates": str(tmp_path / "similarity.parquet"),
    }

    method_paths = experiment_runner._method_input_paths(
        "global_popularity",
        available_paths,
    )
    recent_paths = experiment_runner._method_input_paths(
        "recent_popularity",
        available_paths,
    )

    expected = {
        "weekly_transactions": available_paths["weekly_transactions"],
        "time_windows": available_paths["time_windows"],
        "target_users": available_paths["target_users"],
    }
    assert method_paths == expected
    assert recent_paths == expected


def test_attribute_similarity_method_freshness_uses_profile_and_candidates(
    tmp_path,
) -> None:
    available_paths = {
        "weekly_transactions": str(tmp_path / "weekly.parquet"),
        "article_attributes": str(tmp_path / "articles.csv"),
        "time_windows": str(tmp_path / "time_windows.parquet"),
        "target_users": str(tmp_path / "target_users.parquet"),
        "user_profile": str(tmp_path / "user_profile.parquet"),
        "trend_predictions": str(tmp_path / "predictions.csv"),
        "similarity_candidates": str(tmp_path / "similarity.parquet"),
        "default_candidates": str(tmp_path / "default.parquet"),
    }

    method_paths = experiment_runner._method_input_paths(
        "attribute_similarity",
        available_paths,
    )

    assert method_paths == {
        "weekly_transactions": available_paths["weekly_transactions"],
        "article_attributes": available_paths["article_attributes"],
        "time_windows": available_paths["time_windows"],
        "target_users": available_paths["target_users"],
        "user_profile": available_paths["user_profile"],
        "candidate_items": available_paths["similarity_candidates"],
        "candidate_metadata": str(
            (tmp_path / "similarity.parquet").with_name("metadata.json")
        ),
    }


def test_similarity_feature_cache_inputs_expose_shared_feature_sources(
    tmp_path,
) -> None:
    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        input_paths={
            "weekly_transactions": str(tmp_path / "weekly.parquet"),
            "article_attributes": str(tmp_path / "attributes.csv"),
            "trend_predictions": str(tmp_path / "predictions.csv"),
        },
    )

    input_paths = experiment_runner._feature_cache_input_paths("similarity", context)

    assert input_paths["weekly_transactions"] == str(tmp_path / "weekly.parquet")
    assert input_paths["article_attributes"] == str(tmp_path / "attributes.csv")
    assert input_paths["trend_predictions"] == str(tmp_path / "predictions.csv")
    assert input_paths["candidate_items"].endswith(
        "data/processed/recommend/candidates/similarity/candidate_items.parquet"
    )
    assert input_paths["candidate_metadata"].endswith(
        "data/processed/recommend/candidates/similarity/metadata.json"
    )


def test_trend_method_freshness_uses_profile_candidates_and_predictions(
    tmp_path,
) -> None:
    available_paths = {
        "weekly_transactions": str(tmp_path / "weekly.parquet"),
        "article_attributes": str(tmp_path / "articles.csv"),
        "time_windows": str(tmp_path / "time_windows.parquet"),
        "target_users": str(tmp_path / "target_users.parquet"),
        "user_profile": str(tmp_path / "user_profile.parquet"),
        "trend_predictions": str(tmp_path / "predictions.csv"),
        "similarity_candidates": str(tmp_path / "similarity.parquet"),
        "default_candidates": str(tmp_path / "default.parquet"),
    }

    method_paths = experiment_runner._method_input_paths(
        "pop_similarity_trend",
        available_paths,
    )

    assert method_paths == {
        "weekly_transactions": available_paths["weekly_transactions"],
        "article_attributes": available_paths["article_attributes"],
        "time_windows": available_paths["time_windows"],
        "target_users": available_paths["target_users"],
        "user_profile": available_paths["user_profile"],
        "candidate_items": available_paths["default_candidates"],
        "candidate_metadata": str(
            (tmp_path / "default.parquet").with_name("metadata.json")
        ),
        "trend_predictions": available_paths["trend_predictions"],
    }


def test_experiment_reuses_fresh_cached_method_output_with_feature_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    current_inputs = {
        "weekly_transactions": str(paths.weekly_transactions),
        "article_attributes": str(paths.article_attributes),
        "time_windows": str(paths.time_windows),
        "target_users": str(paths.target_users),
        "user_profile": str(paths.user_profile),
        "default_candidates": str(paths.candidate_items),
        "default_candidate_metadata": str(paths.candidate_metadata),
        "feature_cache_metadata": str(paths.feature_cache_metadata),
    }
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: current_inputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "feature_artifact_paths_for_method_window",
        lambda **kwargs: paths.feature_artifacts,
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_method_output_for_experiment",
        lambda method, context, inputs, force=False: {"method": method, "metrics": {}},
    )

    payloads = run_baseline_methods(
        RecommendationExperimentContext(
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
            input_paths={},
        ),
        RecommendationInputArtifacts(
            time_windows=pd.DataFrame(),
            target_users=pd.DataFrame(),
            evaluation_labels=pd.DataFrame(),
            user_profile=pd.DataFrame(),
        ),
        force=False,
    )

    assert payloads == [{"method": "pop_similarity", "metrics": {}}]


def test_experiment_rejects_cached_method_output_when_feature_cache_changes(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    current_inputs = {
        "weekly_transactions": str(paths.weekly_transactions),
        "article_attributes": str(paths.article_attributes),
        "time_windows": str(paths.time_windows),
        "target_users": str(paths.target_users),
        "user_profile": str(paths.user_profile),
        "default_candidates": str(paths.candidate_items),
        "default_candidate_metadata": str(paths.candidate_metadata),
        "feature_cache_metadata": str(paths.feature_cache_metadata),
    }
    paths.feature_partition.write_text("updated cache", encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: current_inputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "feature_artifact_paths_for_method_window",
        lambda **kwargs: paths.feature_artifacts,
    )

    with pytest.raises(RuntimeError, match="input_fingerprints changed"):
        run_baseline_methods(
            RecommendationExperimentContext(
                transactions=pd.DataFrame(),
                article_attributes=pd.DataFrame(),
                trend_predictions=pd.DataFrame(),
                input_paths={},
            ),
            RecommendationInputArtifacts(
                time_windows=pd.DataFrame(),
                target_users=pd.DataFrame(),
                evaluation_labels=pd.DataFrame(),
                user_profile=pd.DataFrame(),
            ),
            force=False,
        )


def test_experiment_rejects_cached_method_output_missing_required_partition(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    metadata = json.loads(paths.method_outputs.metadata.read_text(encoding="utf-8"))
    metadata["input_artifacts"].pop("feature_partition_0002")
    paths.method_outputs.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    current_inputs = {
        "weekly_transactions": str(paths.weekly_transactions),
        "article_attributes": str(paths.article_attributes),
        "time_windows": str(paths.time_windows),
        "target_users": str(paths.target_users),
        "user_profile": str(paths.user_profile),
        "default_candidates": str(paths.candidate_items),
        "default_candidate_metadata": str(paths.candidate_metadata),
        "feature_cache_metadata": str(paths.feature_cache_metadata),
    }
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: current_inputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "feature_artifact_paths_for_method_window",
        lambda **kwargs: paths.feature_artifacts,
    )

    with pytest.raises(RuntimeError, match="missing feature partitions"):
        run_baseline_methods(
            RecommendationExperimentContext(
                transactions=pd.DataFrame(),
                article_attributes=pd.DataFrame(),
                trend_predictions=pd.DataFrame(),
                input_paths={},
            ),
            RecommendationInputArtifacts(
                time_windows=pd.DataFrame(),
                target_users=pd.DataFrame(),
                evaluation_labels=pd.DataFrame(),
                user_profile=pd.DataFrame(),
            ),
            force=False,
        )


def test_experiment_rejects_cached_method_output_missing_partition_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    metadata = json.loads(paths.method_outputs.metadata.read_text(encoding="utf-8"))
    metadata["input_artifacts"].pop("feature_partition_0003")
    paths.method_outputs.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: _cached_method_current_inputs(paths),
    )
    monkeypatch.setattr(
        experiment_runner,
        "feature_artifact_paths_for_method_window",
        lambda **kwargs: paths.feature_artifacts,
    )

    with pytest.raises(RuntimeError, match="missing feature partitions"):
        run_baseline_methods(
            RecommendationExperimentContext(
                transactions=pd.DataFrame(),
                article_attributes=pd.DataFrame(),
                trend_predictions=pd.DataFrame(),
                input_paths={},
            ),
            RecommendationInputArtifacts(
                time_windows=pd.DataFrame(),
                target_users=pd.DataFrame(),
                evaluation_labels=pd.DataFrame(),
                user_profile=pd.DataFrame(),
            ),
            force=False,
        )


def test_experiment_rejects_cached_method_output_missing_window_summaries(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    metadata = json.loads(paths.method_outputs.metadata.read_text(encoding="utf-8"))
    metadata.pop("window_summaries")
    paths.method_outputs.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: _cached_method_current_inputs(paths),
    )

    with pytest.raises(RuntimeError, match="window_summaries missing"):
        run_baseline_methods(
            RecommendationExperimentContext(
                transactions=pd.DataFrame(),
                article_attributes=pd.DataFrame(),
                trend_predictions=pd.DataFrame(),
                input_paths={},
            ),
            RecommendationInputArtifacts(
                time_windows=pd.DataFrame(),
                target_users=pd.DataFrame(),
                evaluation_labels=pd.DataFrame(),
                user_profile=pd.DataFrame(),
            ),
            force=False,
        )


def test_experiment_rejects_cached_method_output_window_missing_candidate_rows(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    metadata = json.loads(paths.method_outputs.metadata.read_text(encoding="utf-8"))
    metadata["window_summaries"][0].pop("candidate_rows")
    paths.method_outputs.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: _cached_method_current_inputs(paths),
    )

    with pytest.raises(RuntimeError, match="candidate_rows"):
        run_baseline_methods(
            RecommendationExperimentContext(
                transactions=pd.DataFrame(),
                article_attributes=pd.DataFrame(),
                trend_predictions=pd.DataFrame(),
                input_paths={},
            ),
            RecommendationInputArtifacts(
                time_windows=pd.DataFrame(),
                target_users=pd.DataFrame(),
                evaluation_labels=pd.DataFrame(),
                user_profile=pd.DataFrame(),
            ),
            force=False,
        )


def test_experiment_rejects_invalid_method_metadata_json(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    paths.method_outputs.metadata.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: _cached_method_current_inputs(paths),
    )

    with pytest.raises(RuntimeError, match="metadata is invalid"):
        run_baseline_methods(
            RecommendationExperimentContext(
                transactions=pd.DataFrame(),
                article_attributes=pd.DataFrame(),
                trend_predictions=pd.DataFrame(),
                input_paths={},
            ),
            RecommendationInputArtifacts(
                time_windows=pd.DataFrame(),
                target_users=pd.DataFrame(),
                evaluation_labels=pd.DataFrame(),
                user_profile=pd.DataFrame(),
            ),
            force=False,
        )


def test_experiment_rejects_non_object_method_metadata_json(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _write_cached_method_output_fixture(tmp_path, "pop_similarity")
    paths.method_outputs.metadata.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "method_output_paths",
        lambda method: paths.method_outputs,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: _cached_method_current_inputs(paths),
    )

    with pytest.raises(RuntimeError, match="metadata is invalid"):
        run_baseline_methods(
            RecommendationExperimentContext(
                transactions=pd.DataFrame(),
                article_attributes=pd.DataFrame(),
                trend_predictions=pd.DataFrame(),
                input_paths={},
            ),
            RecommendationInputArtifacts(
                time_windows=pd.DataFrame(),
                target_users=pd.DataFrame(),
                evaluation_labels=pd.DataFrame(),
                user_profile=pd.DataFrame(),
            ),
            force=False,
        )


def test_recommendable_pool_cache_excludes_feature_manifest_from_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_experiment_feature_cache_paths(tmp_path, monkeypatch)
    context = _cache_build_context(tmp_path)
    inputs = _cache_build_inputs()

    first = experiment_runner.ensure_or_build_recommendable_pool_cache(
        context,
        inputs,
        force=False,
    )
    manifest = json.loads((tmp_path / "features" / "metadata.json").read_text())
    entry = manifest["entries"]["feature:recommendable_pool:strategy:all"]

    assert "feature_cache_metadata" not in entry["input_artifacts"]

    def fail_rewrite(*args, **kwargs):
        raise AssertionError("fresh recommendable_pool cache must not rebuild")

    monkeypatch.setattr(
        experiment_runner,
        "write_recommendable_pool_cache",
        fail_rewrite,
    )

    second = experiment_runner.ensure_or_build_recommendable_pool_cache(
        context,
        inputs,
        force=False,
    )

    assert first.to_dict("records") == second.to_dict("records")


@pytest.mark.parametrize("manifest_text", ["{invalid", "[]"])
def test_recommendable_pool_cache_rebuilds_invalid_global_manifest(
    tmp_path,
    monkeypatch,
    manifest_text: str,
) -> None:
    _patch_experiment_feature_cache_paths(tmp_path, monkeypatch)
    manifest_path = tmp_path / "features" / "metadata.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    experiment_runner.ensure_or_build_recommendable_pool_cache(
        _cache_build_context(tmp_path),
        _cache_build_inputs(),
        force=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "feature:recommendable_pool:strategy:all" in manifest["entries"]


def test_recommendable_pool_cache_preserves_valid_global_manifest_entries(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_experiment_feature_cache_paths(tmp_path, monkeypatch)
    manifest_path = tmp_path / "features" / "metadata.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"entries": {"strategy:default": {"feature_count": 5}}}),
        encoding="utf-8",
    )

    experiment_runner.ensure_or_build_recommendable_pool_cache(
        _cache_build_context(tmp_path),
        _cache_build_inputs(),
        force=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "strategy:default" in manifest["entries"]
    assert "feature:recommendable_pool:strategy:all" in manifest["entries"]


def test_experiment_rejects_stale_candidate_when_method_output_missing(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "weekly_transactions.parquet"
    input_path.write_text("current input", encoding="utf-8")
    candidate_path = tmp_path / "candidates" / "popularity" / "candidate_items.parquet"
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            (
                "valid",
                10,
                11,
                "popularity",
                "0000001",
                "0000000001",
                "popularity",
                "popularity",
                1,
            )
        ],
        columns=CANDIDATE_ITEM_COLUMNS,
    ).to_parquet(candidate_path, index=False)
    candidate_path.with_name("metadata.json").write_text(
        json.dumps(
            {
                "strategy": "popularity",
                "input_artifacts": {"weekly_transactions": str(input_path)},
                "input_fingerprints": {
                    "weekly_transactions": {
                        "path": str(input_path),
                        "exists": True,
                        "size_bytes": 0,
                        "mtime_ns": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    input_paths = {"weekly_transactions": str(input_path)}
    monkeypatch.setattr(
        experiment_runner,
        "candidate_items_path",
        lambda strategy: candidate_path,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_experiment_input_paths",
        lambda context: input_paths,
    )

    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        input_paths=input_paths,
    )
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="--force-candidates"):
        ensure_or_build_candidate_items(
            "popularity",
            context,
            inputs,
            force=False,
        )


def test_run_baseline_methods_builds_missing_feature_cache_before_cached_method(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_experiment_feature_cache_paths(tmp_path, monkeypatch)
    candidates = _default_candidates_for_cache_test()
    context = _cache_build_context(tmp_path)
    inputs = _cache_build_inputs()
    observed = {}

    monkeypatch.setattr(experiment_runner, "BASELINE_METHODS", ("pop_similarity",))
    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_candidates_for_method",
        lambda *args, **kwargs: candidates,
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_recommendation_method_by_window",
        lambda **kwargs: observed.setdefault(
            "cache_exists",
            (
                tmp_path
                / "features"
                / "popularity_scores"
                / "strategy=default"
                / "split=valid"
                / "cutoff_week=10"
                / "part.parquet"
            ).exists(),
        ),
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_method_output_for_experiment",
        lambda method, context, inputs, force=False: {"method": method, "metrics": {}},
    )

    payloads = run_baseline_methods(context, inputs, force=True)

    assert payloads == [{"method": "pop_similarity", "metrics": {}}]
    assert observed["cache_exists"] is True


def test_ensure_feature_cache_builds_from_empty_cache(tmp_path, monkeypatch) -> None:
    _patch_experiment_feature_cache_paths(tmp_path, monkeypatch)

    ensure_or_build_feature_cache_for_strategy(
        "default",
        _cache_build_context(tmp_path),
        _cache_build_inputs(),
        _default_candidates_for_cache_test(),
        force=False,
    )

    assert (
        tmp_path
        / "features"
        / "similarity_scores"
        / "strategy=default"
        / "split=valid"
        / "cutoff_week=10"
        / "part.parquet"
    ).exists()


@pytest.mark.parametrize("manifest_text", ["{invalid", "[]"])
def test_ensure_feature_cache_rebuilds_invalid_global_manifest(
    tmp_path,
    monkeypatch,
    manifest_text: str,
) -> None:
    _patch_experiment_feature_cache_paths(tmp_path, monkeypatch)
    manifest_path = tmp_path / "features" / "metadata.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    ensure_or_build_feature_cache_for_strategy(
        "default",
        _cache_build_context(tmp_path),
        _cache_build_inputs(),
        _default_candidates_for_cache_test(),
        force=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "strategy:default" in manifest["entries"]


def test_ensure_feature_cache_rebuilds_manifest_missing_strategy_entry(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_experiment_feature_cache_paths(tmp_path, monkeypatch)
    manifest_path = tmp_path / "features" / "metadata.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"entries": {"strategy:similarity": {}}}),
        encoding="utf-8",
    )

    ensure_or_build_feature_cache_for_strategy(
        "default",
        _cache_build_context(tmp_path),
        _cache_build_inputs(),
        _default_candidates_for_cache_test(),
        force=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "strategy:default" in manifest["entries"]
    assert "strategy:similarity" in manifest["entries"]


def test_valid_weight_search_uses_cached_feature_partitions(
    tmp_path, monkeypatch
) -> None:
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text("predictions", encoding="utf-8")
    cache = _write_trend_method_cache_partitions(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "fashion_trend.recommendation.methods.baselines.global_popularity"
        ".build_ranking_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid weight search must use cached features")
        ),
    )
    inputs = RecommendationInputArtifacts(
        time_windows=pd.DataFrame(
            [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
        ),
        target_users=pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "customer_id": "u1",
                }
            ]
        ),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "customer_id": "u1",
                }
            ]
        ),
    )
    context = RecommendationExperimentContext(
        transactions=pd.DataFrame(
            {
                "customer_id": ["u2"],
                "article_id": ["0000000003"],
                "week_id": [9],
            }
        ),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(
            {
                "split": ["valid"],
                "week_id": [10],
                "attr_id": [1],
                "attr_type": ["product_type_name"],
                "attr_value": ["Dress"],
                "pred_target_growth": [1.0],
            }
        ),
        input_paths={"trend_predictions": str(predictions_path)},
        trend_model_source=str(predictions_path),
    )

    result = experiment_runner.build_recommendation_result_in_memory(
        method_name="pop_similarity_trend",
        weights={
            "pop_score": 0.35,
            "sim_score": 0.35,
            "trend_score": 0.25,
            "recent_score": 0.05,
        },
        split_filter="valid",
        context=context,
        inputs=inputs,
        candidates=cache.candidates,
    )

    assert result.recommendation_items["article_id"].iloc[0] == "0000000001"


def _write_cached_method_output_fixture(tmp_path, method: str) -> SimpleNamespace:
    weekly_transactions = tmp_path / "weekly_transactions.parquet"
    article_attributes = tmp_path / "article_attributes.csv"
    time_windows = tmp_path / "time_windows.parquet"
    target_users = tmp_path / "target_users.parquet"
    user_profile = tmp_path / "user_profile.parquet"
    candidate_items = tmp_path / "candidates" / "default" / "candidate_items.parquet"
    candidate_metadata = candidate_items.with_name("metadata.json")
    feature_cache_metadata = tmp_path / "features" / "metadata.json"
    feature_artifacts = [
        tmp_path / "features" / feature_name / filename
        for feature_name in (
            "candidate_seen_flags",
            "popularity_scores",
            "recent_scores",
            "similarity_scores",
        )
        for filename in ("part.parquet", "metadata.json")
    ]
    for path in (
        weekly_transactions,
        article_attributes,
        time_windows,
        target_users,
        user_profile,
        candidate_items,
        candidate_metadata,
        feature_cache_metadata,
        *feature_artifacts,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    output_dir = tmp_path / "outputs" / method
    output_dir.mkdir(parents=True)
    method_outputs = SimpleNamespace(
        recommendations=output_dir / "recommendations.csv",
        recommendation_items=output_dir / "recommendation_items.parquet",
        params=output_dir / "params.json",
        metadata=output_dir / "metadata.json",
    )
    method_outputs.recommendations.write_text("existing output", encoding="utf-8")
    pd.DataFrame({"customer_id": ["u1"]}).to_parquet(
        method_outputs.recommendation_items,
        index=False,
    )
    method_outputs.params.write_text(json.dumps({"method": method}), encoding="utf-8")

    base_inputs = {
        "weekly_transactions": str(weekly_transactions),
        "article_attributes": str(article_attributes),
        "time_windows": str(time_windows),
        "target_users": str(target_users),
        "user_profile": str(user_profile),
        "candidate_items": str(candidate_items),
        "candidate_metadata": str(candidate_metadata),
        "feature_cache_metadata": str(feature_cache_metadata),
        **{
            f"feature_partition_{index:04d}": str(path)
            for index, path in enumerate(feature_artifacts)
        },
    }
    method_outputs.metadata.write_text(
        json.dumps(
            build_artifact_metadata(
                name=f"recommendation_method:{method}",
                input_artifacts=base_inputs,
                output_artifacts={
                    "recommendations": str(method_outputs.recommendations),
                    "recommendation_items": str(method_outputs.recommendation_items),
                    "params": str(method_outputs.params),
                    "metadata": str(method_outputs.metadata),
                },
                schema_version=1,
                algorithm_version="recommendation-method-v1",
                config={
                    "method": method,
                    "top_k": 12,
                    "candidate_strategy": "default",
                    "exclude_seen": True,
                    "weights": {
                        "pop_score": 0.45,
                        "sim_score": 0.45,
                        "recent_score": 0.10,
                    },
                },
                row_counts={"recommendation_rows": 1},
            )
            | {
                "window_summaries": [
                    {
                        "split": "valid",
                        "cutoff_week": 10,
                        "label_week": 11,
                        "candidate_rows": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        weekly_transactions=weekly_transactions,
        article_attributes=article_attributes,
        time_windows=time_windows,
        target_users=target_users,
        user_profile=user_profile,
        candidate_items=candidate_items,
        candidate_metadata=candidate_metadata,
        feature_cache_metadata=feature_cache_metadata,
        feature_partition=feature_artifacts[0],
        feature_artifacts=[str(path) for path in feature_artifacts],
        method_outputs=method_outputs,
    )


def _load_experiment_cli_module():
    module_name = "_recommendation_experiment_cli_for_tests"
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / ("16_run_recommendation_experiment.py")
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_experiment_feature_cache_paths(tmp_path, monkeypatch) -> None:
    feature_root = tmp_path / "features"
    manifest_path = feature_root / "metadata.json"

    def partition_path(feature_name, strategy, split, cutoff_week):
        return (
            feature_root
            / feature_name
            / f"strategy={strategy}"
            / f"split={split}"
            / f"cutoff_week={cutoff_week}"
            / "part.parquet"
        )

    def partition_metadata_path(feature_name, strategy, split, cutoff_week):
        return partition_path(
            feature_name,
            strategy,
            split,
            cutoff_week,
        ).with_name("metadata.json")

    monkeypatch.setattr(
        experiment_runner,
        "FEATURE_CACHE_METADATA_PATH",
        manifest_path,
    )
    monkeypatch.setattr(
        experiment_runner,
        "feature_cache_partition_path",
        partition_path,
    )
    monkeypatch.setattr(
        experiment_runner,
        "feature_cache_partition_metadata_path",
        partition_metadata_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.FEATURE_CACHE_METADATA_PATH",
        manifest_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache.feature_cache_partition_path",
        partition_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.features.cache."
        "feature_cache_partition_metadata_path",
        partition_metadata_path,
    )


def _cache_build_context(tmp_path) -> RecommendationExperimentContext:
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text("predictions", encoding="utf-8")
    return RecommendationExperimentContext(
        transactions=pd.DataFrame(
            {
                "customer_id": ["u1", "u2"],
                "article_id": ["0000000002", "0000000001"],
                "week_id": [9, 9],
            }
        ),
        article_attributes=pd.DataFrame(
            {
                "article_id": ["0000000001", "0000000002"],
                "attr_type": ["product_type_name", "product_type_name"],
                "attr_value": ["Dress", "Shirt"],
            }
        ),
        trend_predictions=pd.DataFrame(
            {
                "split": ["valid"],
                "week_id": [10],
                "attr_id": [1],
                "attr_type": ["product_type_name"],
                "attr_value": ["Dress"],
                "pred_target_growth": [1.0],
            }
        ),
        input_paths={"trend_predictions": str(predictions_path)},
        trend_model_source=str(predictions_path),
    )


def _enhanced_artifact_context() -> RecommendationExperimentContext:
    return RecommendationExperimentContext(
        transactions=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        trend_predictions=pd.DataFrame(),
        customers=pd.DataFrame(),
        clean_articles=pd.DataFrame(),
        input_paths={
            "weekly_transactions": "data/interim/transactions_train_weekly.parquet",
            "article_attributes": "data/processed/graph/edges_article_attribute.csv",
            "trend_predictions": "outputs/models/lightgbm/predictions.csv",
            "raw_customers": "data/raw/customers.csv",
            "clean_articles": "data/interim/articles_clean.csv",
        },
    )


def _enhanced_input_artifacts() -> RecommendationInputArtifacts:
    return RecommendationInputArtifacts(
        time_windows=pd.DataFrame(
            [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
        ),
        target_users=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        customer_profile=pd.DataFrame(
            [{"customer_id": "u1", "age": 25, "age_bucket": "20-29"}]
        ),
        article_product_map=pd.DataFrame(
            [{"article_id": "0000000001", "product_code": "123"}]
        ),
    )


def _cache_build_inputs() -> RecommendationInputArtifacts:
    return RecommendationInputArtifacts(
        time_windows=pd.DataFrame(
            [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
        ),
        target_users=pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "customer_id": "u1",
                }
            ]
        ),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "customer_id": "u1",
                    "attr_type": "product_type_name",
                    "attr_value": "Dress",
                    "preference_score": 1.0,
                }
            ]
        ),
    )


def _default_candidates_for_cache_test() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "default",
                "customer_id": "u1",
                "article_id": "0000000001",
                "candidate_sources": "popularity|similarity|trend",
                "primary_source": "similarity",
                "best_source_rank": 1,
            }
        ],
        columns=CANDIDATE_ITEM_COLUMNS,
    )


def _cached_method_current_inputs(paths: SimpleNamespace) -> dict[str, str]:
    return {
        "weekly_transactions": str(paths.weekly_transactions),
        "article_attributes": str(paths.article_attributes),
        "time_windows": str(paths.time_windows),
        "target_users": str(paths.target_users),
        "user_profile": str(paths.user_profile),
        "default_candidates": str(paths.candidate_items),
        "default_candidate_metadata": str(paths.candidate_metadata),
        "feature_cache_metadata": str(paths.feature_cache_metadata),
    }


def _write_trend_method_cache_partitions(tmp_path, monkeypatch) -> SimpleNamespace:
    feature_root = tmp_path / "features"
    candidates = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "default",
                "customer_id": "u1",
                "article_id": "0000000001",
                "candidate_sources": "popularity|similarity|trend",
                "primary_source": "trend",
                "best_source_rank": 1,
            }
        ],
        columns=CANDIDATE_ITEM_COLUMNS,
    )

    def partition_path(feature_name, strategy, split, cutoff_week):
        return (
            feature_root
            / feature_name
            / f"strategy={strategy}"
            / f"split={split}"
            / f"cutoff_week={cutoff_week}"
            / "part.parquet"
        )

    def metadata_path(feature_name, strategy, split, cutoff_week):
        return partition_path(
            feature_name,
            strategy,
            split,
            cutoff_week,
        ).with_name("metadata.json")

    monkeypatch.setattr(
        "fashion_trend.recommendation.runner.feature_cache_partition_path",
        partition_path,
    )
    monkeypatch.setattr(
        "fashion_trend.recommendation.runner.feature_cache_partition_metadata_path",
        metadata_path,
    )

    window = {
        "split": "valid",
        "cutoff_week": 10,
        "label_week": 11,
        "strategy": "default",
    }
    score_frames = {
        "candidate_seen_flags": candidates.iloc[0:0].assign(seen=True),
        "popularity_scores": candidates.loc[
            :, ["split", "cutoff_week", "label_week", "strategy", "article_id"]
        ].assign(pop_score=0.8),
        "recent_scores": candidates.loc[
            :, ["split", "cutoff_week", "label_week", "strategy", "article_id"]
        ].assign(recent_score=0.7),
        "trend_scores": candidates.loc[
            :, ["split", "cutoff_week", "label_week", "strategy", "article_id"]
        ].assign(trend_score=0.9),
        "similarity_scores": candidates.loc[
            :,
            [
                "split",
                "cutoff_week",
                "label_week",
                "strategy",
                "customer_id",
                "article_id",
            ],
        ].assign(sim_score=0.6),
    }
    for feature_name, frame in score_frames.items():
        parquet_path = partition_path(
            feature_name,
            "default",
            "valid",
            10,
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet_path, index=False)
        metadata_file = metadata_path(
            feature_name,
            "default",
            "valid",
            10,
        )
        metadata_file.write_text(
            json.dumps(
                build_artifact_metadata(
                    name=f"recommendation_feature_cache_{feature_name}",
                    input_artifacts=_feature_cache_test_input_artifacts(
                        feature_name,
                        tmp_path,
                    ),
                    output_artifacts={
                        "partition": str(parquet_path),
                        "partition_metadata": str(metadata_file),
                    },
                    schema_version=1,
                    algorithm_version="recommendation-feature-cache-v1",
                    config={
                        "feature_name": feature_name,
                        "strategy": "default",
                        "split": "valid",
                        "cutoff_week": 10,
                        "label_week": 11,
                    },
                    row_counts={"rows": int(len(frame))},
                )
            ),
            encoding="utf-8",
        )

    return SimpleNamespace(candidates=candidates, window=window)


def _feature_cache_test_input_artifacts(
    feature_name: str,
    tmp_path,
) -> dict[str, str]:
    input_artifacts = {
        "candidate_items": str(experiment_runner.candidate_items_path("default")),
    }
    if feature_name == "similarity_scores":
        input_artifacts["user_profile"] = str(experiment_runner.USER_PROFILE_PATH)
    if feature_name == "trend_scores":
        input_artifacts["trend_predictions"] = str(tmp_path / "predictions.csv")
    return input_artifacts


@pytest.mark.parametrize("bad", ["", ".", "..", "main/evil", "main\\evil"])
def test_experiment_id_rejects_unsafe_path_segments(bad: str) -> None:
    with pytest.raises(ValueError, match="安全"):
        experiment_run_dir(bad, "20260510-120000-1234abcd")


def test_build_ablation_summary_flattens_and_sorts_by_split_then_method() -> None:
    summary = build_ablation_summary(
        [
            {
                "method": "recent_popularity",
                "metrics": {"test": {"map_at_12": 0.2}},
            },
            {
                "method": "global_popularity",
                "metrics": {
                    "valid": {"map_at_12": 0.1},
                    "test": {"map_at_12": 0.3},
                },
            },
        ]
    )

    assert summary == [
        {"method": "global_popularity", "split": "test", "map_at_12": 0.3},
        {"method": "recent_popularity", "split": "test", "map_at_12": 0.2},
        {"method": "global_popularity", "split": "valid", "map_at_12": 0.1},
    ]


def test_drop_and_renormalize_weights_derive_from_best_weights() -> None:
    from fashion_trend.recommendation.experiments.ablation import (
        derive_strict_ablation_weights,
    )

    best_weights = {
        "pop_score": 0.2,
        "sim_score": 0.2,
        "trend_score": 0.1,
        "recent_score": 0.5,
    }

    without_trend = derive_strict_ablation_weights(best_weights, "trend_score")
    assert without_trend == pytest.approx(
        {
            "pop_score": 2 / 9,
            "sim_score": 2 / 9,
            "trend_score": 0.0,
            "recent_score": 5 / 9,
        }
    )

    without_similarity = derive_strict_ablation_weights(best_weights, "sim_score")
    assert without_similarity == pytest.approx(
        {
            "pop_score": 0.25,
            "sim_score": 0.0,
            "trend_score": 0.125,
            "recent_score": 0.625,
        }
    )

    without_recent = derive_strict_ablation_weights(best_weights, "recent_score")
    assert without_recent == pytest.approx(
        {
            "pop_score": 0.4,
            "sim_score": 0.4,
            "trend_score": 0.2,
            "recent_score": 0.0,
        }
    )


def test_drop_and_renormalize_rejects_invalid_weights() -> None:
    from fashion_trend.recommendation.experiments.ablation import (
        derive_strict_ablation_weights,
    )

    with pytest.raises(ValueError, match="best_weights"):
        derive_strict_ablation_weights(
            {"pop_score": 0.0, "trend_score": 0.0},
            "trend_score",
        )

    with pytest.raises(ValueError, match="unknown_score"):
        derive_strict_ablation_weights({"pop_score": 1.0}, "unknown_score")


def test_build_named_ablation_rows_keeps_audit_fields_and_selection_split() -> None:
    from fashion_trend.recommendation.experiments.ablation import (
        build_named_ablation_rows,
    )

    best_weights = {
        "pop_score": 0.2,
        "sim_score": 0.2,
        "trend_score": 0.1,
        "recent_score": 0.5,
    }
    rows = build_named_ablation_rows(
        best_weights=best_weights,
        strict_metrics={
            "without_trend_in_rec": {
                "valid": {"ndcg_at_12": 0.1},
                "test": {"ndcg_at_12": 0.2},
            },
            "without_similarity": {
                "valid": {"ndcg_at_12": 0.3},
                "test": {"ndcg_at_12": 0.4},
            },
            "without_recent": {
                "valid": {"ndcg_at_12": 0.5},
                "test": {"ndcg_at_12": 0.6},
            },
        },
        full_model_metrics={
            "valid": {
                "ndcg_at_12": 0.7,
                "coverage_by_window": [{"cutoff_week": 88, "coverage": 0.01}],
            },
            "test": {"ndcg_at_12": 0.8},
        },
        stable_baseline_metrics={
            "recent_only_baseline": {
                "method": "recent_popularity",
                "display_name": "Recent Only",
                "metrics": {
                    "valid": {"ndcg_at_12": 0.9},
                    "test": {"ndcg_at_12": 1.0},
                },
            },
            "pop_similarity_baseline": {
                "method": "pop_similarity",
                "display_name": "Pop + Similarity baseline",
                "metrics": {
                    "valid": {"ndcg_at_12": 1.1},
                    "test": {"ndcg_at_12": 1.2},
                },
            },
        },
    )

    by_id = {row["variant_id"]: row for row in rows}
    assert tuple(by_id) == (
        "full_model",
        "without_trend_in_rec",
        "without_similarity",
        "without_recent",
        "recent_only_baseline",
        "pop_similarity_baseline",
    )
    assert by_id["without_trend_in_rec"]["selection_split"] == "valid"
    assert by_id["without_trend_in_rec"]["weight_policy"] == (
        "strict_drop_and_renormalize_from_full"
    )
    assert by_id["without_trend_in_rec"]["weights"] == pytest.approx(
        {
            "pop_score": 2 / 9,
            "sim_score": 2 / 9,
            "trend_score": 0.0,
            "recent_score": 5 / 9,
        }
    )
    assert by_id["full_model"]["metrics"]["valid"]["coverage_by_window"] == [
        {"cutoff_week": 88, "coverage": 0.01}
    ]
    assert by_id["recent_only_baseline"]["selection_split"] == "not_applicable"
    assert by_id["recent_only_baseline"]["weight_policy"] == "stable_method_baseline"


def test_select_trend_bucket_best_by_valid_uses_ndcg_and_grid_order() -> None:
    from fashion_trend.recommendation.experiments.ablation import (
        select_trend_bucket_representatives,
    )

    rows = select_trend_bucket_representatives(
        [
            {
                "grid_index": 2,
                "weights": {
                    "pop_score": 0.3,
                    "sim_score": 0.2,
                    "trend_score": 0.1,
                    "recent_score": 0.4,
                },
                "valid_metrics": {
                    "ndcg_at_12": 0.4,
                    "coverage_by_window": [{"cutoff_week": 88, "coverage": 0.01}],
                },
            },
            {
                "grid_index": 1,
                "weights": {
                    "pop_score": 0.2,
                    "sim_score": 0.2,
                    "trend_score": 0.1,
                    "recent_score": 0.5,
                },
                "valid_metrics": {
                    "ndcg_at_12": 0.4,
                    "coverage_by_window": [{"cutoff_week": 88, "coverage": 0.01}],
                },
            },
            {
                "grid_index": 3,
                "weights": {
                    "pop_score": 0.4,
                    "sim_score": 0.2,
                    "trend_score": 0.2,
                    "recent_score": 0.2,
                },
                "valid_metrics": {"ndcg_at_12": 0.5},
            },
        ],
        required_trend_scores=(0.1, 0.2),
    )

    assert [row["trend_score"] for row in rows] == [0.1, 0.2]
    assert rows[0]["grid_index"] == 1
    assert rows[0]["metrics"]["valid"]["coverage_by_window"] == [
        {"cutoff_week": 88, "coverage": 0.01}
    ]
