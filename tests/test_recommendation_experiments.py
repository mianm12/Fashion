from __future__ import annotations

import inspect
import json
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from fashion_trend.recommendation.experiments import runner as experiment_runner
from fashion_trend.recommendation.experiments.ablation import build_ablation_summary
from fashion_trend.recommendation.experiments.grid_search import (
    iter_weight_grid,
    select_best_weights,
)
from fashion_trend.recommendation.experiments.runner import (
    RecommendationExperimentContext,
    candidate_strategy_for_method,
    generate_experiment_run_id,
    run_baseline_methods,
    run_recommendation_experiment,
)
from fashion_trend.recommendation.inputs import RecommendationInputArtifacts
from fashion_trend.recommendation.paths import experiment_run_dir


def test_weight_grid_contains_only_valid_normalized_weights() -> None:
    weights = list(iter_weight_grid())

    assert weights
    assert all(abs(sum(item.values()) - 1.0) <= 1e-9 for item in weights)
    assert all(all(value >= 0.0 for value in item.values()) for item in weights)
    assert {
        "pop_score": 0.4,
        "sim_score": 0.3,
        "trend_score": 0.2,
        "recent_score": 0.1,
    } in weights


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

    assert select_best_weights(results) == {
        "pop_score": 0.4,
        "sim_score": 0.3,
        "trend_score": 0.2,
        "recent_score": 0.1,
    }


def test_select_best_weights_rejects_empty_grid_results() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_best_weights([])


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


def test_experiment_runner_exposes_force_rebuild_switch() -> None:
    signature = inspect.signature(run_recommendation_experiment)

    assert "force" in signature.parameters
    assert signature.parameters["force"].default is False


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
        lambda method, context, inputs: {"method": method, "metrics": {}},
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
    with pytest.raises(RuntimeError, match="--force"):
        run_baseline_methods(context, inputs, force=False)


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
