from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.recommendation.evaluation.metrics import (
    evaluate_recommendations,
    parse_prediction_items,
)
from fashion_trend.recommendation.experiments import runner as experiment_runner
from fashion_trend.recommendation.experiments.runner import (
    RecommendationExperimentContext,
    evaluate_result_for_experiment,
)
from fashion_trend.recommendation.inputs import RecommendationInputArtifacts
from fashion_trend.recommendation.methods.base import RecommendationResult


def test_missing_recommendation_user_scores_zero_by_default() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u2",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "customer_id": ["u1", "u2"],
            "article_id": ["0000000001", "0000000002"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "method": ["recent_popularity"],
            "prediction": ["0000000001 0000000003 0000000004"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "article_id": ["0000000001"],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["user_count"] == 2
    assert metrics["valid"]["missing_recommendation_user_count"] == 1
    assert metrics["valid"]["hit_rate_at_12"] == 0.5
    assert metrics["valid"]["recall_at_12"] == 0.5


def test_missing_recommendation_user_fails_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="missing"):
        evaluate_recommendations(
            pd.DataFrame(
                columns=[
                    "customer_id",
                    "split",
                    "cutoff_week",
                    "label_week",
                    "method",
                    "prediction",
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "split": "valid",
                        "cutoff_week": 10,
                        "label_week": 11,
                        "customer_id": "u1",
                        "history_purchase_count": 1,
                        "label_purchase_count": 1,
                    }
                ]
            ),
            pd.DataFrame(
                {
                    "split": ["valid"],
                    "cutoff_week": [10],
                    "label_week": [11],
                    "customer_id": ["u1"],
                    "article_id": ["0000000001"],
                }
            ),
            pd.DataFrame(
                {
                    "split": ["valid"],
                    "cutoff_week": [10],
                    "label_week": [11],
                    "article_id": ["0000000001"],
                }
            ),
            top_k=12,
            strict_missing_users=True,
        )


def test_ranking_metrics_use_exact_relevant_sets() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 2,
            }
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid", "valid"],
            "cutoff_week": [10, 10, 10],
            "label_week": [11, 11, 11],
            "customer_id": ["u1", "u1", "u1"],
            "article_id": ["0000000001", "0000000003", "0000000003"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "method": ["pop_similarity_trend"],
            "prediction": ["0000000001 0000000002 0000000003"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"] * 4,
            "cutoff_week": [10] * 4,
            "label_week": [11] * 4,
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
            ],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["map_at_12"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert metrics["valid"]["recall_at_12"] == 1.0
    assert metrics["valid"]["hit_rate_at_12"] == 1.0
    assert metrics["valid"]["ndcg_at_12"] == pytest.approx(
        (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
    )
    assert metrics["valid"]["coverage"] == 0.75


def test_coverage_is_computed_per_window_before_split_average() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
            {
                "split": "valid",
                "cutoff_week": 20,
                "label_week": 21,
                "customer_id": "u2",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 20],
            "label_week": [11, 21],
            "customer_id": ["u1", "u2"],
            "article_id": ["0000000001", "0000000005"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "split": ["valid", "valid"],
            "cutoff_week": [10, 20],
            "label_week": [11, 21],
            "method": ["recent_popularity", "recent_popularity"],
            "prediction": ["0000000001 0000000002", "0000000005"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"] * 6,
            "cutoff_week": [10, 10, 10, 10, 20, 20],
            "label_week": [11, 11, 11, 11, 21, 21],
            "article_id": [
                "0000000001",
                "0000000002",
                "0000000003",
                "0000000004",
                "0000000005",
                "0000000006",
            ],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["coverage_by_window"] == [
        {"cutoff_week": 10, "label_week": 11, "coverage": 0.5},
        {"cutoff_week": 20, "label_week": 21, "coverage": 0.5},
    ]
    assert metrics["valid"]["coverage"] == 0.5


def test_coverage_ignores_recommendations_outside_recommendable_pool() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            }
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "customer_id": ["u1"],
            "article_id": ["0000000001"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "method": ["recent_popularity"],
            "prediction": ["0000000001 0000000999"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "article_id": ["0000000001"],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
    )

    assert metrics["valid"]["coverage"] == 1.0


def test_parse_prediction_items_rejects_duplicate_or_too_many_articles() -> None:
    assert parse_prediction_items("0000000001 0000000002", top_k=2) == [
        "0000000001",
        "0000000002",
    ]

    with pytest.raises(ValueError, match="duplicate"):
        parse_prediction_items("0000000001 0000000001", top_k=12)

    with pytest.raises(ValueError, match="more than 1"):
        parse_prediction_items("0000000001 0000000002", top_k=1)


def test_empty_target_users_fails() -> None:
    with pytest.raises(ValueError, match="target_users"):
        evaluate_recommendations(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            top_k=12,
        )


def test_experiment_evaluation_reuses_recommendable_pool_cache(monkeypatch) -> None:
    cached_pool = pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "article_id": ["0000000001"],
        }
    )
    observed = {}

    monkeypatch.setattr(
        experiment_runner,
        "ensure_or_build_recommendable_pool_cache",
        lambda context, inputs, *, force: cached_pool,
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_recommendation_evaluation",
        lambda **kwargs: _capture_recommendable_pool(observed, kwargs),
    )

    evaluate_result_for_experiment(
        method="pop_similarity_trend",
        result=RecommendationResult(
            recommendations=pd.DataFrame(),
            recommendation_items=pd.DataFrame(),
            params={},
            metadata={},
        ),
        context=RecommendationExperimentContext(
            transactions=pd.DataFrame({"article_id": ["0000000999"], "week_id": [10]}),
            article_attributes=pd.DataFrame(),
            trend_predictions=pd.DataFrame(),
        ),
        inputs=RecommendationInputArtifacts(
            time_windows=pd.DataFrame(
                [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
            ),
            target_users=pd.DataFrame(),
            evaluation_labels=pd.DataFrame(),
            user_profile=pd.DataFrame(),
        ),
    )

    assert observed["recommendable_pool"] is cached_pool


def test_eval_cli_helper_fails_when_recommendable_pool_cache_missing(
    tmp_path,
    monkeypatch,
) -> None:
    eval_module = _load_eval_script_module()
    monkeypatch.setattr(
        eval_module,
        "FEATURE_CACHE_METADATA_PATH",
        tmp_path / "features" / "metadata.json",
    )

    with pytest.raises(RuntimeError) as exc_info:
        eval_module.read_cached_recommendable_pool_for_evaluation(
            pd.DataFrame(columns=["split", "cutoff_week", "label_week"])
        )
    message = str(exc_info.value)
    assert "16_run_recommendation_experiment.py" in message
    assert "--force-cache" in message
    assert "--force-rebuild-all" in message
    assert "--experiment main --force`" not in message


def _capture_recommendable_pool(
    observed: dict[str, pd.DataFrame],
    kwargs: dict[str, object],
) -> dict[str, object]:
    observed["recommendable_pool"] = kwargs["recommendable_pool"]
    return {"metrics": {}}


def _load_eval_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "src" / "15_eval_recommendations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "eval_recommendations_15", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load eval script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
