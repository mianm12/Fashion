from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.reports.loaders import (
    REPORT_TREND_JOIN_KEY,
    build_lightgbm_prediction_sample_view,
    flatten_recommendation_metrics,
    flatten_trend_metrics,
    flatten_trend_metrics_by_attr_type,
    read_feature_importance,
    read_json_object,
    read_trend_samples,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "week_id": 103,
                "attr_id": "colour_group_name::Light Green",
                "attr_type": "colour_group_name",
                "attr_value": "Light Green",
                "model_name": "lightgbm",
                "split": "test",
                "share_t": 0.1,
                "pred_share_t1": 0.12,
                "target_growth": 0.2,
                "pred_target_growth": 0.1,
                "target_rank_in_type_t1": 1,
            }
        ]
    )


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "week_id": 103,
                "attr_id": "colour_group_name::Light Green",
                "attr_type": "colour_group_name",
                "attr_value": "Light Green",
                "heat_t": 120,
                "share_t": 0.1,
                "target_growth": 0.2,
                "history_total_heat_t": 1000,
                "history_active_weeks_t": 20,
                "is_trend_eligible_t": 1,
            }
        ]
    )


def test_build_lightgbm_prediction_sample_view_adds_filter_columns() -> None:
    view = build_lightgbm_prediction_sample_view(_prediction_frame(), _sample_frame())

    assert tuple(view.loc[0, REPORT_TREND_JOIN_KEY]) == (
        103,
        "colour_group_name::Light Green",
        "colour_group_name",
        "Light Green",
    )
    assert view.loc[0, "heat_t"] == 120
    assert view.loc[0, "history_total_heat_t"] == 1000
    assert view.loc[0, "history_active_weeks_t"] == 20
    assert view.loc[0, "is_trend_eligible_t"] == 1


def test_build_lightgbm_prediction_sample_view_rejects_duplicate_prediction_key() -> (
    None
):
    predictions = pd.concat([_prediction_frame(), _prediction_frame()])

    try:
        build_lightgbm_prediction_sample_view(predictions, _sample_frame())
    except ValueError as exc:
        assert "LightGBM predictions 存在重复 join key" in str(exc)
    else:
        raise AssertionError("duplicate prediction key should fail")


def test_build_lightgbm_prediction_sample_view_rejects_missing_sample_match() -> None:
    samples = _sample_frame()
    samples.loc[0, "attr_value"] = "Dark Green"

    try:
        build_lightgbm_prediction_sample_view(_prediction_frame(), samples)
    except ValueError as exc:
        assert "无法 1:1 join" in str(exc)
    else:
        raise AssertionError("missing sample match should fail")


def test_build_lightgbm_prediction_sample_view_rejects_extra_sample_key() -> None:
    extra_sample = _sample_frame()
    extra_sample.loc[0, "attr_value"] = "Dark Green"
    samples = pd.concat([_sample_frame(), extra_sample], ignore_index=True)

    try:
        build_lightgbm_prediction_sample_view(_prediction_frame(), samples)
    except ValueError as exc:
        message = str(exc)
        assert "无法 1:1 join" in message
        assert "Dark Green" in message
    else:
        raise AssertionError("extra sample key should fail")


def test_build_lightgbm_prediction_sample_view_rejects_conflicting_shared_values() -> (
    None
):
    samples = _sample_frame()
    samples.loc[0, "share_t"] = 0.2

    try:
        build_lightgbm_prediction_sample_view(_prediction_frame(), samples)
    except ValueError as exc:
        assert "share_t 不一致" in str(exc)
    else:
        raise AssertionError("conflicting share_t should fail")


def test_read_json_object_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    try:
        read_json_object(path, artifact_name="payload")
    except ValueError as exc:
        assert "必须是 JSON object" in str(exc)
    else:
        raise AssertionError("non-object json should fail")


def test_read_trend_samples_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "trend_model_samples.parquet"
    dataframe = _sample_frame().drop(columns=["history_active_weeks_t"])
    dataframe.to_parquet(path, index=False)

    with pytest.raises(
        ValueError,
        match="trend_model_samples 缺少列.*history_active_weeks_t",
    ):
        read_trend_samples(path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("split_importance", "nan"),
        ("gain_importance", "inf"),
        ("normalized_gain_importance", "bad"),
        ("normalized_gain_importance", "-0.1"),
    ],
)
def test_read_feature_importance_rejects_bad_numeric_values(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    path = tmp_path / "feature_importance.csv"
    row = {
        "feature": "lag_1",
        "split_importance": "1",
        "gain_importance": "2.0",
        "normalized_gain_importance": "0.5",
    }
    row[column] = value
    path.write_text(
        ",".join(row) + "\n" + ",".join(row.values()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"feature_importance.*{column}"):
        read_feature_importance(path)


def test_flatten_trend_metrics_extracts_report_columns() -> None:
    payload = {
        "model_name": "lightgbm",
        "run_id": "run-1",
        "overall": {
            "test": {
                "mae": 0.1,
                "rmse": 0.2,
                "spearman": 0.3,
                "ndcg_at_k": {"10": 0.4},
                "precision_at_k": {"10": 0.5},
                "recall_at_k": {"10": 0.6},
            }
        },
    }

    rows = flatten_trend_metrics(payload)

    assert rows == [
        {
            "model_name": "lightgbm",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-1",
        }
    ]


def test_flatten_trend_metrics_requires_model_name() -> None:
    payload = {
        "overall": {
            "test": {
                "mae": 0.1,
                "rmse": 0.2,
                "spearman": 0.3,
                "ndcg_at_k": {"10": 0.4},
                "precision_at_k": {"10": 0.5},
                "recall_at_k": {"10": 0.6},
            }
        },
    }

    with pytest.raises(ValueError, match="model_name"):
        flatten_trend_metrics(payload)


def test_flatten_trend_metrics_rejects_bool_numbers() -> None:
    payload = {
        "model_name": "lightgbm",
        "overall": {
            "test": {
                "mae": True,
                "rmse": 0.2,
                "spearman": 0.3,
                "ndcg_at_k": {"10": 0.4},
                "precision_at_k": {"10": 0.5},
                "recall_at_k": {"10": 0.6},
            }
        },
    }

    with pytest.raises(ValueError, match="不是有限数值.*mae"):
        flatten_trend_metrics(payload)


def test_flatten_trend_metrics_by_attr_type_extracts_design_columns() -> None:
    payload = {
        "model_name": "lightgbm",
        "by_attr_type": {
            "test": {
                "colour_group_name": {
                    "mae": 0.11,
                    "rmse": 0.21,
                    "spearman": 0.31,
                    "ndcg_at_k": {"10": 0.41},
                    "precision_at_k": {"10": 0.51},
                    "recall_at_k": {"10": 0.61},
                }
            }
        },
    }

    rows = flatten_trend_metrics_by_attr_type(payload)

    assert rows == [
        {
            "model_name": "lightgbm",
            "split": "test",
            "attr_type": "colour_group_name",
            "mae": 0.11,
            "rmse": 0.21,
            "spearman": 0.31,
            "ndcg_at_10": 0.41,
            "precision_at_10": 0.51,
            "recall_at_10": 0.61,
        }
    ]


def test_flatten_trend_metrics_by_attr_type_allows_undefined_spearman() -> None:
    payload = {
        "model_name": "last_week",
        "by_attr_type": {
            "valid": {
                "index_group_name": {
                    "mae": 0.11,
                    "rmse": 0.21,
                    "spearman": None,
                    "ndcg_at_k": {"10": 0.41},
                    "precision_at_k": {"10": 0.51},
                    "recall_at_k": {"10": 0.61},
                }
            }
        },
    }

    rows = flatten_trend_metrics_by_attr_type(payload)

    assert rows[0]["spearman"] == ""


def test_flatten_recommendation_metrics_requires_method() -> None:
    payload = {
        "metrics": {
            "valid": {
                "map_at_12": 0.1,
                "recall_at_12": 0.2,
                "hit_rate_at_12": 0.3,
                "ndcg_at_12": 0.4,
                "coverage": 0.5,
                "user_count": 10,
                "missing_recommendation_user_count": 0,
            }
        },
    }

    with pytest.raises(ValueError, match="method"):
        flatten_recommendation_metrics(payload)


def test_flatten_recommendation_metrics_rejects_fractional_counts() -> None:
    payload = {
        "method": "pop_similarity_trend",
        "metrics": {
            "valid": {
                "map_at_12": 0.1,
                "recall_at_12": 0.2,
                "hit_rate_at_12": 0.3,
                "ndcg_at_12": 0.4,
                "coverage": 0.5,
                "user_count": 10.9,
                "missing_recommendation_user_count": 0,
            }
        },
    }

    with pytest.raises(ValueError, match="user_count"):
        flatten_recommendation_metrics(payload)


def test_flatten_recommendation_metrics_rejects_bool_counts() -> None:
    payload = {
        "method": "pop_similarity_trend",
        "metrics": {
            "valid": {
                "map_at_12": 0.1,
                "recall_at_12": 0.2,
                "hit_rate_at_12": 0.3,
                "ndcg_at_12": 0.4,
                "coverage": 0.5,
                "user_count": True,
                "missing_recommendation_user_count": 0,
            }
        },
    }

    with pytest.raises(ValueError, match="user_count"):
        flatten_recommendation_metrics(payload)


def test_flatten_recommendation_metrics_extracts_report_columns() -> None:
    payload = {
        "method": "pop_similarity_trend",
        "metrics": {
            "valid": {
                "map_at_12": 0.1,
                "recall_at_12": 0.2,
                "hit_rate_at_12": 0.3,
                "ndcg_at_12": 0.4,
                "coverage": 0.5,
                "user_count": 10,
                "missing_recommendation_user_count": 0,
            }
        },
    }

    rows = flatten_recommendation_metrics(payload)

    assert rows == [
        {
            "method": "pop_similarity_trend",
            "split": "valid",
            "map_at_12": 0.1,
            "recall_at_12": 0.2,
            "hit_rate_at_12": 0.3,
            "ndcg_at_12": 0.4,
            "coverage": 0.5,
            "user_count": 10,
            "missing_recommendation_user_count": 0,
        }
    ]
