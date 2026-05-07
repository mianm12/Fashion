from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest


def test_trend_entrypoint_is_package_facade() -> None:
    trend = importlib.import_module("fashion_trend.trend")

    assert Path(trend.__file__).name == "__init__.py"
    assert hasattr(trend, "build_article_week_sales_frame")
    assert hasattr(trend, "TREND_MODEL_PREDICTION_COLUMNS")


def test_trend_schema_module_exports_core_contracts() -> None:
    from fashion_trend.trend.schema import (
        ARTICLE_WEEK_SALES_COLUMNS,
        TREND_MODEL_PREDICTION_COLUMNS,
        TREND_MODEL_SPLIT_VALUES,
    )

    assert ARTICLE_WEEK_SALES_COLUMNS == (
        "week_id",
        "article_id",
        "sales_cnt",
        "sales_user_cnt",
        "sales_amount",
    )
    assert TREND_MODEL_SPLIT_VALUES == ("train", "valid", "test")
    assert TREND_MODEL_PREDICTION_COLUMNS[:6] == (
        "week_id",
        "attr_id",
        "attr_type",
        "attr_value",
        "model_name",
        "split",
    )


def test_trend_validation_module_rejects_missing_columns() -> None:
    from fashion_trend.trend.validation import validate_required_columns

    with pytest.raises(ValueError, match="missing_col"):
        validate_required_columns(
            pd.DataFrame({"present": [1]}).columns.tolist(),
            ("present", "missing_col"),
            source_name="测试表",
        )


def test_article_sales_and_io_modules_export_stage_api() -> None:
    from fashion_trend.trend.article_sales import (
        build_article_week_sales_frame,
        read_article_week_sales,
        read_weekly_transactions,
        validate_article_week_sales,
    )
    from fashion_trend.trend.io import write_json, write_trend_csv, write_trend_parquet

    assert callable(read_weekly_transactions)
    assert callable(build_article_week_sales_frame)
    assert callable(validate_article_week_sales)
    assert callable(read_article_week_sales)
    assert callable(write_json)
    assert callable(write_trend_csv)
    assert callable(write_trend_parquet)


def test_heat_and_target_modules_export_stage_api() -> None:
    from fashion_trend.trend.attribute_heat import (
        build_attribute_week_heat_frame,
        read_article_attribute_edges,
        read_attribute_nodes,
        read_attribute_week_heat,
        validate_attribute_week_heat,
    )
    from fashion_trend.trend.targets import (
        build_attribute_week_target_frame,
        read_attribute_week_target,
        validate_attribute_week_target,
        validate_attribute_week_target_matches_heat,
    )

    assert callable(read_article_attribute_edges)
    assert callable(read_attribute_nodes)
    assert callable(read_attribute_week_heat)
    assert callable(build_attribute_week_heat_frame)
    assert callable(validate_attribute_week_heat)
    assert callable(read_attribute_week_target)
    assert callable(build_attribute_week_target_frame)
    assert callable(validate_attribute_week_target)
    assert callable(validate_attribute_week_target_matches_heat)


def test_sample_and_split_modules_export_stage_api() -> None:
    from fashion_trend.trend.samples import (
        build_attribute_graph_features_frame,
        build_trend_model_samples_frame,
        read_attribute_hierarchy_edges,
        validate_trend_model_samples,
    )
    from fashion_trend.trend.splits import (
        build_trend_model_split_frames,
        build_trend_model_split_metadata,
        read_trend_model_split,
        validate_trend_model_split_frame,
        validate_trend_model_split_frames,
    )

    assert callable(read_attribute_hierarchy_edges)
    assert callable(build_attribute_graph_features_frame)
    assert callable(build_trend_model_samples_frame)
    assert callable(validate_trend_model_samples)
    assert callable(build_trend_model_split_frames)
    assert callable(validate_trend_model_split_frames)
    assert callable(validate_trend_model_split_frame)
    assert callable(build_trend_model_split_metadata)
    assert callable(read_trend_model_split)


def test_prediction_module_exports_contract_api() -> None:
    from fashion_trend.trend.predictions import (
        derive_normalized_pred_share_t1,
        validate_pred_share_t1_distribution,
        validate_trend_model_predictions,
    )

    assert callable(validate_trend_model_predictions)
    assert callable(derive_normalized_pred_share_t1)
    assert callable(validate_pred_share_t1_distribution)
