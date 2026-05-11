from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.reports.markdown import markdown_table
from fashion_trend.reports.paths import (
    case_study_output_paths,
    default_report_input_paths,
    figure_output_paths,
    manifest_output_path,
    table_output_paths,
)


def test_markdown_table_uses_stable_column_order_and_escapes_cells() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "name": "A|B",
                "note": "line 1\nline 2",
                "value": 0.123456,
                "missing": None,
            }
        ]
    )

    text = markdown_table(
        dataframe,
        columns=("name", "note", "value", "missing"),
        float_format="{:.3f}",
    )

    assert text == (
        "| name | note | value | missing |\n"
        "| --- | --- | --- | --- |\n"
        "| A\\|B | line 1<br>line 2 | 0.123 |  |\n"
    )


def test_markdown_table_rejects_missing_columns() -> None:
    dataframe = pd.DataFrame([{"name": "A"}])

    try:
        markdown_table(dataframe, columns=("name", "value"))
    except ValueError as exc:
        assert "Markdown 表格缺少列" in str(exc)
    else:
        raise AssertionError("missing column should fail")


def test_markdown_table_handles_empty_frame() -> None:
    dataframe = pd.DataFrame(columns=["name", "value"])

    assert markdown_table(dataframe, columns=("name", "value")) == (
        "| name | value |\n" "| --- | --- |\n"
    )


def test_output_path_helpers_honor_custom_output_root(tmp_path) -> None:
    root = tmp_path / "paper-assets"

    assert figure_output_paths("chart", output_root=root)["svg"] == (
        root / "figures" / "chart.svg"
    )
    assert table_output_paths("metrics", output_root=root)["csv"] == (
        root / "tables" / "metrics.csv"
    )
    assert case_study_output_paths("case_01", output_root=root)["json"] == (
        root / "case_studies" / "case_01.json"
    )
    assert manifest_output_path(root) == root / "manifest.json"


@pytest.mark.parametrize("unsafe_name", ["../x", "a/b", ".."])
def test_output_path_helpers_reject_unsafe_artifact_names(unsafe_name: str) -> None:
    with pytest.raises(ValueError, match="报告产物名称不是安全路径片段"):
        figure_output_paths(unsafe_name)

    with pytest.raises(ValueError, match="报告产物名称不是安全路径片段"):
        table_output_paths(unsafe_name)

    with pytest.raises(ValueError, match="报告产物名称不是安全路径片段"):
        case_study_output_paths(unsafe_name)


def test_default_report_input_paths_cover_core_sources() -> None:
    paths = default_report_input_paths()

    assert paths.lightgbm_predictions.as_posix().endswith(
        "outputs/models/lightgbm/predictions.csv"
    )
    assert (
        paths.trend_metrics["lightgbm"]
        .as_posix()
        .endswith("outputs/metrics/lightgbm/trend_metrics.json")
    )
    assert (
        paths.recommendation_items["pop_similarity_trend"]
        .as_posix()
        .endswith(
            "outputs/recommendation/pop_similarity_trend/recommendation_items.parquet"
        )
    )
    assert (
        paths.recommendation_items_csv["pop_similarity_trend"]
        .as_posix()
        .endswith(
            "outputs/recommendation/pop_similarity_trend/recommendation_items.csv"
        )
    )
    assert "trend_model_samples" in paths.data_artifacts
