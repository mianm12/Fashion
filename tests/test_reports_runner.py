from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from fashion_trend.reports.figures import build_recommendation_weight_analysis_figure
from fashion_trend.reports.manifest import build_manifest_payload
from fashion_trend.reports.runner import (
    PaperAssetsExportConfig,
    run_paper_assets_export,
)
from fashion_trend.reports.tables import REPORT_TABLE_COLUMNS


def test_build_manifest_payload_records_outputs() -> None:
    payload = build_manifest_payload(
        parameters={"case_count": 3},
        input_artifacts={"predictions": "outputs/models/lightgbm/predictions.csv"},
        output_artifacts={
            "figures": [
                "outputs/reports/figures/a.svg",
                "outputs/reports/figures/a.png",
            ],
            "tables": [
                "outputs/reports/tables/a.csv",
                "outputs/reports/tables/a.md",
            ],
            "case_studies": ["outputs/reports/case_studies/case_1.json"],
        },
        row_counts={"trend_model_metrics": 8},
        case_user_ids=["customer-a"],
        warnings=["current grid has valid metrics only"],
    )

    assert payload["schema_version"] == "paper_assets_manifest/v1"
    assert payload["figure_count"] == 2
    assert payload["table_count"] == 2
    assert payload["case_count"] == 1
    assert payload["case_user_ids"] == ["customer-a"]


def test_run_paper_assets_export_writes_manifest_to_output_dir(
    tmp_path,
    monkeypatch,
) -> None:
    from fashion_trend.reports import runner

    fake_inputs = SimpleNamespace(
        input_artifacts={},
        report_table_rows={name: [{"row": 1}] for name in REPORT_TABLE_COLUMNS},
        trend_metrics=pd.DataFrame(),
        recommendation_metrics=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        trend_view=pd.DataFrame(),
        search_results=pd.DataFrame(),
        recommendation_items=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        representative_trends=pd.DataFrame(),
        best_weights={},
        warnings=[],
    )
    monkeypatch.setattr(runner, "configure_matplotlib_for_reports", lambda: "Test Font")
    monkeypatch.setattr(runner, "_load_report_inputs", lambda config: fake_inputs)
    monkeypatch.setattr(runner, "_write_tables", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(runner, "_write_figures", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "_write_cases", lambda *args, **kwargs: ([], []))

    payload = run_paper_assets_export(
        PaperAssetsExportConfig(
            case_count=2, top_k=5, trend_week=102, output_dir=tmp_path
        )
    )

    manifest_path = tmp_path / "manifest.json"
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["manifest_path"] == str(manifest_path)
    assert written["parameters"]["case_count"] == 2
    assert written["parameters"]["top_k"] == 5
    assert written["parameters"]["trend_week"] == 102
    assert written["figure_count"] == 0


def test_export_paper_assets_cli_passes_args(monkeypatch) -> None:
    module = importlib.import_module("17_export_paper_assets")
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {
            "manifest_path": "outputs/reports/manifest.json",
            "figure_count": 0,
            "table_count": 0,
            "case_count": 0,
        }

    monkeypatch.setattr(module, "run_paper_assets_export", fake_run)

    exit_code = module.main(
        [
            "--case-count",
            "2",
            "--top-k",
            "5",
            "--trend-week",
            "102",
            "--figure-format",
            "svg,png",
            "--output-dir",
            "outputs/reports-paper",
        ]
    )

    assert exit_code == 0
    assert captured["config"].case_count == 2
    assert captured["config"].top_k == 5
    assert captured["config"].trend_week == 102
    assert captured["config"].figure_formats == ("svg", "png")
    assert captured["config"].output_dir == Path("outputs/reports-paper")


def test_parse_figure_formats_rejects_duplicates() -> None:
    module = importlib.import_module("17_export_paper_assets")

    with pytest.raises(ValueError, match="不能重复"):
        module._parse_figure_formats("svg,svg")


def test_run_paper_assets_export_uses_monkeypatched_small_data(
    tmp_path,
    monkeypatch,
) -> None:
    from fashion_trend.reports import runner

    figure_names = (
        "data_pipeline",
        "attribute_graph_schema",
        "trend_curve_examples",
        "lightgbm_feature_importance",
        "trend_model_metrics",
        "recommendation_method_metrics",
        "topk_trend_attributes",
        "recommendation_weight_analysis",
    )
    fake_inputs = SimpleNamespace(
        input_artifacts={"predictions": "outputs/models/lightgbm/predictions.csv"},
        report_table_rows={name: [{"row": 1}] for name in REPORT_TABLE_COLUMNS},
        trend_metrics=pd.DataFrame(),
        recommendation_metrics=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        trend_view=pd.DataFrame(),
        search_results=pd.DataFrame(),
        recommendation_items=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        representative_trends=pd.DataFrame(),
        best_weights={
            "pop_score": 0.2,
            "sim_score": 0.3,
            "trend_score": 0.4,
            "recent_score": 0.1,
        },
        warnings=[],
    )

    monkeypatch.setattr(runner, "configure_matplotlib_for_reports", lambda: "Test Font")
    monkeypatch.setattr(runner, "_load_report_inputs", lambda config: fake_inputs)

    def fake_write_tables(report_table_rows, *, output_root):
        assert output_root == tmp_path
        assert set(report_table_rows) == set(REPORT_TABLE_COLUMNS)
        paths = [
            str(output_root / "tables" / f"{name}.{suffix}")
            for name in REPORT_TABLE_COLUMNS
            for suffix in ("csv", "md")
        ]
        return paths, {name: 1 for name in REPORT_TABLE_COLUMNS}

    def fake_write_figures(
        *args,
        trend_week,
        top_k,
        figure_formats,
        best_weights,
        output_root,
    ):
        assert trend_week == 103
        assert top_k == 10
        assert figure_formats == ("svg",)
        assert best_weights["trend_score"] == 0.4
        assert output_root == tmp_path
        return [
            str(output_root / "figures" / f"{name}.{suffix}")
            for name in figure_names
            for suffix in figure_formats
        ]

    def fake_write_cases(*args, case_count, output_root):
        assert case_count == 3
        assert output_root == tmp_path
        paths = [
            str(output_root / "case_studies" / f"case_{index:02d}.{suffix}")
            for index in range(1, 4)
            for suffix in ("json", "md")
        ]
        return paths, ["customer-a", "customer-b", "customer-c"]

    monkeypatch.setattr(runner, "_write_tables", fake_write_tables)
    monkeypatch.setattr(runner, "_write_figures", fake_write_figures)
    monkeypatch.setattr(runner, "_write_cases", fake_write_cases)

    payload = runner.run_paper_assets_export(
        runner.PaperAssetsExportConfig(
            case_count=3,
            top_k=10,
            trend_week=103,
            figure_formats=("svg",),
            output_dir=tmp_path,
        )
    )

    assert payload["schema_version"] == "paper_assets_manifest/v1"
    assert payload["table_count"] == 16
    assert payload["figure_count"] == 8
    assert payload["case_count"] == 3
    assert set(payload["row_counts"]) == set(REPORT_TABLE_COLUMNS)
    assert all(path.endswith(".svg") for path in payload["output_artifacts"]["figures"])
    assert (tmp_path / "manifest.json").exists()


def test_report_input_helper_rows_use_design_contracts(tmp_path) -> None:
    from fashion_trend.reports import runner

    artifact_path = tmp_path / "artifact.csv"
    pd.DataFrame([{"a": 1, "b": 2}]).to_csv(artifact_path, index=False)

    artifact_rows = runner.build_data_artifact_summary_rows(
        artifacts={"trend_samples": artifact_path},
        sections={"trend_samples": "trend"},
        paper_usage={"trend_samples": "趋势样本规模"},
    )
    assert tuple(artifact_rows[0]) == REPORT_TABLE_COLUMNS["data_artifact_summary"]
    assert artifact_rows[0]["row_count"] == 1
    assert artifact_rows[0]["column_count"] == 2

    split_rows = runner.build_time_split_summary_rows(
        split_frames={
            "test": pd.DataFrame(
                {"week_id": [101, 102], "attr_id": ["colour::black", "index::ladies"]}
            )
        },
        domain="trend",
    )
    assert tuple(split_rows[0]) == REPORT_TABLE_COLUMNS["time_split_summary"]
    assert split_rows[0]["week_start"] == 101
    assert split_rows[0]["week_count"] == 2
    assert split_rows[0]["attribute_count"] == 2

    graph_rows = runner.build_attribute_graph_summary_rows(
        graph_frames={
            "nodes_article": pd.DataFrame({"article_id": ["000000001"]}),
            "nodes_attribute": pd.DataFrame(
                {"attr_id": ["colour::black"], "attr_type": ["colour_group_name"]}
            ),
            "edges_article_attribute": pd.DataFrame(
                {"article_id": ["000000001"], "attr_type": ["colour_group_name"]}
            ),
            "edges_attribute_hierarchy": pd.DataFrame(
                {"parent_attr_type": ["product_group_name"]}
            ),
        },
        graph_paths={"edges_article_attribute": "graph/edges_article_attribute.csv"},
    )
    assert all(
        tuple(row) == REPORT_TABLE_COLUMNS["attribute_graph_summary"]
        for row in graph_rows
    )

    feature_rows = runner.build_trend_feature_summary_rows()
    assert tuple(feature_rows[0]) == REPORT_TABLE_COLUMNS["trend_feature_summary"]


def test_experiment_helpers_flatten_real_payload_shape() -> None:
    from fashion_trend.reports import runner

    payload = {
        "best_weights": {
            "pop_score": 0.20,
            "sim_score": 0.30,
            "trend_score": 0.40,
            "recent_score": 0.10,
        },
        "search_results": [
            {
                "weights": {
                    "pop_score": 0.20,
                    "sim_score": 0.30,
                    "trend_score": 0.40,
                    "recent_score": 0.10,
                },
                "valid_metrics": {
                    "map_at_12": 0.01,
                    "recall_at_12": 0.02,
                    "hit_rate_at_12": 0.03,
                    "ndcg_at_12": 0.04,
                    "coverage": 0.05,
                },
            }
        ],
        "ablation": [
            {
                "method": "pop_similarity_trend",
                "split": "test",
                "map_at_12": 0.11,
                "recall_at_12": 0.12,
                "hit_rate_at_12": 0.13,
                "ndcg_at_12": 0.14,
                "coverage": 0.15,
            },
            {
                "method": "pop_similarity",
                "split": "test",
                "map_at_12": 0.21,
                "recall_at_12": 0.22,
                "hit_rate_at_12": 0.23,
                "ndcg_at_12": 0.24,
                "coverage": 0.25,
            },
        ],
    }

    search_results = runner.flatten_experiment_search_results(payload)
    assert tuple(search_results.columns) == (
        "section",
        "rank",
        "method",
        "split",
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
        "map_at_12",
        "recall_at_12",
        "hit_rate_at_12",
        "ndcg_at_12",
        "coverage",
    )
    assert search_results.loc[0, "section"] == "search_results"
    assert search_results.loc[0, "trend_score"] == 0.40

    rows = runner.flatten_recommendation_experiment_rows(payload)
    assert [row["section"] for row in rows] == [
        "search_results",
        "ablation",
        "ablation",
    ]
    assert rows[1]["trend_score"] == 0.40
    assert rows[2]["trend_score"] == ""


def test_flatten_recommendation_experiment_rows_includes_new_sections() -> None:
    from fashion_trend.reports import runner

    payload = {
        "best_weights": {
            "pop_score": 0.2,
            "sim_score": 0.2,
            "trend_score": 0.1,
            "recent_score": 0.5,
        },
        "search_results": [],
        "ablation": [],
        "named_ablation": [
            {
                "variant_id": "without_recent",
                "display_name": "w/o Recent",
                "weights": {
                    "pop_score": 0.4,
                    "sim_score": 0.4,
                    "trend_score": 0.2,
                    "recent_score": 0.0,
                },
                "metrics": {
                    "valid": {
                        "map_at_12": 0.1,
                        "recall_at_12": 0.2,
                        "hit_rate_at_12": 0.3,
                        "ndcg_at_12": 0.4,
                        "coverage": 0.5,
                    },
                    "test": {
                        "map_at_12": 0.6,
                        "recall_at_12": 0.7,
                        "hit_rate_at_12": 0.8,
                        "ndcg_at_12": 0.9,
                        "coverage": 1.0,
                    },
                },
            }
        ],
        "trend_bucket_best_by_valid": [
            {
                "variant_id": "trend_bucket_0_1",
                "display_name": "trend_score=0.1 valid-best",
                "weights": {
                    "pop_score": 0.2,
                    "sim_score": 0.2,
                    "trend_score": 0.1,
                    "recent_score": 0.5,
                },
                "metrics": {
                    "valid": {
                        "map_at_12": 0.11,
                        "recall_at_12": 0.12,
                        "hit_rate_at_12": 0.13,
                        "ndcg_at_12": 0.14,
                        "coverage": 0.15,
                    },
                    "test": {
                        "map_at_12": 0.21,
                        "recall_at_12": 0.22,
                        "hit_rate_at_12": 0.23,
                        "ndcg_at_12": 0.24,
                        "coverage": 0.25,
                    },
                },
            }
        ],
    }

    rows = runner.flatten_recommendation_experiment_rows(payload)

    assert [row["section"] for row in rows] == [
        "named_ablation",
        "named_ablation",
        "trend_bucket_best_by_valid",
        "trend_bucket_best_by_valid",
    ]
    assert rows[0]["method"] == "w/o Recent"
    assert rows[0]["split"] == "valid"
    assert rows[1]["split"] == "test"
    assert rows[2]["method"] == "trend_score=0.1 valid-best"


def test_manifest_helpers_capture_all_inputs_and_warnings(tmp_path) -> None:
    from fashion_trend.reports import runner

    legacy_csv = tmp_path / "recommendation_items.csv"
    legacy_csv.write_text("customer_id,article_id\n", encoding="utf-8")
    input_paths = SimpleNamespace(
        data_artifacts={"trend_model_samples": tmp_path / "samples.parquet"},
        trend_split_samples={"test": tmp_path / "samples_test.parquet"},
        graph_artifacts={"nodes_article": tmp_path / "nodes_article.csv"},
        trend_metrics={"lightgbm": tmp_path / "trend_metrics.json"},
        recommendation_metrics={"pop_similarity_trend": tmp_path / "metrics.json"},
        recommendation_items={
            "pop_similarity_trend": tmp_path / "recommendation_items.parquet"
        },
        recommendation_items_csv={"pop_similarity_trend": legacy_csv},
        lightgbm_predictions=tmp_path / "predictions.csv",
        lightgbm_feature_importance=tmp_path / "feature_importance.csv",
        trend_model_samples=tmp_path / "samples.parquet",
        recommendation_experiment=tmp_path / "experiment.json",
        evaluation_labels=tmp_path / "evaluation_labels.parquet",
        user_profile=tmp_path / "user_profile.parquet",
        article_attributes=tmp_path / "edges_article_attribute.csv",
    )
    payload = {
        "search_results": [
            {
                "weights": {"trend_score": 0.4},
                "valid_metrics": {"ndcg_at_12": 0.04},
            }
        ],
        "ablation": [{"method": "pop_similarity_trend"}],
    }

    artifacts = runner._build_input_artifacts(input_paths)
    warnings = runner._build_report_warnings(payload, input_paths)
    payload_with_strict_ablation = {
        **payload,
        "named_ablation": [
            {
                "display_name": "w/o Recent",
                "weight_policy": "strict_drop_and_renormalize_from_full",
                "weights": {"recent_score": 0.0},
                "metrics": {
                    "valid": {"ndcg_at_12": 0.1},
                    "test": {"ndcg_at_12": 0.2},
                },
            }
        ],
    }
    warnings_with_strict_ablation = runner._build_report_warnings(
        payload_with_strict_ablation,
        input_paths,
    )

    assert artifacts["data_artifact__trend_model_samples"].endswith("samples.parquet")
    assert artifacts["trend_split_sample__test"].endswith("samples_test.parquet")
    assert artifacts["graph_artifact__nodes_article"].endswith("nodes_article.csv")
    assert artifacts["recommendation_items_csv__pop_similarity_trend"].endswith(
        "recommendation_items.csv"
    )
    assert any("grid search 只有 valid 指标" in warning for warning in warnings)
    assert any("缺少严格 w/o Recent" in warning for warning in warnings)
    assert any("recommendation_items.csv" in warning for warning in warnings)
    assert not any(
        "缺少严格 w/o Recent" in warning for warning in warnings_with_strict_ablation
    )


def test_recommendation_weight_analysis_includes_best_weights(monkeypatch) -> None:
    monkeypatch.setattr("matplotlib.figure.Figure.tight_layout", lambda self: None)
    search_results = pd.DataFrame(
        [
            {"trend_score": 0.1, "ndcg_at_12": 0.02},
            {"trend_score": 0.4, "ndcg_at_12": 0.04},
        ]
    )
    figure = build_recommendation_weight_analysis_figure(
        search_results,
        best_weights={
            "pop_score": 0.20,
            "sim_score": 0.30,
            "trend_score": 0.40,
            "recent_score": 0.10,
        },
    )

    try:
        assert len(figure.axes) == 2
        assert "主实验权重构成" in figure.axes[1].get_title()
    finally:
        plt.close(figure)


def test_build_representative_trend_attributes_uses_top_eligible_rows() -> None:
    from fashion_trend.reports import runner

    trend_view = pd.DataFrame(
        [
            {
                "week_id": 102,
                "attr_type": "colour_group_name",
                "attr_value": "Red",
                "pred_target_growth": 0.40,
                "heat_t": 700,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": 103,
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "pred_target_growth": 0.30,
                "heat_t": 900,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": 103,
                "attr_type": "colour_group_name",
                "attr_value": "Blue",
                "pred_target_growth": 0.80,
                "heat_t": 100,
                "is_trend_eligible_t": False,
            },
            {
                "week_id": 103,
                "attr_type": "index_name",
                "attr_value": "Ladieswear",
                "pred_target_growth": 0.50,
                "heat_t": 800,
                "is_trend_eligible_t": True,
            },
        ]
    )

    rows = runner.build_representative_trend_attributes(
        trend_view,
        week_id=103,
        top_n=2,
    )

    assert tuple(rows.columns) == (
        "week_id",
        "attr_type",
        "attr_value",
        "pred_target_growth",
        "heat_t",
    )
    assert rows["attr_value"].tolist() == ["Ladieswear", "Black"]

    case_rows = runner.build_representative_trend_attributes(
        trend_view,
        week_ids=[102, 103],
        top_n=1,
    )
    assert case_rows["week_id"].tolist() == [103, 102]
    assert set(case_rows["attr_value"]) == {"Ladieswear", "Red"}
