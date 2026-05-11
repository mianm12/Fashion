from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from fashion_trend.reports.manifest import build_manifest_payload
from fashion_trend.reports.runner import (
    PaperAssetsExportConfig,
    run_paper_assets_export,
)


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


def test_run_paper_assets_export_writes_manifest_to_output_dir(tmp_path) -> None:
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
