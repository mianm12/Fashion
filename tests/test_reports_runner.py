from __future__ import annotations

import json

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
