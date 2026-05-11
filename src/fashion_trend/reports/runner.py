from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fashion_trend.reports.manifest import build_manifest_payload, write_manifest
from fashion_trend.reports.paths import ReportInputPaths, manifest_output_path


@dataclass(frozen=True)
class PaperAssetsExportConfig:
    case_count: int = 3
    top_k: int = 10
    trend_week: int = 103
    figure_formats: tuple[str, ...] = ("svg", "png")
    output_dir: Path | None = None
    input_paths: ReportInputPaths | None = None


def run_paper_assets_export(config: PaperAssetsExportConfig) -> dict[str, Any]:
    """Export paper assets from stable artifacts.

    Task 7 establishes the public runner contract. Full artifact export is
    connected in the integration task.
    """
    manifest_path = manifest_output_path(config.output_dir)
    payload = build_manifest_payload(
        parameters={
            "case_count": config.case_count,
            "top_k": config.top_k,
            "trend_week": config.trend_week,
            "figure_formats": list(config.figure_formats),
            "output_dir": str(manifest_path.parent),
        },
        input_artifacts={},
        output_artifacts={"figures": [], "tables": [], "case_studies": []},
        row_counts={},
        case_user_ids=[],
        warnings=[],
    )
    payload["manifest_path"] = str(manifest_path)
    write_manifest(payload, manifest_path)
    return payload
