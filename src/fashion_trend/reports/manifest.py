from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fashion_trend.foundation.io import write_json_atomic

MANIFEST_SCHEMA_VERSION = "paper_assets_manifest/v1"


def build_manifest_payload(
    *,
    parameters: dict[str, Any],
    input_artifacts: dict[str, str],
    output_artifacts: dict[str, list[str]],
    row_counts: dict[str, int],
    case_user_ids: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the audit manifest payload for reports exports."""
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "row_counts": row_counts,
        "figure_count": len(output_artifacts.get("figures", [])),
        "table_count": len(output_artifacts.get("tables", [])),
        "case_count": len(case_user_ids),
        "case_user_ids": case_user_ids,
        "warnings": warnings,
    }
    _validate_manifest(payload)
    return payload


def write_manifest(payload: dict[str, Any], output_path: Path) -> None:
    """Write a validated reports manifest atomically."""
    _validate_manifest(payload)
    write_json_atomic(payload, output_path)


def _validate_manifest(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "generated_at",
        "parameters",
        "input_artifacts",
        "output_artifacts",
        "row_counts",
        "figure_count",
        "table_count",
        "case_count",
        "case_user_ids",
        "warnings",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"reports manifest 缺少字段: {missing}")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"reports manifest schema_version 不匹配: {payload['schema_version']}"
        )
