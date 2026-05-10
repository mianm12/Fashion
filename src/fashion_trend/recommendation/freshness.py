from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fashion_trend.recommendation.fingerprints import build_input_fingerprints


def build_artifact_metadata(
    name: str,
    input_artifacts: dict[str, str],
    output_artifacts: dict[str, str],
    schema_version: int,
    algorithm_version: str,
    config: dict[str, object],
    row_counts: dict[str, int],
) -> dict[str, object]:
    """Build a JSON-compatible metadata payload for freshness validation."""
    return _json_compatible(
        {
            "name": name,
            "input_artifacts": dict(input_artifacts),
            "input_fingerprints": build_input_fingerprints(input_artifacts),
            "output_artifacts": dict(output_artifacts),
            "schema_version": schema_version,
            "algorithm_version": algorithm_version,
            "config": dict(config),
            "row_counts": dict(row_counts),
        }
    )


def assert_fresh_metadata(
    metadata_path: Path,
    expected_input_artifacts: dict[str, str],
    expected_output_artifacts: dict[str, str],
    expected_schema_version: int,
    expected_algorithm_version: str,
    expected_config: dict[str, object],
    stale_message: Callable[[str], str],
) -> None:
    """Reject metadata that no longer matches the expected artifact contract."""
    if not metadata_path.exists():
        raise RuntimeError(stale_message(f"{metadata_path.name} is missing"))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError(stale_message("metadata is invalid"))
    expected_input_artifacts = _json_compatible(dict(expected_input_artifacts))
    expected_output_artifacts = _json_compatible(dict(expected_output_artifacts))
    expected_config = _json_compatible(dict(expected_config))
    expected_fingerprints = build_input_fingerprints(expected_input_artifacts)

    checks = (
        ("input_artifacts", expected_input_artifacts),
        ("input_fingerprints", expected_fingerprints),
        ("output_artifacts", expected_output_artifacts),
        ("schema_version", expected_schema_version),
        ("algorithm_version", expected_algorithm_version),
        ("config", expected_config),
    )
    for key, expected_value in checks:
        if metadata.get(key) != expected_value:
            raise RuntimeError(stale_message(f"{key} changed"))


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
