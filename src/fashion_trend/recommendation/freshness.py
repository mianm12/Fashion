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
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                stale_message(
                    _changed_metadata_reason(key, actual_value, expected_value)
                )
            )


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _changed_metadata_reason(
    key: str,
    actual_value: object,
    expected_value: object,
) -> str:
    if not isinstance(actual_value, dict) or not isinstance(expected_value, dict):
        return f"{key} changed"

    actual_keys = set(actual_value)
    expected_keys = set(expected_value)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    changed = sorted(
        item
        for item in expected_keys & actual_keys
        if actual_value.get(item) != expected_value.get(item)
    )
    details = []
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    if changed:
        details.append(f"changed={changed}")
    if not details:
        return f"{key} changed"
    return f"{key} changed: {', '.join(details)}"
