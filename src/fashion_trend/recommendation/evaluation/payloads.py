from __future__ import annotations

import json


def build_recommendation_metrics_payload(
    method: str,
    metrics_by_split: dict[str, dict[str, object]],
    input_paths: dict[str, str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "method": method,
        "metrics": metrics_by_split,
        "input_paths": input_paths,
    }
    return ensure_finite_json_payload(payload)


def ensure_finite_json_payload(payload: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(payload, allow_nan=False)
    return json.loads(encoded)
