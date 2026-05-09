from __future__ import annotations

from typing import Any


def build_ablation_summary(
    metrics_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in metrics_payloads:
        method = str(payload["method"])
        metrics_by_split = dict(payload["metrics"])
        for split, metrics in metrics_by_split.items():
            rows.append({"method": method, "split": split, **dict(metrics)})
    return sorted(rows, key=lambda row: (row["split"], row["method"]))
