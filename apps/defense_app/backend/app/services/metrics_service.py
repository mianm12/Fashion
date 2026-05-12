from __future__ import annotations


def group_metrics_by_domain(
    metrics: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for metric in metrics:
        groups.setdefault(str(metric["metric_domain"]), []).append(metric)
    return groups
