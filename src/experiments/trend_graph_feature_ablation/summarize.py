from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import pandas as pd

from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_VARIANTS,
    SUMMARY_COLUMNS,
)


def build_metrics_summary_frame(
    metrics_payloads: Mapping[str, Mapping[str, object]],
    metadata_payloads: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """按固定 variant 顺序汇总图特征消融的 valid/test 指标。"""

    rows = [
        _build_variant_summary_row(variant, metrics_payloads, metadata_payloads)
        for variant in ABLATION_VARIANTS
    ]
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS), dtype=object)


def render_metrics_summary_markdown(summary: pd.DataFrame) -> str:
    """渲染 metrics summary Markdown 表格，兼容缺少 tabulate 的环境。"""

    try:
        markdown = summary.to_markdown(index=False)
    except ImportError:
        markdown = _render_markdown_table(summary)
    if not markdown.endswith("\n"):
        markdown += "\n"
    return markdown


def _build_variant_summary_row(
    variant: str,
    metrics_payloads: Mapping[str, Mapping[str, object]],
    metadata_payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    metrics_payload = _require_variant_payload(metrics_payloads, variant, "metrics")
    metadata_payload = _require_variant_payload(metadata_payloads, variant, "metadata")

    return {
        "variant": variant,
        "feature_count": _read_feature_count(metadata_payload, variant),
        "best_iteration": _read_best_iteration(metadata_payload, variant),
        "training_elapsed_seconds": _read_finite_number(
            metadata_payload,
            "training_elapsed_seconds",
            variant=variant,
            field_path="metadata.training_elapsed_seconds",
        ),
        "valid_ndcg_at_10": _read_metric(
            metrics_payload,
            variant=variant,
            split="valid",
            metric_name="ndcg_at_k",
            key="10",
        ),
        "valid_spearman": _read_metric(
            metrics_payload,
            variant=variant,
            split="valid",
            metric_name="spearman",
        ),
        "valid_precision_at_10": _read_metric(
            metrics_payload,
            variant=variant,
            split="valid",
            metric_name="precision_at_k",
            key="10",
        ),
        "valid_recall_at_10": _read_metric(
            metrics_payload,
            variant=variant,
            split="valid",
            metric_name="recall_at_k",
            key="10",
        ),
        "test_ndcg_at_10": _read_metric(
            metrics_payload,
            variant=variant,
            split="test",
            metric_name="ndcg_at_k",
            key="10",
        ),
        "test_spearman": _read_metric(
            metrics_payload,
            variant=variant,
            split="test",
            metric_name="spearman",
        ),
        "test_precision_at_10": _read_metric(
            metrics_payload,
            variant=variant,
            split="test",
            metric_name="precision_at_k",
            key="10",
        ),
        "test_recall_at_10": _read_metric(
            metrics_payload,
            variant=variant,
            split="test",
            metric_name="recall_at_k",
            key="10",
        ),
    }


def _require_variant_payload(
    payloads: Mapping[str, Mapping[str, object]],
    variant: str,
    source_name: str,
) -> Mapping[str, object]:
    if variant not in payloads:
        raise ValueError(f"缺少 {source_name} payload: {variant}")
    payload = payloads[variant]
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source_name} payload 必须为对象: {variant}")
    return payload


def _read_feature_count(metadata_payload: Mapping[str, object], variant: str) -> int:
    feature_mask = _require_key(
        metadata_payload,
        "feature_mask",
        variant=variant,
        source_name="metadata",
    )
    if not isinstance(feature_mask, Mapping):
        raise ValueError(f"metadata feature_mask 必须为对象: {variant}")
    numeric_features = _read_feature_list(feature_mask, "numeric_features", variant)
    categorical_features = _read_feature_list(
        feature_mask,
        "categorical_features",
        variant,
    )
    return len(numeric_features) + len(categorical_features)


def _read_best_iteration(
    metadata_payload: Mapping[str, object],
    variant: str,
) -> int | None:
    value = _require_key(
        metadata_payload,
        "best_iteration",
        variant=variant,
        source_name="metadata",
    )
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{variant} metadata.best_iteration 必须为整数或 None")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{variant} metadata.best_iteration 必须为整数或 None")
    if isinstance(value, Integral):
        return int(value)
    return int(number)


def _read_feature_list(
    feature_mask: Mapping[str, object],
    key: str,
    variant: str,
) -> Sequence[object]:
    value = _require_key(
        feature_mask,
        key,
        variant=variant,
        source_name="metadata feature_mask",
    )
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"metadata feature_mask {key} 必须为序列: {variant}")
    return value


def _read_metric(
    metrics_payload: Mapping[str, object],
    *,
    variant: str,
    split: str,
    metric_name: str,
    key: str | None = None,
) -> object:
    overall = _require_key(
        metrics_payload,
        "overall",
        variant=variant,
        source_name="metrics",
    )
    if not isinstance(overall, Mapping):
        raise ValueError(f"metrics overall 必须为对象: {variant}")
    split_metrics = _require_key(
        overall,
        split,
        variant=variant,
        source_name="metrics overall",
    )
    if not isinstance(split_metrics, Mapping):
        raise ValueError(f"metrics overall.{split} 必须为对象: {variant}")
    metric_value = _require_key(
        split_metrics,
        metric_name,
        variant=variant,
        source_name=f"metrics overall.{split}",
    )
    if key is None:
        return _read_nullable_finite_number(
            metric_value,
            variant=variant,
            field_path=f"metrics.overall.{split}.{metric_name}",
        )
    if not isinstance(metric_value, Mapping):
        raise ValueError(f"metrics {split}.{metric_name} 必须为对象: {variant}")
    nested_value = _require_key(
        metric_value,
        key,
        variant=variant,
        source_name=f"metrics {split}.{metric_name}",
    )
    return _read_nullable_finite_number(
        nested_value,
        variant=variant,
        field_path=f"metrics.overall.{split}.{metric_name}.{key}",
    )


def _read_finite_number(
    payload: Mapping[str, object],
    key: str,
    *,
    variant: str,
    field_path: str,
) -> float:
    value = _require_key(
        payload,
        key,
        variant=variant,
        source_name=field_path.rsplit(".", maxsplit=1)[0],
    )
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{variant} {field_path} 必须为有限数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{variant} {field_path} 必须为有限数值")
    return number


def _read_nullable_finite_number(
    value: object,
    *,
    variant: str,
    field_path: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{variant} {field_path} 必须为有限数值或 None")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{variant} {field_path} 必须为有限数值或 None")
    return number


def _require_key(
    payload: Mapping[str, object],
    key: str,
    *,
    variant: str,
    source_name: str,
) -> object:
    if key not in payload:
        raise ValueError(f"{variant} 缺少指标或字段 {source_name}.{key}")
    return payload[key]


def _render_markdown_table(summary: pd.DataFrame) -> str:
    headers = [str(column) for column in summary.columns]
    rows = [
        [_format_markdown_cell(value) for value in row]
        for row in summary.itertuples(index=False, name=None)
    ]
    widths = _column_widths(headers, rows)
    header = _render_markdown_row(headers, widths)
    separator = _render_markdown_row(["-" * width for width in widths], widths)
    body = [_render_markdown_row(row, widths) for row in rows]
    return "\n".join([header, separator, *body])


def _column_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    return widths


def _render_markdown_row(values: list[str], widths: list[int]) -> str:
    padded = [value.ljust(widths[index]) for index, value in enumerate(values)]
    return "| " + " | ".join(padded) + " |"


def _format_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)
