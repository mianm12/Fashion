from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from fashion_trend.foundation.io import write_text_atomic


def read_run_id_from_model_metadata(metadata_path: Path) -> str | None:
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LightGBM metadata {metadata_path} 不是合法 JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"LightGBM metadata {metadata_path} 顶层必须是 object。")
    run_id = payload.get("run_id")
    return str(run_id) if run_id is not None else None


def build_lightgbm_evaluation_summary(
    *,
    run_id: str,
    metrics_path: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    overall = payload["overall"]
    valid = overall["valid"]
    return {
        "run_id": run_id,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metrics_path": str(metrics_path),
        "selection_metrics": {
            "split": "valid",
            "ndcg_at_10": valid["ndcg_at_k"]["10"],
            "spearman": valid["spearman"],
            "mae": valid["mae"],
            "rmse": valid["rmse"],
        },
        "report_metrics": {
            "valid": overall["valid"],
            "test": overall["test"],
        },
    }


def validate_lightgbm_run_metrics_payload(
    payload: object,
    *,
    run_id: str,
    prediction_path: Path,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("LightGBM run metrics payload 顶层必须是 object。")
    if payload.get("model_name") != "lightgbm":
        raise ValueError("LightGBM run metrics 的 model_name 必须是 lightgbm。")
    if payload.get("run_id") != run_id:
        raise ValueError(
            f"LightGBM run metrics 的 run_id 不匹配: {payload.get('run_id')}"
        )
    if payload.get("prediction_path") != str(prediction_path):
        raise ValueError("LightGBM run metrics 的 prediction_path 不指向当前 run。")
    _validate_trend_metrics_contract(payload)


def build_stable_metrics_payload(
    payload: dict[str, object],
    *,
    stable_prediction_path: Path,
    stable_metrics_path: Path,
) -> dict[str, object]:
    stable_payload = dict(payload)
    stable_payload["prediction_path"] = str(stable_prediction_path)
    stable_payload["output_path"] = str(stable_metrics_path)
    return stable_payload


def upsert_lightgbm_evaluation_index(
    index_path: Path,
    summary: dict[str, object],
) -> None:
    summaries: dict[str, dict[str, object]] = {}
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                summaries[str(payload["run_id"])] = payload
    summaries[str(summary["run_id"])] = summary
    lines = [
        json.dumps(summaries[key], ensure_ascii=False, sort_keys=True)
        for key in sorted(summaries)
    ]
    write_text_atomic("\n".join(lines) + "\n", index_path)


def _validate_trend_metrics_contract(payload: dict[str, object]) -> None:
    _validate_strict_json_payload(payload)
    evaluated_splits = payload.get("evaluated_splits")
    if evaluated_splits != ["valid", "test"]:
        raise ValueError("LightGBM run metrics 的 evaluated_splits 必须是 valid/test。")
    ranking = _require_mapping(payload, "ranking", "LightGBM run metrics")
    for key in ("target_column", "prediction_column", "group_by", "k_values"):
        if key not in ranking:
            raise ValueError(f"LightGBM run metrics 的 ranking 缺少 {key}。")
    k_values = ranking["k_values"]
    if not isinstance(k_values, list) or not k_values:
        raise ValueError("LightGBM run metrics 的 ranking.k_values 必须是非空列表。")
    k_keys = {str(k_value) for k_value in k_values}
    overall = _require_mapping(payload, "overall", "LightGBM run metrics")
    by_attr_type = _require_mapping(payload, "by_attr_type", "LightGBM run metrics")
    groups = _require_mapping(payload, "groups", "LightGBM run metrics")
    for split in ("valid", "test"):
        split_metrics = _require_mapping(overall, split, "LightGBM run metrics overall")
        _validate_split_metric_values(
            split_metrics,
            source=f"overall.{split}",
            k_keys=k_keys,
        )
        attr_type_metrics = _require_mapping(
            by_attr_type,
            split,
            "LightGBM run metrics by_attr_type",
        )
        for attr_type, attr_metrics in attr_type_metrics.items():
            if not isinstance(attr_metrics, dict):
                raise ValueError(
                    f"LightGBM run metrics 的 by_attr_type.{split}.{attr_type} "
                    "必须是 object。"
                )
            _validate_split_metric_values(
                attr_metrics,
                source=f"by_attr_type.{split}.{attr_type}",
                k_keys=k_keys,
            )
        split_groups = _require_mapping(groups, split, "LightGBM run metrics groups")
        if "ranking_groups" not in split_groups:
            raise ValueError(
                f"LightGBM run metrics 的 groups.{split} 缺少 ranking_groups。"
            )


def _validate_split_metric_values(
    split_metrics: dict[str, object],
    *,
    source: str,
    k_keys: set[str],
) -> None:
    for key in ("mae", "rmse", "spearman"):
        if key not in split_metrics:
            raise ValueError(f"LightGBM run metrics 的 {source} 缺少 {key}。")
    for key in ("mae", "rmse"):
        _validate_finite_metric_value(
            split_metrics[key],
            path=f"{source}.{key}",
            allow_none=False,
        )
    _validate_finite_metric_value(
        split_metrics["spearman"],
        path=f"{source}.spearman",
        allow_none=True,
    )
    for key in ("precision_at_k", "recall_at_k", "ndcg_at_k"):
        ranking_metrics = _require_mapping(
            split_metrics,
            key,
            f"LightGBM run metrics {source}",
        )
        missing_k = sorted(k_keys - set(ranking_metrics))
        if missing_k:
            raise ValueError(
                f"LightGBM run metrics 的 {source}.{key} 缺少 k={missing_k}。"
            )
        for k in k_keys:
            _validate_finite_metric_value(
                ranking_metrics[k],
                path=f"{source}.{key}.{k}",
                allow_none=key == "ndcg_at_k",
            )


def _require_mapping(
    payload: dict[str, object],
    key: str,
    source: str,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source} 的 {key} 必须是 object。")
    return value


def _validate_strict_json_payload(payload: dict[str, object]) -> None:
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("LightGBM run metrics 必须是 strict JSON 载荷。") from exc


def _validate_finite_metric_value(
    value: object,
    *,
    path: str,
    allow_none: bool,
) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LightGBM run metrics 的 {path} 必须是有限数值。")
    if not math.isfinite(float(value)):
        raise ValueError(f"LightGBM run metrics 的 {path} 必须是有限数值。")
