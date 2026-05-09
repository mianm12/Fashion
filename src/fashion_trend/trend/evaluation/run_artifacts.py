from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fashion_trend.foundation.io import write_text_atomic


def read_run_id_from_model_metadata(metadata_path: Path) -> str | None:
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
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
