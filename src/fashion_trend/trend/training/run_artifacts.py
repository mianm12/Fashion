from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fashion_trend.foundation import logging as log
from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.foundation.io import (
    remove_file_if_exists,
    write_json_atomic,
    write_text_atomic,
)

LIGHTGBM_RUN_RESERVED_NAMES: frozenset[str] = frozenset(
    {"index.jsonl", "evaluations.jsonl"}
)
LIGHTGBM_RUN_ID_RETRY_LIMIT = 10
LIGHTGBM_RUN_ID_SUFFIX_LENGTH = 8
LIGHTGBM_PROMOTION_STATUSES: frozenset[str] = frozenset(
    {"not_requested", "succeeded", "failed"}
)


@dataclass(frozen=True)
class LightGBMRunSummary:
    run_id: str
    created_at: str
    run_dir: str
    promotion_status: str
    params_path: str
    metadata_path: str
    promotion_error: str | None = None


@dataclass(frozen=True)
class PromotionItem:
    final_path: Path
    payload: bytes | dict[str, object]


def validate_lightgbm_run_id(run_id: str) -> None:
    """校验 LightGBM run_id 能安全作为 runs/ 下的单级目录名。"""

    validate_safe_path_segment(run_id, "run_id")
    if run_id in LIGHTGBM_RUN_RESERVED_NAMES:
        raise ValueError(f"run_id 是保留名称，不能作为实验目录: {run_id}")


def generate_lightgbm_run_id(
    run_root: Path,
    *,
    now_factory: Callable[[], datetime] = datetime.now,
    token_factory: Callable[[], str] | None = None,
) -> str:
    """生成稳定格式的 LightGBM run_id，并避开已存在的 run 目录。"""

    timestamp = now_factory().strftime("%Y%m%d-%H%M%S")
    make_token = token_factory or (
        lambda: uuid.uuid4().hex[:LIGHTGBM_RUN_ID_SUFFIX_LENGTH]
    )
    for _attempt in range(LIGHTGBM_RUN_ID_RETRY_LIMIT):
        suffix = make_token()
        run_id = f"{timestamp}-{suffix}"
        validate_lightgbm_run_id(run_id)
        if not (run_root / run_id).exists():
            return run_id
    raise FileExistsError(
        "自动生成 lightgbm run_id 连续冲突 "
        f"{LIGHTGBM_RUN_ID_RETRY_LIMIT} 次: {run_root}"
    )


def build_lightgbm_run_summary(
    *,
    run_id: str,
    metadata: dict[str, object],
    promotion_status: str,
    promotion_error: str | None = None,
) -> LightGBMRunSummary:
    if promotion_status not in LIGHTGBM_PROMOTION_STATUSES:
        raise ValueError(f"未知 LightGBM promotion_status: {promotion_status}")
    return LightGBMRunSummary(
        run_id=run_id,
        created_at=str(metadata.get("created_at", "")),
        run_dir=str(metadata["run_dir"]),
        promotion_status=promotion_status,
        params_path=str(metadata["params_path"]),
        metadata_path=str(Path(str(metadata["run_dir"])) / "metadata.json"),
        promotion_error=promotion_error,
    )


def upsert_lightgbm_run_index(
    index_path: Path,
    summary: LightGBMRunSummary,
) -> None:
    summaries: dict[str, dict[str, object]] = {}
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            summaries[str(payload["run_id"])] = payload
    row: dict[str, object] = {
        "run_id": summary.run_id,
        "created_at": summary.created_at,
        "run_dir": summary.run_dir,
        "promotion_status": summary.promotion_status,
        "params_path": summary.params_path,
        "metadata_path": summary.metadata_path,
    }
    if summary.promotion_error is not None:
        row["promotion_error"] = summary.promotion_error
    summaries[summary.run_id] = row
    lines = [
        json.dumps(summaries[key], ensure_ascii=False, sort_keys=True)
        for key in sorted(summaries)
    ]
    write_text_atomic("\n".join(lines) + "\n", index_path)


def write_promotion_items_atomic(
    items: list[PromotionItem],
    staging_root: Path,
) -> None:
    staging_dir = staging_root / f".tmp-lightgbm-promotion-{uuid.uuid4().hex}"
    published_paths: list[tuple[Path, Path | None]] = []
    try:
        staged_paths: list[tuple[Path, Path]] = []
        for index, item in enumerate(items):
            suffix = item.final_path.suffix
            staging_path = staging_dir / f"item-{index}{suffix}"
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(item.payload, dict):
                write_json_atomic(item.payload, staging_path)
            else:
                staging_path.write_bytes(item.payload)
            staged_paths.append((item.final_path, staging_path))

        for final_path, staging_path in staged_paths:
            backup_path = None
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                backup_path = final_path.with_name(
                    f".{final_path.name}.bak-{uuid.uuid4().hex}"
                )
                final_path.replace(backup_path)
            published_paths.append((final_path, backup_path))
            staging_path.replace(final_path)
    except Exception:
        _rollback_promoted_outputs(published_paths)
        raise
    else:
        _remove_promotion_backups(published_paths)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def publish_lightgbm_run_to_stable(
    *,
    result,
    run_metadata: dict[str, object],
    run_context,
    stable_paths: dict[str, Path],
    include_metrics: bool = False,
    metrics_item: PromotionItem | None = None,
) -> dict[str, object]:
    from fashion_trend.trend.training.outputs import build_trend_train_metadata

    stable_metadata = build_trend_train_metadata(
        result,
        run_context,
        stable_paths,
        run_id=str(run_metadata["run_id"]),
        run_dir=Path(str(run_metadata["run_dir"])),
        stable_output_dir=stable_paths["output_dir"],
        promotion_requested=True,
    )
    items = [
        PromotionItem(
            stable_paths["predictions"],
            run_metadata_path(run_metadata, "predictions").read_bytes(),
        ),
        PromotionItem(
            stable_paths["params"],
            run_metadata_path(run_metadata, "params").read_bytes(),
        ),
    ]
    for artifact in result.artifacts:
        source_path = Path(str(run_metadata["run_dir"])) / artifact.relative_path
        items.append(
            PromotionItem(
                stable_paths["output_dir"] / artifact.relative_path,
                source_path.read_bytes(),
            )
        )
    items.append(PromotionItem(stable_paths["metadata"], stable_metadata))
    if include_metrics:
        if metrics_item is None:
            raise ValueError("发布 stable metrics 时必须提供 metrics_item。")
        items.append(metrics_item)
    write_promotion_items_atomic(items, stable_paths["output_dir"])
    return stable_metadata


def run_metadata_path(run_metadata: dict[str, object], artifact_name: str) -> Path:
    if artifact_name == "predictions":
        return Path(str(run_metadata["prediction_path"]))
    if artifact_name == "params":
        return Path(str(run_metadata["params_path"]))
    if artifact_name == "metadata":
        return Path(str(run_metadata["run_dir"])) / "metadata.json"
    raise ValueError(f"未知 LightGBM run artifact: {artifact_name}")


def record_lightgbm_promotion_failure(
    *,
    index_path: Path,
    summary: LightGBMRunSummary,
    run_dir: Path,
    stable_dir: Path,
    promotion_error: BaseException,
) -> None:
    try:
        upsert_lightgbm_run_index(index_path, summary)
    except Exception as index_error:
        log.error(
            "LightGBM promotion 失败，且 run index 更新失败: "
            f"run_dir={run_dir}, stable_dir={stable_dir}, "
            f"promotion_error={promotion_error}, index_error={index_error}",
            source="lightgbm-run-artifacts",
        )


def record_lightgbm_index_update_failure(
    *,
    run_dir: Path,
    stable_dir: Path,
    index_error: BaseException,
    attempted_status: str,
) -> None:
    log.error(
        "LightGBM promotion succeeded, but run index update failed: "
        f"run_dir={run_dir}, stable_dir={stable_dir}, "
        f"attempted_status={attempted_status}, index_error={index_error}",
        source="lightgbm-run-artifacts",
    )


def _rollback_promoted_outputs(
    published_paths: list[tuple[Path, Path | None]],
) -> None:
    for final_path, backup_path in reversed(published_paths):
        if backup_path is None:
            remove_file_if_exists(final_path)
            continue
        remove_file_if_exists(final_path)
        backup_path.replace(final_path)


def _remove_promotion_backups(
    published_paths: list[tuple[Path, Path | None]],
) -> None:
    for _final_path, backup_path in published_paths:
        if backup_path is not None:
            remove_file_if_exists(backup_path)
