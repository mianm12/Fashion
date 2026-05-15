from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.trend_graph_feature_ablation.paths import EXPERIMENT_ROOT
from fashion_trend.foundation.paths import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
from fashion_trend.trend.paths import (
    TREND_MODEL_SAMPLES_PATH,
    TREND_MODEL_SAMPLES_TEST_PATH,
    TREND_MODEL_SAMPLES_TRAIN_PATH,
    TREND_MODEL_SAMPLES_VALID_PATH,
)

HASH_CHUNK_SIZE = 1024 * 1024


def digest_json_payload(payload: Mapping[str, Any]) -> str:
    """返回 JSON payload 的稳定 SHA-256 摘要。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def digest_file(path: Path) -> str:
    """返回文件内容的 SHA-256 摘要。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_hash_entry(
    path: Path,
    *,
    required: bool = True,
    row_count: int | None = None,
) -> dict[str, object]:
    """构建输入 artifact 的可审计 hash 记录。"""

    artifact_path = Path(path)
    if not artifact_path.exists():
        if required:
            raise FileNotFoundError(f"必需输入 artifact 不存在: {artifact_path}")
        return {
            "path": str(artifact_path),
            "exists": False,
            "hash": None,
            "size": None,
            "mtime": None,
            "row_count": row_count,
        }

    stat = artifact_path.stat()
    return {
        "path": str(artifact_path),
        "exists": True,
        "hash": digest_file(artifact_path),
        "size": int(stat.st_size),
        "mtime": stat.st_mtime,
        "row_count": row_count,
    }


def assert_experiment_write_path(
    path: Path,
    *,
    root: Path = EXPERIMENT_ROOT,
) -> Path:
    """校验写入路径只能落在趋势图消融实验根目录内。"""

    output_path = Path(path).resolve(strict=False)
    allowed_root = Path(root).resolve(strict=False)

    _reject_forbidden_production_path(output_path)

    if not output_path.is_relative_to(allowed_root):
        raise ValueError(f"实验写入路径不在允许根目录内: {output_path}")
    return output_path


def prepare_output_path(path: Path, *, root: Path = EXPERIMENT_ROOT) -> Path:
    """校验实验输出路径并创建父目录。"""

    output_path = assert_experiment_write_path(path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _reject_forbidden_production_path(path: Path) -> None:
    forbidden_roots = (
        OUTPUT_DIR / "models" / "lightgbm",
        OUTPUT_DIR / "metrics" / "lightgbm",
        OUTPUT_DIR / "reports",
        OUTPUT_DIR / "defense_app",
        PROJECT_ROOT / "apps" / "defense_app",
        DATA_DIR / "processed" / "features",
    )
    forbidden_files = (
        TREND_MODEL_SAMPLES_PATH,
        TREND_MODEL_SAMPLES_TRAIN_PATH,
        TREND_MODEL_SAMPLES_VALID_PATH,
        TREND_MODEL_SAMPLES_TEST_PATH,
    )

    resolved_roots = [root.resolve(strict=False) for root in forbidden_roots]
    resolved_files = [file_path.resolve(strict=False) for file_path in forbidden_files]
    if path in resolved_files or any(
        path.is_relative_to(root) for root in resolved_roots
    ):
        raise ValueError(f"实验禁止写入稳定产物路径: {path}")
