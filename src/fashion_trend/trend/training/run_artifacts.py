from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fashion_trend.foundation.artifacts import validate_safe_path_segment

LIGHTGBM_RUN_RESERVED_NAMES: frozenset[str] = frozenset(
    {"index.jsonl", "evaluations.jsonl"}
)
LIGHTGBM_RUN_ID_RETRY_LIMIT = 10
LIGHTGBM_RUN_ID_SUFFIX_LENGTH = 8


@dataclass(frozen=True)
class LightGBMRunSummary:
    run_id: str
    created_at: str
    run_dir: str
    promotion_status: str
    params_path: str
    metadata_path: str
    promotion_error: str | None = None


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
