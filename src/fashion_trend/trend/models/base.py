from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

import pandas as pd

# 模型类型由通用训练 runner 校验，避免未登记类型写入标准训练产物。
MODEL_TYPE_BASELINE = "baseline"
MODEL_TYPE_SUPERVISED = "supervised"
KNOWN_MODEL_TYPES: tuple[str, ...] = (MODEL_TYPE_BASELINE, MODEL_TYPE_SUPERVISED)


@dataclass(frozen=True)
class TrendArtifact:
    """训练器随标准产物一起交给 runner 写出的附加产物。"""

    relative_path: str
    kind: str
    payload: pd.DataFrame | dict[str, object] | bytes


@dataclass(frozen=True)
class TrendTrainContext:
    """通用训练 runner 传给具体趋势模型训练器的输入上下文。"""

    model_name: str
    split_frames: Mapping[str, pd.DataFrame]
    input_paths: Mapping[str, Path]
    output_dir: Path
    split_order: tuple[str, ...] = ("train", "valid", "test")


@dataclass(frozen=True)
class TrendTrainResult:
    """具体训练器返回给通用 runner 的标准趋势训练结果。"""

    model_name: str
    model_type: str
    predictions: pd.DataFrame
    params: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)
    artifacts: tuple[TrendArtifact, ...] = ()


class TrendModelTrainer(Protocol):
    """趋势模型训练器协议，供注册表和通用 runner 统一调用。"""

    name: str
    model_type: str

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        raise NotImplementedError
