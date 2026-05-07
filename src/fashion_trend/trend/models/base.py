from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

import pandas as pd

MODEL_TYPE_BASELINE = "baseline"
MODEL_TYPE_SUPERVISED = "supervised"
KNOWN_MODEL_TYPES: tuple[str, ...] = (MODEL_TYPE_BASELINE, MODEL_TYPE_SUPERVISED)


@dataclass(frozen=True)
class TrendArtifact:
    relative_path: str
    kind: str
    payload: pd.DataFrame | dict[str, object] | bytes


@dataclass(frozen=True)
class TrendTrainContext:
    model_name: str
    split_frames: Mapping[str, pd.DataFrame]
    input_paths: Mapping[str, Path]
    output_dir: Path
    split_order: tuple[str, ...] = ("train", "valid", "test")


@dataclass(frozen=True)
class TrendTrainResult:
    model_name: str
    model_type: str
    predictions: pd.DataFrame
    params: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)
    artifacts: tuple[TrendArtifact, ...] = ()


class TrendModelTrainer(Protocol):
    name: str
    model_type: str

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        raise NotImplementedError
