# Trend Model Training Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the baseline-specific training entrypoint with a general trend model training framework that supports all future trend models while only implementing `last_week` now.

**Architecture:** Add a `TrendModelTrainer` protocol, registry, and common runner. Move `last_week` into a trainer module under `fashion_trend.models`, derive model output paths from `outputs/models/<model_name>/`, and keep the top-level CLI as orchestration only.

**Tech Stack:** Python 3.10-3.12, pandas, numpy, standard-library `argparse`, `dataclasses`, `typing.Protocol`, `unittest`, existing CSV/JSON/Parquet helpers in `fashion_trend.trend`.

---

## File Structure

- Create: `src/10_train_trend_model.py`
  - General CLI entrypoint. Parses `--model`, delegates to `run_trend_model_training()`, logs metadata summary, returns stable exit codes.
- Create: `src/fashion_trend/models/base.py`
  - Defines `TrendArtifact`, `TrendTrainContext`, `TrendTrainResult`, `TrendModelTrainer`, and known model type constants.
- Create: `src/fashion_trend/models/last_week.py`
  - Owns `last_week` constants, `predict_last_week()`, and `LastWeekTrainer`.
- Create: `src/fashion_trend/models/registry.py`
  - Owns model registration, `list_trend_model_names()`, `get_trend_model_trainer()`, and `UnknownTrendModelError`.
- Create: `src/fashion_trend/training.py`
  - Owns split loading, output path derivation, result validation, metadata construction, artifact validation, output writing, and runner.
- Delete: `src/10_train_trend_baseline.py`
  - Old entrypoint is not retained.
- Delete: `src/fashion_trend/models/baseline_last_week.py`
  - Replaced by `src/fashion_trend/models/last_week.py`.
- Modify: `src/fashion_trend/models/__init__.py`
  - Keep package docstring.
- Modify: `src/fashion_trend/trend.py`
  - Rename generic prediction contract from baseline-specific names to trend-model names.
- Modify: `src/fashion_trend/config.py`
  - Remove model-specific `PATH["output_model_last_week_*"]` entries; keep `OUTPUT_MODELS_DIR`.
- Modify: `tests/test_trend.py`
  - Update imports and tests to target the generic training framework.
- Modify: `README.md`
  - Use `uv run python src/10_train_trend_model.py --model last_week`.

Historical planning/spec files under `docs/superpowers/plans/2026-05-05-trend-last-week-baseline.md` and `docs/superpowers/specs/2026-05-04-trend-last-week-baseline-design.md` can remain as historical records unless the user explicitly asks to rewrite past docs.

Commit steps below are implementation checkpoints. Execute them only if the user explicitly authorizes commits.

---

### Task 1: Rename Generic Prediction Contract

**Files:**
- Modify: `src/fashion_trend/trend.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Write the failing import/test update**

In `tests/test_trend.py`, update imports and assertions so the prediction contract is model-wide, not baseline-specific. The `fashion_trend.trend` import block should include these names:

```python
from fashion_trend.trend import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
    TREND_MODEL_SPLIT_COLUMNS,
    build_article_week_sales_frame,
    build_attribute_graph_features_frame,
    build_attribute_week_heat_frame,
    build_attribute_week_target_frame,
    build_trend_model_samples_frame,
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    read_article_attribute_edges,
    read_attribute_hierarchy_edges,
    read_attribute_nodes,
    read_article_week_sales,
    read_attribute_week_target,
    read_trend_model_split,
    read_weekly_transactions,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_article_week_sales,
    validate_attribute_week_heat,
    validate_attribute_week_target,
    validate_trend_model_predictions,
    validate_trend_model_samples,
    validate_trend_model_split_frames,
    write_json,
    write_trend_csv,
    write_trend_parquet,
)
```

Update existing references:

```python
self.assertEqual(
    predictions.columns.tolist(), list(TREND_MODEL_PREDICTION_COLUMNS)
)

with self.assertRaisesRegex(ValueError, "趋势模型预测"):
    validate_trend_model_predictions(predictions, samples)
```

Replace every call to `validate_trend_baseline_predictions` with:

```python
validate_trend_model_predictions(predictions, samples)
```

- [ ] **Step 2: Run tests and verify the expected failure**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: ERROR with an import failure because `TREND_MODEL_PREDICTION_COLUMNS` and `validate_trend_model_predictions` do not exist yet.

- [ ] **Step 3: Rename the contract in `trend.py`**

In `src/fashion_trend/trend.py`, replace:

```python
TREND_BASELINE_PREDICTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "model_name",
    "split",
    "share_t",
    "pred_share_t1",
    "target_growth",
    "pred_target_growth",
    "target_rank_in_type_t1",
)
```

with:

```python
TREND_MODEL_PREDICTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "model_name",
    "split",
    "share_t",
    "pred_share_t1",
    "target_growth",
    "pred_target_growth",
    "target_rank_in_type_t1",
)
```

Replace `validate_trend_baseline_predictions()` with:

```python
def validate_trend_model_predictions(
    predictions: pd.DataFrame,
    split_samples: pd.DataFrame,
) -> None:
    """校验趋势模型预测表的列契约、split 对齐和数值可评价性。"""
    validate_required_columns(
        predictions.columns.tolist(),
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name="趋势模型预测表",
    )
    validate_no_missing_values(
        predictions,
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name="趋势模型预测表",
    )
    validate_unique_key(
        predictions,
        ["week_id", "attr_id", "model_name"],
        source_name="趋势模型预测表",
    )
    if not set(predictions["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("趋势模型预测表存在非法 split。")

    copied_sample_columns = (
        "week_id",
        "attr_id",
        "attr_type",
        "attr_value",
        "split",
        "share_t",
        "target_growth",
        "target_rank_in_type_t1",
    )
    validate_required_columns(
        split_samples.columns.tolist(),
        copied_sample_columns,
        source_name="趋势模型输入样本",
    )
    sorted_predictions = predictions.sort_values(
        ["week_id", "attr_id"],
        ignore_index=True,
    )
    sorted_samples = split_samples.sort_values(
        ["week_id", "attr_id"],
        ignore_index=True,
    )
    prediction_split = sorted_predictions.loc[:, ["week_id", "attr_id", "split"]]
    sample_split = sorted_samples.loc[:, ["week_id", "attr_id", "split"]]
    if not prediction_split.equals(sample_split):
        raise ValueError("趋势模型预测 split 与输入不一致。")

    prediction_copied_values = sorted_predictions.loc[:, list(copied_sample_columns)]
    sample_copied_values = sorted_samples.loc[:, list(copied_sample_columns)]
    if not prediction_copied_values.equals(sample_copied_values):
        raise ValueError("趋势模型预测字段与输入不一致。")

    numeric_values = sorted_predictions.drop(
        columns=["attr_id", "attr_type", "attr_value", "model_name", "split"]
    )
    try:
        finite_numeric_values = numeric_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势模型预测表无法校验数值字段。") from exc
    if not np.isfinite(finite_numeric_values).all():
        raise ValueError("趋势模型预测表存在非有限数值。")
```

- [ ] **Step 4: Run focused tests**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: PASS for the updated prediction contract tests.

- [ ] **Step 5: Optional commit checkpoint**

Only if commits are explicitly authorized:

```sh
git add src/fashion_trend/trend.py tests/test_trend.py
git commit -m "refactor(trend): 泛化趋势模型预测契约"
```

---

### Task 2: Add Model Base Types and Registry

**Files:**
- Create: `src/fashion_trend/models/base.py`
- Create: `src/fashion_trend/models/registry.py`
- Create: `src/fashion_trend/models/last_week.py`
- Delete: `src/fashion_trend/models/baseline_last_week.py`
- Modify: `src/fashion_trend/models/__init__.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Write failing tests for base types, trainer, and registry**

Update `tests/test_trend.py` imports:

```python
from fashion_trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.models.last_week import (
    LAST_WEEK_MODEL_NAME,
    LAST_WEEK_PARAMS,
    LastWeekTrainer,
    predict_last_week,
)
from fashion_trend.models.registry import (
    UnknownTrendModelError,
    get_trend_model_trainer,
    list_trend_model_names,
)
```

Replace the old `fashion_trend.models.baseline_last_week` import.

Add tests inside `LastWeekBaselineTests`:

```python
def test_registry_lists_last_week(self) -> None:
    self.assertEqual(list_trend_model_names(), (LAST_WEEK_MODEL_NAME,))

def test_registry_returns_last_week_trainer(self) -> None:
    trainer = get_trend_model_trainer(LAST_WEEK_MODEL_NAME)

    self.assertIsInstance(trainer, LastWeekTrainer)
    self.assertEqual(trainer.name, LAST_WEEK_MODEL_NAME)
    self.assertEqual(trainer.model_type, MODEL_TYPE_BASELINE)

def test_registry_rejects_unknown_model(self) -> None:
    with self.assertRaisesRegex(UnknownTrendModelError, "moving_average"):
        get_trend_model_trainer("moving_average")

def test_last_week_trainer_returns_train_result(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    context = TrendTrainContext(
        model_name=LAST_WEEK_MODEL_NAME,
        split_frames=split_frames,
        input_paths={
            "train": Path("train.parquet"),
            "valid": Path("valid.parquet"),
            "test": Path("test.parquet"),
        },
        output_dir=Path("outputs/models/last_week"),
    )

    result = LastWeekTrainer().train(context)

    self.assertIsInstance(result, TrendTrainResult)
    self.assertEqual(result.model_name, LAST_WEEK_MODEL_NAME)
    self.assertEqual(result.model_type, MODEL_TYPE_BASELINE)
    self.assertEqual(result.params, LAST_WEEK_PARAMS)
    self.assertEqual(result.artifacts, ())
    self.assertEqual(result.metadata, {})
    self.assertEqual(len(result.predictions), 40)
```

- [ ] **Step 2: Run tests and verify the expected failure**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: ERROR because `models.base`, `models.last_week`, and `models.registry` do not exist yet.

- [ ] **Step 3: Create `models/base.py`**

Create `src/fashion_trend/models/base.py`:

```python
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
```

- [ ] **Step 4: Create `models/last_week.py`**

Move the current formula from `src/fashion_trend/models/baseline_last_week.py` into `src/fashion_trend/models/last_week.py` with this shape:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.trend import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
    validate_required_columns,
    validate_trend_model_predictions,
)

LAST_WEEK_MODEL_NAME = "last_week"
LAST_WEEK_PARAMS: dict[str, object] = {
    "model_name": LAST_WEEK_MODEL_NAME,
    "formula": "pred_target_growth = growth_lag_1",
    "derived_formula": (
        "pred_share_t1 = exp(pred_target_growth) * "
        "(share_t + epsilon) - epsilon"
    ),
    "epsilon": 1e-6,
}

LAST_WEEK_REQUIRED_COLUMNS: tuple[str, ...] = (
    "split",
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "share_t",
    "growth_lag_1",
    "target_growth",
    "target_rank_in_type_t1",
)


def predict_last_week(split_samples: pd.DataFrame) -> pd.DataFrame:
    """用上一样本周占比增长 growth_lag_1 预测下一段 target_growth。"""
    missing_columns = sorted(
        set(LAST_WEEK_REQUIRED_COLUMNS) - set(split_samples.columns)
    )
    if missing_columns:
        raise ValueError(
            "last_week 模型输入样本缺少必需列: " + ", ".join(missing_columns)
        )
    validate_required_columns(
        split_samples.columns.tolist(),
        LAST_WEEK_REQUIRED_COLUMNS,
        source_name="last_week 模型输入样本",
    )
    if not set(split_samples["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("last_week 模型输入样本存在非法 split。")

    predictions = split_samples.loc[
        :,
        [
            "week_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "split",
            "share_t",
            "growth_lag_1",
            "target_growth",
            "target_rank_in_type_t1",
        ],
    ].copy()
    predictions.insert(4, "model_name", LAST_WEEK_MODEL_NAME)
    predictions["pred_target_growth"] = predictions["growth_lag_1"]
    epsilon = float(LAST_WEEK_PARAMS["epsilon"])
    predictions["pred_share_t1"] = (
        np.exp(predictions["pred_target_growth"]) * (predictions["share_t"] + epsilon)
        - epsilon
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    return predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )


class LastWeekTrainer:
    name = LAST_WEEK_MODEL_NAME
    model_type = MODEL_TYPE_BASELINE

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        split_samples = pd.concat(
            [context.split_frames[split_name] for split_name in context.split_order],
            ignore_index=True,
        )
        predictions = predict_last_week(split_samples)
        validate_trend_model_predictions(predictions, split_samples)
        return TrendTrainResult(
            model_name=self.name,
            model_type=self.model_type,
            predictions=predictions,
            params=dict(LAST_WEEK_PARAMS),
        )
```

- [ ] **Step 5: Create `models/registry.py`**

Create `src/fashion_trend/models/registry.py`:

```python
from __future__ import annotations

from fashion_trend.models.base import TrendModelTrainer
from fashion_trend.models.last_week import LAST_WEEK_MODEL_NAME, LastWeekTrainer


class UnknownTrendModelError(ValueError):
    """Raised when a requested trend model is not registered."""


TREND_MODEL_REGISTRY: dict[str, TrendModelTrainer] = {
    LAST_WEEK_MODEL_NAME: LastWeekTrainer(),
}


def list_trend_model_names() -> tuple[str, ...]:
    return tuple(sorted(TREND_MODEL_REGISTRY))


def get_trend_model_trainer(model_name: str) -> TrendModelTrainer:
    try:
        return TREND_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(list_trend_model_names())
        raise UnknownTrendModelError(
            f"不支持的趋势模型: {model_name}。可用模型: {available}"
        ) from exc
```

- [ ] **Step 6: Update package init and remove old module**

Set `src/fashion_trend/models/__init__.py` to:

```python
"""Trend model implementations and registry."""
```

Delete:

```text
src/fashion_trend/models/baseline_last_week.py
```

- [ ] **Step 7: Run focused tests**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: PASS.

- [ ] **Step 8: Optional commit checkpoint**

Only if commits are explicitly authorized:

```sh
git add src/fashion_trend/models tests/test_trend.py
git rm src/fashion_trend/models/baseline_last_week.py
git commit -m "refactor(trend): 添加趋势模型 registry"
```

---

### Task 3: Add Training Runner and Output Contract

**Files:**
- Create: `src/fashion_trend/training.py`
- Modify: `src/fashion_trend/config.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Write failing tests for runner metadata and validation**

Update imports in `tests/test_trend.py`:

```python
from fashion_trend.config import OUTPUT_MODELS_DIR
from fashion_trend.models.base import TrendArtifact
from fashion_trend.training import (
    build_trend_train_metadata,
    derive_trend_model_output_paths,
    run_trend_model_training,
    validate_trend_train_result,
    write_trend_model_outputs,
)
```

Add tests:

```python
def test_derive_trend_model_output_paths_uses_model_name(self) -> None:
    paths = derive_trend_model_output_paths("last_week", Path("outputs/models"))

    self.assertEqual(paths["output_dir"], Path("outputs/models/last_week"))
    self.assertEqual(paths["predictions"], Path("outputs/models/last_week/predictions.csv"))
    self.assertEqual(paths["params"], Path("outputs/models/last_week/params.json"))
    self.assertEqual(paths["metadata"], Path("outputs/models/last_week/metadata.json"))

def test_validate_trend_train_result_rejects_wrong_model_name(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    samples = pd.concat(split_frames.values(), ignore_index=True)
    result = TrendTrainResult(
        model_name="wrong",
        model_type=MODEL_TYPE_BASELINE,
        predictions=predict_last_week(samples),
        params=dict(LAST_WEEK_PARAMS),
    )
    context = TrendTrainContext(
        model_name=LAST_WEEK_MODEL_NAME,
        split_frames=split_frames,
        input_paths={"train": Path("train.parquet"), "valid": Path("valid.parquet"), "test": Path("test.parquet")},
        output_dir=Path("outputs/models/last_week"),
    )

    with self.assertRaisesRegex(ValueError, "model_name"):
        validate_trend_train_result(result, context)

def test_build_trend_train_metadata_rejects_core_key_override(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    context = TrendTrainContext(
        model_name=LAST_WEEK_MODEL_NAME,
        split_frames=split_frames,
        input_paths={"train": Path("train.parquet"), "valid": Path("valid.parquet"), "test": Path("test.parquet")},
        output_dir=Path("outputs/models/last_week"),
    )
    result = LastWeekTrainer().train(context)
    result = TrendTrainResult(
        model_name=result.model_name,
        model_type=result.model_type,
        predictions=result.predictions,
        params=result.params,
        metadata={"rows": 999},
    )
    paths = derive_trend_model_output_paths("last_week", Path("outputs/models"))

    with self.assertRaisesRegex(ValueError, "metadata"):
        build_trend_train_metadata(result, context, paths)

def test_write_trend_model_outputs_rejects_unsafe_artifact_path_before_writing(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    with TemporaryDirectory() as tmp_dir:
        output_root = Path(tmp_dir) / "models"
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={"train": Path("train.parquet"), "valid": Path("valid.parquet"), "test": Path("test.parquet")},
            output_dir=output_root / "last_week",
        )
        result = LastWeekTrainer().train(context)
        bad_result = TrendTrainResult(
            model_name=result.model_name,
            model_type=result.model_type,
            predictions=result.predictions,
            params=result.params,
            artifacts=(TrendArtifact("../leak.txt", "binary", b"bad"),),
        )
        paths = derive_trend_model_output_paths("last_week", output_root)
        metadata = build_trend_train_metadata(bad_result, context, paths)

        with self.assertRaisesRegex(ValueError, "artifact"):
            write_trend_model_outputs(bad_result, metadata, paths)

        self.assertFalse(paths["predictions"].exists())

def test_run_trend_model_training_writes_standard_outputs(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_paths = {
            "train": tmp_path / "trend_model_samples_train.parquet",
            "valid": tmp_path / "trend_model_samples_valid.parquet",
            "test": tmp_path / "trend_model_samples_test.parquet",
        }
        for split_name, split_frame in split_frames.items():
            write_trend_parquet(split_frame, input_paths[split_name])

        metadata = run_trend_model_training(
            LAST_WEEK_MODEL_NAME,
            input_paths=input_paths,
            output_root=tmp_path / "outputs" / "models",
        )

        output_dir = tmp_path / "outputs" / "models" / "last_week"
        self.assertTrue((output_dir / "predictions.csv").exists())
        self.assertTrue((output_dir / "params.json").exists())
        self.assertTrue((output_dir / "metadata.json").exists())
        self.assertEqual(metadata["model_name"], LAST_WEEK_MODEL_NAME)
        self.assertEqual(metadata["model_type"], MODEL_TYPE_BASELINE)
        self.assertEqual(metadata["rows"], 40)
        self.assertEqual(metadata["extra_artifacts"], [])
```

- [ ] **Step 2: Run tests and verify the expected failure**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: ERROR because `fashion_trend.training` does not exist yet.

- [ ] **Step 3: Simplify model-specific config paths**

In `src/fashion_trend/config.py`, remove these `PATH` keys:

```python
"output_model_last_week_dir": OUTPUT_MODELS_DIR / "last_week",
"output_model_last_week_predictions": OUTPUT_MODELS_DIR / "last_week" / "predictions.csv",
"output_model_last_week_params": OUTPUT_MODELS_DIR / "last_week" / "params.json",
"output_model_last_week_metadata": OUTPUT_MODELS_DIR / "last_week" / "metadata.json",
```

Keep:

```python
OUTPUT_MODELS_DIR = OUTPUT_DIR / "models"
```

- [ ] **Step 4: Create `training.py`**

Create `src/fashion_trend/training.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from fashion_trend.config import OUTPUT_MODELS_DIR, PATH
from fashion_trend.models.base import (
    KNOWN_MODEL_TYPES,
    TrendArtifact,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.models.registry import get_trend_model_trainer
from fashion_trend.trend import (
    TREND_MODEL_SPLIT_VALUES,
    read_trend_model_split,
    validate_trend_model_predictions,
    write_json,
    write_trend_csv,
)


def default_trend_model_input_paths() -> dict[str, Path]:
    return {
        "train": PATH["features_trend_model_samples_train"],
        "valid": PATH["features_trend_model_samples_valid"],
        "test": PATH["features_trend_model_samples_test"],
    }


def derive_trend_model_output_paths(
    model_name: str,
    output_root: Path = OUTPUT_MODELS_DIR,
) -> dict[str, Path]:
    output_dir = output_root / model_name
    return {
        "output_dir": output_dir,
        "predictions": output_dir / "predictions.csv",
        "params": output_dir / "params.json",
        "metadata": output_dir / "metadata.json",
    }


def read_trend_model_split_frames(
    input_paths: Mapping[str, Path],
) -> dict[str, pd.DataFrame]:
    missing_splits = set(TREND_MODEL_SPLIT_VALUES) - set(input_paths)
    if missing_splits:
        raise ValueError(f"趋势模型输入路径缺少 split: {sorted(missing_splits)}")
    return {
        split_name: read_trend_model_split(input_paths[split_name])
        for split_name in TREND_MODEL_SPLIT_VALUES
    }


def validate_trend_train_result(
    result: TrendTrainResult,
    context: TrendTrainContext,
) -> None:
    if result.model_name != context.model_name:
        raise ValueError(
            "趋势模型训练结果 model_name 与请求不一致: "
            f"result={result.model_name}, context={context.model_name}"
        )
    if result.model_type not in KNOWN_MODEL_TYPES:
        raise ValueError(f"趋势模型训练结果存在未知 model_type: {result.model_type}")
    if not isinstance(result.params, dict):
        raise ValueError("趋势模型训练结果 params 必须是字典。")

    split_samples = pd.concat(
        [context.split_frames[split_name] for split_name in context.split_order],
        ignore_index=True,
    )
    validate_trend_model_predictions(result.predictions, split_samples)
    _validate_artifacts(result.artifacts)


def build_trend_train_metadata(
    result: TrendTrainResult,
    context: TrendTrainContext,
    output_paths: Mapping[str, Path],
) -> dict[str, object]:
    split_metadata: dict[str, dict[str, object]] = {}
    for split_name in context.split_order:
        split_predictions = result.predictions[result.predictions["split"] == split_name]
        if split_predictions.empty:
            raise ValueError(f"趋势模型 metadata 缺少 {split_name} split。")
        split_metadata[split_name] = {
            "rows": int(len(split_predictions)),
            "weeks": int(split_predictions["week_id"].nunique()),
            "attributes": int(split_predictions["attr_id"].nunique()),
            "week_min": int(split_predictions["week_id"].min()),
            "week_max": int(split_predictions["week_id"].max()),
        }

    core_metadata: dict[str, object] = {
        "model_name": result.model_name,
        "model_type": result.model_type,
        "input_paths": {
            split_name: str(context.input_paths[split_name])
            for split_name in context.split_order
        },
        "output_dir": str(output_paths["output_dir"]),
        "prediction_path": str(output_paths["predictions"]),
        "params_path": str(output_paths["params"]),
        "rows": int(len(result.predictions)),
        "weeks": int(result.predictions["week_id"].nunique()),
        "attributes": int(result.predictions["attr_id"].nunique()),
        "splits": split_metadata,
        "extra_artifacts": [
            {"path": artifact.relative_path, "kind": artifact.kind}
            for artifact in result.artifacts
        ],
    }

    overlapping_keys = sorted(set(core_metadata) & set(result.metadata))
    if overlapping_keys:
        raise ValueError(
            "趋势模型 metadata 不能覆盖 runner 核心字段: "
            + ", ".join(overlapping_keys)
        )
    return {**core_metadata, **result.metadata}


def write_trend_model_outputs(
    result: TrendTrainResult,
    metadata: dict[str, object],
    output_paths: Mapping[str, Path],
) -> None:
    _validate_artifacts(result.artifacts)
    write_trend_csv(result.predictions, output_paths["predictions"])
    write_json(result.params, output_paths["params"])
    for artifact in result.artifacts:
        _write_artifact(artifact, output_paths["output_dir"])
    write_json(metadata, output_paths["metadata"])


def run_trend_model_training(
    model_name: str,
    input_paths: Mapping[str, Path] | None = None,
    output_root: Path = OUTPUT_MODELS_DIR,
) -> dict[str, object]:
    trainer = get_trend_model_trainer(model_name)
    resolved_input_paths = dict(input_paths or default_trend_model_input_paths())
    split_frames = read_trend_model_split_frames(resolved_input_paths)
    output_paths = derive_trend_model_output_paths(model_name, output_root)
    context = TrendTrainContext(
        model_name=model_name,
        split_frames=split_frames,
        input_paths=resolved_input_paths,
        output_dir=output_paths["output_dir"],
    )
    result = trainer.train(context)
    validate_trend_train_result(result, context)
    metadata = build_trend_train_metadata(result, context, output_paths)
    write_trend_model_outputs(result, metadata, output_paths)
    return metadata


def _validate_artifacts(artifacts: tuple[TrendArtifact, ...]) -> None:
    for artifact in artifacts:
        artifact_path = Path(artifact.relative_path)
        if (
            not artifact.relative_path
            or artifact_path.is_absolute()
            or ".." in artifact_path.parts
            or artifact_path == Path(".")
        ):
            raise ValueError(f"趋势模型 artifact 路径不安全: {artifact.relative_path}")


def _write_artifact(artifact: TrendArtifact, output_dir: Path) -> None:
    output_path = output_dir / artifact.relative_path
    if isinstance(artifact.payload, pd.DataFrame):
        write_trend_csv(artifact.payload, output_path)
        return
    if isinstance(artifact.payload, dict):
        write_json(artifact.payload, output_path)
        return
    if isinstance(artifact.payload, bytes):
        _write_binary(artifact.payload, output_path)
        return
    raise ValueError(f"不支持的趋势模型 artifact payload: {artifact.relative_path}")


def _write_binary(payload: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        tmp_output_path.write_bytes(payload)
        tmp_output_path.replace(output_path)
    except Exception:
        try:
            tmp_output_path.unlink()
        except FileNotFoundError:
            pass
        raise
```

- [ ] **Step 5: Run focused tests**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: PASS.

- [ ] **Step 6: Optional commit checkpoint**

Only if commits are explicitly authorized:

```sh
git add src/fashion_trend/config.py src/fashion_trend/training.py tests/test_trend.py
git commit -m "feat(trend): 添加通用模型训练 runner"
```

---

### Task 4: Replace CLI Entrypoint

**Files:**
- Create: `src/10_train_trend_model.py`
- Delete: `src/10_train_trend_baseline.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Write failing CLI tests**

Replace tests that import `10_train_trend_baseline` with `10_train_trend_model`.

Add:

```python
def test_train_trend_model_main_preserves_argparse_usage_error_code(self) -> None:
    train_model = importlib.import_module("10_train_trend_model")

    self.assertEqual(train_model.main(["--unknown"]), 2)

def test_train_trend_model_main_rejects_unknown_model(self) -> None:
    train_model = importlib.import_module("10_train_trend_model")

    self.assertEqual(train_model.main(["--model", "moving_average"]), 1)
```

Remove old tests that directly patch `train_baseline.build_prediction_metadata`, because metadata is now owned by `fashion_trend.training`.

- [ ] **Step 2: Run tests and verify the expected failure**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: ERROR because `10_train_trend_model` does not exist yet.

- [ ] **Step 3: Create `src/10_train_trend_model.py`**

Create:

```python
from __future__ import annotations

import argparse
from typing import Sequence

from fashion_trend import log
from fashion_trend.models.registry import UnknownTrendModelError, list_trend_model_names
from fashion_trend.training import run_trend_model_training
from fashion_trend.trend import TREND_MODEL_SPLIT_VALUES

LOG_SOURCE = "trend-model-train"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析趋势模型训练入口参数。"""
    parser = argparse.ArgumentParser(description="训练趋势预测模型并写出预测结果。")
    parser.add_argument(
        "--model",
        required=True,
        help="需要训练的趋势模型名称。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """运行趋势模型训练 CLI，保留 argparse 用法错误码并记录运行摘要。"""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        metadata = run_trend_model_training(args.model)
    except UnknownTrendModelError as exc:
        available = ", ".join(list_trend_model_names())
        log.error(f"{exc}。可用模型: {available}", source=LOG_SOURCE)
        return 1
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"模型名称: {metadata['model_name']}", source=LOG_SOURCE)
    log.info(f"模型类型: {metadata['model_type']}", source=LOG_SOURCE)
    log.info(f"预测行数: {metadata['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖样本周数: {metadata['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {metadata['attributes']:,}", source=LOG_SOURCE)
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_stats = metadata["splits"][split_name]
        log.info(
            f"{split_name} 预测: rows={split_stats['rows']:,}, "
            f"weeks={split_stats['weeks']:,}, "
            f"attributes={split_stats['attributes']:,}, "
            f"week_range={split_stats['week_min']}..{split_stats['week_max']}",
            source=LOG_SOURCE,
        )
    log.info(f"输出目录: {metadata['output_dir']}", source=LOG_SOURCE)
    log.info(f"预测输出文件: {metadata['prediction_path']}", source=LOG_SOURCE)
    log.info(f"参数输出文件: {metadata['params_path']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Delete old CLI**

Delete:

```text
src/10_train_trend_baseline.py
```

- [ ] **Step 5: Run focused tests**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: PASS.

- [ ] **Step 6: Optional commit checkpoint**

Only if commits are explicitly authorized:

```sh
git add src/10_train_trend_model.py tests/test_trend.py
git rm src/10_train_trend_baseline.py
git commit -m "feat(trend): 替换趋势模型训练入口"
```

---

### Task 5: Update README and Final Verification

**Files:**
- Modify: `README.md`
- Test: repository verification commands

- [ ] **Step 1: Update pipeline command in README**

In `README.md`, replace:

```sh
uv run python src/10_train_trend_baseline.py --model last_week
```

with:

```sh
uv run python src/10_train_trend_model.py --model last_week
```

This appears in the pipeline command block and in the `last_week baseline` section.

- [ ] **Step 2: Update last_week wording**

Keep the stage name `Last Week baseline`, but describe the entrypoint as the general trend model trainer:

```markdown
`last_week` baseline 通过通用趋势模型训练入口运行，模型细节位于 `src/fashion_trend/models/last_week.py`。
```

- [ ] **Step 3: Run old-reference scan**

Run:

```sh
rg -n "10_train_trend_baseline|baseline_last_week|output_model_last_week" README.md src tests
```

Expected: no matches.

- [ ] **Step 4: Run unit tests**

Run:

```sh
uv run python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Run compile check**

Run:

```sh
uv run python -m py_compile src/09_split_trend_model_samples.py src/10_train_trend_model.py src/fashion_trend/models/base.py src/fashion_trend/models/registry.py src/fashion_trend/models/last_week.py src/fashion_trend/training.py src/fashion_trend/trend.py src/fashion_trend/config.py
```

Expected: exit code 0.

- [ ] **Step 6: Run real data training**

Run:

```sh
uv run python src/10_train_trend_model.py --model last_week
```

Expected: logs show model `last_week`, model type `baseline`, and output files under `outputs/models/last_week/`.

- [ ] **Step 7: Inspect generated outputs**

Run:

```sh
uv run python -c "import json, numpy as np, pandas as pd; base='data/processed/features'; train=pd.read_parquet(f'{base}/trend_model_samples_train.parquet'); valid=pd.read_parquet(f'{base}/trend_model_samples_valid.parquet'); test=pd.read_parquet(f'{base}/trend_model_samples_test.parquet'); pred=pd.read_csv('outputs/models/last_week/predictions.csv'); meta=json.load(open('outputs/models/last_week/metadata.json', encoding='utf-8')); params=json.load(open('outputs/models/last_week/params.json', encoding='utf-8')); samples=pd.concat([train, valid, test], ignore_index=True); merged=pred.merge(samples[['week_id','attr_id','growth_lag_1','split']], on=['week_id','attr_id','split'], how='left'); print({'train_rows': len(train), 'valid_rows': len(valid), 'test_rows': len(test), 'pred_rows': len(pred), 'metadata_rows': meta['rows'], 'model_name': meta['model_name'], 'model_type': meta['model_type'], 'splits': sorted(pred.split.unique().tolist()), 'model_names': sorted(pred.model_name.unique().tolist()), 'pred_missing': int(pred.isna().sum().sum()), 'pred_formula_ok': bool(np.allclose(merged['pred_target_growth'], merged['growth_lag_1'])), 'params_epsilon': params['epsilon']})"
```

Expected:

```text
{
  'train_rows': 49728,
  'valid_rows': 4736,
  'test_rows': 4736,
  'pred_rows': 59200,
  'metadata_rows': 59200,
  'model_name': 'last_week',
  'model_type': 'baseline',
  'splits': ['test', 'train', 'valid'],
  'model_names': ['last_week'],
  'pred_missing': 0,
  'pred_formula_ok': True,
  'params_epsilon': 1e-06
}
```

If row counts differ because upstream data changed, the invariant still holds: `pred_rows == train_rows + valid_rows + test_rows`, metadata rows match predictions, no missing values, and formula check is true.

- [ ] **Step 8: Review diff**

Run:

```sh
git diff --stat
git diff -- src/10_train_trend_model.py src/fashion_trend/models/base.py src/fashion_trend/models/registry.py src/fashion_trend/models/last_week.py src/fashion_trend/training.py src/fashion_trend/trend.py src/fashion_trend/config.py tests/test_trend.py README.md
```

Expected: diff is limited to the model training framework, tests, and README updates.

- [ ] **Step 9: Optional commit checkpoint**

Only if commits are explicitly authorized:

```sh
git add README.md src/10_train_trend_model.py src/fashion_trend/config.py src/fashion_trend/models src/fashion_trend/training.py src/fashion_trend/trend.py tests/test_trend.py
git rm src/10_train_trend_baseline.py src/fashion_trend/models/baseline_last_week.py
git commit -m "refactor(trend): 搭建通用模型训练框架"
```

---

## Self-Review

- Spec coverage: covered trainer interface, registry, runner, derived output paths, metadata, artifacts, no old entrypoint compatibility, README command update, and verification.
- Placeholder scan: no `TBD`, `TODO`, or "implement later" placeholders remain.
- Type consistency: plan uses `TrendTrainContext`, `TrendTrainResult`, `TrendArtifact`, `TrendModelTrainer`, `LastWeekTrainer`, `run_trend_model_training`, and `validate_trend_model_predictions` consistently across tasks.
- Scope check: this plan implements the framework and migrates only `last_week`; it does not add other models, metrics, recommendation code, or experiment config.
