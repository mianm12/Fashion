# LightGBM 趋势主模型实施计划

> **给 agentic worker 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，并按任务逐项执行。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标：** 实现 `lightgbm` 趋势主模型，让它复用现有 `10_train_trend_model.py --model lightgbm` 和 `11_eval_trend_model.py --model lightgbm` 入口，输出标准预测、趋势评价、模型文件和可解释产物。

**架构：** 新增 `src/fashion_trend/trend/models/supervised/lightgbm.py` 作为唯一 LightGBM 训练事实来源，并在 registry 中注册 `lightgbm`。通用 runner 继续负责读取 split、校验 `TrendTrainResult`、构建 metadata 和原子写盘；LightGBM trainer 负责特征准备、延迟导入原生包、拟合、预测、诊断 metadata 和产物载荷。

**技术栈：** Python 3.10-3.12、pandas、numpy、LightGBM `LGBMRegressor`、pytest、现有 `fashion_trend.trend.models` / `training` / `evaluation` / `predictions` 契约。

---

## 文件结构

- 新建：`src/fashion_trend/trend/models/supervised/lightgbm.py`
  - 负责 LightGBM 常量、特征准备、延迟导入、训练、预测、诊断、特征重要性和 `LightGBMTrendTrainer`。
- 修改：`src/fashion_trend/trend/models/registry.py`
  - 注册 `lightgbm`，并保持 baseline trainer 可在没有 LightGBM 原生运行时依赖时正常导入。
- 新建：`tests/test_trend_lightgbm.py`
  - 覆盖监督模型专属行为：特征清单、稳定分类编码、延迟导入边界、诊断、特征重要性、产物和错误路径。
- 修改：`tests/test_trend_training.py`
  - 只补 registry、runner 和 CLI 对 `lightgbm` 的通用接入点。
- 修改：`tests/test_trend_evaluation.py`
  - 确认 `lightgbm` 的标准预测表能复用评价 runner。
- 修改：`README.md`
  - 同步当前阶段、命令、产物路径和验收说明。
- 修改：`docs/gpt-research/implementation-plan.md`
  - 把 LightGBM 从后续计划更新为当前主模型实现边界。

每个任务结尾包含 commit 命令。只有在实现阶段用户明确授权提交时才执行 commit；否则保留为人工检查点。

---

### 任务 1：添加 LightGBM 单元测试骨架

**文件：**
- 新建：`tests/test_trend_lightgbm.py`
- 阅读：`docs/superpowers/specs/2026-05-08-lightgbm-main-model-design.md`

- [ ] **步骤 1：为常量和导入边界写失败测试**

创建 `tests/test_trend_lightgbm.py`，初始内容如下：

```python
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.trend.models.base import MODEL_TYPE_SUPERVISED


LIGHTGBM_MODULE = "fashion_trend.trend.models.supervised.lightgbm"
LIGHTGBM_SOURCE = Path("src/fashion_trend/trend/models/supervised/lightgbm.py")


class TestLightGBMTrendModel:
    def test_lightgbm_module_does_not_import_native_package_at_top_level(self) -> None:
        source = LIGHTGBM_SOURCE.read_text(encoding="utf-8")
        top_level_source = source.split("def _fit_lightgbm_model", maxsplit=1)[0]

        assert "import lightgbm" not in top_level_source
        assert "from lightgbm" not in top_level_source

    def test_lightgbm_constants_are_stable(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)

        assert lightgbm_model.LIGHTGBM_MODEL_NAME == "lightgbm"
        assert lightgbm_model.LIGHTGBM_TARGET_COLUMN == "target_growth"
        assert lightgbm_model.LIGHTGBM_EPSILON == 1e-6
        assert lightgbm_model.LIGHTGBM_CATEGORICAL_FEATURES == ("attr_type",)
        assert lightgbm_model.LIGHTGBM_ALLOWED_OBJECTIVES == (
            "regression",
            "regression_l1",
        )
        assert lightgbm_model.LIGHTGBM_PARAMS == {
            "objective": "regression",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
        }
        assert lightgbm_model.LIGHTGBM_EARLY_STOPPING == {"stopping_rounds": 30}
        assert lightgbm_model.LIGHTGBM_NUMERIC_FEATURES == (
            "heat_t",
            "share_t",
            "log_heat_t",
            "rank_in_type_t",
            "heat_lag_1",
            "heat_lag_2",
            "heat_lag_3",
            "heat_lag_4",
            "share_lag_1",
            "share_lag_2",
            "share_lag_3",
            "share_lag_4",
            "growth_lag_1",
            "growth_lag_2",
            "acc_lag_1",
            "heat_ma_4",
            "share_ma_4",
            "share_std_4",
            "share_max_4",
            "share_min_4",
            "article_count",
            "is_core_attr",
            "parent_count",
            "child_count",
            "degree",
            "history_total_heat_t",
            "history_active_weeks_t",
            "is_trend_eligible_t",
            "week_index",
            "week_mod_52",
        )
        assert lightgbm_model.LIGHTGBM_EXCLUDED_COLUMNS == (
            "attr_id",
            "attr_value",
            "target_growth",
            "target_log_heat_t1",
            "target_rank_in_type_t1",
            "split",
        )

    def test_lightgbm_trainer_metadata(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        trainer = lightgbm_model.LightGBMTrendTrainer()

        assert trainer.name == "lightgbm"
        assert trainer.model_type == MODEL_TYPE_SUPERVISED
```

- [ ] **步骤 2：运行聚焦测试，确认按预期失败**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py -q
```

预期：测试失败，因为 `src/fashion_trend/trend/models/supervised/lightgbm.py` 尚不存在，或尚未定义被导入的常量。

- [ ] **步骤 3：如已获授权，提交测试骨架**

```sh
git add tests/test_trend_lightgbm.py
git commit -m "test(trend): 添加 LightGBM 主模型测试骨架"
```

---

### 任务 2：实现 LightGBM 常量和特征准备

**文件：**
- 新建：`src/fashion_trend/trend/models/supervised/lightgbm.py`
- 修改：`tests/test_trend_lightgbm.py`

- [ ] **步骤 1：增加特征表测试**

在 `tests/test_trend_lightgbm.py` 的 `TestLightGBMTrendModel` 中追加以下测试：

```python
    def test_prepare_feature_frame_uses_train_categories(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="train")

        prepared = lightgbm_model.prepare_lightgbm_feature_frame(samples)

        assert prepared.attr_type_categories == ("colour_group_name",)
        assert prepared.features.columns.tolist() == [
            *lightgbm_model.LIGHTGBM_NUMERIC_FEATURES,
            *lightgbm_model.LIGHTGBM_CATEGORICAL_FEATURES,
        ]
        assert str(prepared.features["attr_type"].dtype) == "category"
        assert list(prepared.features["attr_type"].cat.categories) == [
            "colour_group_name"
        ]

    def test_prepare_feature_frame_reuses_train_categories(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="valid")

        prepared = lightgbm_model.prepare_lightgbm_feature_frame(
            samples,
            attr_type_categories=("colour_group_name", "product_type_name"),
        )

        assert prepared.attr_type_categories == (
            "colour_group_name",
            "product_type_name",
        )
        assert list(prepared.features["attr_type"].cat.categories) == [
            "colour_group_name",
            "product_type_name",
        ]

    def test_prepare_feature_frame_rejects_unknown_attr_type(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="valid")

        with pytest.raises(ValueError, match="未知 attr_type"):
            lightgbm_model.prepare_lightgbm_feature_frame(
                samples,
                attr_type_categories=("product_type_name",),
            )

    def test_prepare_feature_frame_rejects_non_finite_numeric_feature(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="train")
        samples.loc[samples.index[0], "growth_lag_1"] = float("nan")

        with pytest.raises(ValueError, match="非有限|growth_lag_1"):
            lightgbm_model.prepare_lightgbm_feature_frame(samples)

    def test_prepare_feature_frame_rejects_missing_attr_type(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="train")
        samples.loc[samples.index[0], "attr_type"] = None

        with pytest.raises(ValueError, match="attr_type"):
            lightgbm_model.prepare_lightgbm_feature_frame(samples)
```

- [ ] **步骤 2：运行聚焦测试，确认特征准备尚未实现**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_prepare_feature_frame_uses_train_categories tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_prepare_feature_frame_reuses_train_categories tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_prepare_feature_frame_rejects_unknown_attr_type -q
```

预期：测试失败，因为 `prepare_lightgbm_feature_frame()` 尚不存在。

- [ ] **步骤 3：创建包含常量和特征准备逻辑的 LightGBM 模块**

创建 `src/fashion_trend/trend/models/supervised/lightgbm.py`，初始实现如下：

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.trend.models.base import MODEL_TYPE_SUPERVISED

LIGHTGBM_MODEL_NAME = "lightgbm"
LIGHTGBM_TARGET_COLUMN = "target_growth"
LIGHTGBM_EPSILON = 1e-6
LIGHTGBM_NUMERIC_FEATURES: tuple[str, ...] = (
    "heat_t",
    "share_t",
    "log_heat_t",
    "rank_in_type_t",
    "heat_lag_1",
    "heat_lag_2",
    "heat_lag_3",
    "heat_lag_4",
    "share_lag_1",
    "share_lag_2",
    "share_lag_3",
    "share_lag_4",
    "growth_lag_1",
    "growth_lag_2",
    "acc_lag_1",
    "heat_ma_4",
    "share_ma_4",
    "share_std_4",
    "share_max_4",
    "share_min_4",
    "article_count",
    "is_core_attr",
    "parent_count",
    "child_count",
    "degree",
    "history_total_heat_t",
    "history_active_weeks_t",
    "is_trend_eligible_t",
    "week_index",
    "week_mod_52",
)
LIGHTGBM_CATEGORICAL_FEATURES: tuple[str, ...] = ("attr_type",)
LIGHTGBM_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "attr_id",
    "attr_value",
    "target_growth",
    "target_log_heat_t1",
    "target_rank_in_type_t1",
    "split",
)
LIGHTGBM_ALLOWED_OBJECTIVES: tuple[str, ...] = ("regression", "regression_l1")
LIGHTGBM_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}
LIGHTGBM_EARLY_STOPPING: dict[str, int] = {"stopping_rounds": 30}
_LIGHTGBM_REQUIRED_COLUMNS: tuple[str, ...] = (
    "split",
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "share_t",
    "target_growth",
    "target_rank_in_type_t1",
    *LIGHTGBM_NUMERIC_FEATURES,
)


@dataclass(frozen=True)
class LightGBMFeatureFrame:
    features: pd.DataFrame
    attr_type_categories: tuple[str, ...]


class LightGBMTrendTrainer:
    name = LIGHTGBM_MODEL_NAME
    model_type = MODEL_TYPE_SUPERVISED

    def train(self, context):
        raise ValueError("lightgbm 模型训练流程缺少拟合实现。")


def prepare_lightgbm_feature_frame(
    samples: pd.DataFrame,
    attr_type_categories: tuple[str, ...] | None = None,
) -> LightGBMFeatureFrame:
    if samples.empty:
        raise ValueError("lightgbm 模型输入 split 不能为空。")
    validate_required_columns(
        samples,
        _LIGHTGBM_REQUIRED_COLUMNS,
        source_name="lightgbm 模型输入样本",
    )
    numeric_features = _read_numeric_features(samples)
    attr_type = _read_attr_type(samples, attr_type_categories)
    features = pd.concat([numeric_features, attr_type], axis=1)
    return LightGBMFeatureFrame(
        features=features.loc[
            :, [*LIGHTGBM_NUMERIC_FEATURES, *LIGHTGBM_CATEGORICAL_FEATURES]
        ],
        attr_type_categories=tuple(attr_type.cat.categories.astype(str)),
    )


def _read_numeric_features(samples: pd.DataFrame) -> pd.DataFrame:
    numeric_features = pd.DataFrame(index=samples.index)
    for column in LIGHTGBM_NUMERIC_FEATURES:
        try:
            numeric_features[column] = pd.to_numeric(samples[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"lightgbm 模型数值特征无法解析: {column}") from exc
        if not np.isfinite(numeric_features[column].to_numpy(dtype=float)).all():
            raise ValueError(f"lightgbm 模型数值特征存在非有限值: {column}")
    return numeric_features


def _read_attr_type(
    samples: pd.DataFrame,
    attr_type_categories: tuple[str, ...] | None,
) -> pd.Series:
    if samples["attr_type"].isna().any():
        raise ValueError("lightgbm 模型 attr_type 不能为空。")
    attr_values = samples["attr_type"].astype(str)
    if attr_type_categories is None:
        categories = tuple(sorted(attr_values.unique()))
    else:
        categories = tuple(attr_type_categories)
        unknown_values = sorted(set(attr_values.unique()) - set(categories))
        if unknown_values:
            examples = ", ".join(unknown_values[:5])
            raise ValueError(f"lightgbm 模型存在未知 attr_type: {examples}")
    dtype = CategoricalDtype(categories=list(categories))
    attr_type = attr_values.astype(dtype)
    attr_type.name = "attr_type"
    return attr_type
```

- [ ] **步骤 4：运行聚焦测试，确认常量和特征准备通过**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py -q
```

预期：常量和特征准备测试通过；此时尚未加入调用 `LightGBMTrendTrainer.train()` 的测试。

- [ ] **步骤 5：如已获授权，提交常量和特征准备实现**

```sh
git add src/fashion_trend/trend/models/supervised/lightgbm.py tests/test_trend_lightgbm.py
git commit -m "feat(trend): 添加 LightGBM 特征准备"
```

---

### 任务 3：添加诊断和特征重要性辅助函数

**文件：**
- 修改：`src/fashion_trend/trend/models/supervised/lightgbm.py`
- 修改：`tests/test_trend_lightgbm.py`

- [ ] **步骤 1：增加诊断和特征重要性测试**

在 `TestLightGBMTrendModel` 中追加以下测试：

```python
    def test_describe_target_distribution_returns_split_metric_mapping(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="train")

        distribution = lightgbm_model.describe_target_distribution(
            {"train": samples}
        )

        assert set(distribution) == {"train"}
        assert set(distribution["train"]) == {
            "count",
            "min",
            "max",
            "mean",
            "std",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
            "abs_gt_2",
        }
        assert distribution["train"]["count"] == len(samples)

    def test_describe_residual_distribution_returns_valid_test_only(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS

        samples = _sample_lightgbm_samples("valid")
        predictions = samples.loc[
            :,
            [
                "week_id",
                "attr_id",
                "attr_type",
                "attr_value",
                "split",
                "share_t",
                "target_growth",
                "target_rank_in_type_t1",
            ],
        ].copy()
        predictions.insert(4, "model_name", "lightgbm")
        predictions["pred_share_t1"] = [0.6, 0.4, 0.6, 0.4]
        predictions["pred_target_growth"] = [0.1, -0.1, 0.2, -0.2]
        predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]

        distribution = lightgbm_model.describe_residual_distribution(
            {"valid": predictions}
        )

        assert set(distribution) == {"valid"}
        assert set(distribution["valid"]) == {
            "count",
            "min",
            "max",
            "mean",
            "std",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
            "mae",
            "rmse",
        }

    def test_build_feature_importance_frame_normalizes_gain(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        booster = _FakeBooster(
            feature_names=["growth_lag_1", "attr_type"],
            split_importance=[3, 1],
            gain_importance=[2.0, 6.0],
        )

        importance = lightgbm_model.build_feature_importance_frame(booster)

        assert importance.to_dict(orient="records") == [
            {
                "feature": "growth_lag_1",
                "split_importance": 3,
                "gain_importance": 2.0,
                "normalized_gain_importance": 0.25,
            },
            {
                "feature": "attr_type",
                "split_importance": 1,
                "gain_importance": 6.0,
                "normalized_gain_importance": 0.75,
            },
        ]

    def test_build_feature_importance_frame_handles_zero_gain(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        booster = _FakeBooster(
            feature_names=["growth_lag_1", "attr_type"],
            split_importance=[0, 0],
            gain_importance=[0.0, 0.0],
        )

        importance = lightgbm_model.build_feature_importance_frame(booster)

        assert importance["normalized_gain_importance"].tolist() == [0.0, 0.0]
```

在 `tests/test_trend_lightgbm.py` 底部增加以下模块级辅助对象：

```python
def _sample_lightgbm_samples(split: str):
    from tests.trend_samples import sample_trend_model_samples_for_split

    samples = sample_trend_model_samples_for_split().head(4).copy()
    samples["split"] = split
    return samples


class _FakeBooster:
    def __init__(
        self,
        feature_names: list[str],
        split_importance: list[int],
        gain_importance: list[float],
    ) -> None:
        self._feature_names = feature_names
        self._split_importance = split_importance
        self._gain_importance = gain_importance

    def feature_name(self) -> list[str]:
        return list(self._feature_names)

    def feature_importance(self, importance_type: str):
        if importance_type == "split":
            return list(self._split_importance)
        if importance_type == "gain":
            return list(self._gain_importance)
        raise AssertionError(f"unexpected importance_type={importance_type}")

    def model_to_string(self) -> str:
        return "fake lightgbm model"
```

- [ ] **步骤 2：运行聚焦测试，确认辅助函数尚未实现**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_describe_target_distribution_returns_split_metric_mapping tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_describe_residual_distribution_returns_valid_test_only tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_build_feature_importance_frame_normalizes_gain -q
```

预期：测试失败，因为诊断和特征重要性辅助函数尚不存在。

- [ ] **步骤 3：实现诊断和特征重要性辅助函数**

把以下函数追加到 `src/fashion_trend/trend/models/supervised/lightgbm.py`：

```python
def describe_target_distribution(
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float | int]]:
    return {
        split_name: _describe_target_series(split_frame[LIGHTGBM_TARGET_COLUMN])
        for split_name, split_frame in split_frames.items()
    }


def describe_zero_share_rows(
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, int]:
    return {
        split_name: int(pd.to_numeric(split_frame["share_t"], errors="raise").eq(0).sum())
        for split_name, split_frame in split_frames.items()
    }


def describe_residual_distribution(
    prediction_frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float | int]]:
    distribution: dict[str, dict[str, float | int]] = {}
    for split_name, predictions in prediction_frames.items():
        residual = (
            pd.to_numeric(predictions["target_growth"], errors="raise")
            - pd.to_numeric(predictions["pred_target_growth"], errors="raise")
        )
        summary = _describe_residual_series(residual)
        summary["mae"] = float(residual.abs().mean())
        summary["rmse"] = float(np.sqrt(np.square(residual).mean()))
        distribution[split_name] = summary
    return distribution


def build_feature_importance_frame(booster) -> pd.DataFrame:
    feature_names = list(booster.feature_name())
    split_importance = np.asarray(
        booster.feature_importance(importance_type="split"),
        dtype=np.int64,
    )
    gain_importance = np.asarray(
        booster.feature_importance(importance_type="gain"),
        dtype=float,
    )
    total_gain = float(gain_importance.sum())
    if total_gain > 0:
        normalized_gain = gain_importance / total_gain
    else:
        normalized_gain = np.zeros_like(gain_importance, dtype=float)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "split_importance": split_importance,
            "gain_importance": gain_importance,
            "normalized_gain_importance": normalized_gain,
        }
    )
    if not np.isfinite(
        importance.loc[:, ["gain_importance", "normalized_gain_importance"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("lightgbm 特征重要性存在非有限数值。")
    return importance


def _describe_numeric_series(series: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("lightgbm 诊断数值存在非有限值。")
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "p01": float(numeric.quantile(0.01)),
        "p05": float(numeric.quantile(0.05)),
        "p50": float(numeric.quantile(0.50)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
    }


def _describe_target_series(series: pd.Series) -> dict[str, float | int]:
    summary = _describe_numeric_series(series)
    numeric = pd.to_numeric(series, errors="raise")
    summary["abs_gt_2"] = int(numeric.abs().gt(2).sum())
    return summary


def _describe_residual_series(series: pd.Series) -> dict[str, float | int]:
    return _describe_numeric_series(series)
```

- [ ] **步骤 4：运行聚焦测试，确认辅助函数通过**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py -q
```

预期：当前已经写好的 LightGBM 测试全部通过。

- [ ] **步骤 5：如已获授权，提交诊断辅助函数**

```sh
git add src/fashion_trend/trend/models/supervised/lightgbm.py tests/test_trend_lightgbm.py
git commit -m "feat(trend): 补充 LightGBM 诊断工具"
```

---

### 任务 4：实现带延迟原生包导入的 LightGBM trainer

**文件：**
- 修改：`src/fashion_trend/trend/models/supervised/lightgbm.py`
- 修改：`tests/test_trend_lightgbm.py`

- [ ] **步骤 1：增加使用假模型注入的 trainer 测试**

在 `TestLightGBMTrendModel` 中追加以下测试：

```python
    def test_trainer_returns_standard_train_result(self, monkeypatch) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.models.base import TrendTrainContext, TrendTrainResult
        from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
        from fashion_trend.trend.splits import build_trend_model_split_frames
        from tests.trend_samples import sample_trend_model_samples_for_split

        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )

        def fake_fit(train_features, train_target, valid_features, valid_target):
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)

        result = lightgbm_model.LightGBMTrendTrainer().train(
            TrendTrainContext(
                model_name="lightgbm",
                split_frames=split_frames,
                input_paths={
                    "train": Path("train.parquet"),
                    "valid": Path("valid.parquet"),
                    "test": Path("test.parquet"),
                },
                output_dir=Path("outputs/models/lightgbm"),
            )
        )

        assert isinstance(result, TrendTrainResult)
        assert result.model_name == "lightgbm"
        assert result.model_type == MODEL_TYPE_SUPERVISED
        assert result.predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
        assert set(result.predictions["model_name"]) == {"lightgbm"}
        assert result.params["objective"] == "regression"
        assert result.params["best_iteration"] == 7
        assert result.metadata["attr_type_categories"] == ["colour_group_name"]
        assert set(result.metadata["target_distribution"]) == {"train", "valid", "test"}
        assert set(result.metadata["residual_distribution"]) == {"valid", "test"}
        assert [artifact.relative_path for artifact in result.artifacts] == [
            "feature_importance.csv",
            "model.txt",
        ]

    def test_trainer_rejects_split_frame_with_mismatched_split_value(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.models.base import TrendTrainContext
        from fashion_trend.trend.splits import build_trend_model_split_frames
        from tests.trend_samples import sample_trend_model_samples_for_split

        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        split_frames["valid"] = split_frames["valid"].assign(split="train")

        with pytest.raises(ValueError, match="split.*valid|不一致"):
            lightgbm_model.LightGBMTrendTrainer().train(
                TrendTrainContext(
                    model_name="lightgbm",
                    split_frames=split_frames,
                    input_paths={
                        "train": Path("train.parquet"),
                        "valid": Path("valid.parquet"),
                        "test": Path("test.parquet"),
                    },
                    output_dir=Path("outputs/models/lightgbm"),
                )
            )

    def test_trainer_rejects_split_frame_missing_split_column(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.models.base import TrendTrainContext
        from fashion_trend.trend.splits import build_trend_model_split_frames
        from tests.trend_samples import sample_trend_model_samples_for_split

        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        split_frames["valid"] = split_frames["valid"].drop(columns=["split"])

        with pytest.raises(ValueError, match="lightgbm.*split.*缺少|缺少.*split"):
            lightgbm_model.LightGBMTrendTrainer().train(
                TrendTrainContext(
                    model_name="lightgbm",
                    split_frames=split_frames,
                    input_paths={
                        "train": Path("train.parquet"),
                        "valid": Path("valid.parquet"),
                        "test": Path("test.parquet"),
                    },
                    output_dir=Path("outputs/models/lightgbm"),
                )
            )

    def test_fit_lightgbm_model_wraps_native_import_errors(self, monkeypatch) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)

        def broken_import(name, *args, **kwargs):
            if name == "lightgbm":
                raise OSError("libomp.dylib not found")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", broken_import)

        with pytest.raises(ValueError, match="lightgbm|libomp"):
            lightgbm_model._fit_lightgbm_model(
                _sample_lightgbm_samples("train").loc[:, ["growth_lag_1"]],
                _sample_lightgbm_samples("train")["target_growth"],
                _sample_lightgbm_samples("valid").loc[:, ["growth_lag_1"]],
                _sample_lightgbm_samples("valid")["target_growth"],
            )
```

增加这个假模型辅助对象：

```python
class _FakeLightGBMModel:
    best_iteration_ = 7
    best_score_ = {"valid_0": {"l2": 0.12}}

    def __init__(self, feature_names: list[str]) -> None:
        self.booster_ = _FakeBooster(
            feature_names=feature_names,
            split_importance=[1 for _ in feature_names],
            gain_importance=[float(index + 1) for index, _ in enumerate(feature_names)],
        )

    def predict(self, features, num_iteration=None):
        return features["growth_lag_1"].astype(float).to_numpy()
```

- [ ] **步骤 2：运行 trainer 测试，确认按预期失败**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_trainer_returns_standard_train_result tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_trainer_rejects_split_frame_with_mismatched_split_value tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_trainer_rejects_split_frame_missing_split_column tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_fit_lightgbm_model_wraps_native_import_errors -q
```

预期：测试失败，因为 `LightGBMTrendTrainer.train()` 和 `_fit_lightgbm_model()` 尚未实现。

- [ ] **步骤 3：实现延迟导入、trainer、预测构造和产物**

更新 `src/fashion_trend/trend/models/supervised/lightgbm.py` 中的 imports 和实现：

```python
from pathlib import Path
from typing import Mapping

from fashion_trend.trend.models.base import (
    MODEL_TYPE_SUPERVISED,
    TrendArtifact,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.trend.predictions import (
    derive_normalized_pred_share_t1,
    validate_trend_model_predictions,
)
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)
```

将 `LightGBMTrendTrainer.train()` 替换为：

```python
class LightGBMTrendTrainer:
    name = LIGHTGBM_MODEL_NAME
    model_type = MODEL_TYPE_SUPERVISED

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        split_frames = _copy_context_split_frames(context)
        train_prepared = prepare_lightgbm_feature_frame(split_frames["train"])
        valid_prepared = prepare_lightgbm_feature_frame(
            split_frames["valid"],
            train_prepared.attr_type_categories,
        )
        test_prepared = prepare_lightgbm_feature_frame(
            split_frames["test"],
            train_prepared.attr_type_categories,
        )
        model = _fit_lightgbm_model(
            train_prepared.features,
            _read_target(split_frames["train"]),
            valid_prepared.features,
            _read_target(split_frames["valid"]),
        )
        prepared_frames = {
            "train": train_prepared,
            "valid": valid_prepared,
            "test": test_prepared,
        }
        prediction_frames = {
            split_name: _build_lightgbm_predictions(
                split_frames[split_name],
                _predict_with_model(model, prepared.features),
            )
            for split_name, prepared in prepared_frames.items()
        }
        predictions = pd.concat(
            [prediction_frames[split_name] for split_name in context.split_order],
            ignore_index=True,
        ).sort_values(["week_id", "attr_type", "attr_id"], ignore_index=True)
        split_samples = pd.concat(
            [split_frames[split_name] for split_name in context.split_order],
            ignore_index=True,
        )
        validate_trend_model_predictions(predictions, split_samples)
        booster = model.booster_
        feature_importance = build_feature_importance_frame(booster)
        metadata = {
            "target_column": LIGHTGBM_TARGET_COLUMN,
            "numeric_features": list(LIGHTGBM_NUMERIC_FEATURES),
            "categorical_features": list(LIGHTGBM_CATEGORICAL_FEATURES),
            "attr_type_categories": list(train_prepared.attr_type_categories),
            "best_iteration": _read_best_iteration(model),
            "best_score": _read_best_score(model),
            "target_distribution": describe_target_distribution(split_frames),
            "zero_share_rows": describe_zero_share_rows(split_frames),
            "residual_distribution": describe_residual_distribution(
                {
                    "valid": prediction_frames["valid"],
                    "test": prediction_frames["test"],
                }
            ),
        }
        return TrendTrainResult(
            model_name=self.name,
            model_type=self.model_type,
            predictions=predictions,
            params=_build_lightgbm_params(model),
            metadata=metadata,
            artifacts=(
                TrendArtifact("feature_importance.csv", "csv", feature_importance),
                TrendArtifact("model.txt", "binary", _dump_model_text(booster)),
            ),
        )
```

增加以下支撑函数：

```python
def _copy_context_split_frames(
    context: TrendTrainContext,
) -> dict[str, pd.DataFrame]:
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in context.split_order:
        if split_name not in context.split_frames:
            raise ValueError(f"lightgbm 模型输入缺少 split: {split_name}")
        split_frame = context.split_frames[split_name].copy()
        validate_required_columns(
            split_frame,
            ("split",),
            source_name=f"lightgbm 模型输入 split {split_name}",
        )
        if split_frame.empty:
            raise ValueError(f"lightgbm 模型输入 split 不能为空: {split_name}")
        split_values = set(split_frame["split"].astype(str))
        if not split_values.issubset(set(TREND_MODEL_SPLIT_VALUES)):
            raise ValueError(f"lightgbm 模型输入 split 存在非法值: {split_name}")
        if split_values != {split_name}:
            values = ", ".join(sorted(split_values))
            raise ValueError(
                "lightgbm 模型输入 split 与 frame key 不一致: "
                f"key={split_name}, values={values}"
            )
        split_frames[split_name] = split_frame
    return split_frames


def _read_target(samples: pd.DataFrame) -> pd.Series:
    try:
        target = pd.to_numeric(samples[LIGHTGBM_TARGET_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("lightgbm 模型 target_growth 必须为数值。") from exc
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("lightgbm 模型 target_growth 存在非有限值。")
    return target


def _fit_lightgbm_model(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    valid_features: pd.DataFrame,
    valid_target: pd.Series,
):
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    except (ImportError, OSError) as exc:
        raise ValueError(
            "lightgbm 模型依赖无法导入；请确认 lightgbm 与 native runtime "
            "如 libomp.dylib 已正确安装。"
        ) from exc
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        train_features,
        train_target,
        eval_set=[(valid_features, valid_target)],
        eval_metric="l2",
        categorical_feature=list(LIGHTGBM_CATEGORICAL_FEATURES),
        callbacks=[
            early_stopping(
                int(LIGHTGBM_EARLY_STOPPING["stopping_rounds"]),
                verbose=False,
            ),
            log_evaluation(period=0),
        ],
    )
    return model


def _predict_with_model(model, features: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(features, num_iteration=_read_best_iteration(model))
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("lightgbm 模型预测存在非有限数值。")
    return predictions


def _build_lightgbm_predictions(
    split_samples: pd.DataFrame,
    pred_target_growth: list[float] | np.ndarray | pd.Series,
) -> pd.DataFrame:
    predictions = split_samples.loc[
        :,
        [
            "week_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "split",
            "share_t",
            "target_growth",
            "target_rank_in_type_t1",
        ],
    ].copy()
    predictions.insert(4, "model_name", LIGHTGBM_MODEL_NAME)
    predictions["pred_target_growth"] = np.asarray(pred_target_growth, dtype=float)
    predictions["pred_share_t1"] = derive_normalized_pred_share_t1(
        predictions,
        LIGHTGBM_EPSILON,
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    return predictions.sort_values(["week_id", "attr_type", "attr_id"], ignore_index=True)


def _build_lightgbm_params(model) -> dict[str, object]:
    return {
        "model_name": LIGHTGBM_MODEL_NAME,
        "model_type": MODEL_TYPE_SUPERVISED,
        "target_column": LIGHTGBM_TARGET_COLUMN,
        "numeric_features": list(LIGHTGBM_NUMERIC_FEATURES),
        "categorical_features": list(LIGHTGBM_CATEGORICAL_FEATURES),
        "excluded_columns": list(LIGHTGBM_EXCLUDED_COLUMNS),
        "epsilon": LIGHTGBM_EPSILON,
        "lightgbm_params": dict(LIGHTGBM_PARAMS),
        "early_stopping": dict(LIGHTGBM_EARLY_STOPPING),
        "best_iteration": _read_best_iteration(model),
        "objective": str(LIGHTGBM_PARAMS["objective"]),
        "allowed_objectives": list(LIGHTGBM_ALLOWED_OBJECTIVES),
    }


def _read_best_iteration(model) -> int | None:
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None:
        return None
    return int(best_iteration)


def _read_best_score(model) -> dict[str, object]:
    best_score = getattr(model, "best_score_", {})
    return dict(best_score)


def _dump_model_text(booster) -> bytes:
    model_text = booster.model_to_string()
    return model_text.encode("utf-8")
```

- [ ] **步骤 4：运行 LightGBM 单元测试**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py -q
```

预期：全部测试通过；除了使用 monkeypatch 的 `_fit_lightgbm_model` 导入错误包装测试外，测试不需要真实导入 LightGBM 原生包。

- [ ] **步骤 5：如已获授权，提交 trainer 实现**

```sh
git add src/fashion_trend/trend/models/supervised/lightgbm.py tests/test_trend_lightgbm.py
git commit -m "feat(trend): 实现 LightGBM 主模型训练器"
```

---

### 任务 5：注册 LightGBM 并接入训练 runner 测试

**文件：**
- 修改：`src/fashion_trend/trend/models/registry.py`
- 修改：`tests/test_trend_lightgbm.py`
- 修改：`tests/test_trend_training.py`

- [ ] **步骤 1：增加 registry 和 runner 测试**

在 `tests/test_trend_training.py` 中增加 imports：

```python
from fashion_trend.trend.models.base import MODEL_TYPE_SUPERVISED
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_MODEL_NAME,
    LightGBMTrendTrainer,
)
```

更新 `test_registry_lists_registered_models()`：

```python
def test_registry_lists_registered_models(self) -> None:
    assert list_trend_model_names() == (
        LAST_WEEK_MODEL_NAME,
        LIGHTGBM_MODEL_NAME,
        MOVING_AVERAGE_MODEL_NAME,
        PREVIOUS_GROWTH_MODEL_NAME,
    )
```

增加 registry 测试：

```python
def test_registry_returns_lightgbm_trainer(self) -> None:
    trainer = get_trend_model_trainer(LIGHTGBM_MODEL_NAME)

    assert isinstance(trainer, LightGBMTrendTrainer)
    assert trainer.name == LIGHTGBM_MODEL_NAME
    assert trainer.model_type == MODEL_TYPE_SUPERVISED
```

在现有 CLI 测试附近增加 CLI 接受 `lightgbm` 的测试：

```python
def test_train_trend_model_main_accepts_lightgbm(self) -> None:
    train_model = importlib.import_module("10_train_trend_model")
    calls: list[str] = []
    original_run_trend_model_training = train_model.run_trend_model_training

    def fake_run_trend_model_training(model_name: str) -> dict[str, object]:
        calls.append(model_name)
        return {
            "model_name": LIGHTGBM_MODEL_NAME,
            "model_type": MODEL_TYPE_SUPERVISED,
            "rows": 40,
            "weeks": 20,
            "attributes": 2,
            "splits": {
                "train": {"rows": 24, "weeks": 12, "attributes": 2, "week_min": 4, "week_max": 15},
                "valid": {"rows": 8, "weeks": 4, "attributes": 2, "week_min": 16, "week_max": 19},
                "test": {"rows": 8, "weeks": 4, "attributes": 2, "week_min": 20, "week_max": 23},
            },
            "output_dir": "outputs/models/lightgbm",
            "prediction_path": "outputs/models/lightgbm/predictions.csv",
            "params_path": "outputs/models/lightgbm/params.json",
        }

    try:
        train_model.run_trend_model_training = fake_run_trend_model_training

        assert train_model.main(["--model", LIGHTGBM_MODEL_NAME]) == 0
    finally:
        train_model.run_trend_model_training = original_run_trend_model_training

    assert calls == [LIGHTGBM_MODEL_NAME]
```

在 `tests/test_trend_lightgbm.py` 的 `TestLightGBMTrendModel` 中追加运行时导入隔离测试：

```python
    def test_native_lightgbm_import_is_deferred_until_fit(self, monkeypatch) -> None:
        import builtins

        for module_name in (
            LIGHTGBM_MODULE,
            "fashion_trend.trend.models.registry",
        ):
            sys.modules.pop(module_name, None)

        original_import = builtins.__import__
        blocked_imports: list[str] = []

        def block_lightgbm(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lightgbm" or name.startswith("lightgbm."):
                blocked_imports.append(name)
                raise ImportError("blocked lightgbm import")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", block_lightgbm)

        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        registry = importlib.import_module("fashion_trend.trend.models.registry")

        assert "lightgbm" in registry.list_trend_model_names()
        assert registry.get_trend_model_trainer("last_week").name == "last_week"
        assert registry.get_trend_model_trainer("previous_growth").name == "previous_growth"
        assert registry.get_trend_model_trainer("moving_average").name == "moving_average"
        assert blocked_imports == []

        with pytest.raises(ValueError, match="lightgbm|native runtime|原生运行时"):
            lightgbm_model._fit_lightgbm_model(
                _sample_lightgbm_samples("train").loc[:, ["growth_lag_1"]],
                _sample_lightgbm_samples("train")["target_growth"],
                _sample_lightgbm_samples("valid").loc[:, ["growth_lag_1"]],
                _sample_lightgbm_samples("valid")["target_growth"],
            )

        assert blocked_imports == ["lightgbm"]
```

- [ ] **步骤 2：运行聚焦测试，确认 registry 测试按预期失败**

运行：

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_registry_lists_registered_models tests/test_trend_training.py::TestTrendTraining::test_registry_returns_lightgbm_trainer tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_native_lightgbm_import_is_deferred_until_fit -q
```

预期：测试失败，因为 `lightgbm` 尚未注册。

- [ ] **步骤 3：注册 LightGBM trainer**

修改 `src/fashion_trend/trend/models/registry.py`：

```python
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_MODEL_NAME,
    LightGBMTrendTrainer,
)
```

更新 `TREND_MODEL_REGISTRY`：

```python
TREND_MODEL_REGISTRY: dict[str, TrendModelTrainer] = {
    LAST_WEEK_MODEL_NAME: LastWeekTrainer(),
    LIGHTGBM_MODEL_NAME: LightGBMTrendTrainer(),
    MOVING_AVERAGE_MODEL_NAME: MovingAverageTrainer(),
    PREVIOUS_GROWTH_MODEL_NAME: PreviousGrowthTrainer(),
}
```

- [ ] **步骤 4：运行聚焦训练测试**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py tests/test_trend_training.py -q
```

预期：测试通过。运行时导入隔离测试应证明导入 `supervised.lightgbm`、导入 registry、列出模型和获取 baseline trainer 都不会导入原生 `lightgbm`；只有调用 `_fit_lightgbm_model()` 才会触发并包装为可定位的 `ValueError`。

- [ ] **步骤 5：如已获授权，提交 registry 接入**

```sh
git add src/fashion_trend/trend/models/registry.py tests/test_trend_lightgbm.py tests/test_trend_training.py
git commit -m "feat(trend): 注册 LightGBM 主模型"
```

---

### 任务 6：增加 runner 输出和评价测试

**文件：**
- 修改：`tests/test_trend_training.py`
- 修改：`tests/test_trend_evaluation.py`

- [ ] **步骤 1：增加使用假拟合函数的 runner 输出测试**

在 `tests/test_trend_training.py` 中增加一个测试：调用 `run_trend_model_training()` 前先 monkeypatch `_fit_lightgbm_model`。

```python
def test_run_trend_model_training_writes_lightgbm_outputs(
    self,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

    class FakeBooster:
        def feature_name(self) -> list[str]:
            return [*lightgbm_model.LIGHTGBM_NUMERIC_FEATURES, "attr_type"]

        def feature_importance(self, importance_type: str):
            feature_count = len(self.feature_name())
            if importance_type == "split":
                return [1 for _ in range(feature_count)]
            if importance_type == "gain":
                return [1.0 for _ in range(feature_count)]
            raise AssertionError(f"unexpected importance_type={importance_type}")

        def model_to_string(self) -> str:
            return "fake lightgbm model"

    class FakeModel:
        best_iteration_ = 7
        best_score_ = {"valid_0": {"l2": 0.1}}
        booster_ = FakeBooster()

        def predict(self, features: pd.DataFrame, num_iteration: int | None = None):
            return features["growth_lag_1"].astype(float).to_numpy()

    def fake_fit(train_features, train_target, valid_features, valid_target):
        return FakeModel()

    monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)

    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    input_paths = {
        "train": tmp_path / "trend_model_samples_train.parquet",
        "valid": tmp_path / "trend_model_samples_valid.parquet",
        "test": tmp_path / "trend_model_samples_test.parquet",
    }
    for split_name, split_frame in split_frames.items():
        write_parquet_atomic(split_frame, input_paths[split_name])

    metadata = run_trend_model_training(
        LIGHTGBM_MODEL_NAME,
        input_paths=input_paths,
        output_root=tmp_path / "outputs" / "models",
    )

    output_dir = tmp_path / "outputs" / "models" / "lightgbm"
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "params.json").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "feature_importance.csv").exists()
    assert (output_dir / "model.txt").exists()
    assert metadata["model_name"] == LIGHTGBM_MODEL_NAME
    assert metadata["model_type"] == MODEL_TYPE_SUPERVISED
    assert metadata["extra_artifacts"] == [
        {"path": "feature_importance.csv", "kind": "csv"},
        {"path": "model.txt", "kind": "binary"},
    ]
```

- [ ] **步骤 2：增加评价 runner 测试**

在 `tests/test_trend_evaluation.py` 中增加：

```python
def test_run_trend_model_evaluation_writes_lightgbm_metrics(
    self,
    tmp_path: Path,
) -> None:
    predictions = sample_trend_predictions_for_evaluation().copy()
    predictions["model_name"] = "lightgbm"
    model_output_dir = tmp_path / "outputs" / "models" / "lightgbm"
    metrics_output_root = tmp_path / "outputs" / "metrics"
    write_csv_atomic(predictions, model_output_dir / "predictions.csv")

    payload = run_trend_model_evaluation(
        "lightgbm",
        model_output_root=tmp_path / "outputs" / "models",
        metrics_output_root=metrics_output_root,
    )

    assert payload["model_name"] == "lightgbm"
    assert (metrics_output_root / "lightgbm" / "trend_metrics.json").exists()
```

- [ ] **步骤 3：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_lightgbm_outputs tests/test_trend_evaluation.py::TestTrendEvaluation::test_run_trend_model_evaluation_writes_lightgbm_metrics -q
```

预期：测试通过。

- [ ] **步骤 4：如已获授权，提交 runner / evaluation 测试**

```sh
git add tests/test_trend_training.py tests/test_trend_evaluation.py
git commit -m "test(trend): 覆盖 LightGBM 训练评价接入"
```

---

### 任务 7：更新 README 和 implementation-plan 文档

**文件：**
- 修改：`README.md`
- 修改：`docs/gpt-research/implementation-plan.md`

- [ ] **步骤 1：更新 README 命令列表和阶段表**

在 README 中列出 baseline 命令的代码块里增加：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm
```

更新当前阶段表，加入：

```markdown
| LightGBM 主模型 | 已实现（运行命令后生成） | `outputs/models/lightgbm/predictions.csv`、`params.json`、`metadata.json`、`feature_importance.csv`、`model.txt` |
```

- [ ] **步骤 2：增加 README LightGBM 小节**

在现有 baseline 小节附近增加以下小节，并把当前 README 中已有的
`### 12. 趋势评价` 整体顺延为 `### 13. 趋势评价`。顺延时同步检查模型列表、命令块、metrics 路径和阶段表，确保 README 中不会出现两个 `### 12` 小节。

````markdown
### 12. LightGBM 主模型

`lightgbm` 主模型复用通用趋势模型训练入口，模型细节位于
`src/fashion_trend/trend/models/supervised/lightgbm.py`。第一版使用现有
`trend_model_samples` 中的数值特征和 `attr_type` 分类特征，预测目标为：

```text
target_growth
```

模型使用 train split 拟合，valid split 做 early stopping，test split 只进入统一趋势评价。
标准预测产物和可解释产物写入：

```sh
outputs/models/lightgbm/predictions.csv
outputs/models/lightgbm/params.json
outputs/models/lightgbm/metadata.json
outputs/models/lightgbm/feature_importance.csv
outputs/models/lightgbm/model.txt
```

运行命令：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm
```
````

- [ ] **步骤 3：更新实施计划中的 LightGBM 小节**

在 `docs/gpt-research/implementation-plan.md` 中，把说明 `lightgbm` 尚未注册的文字替换为：

````markdown
当前实现中，`lightgbm` 已注册到统一趋势模型训练入口：

```sh
src/10_train_trend_model.py --model lightgbm
src/11_eval_trend_model.py --model lightgbm
```

标准模型产物位于：

```text
outputs/models/lightgbm/predictions.csv
outputs/models/lightgbm/metadata.json
outputs/models/lightgbm/params.json
outputs/models/lightgbm/feature_importance.csv
outputs/models/lightgbm/model.txt
```

趋势评价产物位于：

```text
outputs/metrics/lightgbm/trend_metrics.json
```
````

- [ ] **步骤 4：运行文档 grep 检查**

运行：

```sh
rg -n "lightgbm|LightGBM|尚未注册|后续实现注册后|12_train_lightgbm" README.md docs/gpt-research/implementation-plan.md
rg -n "^### 12\\.|^### 13\\." README.md
```

预期：
- 相关引用应说明 `lightgbm` 已通过 `10_train_trend_model.py --model lightgbm` 实现；不应再出现需要单独 LightGBM 编号脚本的描述。
- README 只有 `### 12. LightGBM 主模型` 和 `### 13. 趋势评价`，没有重复的 `### 12`。

- [ ] **步骤 5：如已获授权，提交文档同步**

```sh
git add README.md docs/gpt-research/implementation-plan.md
git commit -m "docs: 同步 LightGBM 主模型说明"
```

---

### 任务 8：运行完整验证和 baseline 对比

**文件：**
- 读取/写入被忽略的 `outputs/` 下生成产物
- 预期不修改源代码

- [ ] **步骤 1：运行聚焦测试集**

运行：

```sh
uv run pytest tests/test_trend_lightgbm.py tests/test_trend_training.py tests/test_trend_evaluation.py
```

预期：选中的测试全部通过。如果 `uv run` 因 uv cache 权限失败，则运行：

```sh
.venv/bin/python -m pytest tests/test_trend_lightgbm.py tests/test_trend_training.py tests/test_trend_evaluation.py
```

- [ ] **步骤 2：运行完整测试集和格式检查**

运行：

```sh
uv run pytest
uv run black --check src tests
uv run isort --check-only src tests
```

预期：全部通过。如果 `uv run pytest` 仅因 uv cache 权限失败，则改用 `.venv/bin/python -m pytest` 重新运行，并在最终结果中记录环境问题。

- [ ] **步骤 3：训练并评价 LightGBM**

运行：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm
```

预期输出：

```text
outputs/models/lightgbm/predictions.csv
outputs/models/lightgbm/params.json
outputs/models/lightgbm/metadata.json
outputs/models/lightgbm/feature_importance.csv
outputs/models/lightgbm/model.txt
outputs/metrics/lightgbm/trend_metrics.json
```

如果缺少 LightGBM 原生运行时依赖，预期行为是只有 `lightgbm` 训练命令失败，并给出清晰的 `lightgbm` / 原生运行时错误；baseline 训练命令仍必须可用。

- [ ] **步骤 4：重新生成 baseline predictions 和 metrics**

运行：

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

预期：三个 baseline 的 `predictions.csv` 和 `trend_metrics.json` 都在本轮重新生成，避免拿旧代码、旧预测或旧样本的结果与 LightGBM 比较。

- [ ] **步骤 5：生成 baseline 对比摘要**

运行：

```sh
uv run python - <<'PY'
import json
from pathlib import Path

models = ["last_week", "previous_growth", "moving_average", "lightgbm"]
metrics = {}
for model in models:
    path = Path("outputs/metrics") / model / "trend_metrics.json"
    if not path.exists():
        raise SystemExit(f"missing metrics: {path}")
    metrics[model] = json.loads(path.read_text(encoding="utf-8"))["overall"]


def format_metric(value):
    if value is None:
        return "not_available"
    return f"{float(value):.6f}"


def find_best_baseline(values, reverse=False):
    comparable = {
        model: value
        for model, value in values.items()
        if model != "lightgbm" and value is not None
    }
    if not comparable:
        return None, None
    best_model = (
        max(comparable, key=comparable.get)
        if reverse
        else min(comparable, key=comparable.get)
    )
    return best_model, comparable[best_model]


def print_comparison(metric, values, reverse=False):
    best_model, best_value = find_best_baseline(values, reverse=reverse)
    lightgbm_value = values["lightgbm"]
    if best_value is None or lightgbm_value is None:
        better = "not_available"
        baseline_text = "not_available"
    else:
        better = lightgbm_value > best_value if reverse else lightgbm_value < best_value
        baseline_text = f"{best_model}:{format_metric(best_value)}"
    print(
        f"{metric}: lightgbm={format_metric(lightgbm_value)}, "
        f"best_baseline={baseline_text}, "
        f"lightgbm_better={better}"
    )


for split in ("valid", "test"):
    print(f"[{split}]")
    for metric in ("mae", "rmse", "spearman"):
        values = {model: metrics[model][split][metric] for model in models}
        print_comparison(metric, values, reverse=(metric == "spearman"))
    ndcg_values = {
        model: metrics[model][split]["ndcg_at_k"]["10"] for model in models
    }
    print_comparison("ndcg@10", ndcg_values, reverse=True)
PY
```

预期：打印 valid/test 的 MAE、RMSE、Spearman 和 NDCG@10 对比。若某个指标为 `None`，输出 `not_available`，不应因格式化或 best-baseline 比较失败。最终实现结果中需要包含这段输出摘要。

- [ ] **步骤 6：检查源码 diff 和被忽略的生成产物**

运行：

```sh
git status --short
git diff --check
git diff --stat
```

预期：源码、文档和测试改动可见；`outputs/` 产物仍被忽略，且未被 staged。

- [ ] **步骤 7：如已获授权，提交最终实现**

如果所有验证通过，且前面没有按任务逐项提交：

```sh
git add src/fashion_trend/trend/models/supervised/lightgbm.py src/fashion_trend/trend/models/registry.py tests/test_trend_lightgbm.py tests/test_trend_training.py tests/test_trend_evaluation.py README.md docs/gpt-research/implementation-plan.md
git commit -m "feat(trend): 实现 LightGBM 主模型"
```
