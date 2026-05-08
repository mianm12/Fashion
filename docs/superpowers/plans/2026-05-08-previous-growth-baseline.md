# Previous Growth 基线实施计划

> **给 agentic worker 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，并按任务逐项执行。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标：** 对齐必须基线套件，让 `last_week` 表示 Last Week Heat，让 `previous_growth` 接管现有增长率基线，并让三类基线都能通过标准 runner 完成训练和趋势评价。

**架构：** 保持现有 `src/fashion_trend/trend/models/baselines/` 下“一模型一 trainer 文件”的结构。新增 `previous_growth` 确定性 trainer，把 `last_week` 改为用当前份额预测下一周份额，并保持 `moving_average` 不变。CLI、训练 runner、预测表契约和评价 runner 继续保持模型无关。

**技术栈：** Python 3.12、pandas、numpy、pytest，以及现有 `fashion_trend.trend.models`、`fashion_trend.trend.training`、`fashion_trend.trend.evaluation` 和 `fashion_trend.trend.predictions` 工具。

---

## 文件结构

- 新建：`src/fashion_trend/trend/models/baselines/previous_growth.py`
  - 负责 `PREVIOUS_GROWTH_MODEL_NAME`、`PREVIOUS_GROWTH_PARAMS`、`predict_previous_growth()` 和 `PreviousGrowthTrainer`。
- 修改：`src/fashion_trend/trend/models/baselines/last_week.py`
  - 将 `last_week` 从增长率 lag 语义改为 Last Week Heat 语义。
- 修改：`src/fashion_trend/trend/models/registry.py`
  - 在 `last_week` 和 `moving_average` 旁注册 `previous_growth`。
- 修改：`tests/test_trend_training.py`
  - 增加 `previous_growth` 的单元测试和 runner 覆盖。
  - 更新 `last_week` 测试，断言 share-level 语义。
- 修改：`tests/test_trend_evaluation.py`
  - 增加 `previous_growth` 的评价 runner 覆盖。
- 修改：`README.md`
  - 记录三类已实现基线的命令和输出。
- 修改：`docs/gpt-research/implementation-plan.md`
  - 更新基线表和第 9 步说明，避免继续混淆 Last Week Heat 与 Previous Growth。

下面的实施检查点包含 commit 命令，因为该项目通常偏好小粒度功能提交。只有用户在实现阶段明确授权提交时，才执行 commit 步骤。

---

### 任务 0：固定设计与计划文档

**文件：**
- 修改：`docs/superpowers/specs/2026-05-08-previous-growth-baseline-design.md`
- 新建：`docs/superpowers/plans/2026-05-08-previous-growth-baseline.md`

- [ ] **步骤 1：确认计划阶段文档 diff**

运行：

```sh
git status --short
git diff -- docs/superpowers/specs/2026-05-08-previous-growth-baseline-design.md
git diff -- docs/superpowers/plans/2026-05-08-previous-growth-baseline.md
```

预期：只看到本设计文档的补充修改和本实施计划文件；不应混入模型、测试或 README 实现改动。

- [ ] **步骤 2：如已获授权，单独提交计划阶段文档**

```sh
git add docs/superpowers/specs/2026-05-08-previous-growth-baseline-design.md docs/superpowers/plans/2026-05-08-previous-growth-baseline.md
git commit -m "docs: 收紧 previous_growth 基线实施计划"
```

如果用户没有授权提交，则保留未提交文档改动，但执行后续实现时必须在最终 diff 中单独识别这两个 superpowers 文档，不能把它们误判为实现代码变更。

---

### 任务 1：添加 Previous Growth 基线

**文件：**
- 新建：`src/fashion_trend/trend/models/baselines/previous_growth.py`
- 修改：`src/fashion_trend/trend/models/registry.py`
- 修改：`tests/test_trend_training.py`

- [ ] **步骤 1：先写失败测试和 import**

在 `tests/test_trend_training.py` 中，在现有 `moving_average` import 后增加：

```python
from fashion_trend.trend.models.baselines.previous_growth import (
    PREVIOUS_GROWTH_MODEL_NAME,
    PREVIOUS_GROWTH_PARAMS,
    PreviousGrowthTrainer,
    predict_previous_growth,
)
```

更新 `test_registry_lists_registered_models()`：

```python
def test_registry_lists_registered_models(self) -> None:
    assert list_trend_model_names() == (
        LAST_WEEK_MODEL_NAME,
        MOVING_AVERAGE_MODEL_NAME,
        PREVIOUS_GROWTH_MODEL_NAME,
    )
```

在 `TestTrendTraining` 中，靠近现有模型参数和 registry 测试的位置增加：

```python
def test_previous_growth_params_are_stable(self) -> None:
    assert PREVIOUS_GROWTH_PARAMS == {
        "model_name": "previous_growth",
        "formula": "pred_target_growth = growth_lag_1",
        "derived_formula": (
            "raw_pred_share_t1 = exp(pred_target_growth) * "
            "(share_t + epsilon) - epsilon; "
            "pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))"
        ),
        "epsilon": 1e-6,
    }

def test_registry_returns_previous_growth_trainer(self) -> None:
    trainer = get_trend_model_trainer(PREVIOUS_GROWTH_MODEL_NAME)

    assert isinstance(trainer, PreviousGrowthTrainer)
    assert trainer.name == PREVIOUS_GROWTH_MODEL_NAME
    assert trainer.model_type == MODEL_TYPE_BASELINE
```

在 `test_predict_moving_average_uses_two_growth_lags()` 附近增加公式测试：

```python
def test_predict_previous_growth_uses_growth_lag_1(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    samples = pd.concat(split_frames.values(), ignore_index=True)

    predictions = predict_previous_growth(samples)
    ordered_samples = samples.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )

    assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
    assert set(predictions["model_name"]) == {PREVIOUS_GROWTH_MODEL_NAME}
    pd.testing.assert_series_equal(
        predictions["pred_target_growth"],
        ordered_samples["growth_lag_1"],
        check_names=False,
    )
    expected_share = _expected_normalized_pred_share(
        predictions,
        float(PREVIOUS_GROWTH_PARAMS["epsilon"]),
    )
    pd.testing.assert_series_equal(
        predictions["pred_share_t1"],
        expected_share,
        check_names=False,
    )
    _assert_pred_share_t1_distribution(predictions)
```

在现有 moving average 缺列测试附近增加缺列测试：

```python
def test_predict_previous_growth_rejects_missing_growth_lag(self) -> None:
    samples = sample_trend_model_samples_for_split().assign(split="train")
    samples = samples.drop(columns=["growth_lag_1"])

    with pytest.raises(ValueError, match="growth_lag_1"):
        predict_previous_growth(samples)
```

- [ ] **步骤 2：运行聚焦测试，确认按预期失败**

运行：

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_previous_growth_params_are_stable tests/test_trend_training.py::TestTrendTraining::test_registry_returns_previous_growth_trainer tests/test_trend_training.py::TestTrendTraining::test_predict_previous_growth_uses_growth_lag_1 -q
```

预期：导入阶段失败，因为 `fashion_trend.trend.models.baselines.previous_growth` 尚不存在。

- [ ] **步骤 3：创建 `previous_growth.py`**

创建 `src/fashion_trend/trend/models/baselines/previous_growth.py`，内容如下：

```python
from __future__ import annotations

import pandas as pd

from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.trend.models.base import (
    MODEL_TYPE_BASELINE,
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

PREVIOUS_GROWTH_MODEL_NAME = "previous_growth"
PREVIOUS_GROWTH_PARAMS: dict[str, object] = {
    "model_name": PREVIOUS_GROWTH_MODEL_NAME,
    "formula": "pred_target_growth = growth_lag_1",
    "derived_formula": (
        "raw_pred_share_t1 = exp(pred_target_growth) * "
        "(share_t + epsilon) - epsilon; "
        "pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))"
    ),
    "epsilon": 1e-6,
}

PREVIOUS_GROWTH_REQUIRED_COLUMNS: tuple[str, ...] = (
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


def predict_previous_growth(split_samples: pd.DataFrame) -> pd.DataFrame:
    """生成 previous_growth 基线预测表。"""

    missing_columns = sorted(
        set(PREVIOUS_GROWTH_REQUIRED_COLUMNS) - set(split_samples.columns)
    )
    if missing_columns:
        raise ValueError(
            "previous_growth 模型输入样本缺少必需列: "
            + ", ".join(missing_columns)
        )
    validate_required_columns(
        split_samples,
        PREVIOUS_GROWTH_REQUIRED_COLUMNS,
        source_name="previous_growth 模型输入样本",
    )
    if not set(split_samples["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("previous_growth 模型输入样本存在非法 split。")

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
    predictions.insert(4, "model_name", PREVIOUS_GROWTH_MODEL_NAME)
    predictions["pred_target_growth"] = predictions["growth_lag_1"]
    epsilon = float(PREVIOUS_GROWTH_PARAMS["epsilon"])
    predictions["pred_share_t1"] = derive_normalized_pred_share_t1(
        predictions,
        epsilon,
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return predictions


class PreviousGrowthTrainer:
    """previous_growth 基线训练器，为通用 runner 产出标准 TrendTrainResult。"""

    name = PREVIOUS_GROWTH_MODEL_NAME
    model_type = MODEL_TYPE_BASELINE

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        split_samples = pd.concat(
            [context.split_frames[split_name] for split_name in context.split_order],
            ignore_index=True,
        )
        predictions = predict_previous_growth(split_samples)
        validate_trend_model_predictions(predictions, split_samples)
        return TrendTrainResult(
            model_name=self.name,
            model_type=self.model_type,
            predictions=predictions,
            params=dict(PREVIOUS_GROWTH_PARAMS),
        )
```

- [ ] **步骤 4：注册 `previous_growth`**

将 `src/fashion_trend/trend/models/registry.py` 替换为：

```python
from __future__ import annotations

from fashion_trend.trend.models.base import TrendModelTrainer
from fashion_trend.trend.models.baselines.last_week import (
    LAST_WEEK_MODEL_NAME,
    LastWeekTrainer,
)
from fashion_trend.trend.models.baselines.moving_average import (
    MOVING_AVERAGE_MODEL_NAME,
    MovingAverageTrainer,
)
from fashion_trend.trend.models.baselines.previous_growth import (
    PREVIOUS_GROWTH_MODEL_NAME,
    PreviousGrowthTrainer,
)


class UnknownTrendModelError(ValueError):
    """请求的趋势模型未注册时抛出的错误。"""


TREND_MODEL_REGISTRY: dict[str, TrendModelTrainer] = {
    LAST_WEEK_MODEL_NAME: LastWeekTrainer(),
    MOVING_AVERAGE_MODEL_NAME: MovingAverageTrainer(),
    PREVIOUS_GROWTH_MODEL_NAME: PreviousGrowthTrainer(),
}


def list_trend_model_names() -> tuple[str, ...]:
    """返回当前注册表中可用的趋势模型名。"""

    return tuple(sorted(TREND_MODEL_REGISTRY))


def get_trend_model_trainer(model_name: str) -> TrendModelTrainer:
    """按模型名取得趋势模型训练器。"""

    try:
        return TREND_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(list_trend_model_names())
        raise UnknownTrendModelError(
            f"不支持的趋势模型: {model_name}。可用模型: {available}"
        ) from exc
```

- [ ] **步骤 5：运行 `previous_growth` 聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_previous_growth_params_are_stable tests/test_trend_training.py::TestTrendTraining::test_registry_lists_registered_models tests/test_trend_training.py::TestTrendTraining::test_registry_returns_previous_growth_trainer tests/test_trend_training.py::TestTrendTraining::test_predict_previous_growth_uses_growth_lag_1 tests/test_trend_training.py::TestTrendTraining::test_predict_previous_growth_rejects_missing_growth_lag -q
```

预期：PASS。

- [ ] **步骤 6：如已获授权，提交检查点**

```sh
git add src/fashion_trend/trend/models/baselines/previous_growth.py src/fashion_trend/trend/models/registry.py tests/test_trend_training.py
git commit -m "feat(trend): 添加 previous_growth 基线"
```

---

### 任务 2：将 Last Week 改为 Last Week Heat

**文件：**
- 修改：`src/fashion_trend/trend/models/baselines/last_week.py`
- 修改：`tests/test_trend_training.py`

- [ ] **步骤 1：增加当前 share 归一化测试 helper**

在 `tests/test_trend_training.py` 中，在 `_expected_normalized_pred_share()` 后增加：

```python
def _expected_current_share_distribution(predictions: pd.DataFrame) -> pd.Series:
    """按 last_week 的当前 share 语义计算分组归一化预测 share。"""
    current_share = predictions["share_t"].clip(lower=0.0)
    group_total = current_share.groupby(
        [
            predictions["split"],
            predictions["week_id"],
            predictions["attr_type"],
        ]
    ).transform("sum")
    return current_share / group_total
```

- [ ] **步骤 2：替换 `last_week` 参数和公式测试**

将 `test_last_week_params_are_stable()` 替换为：

```python
def test_last_week_params_are_stable(self) -> None:
    assert LAST_WEEK_PARAMS == {
        "model_name": "last_week",
        "formula": "pred_share_t1 = group_normalize(share_t)",
        "derived_formula": (
            "pred_target_growth = log((pred_share_t1 + epsilon) / "
            "(share_t + epsilon))"
        ),
        "epsilon": 1e-6,
    }
```

将 `test_predict_last_week_uses_growth_lag_1()` 替换为：

```python
def test_predict_last_week_uses_current_share(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    samples = pd.concat(split_frames.values(), ignore_index=True)

    predictions = predict_last_week(samples)

    assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
    assert set(predictions["model_name"]) == {LAST_WEEK_MODEL_NAME}
    expected_share = _expected_current_share_distribution(predictions)
    expected_growth = np.log(
        (expected_share + float(LAST_WEEK_PARAMS["epsilon"]))
        / (predictions["share_t"] + float(LAST_WEEK_PARAMS["epsilon"]))
    )
    pd.testing.assert_series_equal(
        predictions["pred_share_t1"],
        expected_share,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        predictions["pred_target_growth"],
        expected_growth,
        check_names=False,
    )
    _assert_pred_share_t1_distribution(predictions)
```

在 `last_week` 缺列测试附近增加：

```python
def test_predict_last_week_does_not_require_growth_lag_1(self) -> None:
    samples = sample_trend_model_samples_for_split().assign(split="train")
    samples = samples.drop(columns=["growth_lag_1"])

    predictions = predict_last_week(samples)

    assert set(predictions["model_name"]) == {LAST_WEEK_MODEL_NAME}
    _assert_pred_share_t1_distribution(predictions)
```

在同一组 `last_week` 测试附近增加 `share_t` 边界测试：

```python
@pytest.mark.parametrize(
    "case",
    ["negative", "above_one", "bad_total", "all_zero", "non_finite"],
)
def test_predict_last_week_rejects_invalid_share_t(self, case: str) -> None:
    samples = sample_trend_model_samples_for_split().assign(split="train")
    group_mask = (samples["week_id"] == 4) & (
        samples["attr_type"] == "colour_group_name"
    )
    black_mask = group_mask & (samples["attr_value"] == "Black")

    if case == "negative":
        samples.loc[black_mask, "share_t"] = -0.01
    elif case == "above_one":
        samples.loc[black_mask, "share_t"] = 1.01
    elif case == "bad_total":
        samples.loc[black_mask, "share_t"] = 0.80
    elif case == "all_zero":
        samples.loc[group_mask, "share_t"] = 0.0
    elif case == "non_finite":
        samples.loc[black_mask, "share_t"] = float("inf")
    else:
        raise AssertionError(f"未知测试场景: {case}")

    with pytest.raises(ValueError, match="share_t"):
        predict_last_week(samples)
```

- [ ] **步骤 3：运行聚焦测试，确认按预期失败**

运行：

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_last_week_params_are_stable tests/test_trend_training.py::TestTrendTraining::test_predict_last_week_uses_current_share tests/test_trend_training.py::TestTrendTraining::test_predict_last_week_does_not_require_growth_lag_1 tests/test_trend_training.py::TestTrendTraining::test_predict_last_week_rejects_invalid_share_t -q
```

预期：FAIL，因为 `last_week` 仍记录旧 `growth_lag_1` 公式、仍要求 `growth_lag_1`，并且还没有拒绝异常 `share_t`。

- [ ] **步骤 4：重写 `last_week.py`**

将 `src/fashion_trend/trend/models/baselines/last_week.py` 替换为：

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.trend.predictions import (
    validate_trend_model_predictions,
)
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_PRED_SHARE_GROUP_COLUMNS,
    TREND_MODEL_SHARE_TOLERANCE,
    TREND_MODEL_SPLIT_VALUES,
)

LAST_WEEK_MODEL_NAME = "last_week"
LAST_WEEK_PARAMS: dict[str, object] = {
    "model_name": LAST_WEEK_MODEL_NAME,
    "formula": "pred_share_t1 = group_normalize(share_t)",
    "derived_formula": (
        "pred_target_growth = log((pred_share_t1 + epsilon) / "
        "(share_t + epsilon))"
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
    "target_growth",
    "target_rank_in_type_t1",
)


def predict_last_week(split_samples: pd.DataFrame) -> pd.DataFrame:
    """生成 last_week heat 基线预测表。"""

    missing_columns = sorted(
        set(LAST_WEEK_REQUIRED_COLUMNS) - set(split_samples.columns)
    )
    if missing_columns:
        raise ValueError(
            "last_week 模型输入样本缺少必需列: " + ", ".join(missing_columns)
        )
    validate_required_columns(
        split_samples,
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
            "target_growth",
            "target_rank_in_type_t1",
        ],
    ].copy()
    predictions.insert(4, "model_name", LAST_WEEK_MODEL_NAME)
    predictions["pred_share_t1"] = _derive_normalized_current_share(predictions)
    epsilon = float(LAST_WEEK_PARAMS["epsilon"])
    predictions["pred_target_growth"] = _derive_growth_from_pred_share(
        predictions,
        epsilon,
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return predictions


class LastWeekTrainer:
    """last_week heat 基线训练器，为通用 runner 产出标准 TrendTrainResult。"""

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


def _derive_normalized_current_share(predictions: pd.DataFrame) -> pd.Series:
    try:
        current_share = pd.to_numeric(predictions["share_t"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("last_week 模型 share_t 必须为数值。") from exc
    if not np.isfinite(current_share.to_numpy(dtype=float)).all():
        raise ValueError("last_week 模型 share_t 存在非有限数值。")

    below_zero = current_share < -TREND_MODEL_SHARE_TOLERANCE
    above_one = current_share > 1.0 + TREND_MODEL_SHARE_TOLERANCE
    if below_zero.any() or above_one.any():
        raise ValueError("last_week 模型 share_t 必须在 [0, 1] 范围内。")

    group_total = current_share.groupby(
        [predictions[column] for column in TREND_MODEL_PRED_SHARE_GROUP_COLUMNS],
        dropna=False,
    ).transform("sum")
    invalid_total = ~np.isclose(
        group_total,
        1.0,
        rtol=0,
        atol=TREND_MODEL_SHARE_TOLERANCE,
    )
    if invalid_total.any():
        raise ValueError(
            "last_week 模型 share_t 必须在 split/week_id/attr_type 内归一化。"
        )

    non_negative_share = current_share.clip(lower=0.0)
    normalized_total = non_negative_share.groupby(
        [predictions[column] for column in TREND_MODEL_PRED_SHARE_GROUP_COLUMNS],
        dropna=False,
    ).transform("sum")
    if (normalized_total <= 0).any():
        raise ValueError("last_week 模型 share_t 组内总和必须大于 0。")
    normalized_share = non_negative_share / normalized_total
    normalized_share.name = "pred_share_t1"
    return normalized_share


def _derive_growth_from_pred_share(
    predictions: pd.DataFrame,
    epsilon: float,
) -> pd.Series:
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("last_week 模型 epsilon 必须为数值。") from exc
    if epsilon_value < 0 or not np.isfinite(epsilon_value):
        raise ValueError("last_week 模型 epsilon 必须为非负有限数值。")

    try:
        share_t = pd.to_numeric(predictions["share_t"], errors="raise")
        pred_share_t1 = pd.to_numeric(predictions["pred_share_t1"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("last_week 模型增长率派生字段必须为数值。") from exc

    denominator = share_t + epsilon_value
    if (denominator <= 0).any():
        raise ValueError("last_week 模型 share_t 加 epsilon 后必须大于 0。")
    growth = np.log((pred_share_t1 + epsilon_value) / denominator)
    if not np.isfinite(growth.to_numpy(dtype=float)).all():
        raise ValueError("last_week 模型 pred_target_growth 存在非有限数值。")
    growth.name = "pred_target_growth"
    return growth
```

- [ ] **步骤 5：运行 `last_week` 聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_last_week_params_are_stable tests/test_trend_training.py::TestTrendTraining::test_predict_last_week_uses_current_share tests/test_trend_training.py::TestTrendTraining::test_predict_last_week_does_not_require_growth_lag_1 tests/test_trend_training.py::TestTrendTraining::test_predict_last_week_rejects_invalid_share_t tests/test_trend_training.py::TestTrendTraining::test_last_week_trainer_returns_train_result tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_standard_outputs -q
```

预期：PASS。

- [ ] **步骤 6：如已获授权，提交检查点**

```sh
git add src/fashion_trend/trend/models/baselines/last_week.py tests/test_trend_training.py
git commit -m "refactor(trend): 对齐 last_week 热度基线语义"
```

---

### 任务 3：补齐 runner、CLI 和评价覆盖

**文件：**
- 修改：`tests/test_trend_training.py`
- 修改：`tests/test_trend_evaluation.py`

- [ ] **步骤 1：增加 `previous_growth` runner 输出覆盖**

在 `tests/test_trend_training.py` 中，在 `test_run_trend_model_training_writes_moving_average_outputs()` 后增加：

```python
def test_run_trend_model_training_writes_previous_growth_outputs(
    self,
    tmp_path: Path,
) -> None:
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
        PREVIOUS_GROWTH_MODEL_NAME,
        input_paths=input_paths,
        output_root=tmp_path / "outputs" / "models",
    )

    output_dir = tmp_path / "outputs" / "models" / "previous_growth"
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "params.json").exists()
    assert (output_dir / "metadata.json").exists()
    assert metadata["model_name"] == PREVIOUS_GROWTH_MODEL_NAME
    assert metadata["model_type"] == MODEL_TYPE_BASELINE
    assert metadata["rows"] == 40
    assert metadata["extra_artifacts"] == []
```

在 `test_moving_average_trainer_returns_train_result()` 后增加 trainer 结果测试：

```python
def test_previous_growth_trainer_returns_train_result(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    context = TrendTrainContext(
        model_name=PREVIOUS_GROWTH_MODEL_NAME,
        split_frames=split_frames,
        input_paths={
            "train": Path("train.parquet"),
            "valid": Path("valid.parquet"),
            "test": Path("test.parquet"),
        },
        output_dir=Path("outputs/models/previous_growth"),
    )

    result = PreviousGrowthTrainer().train(context)

    assert isinstance(result, TrendTrainResult)
    assert result.model_name == PREVIOUS_GROWTH_MODEL_NAME
    assert result.model_type == MODEL_TYPE_BASELINE
    assert result.params == PREVIOUS_GROWTH_PARAMS
    assert result.artifacts == ()
    assert result.metadata == {}
    assert len(result.predictions) == 40
```

- [ ] **步骤 2：增加 `previous_growth` CLI 覆盖**

在 `tests/test_trend_training.py` 中，在 `test_train_trend_model_main_accepts_moving_average()` 后增加：

```python
def test_train_trend_model_main_accepts_previous_growth(self) -> None:
    train_model = importlib.import_module("10_train_trend_model")
    calls: list[str] = []
    original_run_trend_model_training = train_model.run_trend_model_training

    def fake_run_trend_model_training(model_name: str) -> dict[str, object]:
        calls.append(model_name)
        return {
            "model_name": PREVIOUS_GROWTH_MODEL_NAME,
            "model_type": MODEL_TYPE_BASELINE,
            "rows": 40,
            "weeks": 20,
            "attributes": 2,
            "splits": {
                "train": {
                    "rows": 24,
                    "weeks": 12,
                    "attributes": 2,
                    "week_min": 4,
                    "week_max": 15,
                },
                "valid": {
                    "rows": 8,
                    "weeks": 4,
                    "attributes": 2,
                    "week_min": 16,
                    "week_max": 19,
                },
                "test": {
                    "rows": 8,
                    "weeks": 4,
                    "attributes": 2,
                    "week_min": 20,
                    "week_max": 23,
                },
            },
            "output_dir": "outputs/models/previous_growth",
            "prediction_path": "outputs/models/previous_growth/predictions.csv",
            "params_path": "outputs/models/previous_growth/params.json",
        }

    try:
        train_model.run_trend_model_training = fake_run_trend_model_training

        assert train_model.main(["--model", PREVIOUS_GROWTH_MODEL_NAME]) == 0
    finally:
        train_model.run_trend_model_training = original_run_trend_model_training

    assert calls == [PREVIOUS_GROWTH_MODEL_NAME]
```

- [ ] **步骤 3：增加评价 runner 覆盖**

在 `tests/test_trend_evaluation.py` 中增加 import：

```python
from fashion_trend.trend.models.baselines.previous_growth import (
    PREVIOUS_GROWTH_MODEL_NAME,
)
```

在 `test_run_trend_model_evaluation_reads_moving_average_predictions()` 后增加：

```python
def test_run_trend_model_evaluation_reads_previous_growth_predictions(
    self,
    tmp_path: Path,
) -> None:
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

    model_root = tmp_path / "outputs" / "models"
    metrics_root = tmp_path / "outputs" / "metrics"
    run_trend_model_training(
        PREVIOUS_GROWTH_MODEL_NAME,
        input_paths=input_paths,
        output_root=model_root,
    )

    payload = run_trend_model_evaluation(
        PREVIOUS_GROWTH_MODEL_NAME,
        model_output_root=model_root,
        metrics_output_root=metrics_root,
    )

    metrics_path = metrics_root / "previous_growth" / "trend_metrics.json"
    assert metrics_path.exists()
    assert payload["model_name"] == PREVIOUS_GROWTH_MODEL_NAME
    assert payload["evaluated_splits"] == ["valid", "test"]
    assert "valid" in payload["overall"]
    assert "test" in payload["overall"]
```

- [ ] **步骤 4：运行聚焦集成测试**

运行：

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_previous_growth_outputs tests/test_trend_training.py::TestTrendTraining::test_previous_growth_trainer_returns_train_result tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_accepts_previous_growth tests/test_trend_evaluation.py::TestTrendEvaluation::test_run_trend_model_evaluation_reads_previous_growth_predictions -q
```

预期：PASS。

- [ ] **步骤 5：运行全部趋势训练和评价测试**

运行：

```sh
uv run pytest tests/test_trend_training.py tests/test_trend_evaluation.py -q
```

预期：PASS。

- [ ] **步骤 6：如已获授权，提交检查点**

```sh
git add tests/test_trend_training.py tests/test_trend_evaluation.py
git commit -m "test(trend): 覆盖三类基线训练评价"
```

---

### 任务 4：同步 README 和实施方案

**文件：**
- 修改：`README.md`
- 修改：`docs/gpt-research/implementation-plan.md`

- [ ] **步骤 1：检查残留旧表述**

运行：

```sh
rg -n "Previous Growth|Last Week|last_week|moving_average|outputs/models|outputs/metrics" README.md docs/gpt-research/implementation-plan.md
```

预期：命令输出当前需要审查的所有引用。

- [ ] **步骤 2：更新 README 总述、阶段表和快捷命令**

在 `README.md` 顶部研究主线后，将当前阶段总结改成三类基线都已闭环：

```markdown
现阶段已经完成到趋势 `last_week`、`previous_growth` 与 `moving_average` 三类 baseline 闭环：
```

在 `README.md` 中，更新当前状态表里的基线行：

```markdown
| Last Week Heat 基线 | 已实现 | `outputs/models/last_week/predictions.csv`、`params.json`、`metadata.json` |
| Previous Growth 基线 | 已实现 | `outputs/models/previous_growth/predictions.csv`、`params.json`、`metadata.json` |
| Moving Average 基线 | 已实现 | `outputs/models/moving_average/predictions.csv`、`params.json`、`metadata.json` |
| 趋势评价 | 已实现 | `outputs/metrics/last_week/trend_metrics.json`、`outputs/metrics/previous_growth/trend_metrics.json`、`outputs/metrics/moving_average/trend_metrics.json` |
```

更新列出模型训练和评价命令的代码块，包含：

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
```

- [ ] **步骤 3：更新 README 基线小节**

将当前 `last_week` 小节替换为以下语义：

````markdown
### 9. last_week 基线

`last_week` 基线通过通用趋势模型训练入口运行，模型细节位于
`src/fashion_trend/trend/models/baselines/last_week.py`。当前模型是 Last Week Heat
基线，使用当前周属性占比预测下一周属性占比：

```text
pred_share_t1 = group_normalize(share_t)
pred_target_growth = log((pred_share_t1 + epsilon) / (share_t + epsilon))
```

`pred_share_t1` 在同一 `split/week_id/attr_type` 内归一化，保证输出是合法占比分布。
`pred_target_growth` 由预测占比反推，用于复用统一趋势评价口径。
````

在 `last_week` 小节之后、`moving_average` 小节之前插入：

````markdown
### 10. previous_growth 基线

`previous_growth` 基线复用通用趋势模型训练入口，模型细节位于
`src/fashion_trend/trend/models/baselines/previous_growth.py`。该模型承担实施方案中的
Previous Growth 语义，使用上一段已观测属性占比增长预测下一段增长：

```text
pred_target_growth = growth_lag_1
```

派生 `pred_share_t1` 时，先按目标公式的逆运算得到原始预测占比，再在同一
`split/week_id/attr_type` 内对非负原始值归一化，确保输出是合法占比分布。

预测结果、参数和元数据统一写入：

```sh
outputs/models/previous_growth/predictions.csv
outputs/models/previous_growth/params.json
outputs/models/previous_growth/metadata.json
```

运行命令：

```sh
uv run python src/10_train_trend_model.py --model previous_growth
```
````

随后调整 `moving_average` 和趋势评价标题编号，让章节顺序保持可读。

- [ ] **步骤 4：更新 README 后续阶段和测试覆盖**

在 `README.md` 的“后续阶段”小节中，把两模型表述改成三模型表述：

```markdown
趋势模型训练与评价框架已经落地到 `last_week`、`previous_growth` 与 `moving_average` 三类 baseline，README 继续按计划记录后续边界：
```

同时把“趋势模型扩展”行改为主模型方向：

```markdown
| 趋势模型扩展 | 更多模型文件和趋势预测结果 | 三类必须 baseline 已完成，后续进入 LightGBM 主模型实验 |
```

在“验证”小节的测试覆盖列表中，把 baseline 覆盖说明改为：

```markdown
- `last_week`、`previous_growth` 与 `moving_average` baseline 预测公式、预测表校验、通用训练 runner metadata、artifact 和写出顺序校验。
```

- [ ] **步骤 5：更新实施方案中的基线描述**

在 `docs/gpt-research/implementation-plan.md` 中，更新 `9.1 趋势预测 baseline` 下的基线表，使必选行变为：

```markdown
| Last Week Heat（当前 `last_week` 语义） | $\hat{s}_{a,t+1}=s_{a,t}$ | 必须 | 最简单份额不变基线 |
| Moving Average（当前 `moving_average` 语义） | $\hat{y}_{a,t}=\operatorname{mean}(\mathrm{growth\_lag\_1},\mathrm{growth\_lag\_2})$ | 必须 | 平滑增长基线 |
| Previous Growth（当前 `previous_growth` 语义） | $\hat{y}_{a,t}=y_{a,t-1}$ | 必须 | 增长趋势基线 |
```

将第 9 步命令块更新为：

```text
src/10_train_trend_model.py --model last_week
src/10_train_trend_model.py --model previous_growth
src/10_train_trend_model.py --model moving_average
```

将附近说明替换为：

````markdown
当前实现中，`last_week` 是 Last Week Heat 基线，使用当前周 `share_t` 预测下一周占比；
`previous_growth` 使用 `growth_lag_1` 预测 `target_growth`，承担 Previous Growth 的增长趋势基线语义；
`moving_average` 使用最近两段增长的均值作为平滑基线。三者都通过统一训练入口写出
`outputs/models/<model>/`，并通过统一评价入口写出 `outputs/metrics/<model>/trend_metrics.json`。
````

- [ ] **步骤 6：验证文档引用**

运行：

```sh
rg -n 'last_week` 与 `moving_average|当前 last_week 语义|承担原计划中|后续再做 EWMA|outputs/metrics/last_week/trend_metrics.json' README.md docs/gpt-research/implementation-plan.md
```

预期：没有残留表述说 `last_week` 仍承担 Previous Growth 语义，也没有 README 总述、后续阶段或测试覆盖继续只列 `last_week` 与 `moving_average`。该命令仍可能输出合法的产物路径引用，但上下文必须列出三类基线模型。

- [ ] **步骤 7：如已获授权，提交检查点**

```sh
git add README.md docs/gpt-research/implementation-plan.md
git commit -m "docs: 对齐三类趋势基线语义"
```

---

### 任务 5：完整验证和真实产物检查

**文件：**
- 读取：`data/processed/features/trend_model_samples_train.parquet`
- 读取：`data/processed/features/trend_model_samples_valid.parquet`
- 读取：`data/processed/features/trend_model_samples_test.parquet`
- 读取：`outputs/models/last_week/params.json`
- 读取：`outputs/models/previous_growth/params.json`
- 读取：`outputs/models/moving_average/params.json`
- 读取：`outputs/metrics/last_week/trend_metrics.json`
- 读取：`outputs/metrics/previous_growth/trend_metrics.json`
- 读取：`outputs/metrics/moving_average/trend_metrics.json`

- [ ] **步骤 1：运行完整测试集**

运行：

```sh
uv run pytest
```

预期：PASS。

- [ ] **步骤 2：训练三类基线模型**

运行：

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
```

预期：每条命令退出码为 0，并记录标准 `outputs/models/<model>/` 路径。

- [ ] **步骤 3：评价三类基线模型**

运行：

```sh
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
```

预期：每条命令退出码为 0，并记录 valid/test 趋势指标。

- [ ] **步骤 4：检查参数语义**

运行：

```sh
jq '.model_name, .formula, .derived_formula' outputs/models/last_week/params.json
jq '.model_name, .formula, .derived_formula' outputs/models/previous_growth/params.json
jq '.model_name, .formula, .growth_lags' outputs/models/moving_average/params.json
```

预期：

```text
last_week
pred_share_t1 = group_normalize(share_t)
pred_target_growth = log((pred_share_t1 + epsilon) / (share_t + epsilon))
previous_growth
pred_target_growth = growth_lag_1
moving_average
pred_target_growth = mean(growth_lag_1, growth_lag_2)
```

- [ ] **步骤 5：检查输出文件是否存在**

运行：

```sh
test -f outputs/models/last_week/predictions.csv
test -f outputs/models/last_week/params.json
test -f outputs/models/last_week/metadata.json
test -f outputs/models/previous_growth/predictions.csv
test -f outputs/models/previous_growth/params.json
test -f outputs/models/previous_growth/metadata.json
test -f outputs/models/moving_average/predictions.csv
test -f outputs/models/moving_average/params.json
test -f outputs/models/moving_average/metadata.json
test -f outputs/metrics/last_week/trend_metrics.json
test -f outputs/metrics/previous_growth/trend_metrics.json
test -f outputs/metrics/moving_average/trend_metrics.json
```

预期：每条 `test -f` 命令退出码均为 0。

- [ ] **步骤 6：检查预测行数和指标 split**

运行：

```sh
uv run python -c "import json, pandas as pd; models=['last_week','previous_growth','moving_average']; [print(model, len(pd.read_csv(f'outputs/models/{model}/predictions.csv')), json.load(open(f'outputs/metrics/{model}/trend_metrics.json'))['evaluated_splits']) for model in models]"
```

预期：

```text
last_week 59200 ['valid', 'test']
previous_growth 59200 ['valid', 'test']
moving_average 59200 ['valid', 'test']
```

- [ ] **步骤 7：校验真实预测公式**

运行：

```sh
uv run python - <<'PY'
import numpy as np
import pandas as pd

epsilon = 1e-6
sample_paths = {
    "train": "data/processed/features/trend_model_samples_train.parquet",
    "valid": "data/processed/features/trend_model_samples_valid.parquet",
    "test": "data/processed/features/trend_model_samples_test.parquet",
}
samples = pd.concat(
    [
        pd.read_parquet(path).assign(split=split_name)
        for split_name, path in sample_paths.items()
    ],
    ignore_index=True,
)
sample_lags = samples.loc[:, ["week_id", "attr_id", "split", "growth_lag_1"]]

previous_growth = pd.read_csv("outputs/models/previous_growth/predictions.csv")
previous_growth = previous_growth.merge(
    sample_lags,
    on=["week_id", "attr_id", "split"],
    how="left",
    validate="one_to_one",
)
assert np.allclose(
    previous_growth["pred_target_growth"],
    previous_growth["growth_lag_1"],
    rtol=0,
    atol=1e-12,
)

last_week = pd.read_csv("outputs/models/last_week/predictions.csv")
current_share = last_week["share_t"].clip(lower=0.0)
group_total = current_share.groupby(
    [last_week["split"], last_week["week_id"], last_week["attr_type"]]
).transform("sum")
expected_share = current_share / group_total
expected_growth = np.log(
    (expected_share + epsilon) / (last_week["share_t"] + epsilon)
)
assert np.allclose(
    last_week["pred_share_t1"],
    expected_share,
    rtol=0,
    atol=1e-12,
)
assert np.allclose(
    last_week["pred_target_growth"],
    expected_growth,
    rtol=0,
    atol=1e-12,
)
print("formula_ok=True")
PY
```

预期：

```text
formula_ok=True
```

- [ ] **步骤 8：审查最终 diff**

运行：

```sh
git diff --check
git diff --stat
git diff
```

预期：没有空白字符错误。如果任务 0 已单独提交设计与计划文档，diff 只覆盖计划中的模型、测试、README 和 implementation-plan 文件；如果任务 0 未提交，diff 还会包含 `docs/superpowers/specs/2026-05-08-previous-growth-baseline-design.md` 和 `docs/superpowers/plans/2026-05-08-previous-growth-baseline.md`，需要先把这两个文档作为计划阶段改动单独处理，不能混入最终实现提交。

- [ ] **步骤 9：如已获授权，执行最终提交**

如果前面的实现检查点没有提交，则把完整实现作为一个有边界的提交。这个实现提交不包含 superpowers 设计和计划文档；它们应已在任务 0 单独提交，或在执行本步骤前单独处理。

```sh
git add src/fashion_trend/trend/models/baselines/previous_growth.py src/fashion_trend/trend/models/baselines/last_week.py src/fashion_trend/trend/models/registry.py tests/test_trend_training.py tests/test_trend_evaluation.py README.md docs/gpt-research/implementation-plan.md
git commit -m "feat(trend): 完成三类 baseline 训练评价"
```

如果检查点提交已经完成，则跳过此步骤，并报告已有提交。
