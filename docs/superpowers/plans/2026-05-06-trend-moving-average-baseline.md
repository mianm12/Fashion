# Moving Average 趋势 Baseline 实施计划

> 本文件是历史实施计划。下文出现的 root `src/fashion_trend/models/`、`src/fashion_trend/training.py`、`src/fashion_trend/evaluation.py`、`fashion_trend.models`、`fashion_trend.training` 和 `fashion_trend.evaluation` 均为当时的计划路径或旧实现路径；当前趋势实验层已迁移到 `fashion_trend.trend.models`、`fashion_trend.trend.training` 和 `fashion_trend.trend.evaluation`。

> **给 agentic worker 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，并按任务逐项执行。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标：** 新增 `moving_average` 确定性趋势 baseline，并复用现有趋势模型训练与评价 runner 完成训练和趋势评价。

**架构：** 历史计划写法是在 `fashion_trend.models` 下新增独立 `moving_average` trainer；当前有效架构已迁移到 `fashion_trend.trend.models`，并复用 `fashion_trend.trend.training.run_trend_model_training()` 与 `fashion_trend.trend.evaluation.run_trend_model_evaluation()`。模型使用 `growth_lag_1` 和 `growth_lag_2` 的均值预测 `target_growth`，再用与 `last_week` 一致的 epsilon 公式推导 `pred_share_t1`。

**技术栈：** Python 3.10-3.12、pandas、numpy、标准库 `unittest`，当前实现复用 `fashion_trend.trend.models`、`fashion_trend.trend.training`、`fashion_trend.trend.evaluation` 和其他 `fashion_trend.trend.*` 工具。

---

## 文件结构

- 新建：`src/fashion_trend/models/moving_average.py`
  - 负责 `MOVING_AVERAGE_MODEL_NAME`、`MOVING_AVERAGE_PARAMS`、`MOVING_AVERAGE_GROWTH_LAGS`、`predict_moving_average()` 和 `MovingAverageTrainer`。
- 修改：`src/fashion_trend/models/registry.py`
  - 在 `last_week` 旁注册 `moving_average`。
- 修改：`tests/test_trend.py`
  - 增加 moving average imports、公式测试、registry 测试、trainer 测试、runner 测试、评价复用测试和 CLI 接受测试。
- 修改：`README.md`
  - 增加 `moving_average` 训练/评价命令、输出产物和阶段状态。
- 修改：`docs/gpt-research/implementation-plan.md`
  - 将当前 baseline 训练和趋势评价入口对齐到 `src/10_train_trend_model.py --model <model>` 与 `outputs/metrics/<model>/trend_metrics.json`。

不要修改：

- `last_week` 公式或输出语义。
- 上游趋势样本构造。
- CLI 参数形态；本轮不新增 `--window`、`--lags` 等模型参数。
- 实施任务期间不要修改 `outputs/models/last_week/` 或已有 last week metrics。

下面的 commit 步骤是实施检查点。只有用户在实现阶段明确授权提交时才执行。

---

### 任务 1：添加 Moving Average Registry 契约

**文件：**
- 新建：`src/fashion_trend/models/moving_average.py`
- 修改：`src/fashion_trend/models/registry.py`
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：先写失败 imports 和 registry 测试**

在 `tests/test_trend.py` 中，靠近现有 `last_week` import 增加：

```python
from fashion_trend.models.moving_average import (
    MOVING_AVERAGE_GROWTH_LAGS,
    MOVING_AVERAGE_MODEL_NAME,
    MOVING_AVERAGE_PARAMS,
    MovingAverageTrainer,
    predict_moving_average,
)
```

更新现有 registry 列表测试：

```python
def test_registry_lists_registered_models(self) -> None:
    self.assertEqual(
        list_trend_model_names(),
        (LAST_WEEK_MODEL_NAME, MOVING_AVERAGE_MODEL_NAME),
    )
```

替换现有 unknown model 测试，确保 `moving_average` 不再被视为未知模型：

```python
def test_registry_rejects_unknown_model(self) -> None:
    with self.assertRaisesRegex(UnknownTrendModelError, "unknown_model"):
        get_trend_model_trainer("unknown_model")
```

在现有 `LastWeekBaselineTests` 模型测试附近增加：

```python
def test_moving_average_params_are_stable(self) -> None:
    self.assertEqual(
        MOVING_AVERAGE_PARAMS,
        {
            "model_name": "moving_average",
            "formula": "pred_target_growth = mean(growth_lag_1, growth_lag_2)",
            "derived_formula": (
                "pred_share_t1 = exp(pred_target_growth) * "
                "(share_t + epsilon) - epsilon"
            ),
            "epsilon": 1e-6,
            "growth_lags": ["growth_lag_1", "growth_lag_2"],
        },
    )
    self.assertEqual(MOVING_AVERAGE_GROWTH_LAGS, ("growth_lag_1", "growth_lag_2"))

def test_registry_returns_moving_average_trainer(self) -> None:
    trainer = get_trend_model_trainer(MOVING_AVERAGE_MODEL_NAME)

    self.assertIsInstance(trainer, MovingAverageTrainer)
    self.assertEqual(trainer.name, MOVING_AVERAGE_MODEL_NAME)
    self.assertEqual(trainer.model_type, MODEL_TYPE_BASELINE)
```

更新现有 CLI unknown model 测试：

```python
def test_train_trend_model_main_rejects_unknown_model(self) -> None:
    train_model = importlib.import_module("10_train_trend_model")

    self.assertEqual(train_model.main(["--model", "unknown_model"]), 1)
```

- [ ] **步骤 2：运行聚焦测试，确认按预期失败**

运行：

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

预期：失败或报错，因为 `fashion_trend.models.moving_average` 尚不存在。

- [ ] **步骤 3：新建 moving average trainer 模块**

创建 `src/fashion_trend/models/moving_average.py`：

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.trend.predictions import validate_trend_model_predictions
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)

MOVING_AVERAGE_MODEL_NAME = "moving_average"
MOVING_AVERAGE_GROWTH_LAGS: tuple[str, ...] = ("growth_lag_1", "growth_lag_2")
MOVING_AVERAGE_PARAMS: dict[str, object] = {
    "model_name": MOVING_AVERAGE_MODEL_NAME,
    "formula": "pred_target_growth = mean(growth_lag_1, growth_lag_2)",
    "derived_formula": (
        "pred_share_t1 = exp(pred_target_growth) * "
        "(share_t + epsilon) - epsilon"
    ),
    "epsilon": 1e-6,
    "growth_lags": list(MOVING_AVERAGE_GROWTH_LAGS),
}

MOVING_AVERAGE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "split",
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "share_t",
    *MOVING_AVERAGE_GROWTH_LAGS,
    "target_growth",
    "target_rank_in_type_t1",
)


def predict_moving_average(split_samples: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(
        set(MOVING_AVERAGE_REQUIRED_COLUMNS) - set(split_samples.columns)
    )
    if missing_columns:
        raise ValueError(
            "moving_average 模型输入样本缺少必需列: "
            + ", ".join(missing_columns)
        )
    validate_required_columns(
        split_samples.columns.tolist(),
        MOVING_AVERAGE_REQUIRED_COLUMNS,
        source_name="moving_average 模型输入样本",
    )
    if not set(split_samples["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("moving_average 模型输入样本存在非法 split。")

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
    predictions.insert(4, "model_name", MOVING_AVERAGE_MODEL_NAME)
    growth_lags = _read_growth_lags(split_samples)
    predictions["pred_target_growth"] = growth_lags.mean(axis=1)
    epsilon = float(MOVING_AVERAGE_PARAMS["epsilon"])
    predictions["pred_share_t1"] = (
        np.exp(predictions["pred_target_growth"]) * (predictions["share_t"] + epsilon)
        - epsilon
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
    _validate_finite_predictions(predictions)
    predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return predictions


class MovingAverageTrainer:
    name = MOVING_AVERAGE_MODEL_NAME
    model_type = MODEL_TYPE_BASELINE

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        split_samples = pd.concat(
            [context.split_frames[split_name] for split_name in context.split_order],
            ignore_index=True,
        )
        predictions = predict_moving_average(split_samples)
        validate_trend_model_predictions(predictions, split_samples)
        return TrendTrainResult(
            model_name=self.name,
            model_type=self.model_type,
            predictions=predictions,
            params=dict(MOVING_AVERAGE_PARAMS),
        )


def _read_growth_lags(split_samples: pd.DataFrame) -> pd.DataFrame:
    try:
        return split_samples.loc[:, list(MOVING_AVERAGE_GROWTH_LAGS)].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("moving_average 模型输入增长 lag 必须为数值。") from exc


def _validate_finite_predictions(predictions: pd.DataFrame) -> None:
    numeric_columns = ["share_t", "pred_share_t1", "target_growth", "pred_target_growth"]
    try:
        numeric_values = predictions.loc[:, numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("moving_average 模型预测存在无法解析的数值字段。") from exc
    if not np.isfinite(numeric_values).all():
        raise ValueError("moving_average 模型预测存在非有限数值。")
```

- [ ] **步骤 4：注册 `moving_average`**

更新 `src/fashion_trend/models/registry.py`：

```python
from __future__ import annotations

from fashion_trend.models.base import TrendModelTrainer
from fashion_trend.models.last_week import LAST_WEEK_MODEL_NAME, LastWeekTrainer
from fashion_trend.models.moving_average import (
    MOVING_AVERAGE_MODEL_NAME,
    MovingAverageTrainer,
)


class UnknownTrendModelError(ValueError):
    """Raised when a requested trend model is not registered."""


TREND_MODEL_REGISTRY: dict[str, TrendModelTrainer] = {
    LAST_WEEK_MODEL_NAME: LastWeekTrainer(),
    MOVING_AVERAGE_MODEL_NAME: MovingAverageTrainer(),
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

- [ ] **步骤 5：运行聚焦测试，确认 registry 通过**

运行：

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests.test_moving_average_params_are_stable tests.test_trend.LastWeekBaselineTests.test_registry_lists_registered_models tests.test_trend.LastWeekBaselineTests.test_registry_returns_moving_average_trainer tests.test_trend.LastWeekBaselineTests.test_registry_rejects_unknown_model tests.test_trend.LastWeekBaselineTests.test_train_trend_model_main_rejects_unknown_model -v
```

预期：PASS。

- [ ] **步骤 6：检查点**

如果已获得 commit 授权：

```sh
git add src/fashion_trend/models/moving_average.py src/fashion_trend/models/registry.py tests/test_trend.py
git commit -m "feat(trend): 注册 moving average baseline"
```

---

### 任务 2：补充 Moving Average 公式覆盖

**文件：**
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：写公式和 trainer 测试**

在现有 `predict_last_week` 测试附近增加：

```python
def test_predict_moving_average_uses_two_growth_lags(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    samples = pd.concat(split_frames.values(), ignore_index=True)

    predictions = predict_moving_average(samples)
    ordered_samples = samples.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    expected_growth = ordered_samples.loc[
        :, ["growth_lag_1", "growth_lag_2"]
    ].mean(axis=1)

    self.assertEqual(
        predictions.columns.tolist(), list(TREND_MODEL_PREDICTION_COLUMNS)
    )
    self.assertEqual(set(predictions["model_name"]), {MOVING_AVERAGE_MODEL_NAME})
    pd.testing.assert_series_equal(
        predictions["pred_target_growth"],
        expected_growth,
        check_names=False,
    )
    expected_share = (
        predictions["pred_target_growth"].map(math.exp)
        * (predictions["share_t"] + MOVING_AVERAGE_PARAMS["epsilon"])
        - MOVING_AVERAGE_PARAMS["epsilon"]
    )
    pd.testing.assert_series_equal(
        predictions["pred_share_t1"],
        expected_share,
        check_names=False,
    )

def test_predict_moving_average_rejects_missing_growth_lag(self) -> None:
    samples = sample_trend_model_samples_for_split().assign(split="train")
    samples = samples.drop(columns=["growth_lag_2"])

    with self.assertRaisesRegex(ValueError, "growth_lag_2"):
        predict_moving_average(samples)

def test_predict_moving_average_rejects_illegal_split(self) -> None:
    samples = sample_trend_model_samples_for_split().assign(split="holdout")

    with self.assertRaisesRegex(ValueError, "非法 split"):
        predict_moving_average(samples)

def test_moving_average_trainer_returns_train_result(self) -> None:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    context = TrendTrainContext(
        model_name=MOVING_AVERAGE_MODEL_NAME,
        split_frames=split_frames,
        input_paths={
            "train": Path("train.parquet"),
            "valid": Path("valid.parquet"),
            "test": Path("test.parquet"),
        },
        output_dir=Path("outputs/models/moving_average"),
    )

    result = MovingAverageTrainer().train(context)

    self.assertIsInstance(result, TrendTrainResult)
    self.assertEqual(result.model_name, MOVING_AVERAGE_MODEL_NAME)
    self.assertEqual(result.model_type, MODEL_TYPE_BASELINE)
    self.assertEqual(result.params, MOVING_AVERAGE_PARAMS)
    self.assertEqual(result.artifacts, ())
    self.assertEqual(result.metadata, {})
    self.assertEqual(len(result.predictions), 40)
```

- [ ] **步骤 2：运行聚焦公式测试**

运行：

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests.test_predict_moving_average_uses_two_growth_lags tests.test_trend.LastWeekBaselineTests.test_predict_moving_average_rejects_missing_growth_lag tests.test_trend.LastWeekBaselineTests.test_predict_moving_average_rejects_illegal_split tests.test_trend.LastWeekBaselineTests.test_moving_average_trainer_returns_train_result -v
```

预期：PASS。

- [ ] **步骤 3：运行 baseline 模型测试类**

运行：

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

预期：PASS。

- [ ] **步骤 4：检查点**

如果已获得 commit 授权：

```sh
git add src/fashion_trend/models/moving_average.py tests/test_trend.py
git commit -m "test(trend): 覆盖 moving average 公式"
```

---

### 任务 3：验证 Runner 和评价复用

**文件：**
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：增加 `moving_average` runner 输出测试**

在 `test_run_trend_model_training_writes_standard_outputs` 附近增加：

```python
def test_run_trend_model_training_writes_moving_average_outputs(self) -> None:
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
            MOVING_AVERAGE_MODEL_NAME,
            input_paths=input_paths,
            output_root=tmp_path / "outputs" / "models",
        )

        output_dir = tmp_path / "outputs" / "models" / "moving_average"
        self.assertTrue((output_dir / "predictions.csv").exists())
        self.assertTrue((output_dir / "params.json").exists())
        self.assertTrue((output_dir / "metadata.json").exists())
        self.assertEqual(metadata["model_name"], MOVING_AVERAGE_MODEL_NAME)
        self.assertEqual(metadata["model_type"], MODEL_TYPE_BASELINE)
        self.assertEqual(metadata["rows"], 40)
        self.assertEqual(metadata["extra_artifacts"], [])
        params = json.loads((output_dir / "params.json").read_text(encoding="utf-8"))
        self.assertEqual(params["growth_lags"], ["growth_lag_1", "growth_lag_2"])
```

- [ ] **步骤 2：增加 `moving_average` 评价复用测试**

在现有 `run_trend_model_evaluation` 测试附近增加：

```python
def test_run_trend_model_evaluation_reads_moving_average_predictions(
    self,
) -> None:
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

        model_root = tmp_path / "outputs" / "models"
        metrics_root = tmp_path / "outputs" / "metrics"
        run_trend_model_training(
            MOVING_AVERAGE_MODEL_NAME,
            input_paths=input_paths,
            output_root=model_root,
        )

        payload = run_trend_model_evaluation(
            MOVING_AVERAGE_MODEL_NAME,
            model_output_root=model_root,
            metrics_output_root=metrics_root,
        )

        metrics_path = metrics_root / "moving_average" / "trend_metrics.json"
        self.assertTrue(metrics_path.exists())
        self.assertEqual(payload["model_name"], MOVING_AVERAGE_MODEL_NAME)
        self.assertEqual(payload["evaluated_splits"], ["valid", "test"])
        self.assertIn("valid", payload["overall"])
        self.assertIn("test", payload["overall"])
```

- [ ] **步骤 3：运行聚焦集成测试**

运行：

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests.test_run_trend_model_training_writes_moving_average_outputs tests.test_trend.TrendEvaluationTests.test_run_trend_model_evaluation_reads_moving_average_predictions -v
```

预期：PASS。模型注册后，通用训练 runner 和评价 runner 应该无需生产代码调整即可复用。

- [ ] **步骤 4：检查点**

如果已获得 commit 授权：

```sh
git add tests/test_trend.py
git commit -m "test(trend): 覆盖 moving average 训练评价"
```

---

### 任务 4：补充 Moving Average CLI 覆盖

**文件：**
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：增加 CLI 接受测试**

在 `test_train_trend_model_main_runs_training_and_logs_summary` 附近增加：

```python
def test_train_trend_model_main_accepts_moving_average(self) -> None:
    train_model = importlib.import_module("10_train_trend_model")
    calls: list[str] = []
    original_run_trend_model_training = train_model.run_trend_model_training

    def fake_run_trend_model_training(model_name: str) -> dict[str, object]:
        calls.append(model_name)
        return {
            "model_name": MOVING_AVERAGE_MODEL_NAME,
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
            "output_dir": "outputs/models/moving_average",
            "prediction_path": "outputs/models/moving_average/predictions.csv",
            "params_path": "outputs/models/moving_average/params.json",
        }

    try:
        train_model.run_trend_model_training = fake_run_trend_model_training

        self.assertEqual(
            train_model.main(["--model", MOVING_AVERAGE_MODEL_NAME]),
            0,
        )
    finally:
        train_model.run_trend_model_training = original_run_trend_model_training

    self.assertEqual(calls, [MOVING_AVERAGE_MODEL_NAME])
```

- [ ] **步骤 2：运行 CLI 测试**

运行：

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests.test_train_trend_model_main_accepts_moving_average tests.test_trend.LastWeekBaselineTests.test_train_trend_model_main_rejects_unknown_model -v
```

预期：PASS。

- [ ] **步骤 3：检查点**

如果已获得 commit 授权：

```sh
git add tests/test_trend.py
git commit -m "test(trend): 验证 moving average CLI"
```

---

### 任务 5：同步文档

**文件：**
- 修改：`README.md`
- 修改：`docs/gpt-research/implementation-plan.md`

- [ ] **步骤 1：更新 README 阶段表**

在 `README.md` 中更新当前状态表，加入新 baseline 和 metrics：

```markdown
| Last Week baseline | 已实现 | `outputs/models/last_week/predictions.csv`、`params.json`、`metadata.json` |
| Moving Average baseline | 已实现 | `outputs/models/moving_average/predictions.csv`、`params.json`、`metadata.json` |
| 趋势评价 | 已实现 | `outputs/metrics/last_week/trend_metrics.json`、`outputs/metrics/moving_average/trend_metrics.json` |
```

- [ ] **步骤 2：更新 README 流水线命令**

在流水线命令块中增加：

```sh
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

将 `moving_average` 训练命令放在 `last_week` 训练命令之后，将 `moving_average` 评价命令放在 `last_week` 评价命令之后。

- [ ] **步骤 3：增加 README moving_average 小节**

在现有 `last_week baseline` 小节之后增加：

```markdown
### 10. moving_average baseline

`moving_average` baseline 复用通用趋势模型训练入口，模型细节位于
`src/fashion_trend/models/moving_average.py`。当前模型使用最近两段已观测属性占比增长的简单平均预测下一段增长：

```text
pred_target_growth = mean(growth_lag_1, growth_lag_2)
```

预测结果、参数和元数据统一写入：

```sh
outputs/models/moving_average/predictions.csv
outputs/models/moving_average/params.json
outputs/models/moving_average/metadata.json
```

运行命令：

```sh
uv run python src/10_train_trend_model.py --model moving_average
```
```

将后续 `趋势评价` 标题从 `### 10.` 调整为 `### 11.`。

- [ ] **步骤 4：更新 README 趋势评价小节**

在趋势评价小节说明两个模型各自的评价输出：

```markdown
评价结果按模型写入：

```sh
outputs/metrics/last_week/trend_metrics.json
outputs/metrics/moving_average/trend_metrics.json
```

运行命令：

```sh
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model moving_average
```
```

- [ ] **步骤 5：更新 README 后续阶段文案**

替换：

```markdown
| 趋势模型扩展 | 更多模型文件和趋势预测结果 | 后续再做 Moving Average、EWMA baseline，再考虑 LightGBM |
```

为：

```markdown
| 趋势模型扩展 | 更多模型文件和趋势预测结果 | 后续再做 EWMA baseline，再考虑 LightGBM |
```

将验证 bullet 从：

```markdown
- `last_week` baseline 预测公式、预测表校验、通用训练 runner metadata、artifact 和写出顺序校验。
```

改为：

```markdown
- `last_week` 与 `moving_average` baseline 预测公式、预测表校验、通用训练 runner metadata、artifact 和写出顺序校验。
```

- [ ] **步骤 6：更新 implementation plan 当前入口**

在 `docs/gpt-research/implementation-plan.md` 中，将 baseline 训练产物表行更新为：

```markdown
| baseline 训练 | `10_train_trend_model.py --model <model>` | `outputs/models/<model>/predictions.csv`, `params.json`, `metadata.json` |
```

在“第 8 步：先跑 baseline”中，将旧脚本块替换为：

```text
src/10_train_trend_model.py --model last_week
src/10_train_trend_model.py --model moving_average
```

研究计划 baseline 列表中的 `Previous Growth` 仍作为概念 baseline 保留。不要重写历史研究叙述，只对齐当前实现入口和产物路径。

- [ ] **步骤 7：运行文档 grep 检查**

运行：

```sh
rg -n "moving_average|Moving Average|10_train_trend_model.py --model|outputs/metrics/<model>|08_train_trend_baselines.py|trend_baseline_predictions.csv" README.md docs/gpt-research/implementation-plan.md
```

预期：

- `moving_average` 出现在 README 命令、产物和后续阶段文案中。
- `10_train_trend_model.py --model <model>` 出现在 implementation plan 的 baseline 产物行中。
- `08_train_trend_baselines.py` 的历史引用如果仍存在，只应出现在明确属于原始规划的宽泛路线图中；当前实现章节应使用 `10_train_trend_model.py`。

- [ ] **步骤 8：检查点**

如果已获得 commit 授权：

```sh
git add README.md docs/gpt-research/implementation-plan.md
git commit -m "docs(trend): 同步 moving average baseline"
```

---

### 任务 6：使用真实产物做完整验证

**文件：**
- 预期不修改源码。
- 会读取或写出被 ignore 的运行产物：`outputs/models/moving_average/` 和 `outputs/metrics/moving_average/`。

- [ ] **步骤 1：运行编译检查**

运行：

```sh
uv run python -m py_compile src/fashion_trend/models/moving_average.py src/fashion_trend/models/registry.py
```

预期：命令退出码为 0。

- [ ] **步骤 2：运行完整单元测试**

运行：

```sh
uv run python -m unittest discover -s tests -v
```

预期：全部测试通过。

- [ ] **步骤 3：训练真实 moving average baseline**

运行：

```sh
uv run python src/10_train_trend_model.py --model moving_average
```

预期日志包含：

```text
模型名称: moving_average
模型类型: baseline
预测输出文件: outputs/models/moving_average/predictions.csv
参数输出文件: outputs/models/moving_average/params.json
```

- [ ] **步骤 4：评价真实 moving average baseline**

运行：

```sh
uv run python src/11_eval_trend_model.py --model moving_average
```

预期日志包含：

```text
模型名称: moving_average
评价 split: valid, test
评价输出文件: outputs/metrics/moving_average/trend_metrics.json
```

- [ ] **步骤 5：检查真实输出形状和公式**

运行：

```sh
uv run python -c "import json; from pathlib import Path; import numpy as np; import pandas as pd; pred = pd.read_csv('outputs/models/moving_average/predictions.csv'); params = json.loads(Path('outputs/models/moving_average/params.json').read_text(encoding='utf-8')); meta = json.loads(Path('outputs/models/moving_average/metadata.json').read_text(encoding='utf-8')); frames = [pd.read_parquet(f'data/processed/features/trend_model_samples_{split}.parquet') for split in ('train', 'valid', 'test')]; samples = pd.concat(frames, ignore_index=True); merged = pred.merge(samples[['week_id', 'attr_id', 'growth_lag_1', 'growth_lag_2']], on=['week_id', 'attr_id'], how='left'); expected = merged[['growth_lag_1', 'growth_lag_2']].mean(axis=1); metrics = json.loads(Path('outputs/metrics/moving_average/trend_metrics.json').read_text(encoding='utf-8')); print({'pred_rows': len(pred), 'metadata_rows': meta['rows'], 'model_names': sorted(pred.model_name.unique().tolist()), 'pred_missing': int(pred.isna().sum().sum()), 'formula_ok': bool(np.allclose(merged['pred_target_growth'], expected)), 'params_lags': params['growth_lags'], 'metric_model': metrics['model_name'], 'metric_splits': metrics['evaluated_splits'], 'overall_keys': sorted(metrics['overall'].keys())})"
```

预期输出：

```text
{
  'pred_rows': 59200,
  'metadata_rows': 59200,
  'model_names': ['moving_average'],
  'pred_missing': 0,
  'formula_ok': True,
  'params_lags': ['growth_lag_1', 'growth_lag_2'],
  'metric_model': 'moving_average',
  'metric_splits': ['valid', 'test'],
  'overall_keys': ['test', 'valid']
}
```

如果上游样本数据变化导致行数不同，应至少确认 `pred_rows == metadata_rows`、`pred_missing == 0`、`formula_ok == True`、`metric_splits == ['valid', 'test']`。

- [ ] **步骤 6：确认 git diff 范围**

运行：

```sh
git status --short
git diff --stat
git diff -- src/fashion_trend/models/moving_average.py src/fashion_trend/models/registry.py tests/test_trend.py README.md docs/gpt-research/implementation-plan.md
```

预期：

- tracked code/docs 改动只限于五个计划内文件。
- 被 ignore 的 `outputs/` 产物即使已生成，也不应出现在 `git status --short` 中。

- [ ] **步骤 7：最终检查点**

如果已获得 commit 授权且前面检查点未分开提交：

```sh
git add src/fashion_trend/models/moving_average.py src/fashion_trend/models/registry.py tests/test_trend.py README.md docs/gpt-research/implementation-plan.md
git commit -m "feat(trend): 添加 moving average baseline"
```

---

## 自审

- Spec 覆盖：已覆盖独立 `moving_average` trainer、registry 注册、固定两段增长均值公式、标准模型产物、评价复用、测试、文档和真实产物验证。
- 范围检查：本计划只实现一个确定性 baseline 和对应趋势评价，不新增 EWMA、LightGBM、推荐逻辑、模型对比汇总表、可配置窗口或上游样本改动。
- 类型一致性：计划中一致使用现有 `TrendTrainContext`、`TrendTrainResult`、`MODEL_TYPE_BASELINE`、`TREND_MODEL_PREDICTION_COLUMNS`、`run_trend_model_training()` 和 `run_trend_model_evaluation()`。
- 占位检查：没有开放占位标记或临时未实现代码。
