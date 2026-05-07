# 趋势评价模块实施计划

> **给 agentic worker 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，并按任务逐项执行。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标：** 构建可复用的趋势模型评价 runner，读取 `outputs/models/<model>/predictions.csv`，写出 `outputs/metrics/<model>/trend_metrics.json`。

**架构：** 新增 `fashion_trend.evaluation` 模块，负责路径推导、预测读取、输入校验、指标计算、JSON payload 构造和写出。新增 `src/11_eval_trend_model.py` 作为薄 CLI，保持训练产物在 `outputs/models/<model>/`，趋势评价产物在 `outputs/metrics/<model>/`。

**技术栈：** Python 3.10-3.12、pandas、numpy、标准库 `argparse`、`json`、`math`、`unittest`，复用现有 `fashion_trend.config`、`fashion_trend.trend`、`fashion_trend.training` 中的工具。

---

## 文件结构

- 新建：`src/fashion_trend/evaluation.py`
  - 负责趋势评价常量、输出路径推导、预测 CSV 读取、评价输入校验、分组指标、payload 构造、metrics 写出和 `run_trend_model_evaluation()`。
- 新建：`src/11_eval_trend_model.py`
  - 薄 CLI 入口。解析 `--model`，调用 `run_trend_model_evaluation()`，打印摘要，返回稳定退出码。
- 修改：`tests/test_trend.py`
  - 增加评价输入校验、指标公式、聚合逻辑、JSON 安全、写出边界、runner 和 CLI 行为测试。
- 修改：`README.md`
  - 记录新的趋势评价阶段、命令、输出文件和测试覆盖。
- 修改：`docs/gpt-research/implementation-plan.md`
  - 轻量同步当前实现脚本名和产物路径。

不要修改：

- `outputs/models/<model>/` 下的训练产物。
- 既有历史 specs 或 plans。
- 依赖文件；本任务不需要新增依赖。

下面的 commit 步骤是实施检查点。只有用户明确授权实现阶段提交时才执行。

---

### Task 1: 添加评价路径推导和预测输入校验

**文件：**
- 新建：`src/fashion_trend/evaluation.py`
- 修改：`tests/test_trend.py`

- [ ] **Step 1: 先写失败测试和导入**

在 `tests/test_trend.py` 顶部 imports 增加：

```python
import json
```

在现有项目 imports 附近增加：

```python
from fashion_trend.evaluation import (
    TREND_EVALUATION_K_VALUES,
    TREND_EVALUATION_SPLITS,
    derive_trend_metric_output_paths,
    read_trend_model_predictions,
    validate_trend_model_predictions_for_evaluation,
)
```

在现有趋势模型测试附近追加 helper 和测试类：

```python
def sample_trend_predictions_for_evaluation() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("train", 8),
        ("valid", 10),
        ("valid", 11),
        ("test", 12),
        ("test", 13),
    ]
    for split_name, week_id in specs:
        rows.extend(
            [
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.40,
                    "pred_share_t1": 0.52,
                    "target_growth": 3.0,
                    "pred_target_growth": 2.8,
                    "target_rank_in_type_t1": 1,
                },
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::White",
                    "attr_type": "colour_group_name",
                    "attr_value": "White",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.30,
                    "pred_share_t1": 0.20,
                    "target_growth": 2.0,
                    "pred_target_growth": 1.0,
                    "target_rank_in_type_t1": 2,
                },
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::Blue",
                    "attr_type": "colour_group_name",
                    "attr_value": "Blue",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.20,
                    "pred_share_t1": 0.25,
                    "target_growth": 1.0,
                    "pred_target_growth": 1.5,
                    "target_rank_in_type_t1": 3,
                },
                {
                    "week_id": week_id,
                    "attr_id": "product_type_name::Dress",
                    "attr_type": "product_type_name",
                    "attr_value": "Dress",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.55,
                    "pred_share_t1": 0.62,
                    "target_growth": 1.5,
                    "pred_target_growth": 1.0,
                    "target_rank_in_type_t1": 1,
                },
                {
                    "week_id": week_id,
                    "attr_id": "product_type_name::Vest top",
                    "attr_type": "product_type_name",
                    "attr_value": "Vest top",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.45,
                    "pred_share_t1": 0.40,
                    "target_growth": 0.5,
                    "pred_target_growth": 0.7,
                    "target_rank_in_type_t1": 2,
                },
            ]
        )
    return pd.DataFrame(rows).loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]


class TrendEvaluationTests(unittest.TestCase):
    def test_derive_trend_metric_output_paths_uses_model_name(self) -> None:
        paths = derive_trend_metric_output_paths(
            "last_week",
            model_output_root=Path("outputs/models"),
            metrics_output_root=Path("outputs/metrics"),
        )

        self.assertEqual(paths["output_dir"], Path("outputs/metrics/last_week"))
        self.assertEqual(
            paths["predictions"], Path("outputs/models/last_week/predictions.csv")
        )
        self.assertEqual(
            paths["metrics"], Path("outputs/metrics/last_week/trend_metrics.json")
        )

    def test_read_trend_model_predictions_preserves_contract_columns(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        with TemporaryDirectory() as tmp_dir:
            prediction_path = Path(tmp_dir) / "predictions.csv"
            write_trend_csv(predictions, prediction_path)

            loaded = read_trend_model_predictions(prediction_path)

        self.assertEqual(loaded.columns.tolist(), list(TREND_MODEL_PREDICTION_COLUMNS))
        self.assertEqual(len(loaded), len(predictions))

    def test_validate_trend_model_predictions_for_evaluation_accepts_valid_table(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()

        validate_trend_model_predictions_for_evaluation(predictions, "last_week")

    def test_validate_trend_model_predictions_for_evaluation_rejects_missing_test(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions = predictions[predictions["split"] != "test"].copy()

        with self.assertRaisesRegex(ValueError, "缺少评价 split"):
            validate_trend_model_predictions_for_evaluation(predictions, "last_week")

    def test_validate_trend_model_predictions_for_evaluation_rejects_wrong_model(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()

        with self.assertRaisesRegex(ValueError, "model_name"):
            validate_trend_model_predictions_for_evaluation(predictions, "moving_average")

    def test_validate_trend_model_predictions_for_evaluation_rejects_non_finite(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions.loc[predictions.index[0], "pred_target_growth"] = float("nan")

        with self.assertRaisesRegex(ValueError, "非有限数值"):
            validate_trend_model_predictions_for_evaluation(predictions, "last_week")
```

- [ ] **Step 2: 运行测试，确认按预期失败**

运行：

```sh
uv run python -m unittest tests.test_trend.TrendEvaluationTests -v
```

预期：失败，报 `ModuleNotFoundError: No module named 'fashion_trend.evaluation'`。

- [ ] **Step 3: 新建 `evaluation.py`，实现路径、读取和校验**

创建 `src/fashion_trend/evaluation.py`：

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from fashion_trend.config import OUTPUT_METRICS_DIR, OUTPUT_MODELS_DIR
from fashion_trend.foundation.dataframe import (
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_key,
)
from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)

TREND_EVALUATION_SPLITS: tuple[str, ...] = ("valid", "test")
TREND_EVALUATION_K_VALUES: tuple[int, ...] = (5, 10, 20)
TREND_EVALUATION_GROUP_COLUMNS: tuple[str, ...] = ("split", "week_id", "attr_type")
TREND_EVALUATION_TARGET_COLUMN = "target_growth"
TREND_EVALUATION_PREDICTION_COLUMN = "pred_target_growth"

_TEXT_PREDICTION_DTYPES: dict[str, str] = {
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "model_name": "string",
    "split": "string",
}


def derive_trend_metric_output_paths(
    model_name: str,
    model_output_root: Path = OUTPUT_MODELS_DIR,
    metrics_output_root: Path = OUTPUT_METRICS_DIR,
) -> dict[str, Path]:
    """根据模型名推导预测输入路径和趋势评价输出路径。"""
    output_dir = metrics_output_root / model_name
    return {
        "output_dir": output_dir,
        "predictions": model_output_root / model_name / "predictions.csv",
        "metrics": output_dir / "trend_metrics.json",
    }


def read_trend_model_predictions(prediction_path: Path) -> pd.DataFrame:
    """读取趋势模型预测 CSV，并保留标准列契约。"""
    if not prediction_path.exists():
        raise FileNotFoundError(f"趋势模型预测文件不存在: {prediction_path}")
    predictions = pd.read_csv(prediction_path, dtype=_TEXT_PREDICTION_DTYPES)
    validate_required_columns(
        predictions.columns.tolist(),
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name=f"趋势模型预测文件: {prediction_path}",
    )
    predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)].copy()
    return predictions


def validate_trend_model_predictions_for_evaluation(
    predictions: pd.DataFrame,
    model_name: str,
) -> None:
    """在计算趋势评价指标前校验预测表。"""
    if predictions.columns.tolist() != list(TREND_MODEL_PREDICTION_COLUMNS):
        raise ValueError("趋势模型评价预测表列必须与契约完全一致。")
    validate_required_columns(
        predictions.columns.tolist(),
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name="趋势模型评价预测表",
    )
    validate_no_missing_values(
        predictions,
        TREND_MODEL_PREDICTION_COLUMNS,
        source_name="趋势模型评价预测表",
    )
    validate_unique_key(
        predictions,
        ["week_id", "attr_id", "model_name"],
        source_name="趋势模型评价预测表",
    )
    split_values = set(predictions["split"].astype(str))
    if not split_values.issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("趋势模型评价预测表存在非法 split。")
    missing_eval_splits = set(TREND_EVALUATION_SPLITS) - split_values
    if missing_eval_splits:
        raise ValueError(f"趋势模型评价预测表缺少评价 split: {sorted(missing_eval_splits)}")

    model_values = set(predictions["model_name"].astype(str))
    if model_values != {model_name}:
        raise ValueError(
            "趋势模型评价预测表 model_name 与请求不一致: "
            f"expected={model_name}, actual={sorted(model_values)}"
        )

    _validate_integer_week_ids(predictions["week_id"], "趋势模型评价预测表")
    numeric_columns = [
        "week_id",
        "share_t",
        "pred_share_t1",
        "target_growth",
        "pred_target_growth",
        "target_rank_in_type_t1",
    ]
    numeric_values = predictions.loc[:, numeric_columns]
    try:
        finite_numeric_values = numeric_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势模型评价预测表无法校验数值字段。") from exc
    if not np.isfinite(finite_numeric_values).all():
        raise ValueError("趋势模型评价预测表存在非有限数值。")


def _validate_integer_week_ids(week_ids: pd.Series, source_name: str) -> pd.Series:
    try:
        numeric_week_ids = pd.to_numeric(week_ids, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} week_id 必须为整数。") from exc
    if numeric_week_ids.isna().any() or not (numeric_week_ids % 1 == 0).all():
        raise ValueError(f"{source_name} week_id 必须为整数。")
    return numeric_week_ids.astype("int64")
```

- [ ] **Step 4: 运行本任务测试**

运行：

```sh
uv run python -m unittest tests.test_trend.TrendEvaluationTests -v
```

预期：本任务新增的 6 个路径和输入校验测试通过。

- [ ] **Step 5: 运行完整趋势测试模块**

运行：

```sh
uv run python -m unittest tests.test_trend -v
```

预期：通过。

- [ ] **Step 6: 如已授权 commit，提交本检查点**

只有明确授权 commit 时才运行：

```sh
git add src/fashion_trend/evaluation.py tests/test_trend.py
git commit -m "feat(trend): 添加趋势评价输入校验"
```

---

### Task 2: 添加分组指标与聚合逻辑

**文件：**
- 修改：`src/fashion_trend/evaluation.py`
- 修改：`tests/test_trend.py`

- [ ] **Step 1: 先写失败的指标测试**

扩展 `tests/test_trend.py` 中的 `fashion_trend.evaluation` import：

```python
from fashion_trend.evaluation import (
    TREND_EVALUATION_K_VALUES,
    TREND_EVALUATION_SPLITS,
    build_trend_metrics_payload,
    compute_trend_group_metrics,
    compute_trend_metrics,
    derive_trend_metric_output_paths,
    read_trend_model_predictions,
    validate_trend_model_predictions_for_evaluation,
)
```

在 `TrendEvaluationTests` 中追加：

```python
    def test_compute_trend_group_metrics_reports_regression_and_ranking(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        group = predictions[
            (predictions["split"] == "valid")
            & (predictions["week_id"] == 10)
            & (predictions["attr_type"] == "colour_group_name")
        ].copy()

        metrics = compute_trend_group_metrics(group, k_values=(2, 3))

        self.assertTrue(math.isclose(metrics["mae"], 0.5666666667, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["rmse"], math.sqrt(0.43), rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["spearman"], 0.5, rel_tol=1e-9))
        self.assertEqual(metrics["precision_at_k"]["2"], 0.5)
        self.assertEqual(metrics["recall_at_k"]["2"], 0.5)
        self.assertEqual(metrics["precision_at_k"]["3"], 1.0)
        self.assertEqual(metrics["recall_at_k"]["3"], 1.0)
        expected_ndcg_2 = 2.0 / (2.0 + (1.0 / math.log2(3.0)))
        self.assertTrue(
            math.isclose(metrics["ndcg_at_k"]["2"], expected_ndcg_2, rel_tol=1e-9)
        )
        self.assertEqual(metrics["ndcg_at_k"]["3"], 1.0)

    def test_compute_trend_group_metrics_uses_effective_k_for_small_group(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        group = predictions[
            (predictions["split"] == "valid")
            & (predictions["week_id"] == 10)
            & (predictions["attr_type"] == "product_type_name")
        ].copy()

        metrics = compute_trend_group_metrics(group, k_values=(5,))

        self.assertEqual(metrics["precision_at_k"]["5"], 1.0)
        self.assertEqual(metrics["recall_at_k"]["5"], 1.0)
        self.assertEqual(metrics["ndcg_at_k"]["5"], 1.0)

    def test_compute_trend_group_metrics_returns_null_for_constant_ranking(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        group = predictions[
            (predictions["split"] == "valid")
            & (predictions["week_id"] == 10)
            & (predictions["attr_type"] == "colour_group_name")
        ].copy()
        group["target_growth"] = 1.0
        group["pred_target_growth"] = 1.0

        metrics = compute_trend_group_metrics(group, k_values=(2,))

        self.assertIsNone(metrics["spearman"])
        self.assertIsNone(metrics["ndcg_at_k"]["2"])

    def test_compute_trend_metrics_summarizes_valid_and_test_only(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()

        metrics = compute_trend_metrics(predictions, k_values=(2, 3))

        self.assertEqual(set(metrics["overall"]), {"valid", "test"})
        self.assertEqual(set(metrics["by_attr_type"]), {"valid", "test"})
        self.assertEqual(metrics["groups"]["valid"]["rows"], 10)
        self.assertEqual(metrics["groups"]["valid"]["weeks"], 2)
        self.assertEqual(metrics["groups"]["valid"]["attr_types"], 2)
        self.assertEqual(metrics["groups"]["valid"]["ranking_groups"], 4)
        self.assertNotIn("train", metrics["overall"])
        self.assertIn("colour_group_name", metrics["by_attr_type"]["test"])
        self.assertIn("product_type_name", metrics["by_attr_type"]["test"])

    def test_build_trend_metrics_payload_records_contract(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        paths = derive_trend_metric_output_paths(
            "last_week",
            model_output_root=Path("outputs/models"),
            metrics_output_root=Path("outputs/metrics"),
        )

        payload = build_trend_metrics_payload(
            predictions,
            model_name="last_week",
            prediction_path=paths["predictions"],
            output_path=paths["metrics"],
            k_values=(2, 3),
        )

        self.assertEqual(payload["model_name"], "last_week")
        self.assertEqual(
            payload["prediction_path"], "outputs/models/last_week/predictions.csv"
        )
        self.assertEqual(
            payload["output_path"], "outputs/metrics/last_week/trend_metrics.json"
        )
        self.assertEqual(payload["evaluated_splits"], ["valid", "test"])
        self.assertEqual(payload["ranking"]["k_values"], [2, 3])
        self.assertEqual(payload["ranking"]["group_by"], ["split", "week_id", "attr_type"])
        json.dumps(payload, allow_nan=False)
```

- [ ] **Step 2: 运行测试，确认按预期失败**

运行：

```sh
uv run python -m unittest tests.test_trend.TrendEvaluationTests -v
```

预期：失败，提示 `compute_trend_group_metrics`、`compute_trend_metrics`、`build_trend_metrics_payload` 无法导入。

- [ ] **Step 3: 实现指标与聚合逻辑**

在 `src/fashion_trend/evaluation.py` 的 `_validate_integer_week_ids()` 后追加：

```python
def compute_trend_group_metrics(
    group_predictions: pd.DataFrame,
    k_values: Sequence[int] = TREND_EVALUATION_K_VALUES,
) -> dict[str, object]:
    """计算单个 split-week-attr_type 分组的趋势评价指标。"""
    if group_predictions.empty:
        raise ValueError("趋势评价分组不能为空。")
    _validate_k_values(k_values)

    target = pd.to_numeric(
        group_predictions[TREND_EVALUATION_TARGET_COLUMN],
        errors="raise",
    ).astype(float)
    prediction = pd.to_numeric(
        group_predictions[TREND_EVALUATION_PREDICTION_COLUMN],
        errors="raise",
    ).astype(float)
    errors = target - prediction

    metrics: dict[str, object] = {
        "mae": _json_float(np.abs(errors).mean()),
        "rmse": _json_float(math.sqrt(float(np.square(errors).mean()))),
        "spearman": _spearman_or_none(target, prediction),
        "precision_at_k": {},
        "recall_at_k": {},
        "ndcg_at_k": {},
    }
    for k in k_values:
        key = str(k)
        effective_k = min(k, len(group_predictions))
        predicted_top = _top_attr_ids(
            group_predictions,
            TREND_EVALUATION_PREDICTION_COLUMN,
            effective_k,
        )
        actual_top = _top_attr_ids(
            group_predictions,
            TREND_EVALUATION_TARGET_COLUMN,
            effective_k,
        )
        hits = len(set(predicted_top) & set(actual_top))
        metrics["precision_at_k"][key] = _json_float(hits / effective_k)
        metrics["recall_at_k"][key] = _json_float(hits / effective_k)
        metrics["ndcg_at_k"][key] = _ndcg_or_none(group_predictions, effective_k)
    return metrics


def compute_trend_metrics(
    predictions: pd.DataFrame,
    k_values: Sequence[int] = TREND_EVALUATION_K_VALUES,
) -> dict[str, object]:
    """计算 valid/test 的 overall 与 by_attr_type 趋势评价指标。"""
    _validate_k_values(k_values)
    eval_predictions = predictions[
        predictions["split"].astype(str).isin(TREND_EVALUATION_SPLITS)
    ].copy()
    group_records: list[dict[str, object]] = []
    for group_key, group_predictions in eval_predictions.groupby(
        list(TREND_EVALUATION_GROUP_COLUMNS),
        sort=True,
        dropna=False,
    ):
        split_name, week_id, attr_type = group_key
        group_metrics = compute_trend_group_metrics(group_predictions, k_values)
        group_records.append(
            {
                "split": str(split_name),
                "week_id": int(week_id),
                "attr_type": str(attr_type),
                "rows": int(len(group_predictions)),
                "metrics": group_metrics,
            }
        )

    overall: dict[str, dict[str, object]] = {}
    by_attr_type: dict[str, dict[str, dict[str, object]]] = {}
    groups: dict[str, dict[str, int]] = {}
    for split_name in TREND_EVALUATION_SPLITS:
        split_predictions = eval_predictions[eval_predictions["split"] == split_name]
        split_group_records = [
            record for record in group_records if record["split"] == split_name
        ]
        overall[split_name] = _summarize_metric_records(split_group_records, k_values)
        by_attr_type[split_name] = {}
        for attr_type in sorted(split_predictions["attr_type"].astype(str).unique()):
            attr_records = [
                record
                for record in split_group_records
                if record["attr_type"] == attr_type
            ]
            by_attr_type[split_name][attr_type] = _summarize_metric_records(
                attr_records,
                k_values,
            )
        groups[split_name] = {
            "rows": int(len(split_predictions)),
            "weeks": int(split_predictions["week_id"].nunique()),
            "attr_types": int(split_predictions["attr_type"].nunique()),
            "ranking_groups": int(len(split_group_records)),
        }

    return {
        "overall": overall,
        "by_attr_type": by_attr_type,
        "groups": groups,
    }


def build_trend_metrics_payload(
    predictions: pd.DataFrame,
    model_name: str,
    prediction_path: Path,
    output_path: Path,
    k_values: Sequence[int] = TREND_EVALUATION_K_VALUES,
) -> dict[str, object]:
    """构建稳定的 trend_metrics.json payload。"""
    validate_trend_model_predictions_for_evaluation(predictions, model_name)
    metrics = compute_trend_metrics(predictions, k_values)
    payload: dict[str, object] = {
        "model_name": model_name,
        "prediction_path": str(prediction_path),
        "output_path": str(output_path),
        "evaluated_splits": list(TREND_EVALUATION_SPLITS),
        "ranking": {
            "target_column": TREND_EVALUATION_TARGET_COLUMN,
            "prediction_column": TREND_EVALUATION_PREDICTION_COLUMN,
            "group_by": list(TREND_EVALUATION_GROUP_COLUMNS),
            "k_values": [int(k) for k in k_values],
        },
        "overall": metrics["overall"],
        "by_attr_type": metrics["by_attr_type"],
        "groups": metrics["groups"],
    }
    _validate_json_payload(payload)
    return payload


def _validate_k_values(k_values: Sequence[int]) -> None:
    if not k_values:
        raise ValueError("趋势评价 K 值不能为空。")
    invalid_values = [k for k in k_values if int(k) <= 0]
    if invalid_values:
        raise ValueError(f"趋势评价 K 值必须为正整数: {invalid_values}")


def _top_attr_ids(
    predictions: pd.DataFrame,
    score_column: str,
    k: int,
) -> list[str]:
    sorted_predictions = predictions.sort_values(
        [score_column, "attr_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return sorted_predictions.head(k)["attr_id"].astype(str).tolist()


def _spearman_or_none(target: pd.Series, prediction: pd.Series) -> float | None:
    if target.nunique(dropna=False) < 2 or prediction.nunique(dropna=False) < 2:
        return None
    target_rank = target.rank(method="average")
    prediction_rank = prediction.rank(method="average")
    correlation = target_rank.corr(prediction_rank, method="pearson")
    if pd.isna(correlation):
        return None
    return _json_float(correlation)


def _ndcg_or_none(predictions: pd.DataFrame, k: int) -> float | None:
    target = pd.to_numeric(
        predictions[TREND_EVALUATION_TARGET_COLUMN],
        errors="raise",
    ).astype(float)
    relevance = target - float(target.min())
    if bool((relevance == 0).all()):
        return None

    relevance_by_attr = dict(zip(predictions["attr_id"].astype(str), relevance))
    predicted_top = _top_attr_ids(
        predictions,
        TREND_EVALUATION_PREDICTION_COLUMN,
        k,
    )
    ideal_top = _top_attr_ids(
        predictions.assign(_relevance=relevance),
        "_relevance",
        k,
    )
    dcg = _discounted_gain(predicted_top, relevance_by_attr)
    ideal_dcg = _discounted_gain(ideal_top, relevance_by_attr)
    if ideal_dcg == 0:
        return None
    return _json_float(dcg / ideal_dcg)


def _discounted_gain(attr_ids: list[str], relevance_by_attr: Mapping[str, float]) -> float:
    return float(
        sum(
            relevance_by_attr[attr_id] / math.log2(position + 2)
            for position, attr_id in enumerate(attr_ids)
        )
    )


def _summarize_metric_records(
    group_records: list[dict[str, object]],
    k_values: Sequence[int],
) -> dict[str, object]:
    if not group_records:
        return {
            "mae": None,
            "rmse": None,
            "spearman": None,
            "precision_at_k": {str(k): None for k in k_values},
            "recall_at_k": {str(k): None for k in k_values},
            "ndcg_at_k": {str(k): None for k in k_values},
        }

    metric_payloads = [record["metrics"] for record in group_records]
    return {
        "mae": _mean_or_none([metrics["mae"] for metrics in metric_payloads]),
        "rmse": _mean_or_none([metrics["rmse"] for metrics in metric_payloads]),
        "spearman": _mean_or_none(
            [metrics["spearman"] for metrics in metric_payloads]
        ),
        "precision_at_k": {
            str(k): _mean_or_none(
                [metrics["precision_at_k"][str(k)] for metrics in metric_payloads]
            )
            for k in k_values
        },
        "recall_at_k": {
            str(k): _mean_or_none(
                [metrics["recall_at_k"][str(k)] for metrics in metric_payloads]
            )
            for k in k_values
        },
        "ndcg_at_k": {
            str(k): _mean_or_none(
                [metrics["ndcg_at_k"][str(k)] for metrics in metric_payloads]
            )
            for k in k_values
        },
    }


def _mean_or_none(values: Sequence[object]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return _json_float(sum(numeric_values) / len(numeric_values))


def _json_float(value: object) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("趋势评价指标存在非有限数值。")
    return numeric_value


def _validate_json_payload(payload: dict[str, object]) -> None:
    try:
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势评价指标不能序列化为合法 JSON。") from exc
```

- [ ] **Step 4: 运行指标测试**

运行：

```sh
uv run python -m unittest tests.test_trend.TrendEvaluationTests -v
```

预期：截至目前所有评价测试通过。

- [ ] **Step 5: 运行完整测试**

运行：

```sh
uv run python -m unittest discover -s tests -v
```

预期：通过。

- [ ] **Step 6: 如已授权 commit，提交本检查点**

只有明确授权 commit 时才运行：

```sh
git add src/fashion_trend/evaluation.py tests/test_trend.py
git commit -m "feat(trend): 计算趋势评价指标"
```

---

### Task 3: 添加 metrics 写出和评价 runner

**文件：**
- 修改：`src/fashion_trend/evaluation.py`
- 修改：`tests/test_trend.py`

- [ ] **Step 1: 先写失败的 writer 和 runner 测试**

扩展 `tests/test_trend.py` 中的 `fashion_trend.evaluation` import：

```python
from fashion_trend.evaluation import (
    TREND_EVALUATION_K_VALUES,
    TREND_EVALUATION_SPLITS,
    build_trend_metrics_payload,
    compute_trend_group_metrics,
    compute_trend_metrics,
    derive_trend_metric_output_paths,
    read_trend_model_predictions,
    run_trend_model_evaluation,
    validate_trend_model_predictions_for_evaluation,
    write_trend_metrics,
)
```

在 `TrendEvaluationTests` 中追加：

```python
    def test_write_trend_metrics_writes_json_without_touching_model_outputs(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prediction_path = tmp_path / "outputs" / "models" / "last_week" / "predictions.csv"
            metrics_path = tmp_path / "outputs" / "metrics" / "last_week" / "trend_metrics.json"
            model_metadata_path = prediction_path.parent / "metadata.json"
            write_trend_csv(predictions, prediction_path)
            write_json({"model_name": "last_week"}, model_metadata_path)
            payload = build_trend_metrics_payload(
                predictions,
                model_name="last_week",
                prediction_path=prediction_path,
                output_path=metrics_path,
                k_values=(2,),
            )

            write_trend_metrics(payload, metrics_path)

            self.assertTrue(metrics_path.exists())
            self.assertTrue(prediction_path.exists())
            self.assertTrue(model_metadata_path.exists())
            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(written["model_name"], "last_week")
            self.assertEqual(set(written["overall"]), {"valid", "test"})

    def test_run_trend_model_evaluation_reads_predictions_and_writes_metrics(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_root = tmp_path / "outputs" / "models"
            metrics_root = tmp_path / "outputs" / "metrics"
            prediction_path = model_root / "last_week" / "predictions.csv"
            write_trend_csv(predictions, prediction_path)

            payload = run_trend_model_evaluation(
                "last_week",
                model_output_root=model_root,
                metrics_output_root=metrics_root,
            )

            metrics_path = metrics_root / "last_week" / "trend_metrics.json"
            self.assertTrue(metrics_path.exists())
            self.assertEqual(payload["model_name"], "last_week")
            self.assertEqual(payload["groups"]["test"]["ranking_groups"], 4)
            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(written["ranking"]["k_values"], [5, 10, 20])

    def test_run_trend_model_evaluation_rejects_missing_predictions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with self.assertRaisesRegex(FileNotFoundError, "预测文件不存在"):
                run_trend_model_evaluation(
                    "last_week",
                    model_output_root=tmp_path / "outputs" / "models",
                    metrics_output_root=tmp_path / "outputs" / "metrics",
                )
```

- [ ] **Step 2: 运行测试，确认按预期失败**

运行：

```sh
uv run python -m unittest tests.test_trend.TrendEvaluationTests -v
```

预期：失败，提示 `write_trend_metrics` 和 `run_trend_model_evaluation` 无法导入。

- [ ] **Step 3: 实现 writer 和 runner**

在 `src/fashion_trend/evaluation.py` 中靠近 public runner 函数的位置追加：

```python
def write_trend_metrics(payload: dict[str, object], output_path: Path) -> None:
    """确认 payload 是严格 JSON 后写出趋势评价指标。"""
    _validate_json_payload(payload)
    write_json(payload, output_path)


def run_trend_model_evaluation(
    model_name: str,
    model_output_root: Path = OUTPUT_MODELS_DIR,
    metrics_output_root: Path = OUTPUT_METRICS_DIR,
) -> dict[str, object]:
    """运行单个趋势模型的评价，并写出 trend_metrics.json。"""
    output_paths = derive_trend_metric_output_paths(
        model_name,
        model_output_root=model_output_root,
        metrics_output_root=metrics_output_root,
    )
    predictions = read_trend_model_predictions(output_paths["predictions"])
    payload = build_trend_metrics_payload(
        predictions,
        model_name=model_name,
        prediction_path=output_paths["predictions"],
        output_path=output_paths["metrics"],
    )
    write_trend_metrics(payload, output_paths["metrics"])
    return payload
```

- [ ] **Step 4: 运行评价测试**

运行：

```sh
uv run python -m unittest tests.test_trend.TrendEvaluationTests -v
```

预期：通过。

- [ ] **Step 5: 运行完整测试**

运行：

```sh
uv run python -m unittest discover -s tests -v
```

预期：通过。

- [ ] **Step 6: 如已授权 commit，提交本检查点**

只有明确授权 commit 时才运行：

```sh
git add src/fashion_trend/evaluation.py tests/test_trend.py
git commit -m "feat(trend): 写出趋势评价产物"
```

---

### Task 4: 添加薄评价 CLI

**文件：**
- 新建：`src/11_eval_trend_model.py`
- 修改：`tests/test_trend.py`

- [ ] **Step 1: 先写失败的 CLI 测试**

在 `TrendEvaluationTests` 中追加：

```python
    def test_eval_trend_model_main_preserves_argparse_usage_error_code(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")

        exit_code = eval_model.main([])

        self.assertEqual(exit_code, 2)

    def test_eval_trend_model_main_returns_error_for_missing_predictions(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")

        exit_code = eval_model.main(["--model", "missing_model"])

        self.assertEqual(exit_code, 1)

    def test_eval_trend_model_main_runs_evaluation_and_logs_summary(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")
        original_run_trend_model_evaluation = eval_model.run_trend_model_evaluation

        def fake_run_trend_model_evaluation(model_name: str) -> dict[str, object]:
            self.assertEqual(model_name, "last_week")
            return {
                "model_name": "last_week",
                "evaluated_splits": ["valid", "test"],
                "overall": {
                    "valid": {
                        "mae": 0.5,
                        "rmse": 0.7,
                        "spearman": 0.2,
                        "precision_at_k": {"10": 0.4},
                        "recall_at_k": {"10": 0.4},
                        "ndcg_at_k": {"10": 0.6},
                    },
                    "test": {
                        "mae": 0.6,
                        "rmse": 0.8,
                        "spearman": 0.3,
                        "precision_at_k": {"10": 0.5},
                        "recall_at_k": {"10": 0.5},
                        "ndcg_at_k": {"10": 0.7},
                    },
                },
                "groups": {
                    "valid": {"ranking_groups": 4},
                    "test": {"ranking_groups": 4},
                },
                "output_path": "outputs/metrics/last_week/trend_metrics.json",
            }

        try:
            eval_model.run_trend_model_evaluation = fake_run_trend_model_evaluation
            exit_code = eval_model.main(["--model", "last_week"])
        finally:
            eval_model.run_trend_model_evaluation = original_run_trend_model_evaluation

        self.assertEqual(exit_code, 0)
```

- [ ] **Step 2: 运行测试，确认按预期失败**

运行：

```sh
uv run python -m unittest tests.test_trend.TrendEvaluationTests -v
```

预期：失败，报 `ModuleNotFoundError: No module named '11_eval_trend_model'`。

- [ ] **Step 3: 新建 CLI**

创建 `src/11_eval_trend_model.py`：

```python
from __future__ import annotations

import argparse
from typing import Sequence

from fashion_trend import log
from fashion_trend.evaluation import run_trend_model_evaluation

LOG_SOURCE = "trend-model-eval"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析趋势模型评价入口参数。"""
    parser = argparse.ArgumentParser(description="评价趋势预测模型并写出指标。")
    parser.add_argument(
        "--model",
        required=True,
        help="需要评价的趋势模型名称。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """运行趋势模型评价 CLI，并返回稳定退出码。"""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        metrics = run_trend_model_evaluation(args.model)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"模型名称: {metrics['model_name']}", source=LOG_SOURCE)
    log.info(
        "评价 split: " + ", ".join(str(split) for split in metrics["evaluated_splits"]),
        source=LOG_SOURCE,
    )
    for split_name in metrics["evaluated_splits"]:
        split_metrics = metrics["overall"][split_name]
        split_groups = metrics["groups"][split_name]
        log.info(
            f"{split_name} 评价: "
            f"groups={split_groups['ranking_groups']:,}, "
            f"mae={_format_metric(split_metrics['mae'])}, "
            f"rmse={_format_metric(split_metrics['rmse'])}, "
            f"spearman={_format_metric(split_metrics['spearman'])}, "
            f"precision@10={_format_metric(split_metrics['precision_at_k']['10'])}, "
            f"recall@10={_format_metric(split_metrics['recall_at_k']['10'])}, "
            f"ndcg@10={_format_metric(split_metrics['ndcg_at_k']['10'])}",
            source=LOG_SOURCE,
        )
    log.info(f"评价输出文件: {metrics['output_path']}", source=LOG_SOURCE)
    return 0


def _format_metric(value: object) -> str:
    if value is None:
        return "null"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行 CLI 测试**

运行：

```sh
uv run python -m unittest tests.test_trend.TrendEvaluationTests -v
```

预期：通过。

- [ ] **Step 5: 编译新模块**

运行：

```sh
uv run python -m py_compile src/fashion_trend/evaluation.py src/11_eval_trend_model.py
```

预期：退出码为 0，无输出。

- [ ] **Step 6: 运行完整测试**

运行：

```sh
uv run python -m unittest discover -s tests -v
```

预期：通过。

- [ ] **Step 7: 如已授权 commit，提交本检查点**

只有明确授权 commit 时才运行：

```sh
git add src/11_eval_trend_model.py src/fashion_trend/evaluation.py tests/test_trend.py
git commit -m "feat(trend): 添加趋势评价入口"
```

---

### Task 5: 同步 README 和研究计划文档

**文件：**
- 修改：`README.md`
- 修改：`docs/gpt-research/implementation-plan.md`

- [ ] **Step 1: 更新 README 阶段表**

在 `README.md` 顶部阶段表中，将：

```markdown
| Last Week baseline | 已实现 | `outputs/models/last_week/predictions.csv`、`params.json`、`metadata.json` |
| 推荐评价 | 尚未实现 | 后续推荐结果 |
```

替换为：

```markdown
| Last Week baseline | 已实现 | `outputs/models/last_week/predictions.csv`、`params.json`、`metadata.json` |
| 趋势评价 | 已实现 | `outputs/metrics/last_week/trend_metrics.json` |
| 推荐评价 | 尚未实现 | 后续推荐结果 |
```

- [ ] **Step 2: 更新 README 流水线命令**

在 README 的“当前已实现流水线按下面顺序运行”命令块中，将：

```sh
uv run python src/10_train_trend_model.py --model last_week
```

替换为：

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model last_week
```

- [ ] **Step 3: 增加 README 趋势评价小节**

在 `### 9. last_week baseline` 后、`## 后续阶段` 前插入：

````markdown
### 10. 趋势评价

趋势评价通过独立入口运行，读取已经生成的趋势模型预测表：

```sh
outputs/models/last_week/predictions.csv
```

评价结果写入：

```sh
outputs/metrics/last_week/trend_metrics.json
```

第一版趋势评价只评价 `valid` 和 `test` split，不把 `train` 作为正式指标。排序目标与训练目标保持一致：

```text
target_growth vs pred_target_growth
```

指标包含：

```text
MAE
RMSE
Spearman
Precision@5/10/20
Recall@5/10/20
NDCG@5/10/20
```

排序指标按 `split + week_id + attr_type` 逐组计算，再汇总到 overall 和 by_attr_type，便于观察不同属性类型的趋势预测质量。

运行命令：

```sh
uv run python src/11_eval_trend_model.py --model last_week
```
````

- [ ] **Step 4: 更新 README 后续阶段说明**

在 `README.md` 中，将：

```markdown
趋势模型训练框架已经落地到 `last_week` baseline，README 继续按计划记录后续边界：
```

替换为：

```markdown
趋势模型训练与评价框架已经落地到 `last_week` baseline，README 继续按计划记录后续边界：
```

保留现有“趋势模型扩展”和“推荐模块”两行。

- [ ] **Step 5: 更新 README 验证覆盖说明**

在 README “已覆盖的核心逻辑包括”列表中，在：

```markdown
- `last_week` baseline 预测公式、预测表校验、通用训练 runner metadata、artifact 和写出顺序校验。
```

后面增加：

```markdown
- 趋势评价的预测读取、输入校验、分组指标、JSON payload、写出边界和 CLI 行为校验。
```

- [ ] **Step 6: 轻量同步 implementation plan**

在 `docs/gpt-research/implementation-plan.md` 中找到“第 10 步：趋势预测评价”小节，将：

````markdown
写：

```text
src/10_eval_trend.py
```

输出：

```text
trend_eval_results.csv
```
````

替换为：

````markdown
当前实现入口：

```text
src/11_eval_trend_model.py
```

当前标准产物：

```text
outputs/metrics/<model>/trend_metrics.json
```
````

同时找到阶段产出表中的趋势评价行，将：

```markdown
| 趋势评价        | `10_eval_trend.py`                  | `trend_eval_results.csv`                                                                                   |
```

替换为：

```markdown
| 趋势评价        | `11_eval_trend_model.py`            | `outputs/metrics/<model>/trend_metrics.json`                                                               |
```

- [ ] **Step 7: 运行文档漂移检查**

运行：

```sh
rg -n "10_eval_trend|trend_eval_results|11_eval_trend_model|trend_metrics.json" README.md docs/gpt-research/implementation-plan.md
```

预期：

- 没有 `10_eval_trend.py` 命中。
- 没有 `trend_eval_results.csv` 命中。
- `11_eval_trend_model.py` 在 README 和 implementation plan 中出现。
- `trend_metrics.json` 在 README 和 implementation plan 中出现。

- [ ] **Step 8: 文档更新后运行测试**

运行：

```sh
uv run python -m unittest discover -s tests -v
```

预期：通过。

- [ ] **Step 9: 如已授权 commit，提交本检查点**

只有明确授权 commit 时才运行：

```sh
git add README.md docs/gpt-research/implementation-plan.md
git commit -m "docs(trend): 同步趋势评价命令和产物"
```

---

### Task 6: 使用真实 `last_week` 预测产物验证

**文件：**
- 读取：`outputs/models/last_week/predictions.csv`
- 创建或更新被 ignore 的产物：`outputs/metrics/last_week/trend_metrics.json`

- [ ] **Step 1: 确认训练预测文件存在**

运行：

```sh
ls -l outputs/models/last_week/predictions.csv
```

预期：文件存在。当前已知本地产物约 10 MB，且具有标准预测表 header。

- [ ] **Step 2: 运行真实评价命令**

运行：

```sh
uv run python src/11_eval_trend_model.py --model last_week
```

预期：

- 退出码为 0。
- 日志包含 `模型名称: last_week`。
- 日志包含 `valid 评价`。
- 日志包含 `test 评价`。
- 日志包含 `评价输出文件:`。

- [ ] **Step 3: 检查实际 JSON 结构**

运行：

```sh
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/metrics/last_week/trend_metrics.json")
payload = json.loads(path.read_text(encoding="utf-8"))

print("model_name", payload["model_name"])
print("evaluated_splits", payload["evaluated_splits"])
print("overall_splits", sorted(payload["overall"]))
print("by_attr_type_splits", sorted(payload["by_attr_type"]))
print("valid_groups", payload["groups"]["valid"])
print("test_groups", payload["groups"]["test"])
print("test_metric_keys", sorted(payload["overall"]["test"]))
print("test_precision_keys", sorted(payload["overall"]["test"]["precision_at_k"]))
print("test_recall_keys", sorted(payload["overall"]["test"]["recall_at_k"]))
print("test_ndcg_keys", sorted(payload["overall"]["test"]["ndcg_at_k"]))
PY
```

预期输出形状：

```text
model_name last_week
evaluated_splits ['valid', 'test']
overall_splits ['test', 'valid']
by_attr_type_splits ['test', 'valid']
valid_groups {'rows': 4736, 'weeks': 8, 'attr_types': 10, 'ranking_groups': 80}
test_groups {'rows': 4736, 'weeks': 8, 'attr_types': 10, 'ranking_groups': 80}
test_metric_keys ['mae', 'ndcg_at_k', 'precision_at_k', 'recall_at_k', 'rmse', 'spearman']
test_precision_keys ['10', '20', '5']
test_recall_keys ['10', '20', '5']
test_ndcg_keys ['10', '20', '5']
```

- [ ] **Step 4: 确认 strict JSON 没有非标准数值**

运行：

```sh
uv run python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("outputs/metrics/last_week/trend_metrics.json").read_text(encoding="utf-8"))
json.dumps(payload, allow_nan=False)
print("strict-json-ok")
PY
```

预期：

```text
strict-json-ok
```

- [ ] **Step 5: 确认评价没有改动模型产物**

运行：

```sh
git status --short
```

预期：

- 如果尚未提交实现，能看到本次实现涉及的源码和文档文件。
- 不应看到 tracked 的 `outputs/models/last_week/` 文件被修改。
- `outputs/metrics/last_week/trend_metrics.json` 可能不会出现在 git status 中，因为 `outputs/` 被 ignore。

- [ ] **Step 6: 运行最终编译和测试验证**

运行：

```sh
uv run python -m py_compile src/fashion_trend/evaluation.py src/11_eval_trend_model.py
uv run python -m unittest discover -s tests -v
```

预期：

- 编译命令退出码为 0，无输出。
- 完整测试通过。

- [ ] **Step 7: 如已授权 commit，提交最终检查点**

只有明确授权 commit，且已检查 diff 后才运行：

```sh
git diff --stat
git diff -- src/fashion_trend/evaluation.py src/11_eval_trend_model.py tests/test_trend.py README.md docs/gpt-research/implementation-plan.md
git add src/fashion_trend/evaluation.py src/11_eval_trend_model.py tests/test_trend.py README.md docs/gpt-research/implementation-plan.md
git commit -m "feat(trend): 打通趋势评价闭环"
```

---

## 自审清单

Spec 覆盖：

- 独立 runner 和薄 CLI：Task 3 和 Task 4。
- 输入 `outputs/models/<model>/predictions.csv`：Task 1、Task 3、Task 6。
- 输出 `outputs/metrics/<model>/trend_metrics.json`：Task 1、Task 3、Task 6。
- 只评价 valid/test：Task 1 和 Task 2。
- MAE、RMSE、Spearman、Precision/Recall/NDCG at 5/10/20：Task 2 和 Task 6。
- 按 `split + week_id + attr_type` 分组：Task 2 和 Task 6。
- strict JSON 且无 NaN/Infinity：Task 2、Task 3、Task 6。
- 不修改模型产物：Task 3 和 Task 6。
- README 和 implementation plan 同步：Task 5。

模糊项扫描：

- 计划中不保留未解决的模糊说明。
- 每个代码修改任务都包含具体代码片段、命令和预期结果。

类型和命名一致性：

- 公共模块：`fashion_trend.evaluation`。
- CLI 文件：`src/11_eval_trend_model.py`。
- Runner：`run_trend_model_evaluation()`。
- Metrics writer：`write_trend_metrics()`。
- JSON 产物：`trend_metrics.json`。
- 指标 key：`mae`、`rmse`、`spearman`、`precision_at_k`、`recall_at_k`、`ndcg_at_k`。
