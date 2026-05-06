# 趋势 Last Week Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first trend baseline pipeline: split `trend_model_samples.parquet` into train/valid/test files, run the `last_week` baseline, and write model outputs under `outputs/models/last_week/`.

**Architecture:** Keep reusable DataFrame logic in `src/fashion_trend/trend.py` and the baseline formula in `src/fashion_trend/models/baseline_last_week.py`. `src/09_split_trend_model_samples.py` owns time splitting and writes processed feature splits; `src/10_train_trend_baseline.py` only consumes those split files and writes model outputs. Config owns paths and split widths so model scripts do not hard-code dataset layout.

**Tech Stack:** Python 3.10-3.12, `pandas`, `numpy`, `pyarrow`, standard library `argparse` / `json` / `pathlib` / `unittest` / `tempfile`, existing `fashion_trend.config.PATH`, `fashion_trend.trend`, and `fashion_trend.log`.

---

## File Structure

- Modify: `src/fashion_trend/config.py`
  - Add output directories, train/valid/test split widths, split feature paths, and last_week output paths.
- Modify: `src/fashion_trend/trend.py`
  - Add split constants, split construction/validation helpers, JSON writer, split parquet reader, and baseline prediction validation.
- Create: `src/09_split_trend_model_samples.py`
  - Top-level script that reads full trend samples, applies config-owned time split, and writes train/valid/test parquet plus split metadata JSON.
- Create: `src/fashion_trend/models/__init__.py`
  - Package marker for baseline model modules.
- Create: `src/fashion_trend/models/baseline_last_week.py`
  - DataFrame-level `last_week` prediction logic with no path or CLI dependencies.
- Create: `src/10_train_trend_baseline.py`
  - Top-level CLI that reads split parquet files, runs `last_week`, and writes predictions, params, and metadata.
- Modify: `tests/test_trend.py`
  - Add tests for split logic, split metadata, last_week predictions, prediction validation, and JSON writing.
- Modify: `README.md`
  - Add commands and output descriptions for split and baseline stages.

## Task 1: Config And Split Core

**Files:**
- Modify: `src/fashion_trend/config.py`
- Modify: `src/fashion_trend/trend.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Add failing imports and split tests**

Modify the import block in `tests/test_trend.py` so it includes the new names:

```python
from fashion_trend.trend import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
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
    validate_trend_model_samples,
    validate_trend_model_split_frames,
    write_json,
    write_trend_csv,
    write_trend_parquet,
)
```

Add this helper near existing sample helpers in `tests/test_trend.py`:

```python
def sample_trend_model_samples_for_split() -> pd.DataFrame:
    rows = []
    for week_id in range(4, 24):
        for attr_id, attr_type, attr_value in [
            ("colour_group_name::Black", "colour_group_name", "Black"),
            ("colour_group_name::White", "colour_group_name", "White"),
        ]:
            share_t = 0.60 if attr_value == "Black" else 0.40
            rows.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": attr_type,
                    "attr_value": attr_value,
                    "heat_t": 10 + week_id,
                    "share_t": share_t,
                    "log_heat_t": math.log1p(10 + week_id),
                    "rank_in_type_t": 1 if attr_value == "Black" else 2,
                    "heat_lag_1": 9 + week_id,
                    "heat_lag_2": 8 + week_id,
                    "heat_lag_3": 7 + week_id,
                    "heat_lag_4": 6 + week_id,
                    "share_lag_1": share_t - 0.01,
                    "share_lag_2": share_t - 0.02,
                    "share_lag_3": share_t - 0.03,
                    "share_lag_4": share_t - 0.04,
                    "growth_lag_1": 0.10 if attr_value == "Black" else -0.05,
                    "growth_lag_2": 0.05 if attr_value == "Black" else -0.02,
                    "acc_lag_1": 0.05 if attr_value == "Black" else -0.03,
                    "heat_ma_4": 8.5 + week_id,
                    "share_ma_4": share_t - 0.015,
                    "share_std_4": 0.01,
                    "share_max_4": share_t,
                    "share_min_4": share_t - 0.04,
                    "article_count": 10,
                    "is_core_attr": 1,
                    "parent_count": 1,
                    "child_count": 0,
                    "degree": 1,
                    "history_total_heat_t": 100 + week_id,
                    "history_active_weeks_t": week_id,
                    "is_trend_eligible_t": True,
                    "week_index": week_id,
                    "week_mod_52": week_id % 52,
                    "target_growth": 0.12 if attr_value == "Black" else -0.03,
                    "target_log_heat_t1": math.log1p(11 + week_id),
                    "target_rank_in_type_t1": 1 if attr_value == "Black" else 2,
                }
            )
    return pd.DataFrame(rows).loc[:, list(TREND_MODEL_SAMPLE_COLUMNS)]
```

Add these tests near the existing trend sample tests:

```python
class TrendModelSplitFrameTests(unittest.TestCase):
    def test_build_trend_model_split_frames_uses_time_boundaries(self) -> None:
        samples = sample_trend_model_samples_for_split()

        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )

        self.assertEqual(set(split_frames), {"train", "valid", "test"})
        self.assertEqual(split_frames["train"]["week_id"].min(), 4)
        self.assertEqual(split_frames["train"]["week_id"].max(), 15)
        self.assertEqual(split_frames["valid"]["week_id"].min(), 16)
        self.assertEqual(split_frames["valid"]["week_id"].max(), 19)
        self.assertEqual(split_frames["test"]["week_id"].min(), 20)
        self.assertEqual(split_frames["test"]["week_id"].max(), 23)
        self.assertEqual(set(split_frames["train"]["split"]), {"train"})
        self.assertEqual(set(split_frames["valid"]["split"]), {"valid"})
        self.assertEqual(set(split_frames["test"]["split"]), {"test"})

    def test_build_trend_model_split_frames_rejects_too_few_weeks(self) -> None:
        samples = sample_trend_model_samples_for_split()
        samples = samples[samples["week_id"] < 10].copy()

        with self.assertRaisesRegex(ValueError, "样本周数不足"):
            build_trend_model_split_frames(samples, valid_weeks=4, test_weeks=4)

    def test_build_trend_model_split_metadata_reports_ranges(self) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )

        metadata = build_trend_model_split_metadata(
            split_frames,
            input_path=Path("data/processed/features/trend_model_samples.parquet"),
            output_paths={
                "train": Path("data/processed/features/trend_model_samples_train.parquet"),
                "valid": Path("data/processed/features/trend_model_samples_valid.parquet"),
                "test": Path("data/processed/features/trend_model_samples_test.parquet"),
            },
            valid_weeks=4,
            test_weeks=4,
        )

        self.assertEqual(metadata["split_strategy"], "time")
        self.assertEqual(metadata["valid_weeks"], 4)
        self.assertEqual(metadata["test_weeks"], 4)
        self.assertEqual(metadata["splits"]["train"]["week_min"], 4)
        self.assertEqual(metadata["splits"]["train"]["week_max"], 15)
        self.assertEqual(metadata["splits"]["train"]["rows"], 24)
        self.assertEqual(metadata["splits"]["valid"]["week_min"], 16)
        self.assertEqual(metadata["splits"]["test"]["week_max"], 23)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
uv run python -m unittest tests.test_trend.TrendModelSplitFrameTests -v
```

Expected: fail with `ImportError` for `TREND_MODEL_SPLIT_COLUMNS` or `build_trend_model_split_frames`.

- [ ] **Step 3: Add config paths and split widths**

Modify `src/fashion_trend/config.py`:

```python
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_MODELS_DIR = OUTPUT_DIR / "models"
OUTPUT_METRICS_DIR = OUTPUT_DIR / "metrics"
OUTPUT_FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

TREND_SPLIT_VALID_WEEKS = 8
TREND_SPLIT_TEST_WEEKS = 8
```

Update `PATH` so the processed feature and output sections contain these keys:

```python
    "features_trend_model_samples": FEATURES_DIR / "trend_model_samples.parquet",
    "features_trend_model_samples_train": FEATURES_DIR
    / "trend_model_samples_train.parquet",
    "features_trend_model_samples_valid": FEATURES_DIR
    / "trend_model_samples_valid.parquet",
    "features_trend_model_samples_test": FEATURES_DIR
    / "trend_model_samples_test.parquet",
    "features_trend_model_samples_split_metadata": FEATURES_DIR
    / "trend_model_samples_split_metadata.json",
    # ---------------- Model outputs ----------------
    "output_model_last_week_dir": OUTPUT_MODELS_DIR / "last_week",
    "output_model_last_week_predictions": OUTPUT_MODELS_DIR
    / "last_week"
    / "predictions.csv",
    "output_model_last_week_params": OUTPUT_MODELS_DIR / "last_week" / "params.json",
    "output_model_last_week_metadata": OUTPUT_MODELS_DIR
    / "last_week"
    / "metadata.json",
```

- [ ] **Step 4: Add split helpers to trend.py**

Modify imports at the top of `src/fashion_trend/trend.py`:

```python
import json
```

Add constants below `TREND_MODEL_SAMPLE_COLUMNS`:

```python
TREND_MODEL_SPLIT_VALUES: tuple[str, ...] = ("train", "valid", "test")

TREND_MODEL_SPLIT_COLUMNS: tuple[str, ...] = (
    "split",
    *TREND_MODEL_SAMPLE_COLUMNS,
)
```

Add these functions above `remove_file_if_exists`:

```python
def build_trend_model_split_frames(
    trend_model_samples: pd.DataFrame,
    valid_weeks: int,
    test_weeks: int,
) -> dict[str, pd.DataFrame]:
    validate_trend_model_samples(trend_model_samples)
    if valid_weeks <= 0:
        raise ValueError("valid_weeks 必须为正整数。")
    if test_weeks <= 0:
        raise ValueError("test_weeks 必须为正整数。")

    week_ids = sorted(trend_model_samples["week_id"].unique().tolist())
    required_week_count = valid_weeks + test_weeks + 1
    if len(week_ids) < required_week_count:
        raise ValueError(
            "样本周数不足，无法生成非空 train/valid/test: "
            f"当前 {len(week_ids)} 周，valid_weeks={valid_weeks}, "
            f"test_weeks={test_weeks}。"
        )

    max_sample_week = max(week_ids)
    test_start_week = max_sample_week - test_weeks + 1
    valid_start_week = test_start_week - valid_weeks

    split_masks = {
        "train": trend_model_samples["week_id"] < valid_start_week,
        "valid": (trend_model_samples["week_id"] >= valid_start_week)
        & (trend_model_samples["week_id"] < test_start_week),
        "test": trend_model_samples["week_id"] >= test_start_week,
    }
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = trend_model_samples.loc[split_masks[split_name]].copy()
        split_frame.insert(0, "split", split_name)
        split_frame = split_frame.loc[:, list(TREND_MODEL_SPLIT_COLUMNS)].sort_values(
            ["week_id", "attr_type", "attr_id"],
            ignore_index=True,
        )
        split_frames[split_name] = split_frame

    validate_trend_model_split_frames(split_frames, trend_model_samples)
    return split_frames


def validate_trend_model_split_frames(
    split_frames: dict[str, pd.DataFrame],
    original_samples: pd.DataFrame | None = None,
) -> None:
    missing_splits = set(TREND_MODEL_SPLIT_VALUES) - set(split_frames)
    if missing_splits:
        raise ValueError(f"趋势样本切分缺少 split: {sorted(missing_splits)}")

    combined_parts: list[pd.DataFrame] = []
    previous_max_week: int | None = None
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = split_frames[split_name]
        validate_required_columns(
            split_frame.columns.tolist(),
            TREND_MODEL_SPLIT_COLUMNS,
            source_name=f"{split_name} 趋势样本",
        )
        validate_no_missing_values(
            split_frame,
            TREND_MODEL_SPLIT_COLUMNS,
            source_name=f"{split_name} 趋势样本",
        )
        if split_frame.empty:
            raise ValueError(f"{split_name} 趋势样本为空。")
        if set(split_frame["split"]) != {split_name}:
            raise ValueError(f"{split_name} 趋势样本 split 字段不一致。")
        validate_unique_key(
            split_frame,
            ["week_id", "attr_id"],
            source_name=f"{split_name} 趋势样本",
        )
        min_week = int(split_frame["week_id"].min())
        max_week = int(split_frame["week_id"].max())
        if previous_max_week is not None and min_week <= previous_max_week:
            raise ValueError("趋势样本 split 周范围必须按时间递增且互不重叠。")
        previous_max_week = max_week
        combined_parts.append(split_frame.drop(columns=["split"]))

    if original_samples is not None:
        combined = pd.concat(combined_parts, ignore_index=True)
        combined_keys = combined.loc[:, ["week_id", "attr_id"]].sort_values(
            ["week_id", "attr_id"],
            ignore_index=True,
        )
        original_keys = original_samples.loc[:, ["week_id", "attr_id"]].sort_values(
            ["week_id", "attr_id"],
            ignore_index=True,
        )
        if not combined_keys.equals(original_keys):
            raise ValueError("趋势样本 split 合并后无法覆盖原始样本全集。")


def build_trend_model_split_metadata(
    split_frames: dict[str, pd.DataFrame],
    input_path: Path,
    output_paths: dict[str, Path],
    valid_weeks: int,
    test_weeks: int,
) -> dict[str, object]:
    validate_trend_model_split_frames(split_frames)
    split_metadata: dict[str, dict[str, object]] = {}
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = split_frames[split_name]
        split_metadata[split_name] = {
            "path": str(output_paths[split_name]),
            "rows": int(len(split_frame)),
            "weeks": int(split_frame["week_id"].nunique()),
            "attributes": int(split_frame["attr_id"].nunique()),
            "week_min": int(split_frame["week_id"].min()),
            "week_max": int(split_frame["week_id"].max()),
        }
    return {
        "split_strategy": "time",
        "valid_weeks": int(valid_weeks),
        "test_weeks": int(test_weeks),
        "input_path": str(input_path),
        "splits": split_metadata,
    }


def read_trend_model_split(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"趋势样本 split 不存在: {input_path}")
    dataframe = pd.read_parquet(input_path)
    validate_required_columns(
        dataframe.columns.tolist(),
        TREND_MODEL_SPLIT_COLUMNS,
        source_name=f"趋势样本 split: {input_path}",
    )
    return dataframe.loc[:, list(TREND_MODEL_SPLIT_COLUMNS)].copy()


def write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        tmp_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_output_path.replace(output_path)
    except Exception:
        remove_file_if_exists(tmp_output_path)
        raise
```

- [ ] **Step 5: Run split tests**

Run:

```sh
uv run python -m unittest tests.test_trend.TrendModelSplitFrameTests -v
```

Expected: all tests in `TrendModelSplitFrameTests` pass.

- [ ] **Step 6: Commit split core**

Run:

```sh
git add src/fashion_trend/config.py src/fashion_trend/trend.py tests/test_trend.py
git commit -m "feat(trend): 增加趋势样本时间切分"
```

## Task 2: Split CLI

**Files:**
- Create: `src/09_split_trend_model_samples.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Write failing script smoke test**

Add this test class to `tests/test_trend.py`:

```python
class TrendModelSplitWriteTests(unittest.TestCase):
    def test_write_json_creates_parent_and_writes_sorted_keys(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "metadata.json"

            write_json({"b": 2, "a": 1}, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{\n  "a": 1,\n  "b": 2\n}\n',
            )
```

- [ ] **Step 2: Run write test**

Run:

```sh
uv run python -m unittest tests.test_trend.TrendModelSplitWriteTests -v
```

Expected: pass if Task 1 added `write_json`; otherwise fail with `ImportError`.

- [ ] **Step 3: Create split script**

Create `src/09_split_trend_model_samples.py`:

```python
from __future__ import annotations

from fashion_trend import log
from fashion_trend.config import (
    PATH,
    TREND_SPLIT_TEST_WEEKS,
    TREND_SPLIT_VALID_WEEKS,
)
from fashion_trend.trend import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    validate_trend_model_split_frames,
    write_json,
    write_trend_parquet,
)

LOG_SOURCE = "trend-model-split"


def split_trend_model_samples() -> dict[str, object]:
    input_path = PATH["features_trend_model_samples"]
    log.info(f"输入趋势样本表: {input_path}", source=LOG_SOURCE)
    if not input_path.exists():
        raise FileNotFoundError(f"趋势样本表不存在: {input_path}")

    import pandas as pd

    trend_model_samples = pd.read_parquet(input_path)
    split_frames = build_trend_model_split_frames(
        trend_model_samples,
        valid_weeks=TREND_SPLIT_VALID_WEEKS,
        test_weeks=TREND_SPLIT_TEST_WEEKS,
    )
    validate_trend_model_split_frames(split_frames, trend_model_samples)

    output_paths = {
        "train": PATH["features_trend_model_samples_train"],
        "valid": PATH["features_trend_model_samples_valid"],
        "test": PATH["features_trend_model_samples_test"],
    }
    for split_name, split_frame in split_frames.items():
        write_trend_parquet(split_frame, output_paths[split_name])

    metadata = build_trend_model_split_metadata(
        split_frames,
        input_path=input_path,
        output_paths=output_paths,
        valid_weeks=TREND_SPLIT_VALID_WEEKS,
        test_weeks=TREND_SPLIT_TEST_WEEKS,
    )
    write_json(metadata, PATH["features_trend_model_samples_split_metadata"])
    return metadata


def main() -> int:
    try:
        metadata = split_trend_model_samples()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    for split_name in ("train", "valid", "test"):
        split_stats = metadata["splits"][split_name]
        log.info(
            f"{split_name} 样本: rows={split_stats['rows']:,}, "
            f"weeks={split_stats['weeks']:,}, "
            f"week_range={split_stats['week_min']}..{split_stats['week_max']}",
            source=LOG_SOURCE,
        )
        log.info(f"{split_name} 输出文件: {split_stats['path']}", source=LOG_SOURCE)
    log.info(
        f"切分元数据: {PATH['features_trend_model_samples_split_metadata']}",
        source=LOG_SOURCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Compile split script**

Run:

```sh
uv run python -m py_compile src/09_split_trend_model_samples.py
```

Expected: command exits with status 0 and prints no errors.

- [ ] **Step 5: Run split-related tests**

Run:

```sh
uv run python -m unittest tests.test_trend.TrendModelSplitFrameTests tests.test_trend.TrendModelSplitWriteTests -v
```

Expected: all listed tests pass.

- [ ] **Step 6: Commit split script**

Run:

```sh
git add src/09_split_trend_model_samples.py tests/test_trend.py
git commit -m "feat(trend): 添加趋势样本切分脚本"
```

## Task 3: Last Week Model Module

**Files:**
- Create: `src/fashion_trend/models/__init__.py`
- Create: `src/fashion_trend/models/baseline_last_week.py`
- Modify: `src/fashion_trend/trend.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Add failing baseline imports and tests**

Add imports to `tests/test_trend.py`:

```python
from fashion_trend.models.baseline_last_week import (
    LAST_WEEK_MODEL_NAME,
    LAST_WEEK_PARAMS,
    predict_last_week,
)
```

Extend the `fashion_trend.trend` import with:

```python
    TREND_BASELINE_PREDICTION_COLUMNS,
    validate_trend_baseline_predictions,
```

Add this test class:

```python
class LastWeekBaselineTests(unittest.TestCase):
    def test_predict_last_week_uses_growth_lag_1(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)

        predictions = predict_last_week(samples)

        self.assertEqual(predictions.columns.tolist(), list(TREND_BASELINE_PREDICTION_COLUMNS))
        self.assertEqual(set(predictions["model_name"]), {LAST_WEEK_MODEL_NAME})
        pd.testing.assert_series_equal(
            predictions["pred_target_growth"],
            samples.sort_values(["week_id", "attr_type", "attr_id"], ignore_index=True)[
                "growth_lag_1"
            ],
            check_names=False,
        )
        expected_share = (
            predictions["pred_target_growth"].map(math.exp)
            * (predictions["share_t"] + LAST_WEEK_PARAMS["epsilon"])
            - LAST_WEEK_PARAMS["epsilon"]
        )
        pd.testing.assert_series_equal(
            predictions["pred_share_t1"],
            expected_share,
            check_names=False,
        )

    def test_predict_last_week_rejects_missing_split(self) -> None:
        samples = sample_trend_model_samples_for_split()

        with self.assertRaisesRegex(ValueError, "缺少必需列"):
            predict_last_week(samples)

    def test_validate_trend_baseline_predictions_rejects_changed_split(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        predictions = predict_last_week(samples)
        predictions.loc[0, "split"] = "test"

        with self.assertRaisesRegex(ValueError, "baseline 预测 split 与输入不一致"):
            validate_trend_baseline_predictions(predictions, samples)
```

- [ ] **Step 2: Run baseline tests to verify they fail**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: fail with `ModuleNotFoundError: No module named 'fashion_trend.models'`.

- [ ] **Step 3: Add prediction constants and validator**

Add to `src/fashion_trend/trend.py` below split constants:

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

Add this validator above `remove_file_if_exists`:

```python
def validate_trend_baseline_predictions(
    predictions: pd.DataFrame,
    split_samples: pd.DataFrame,
) -> None:
    validate_required_columns(
        predictions.columns.tolist(),
        TREND_BASELINE_PREDICTION_COLUMNS,
        source_name="趋势 baseline 预测表",
    )
    validate_no_missing_values(
        predictions,
        TREND_BASELINE_PREDICTION_COLUMNS,
        source_name="趋势 baseline 预测表",
    )
    validate_unique_key(
        predictions,
        ["week_id", "attr_id", "model_name"],
        source_name="趋势 baseline 预测表",
    )
    if not set(predictions["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("趋势 baseline 预测表存在非法 split。")
    sorted_predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    sorted_samples = split_samples.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    prediction_split = sorted_predictions.loc[:, ["week_id", "attr_id", "split"]]
    sample_split = sorted_samples.loc[:, ["week_id", "attr_id", "split"]]
    if not prediction_split.equals(sample_split):
        raise ValueError("趋势 baseline 预测 split 与输入不一致。")

    numeric_values = sorted_predictions.drop(
        columns=["attr_id", "attr_type", "attr_value", "model_name", "split"]
    )
    try:
        finite_numeric_values = numeric_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("趋势 baseline 预测表无法校验数值字段。") from exc
    if not np.isfinite(finite_numeric_values).all():
        raise ValueError("趋势 baseline 预测表存在非有限数值。")
```

- [ ] **Step 4: Create last_week model module**

Create `src/fashion_trend/models/__init__.py`:

```python
"""Trend model implementations."""
```

Create `src/fashion_trend/models/baseline_last_week.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.trend import (
    TREND_BASELINE_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
    validate_required_columns,
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
    validate_required_columns(
        split_samples.columns.tolist(),
        LAST_WEEK_REQUIRED_COLUMNS,
        source_name="last_week baseline 输入样本",
    )
    if not set(split_samples["split"]).issubset(set(TREND_MODEL_SPLIT_VALUES)):
        raise ValueError("last_week baseline 输入样本存在非法 split。")

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
    predictions = predictions.loc[:, list(TREND_BASELINE_PREDICTION_COLUMNS)]
    predictions = predictions.sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return predictions
```

- [ ] **Step 5: Run baseline tests**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests -v
```

Expected: all tests in `LastWeekBaselineTests` pass.

- [ ] **Step 6: Compile model module**

Run:

```sh
uv run python -m py_compile src/fashion_trend/models/__init__.py src/fashion_trend/models/baseline_last_week.py src/fashion_trend/trend.py
```

Expected: command exits with status 0 and prints no errors.

- [ ] **Step 7: Commit last_week model**

Run:

```sh
git add src/fashion_trend/models/__init__.py src/fashion_trend/models/baseline_last_week.py src/fashion_trend/trend.py tests/test_trend.py
git commit -m "feat(trend): 添加 last week baseline 模型"
```

## Task 4: Baseline CLI And Outputs

**Files:**
- Create: `src/10_train_trend_baseline.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Add metadata helper test**

Add this test to `LastWeekBaselineTests`:

```python
    def test_last_week_params_are_stable(self) -> None:
        self.assertEqual(
            LAST_WEEK_PARAMS,
            {
                "model_name": "last_week",
                "formula": "pred_target_growth = growth_lag_1",
                "derived_formula": (
                    "pred_share_t1 = exp(pred_target_growth) * "
                    "(share_t + epsilon) - epsilon"
                ),
                "epsilon": 1e-6,
            },
        )
```

- [ ] **Step 2: Run metadata helper test**

Run:

```sh
uv run python -m unittest tests.test_trend.LastWeekBaselineTests.test_last_week_params_are_stable -v
```

Expected: pass after Task 3.

- [ ] **Step 3: Create baseline CLI**

Create `src/10_train_trend_baseline.py`:

```python
from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from fashion_trend import log
from fashion_trend.config import PATH
from fashion_trend.models.baseline_last_week import (
    LAST_WEEK_MODEL_NAME,
    LAST_WEEK_PARAMS,
    predict_last_week,
)
from fashion_trend.trend import (
    TREND_MODEL_SPLIT_VALUES,
    read_trend_model_split,
    validate_trend_baseline_predictions,
    write_json,
    write_trend_csv,
)

LOG_SOURCE = "trend-baseline"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练趋势预测 baseline。")
    parser.add_argument(
        "--model",
        choices=[LAST_WEEK_MODEL_NAME],
        required=True,
        help="要运行的 baseline 模型名。",
    )
    return parser.parse_args(argv)


def build_prediction_metadata(predictions: pd.DataFrame) -> dict[str, object]:
    split_metadata: dict[str, dict[str, int]] = {}
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = predictions[predictions["split"] == split_name]
        split_metadata[split_name] = {
            "rows": int(len(split_frame)),
            "weeks": int(split_frame["week_id"].nunique()),
            "attributes": int(split_frame["attr_id"].nunique()),
            "week_min": int(split_frame["week_id"].min()),
            "week_max": int(split_frame["week_id"].max()),
        }
    return {
        "model_name": LAST_WEEK_MODEL_NAME,
        "input_paths": {
            "train": str(PATH["features_trend_model_samples_train"]),
            "valid": str(PATH["features_trend_model_samples_valid"]),
            "test": str(PATH["features_trend_model_samples_test"]),
        },
        "prediction_path": str(PATH["output_model_last_week_predictions"]),
        "rows": int(len(predictions)),
        "weeks": int(predictions["week_id"].nunique()),
        "attributes": int(predictions["attr_id"].nunique()),
        "splits": split_metadata,
    }


def train_trend_baseline(model_name: str) -> dict[str, object]:
    if model_name != LAST_WEEK_MODEL_NAME:
        raise ValueError(f"不支持的 baseline 模型: {model_name}")

    split_samples = pd.concat(
        [
            read_trend_model_split(PATH["features_trend_model_samples_train"]),
            read_trend_model_split(PATH["features_trend_model_samples_valid"]),
            read_trend_model_split(PATH["features_trend_model_samples_test"]),
        ],
        ignore_index=True,
    )
    predictions = predict_last_week(split_samples)
    validate_trend_baseline_predictions(predictions, split_samples)

    write_trend_csv(predictions, PATH["output_model_last_week_predictions"])
    write_json(LAST_WEEK_PARAMS, PATH["output_model_last_week_params"])
    metadata = build_prediction_metadata(predictions)
    write_json(metadata, PATH["output_model_last_week_metadata"])
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = train_trend_baseline(args.model)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"baseline 模型: {metadata['model_name']}", source=LOG_SOURCE)
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_stats = metadata["splits"][split_name]
        log.info(
            f"{split_name} 预测: rows={split_stats['rows']:,}, "
            f"weeks={split_stats['weeks']:,}, "
            f"week_range={split_stats['week_min']}..{split_stats['week_max']}",
            source=LOG_SOURCE,
        )
    log.info(f"预测输出: {PATH['output_model_last_week_predictions']}", source=LOG_SOURCE)
    log.info(f"参数输出: {PATH['output_model_last_week_params']}", source=LOG_SOURCE)
    log.info(f"元数据输出: {PATH['output_model_last_week_metadata']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Compile baseline CLI**

Run:

```sh
uv run python -m py_compile src/10_train_trend_baseline.py
```

Expected: command exits with status 0 and prints no errors.

- [ ] **Step 5: Run all trend tests**

Run:

```sh
uv run python -m unittest tests.test_trend -v
```

Expected: all `tests.test_trend` tests pass.

- [ ] **Step 6: Commit baseline CLI**

Run:

```sh
git add src/10_train_trend_baseline.py tests/test_trend.py
git commit -m "feat(trend): 添加 baseline 训练入口"
```

## Task 5: README And End-To-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README pipeline commands**

Modify the pipeline command block in `README.md` so it includes the new stages after `src/08_build_trend_model_samples.py`:

```sh
uv run python src/09_split_trend_model_samples.py
uv run python src/10_train_trend_baseline.py --model last_week
```

Add a short section after `trend_model_samples.parquet`:

```markdown
### 8. trend_model_samples_train/valid/test.parquet

基于 `trend_model_samples.parquet` 按时间顺序切分训练、验证和测试样本。默认最后 8 个样本周为 test，之前 8 个样本周为 valid，更早样本周为 train。切分配置集中在 `src/fashion_trend/config.py`：

```text
TREND_SPLIT_VALID_WEEKS = 8
TREND_SPLIT_TEST_WEEKS = 8
```

输出文件：

```sh
data/processed/features/trend_model_samples_train.parquet
data/processed/features/trend_model_samples_valid.parquet
data/processed/features/trend_model_samples_test.parquet
data/processed/features/trend_model_samples_split_metadata.json
```

运行命令：

```sh
uv run python src/09_split_trend_model_samples.py
```

### 9. last_week baseline

`last_week` baseline 使用上一段已观测属性占比增长预测下一段增长：

```text
pred_target_growth = growth_lag_1
```

预测结果、参数和元数据统一写入：

```sh
outputs/models/last_week/predictions.csv
outputs/models/last_week/params.json
outputs/models/last_week/metadata.json
```

运行命令：

```sh
uv run python src/10_train_trend_baseline.py --model last_week
```
```

- [ ] **Step 2: Run docs-adjacent verification**

Run:

```sh
uv run python -m py_compile src/09_split_trend_model_samples.py src/10_train_trend_baseline.py src/fashion_trend/models/baseline_last_week.py src/fashion_trend/trend.py src/fashion_trend/config.py
uv run python -m unittest discover -s tests -v
```

Expected: both commands exit with status 0.

- [ ] **Step 3: Run full local data pipeline if input files exist**

Run:

```sh
uv run python src/09_split_trend_model_samples.py
uv run python src/10_train_trend_baseline.py --model last_week
```

Expected if `data/processed/features/trend_model_samples.parquet` exists:
- `data/processed/features/trend_model_samples_train.parquet` exists.
- `data/processed/features/trend_model_samples_valid.parquet` exists.
- `data/processed/features/trend_model_samples_test.parquet` exists.
- `data/processed/features/trend_model_samples_split_metadata.json` exists.
- `outputs/models/last_week/predictions.csv` exists.
- `outputs/models/last_week/params.json` exists.
- `outputs/models/last_week/metadata.json` exists.

If the input parquet is missing, the first command should fail with a clear `趋势样本表不存在` message. Do not fabricate generated output in that case.

- [ ] **Step 4: Inspect generated outputs when full local data pipeline runs**

Run:

```sh
uv run python -c "import json, pandas as pd; base='data/processed/features'; train=pd.read_parquet(f'{base}/trend_model_samples_train.parquet'); valid=pd.read_parquet(f'{base}/trend_model_samples_valid.parquet'); test=pd.read_parquet(f'{base}/trend_model_samples_test.parquet'); pred=pd.read_csv('outputs/models/last_week/predictions.csv'); meta=json.load(open('outputs/models/last_week/metadata.json', encoding='utf-8')); print({'train_rows': len(train), 'valid_rows': len(valid), 'test_rows': len(test), 'pred_rows': len(pred), 'splits': sorted(pred.split.unique().tolist()), 'metadata_rows': meta['rows'], 'pred_missing': int(pred.isna().sum().sum())})"
```

Expected after full local data pipeline:
- `train_rows`, `valid_rows`, and `test_rows` are all greater than 0.
- `pred_rows` equals `train_rows + valid_rows + test_rows`.
- `splits` equals `['test', 'train', 'valid']`.
- `metadata_rows` equals `pred_rows`.
- `pred_missing` equals 0.

- [ ] **Step 5: Commit README and verification-ready pipeline**

Run:

```sh
git add README.md
git commit -m "docs(trend): 更新 baseline 流水线说明"
```

If `git status --short` shows generated files under `data/processed/features/` or `outputs/`, leave them unstaged and report them as generated verification artifacts. Do not commit generated data or model outputs unless the user explicitly asks for that.
