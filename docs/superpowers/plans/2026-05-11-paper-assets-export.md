# Paper Assets Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable reports stage that exports final paper figures, tables, and recommendation case studies from existing stable Fashion artifacts.

**Architecture:** Add a read-only `fashion_trend.reports` layer behind `src/17_export_paper_assets.py`. The reports layer reads stable artifacts through public readers/contracts, validates schemas and joins, exports CSV/Markdown/SVG/PNG/JSON files under `outputs/reports/`, and writes a manifest for auditability. It does not train models, rerun recommendation methods, or depend on upstream computation internals.

**Tech Stack:** Python 3.10-3.12, pandas, numpy, pyarrow, matplotlib, pytest, existing `foundation.io` atomic writers, existing domain readers/contracts.

---

## File Structure

Create or modify these files:

- Modify: `pyproject.toml`
  - Add runtime dependency `matplotlib>=3.10.0`.
- Modify: `uv.lock`
  - Update through `uv sync` after adding matplotlib.
- Modify: `src/fashion_trend/reports/paths.py`
  - Keep existing roots and add concrete path helpers for figures, tables, case studies, and manifest.
- Create: `src/fashion_trend/reports/markdown.py`
  - Dependency-free GitHub Flavored Markdown pipe-table writer.
- Create: `src/fashion_trend/reports/loaders.py`
  - Stable artifact loaders, schema checks, metrics flattening helpers, and LightGBM predictions + trend samples 1:1 join view.
- Create: `src/fashion_trend/reports/tables.py`
  - Build table DataFrames and write CSV + Markdown.
- Create: `src/fashion_trend/reports/plotting.py`
  - Matplotlib backend/font setup, CJK font discovery, and figure save helpers.
- Create: `src/fashion_trend/reports/figures.py`
  - Generate the 8 required paper figures.
- Create: `src/fashion_trend/reports/cases.py`
  - Select and render 3 recommendation case studies as JSON + Markdown.
- Create: `src/fashion_trend/reports/manifest.py`
  - Manifest payload validation and writing.
- Create: `src/fashion_trend/reports/runner.py`
  - Orchestrate full paper asset export.
- Create: `src/17_export_paper_assets.py`
  - Thin CLI entrypoint.
- Create: `tests/test_reports_markdown.py`
- Create: `tests/test_reports_loaders.py`
- Create: `tests/test_reports_tables.py`
- Create: `tests/test_reports_plotting.py`
- Create: `tests/test_reports_cases.py`
- Create: `tests/test_reports_runner.py`
- Modify: `README.md`
  - Document reports stage entrypoint and outputs.
- Modify: `docs/gpt-research/implementation-plan.md`
  - Update reports directory/command from design target to as-built plan.
- Modify: `docs/gpt-research/project-status-summary.md`
  - Note that implementation plan exists; final asset paths are produced by the reports command.

Do not commit generated `outputs/reports/` files.

---

### Task 1: Add Matplotlib Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add the runtime dependency**

In `pyproject.toml`, add `matplotlib` to `[project].dependencies`:

```toml
dependencies = [
    "kagglehub>=1.0.1",
    "lightgbm>=4.6.0",
    "matplotlib>=3.10.0",
    "numpy>=2.2.6",
    "pandas>=2.3.3",
    "pyarrow>=24.0.0",
    "scikit-learn>=1.7.2",
]
```

- [ ] **Step 2: Update the lockfile**

Run:

```sh
uv sync
```

Expected: command succeeds and `uv.lock` includes `matplotlib` and its transitive dependencies. Do not add `tabulate`, `seaborn`, `plotly`, `altair`, `networkx`, or `Pillow` unless the spec is explicitly changed.

- [ ] **Step 3: Verify imports**

Run:

```sh
uv run python - <<'PY'
import importlib.util
print("matplotlib", bool(importlib.util.find_spec("matplotlib")))
print("tabulate", bool(importlib.util.find_spec("tabulate")))
PY
```

Expected:

```text
matplotlib True
tabulate False
```

If `tabulate True` appears only as a transitive dependency, do not use it; the reports table writer must remain dependency-free.

- [ ] **Step 4: Review and commit**

Run:

```sh
git diff -- pyproject.toml uv.lock
git diff --check
```

Expected: dependency-only diff. Then commit:

```sh
git add pyproject.toml uv.lock
git commit -m "build: 添加报告图表绘图依赖"
```

---

### Task 2: Reports Paths and Markdown Writer

**Files:**
- Modify: `src/fashion_trend/reports/paths.py`
- Create: `src/fashion_trend/reports/markdown.py`
- Test: `tests/test_reports_markdown.py`

- [ ] **Step 1: Write Markdown writer tests**

Create `tests/test_reports_markdown.py`:

```python
from __future__ import annotations

import pandas as pd

from fashion_trend.reports.markdown import markdown_table


def test_markdown_table_uses_stable_column_order_and_escapes_cells() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "name": "A|B",
                "note": "line 1\nline 2",
                "value": 0.123456,
                "missing": None,
            }
        ]
    )

    text = markdown_table(
        dataframe,
        columns=("name", "note", "value", "missing"),
        float_format="{:.3f}",
    )

    assert text == (
        "| name | note | value | missing |\n"
        "| --- | --- | --- | --- |\n"
        "| A\\|B | line 1<br>line 2 | 0.123 |  |\n"
    )


def test_markdown_table_rejects_missing_columns() -> None:
    dataframe = pd.DataFrame([{"name": "A"}])

    try:
        markdown_table(dataframe, columns=("name", "value"))
    except ValueError as exc:
        assert "Markdown 表格缺少列" in str(exc)
    else:
        raise AssertionError("missing column should fail")


def test_markdown_table_handles_empty_frame() -> None:
    dataframe = pd.DataFrame(columns=["name", "value"])

    assert markdown_table(dataframe, columns=("name", "value")) == (
        "| name | value |\n"
        "| --- | --- |\n"
    )
```

- [ ] **Step 2: Run failing tests**

Run:

```sh
uv run pytest tests/test_reports_markdown.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fashion_trend.reports.markdown'`.

- [ ] **Step 3: Add concrete report path helpers**

Modify `src/fashion_trend/reports/paths.py`:

```python
from __future__ import annotations

from pathlib import Path

from fashion_trend.foundation.artifacts import validate_output_parent_dirs
from fashion_trend.foundation.paths import OUTPUT_DIR

# 报告阶段输出根目录。
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

# 报告阶段图表、表格和案例输出位置。
OUTPUT_FIGURES_DIR = OUTPUT_REPORTS_DIR / "figures"
OUTPUT_TABLES_DIR = OUTPUT_REPORTS_DIR / "tables"
OUTPUT_CASE_STUDIES_DIR = OUTPUT_REPORTS_DIR / "case_studies"
OUTPUT_REPORTS_MANIFEST_PATH = OUTPUT_REPORTS_DIR / "manifest.json"


def figure_output_paths(name: str) -> dict[str, Path]:
    """返回同一图表的 SVG 和 PNG 输出路径。"""
    _validate_report_artifact_name(name)
    return {
        "svg": OUTPUT_FIGURES_DIR / f"{name}.svg",
        "png": OUTPUT_FIGURES_DIR / f"{name}.png",
    }


def table_output_paths(name: str) -> dict[str, Path]:
    """返回同一表格的 CSV 和 Markdown 输出路径。"""
    _validate_report_artifact_name(name)
    return {
        "csv": OUTPUT_TABLES_DIR / f"{name}.csv",
        "markdown": OUTPUT_TABLES_DIR / f"{name}.md",
    }


def case_study_output_paths(case_id: str) -> dict[str, Path]:
    """返回单个案例的 JSON 和 Markdown 输出路径。"""
    _validate_report_artifact_name(case_id)
    return {
        "json": OUTPUT_CASE_STUDIES_DIR / f"{case_id}.json",
        "markdown": OUTPUT_CASE_STUDIES_DIR / f"{case_id}.md",
    }


def validate_report_output_path(path: Path) -> None:
    """确认报告产物仍写在 outputs/reports/ 内。"""
    validate_output_parent_dirs(path.parent, OUTPUT_REPORTS_DIR)


def _validate_report_artifact_name(name: str) -> None:
    if not name:
        raise ValueError("报告产物名称不能为空。")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"报告产物名称不是安全路径片段: {name}")
```

- [ ] **Step 4: Implement dependency-free Markdown writer**

Create `src/fashion_trend/reports/markdown.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def markdown_table(
    dataframe: pd.DataFrame,
    *,
    columns: Sequence[str],
    float_format: str = "{:.6f}",
) -> str:
    """Render a small DataFrame as a GitHub Flavored Markdown pipe table.

    This deliberately avoids pandas.DataFrame.to_markdown(), which depends on
    the optional tabulate package that this project does not declare.
    """
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Markdown 表格缺少列: {missing_columns}")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(
            _format_markdown_cell(row[column], float_format=float_format)
            for column in columns
        )
        + " |"
        for _, row in dataframe.loc[:, list(columns)].iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def _format_markdown_cell(value: Any, *, float_format: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        text = float_format.format(value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")
```

- [ ] **Step 5: Run tests**

Run:

```sh
uv run pytest tests/test_reports_markdown.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```sh
git diff -- src/fashion_trend/reports/paths.py src/fashion_trend/reports/markdown.py tests/test_reports_markdown.py
git diff --check
git add src/fashion_trend/reports/paths.py src/fashion_trend/reports/markdown.py tests/test_reports_markdown.py
git commit -m "feat(reports): 添加报告路径和 Markdown 表格 writer"
```

---

### Task 3: Load Stable Report Inputs

**Files:**
- Create: `src/fashion_trend/reports/loaders.py`
- Test: `tests/test_reports_loaders.py`

- [ ] **Step 1: Write loader tests for prediction/sample join**

Create `tests/test_reports_loaders.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fashion_trend.reports.loaders import (
    REPORT_TREND_JOIN_KEY,
    build_lightgbm_prediction_sample_view,
    flatten_recommendation_metrics,
    flatten_trend_metrics,
    read_json_object,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "week_id": 103,
                "attr_id": "colour_group_name::Light Green",
                "attr_type": "colour_group_name",
                "attr_value": "Light Green",
                "model_name": "lightgbm",
                "split": "test",
                "share_t": 0.1,
                "pred_share_t1": 0.12,
                "target_growth": 0.2,
                "pred_target_growth": 0.1,
                "target_rank_in_type_t1": 1,
            }
        ]
    )


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "week_id": 103,
                "attr_id": "colour_group_name::Light Green",
                "attr_type": "colour_group_name",
                "attr_value": "Light Green",
                "heat_t": 120,
                "share_t": 0.1,
                "target_growth": 0.2,
                "history_total_heat_t": 1000,
                "history_active_weeks_t": 20,
                "is_trend_eligible_t": 1,
            }
        ]
    )


def test_build_lightgbm_prediction_sample_view_adds_filter_columns() -> None:
    view = build_lightgbm_prediction_sample_view(_prediction_frame(), _sample_frame())

    assert tuple(view.loc[0, REPORT_TREND_JOIN_KEY]) == (
        103,
        "colour_group_name::Light Green",
        "colour_group_name",
        "Light Green",
    )
    assert view.loc[0, "heat_t"] == 120
    assert view.loc[0, "history_total_heat_t"] == 1000
    assert view.loc[0, "history_active_weeks_t"] == 20
    assert view.loc[0, "is_trend_eligible_t"] == 1


def test_build_lightgbm_prediction_sample_view_rejects_duplicate_prediction_key() -> None:
    predictions = pd.concat([_prediction_frame(), _prediction_frame()])

    try:
        build_lightgbm_prediction_sample_view(predictions, _sample_frame())
    except ValueError as exc:
        assert "LightGBM predictions 存在重复 join key" in str(exc)
    else:
        raise AssertionError("duplicate prediction key should fail")


def test_build_lightgbm_prediction_sample_view_rejects_missing_sample_match() -> None:
    samples = _sample_frame()
    samples.loc[0, "attr_value"] = "Dark Green"

    try:
        build_lightgbm_prediction_sample_view(_prediction_frame(), samples)
    except ValueError as exc:
        assert "无法 1:1 join" in str(exc)
    else:
        raise AssertionError("missing sample match should fail")


def test_build_lightgbm_prediction_sample_view_rejects_conflicting_shared_values() -> None:
    samples = _sample_frame()
    samples.loc[0, "share_t"] = 0.2

    try:
        build_lightgbm_prediction_sample_view(_prediction_frame(), samples)
    except ValueError as exc:
        assert "share_t 不一致" in str(exc)
    else:
        raise AssertionError("conflicting share_t should fail")


def test_read_json_object_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    try:
        read_json_object(path, artifact_name="payload")
    except ValueError as exc:
        assert "必须是 JSON object" in str(exc)
    else:
        raise AssertionError("non-object json should fail")


def test_flatten_trend_metrics_extracts_report_columns() -> None:
    payload = {
        "model_name": "lightgbm",
        "run_id": "run-1",
        "overall": {
            "test": {
                "mae": 0.1,
                "rmse": 0.2,
                "spearman": 0.3,
                "ndcg_at_k": {"10": 0.4},
                "precision_at_k": {"10": 0.5},
                "recall_at_k": {"10": 0.6},
            }
        },
    }

    rows = flatten_trend_metrics(payload)

    assert rows == [
        {
            "model_name": "lightgbm",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-1",
        }
    ]


def test_flatten_recommendation_metrics_extracts_report_columns() -> None:
    payload = {
        "method": "pop_similarity_trend",
        "metrics": {
            "valid": {
                "map_at_12": 0.1,
                "recall_at_12": 0.2,
                "hit_rate_at_12": 0.3,
                "ndcg_at_12": 0.4,
                "coverage": 0.5,
                "user_count": 10,
                "missing_recommendation_user_count": 0,
            }
        },
    }

    rows = flatten_recommendation_metrics(payload)

    assert rows == [
        {
            "method": "pop_similarity_trend",
            "split": "valid",
            "map_at_12": 0.1,
            "recall_at_12": 0.2,
            "hit_rate_at_12": 0.3,
            "ndcg_at_12": 0.4,
            "coverage": 0.5,
            "user_count": 10,
            "missing_recommendation_user_count": 0,
        }
    ]
```

- [ ] **Step 2: Run failing tests**

Run:

```sh
uv run pytest tests/test_reports_loaders.py -q
```

Expected: FAIL because `fashion_trend.reports.loaders` does not exist.

- [ ] **Step 3: Implement loaders**

Create `src/fashion_trend/reports/loaders.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPORT_TREND_JOIN_KEY = ("week_id", "attr_id", "attr_type", "attr_value")
REPORT_TREND_SAMPLE_COLUMNS = (
    *REPORT_TREND_JOIN_KEY,
    "heat_t",
    "share_t",
    "target_growth",
    "history_total_heat_t",
    "history_active_weeks_t",
    "is_trend_eligible_t",
)
REPORT_TREND_VIEW_COLUMNS = (
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
    "heat_t",
    "history_total_heat_t",
    "history_active_weeks_t",
    "is_trend_eligible_t",
)


def read_json_object(path: Path, *, artifact_name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{artifact_name} 文件不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 {artifact_name}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} 必须是 JSON object: {path}")
    return payload


def read_feature_importance(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = {"feature", "importance_gain", "importance_split", "importance_gain_normalized"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"LightGBM feature_importance 缺少列: {missing}")
    return dataframe


def read_trend_samples(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path, columns=list(REPORT_TREND_SAMPLE_COLUMNS))
    missing = sorted(set(REPORT_TREND_SAMPLE_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(f"trend_model_samples 缺少列: {missing}")
    return dataframe


def build_lightgbm_prediction_sample_view(
    predictions: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    _reject_duplicate_join_key(
        predictions,
        artifact_name="LightGBM predictions",
    )
    _reject_duplicate_join_key(samples, artifact_name="trend_model_samples")

    joined = predictions.merge(
        samples.loc[:, list(REPORT_TREND_SAMPLE_COLUMNS)],
        on=list(REPORT_TREND_JOIN_KEY),
        how="left",
        suffixes=("", "_sample"),
        validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all() or len(joined) != len(predictions):
        sample = joined.loc[joined["_merge"] != "both", list(REPORT_TREND_JOIN_KEY)]
        raise ValueError(
            "LightGBM predictions 与 trend_model_samples 无法 1:1 join: "
            f"{sample.head(3).to_dict('records')}"
        )
    _validate_joined_numeric_consistency(joined, column="share_t")
    _validate_joined_numeric_consistency(joined, column="target_growth")
    return joined.loc[:, list(REPORT_TREND_VIEW_COLUMNS)].copy()


def flatten_trend_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    model_name = str(payload.get("model_name"))
    run_id = payload.get("run_id")
    rows: list[dict[str, Any]] = []
    for split, metrics in sorted(_required_dict(payload, "overall").items()):
        metric_payload = _as_dict(metrics, f"overall.{split}")
        rows.append(
            {
                "model_name": model_name,
                "split": split,
                "mae": _finite_number(metric_payload, "mae"),
                "rmse": _finite_number(metric_payload, "rmse"),
                "spearman": _finite_number(metric_payload, "spearman"),
                "ndcg_at_10": _metric_at_k(metric_payload, "ndcg_at_k", "10"),
                "precision_at_10": _metric_at_k(metric_payload, "precision_at_k", "10"),
                "recall_at_10": _metric_at_k(metric_payload, "recall_at_k", "10"),
                "run_id": "" if run_id is None else str(run_id),
            }
        )
    return rows


def flatten_recommendation_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    method = str(payload.get("method"))
    rows: list[dict[str, Any]] = []
    for split, metrics in sorted(_required_dict(payload, "metrics").items()):
        metric_payload = _as_dict(metrics, f"metrics.{split}")
        rows.append(
            {
                "method": method,
                "split": split,
                "map_at_12": _finite_number(metric_payload, "map_at_12"),
                "recall_at_12": _finite_number(metric_payload, "recall_at_12"),
                "hit_rate_at_12": _finite_number(metric_payload, "hit_rate_at_12"),
                "ndcg_at_12": _finite_number(metric_payload, "ndcg_at_12"),
                "coverage": _finite_number(metric_payload, "coverage"),
                "user_count": int(_finite_number(metric_payload, "user_count")),
                "missing_recommendation_user_count": int(
                    _finite_number(metric_payload, "missing_recommendation_user_count")
                ),
            }
        )
    return rows


def _reject_duplicate_join_key(dataframe: pd.DataFrame, *, artifact_name: str) -> None:
    duplicated = dataframe.duplicated(list(REPORT_TREND_JOIN_KEY), keep=False)
    if duplicated.any():
        sample = dataframe.loc[duplicated, list(REPORT_TREND_JOIN_KEY)].head(3)
        raise ValueError(
            f"{artifact_name} 存在重复 join key: {sample.to_dict('records')}"
        )


def _validate_joined_numeric_consistency(dataframe: pd.DataFrame, *, column: str) -> None:
    left = pd.to_numeric(dataframe[column], errors="raise")
    right = pd.to_numeric(dataframe[f"{column}_sample"], errors="raise")
    if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), atol=1e-12):
        raise ValueError(f"LightGBM predictions 与 samples 的 {column} 不一致。")


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"metrics payload 缺少字段: {key}")
    return _as_dict(payload[key], key)


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 JSON object。")
    return value


def _finite_number(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"metrics payload 缺少字段: {key}")
    value = float(payload[key])
    if not np.isfinite(value):
        raise ValueError(f"metrics payload 字段不是有限数值: {key}")
    return value


def _metric_at_k(payload: dict[str, Any], metric_name: str, k: str) -> float:
    metrics = _required_dict(payload, metric_name)
    return _finite_number(metrics, k)
```

- [ ] **Step 4: Run loader tests**

Run:

```sh
uv run pytest tests/test_reports_loaders.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```sh
git diff -- src/fashion_trend/reports/loaders.py tests/test_reports_loaders.py
git diff --check
git add src/fashion_trend/reports/loaders.py tests/test_reports_loaders.py
git commit -m "feat(reports): 添加论文素材读取校验"
```

---

### Task 4: Table Export

**Files:**
- Create: `src/fashion_trend/reports/tables.py`
- Test: `tests/test_reports_tables.py`

- [ ] **Step 1: Write table contract tests**

Create `tests/test_reports_tables.py`:

```python
from __future__ import annotations

import pandas as pd

from fashion_trend.reports.tables import (
    RECOMMENDATION_METHOD_METRICS_COLUMNS,
    TREND_MODEL_METRICS_COLUMNS,
    build_recommendation_method_metrics_table,
    build_trend_model_metrics_table,
    write_report_table,
)


def test_build_trend_model_metrics_table_uses_contract_order() -> None:
    rows = [
        {
            "model_name": "lightgbm",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-1",
        }
    ]

    table = build_trend_model_metrics_table(rows)

    assert tuple(table.columns) == TREND_MODEL_METRICS_COLUMNS
    assert table.loc[0, "model_name"] == "lightgbm"


def test_build_recommendation_method_metrics_table_uses_contract_order() -> None:
    rows = [
        {
            "method": "pop_similarity_trend",
            "split": "valid",
            "map_at_12": 0.1,
            "recall_at_12": 0.2,
            "hit_rate_at_12": 0.3,
            "ndcg_at_12": 0.4,
            "coverage": 0.5,
            "user_count": 10,
            "missing_recommendation_user_count": 0,
        }
    ]

    table = build_recommendation_method_metrics_table(rows)

    assert tuple(table.columns) == RECOMMENDATION_METHOD_METRICS_COLUMNS
    assert table.loc[0, "method"] == "pop_similarity_trend"


def test_write_report_table_writes_csv_and_markdown(tmp_path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "split": "test",
                "mae": 0.1,
                "rmse": 0.2,
                "spearman": 0.3,
                "ndcg_at_10": 0.4,
                "precision_at_10": 0.5,
                "recall_at_10": 0.6,
                "run_id": "run-1",
            }
        ]
    )
    paths = {
        "csv": tmp_path / "trend_model_metrics.csv",
        "markdown": tmp_path / "trend_model_metrics.md",
    }

    written = write_report_table(
        dataframe,
        columns=TREND_MODEL_METRICS_COLUMNS,
        output_paths=paths,
    )

    assert written == [paths["csv"], paths["markdown"]]
    assert paths["csv"].read_text(encoding="utf-8").startswith('"model_name"')
    assert paths["markdown"].read_text(encoding="utf-8").startswith("| model_name |")
```

- [ ] **Step 2: Run failing tests**

Run:

```sh
uv run pytest tests/test_reports_tables.py -q
```

Expected: FAIL because `fashion_trend.reports.tables` does not exist.

- [ ] **Step 3: Implement table builders and writers**

Create `src/fashion_trend/reports/tables.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fashion_trend.foundation.io import write_csv_atomic, write_text_atomic
from fashion_trend.reports.markdown import markdown_table

TREND_MODEL_METRICS_COLUMNS = (
    "model_name",
    "split",
    "mae",
    "rmse",
    "spearman",
    "ndcg_at_10",
    "precision_at_10",
    "recall_at_10",
    "run_id",
)
RECOMMENDATION_METHOD_METRICS_COLUMNS = (
    "method",
    "split",
    "map_at_12",
    "recall_at_12",
    "hit_rate_at_12",
    "ndcg_at_12",
    "coverage",
    "user_count",
    "missing_recommendation_user_count",
)


def build_trend_model_metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(rows)
    return _select_and_sort(
        dataframe,
        columns=TREND_MODEL_METRICS_COLUMNS,
        sort_columns=("model_name", "split"),
        table_name="trend_model_metrics",
    )


def build_recommendation_method_metrics_table(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    dataframe = pd.DataFrame(rows)
    return _select_and_sort(
        dataframe,
        columns=RECOMMENDATION_METHOD_METRICS_COLUMNS,
        sort_columns=("method", "split"),
        table_name="recommendation_method_metrics",
    )


def write_report_table(
    dataframe: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    output_paths: dict[str, Path],
) -> list[Path]:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"报告表格缺少列: {missing}")
    table = dataframe.loc[:, list(columns)]
    csv_path = output_paths["csv"]
    markdown_path = output_paths["markdown"]
    write_csv_atomic(table, csv_path)
    write_text_atomic(markdown_table(table, columns=columns), markdown_path)
    _validate_non_empty_file(csv_path)
    _validate_non_empty_file(markdown_path)
    return [csv_path, markdown_path]


def _select_and_sort(
    dataframe: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    sort_columns: tuple[str, ...],
    table_name: str,
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{table_name} 缺少列: {missing}")
    return dataframe.loc[:, list(columns)].sort_values(list(sort_columns)).reset_index(
        drop=True
    )


def _validate_non_empty_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"报告表格输出为空: {path}")
```

- [ ] **Step 4: Run table tests**

Run:

```sh
uv run pytest tests/test_reports_markdown.py tests/test_reports_tables.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```sh
git diff -- src/fashion_trend/reports/tables.py tests/test_reports_tables.py
git diff --check
git add src/fashion_trend/reports/tables.py tests/test_reports_tables.py
git commit -m "feat(reports): 添加论文表格导出"
```

---

### Task 5: Plotting Environment and Figure Export

**Files:**
- Create: `src/fashion_trend/reports/plotting.py`
- Create: `src/fashion_trend/reports/figures.py`
- Test: `tests/test_reports_plotting.py`

- [ ] **Step 1: Write plotting tests**

Create `tests/test_reports_plotting.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt

from fashion_trend.reports.plotting import (
    configure_matplotlib_for_reports,
    save_report_figure,
)


def test_configure_matplotlib_requires_cjk_font(monkeypatch) -> None:
    monkeypatch.setattr("fashion_trend.reports.plotting.available_cjk_fonts", lambda: [])

    try:
        configure_matplotlib_for_reports()
    except RuntimeError as exc:
        assert "缺少可用中文字体" in str(exc)
    else:
        raise AssertionError("missing CJK font should fail")


def test_save_report_figure_writes_svg_and_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "fashion_trend.reports.plotting.available_cjk_fonts",
        lambda: ["DejaVu Sans"],
    )
    configure_matplotlib_for_reports()
    figure, axis = plt.subplots()
    axis.set_title("中文标题 NDCG@12")
    axis.plot([-1, 0, 1], [1, 0, -1])
    paths = {"svg": tmp_path / "figure.svg", "png": tmp_path / "figure.png"}

    written = save_report_figure(figure, paths)

    assert written == [paths["svg"], paths["png"]]
    assert paths["svg"].stat().st_size > 0
    assert paths["png"].stat().st_size > 0
```

- [ ] **Step 2: Run failing tests**

Run:

```sh
uv run pytest tests/test_reports_plotting.py -q
```

Expected: FAIL because `fashion_trend.reports.plotting` does not exist.

- [ ] **Step 3: Implement plotting environment**

Create `src/fashion_trend/reports/plotting.py`:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
)


def available_cjk_fonts() -> list[str]:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [font_name for font_name in CJK_FONT_CANDIDATES if font_name in installed]


def configure_matplotlib_for_reports() -> str:
    fonts = available_cjk_fonts()
    if not fonts:
        raise RuntimeError(
            "缺少可用中文字体，无法可靠导出中文 SVG/PNG。"
            "请安装 PingFang SC、Noto Sans CJK SC 或 Microsoft YaHei 等字体。"
        )
    selected_font = fonts[0]
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    return selected_font


def save_report_figure(figure: Figure, output_paths: dict[str, Path]) -> list[Path]:
    written: list[Path] = []
    for suffix in ("svg", "png"):
        output_path = output_paths[suffix]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ValueError(f"报告图表输出为空: {output_path}")
        written.append(output_path)
    plt.close(figure)
    return written
```

- [ ] **Step 4: Implement initial figure builders**

Create `src/fashion_trend/reports/figures.py` with the first three reusable chart functions:

```python
from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def build_trend_model_metrics_figure(metrics: pd.DataFrame) -> Figure:
    pivot = metrics.pivot(index="model_name", columns="split", values="ndcg_at_10")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("趋势模型排序指标对比 NDCG@10")
    axis.set_xlabel("model_name")
    axis.set_ylabel("NDCG@10")
    axis.legend(title="split")
    figure.tight_layout()
    return figure


def build_recommendation_method_metrics_figure(metrics: pd.DataFrame) -> Figure:
    pivot = metrics.pivot(index="method", columns="split", values="ndcg_at_12")
    figure, axis = plt.subplots(figsize=(9, 4.8))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("推荐方法排序指标对比 NDCG@12")
    axis.set_xlabel("method")
    axis.set_ylabel("NDCG@12")
    axis.legend(title="split")
    figure.tight_layout()
    return figure


def build_feature_importance_figure(
    feature_importance: pd.DataFrame,
    *,
    top_n: int = 15,
) -> Figure:
    top_features = (
        feature_importance.sort_values("importance_gain_normalized", ascending=False)
        .head(top_n)
        .sort_values("importance_gain_normalized")
    )
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(top_features["feature"], top_features["importance_gain_normalized"])
    axis.set_title("LightGBM 特征重要性 Top-N")
    axis.set_xlabel("normalized gain")
    axis.set_ylabel("feature")
    figure.tight_layout()
    return figure
```

Task 9 will call these functions from the runner and add the remaining five figure builders in the same module.

- [ ] **Step 5: Run plotting tests**

Run:

```sh
uv run pytest tests/test_reports_plotting.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```sh
git diff -- src/fashion_trend/reports/plotting.py src/fashion_trend/reports/figures.py tests/test_reports_plotting.py
git diff --check
git add src/fashion_trend/reports/plotting.py src/fashion_trend/reports/figures.py tests/test_reports_plotting.py
git commit -m "feat(reports): 添加论文图表绘图基础"
```

---

### Task 6: Case Study Selection and Rendering

**Files:**
- Create: `src/fashion_trend/reports/cases.py`
- Test: `tests/test_reports_cases.py`

- [ ] **Step 1: Write case tests**

Create `tests/test_reports_cases.py`:

```python
from __future__ import annotations

import pandas as pd

from fashion_trend.reports.cases import (
    build_case_payload,
    select_recommendation_cases,
)


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "customer-a",
                "split": "test",
                "cutoff_week": 103,
                "label_week": 104,
                "method": "pop_similarity_trend",
                "article_id": "0000000001",
                "rank": 1,
                "score": 0.9,
                "pop_score": 1.0,
                "sim_score": 0.8,
                "trend_score": 0.7,
                "recent_score": 0.6,
                "candidate_sources": "popularity",
            },
            {
                "customer_id": "customer-b",
                "split": "test",
                "cutoff_week": 103,
                "label_week": 104,
                "method": "pop_similarity_trend",
                "article_id": "0000000002",
                "rank": 1,
                "score": 0.8,
                "pop_score": 0.9,
                "sim_score": 0.7,
                "trend_score": 0.6,
                "recent_score": 0.5,
                "candidate_sources": "trend_union",
            },
        ]
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "customer-a",
                "split": "test",
                "cutoff_week": 103,
                "label_week": 104,
                "article_id": "0000000001",
            }
        ]
    )


def _profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "customer-a",
                "split": "test",
                "cutoff_week": 103,
                "label_week": 104,
                "attr_id": "colour_group_name::Black",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "preference_score": 0.5,
                "purchase_count": 4,
                "last_purchase_week": 100,
            },
            {
                "customer_id": "customer-b",
                "split": "test",
                "cutoff_week": 103,
                "label_week": 104,
                "attr_id": "colour_group_name::Blue",
                "attr_type": "colour_group_name",
                "attr_value": "Blue",
                "preference_score": 0.4,
                "purchase_count": 3,
                "last_purchase_week": 101,
            },
        ]
    )


def test_select_recommendation_cases_prioritizes_hits() -> None:
    selected = select_recommendation_cases(
        recommendation_items=_items(),
        evaluation_labels=_labels(),
        user_profile=_profiles(),
        case_count=1,
    )

    assert selected == [("customer-a", "test", 103, 104)]


def test_select_recommendation_cases_fails_when_not_enough_cases() -> None:
    try:
        select_recommendation_cases(
            recommendation_items=_items(),
            evaluation_labels=_labels(),
            user_profile=_profiles().iloc[0:0],
            case_count=1,
        )
    except ValueError as exc:
        assert "不足 1 个推荐案例" in str(exc)
    else:
        raise AssertionError("missing profile should fail")


def test_build_case_payload_marks_hits() -> None:
    payload = build_case_payload(
        case_key=("customer-a", "test", 103, 104),
        recommendation_items=_items(),
        evaluation_labels=_labels(),
        user_profile=_profiles(),
    )

    assert payload["customer_id"] == "customer-a"
    assert payload["hit_count"] == 1
    assert payload["recommendations"][0]["is_hit"] is True
    assert payload["profile"][0]["attr_value"] == "Black"
```

- [ ] **Step 2: Run failing tests**

Run:

```sh
uv run pytest tests/test_reports_cases.py -q
```

Expected: FAIL because `fashion_trend.reports.cases` does not exist.

- [ ] **Step 3: Implement cases module**

Create `src/fashion_trend/reports/cases.py`:

```python
from __future__ import annotations

from typing import Any

import pandas as pd

CASE_KEY_COLUMNS = ("customer_id", "split", "cutoff_week", "label_week")


def select_recommendation_cases(
    *,
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    user_profile: pd.DataFrame,
    case_count: int,
) -> list[tuple[str, str, int, int]]:
    test_items = recommendation_items.loc[
        recommendation_items["split"].astype(str) == "test"
    ].copy()
    _require_score_columns(test_items)
    test_labels = evaluation_labels.loc[
        evaluation_labels["split"].astype(str) == "test"
    ].copy()
    test_profiles = user_profile.loc[user_profile["split"].astype(str) == "test"].copy()

    hits = test_items.merge(
        test_labels.loc[:, [*CASE_KEY_COLUMNS, "article_id"]],
        on=[*CASE_KEY_COLUMNS, "article_id"],
        how="left",
        indicator=True,
    )
    hits["is_hit"] = hits["_merge"] == "both"
    hit_counts = (
        hits.groupby(list(CASE_KEY_COLUMNS), as_index=False)["is_hit"]
        .sum()
        .rename(columns={"is_hit": "hit_count"})
    )
    profile_counts = (
        test_profiles.groupby(list(CASE_KEY_COLUMNS), as_index=False)
        .size()
        .rename(columns={"size": "profile_count"})
    )
    candidates = hit_counts.merge(profile_counts, on=list(CASE_KEY_COLUMNS), how="inner")
    candidates = candidates.loc[candidates["profile_count"] > 0]
    candidates = candidates.sort_values(
        ["hit_count", "profile_count", "customer_id"],
        ascending=[False, False, True],
    )
    if len(candidates) < case_count:
        raise ValueError(f"不足 {case_count} 个推荐案例。")
    return [
        (
            str(row.customer_id),
            str(row.split),
            int(row.cutoff_week),
            int(row.label_week),
        )
        for row in candidates.head(case_count).itertuples(index=False)
    ]


def build_case_payload(
    *,
    case_key: tuple[str, str, int, int],
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    user_profile: pd.DataFrame,
) -> dict[str, Any]:
    customer_id, split, cutoff_week, label_week = case_key
    item_mask = _case_mask(recommendation_items, case_key)
    label_mask = _case_mask(evaluation_labels, case_key)
    profile_mask = _case_mask(user_profile, case_key)
    items = recommendation_items.loc[item_mask].sort_values("rank")
    labels = set(evaluation_labels.loc[label_mask, "article_id"].astype(str))
    profile = user_profile.loc[profile_mask].sort_values(
        ["preference_score", "purchase_count"],
        ascending=[False, False],
    )
    recommendations = []
    for row in items.itertuples(index=False):
        article_id = str(row.article_id)
        recommendations.append(
            {
                "rank": int(row.rank),
                "article_id": article_id,
                "is_hit": article_id in labels,
                "score": float(row.score),
                "pop_score": float(row.pop_score),
                "sim_score": float(row.sim_score),
                "trend_score": float(row.trend_score),
                "recent_score": float(row.recent_score),
                "candidate_sources": str(row.candidate_sources),
            }
        )
    return {
        "customer_id": customer_id,
        "split": split,
        "cutoff_week": cutoff_week,
        "label_week": label_week,
        "hit_count": sum(1 for row in recommendations if row["is_hit"]),
        "profile": [
            {
                "attr_type": str(row.attr_type),
                "attr_value": str(row.attr_value),
                "preference_score": float(row.preference_score),
                "purchase_count": int(row.purchase_count),
                "last_purchase_week": int(row.last_purchase_week),
            }
            for row in profile.head(5).itertuples(index=False)
        ],
        "recommendations": recommendations,
    }


def _case_mask(dataframe: pd.DataFrame, case_key: tuple[str, str, int, int]) -> pd.Series:
    customer_id, split, cutoff_week, label_week = case_key
    return (
        (dataframe["customer_id"].astype(str) == customer_id)
        & (dataframe["split"].astype(str) == split)
        & (dataframe["cutoff_week"].astype(int) == cutoff_week)
        & (dataframe["label_week"].astype(int) == label_week)
    )


def _require_score_columns(dataframe: pd.DataFrame) -> None:
    required = {
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
        "candidate_sources",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"推荐案例缺少解释字段: {missing}")
    if dataframe.loc[:, list(required)].isna().any().any():
        raise ValueError("推荐案例解释字段存在缺失值。")
```

- [ ] **Step 4: Run case tests**

Run:

```sh
uv run pytest tests/test_reports_cases.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```sh
git diff -- src/fashion_trend/reports/cases.py tests/test_reports_cases.py
git diff --check
git add src/fashion_trend/reports/cases.py tests/test_reports_cases.py
git commit -m "feat(reports): 添加推荐案例筛选"
```

---

### Task 7: Manifest and Runner Orchestration

**Files:**
- Create: `src/fashion_trend/reports/manifest.py`
- Create: `src/fashion_trend/reports/runner.py`
- Test: `tests/test_reports_runner.py`

- [ ] **Step 1: Write runner smoke test**

Create `tests/test_reports_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

from fashion_trend.reports.manifest import build_manifest_payload


def test_build_manifest_payload_records_outputs() -> None:
    payload = build_manifest_payload(
        parameters={"case_count": 3},
        input_artifacts={"predictions": "outputs/models/lightgbm/predictions.csv"},
        output_artifacts={
            "figures": ["outputs/reports/figures/a.svg", "outputs/reports/figures/a.png"],
            "tables": ["outputs/reports/tables/a.csv", "outputs/reports/tables/a.md"],
            "case_studies": ["outputs/reports/case_studies/case_1.json"],
        },
        row_counts={"trend_model_metrics": 8},
        case_user_ids=["customer-a"],
        warnings=["current grid has valid metrics only"],
    )

    assert payload["schema_version"] == "paper_assets_manifest/v1"
    assert payload["figure_count"] == 2
    assert payload["table_count"] == 2
    assert payload["case_count"] == 1
    assert payload["case_user_ids"] == ["customer-a"]
```

- [ ] **Step 2: Run failing test**

Run:

```sh
uv run pytest tests/test_reports_runner.py -q
```

Expected: FAIL because `fashion_trend.reports.manifest` does not exist.

- [ ] **Step 3: Implement manifest module**

Create `src/fashion_trend/reports/manifest.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fashion_trend.foundation.io import write_json_atomic

MANIFEST_SCHEMA_VERSION = "paper_assets_manifest/v1"


def build_manifest_payload(
    *,
    parameters: dict[str, Any],
    input_artifacts: dict[str, str],
    output_artifacts: dict[str, list[str]],
    row_counts: dict[str, int],
    case_user_ids: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "row_counts": row_counts,
        "figure_count": len(output_artifacts.get("figures", [])),
        "table_count": len(output_artifacts.get("tables", [])),
        "case_count": len(case_user_ids),
        "case_user_ids": case_user_ids,
        "warnings": warnings,
    }
    _validate_manifest(payload)
    return payload


def write_manifest(payload: dict[str, Any], output_path) -> None:
    _validate_manifest(payload)
    write_json_atomic(payload, output_path)


def _validate_manifest(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "generated_at",
        "parameters",
        "input_artifacts",
        "output_artifacts",
        "row_counts",
        "figure_count",
        "table_count",
        "case_count",
        "case_user_ids",
        "warnings",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"reports manifest 缺少字段: {missing}")
```

- [ ] **Step 4: Implement runner skeleton**

Create `src/fashion_trend/reports/runner.py`. Start with the orchestration contract and a minimal implementation that writes a valid manifest; Task 9 will replace the empty output lists with real artifacts:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fashion_trend.reports.manifest import build_manifest_payload, write_manifest
from fashion_trend.reports.paths import OUTPUT_REPORTS_MANIFEST_PATH


@dataclass(frozen=True)
class PaperAssetsExportConfig:
    case_count: int = 3
    top_k: int = 10
    trend_week: int = 103
    figure_formats: tuple[str, ...] = ("svg", "png")
    output_dir: Path | None = None


def run_paper_assets_export(config: PaperAssetsExportConfig) -> dict[str, Any]:
    """Export paper assets from stable artifacts.

    This initial function writes a valid manifest and establishes the public
    runner contract before Task 9 connects real tables, figures, and cases.
    """
    payload = build_manifest_payload(
        parameters={
            "case_count": config.case_count,
            "top_k": config.top_k,
            "trend_week": config.trend_week,
            "figure_formats": list(config.figure_formats),
        },
        input_artifacts={},
        output_artifacts={"figures": [], "tables": [], "case_studies": []},
        row_counts={},
        case_user_ids=[],
        warnings=[],
    )
    write_manifest(payload, OUTPUT_REPORTS_MANIFEST_PATH)
    return payload
```

This runner is an intermediate contract. Task 9 must replace the empty artifact lists with real table, figure, and case outputs before final validation.

- [ ] **Step 5: Run runner tests**

Run:

```sh
uv run pytest tests/test_reports_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```sh
git diff -- src/fashion_trend/reports/manifest.py src/fashion_trend/reports/runner.py tests/test_reports_runner.py
git diff --check
git add src/fashion_trend/reports/manifest.py src/fashion_trend/reports/runner.py tests/test_reports_runner.py
git commit -m "feat(reports): 添加报告导出 manifest"
```

---

### Task 8: CLI Entrypoint

**Files:**
- Create: `src/17_export_paper_assets.py`
- Test: `tests/test_reports_runner.py`

- [ ] **Step 1: Add CLI test**

Append to `tests/test_reports_runner.py`:

```python
import importlib


def test_export_paper_assets_cli_passes_args(monkeypatch) -> None:
    module = importlib.import_module("17_export_paper_assets")
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {"manifest_path": "outputs/reports/manifest.json"}

    monkeypatch.setattr(module, "run_paper_assets_export", fake_run)

    exit_code = module.main(
        [
            "--case-count",
            "2",
            "--top-k",
            "5",
            "--trend-week",
            "102",
            "--figure-format",
            "svg,png",
        ]
    )

    assert exit_code == 0
    assert captured["config"].case_count == 2
    assert captured["config"].top_k == 5
    assert captured["config"].trend_week == 102
    assert captured["config"].figure_formats == ("svg", "png")
```

- [ ] **Step 2: Run failing CLI test**

Run:

```sh
uv run pytest tests/test_reports_runner.py::test_export_paper_assets_cli_passes_args -q
```

Expected: FAIL because `src/17_export_paper_assets.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `src/17_export_paper_assets.py`:

```python
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fashion_trend.foundation import logging as log
from fashion_trend.reports.runner import (
    PaperAssetsExportConfig,
    run_paper_assets_export,
)

LOG_SOURCE = "paper-assets-export"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--trend-week", type=int, default=103)
    parser.add_argument("--figure-format", default="svg,png")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_paper_assets_export(
            PaperAssetsExportConfig(
                case_count=args.case_count,
                top_k=args.top_k,
                trend_week=args.trend_week,
                figure_formats=_parse_figure_formats(args.figure_format),
                output_dir=args.output_dir,
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1
    log.info(
        f"论文素材导出完成: figures={payload['figure_count']}, "
        f"tables={payload['table_count']}, cases={payload['case_count']}",
        source=LOG_SOURCE,
    )
    return 0


def _parse_figure_formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {"svg", "png"}
    if not formats or not set(formats).issubset(allowed):
        raise ValueError(f"figure-format 只支持 svg,png: {value}")
    return formats


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```sh
uv run pytest tests/test_reports_runner.py::test_export_paper_assets_cli_passes_args -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```sh
git diff -- src/17_export_paper_assets.py tests/test_reports_runner.py
git diff --check
git add src/17_export_paper_assets.py tests/test_reports_runner.py
git commit -m "feat(reports): 添加论文素材导出入口"
```

---

### Task 9: Full Export Integration

**Files:**
- Modify: `src/fashion_trend/reports/runner.py`
- Modify: `src/fashion_trend/reports/figures.py`
- Modify: `src/fashion_trend/reports/tables.py`
- Test: `tests/test_reports_runner.py`

- [ ] **Step 1: Add integration test with monkeypatched small data**

Append to `tests/test_reports_runner.py`:

```python
from pathlib import Path

import pandas as pd


def test_run_paper_assets_export_writes_non_empty_manifest(tmp_path, monkeypatch) -> None:
    from fashion_trend.reports import runner

    monkeypatch.setattr(runner, "OUTPUT_REPORTS_MANIFEST_PATH", tmp_path / "manifest.json")

    payload = runner.run_paper_assets_export(
        runner.PaperAssetsExportConfig(case_count=3, top_k=10, trend_week=103)
    )

    assert payload["schema_version"] == "paper_assets_manifest/v1"
    assert (tmp_path / "manifest.json").exists()
```

This test is intentionally shallow until all real artifact reads are wired. After wiring real reads, replace monkeypatching with fixture path injection if the runner accepts path config.

- [ ] **Step 2: Extend figures module with remaining chart builders**

Add these functions to `src/fashion_trend/reports/figures.py`:

```python
def build_topk_trend_attributes_figure(
    trend_view: pd.DataFrame,
    *,
    week_id: int,
    top_k: int,
) -> Figure:
    filtered = trend_view.loc[
        (trend_view["split"].astype(str) == "test")
        & (trend_view["week_id"].astype(int) == int(week_id))
        & (trend_view["is_trend_eligible_t"].astype(int) == 1)
        & (trend_view["heat_t"].astype(float) >= 20)
        & (trend_view["history_total_heat_t"].astype(float) >= 100)
        & (trend_view["history_active_weeks_t"].astype(float) >= 8)
    ].copy()
    target_types = ("colour_group_name", "product_type_name", "graphical_appearance_name")
    chart_data = (
        filtered.loc[filtered["attr_type"].isin(target_types)]
        .sort_values(["attr_type", "pred_target_growth"], ascending=[True, False])
        .groupby("attr_type")
        .head(top_k)
    )
    if chart_data.empty:
        raise ValueError("Top-K 趋势属性图没有可绘制数据。")
    figure, axes = plt.subplots(1, len(target_types), figsize=(13, 4.8), sharex=False)
    for axis, attr_type in zip(axes, target_types):
        subset = chart_data.loc[chart_data["attr_type"] == attr_type].sort_values(
            "pred_target_growth"
        )
        axis.barh(subset["attr_value"], subset["pred_target_growth"])
        axis.set_title(attr_type)
        axis.set_xlabel("pred_target_growth")
    figure.suptitle(f"test week {week_id} Top-K 趋势属性")
    figure.tight_layout()
    return figure


def build_recommendation_weight_analysis_figure(search_results: pd.DataFrame) -> Figure:
    dataframe = search_results.copy()
    if dataframe.empty:
        raise ValueError("推荐权重分析图没有可绘制数据。")
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for trend_score, group in dataframe.groupby("trend_score"):
        axis.scatter(group["trend_score"], group["ndcg_at_12"], label=f"trend={trend_score}")
    axis.set_title("trend_score 权重与 valid NDCG@12")
    axis.set_xlabel("trend_score")
    axis.set_ylabel("valid NDCG@12")
    figure.tight_layout()
    return figure
```

Also add deterministic schematic builders for `data_pipeline` and `attribute_graph_schema` using `matplotlib.patches.FancyBboxPatch` and arrows:

```python
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def build_data_pipeline_figure() -> Figure:
    labels = [
        "H&M articles.csv",
        "属性图",
        "属性周热度",
        "LightGBM 趋势预测",
        "Top-N 推荐",
        "论文图表与案例",
    ]
    figure, axis = plt.subplots(figsize=(12, 2.8))
    axis.axis("off")
    x_positions = [0.02, 0.20, 0.38, 0.56, 0.74, 0.90]
    for index, (label, x_pos) in enumerate(zip(labels, x_positions)):
        box = FancyBboxPatch(
            (x_pos, 0.42),
            0.13,
            0.22,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            edgecolor="#334155",
            facecolor="#e0f2fe",
            transform=axis.transAxes,
        )
        axis.add_patch(box)
        axis.text(
            x_pos + 0.065,
            0.53,
            label,
            ha="center",
            va="center",
            fontsize=9,
            transform=axis.transAxes,
        )
        if index < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x_pos + 0.13, 0.53),
                (x_positions[index + 1], 0.53),
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.0,
                color="#334155",
                transform=axis.transAxes,
            )
            axis.add_patch(arrow)
    axis.set_title("数据处理与论文素材导出流程")
    return figure


def build_attribute_graph_schema_figure() -> Figure:
    nodes = {
        "商品节点": (0.15, 0.55),
        "属性节点": (0.48, 0.70),
        "父属性": (0.78, 0.78),
        "子属性": (0.78, 0.42),
    }
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    axis.axis("off")
    for label, (x_pos, y_pos) in nodes.items():
        box = FancyBboxPatch(
            (x_pos - 0.09, y_pos - 0.06),
            0.18,
            0.12,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            edgecolor="#475569",
            facecolor="#f8fafc",
            transform=axis.transAxes,
        )
        axis.add_patch(box)
        axis.text(x_pos, y_pos, label, ha="center", va="center", transform=axis.transAxes)
    for start, end, label in [
        ("商品节点", "属性节点", "article_has_attribute"),
        ("父属性", "子属性", "parent_contains_child"),
        ("属性节点", "父属性", "belongs_to_parent"),
    ]:
        start_xy = nodes[start]
        end_xy = nodes[end]
        axis.add_patch(
            FancyArrowPatch(
                start_xy,
                end_xy,
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.0,
                color="#475569",
                transform=axis.transAxes,
            )
        )
        axis.text(
            (start_xy[0] + end_xy[0]) / 2,
            (start_xy[1] + end_xy[1]) / 2 + 0.04,
            label,
            ha="center",
            fontsize=8,
            transform=axis.transAxes,
        )
    axis.set_title("商品属性层次图示意")
    return figure
```

- [ ] **Step 3: Wire runner to real artifacts**

Modify `src/fashion_trend/reports/runner.py` to:

- Configure matplotlib once at the beginning.
- Read trend metrics for `last_week`, `previous_growth`, `moving_average`, `lightgbm`.
- Read recommendation metrics for all five methods.
- Build and write `trend_model_metrics` and `recommendation_method_metrics`.
- Read feature importance and export `lightgbm_feature_importance`.
- Read LightGBM predictions and trend samples, build join view, export `topk_trend_attributes`.
- Read experiment JSON, export `recommendation_weight_analysis` and table rows.
- Select and write 3 case studies.
- Build manifest with real inputs/outputs/warnings.

Keep the function small enough by adding private helpers with concrete return contracts:

```python
def _write_metric_tables(
    trend_metric_rows: list[dict[str, object]],
    recommendation_metric_rows: list[dict[str, object]],
) -> tuple[list[str], dict[str, int]]:
    trend_table = build_trend_model_metrics_table(trend_metric_rows)
    recommendation_table = build_recommendation_method_metrics_table(
        recommendation_metric_rows
    )
    output_paths: list[str] = []
    row_counts: dict[str, int] = {}
    for name, table, columns in [
        ("trend_model_metrics", trend_table, TREND_MODEL_METRICS_COLUMNS),
        (
            "recommendation_method_metrics",
            recommendation_table,
            RECOMMENDATION_METHOD_METRICS_COLUMNS,
        ),
    ]:
        written = write_report_table(
            table,
            columns=columns,
            output_paths=table_output_paths(name),
        )
        output_paths.extend(str(path) for path in written)
        row_counts[name] = len(table)
    return output_paths, row_counts


def _write_figures(
    trend_metrics: pd.DataFrame,
    recommendation_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    trend_view: pd.DataFrame,
    search_results: pd.DataFrame,
    *,
    trend_week: int,
    top_k: int,
) -> list[str]:
    figure_builders = {
        "data_pipeline": build_data_pipeline_figure(),
        "attribute_graph_schema": build_attribute_graph_schema_figure(),
        "lightgbm_feature_importance": build_feature_importance_figure(
            feature_importance,
            top_n=15,
        ),
        "trend_model_metrics": build_trend_model_metrics_figure(trend_metrics),
        "recommendation_method_metrics": build_recommendation_method_metrics_figure(
            recommendation_metrics
        ),
        "topk_trend_attributes": build_topk_trend_attributes_figure(
            trend_view,
            week_id=trend_week,
            top_k=top_k,
        ),
        "recommendation_weight_analysis": build_recommendation_weight_analysis_figure(
            search_results
        ),
    }
    output_paths: list[str] = []
    for name, figure in figure_builders.items():
        written = save_report_figure(figure, figure_output_paths(name))
        output_paths.extend(str(path) for path in written)
    return output_paths


def _write_cases(
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    user_profile: pd.DataFrame,
    *,
    case_count: int,
) -> tuple[list[str], list[str]]:
    case_keys = select_recommendation_cases(
        recommendation_items=recommendation_items,
        evaluation_labels=evaluation_labels,
        user_profile=user_profile,
        case_count=case_count,
    )
    output_paths: list[str] = []
    case_user_ids: list[str] = []
    for index, case_key in enumerate(case_keys, start=1):
        payload = build_case_payload(
            case_key=case_key,
            recommendation_items=recommendation_items,
            evaluation_labels=evaluation_labels,
            user_profile=user_profile,
        )
        case_id = f"case_{index:02d}"
        paths = case_study_output_paths(case_id)
        write_json_atomic(payload, paths["json"])
        write_text_atomic(render_case_markdown(payload), paths["markdown"])
        output_paths.extend(str(path) for path in paths.values())
        case_user_ids.append(str(payload["customer_id"]))
    return output_paths, case_user_ids
```

Each helper should return paths as strings for manifest recording.

- [ ] **Step 4: Run focused tests**

Run:

```sh
uv run pytest tests/test_reports_*.py tests/test_architecture_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```sh
git diff -- src/fashion_trend/reports tests/test_reports_*.py
git diff --check
git add src/fashion_trend/reports tests/test_reports_*.py
git commit -m "feat(reports): 编排论文素材导出"
```

---

### Task 10: Documentation and Real Artifact Validation

**Files:**
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`
- Modify: `docs/gpt-research/project-status-summary.md`

- [ ] **Step 1: Update README reports section**

In `README.md`, update the reports description to include:

```markdown
### 17. 论文图表与案例导出

reports 阶段只读取已经发布的稳定数据、模型和推荐 artifact，不重新训练模型，也不重跑推荐方法。

```sh
uv run python src/17_export_paper_assets.py
```

默认输出：

```text
outputs/reports/figures/*.svg
outputs/reports/figures/*.png
outputs/reports/tables/*.csv
outputs/reports/tables/*.md
outputs/reports/case_studies/*.json
outputs/reports/case_studies/*.md
outputs/reports/manifest.json
```

本阶段新增 `matplotlib` 用于静态 SVG/PNG 图表导出；Markdown 表格由项目内 writer 生成，不依赖 `tabulate`。
```

- [ ] **Step 2: Update implementation plan**

In `docs/gpt-research/implementation-plan.md`, update the reports tree and command examples to include `src/17_export_paper_assets.py` and the final reports output paths. Preserve the existing caution that reports are paper/export artifacts, not an online dashboard.

- [ ] **Step 3: Update project status summary**

In `docs/gpt-research/project-status-summary.md`, add a short note under the remaining paper work section:

```markdown
报告导出阶段已设计并实现为 `src/17_export_paper_assets.py`。最终提交前运行该命令生成 `outputs/reports/` 下的 figures、tables、case_studies 和 manifest。
```

If the implementation has already been validated against real artifacts in this task, include the actual generated file counts. Do not commit generated report outputs.

- [ ] **Step 4: Run real report export**

Run:

```sh
uv run python src/17_export_paper_assets.py
```

Expected:

```text
论文素材导出完成: figures=16, tables>=16, cases=3
```

Exact table count depends on whether each CSV and Markdown path is counted separately. Manifest must record file paths and warnings.

- [ ] **Step 5: Verify output files**

Run:

```sh
test -f outputs/reports/manifest.json
find outputs/reports/figures -type f | sort
find outputs/reports/tables -type f | sort
find outputs/reports/case_studies -type f | sort
```

Expected:

- 8 SVG files and 8 PNG files under `outputs/reports/figures/`.
- CSV and Markdown files under `outputs/reports/tables/`.
- 3 JSON files and 3 Markdown files under `outputs/reports/case_studies/`.
- No generated reports are staged for git.

- [ ] **Step 6: Run final verification**

Run:

```sh
uv run pytest tests/test_reports_*.py tests/test_architecture_boundaries.py
uv run python -m compileall -q src
uv run black --check src tests
uv run isort --check-only src tests
git diff --check
```

Expected: all pass. If `uv run python src/17_export_paper_assets.py` fails because the local machine lacks CJK fonts, keep the fail-fast behavior and report the exact font error; do not weaken it silently.

- [ ] **Step 7: Commit docs**

Run:

```sh
git diff -- README.md docs/gpt-research/implementation-plan.md docs/gpt-research/project-status-summary.md
git status --short
git add README.md docs/gpt-research/implementation-plan.md docs/gpt-research/project-status-summary.md
git commit -m "docs: 说明论文素材导出流程"
```

Before committing, ensure `git status --short` does not show staged files under `outputs/reports/`.

---

## Self-Review Checklist

- Spec coverage:
  - Reports layer and `src/17_export_paper_assets.py`: Tasks 7-9.
  - Matplotlib dependency and CJK font behavior: Tasks 1 and 5.
  - No `tabulate` dependency and custom Markdown writer: Task 2.
  - Predictions + trend samples 1:1 join contract: Task 3.
  - Table column contracts: Task 4.
  - Eight figure outputs: Tasks 5 and 9.
  - Three recommendation cases: Task 6 and Task 9.
  - Manifest: Task 7 and Task 9.
  - Docs and real validation: Task 10.
- Architecture boundaries:
  - `reports` imports only public readers/contracts and `foundation` helpers.
  - No imports from trend training/evaluation runner, recommendation runner, candidate builders, or catalog graph builders.
- Dependency boundaries:
  - Add only `matplotlib`.
  - Do not use `DataFrame.to_markdown()` or `tabulate`.
- Generated artifacts:
  - Do not commit `outputs/reports/`.
