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

Expected: command succeeds and `uv.lock` includes `matplotlib` and its transitive dependencies. Do not directly declare `tabulate`, `seaborn`, `plotly`, `altair`, `networkx`, or `Pillow` unless the spec is explicitly changed. `Pillow` may appear in `uv.lock` as a `matplotlib` transitive dependency; that is expected, but reports code must not directly import or use `PIL`.

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

If `tabulate True` appears only as a transitive dependency, do not use it; the reports table writer must remain dependency-free. `PIL/Pillow` import availability is not a failure condition because Matplotlib 3.10 lists Pillow as a required runtime dependency.

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
from fashion_trend.reports.paths import (
    case_study_output_paths,
    default_report_input_paths,
    figure_output_paths,
    manifest_output_path,
    table_output_paths,
)


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


def test_output_path_helpers_honor_custom_output_root(tmp_path) -> None:
    root = tmp_path / "paper-assets"

    assert figure_output_paths("chart", output_root=root)["svg"] == (
        root / "figures" / "chart.svg"
    )
    assert table_output_paths("metrics", output_root=root)["csv"] == (
        root / "tables" / "metrics.csv"
    )
    assert case_study_output_paths("case_01", output_root=root)["json"] == (
        root / "case_studies" / "case_01.json"
    )
    assert manifest_output_path(root) == root / "manifest.json"


def test_default_report_input_paths_cover_core_sources() -> None:
    paths = default_report_input_paths()

    assert paths.lightgbm_predictions.as_posix().endswith(
        "outputs/models/lightgbm/predictions.csv"
    )
    assert paths.trend_metrics["lightgbm"].as_posix().endswith(
        "outputs/metrics/lightgbm/trend_metrics.json"
    )
    assert paths.recommendation_items["pop_similarity_trend"].as_posix().endswith(
        "outputs/recommendation/pop_similarity_trend/recommendation_items.parquet"
    )
    assert paths.recommendation_items_csv["pop_similarity_trend"].as_posix().endswith(
        "outputs/recommendation/pop_similarity_trend/recommendation_items.csv"
    )
    assert "trend_model_samples" in paths.data_artifacts
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

from dataclasses import dataclass
from pathlib import Path

from fashion_trend.foundation.artifacts import validate_output_parent_dirs
from fashion_trend.foundation.paths import INTERIM_DIR, OUTPUT_DIR, PROCESSED_DIR

# 报告阶段输出根目录。
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

# 报告阶段图表、表格和案例输出位置。
OUTPUT_FIGURES_DIR = OUTPUT_REPORTS_DIR / "figures"
OUTPUT_TABLES_DIR = OUTPUT_REPORTS_DIR / "tables"
OUTPUT_CASE_STUDIES_DIR = OUTPUT_REPORTS_DIR / "case_studies"
OUTPUT_REPORTS_MANIFEST_PATH = OUTPUT_REPORTS_DIR / "manifest.json"


@dataclass(frozen=True)
class ReportInputPaths:
    trend_metrics: dict[str, Path]
    recommendation_metrics: dict[str, Path]
    recommendation_items: dict[str, Path]
    recommendation_items_csv: dict[str, Path]
    lightgbm_predictions: Path
    lightgbm_feature_importance: Path
    trend_model_samples: Path
    trend_split_samples: dict[str, Path]
    recommendation_experiment: Path
    time_windows: Path
    target_users: Path
    evaluation_labels: Path
    user_profile: Path
    article_attributes: Path
    graph_artifacts: dict[str, Path]
    data_artifacts: dict[str, Path]


def reports_output_dir(output_root: Path | None = None) -> Path:
    """返回本次报告导出的输出根目录。"""
    return output_root if output_root is not None else OUTPUT_REPORTS_DIR


def default_report_input_paths() -> ReportInputPaths:
    """Return read-only default inputs for the reports stage.

    These defaults live in reports so the runner does not import upstream path
    modules such as trend.paths, recommendation.paths, or catalog.paths.
    """
    graph_dir = PROCESSED_DIR / "graph"
    trend_dir = PROCESSED_DIR / "trend"
    features_dir = PROCESSED_DIR / "features"
    recommend_dir = PROCESSED_DIR / "recommend"
    output_models_dir = OUTPUT_DIR / "models"
    output_metrics_dir = OUTPUT_DIR / "metrics"
    output_recommendation_dir = OUTPUT_DIR / "recommendation"

    trend_models = ("last_week", "previous_growth", "moving_average", "lightgbm")
    recommendation_methods = (
        "global_popularity",
        "recent_popularity",
        "attribute_similarity",
        "pop_similarity",
        "pop_similarity_trend",
    )
    graph_artifacts = {
        "nodes_article": graph_dir / "nodes_article.csv",
        "nodes_attribute": graph_dir / "nodes_attribute.csv",
        "edges_article_attribute": graph_dir / "edges_article_attribute.csv",
        "edges_attribute_hierarchy": graph_dir / "edges_attribute_hierarchy.csv",
    }
    trend_split_samples = {
        "train": features_dir / "trend_model_samples_train.parquet",
        "valid": features_dir / "trend_model_samples_valid.parquet",
        "test": features_dir / "trend_model_samples_test.parquet",
    }
    data_artifacts = {
        "articles_clean": INTERIM_DIR / "articles_clean.csv",
        **graph_artifacts,
        "article_week_sales": trend_dir / "article_week_sales.csv",
        "attribute_week_heat": trend_dir / "attribute_week_heat.csv",
        "attribute_week_target": trend_dir / "attribute_week_target.csv",
        "trend_model_samples": features_dir / "trend_model_samples.parquet",
        "time_windows": recommend_dir / "time_windows.parquet",
        "target_users": recommend_dir / "target_users.parquet",
        "evaluation_labels": recommend_dir / "evaluation_labels.parquet",
        "user_profile": recommend_dir / "user_profile.parquet",
    }
    return ReportInputPaths(
        trend_metrics={
            model_name: output_metrics_dir / model_name / "trend_metrics.json"
            for model_name in trend_models
        },
        recommendation_metrics={
            method: output_recommendation_dir / method / "metrics.json"
            for method in recommendation_methods
        },
        recommendation_items={
            method: output_recommendation_dir / method / "recommendation_items.parquet"
            for method in recommendation_methods
        },
        recommendation_items_csv={
            method: output_recommendation_dir / method / "recommendation_items.csv"
            for method in recommendation_methods
        },
        lightgbm_predictions=output_models_dir / "lightgbm" / "predictions.csv",
        lightgbm_feature_importance=(
            output_models_dir / "lightgbm" / "feature_importance.csv"
        ),
        trend_model_samples=features_dir / "trend_model_samples.parquet",
        trend_split_samples=trend_split_samples,
        recommendation_experiment=(
            output_recommendation_dir / "experiments" / "main" / "experiment.json"
        ),
        time_windows=recommend_dir / "time_windows.parquet",
        target_users=recommend_dir / "target_users.parquet",
        evaluation_labels=recommend_dir / "evaluation_labels.parquet",
        user_profile=recommend_dir / "user_profile.parquet",
        article_attributes=graph_artifacts["edges_article_attribute"],
        graph_artifacts=graph_artifacts,
        data_artifacts=data_artifacts,
    )


def figure_output_paths(
    name: str,
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """返回同一图表的 SVG 和 PNG 输出路径。"""
    _validate_report_artifact_name(name)
    root = reports_output_dir(output_root)
    return {
        "svg": root / "figures" / f"{name}.svg",
        "png": root / "figures" / f"{name}.png",
    }


def table_output_paths(
    name: str,
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """返回同一表格的 CSV 和 Markdown 输出路径。"""
    _validate_report_artifact_name(name)
    root = reports_output_dir(output_root)
    return {
        "csv": root / "tables" / f"{name}.csv",
        "markdown": root / "tables" / f"{name}.md",
    }


def case_study_output_paths(
    case_id: str,
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """返回单个案例的 JSON 和 Markdown 输出路径。"""
    _validate_report_artifact_name(case_id)
    root = reports_output_dir(output_root)
    return {
        "json": root / "case_studies" / f"{case_id}.json",
        "markdown": root / "case_studies" / f"{case_id}.md",
    }


def manifest_output_path(output_root: Path | None = None) -> Path:
    """返回本次报告 manifest 输出路径。"""
    return reports_output_dir(output_root) / "manifest.json"


def validate_report_output_path(
    path: Path,
    *,
    output_root: Path | None = None,
) -> None:
    """确认报告产物仍写在本次 output root 内。"""
    validate_output_parent_dirs(path.parent, reports_output_dir(output_root))


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
    flatten_trend_metrics_by_attr_type,
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


def test_flatten_trend_metrics_by_attr_type_extracts_design_columns() -> None:
    payload = {
        "model_name": "lightgbm",
        "by_attr_type": {
            "test": {
                "colour_group_name": {
                    "mae": 0.11,
                    "rmse": 0.21,
                    "spearman": 0.31,
                    "ndcg_at_k": {"10": 0.41},
                    "precision_at_k": {"10": 0.51},
                    "recall_at_k": {"10": 0.61},
                }
            }
        },
    }

    rows = flatten_trend_metrics_by_attr_type(payload)

    assert rows == [
        {
            "model_name": "lightgbm",
            "split": "test",
            "attr_type": "colour_group_name",
            "mae": 0.11,
            "rmse": 0.21,
            "spearman": 0.31,
            "ndcg_at_10": 0.41,
            "precision_at_10": 0.51,
            "recall_at_10": 0.61,
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
    required = {
        "feature",
        "split_importance",
        "gain_importance",
        "normalized_gain_importance",
    }
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


def flatten_trend_metrics_by_attr_type(payload: dict[str, Any]) -> list[dict[str, Any]]:
    model_name = str(payload.get("model_name"))
    rows: list[dict[str, Any]] = []
    for split, attr_type_metrics in sorted(
        _required_dict(payload, "by_attr_type").items()
    ):
        attr_payload = _as_dict(attr_type_metrics, f"by_attr_type.{split}")
        for attr_type, metrics in sorted(attr_payload.items()):
            metric_payload = _as_dict(metrics, f"by_attr_type.{split}.{attr_type}")
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "attr_type": attr_type,
                    "mae": _finite_number(metric_payload, "mae"),
                    "rmse": _finite_number(metric_payload, "rmse"),
                    "spearman": _finite_number(metric_payload, "spearman"),
                    "ndcg_at_10": _metric_at_k(metric_payload, "ndcg_at_k", "10"),
                    "precision_at_10": _metric_at_k(
                        metric_payload,
                        "precision_at_k",
                        "10",
                    ),
                    "recall_at_10": _metric_at_k(metric_payload, "recall_at_k", "10"),
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
    REPORT_TABLE_COLUMNS,
    REPORT_TABLE_SORT_COLUMNS,
    RECOMMENDATION_METHOD_METRICS_COLUMNS,
    TREND_MODEL_METRICS_COLUMNS,
    build_report_table,
    build_recommendation_method_metrics_table,
    build_trend_model_metrics_table,
    write_report_table,
)


def test_report_table_contracts_cover_design_outputs() -> None:
    assert set(REPORT_TABLE_COLUMNS) == {
        "data_artifact_summary",
        "time_split_summary",
        "attribute_graph_summary",
        "trend_feature_summary",
        "trend_model_metrics",
        "trend_metrics_by_attr_type",
        "recommendation_method_metrics",
        "recommendation_experiment_summary",
    }
    assert set(REPORT_TABLE_SORT_COLUMNS) == set(REPORT_TABLE_COLUMNS)


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


def test_build_report_table_selects_each_design_contract() -> None:
    samples = {
        "data_artifact_summary": {
            "section": "trend",
            "artifact": "trend_model_samples",
            "path": "data/processed/features/trend_model_samples.parquet",
            "row_count": 59200,
            "column_count": 22,
            "paper_usage": "趋势模型样本规模说明",
        },
        "time_split_summary": {
            "domain": "trend",
            "split": "test",
            "week_start": 96,
            "week_end": 104,
            "week_count": 8,
            "row_count": 5920,
            "attribute_count": 740,
            "user_count": 0,
        },
        "attribute_graph_summary": {
            "entity_type": "article",
            "attr_type": "",
            "relation_type": "article_attribute",
            "count": 105542,
            "path": "data/processed/graph/edges_article_attribute.csv",
            "paper_usage": "属性图规模说明",
        },
        "trend_feature_summary": {
            "feature_group": "lag",
            "feature_name": "lag_1_heat",
            "source_table": "trend_model_samples",
            "model_input": True,
            "description": "上一周属性热度",
        },
        "trend_model_metrics": {
            "model_name": "lightgbm",
            "split": "test",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
            "run_id": "run-1",
        },
        "trend_metrics_by_attr_type": {
            "model_name": "lightgbm",
            "split": "test",
            "attr_type": "colour_group_name",
            "mae": 0.1,
            "rmse": 0.2,
            "spearman": 0.3,
            "ndcg_at_10": 0.4,
            "precision_at_10": 0.5,
            "recall_at_10": 0.6,
        },
        "recommendation_method_metrics": {
            "method": "pop_similarity_trend",
            "split": "test",
            "map_at_12": 0.1,
            "recall_at_12": 0.2,
            "hit_rate_at_12": 0.3,
            "ndcg_at_12": 0.4,
            "coverage": 0.5,
            "user_count": 10,
            "missing_recommendation_user_count": 0,
        },
        "recommendation_experiment_summary": {
            "section": "search_results",
            "rank": 1,
            "method": "pop_similarity_trend",
            "split": "valid",
            "pop_score": 0.2,
            "sim_score": 0.2,
            "trend_score": 0.1,
            "recent_score": 0.5,
            "map_at_12": 0.1,
            "recall_at_12": 0.2,
            "hit_rate_at_12": 0.3,
            "ndcg_at_12": 0.4,
            "coverage": 0.5,
        },
    }

    for table_name, row in samples.items():
        table = build_report_table([row], table_name=table_name)
        assert tuple(table.columns) == REPORT_TABLE_COLUMNS[table_name]


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

DATA_ARTIFACT_SUMMARY_COLUMNS = (
    "section",
    "artifact",
    "path",
    "row_count",
    "column_count",
    "paper_usage",
)
TIME_SPLIT_SUMMARY_COLUMNS = (
    "domain",
    "split",
    "week_start",
    "week_end",
    "week_count",
    "row_count",
    "attribute_count",
    "user_count",
)
ATTRIBUTE_GRAPH_SUMMARY_COLUMNS = (
    "entity_type",
    "attr_type",
    "relation_type",
    "count",
    "path",
    "paper_usage",
)
TREND_FEATURE_SUMMARY_COLUMNS = (
    "feature_group",
    "feature_name",
    "source_table",
    "model_input",
    "description",
)
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
TREND_METRICS_BY_ATTR_TYPE_COLUMNS = (
    "model_name",
    "split",
    "attr_type",
    "mae",
    "rmse",
    "spearman",
    "ndcg_at_10",
    "precision_at_10",
    "recall_at_10",
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
RECOMMENDATION_EXPERIMENT_SUMMARY_COLUMNS = (
    "section",
    "rank",
    "method",
    "split",
    "pop_score",
    "sim_score",
    "trend_score",
    "recent_score",
    "map_at_12",
    "recall_at_12",
    "hit_rate_at_12",
    "ndcg_at_12",
    "coverage",
)

REPORT_TABLE_COLUMNS = {
    "data_artifact_summary": DATA_ARTIFACT_SUMMARY_COLUMNS,
    "time_split_summary": TIME_SPLIT_SUMMARY_COLUMNS,
    "attribute_graph_summary": ATTRIBUTE_GRAPH_SUMMARY_COLUMNS,
    "trend_feature_summary": TREND_FEATURE_SUMMARY_COLUMNS,
    "trend_model_metrics": TREND_MODEL_METRICS_COLUMNS,
    "trend_metrics_by_attr_type": TREND_METRICS_BY_ATTR_TYPE_COLUMNS,
    "recommendation_method_metrics": RECOMMENDATION_METHOD_METRICS_COLUMNS,
    "recommendation_experiment_summary": RECOMMENDATION_EXPERIMENT_SUMMARY_COLUMNS,
}
REPORT_TABLE_SORT_COLUMNS = {
    "data_artifact_summary": ("section", "artifact"),
    "time_split_summary": ("domain", "split", "week_start"),
    "attribute_graph_summary": ("entity_type", "attr_type", "relation_type"),
    "trend_feature_summary": ("feature_group", "feature_name"),
    "trend_model_metrics": ("model_name", "split"),
    "trend_metrics_by_attr_type": ("model_name", "split", "attr_type"),
    "recommendation_method_metrics": ("method", "split"),
    "recommendation_experiment_summary": ("section", "rank", "split", "method"),
}


def build_report_table(rows: list[dict[str, Any]], *, table_name: str) -> pd.DataFrame:
    if table_name not in REPORT_TABLE_COLUMNS:
        raise ValueError(f"未知报告表格: {table_name}")
    dataframe = pd.DataFrame(rows)
    return _select_and_sort(
        dataframe,
        columns=REPORT_TABLE_COLUMNS[table_name],
        sort_columns=REPORT_TABLE_SORT_COLUMNS[table_name],
        table_name=table_name,
    )


def build_trend_model_metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return build_report_table(rows, table_name="trend_model_metrics")


def build_recommendation_method_metrics_table(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    return build_report_table(rows, table_name="recommendation_method_metrics")


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
import pytest

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


@pytest.mark.parametrize(
    ("formats", "expected_suffixes", "unexpected_suffixes"),
    [
        (("svg",), ("svg",), ("png",)),
        (("png",), ("png",), ("svg",)),
        (("svg", "png"), ("svg", "png"), ()),
    ],
)
def test_save_report_figure_honors_requested_formats(
    tmp_path,
    monkeypatch,
    formats,
    expected_suffixes,
    unexpected_suffixes,
) -> None:
    monkeypatch.setattr(
        "fashion_trend.reports.plotting.available_cjk_fonts",
        lambda: ["DejaVu Sans"],
    )
    configure_matplotlib_for_reports()
    figure, axis = plt.subplots()
    axis.set_title("中文标题 NDCG@12")
    axis.plot([-1, 0, 1], [1, 0, -1])
    paths = {"svg": tmp_path / "figure.svg", "png": tmp_path / "figure.png"}

    written = save_report_figure(figure, paths, formats=formats)

    assert written == [paths[suffix] for suffix in expected_suffixes]
    for suffix in expected_suffixes:
        assert paths[suffix].stat().st_size > 0
    for suffix in unexpected_suffixes:
        assert not paths[suffix].exists()


def test_save_report_figure_rejects_duplicate_formats(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "fashion_trend.reports.plotting.available_cjk_fonts",
        lambda: ["DejaVu Sans"],
    )
    configure_matplotlib_for_reports()
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    paths = {"svg": tmp_path / "figure.svg", "png": tmp_path / "figure.png"}

    try:
        with pytest.raises(ValueError, match="不能重复"):
            save_report_figure(figure, paths, formats=("svg", "svg"))
    finally:
        plt.close(figure)
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


def save_report_figure(
    figure: Figure,
    output_paths: dict[str, Path],
    *,
    formats: tuple[str, ...] = ("svg", "png"),
) -> list[Path]:
    allowed_formats = {"svg", "png"}
    unknown_formats = sorted(set(formats) - allowed_formats)
    if not formats or unknown_formats:
        raise ValueError(f"figure formats 只支持 svg,png: {formats}")
    if len(set(formats)) != len(formats):
        raise ValueError(f"figure formats 不能重复: {formats}")
    missing_paths = sorted(set(formats) - set(output_paths))
    if missing_paths:
        raise ValueError(f"图表输出路径缺少格式: {missing_paths}")

    written: list[Path] = []
    for suffix in formats:
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

Create `src/fashion_trend/reports/figures.py` with the first four reusable chart functions:

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
        feature_importance.sort_values("normalized_gain_importance", ascending=False)
        .head(top_n)
        .sort_values("normalized_gain_importance")
    )
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(top_features["feature"], top_features["normalized_gain_importance"])
    axis.set_title("LightGBM 特征重要性 Top-N")
    axis.set_xlabel("normalized gain")
    axis.set_ylabel("feature")
    figure.tight_layout()
    return figure


def build_trend_curve_examples_figure(
    trend_view: pd.DataFrame,
    *,
    week_id: int,
    lookback_weeks: int = 8,
    top_n: int = 3,
) -> Figure:
    current = trend_view.loc[
        (trend_view["week_id"].astype(int) == week_id)
        & (trend_view["is_trend_eligible_t"].astype(bool))
    ].copy()
    examples = (
        current.sort_values("pred_target_growth", ascending=False)
        .drop_duplicates("attr_type")
        .head(top_n)
    )
    if examples.empty:
        raise ValueError(f"week_id={week_id} 没有可绘制的趋势曲线案例。")

    figure, axes = plt.subplots(
        len(examples),
        1,
        figsize=(9, 2.7 * len(examples)),
        sharex=True,
    )
    if len(examples) == 1:
        axes = [axes]

    min_week = week_id - lookback_weeks + 1
    for axis, row in zip(axes, examples.itertuples(index=False)):
        history = trend_view.loc[
            (trend_view["attr_id"].astype(str) == str(row.attr_id))
            & (trend_view["attr_type"].astype(str) == str(row.attr_type))
            & (trend_view["attr_value"].astype(str) == str(row.attr_value))
            & (trend_view["week_id"].astype(int).between(min_week, week_id))
        ].sort_values("week_id")
        axis.plot(history["week_id"], history["heat_t"], marker="o", label="heat_t")
        axis.plot(
            history["week_id"],
            history["pred_share_t1"],
            marker="s",
            label="pred_share_t1",
        )
        axis.bar(
            history["week_id"],
            history["pred_target_growth"],
            alpha=0.25,
            label="pred_target_growth",
        )
        axis.set_title(f"{row.attr_type}: {row.attr_value}")
        axis.set_ylabel("value")
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("week_id")
    figure.suptitle("典型趋势属性最近 8 周曲线")
    figure.tight_layout()
    return figure
```

Task 9 will call these functions from the runner and add the remaining four figure builders in the same module.

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
    render_case_markdown,
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


def _article_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": "0000000001",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
            },
            {
                "article_id": "0000000001",
                "attr_type": "product_group_name",
                "attr_value": "Garment Upper body",
            },
        ]
    )


def _representative_trends() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "week_id": 103,
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "pred_target_growth": 0.31,
                "heat_t": 1200.0,
            }
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


def test_select_recommendation_cases_requires_clear_preference_attr_type() -> None:
    profiles = _profiles()
    profiles.loc[profiles["customer_id"] == "customer-a", "attr_type"] = "department_name"
    profiles.loc[profiles["customer_id"] == "customer-a", "attr_value"] = "Divided"

    selected = select_recommendation_cases(
        recommendation_items=_items(),
        evaluation_labels=_labels(),
        user_profile=profiles,
        case_count=1,
    )

    assert selected == [("customer-b", "test", 103, 104)]


def test_build_case_payload_marks_hits() -> None:
    payload = build_case_payload(
        case_key=("customer-a", "test", 103, 104),
        recommendation_items=_items(),
        evaluation_labels=_labels(),
        user_profile=_profiles(),
        article_attributes=_article_attributes(),
        representative_trends=_representative_trends(),
    )

    assert payload["customer_id"] == "customer-a"
    assert payload["hit_count"] == 1
    assert payload["recommendations"][0]["is_hit"] is True
    assert payload["recommendations"][0]["attributes"]["colour_group_name"] == "Black"
    assert payload["profile"][0]["attr_value"] == "Black"
    assert payload["representative_trends"][0]["attr_value"] == "Black"


def test_build_case_payload_uses_case_cutoff_week_trends() -> None:
    items = _items().assign(cutoff_week=102, label_week=103)
    labels = _labels().assign(cutoff_week=102, label_week=103)
    profiles = _profiles().assign(cutoff_week=102, label_week=103)
    representative_trends = pd.DataFrame(
        [
            {
                "week_id": 102,
                "attr_type": "colour_group_name",
                "attr_value": "Red",
                "pred_target_growth": 0.42,
                "heat_t": 900.0,
            },
            {
                "week_id": 103,
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "pred_target_growth": 0.31,
                "heat_t": 1200.0,
            },
        ]
    )

    payload = build_case_payload(
        case_key=("customer-a", "test", 102, 103),
        recommendation_items=items,
        evaluation_labels=labels,
        user_profile=profiles,
        article_attributes=_article_attributes(),
        representative_trends=representative_trends,
    )

    assert payload["representative_trends"][0]["attr_value"] == "Red"


def test_render_case_markdown_includes_explanations() -> None:
    payload = build_case_payload(
        case_key=("customer-a", "test", 103, 104),
        recommendation_items=_items(),
        evaluation_labels=_labels(),
        user_profile=_profiles(),
        article_attributes=_article_attributes(),
        representative_trends=_representative_trends(),
    )

    text = render_case_markdown(payload)

    assert "# 推荐案例" in text
    assert "代表性趋势属性" in text
    assert "商品属性" in text
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
CORE_ARTICLE_ATTR_TYPES = (
    "product_group_name",
    "product_type_name",
    "graphical_appearance_name",
    "colour_group_name",
    "department_name",
)
CASE_EXPLANATORY_PROFILE_ATTR_TYPES = (
    "graphical_appearance_name",
    "product_group_name",
    "colour_group_name",
)


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
    explanatory_profiles = test_profiles.loc[
        test_profiles["attr_type"].astype(str).isin(CASE_EXPLANATORY_PROFILE_ATTR_TYPES)
    ].copy()

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
        explanatory_profiles.groupby(list(CASE_KEY_COLUMNS), as_index=False)
        .size()
        .rename(columns={"size": "explanatory_profile_count"})
    )
    candidates = hit_counts.merge(profile_counts, on=list(CASE_KEY_COLUMNS), how="inner")
    candidates = candidates.loc[candidates["explanatory_profile_count"] > 0]
    candidates = candidates.sort_values(
        ["hit_count", "explanatory_profile_count", "customer_id"],
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
    article_attributes: pd.DataFrame,
    representative_trends: pd.DataFrame,
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
    attrs_by_article = _article_attributes_by_article(article_attributes)
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
                "attributes": attrs_by_article.get(article_id, {}),
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
        "representative_trends": _representative_trend_rows(
            representative_trends,
            week_id=cutoff_week,
        ),
        "recommendations": recommendations,
    }


def render_case_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# 推荐案例 {payload['customer_id']}",
        "",
        f"- split: {payload['split']}",
        f"- cutoff_week: {payload['cutoff_week']}",
        f"- label_week: {payload['label_week']}",
        f"- hit_count: {payload['hit_count']}",
        "",
        "## 用户偏好属性",
    ]
    for row in payload["profile"]:
        lines.append(
            "- {attr_type}: {attr_value} "
            "(score={preference_score:.4f}, count={purchase_count})".format(**row)
        )
    lines.extend(["", "## 代表性趋势属性"])
    for row in payload["representative_trends"]:
        lines.append(
            "- {attr_type}: {attr_value} "
            "(pred_growth={pred_target_growth:.4f}, heat_t={heat_t:.2f})".format(**row)
        )
    lines.extend(["", "## 推荐商品与解释"])
    for row in payload["recommendations"]:
        attrs = ", ".join(
            f"{name}={value}" for name, value in sorted(row["attributes"].items())
        )
        lines.append(
            "- rank {rank}: {article_id} hit={is_hit} score={score:.4f}; "
            "pop={pop_score:.4f}, sim={sim_score:.4f}, trend={trend_score:.4f}, "
            "recent={recent_score:.4f}; sources={candidate_sources}; "
            "商品属性: {attrs}".format(attrs=attrs or "未补全", **row)
        )
    return "\n".join(lines) + "\n"


def _article_attributes_by_article(
    article_attributes: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    required = {"article_id", "attr_type", "attr_value"}
    missing = sorted(required - set(article_attributes.columns))
    if missing:
        raise ValueError(f"商品属性表缺少列: {missing}")
    scoped = article_attributes.loc[
        article_attributes["attr_type"].astype(str).isin(CORE_ARTICLE_ATTR_TYPES)
    ]
    result: dict[str, dict[str, str]] = {}
    for row in scoped.itertuples(index=False):
        result.setdefault(str(row.article_id), {})[str(row.attr_type)] = str(
            row.attr_value
        )
    return result


def _representative_trend_rows(
    representative_trends: pd.DataFrame,
    *,
    week_id: int,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    required = {"week_id", "attr_type", "attr_value", "pred_target_growth", "heat_t"}
    missing = sorted(required - set(representative_trends.columns))
    if missing:
        raise ValueError(f"代表性趋势属性缺少列: {missing}")
    scoped = representative_trends.loc[
        representative_trends["week_id"].astype(int) == week_id
    ].sort_values("pred_target_growth", ascending=False)
    if scoped.empty:
        raise ValueError(f"week_id={week_id} 缺少代表性趋势属性。")
    return [
        {
            "attr_type": str(row.attr_type),
            "attr_value": str(row.attr_value),
            "pred_target_growth": float(row.pred_target_growth),
            "heat_t": float(row.heat_t),
        }
        for row in scoped.head(top_n).itertuples(index=False)
    ]


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
from fashion_trend.reports.paths import ReportInputPaths, manifest_output_path


@dataclass(frozen=True)
class PaperAssetsExportConfig:
    case_count: int = 3
    top_k: int = 10
    trend_week: int = 103
    figure_formats: tuple[str, ...] = ("svg", "png")
    output_dir: Path | None = None
    input_paths: ReportInputPaths | None = None


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
            "output_dir": str(manifest_output_path(config.output_dir).parent),
        },
        input_artifacts={},
        output_artifacts={"figures": [], "tables": [], "case_studies": []},
        row_counts={},
        case_user_ids=[],
        warnings=[],
    )
    write_manifest(payload, manifest_output_path(config.output_dir))
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
from pathlib import Path

import pytest


def test_export_paper_assets_cli_passes_args(monkeypatch) -> None:
    module = importlib.import_module("17_export_paper_assets")
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {
            "manifest_path": "outputs/reports/manifest.json",
            "figure_count": 0,
            "table_count": 0,
            "case_count": 0,
        }

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
            "--output-dir",
            "outputs/reports-paper",
        ]
    )

    assert exit_code == 0
    assert captured["config"].case_count == 2
    assert captured["config"].top_k == 5
    assert captured["config"].trend_week == 102
    assert captured["config"].figure_formats == ("svg", "png")
    assert captured["config"].output_dir == Path("outputs/reports-paper")


def test_parse_figure_formats_rejects_duplicates() -> None:
    module = importlib.import_module("17_export_paper_assets")

    with pytest.raises(ValueError, match="不能重复"):
        module._parse_figure_formats("svg,svg")
```

- [ ] **Step 2: Run failing CLI test**

Run:

```sh
uv run pytest tests/test_reports_runner.py::test_export_paper_assets_cli_passes_args tests/test_reports_runner.py::test_parse_figure_formats_rejects_duplicates -q
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
    if len(set(formats)) != len(formats):
        raise ValueError(f"figure-format 不能重复: {value}")
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
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from fashion_trend.reports.figures import build_recommendation_weight_analysis_figure
from fashion_trend.reports.tables import REPORT_TABLE_COLUMNS


def test_run_paper_assets_export_uses_monkeypatched_small_data(
    tmp_path,
    monkeypatch,
) -> None:
    from fashion_trend.reports import runner

    figure_names = (
        "data_pipeline",
        "attribute_graph_schema",
        "trend_curve_examples",
        "lightgbm_feature_importance",
        "trend_model_metrics",
        "recommendation_method_metrics",
        "topk_trend_attributes",
        "recommendation_weight_analysis",
    )
    fake_inputs = SimpleNamespace(
        input_artifacts={"predictions": "outputs/models/lightgbm/predictions.csv"},
        report_table_rows={name: [{"row": 1}] for name in REPORT_TABLE_COLUMNS},
        trend_metrics=pd.DataFrame(),
        recommendation_metrics=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        trend_view=pd.DataFrame(),
        search_results=pd.DataFrame(),
        recommendation_items=pd.DataFrame(),
        evaluation_labels=pd.DataFrame(),
        user_profile=pd.DataFrame(),
        article_attributes=pd.DataFrame(),
        representative_trends=pd.DataFrame(),
        best_weights={
            "pop_score": 0.2,
            "sim_score": 0.3,
            "trend_score": 0.4,
            "recent_score": 0.1,
        },
        warnings=[],
    )

    monkeypatch.setattr(runner, "configure_matplotlib_for_reports", lambda: "Test Font")
    monkeypatch.setattr(runner, "_load_report_inputs", lambda: fake_inputs)

    def fake_write_tables(report_table_rows, *, output_root):
        assert output_root == tmp_path
        assert set(report_table_rows) == set(REPORT_TABLE_COLUMNS)
        paths = [
            str(output_root / "tables" / f"{name}.{suffix}")
            for name in REPORT_TABLE_COLUMNS
            for suffix in ("csv", "md")
        ]
        return paths, {name: 1 for name in REPORT_TABLE_COLUMNS}

    def fake_write_figures(
        *args,
        trend_week,
        top_k,
        figure_formats,
        best_weights,
        output_root,
    ):
        assert trend_week == 103
        assert top_k == 10
        assert figure_formats == ("svg",)
        assert best_weights["trend_score"] == 0.4
        assert output_root == tmp_path
        return [
            str(output_root / "figures" / f"{name}.{suffix}")
            for name in figure_names
            for suffix in figure_formats
        ]

    def fake_write_cases(*args, case_count, output_root):
        assert case_count == 3
        assert output_root == tmp_path
        paths = [
            str(output_root / "case_studies" / f"case_{index:02d}.{suffix}")
            for index in range(1, 4)
            for suffix in ("json", "md")
        ]
        return paths, ["customer-a", "customer-b", "customer-c"]

    monkeypatch.setattr(runner, "_write_tables", fake_write_tables)
    monkeypatch.setattr(runner, "_write_figures", fake_write_figures)
    monkeypatch.setattr(runner, "_write_cases", fake_write_cases)

    payload = runner.run_paper_assets_export(
        runner.PaperAssetsExportConfig(
            case_count=3,
            top_k=10,
            trend_week=103,
            figure_formats=("svg",),
            output_dir=tmp_path,
        )
    )

    assert payload["schema_version"] == "paper_assets_manifest/v1"
    assert payload["table_count"] == 16
    assert payload["figure_count"] == 8
    assert payload["case_count"] == 3
    assert set(payload["row_counts"]) == set(REPORT_TABLE_COLUMNS)
    assert all(path.endswith(".svg") for path in payload["output_artifacts"]["figures"])
    assert (tmp_path / "manifest.json").exists()


def test_report_input_helper_rows_use_design_contracts(tmp_path) -> None:
    from fashion_trend.reports import runner

    artifact_path = tmp_path / "artifact.csv"
    pd.DataFrame([{"a": 1, "b": 2}]).to_csv(artifact_path, index=False)

    artifact_rows = runner.build_data_artifact_summary_rows(
        artifacts={"trend_samples": artifact_path},
        sections={"trend_samples": "trend"},
        paper_usage={"trend_samples": "趋势样本规模"},
    )
    assert tuple(artifact_rows[0]) == REPORT_TABLE_COLUMNS["data_artifact_summary"]
    assert artifact_rows[0]["row_count"] == 1
    assert artifact_rows[0]["column_count"] == 2

    split_rows = runner.build_time_split_summary_rows(
        split_frames={
            "test": pd.DataFrame(
                {"week_id": [101, 102], "attr_id": ["colour::black", "index::ladies"]}
            )
        },
        domain="trend",
    )
    assert tuple(split_rows[0]) == REPORT_TABLE_COLUMNS["time_split_summary"]
    assert split_rows[0]["week_start"] == 101
    assert split_rows[0]["week_count"] == 2
    assert split_rows[0]["attribute_count"] == 2

    graph_rows = runner.build_attribute_graph_summary_rows(
        graph_frames={
            "nodes_article": pd.DataFrame({"article_id": ["000000001"]}),
            "nodes_attribute": pd.DataFrame(
                {"attr_id": ["colour::black"], "attr_type": ["colour_group_name"]}
            ),
            "edges_article_attribute": pd.DataFrame(
                {"article_id": ["000000001"], "attr_type": ["colour_group_name"]}
            ),
            "edges_attribute_hierarchy": pd.DataFrame(
                {"parent_attr_type": ["product_group_name"]}
            ),
        },
        graph_paths={"edges_article_attribute": "graph/edges_article_attribute.csv"},
    )
    assert all(
        tuple(row) == REPORT_TABLE_COLUMNS["attribute_graph_summary"]
        for row in graph_rows
    )

    feature_rows = runner.build_trend_feature_summary_rows()
    assert tuple(feature_rows[0]) == REPORT_TABLE_COLUMNS["trend_feature_summary"]


def test_experiment_helpers_flatten_real_payload_shape() -> None:
    from fashion_trend.reports import runner

    payload = {
        "best_weights": {
            "pop_score": 0.20,
            "sim_score": 0.30,
            "trend_score": 0.40,
            "recent_score": 0.10,
        },
        "search_results": [
            {
                "weights": {
                    "pop_score": 0.20,
                    "sim_score": 0.30,
                    "trend_score": 0.40,
                    "recent_score": 0.10,
                },
                "valid_metrics": {
                    "map_at_12": 0.01,
                    "recall_at_12": 0.02,
                    "hit_rate_at_12": 0.03,
                    "ndcg_at_12": 0.04,
                    "coverage": 0.05,
                },
            }
        ],
        "ablation": [
            {
                "method": "pop_similarity_trend",
                "split": "test",
                "map_at_12": 0.11,
                "recall_at_12": 0.12,
                "hit_rate_at_12": 0.13,
                "ndcg_at_12": 0.14,
                "coverage": 0.15,
            },
            {
                "method": "pop_similarity",
                "split": "test",
                "map_at_12": 0.21,
                "recall_at_12": 0.22,
                "hit_rate_at_12": 0.23,
                "ndcg_at_12": 0.24,
                "coverage": 0.25,
            },
        ],
    }

    search_results = runner.flatten_experiment_search_results(payload)
    assert tuple(search_results.columns) == (
        "section",
        "rank",
        "method",
        "split",
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
        "map_at_12",
        "recall_at_12",
        "hit_rate_at_12",
        "ndcg_at_12",
        "coverage",
    )
    assert search_results.loc[0, "section"] == "search_results"
    assert search_results.loc[0, "trend_score"] == 0.40

    rows = runner.flatten_recommendation_experiment_rows(payload)
    assert [row["section"] for row in rows] == [
        "search_results",
        "ablation",
        "ablation",
    ]
    assert rows[1]["trend_score"] == 0.40
    assert rows[2]["trend_score"] == ""


def test_manifest_helpers_capture_all_inputs_and_warnings(tmp_path) -> None:
    from fashion_trend.reports import runner

    legacy_csv = tmp_path / "recommendation_items.csv"
    legacy_csv.write_text("customer_id,article_id\n", encoding="utf-8")
    input_paths = SimpleNamespace(
        data_artifacts={"trend_model_samples": tmp_path / "samples.parquet"},
        trend_split_samples={"test": tmp_path / "samples_test.parquet"},
        graph_artifacts={"nodes_article": tmp_path / "nodes_article.csv"},
        trend_metrics={"lightgbm": tmp_path / "trend_metrics.json"},
        recommendation_metrics={"pop_similarity_trend": tmp_path / "metrics.json"},
        recommendation_items={
            "pop_similarity_trend": tmp_path / "recommendation_items.parquet"
        },
        recommendation_items_csv={"pop_similarity_trend": legacy_csv},
        lightgbm_predictions=tmp_path / "predictions.csv",
        lightgbm_feature_importance=tmp_path / "feature_importance.csv",
        trend_model_samples=tmp_path / "samples.parquet",
        recommendation_experiment=tmp_path / "experiment.json",
        evaluation_labels=tmp_path / "evaluation_labels.parquet",
        user_profile=tmp_path / "user_profile.parquet",
        article_attributes=tmp_path / "edges_article_attribute.csv",
    )
    payload = {
        "search_results": [
            {
                "weights": {"trend_score": 0.4},
                "valid_metrics": {"ndcg_at_12": 0.04},
            }
        ],
        "ablation": [{"method": "pop_similarity_trend"}],
    }

    artifacts = runner._build_input_artifacts(input_paths)
    warnings = runner._build_report_warnings(payload, input_paths)

    assert artifacts["data_artifact__trend_model_samples"].endswith("samples.parquet")
    assert artifacts["trend_split_sample__test"].endswith("samples_test.parquet")
    assert artifacts["graph_artifact__nodes_article"].endswith("nodes_article.csv")
    assert artifacts["recommendation_items_csv__pop_similarity_trend"].endswith(
        "recommendation_items.csv"
    )
    assert any("grid search 只有 valid 指标" in warning for warning in warnings)
    assert any("缺少严格 w/o Recent" in warning for warning in warnings)
    assert any("recommendation_items.csv" in warning for warning in warnings)


def test_recommendation_weight_analysis_includes_best_weights() -> None:
    search_results = pd.DataFrame(
        [
            {"trend_score": 0.1, "ndcg_at_12": 0.02},
            {"trend_score": 0.4, "ndcg_at_12": 0.04},
        ]
    )
    figure = build_recommendation_weight_analysis_figure(
        search_results,
        best_weights={
            "pop_score": 0.20,
            "sim_score": 0.30,
            "trend_score": 0.40,
            "recent_score": 0.10,
        },
    )

    try:
        assert len(figure.axes) == 2
        assert "主实验权重构成" in figure.axes[1].get_title()
    finally:
        plt.close(figure)


def test_build_representative_trend_attributes_uses_top_eligible_rows() -> None:
    from fashion_trend.reports import runner

    trend_view = pd.DataFrame(
        [
            {
                "week_id": 102,
                "attr_type": "colour_group_name",
                "attr_value": "Red",
                "pred_target_growth": 0.40,
                "heat_t": 700,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": 103,
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "pred_target_growth": 0.30,
                "heat_t": 900,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": 103,
                "attr_type": "colour_group_name",
                "attr_value": "Blue",
                "pred_target_growth": 0.80,
                "heat_t": 100,
                "is_trend_eligible_t": False,
            },
            {
                "week_id": 103,
                "attr_type": "index_name",
                "attr_value": "Ladieswear",
                "pred_target_growth": 0.50,
                "heat_t": 800,
                "is_trend_eligible_t": True,
            },
        ]
    )

    rows = runner.build_representative_trend_attributes(
        trend_view,
        week_id=103,
        top_n=2,
    )

    assert tuple(rows.columns) == (
        "week_id",
        "attr_type",
        "attr_value",
        "pred_target_growth",
        "heat_t",
    )
    assert rows["attr_value"].tolist() == ["Ladieswear", "Black"]

    case_rows = runner.build_representative_trend_attributes(
        trend_view,
        week_ids=[102, 103],
        top_n=1,
    )
    assert case_rows["week_id"].tolist() == [103, 102]
    assert set(case_rows["attr_value"]) == {"Ladieswear", "Red"}
```

This test must not read `data/` or `outputs/`, must not require a real CJK font, and must not run real Matplotlib rendering. Keep real artifact validation in Task 10. Task 9 should introduce a private `_load_report_inputs()` seam that returns all DataFrames, JSON payloads, `input_artifacts`, and `report_table_rows`; this test monkeypatches that seam and the writer helpers to verify orchestration counts and argument propagation only.

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


def build_recommendation_weight_analysis_figure(
    search_results: pd.DataFrame,
    *,
    best_weights: dict[str, float],
) -> Figure:
    dataframe = search_results.copy()
    if dataframe.empty:
        raise ValueError("推荐权重分析图没有可绘制数据。")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    scatter_axis, weight_axis = axes
    for trend_score, group in dataframe.groupby("trend_score"):
        scatter_axis.scatter(
            group["trend_score"],
            group["ndcg_at_12"],
            label=f"trend={trend_score}",
        )
    scatter_axis.set_title("trend_score 权重与 valid NDCG@12")
    scatter_axis.set_xlabel("trend_score")
    scatter_axis.set_ylabel("valid NDCG@12")

    weight_names = ("pop_score", "sim_score", "trend_score", "recent_score")
    weight_values = [float(best_weights[name]) for name in weight_names]
    weight_axis.bar(weight_names, weight_values)
    weight_axis.set_title("主实验权重构成")
    weight_axis.set_ylabel("weight")
    weight_axis.tick_params(axis="x", labelrotation=30)
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
- Flatten trend metrics twice: `flatten_trend_metrics()` for `trend_model_metrics`, and `flatten_trend_metrics_by_attr_type()` for `trend_metrics_by_attr_type`.
- Read recommendation metrics for all five methods.
- Build and write all eight design tables: `data_artifact_summary`, `time_split_summary`, `attribute_graph_summary`, `trend_feature_summary`, `trend_model_metrics`, `trend_metrics_by_attr_type`, `recommendation_method_metrics`, and `recommendation_experiment_summary`.
- Read feature importance and export `lightgbm_feature_importance`.
- Read LightGBM predictions and trend samples, build join view, export `trend_curve_examples` and `topk_trend_attributes`.
- Read experiment JSON, export `recommendation_weight_analysis` and table rows.
- Select and write 3 case studies with representative trend attributes and product attributes.
- Pass `config.figure_formats` into `_write_figures()` so requested SVG-only, PNG-only, or SVG+PNG outputs match the manifest `figure_count`.
- Build manifest with real inputs/outputs/warnings.

Keep the function small enough by adding private helpers with concrete return contracts:

`_load_report_inputs()` must be the only helper that reads real artifacts. It returns the DataFrames and JSON-derived rows needed by `_write_tables()`, `_write_figures()`, and `_write_cases()`, making Task 9 unit tests independent from local `data/`, `outputs/`, and CJK fonts.

`report_table_rows` must be built before writing and must include all eight table names in `REPORT_TABLE_COLUMNS`. Use these sources: core artifact paths and row/column counts for `data_artifact_summary`; split metadata plus split parquet row counts for `time_split_summary`; graph node/edge artifacts for `attribute_graph_summary`; documented feature groups for `trend_feature_summary`; `flatten_trend_metrics()` for `trend_model_metrics`; `flatten_trend_metrics_by_attr_type()` for `trend_metrics_by_attr_type`; method metrics JSON for `recommendation_method_metrics`; and `experiment.json` top-level `search_results`, `ablation`, and `best_weights` for `recommendation_experiment_summary`. For `search_results`, flatten each row's `weights` into `pop_score/sim_score/trend_score/recent_score` and `valid_metrics` into metrics columns with `section="search_results"` and `split="valid"`. For `ablation`, use top-level metric fields with `section="ablation"` and use `best_weights` only for the selected trend method row; otherwise leave weight columns blank when a component is not applicable. For case studies, build `article_attributes` from `data/processed/graph/edges_article_attribute.csv` and build `representative_trends` from the same LightGBM prediction + sample join view used by `topk_trend_attributes`, covering every `test` `cutoff_week` present in `pop_similarity_trend` recommendation items. Manifest `input_artifacts` must flatten every path group from `ReportInputPaths`, including `data_artifacts`, `trend_split_samples`, `graph_artifacts`, metrics, recommendation parquet items, and historical recommendation item CSV paths. Manifest warnings must be generated through `_build_report_warnings()` and cover known non-blocking caveats: grid search only has valid metrics, strict w/o Recent ablation is missing, and historical `recommendation_items.csv` exists but is not used. Every table, figure, case, and manifest write must pass `config.output_dir` through the path helpers so `--output-dir` controls the complete export tree.

The helper names referenced by `_load_report_inputs()` are part of the Task 9 contract, not pseudo-code. Implement them in `runner.py` with these return contracts: summary row builders return `list[dict[str, object]]` already matching `REPORT_TABLE_COLUMNS`; `_build_input_artifacts()` returns manifest-ready audit paths for every `ReportInputPaths` group; `_build_report_warnings()` returns design-specified non-blocking caveats; `flatten_experiment_search_results()` returns a DataFrame with `pop_score/sim_score/trend_score/recent_score/ndcg_at_12` for plotting; `flatten_recommendation_experiment_rows()` returns design-table rows for both `search_results` and `ablation`; `build_representative_trend_attributes()` returns Top-N trend rows per selected week, so case rendering can use each case's own `cutoff_week`.

Update `runner.py` imports to stay inside the reports architecture allowlist: use `catalog.readers`, `trend.readers`, `recommendation.readers`, and the new `reports.loaders/tables/paths/plotting/manifest` modules. Do not import `trend.paths`, `recommendation.paths`, `catalog.paths`, `trend.evaluation`, trend training/model modules, recommendation experiment runners, or catalog graph builders from `reports`. Default input paths must come from `reports.paths.default_report_input_paths()` / `ReportInputPaths`; `runner.py` must not define raw upstream artifact path constants.

```python
@dataclass(frozen=True)
class ReportInputs:
    input_artifacts: dict[str, str]
    report_table_rows: dict[str, list[dict[str, object]]]
    trend_metrics: pd.DataFrame
    recommendation_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    trend_view: pd.DataFrame
    search_results: pd.DataFrame
    recommendation_items: pd.DataFrame
    evaluation_labels: pd.DataFrame
    user_profile: pd.DataFrame
    article_attributes: pd.DataFrame
    representative_trends: pd.DataFrame
    best_weights: dict[str, float]
    warnings: list[str]


def run_paper_assets_export(config: PaperAssetsExportConfig) -> dict[str, Any]:
    selected_font = configure_matplotlib_for_reports()
    inputs = _load_report_inputs(config)
    table_paths, row_counts = _write_tables(
        inputs.report_table_rows,
        output_root=config.output_dir,
    )
    figure_paths = _write_figures(
        inputs.trend_metrics,
        inputs.recommendation_metrics,
        inputs.feature_importance,
        inputs.trend_view,
        inputs.search_results,
        best_weights=inputs.best_weights,
        trend_week=config.trend_week,
        top_k=config.top_k,
        figure_formats=config.figure_formats,
        output_root=config.output_dir,
    )
    case_paths, case_user_ids = _write_cases(
        inputs.recommendation_items,
        inputs.evaluation_labels,
        inputs.user_profile,
        inputs.article_attributes,
        inputs.representative_trends,
        case_count=config.case_count,
        output_root=config.output_dir,
    )
    payload = build_manifest_payload(
        parameters={
            "case_count": config.case_count,
            "top_k": config.top_k,
            "trend_week": config.trend_week,
            "figure_formats": list(config.figure_formats),
            "output_dir": str(manifest_output_path(config.output_dir).parent),
            "selected_font": selected_font,
        },
        input_artifacts=inputs.input_artifacts,
        output_artifacts={
            "figures": figure_paths,
            "tables": table_paths,
            "case_studies": case_paths,
        },
        row_counts=row_counts,
        case_user_ids=case_user_ids,
        warnings=inputs.warnings,
    )
    write_manifest(payload, manifest_output_path(config.output_dir))
    return payload


def _load_report_inputs(config: PaperAssetsExportConfig) -> ReportInputs:
    input_paths = config.input_paths or default_report_input_paths()
    trend_metric_paths = input_paths.trend_metrics
    trend_metric_payloads = [
        read_trend_metrics(path)
        for path in trend_metric_paths.values()
    ]
    trend_metric_rows, trend_attr_type_rows = _build_trend_metric_rows(
        trend_metric_payloads
    )

    recommendation_metric_rows = []
    for method, path in input_paths.recommendation_metrics.items():
        payload = read_json_object(path, artifact_name=f"{method} metrics")
        recommendation_metric_rows.extend(flatten_recommendation_metrics(payload))

    predictions = read_trend_model_predictions(input_paths.lightgbm_predictions)
    trend_samples = read_trend_samples(input_paths.trend_model_samples)
    trend_view = build_lightgbm_prediction_sample_view(predictions, trend_samples)
    feature_importance = read_feature_importance(input_paths.lightgbm_feature_importance)
    experiment_payload = read_json_object(
        input_paths.recommendation_experiment,
        artifact_name="main recommendation experiment",
    )
    best_weights = _extract_best_weights(experiment_payload)

    recommendation_items = read_recommendation_items(
        input_paths.recommendation_items["pop_similarity_trend"]
    )
    evaluation_labels = read_evaluation_labels(input_paths.evaluation_labels)
    user_profile = read_user_profile(input_paths.user_profile)
    article_attributes = read_article_attribute_edges(input_paths.article_attributes)
    case_cutoff_weeks = sorted(
        recommendation_items.loc[
            recommendation_items["split"].astype(str) == "test",
            "cutoff_week",
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    representative_trends = build_representative_trend_attributes(
        trend_view,
        week_ids=case_cutoff_weeks,
    )
    search_results = flatten_experiment_search_results(experiment_payload)
    report_table_rows = {
        "data_artifact_summary": build_data_artifact_summary_rows(
            input_paths.data_artifacts
        ),
        "time_split_summary": build_time_split_summary_rows(
            split_paths=input_paths.trend_split_samples,
            time_windows_path=input_paths.time_windows,
            target_users_path=input_paths.target_users,
        ),
        "attribute_graph_summary": build_attribute_graph_summary_rows(
            graph_paths=input_paths.graph_artifacts
        ),
        "trend_feature_summary": build_trend_feature_summary_rows(),
        "trend_model_metrics": trend_metric_rows,
        "trend_metrics_by_attr_type": trend_attr_type_rows,
        "recommendation_method_metrics": recommendation_metric_rows,
        "recommendation_experiment_summary": flatten_recommendation_experiment_rows(
            experiment_payload
        ),
    }
    return ReportInputs(
        input_artifacts=_build_input_artifacts(input_paths),
        report_table_rows=report_table_rows,
        trend_metrics=build_report_table(
            trend_metric_rows,
            table_name="trend_model_metrics",
        ),
        recommendation_metrics=build_report_table(
            recommendation_metric_rows,
            table_name="recommendation_method_metrics",
        ),
        feature_importance=feature_importance,
        trend_view=trend_view,
        search_results=search_results,
        recommendation_items=recommendation_items,
        evaluation_labels=evaluation_labels,
        user_profile=user_profile,
        article_attributes=article_attributes,
        representative_trends=representative_trends,
        best_weights=best_weights,
        warnings=_build_report_warnings(experiment_payload, input_paths),
    )


def _build_input_artifacts(input_paths: ReportInputPaths) -> dict[str, str]:
    artifacts = {
        "lightgbm_predictions": str(input_paths.lightgbm_predictions),
        "trend_model_samples": str(input_paths.trend_model_samples),
        "feature_importance": str(input_paths.lightgbm_feature_importance),
        "recommendation_experiment": str(input_paths.recommendation_experiment),
        "evaluation_labels": str(input_paths.evaluation_labels),
        "user_profile": str(input_paths.user_profile),
        "article_attributes": str(input_paths.article_attributes),
    }
    _extend_prefixed_paths(artifacts, "data_artifact", input_paths.data_artifacts)
    _extend_prefixed_paths(
        artifacts,
        "trend_split_sample",
        input_paths.trend_split_samples,
    )
    _extend_prefixed_paths(artifacts, "graph_artifact", input_paths.graph_artifacts)
    _extend_prefixed_paths(artifacts, "trend_metrics", input_paths.trend_metrics)
    _extend_prefixed_paths(
        artifacts,
        "recommendation_metrics",
        input_paths.recommendation_metrics,
    )
    _extend_prefixed_paths(
        artifacts,
        "recommendation_items",
        input_paths.recommendation_items,
    )
    _extend_prefixed_paths(
        artifacts,
        "recommendation_items_csv",
        input_paths.recommendation_items_csv,
    )
    return artifacts


def _build_report_warnings(
    experiment_payload: dict[str, object],
    input_paths: ReportInputPaths,
) -> list[str]:
    warnings: list[str] = []
    search_results = experiment_payload.get("search_results", [])
    if isinstance(search_results, list) and any(
        isinstance(row, dict)
        and "valid_metrics" in row
        and "test_metrics" not in row
        for row in search_results
    ):
        warnings.append(
            "recommendation grid search 只有 valid 指标，test 指标仅来自最终方法评价。"
        )
    if not _has_strict_without_recent_ablation(experiment_payload):
        warnings.append("recommendation ablation 缺少严格 w/o Recent 消融行。")

    legacy_csv_paths = [
        path
        for path in input_paths.recommendation_items_csv.values()
        if Path(path).exists()
    ]
    if legacy_csv_paths:
        warnings.append(
            "检测到历史 recommendation_items.csv，但报告读取 parquet 长表: "
            + ", ".join(str(path) for path in legacy_csv_paths)
        )
    return warnings


def _extend_prefixed_paths(
    target: dict[str, str],
    prefix: str,
    paths: dict[str, Path],
) -> None:
    for name, path in paths.items():
        target[f"{prefix}__{name}"] = str(path)


def _extract_best_weights(payload: dict[str, object]) -> dict[str, float]:
    weights = payload.get("best_weights")
    if not isinstance(weights, dict):
        raise ValueError("experiment.json best_weights 必须是对象")
    return {
        name: float(weights[name])
        for name in ("pop_score", "sim_score", "trend_score", "recent_score")
    }


def _has_strict_without_recent_ablation(payload: dict[str, object]) -> bool:
    for row in payload.get("ablation", []):
        if not isinstance(row, dict):
            continue
        fields = " ".join(
            str(row.get(key, "")).lower()
            for key in ("section", "name", "method", "variant")
        )
        if any(
            token in fields
            for token in ("w/o recent", "without_recent", "no_recent")
        ):
            return True
        weights = row.get("weights")
        if isinstance(weights, dict):
            try:
                recent_score = float(weights.get("recent_score", -1.0))
            except (TypeError, ValueError):
                continue
            if recent_score == 0.0:
                return True
    return False


def build_data_artifact_summary_rows(
    artifacts: dict[str, Path | str] | None = None,
    *,
    sections: dict[str, str] | None = None,
    paper_usage: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    artifacts = artifacts or default_report_input_paths().data_artifacts
    default_sections = {
        "articles_clean": "catalog",
        "nodes_article": "attribute_graph",
        "nodes_attribute": "attribute_graph",
        "edges_article_attribute": "attribute_graph",
        "edges_attribute_hierarchy": "attribute_graph",
        "article_week_sales": "trend",
        "attribute_week_heat": "trend",
        "attribute_week_target": "trend",
        "trend_model_samples": "trend",
        "time_windows": "recommendation",
        "target_users": "recommendation",
        "evaluation_labels": "recommendation",
        "user_profile": "recommendation",
    }
    default_usage = {
        "articles_clean": "商品属性清洗规模",
        "nodes_article": "属性图商品节点规模",
        "nodes_attribute": "属性图属性节点规模",
        "edges_article_attribute": "商品-属性边规模",
        "edges_attribute_hierarchy": "属性层级边规模",
        "article_week_sales": "商品周销量聚合规模",
        "attribute_week_heat": "属性周热度规模",
        "attribute_week_target": "趋势标签规模",
        "trend_model_samples": "趋势模型样本规模",
        "time_windows": "推荐评价时间窗规模",
        "target_users": "推荐目标用户规模",
        "evaluation_labels": "推荐真实标签规模",
        "user_profile": "用户画像规模",
    }
    section_map = {**default_sections, **(sections or {})}
    usage_map = {**default_usage, **(paper_usage or {})}

    rows: list[dict[str, object]] = []
    for artifact, path_value in artifacts.items():
        path = Path(path_value)
        row_count, column_count = _artifact_shape(path)
        rows.append(
            {
                "section": section_map.get(artifact, "other"),
                "artifact": artifact,
                "path": str(path),
                "row_count": row_count,
                "column_count": column_count,
                "paper_usage": usage_map.get(artifact, ""),
            }
        )
    return rows


def build_time_split_summary_rows(
    split_frames: dict[str, pd.DataFrame] | None = None,
    *,
    split_paths: dict[str, Path] | None = None,
    time_windows_path: Path | None = None,
    target_users_path: Path | None = None,
    domain: str = "trend",
) -> list[dict[str, object]]:
    injected_split_frames = split_frames is not None
    input_paths = default_report_input_paths()
    split_paths = split_paths or input_paths.trend_split_samples
    split_frames = split_frames or {
        split: pd.read_parquet(path) for split, path in split_paths.items()
    }
    rows = [
        _time_split_row(domain=domain, split=split, dataframe=dataframe)
        for split, dataframe in split_frames.items()
    ]

    if injected_split_frames:
        return rows

    time_windows = read_time_windows(time_windows_path or input_paths.time_windows)
    target_users = read_target_users(target_users_path or input_paths.target_users)
    for split, dataframe in time_windows.groupby("split", sort=True):
        split_users = target_users.loc[target_users["split"].astype(str) == str(split)]
        rows.append(
            _time_split_row(
                domain="recommendation",
                split=str(split),
                dataframe=dataframe,
                user_count=split_users["customer_id"].nunique(),
            )
        )
    return rows


def build_attribute_graph_summary_rows(
    graph_frames: dict[str, pd.DataFrame] | None = None,
    *,
    graph_paths: dict[str, Path | str] | None = None,
) -> list[dict[str, object]]:
    default_graph_paths = default_report_input_paths().graph_artifacts
    graph_paths = {**default_graph_paths, **(graph_paths or {})}
    graph_frames = graph_frames or {
        "nodes_article": pd.read_csv(graph_paths["nodes_article"]),
        "nodes_attribute": read_attribute_nodes(Path(graph_paths["nodes_attribute"])),
        "edges_article_attribute": read_article_attribute_edges(
            Path(graph_paths["edges_article_attribute"])
        ),
        "edges_attribute_hierarchy": read_attribute_hierarchy_edges(
            Path(graph_paths["edges_attribute_hierarchy"])
        ),
    }
    rows = [
        {
            "entity_type": "article",
            "attr_type": "",
            "relation_type": "node",
            "count": len(graph_frames["nodes_article"]),
            "path": str(graph_paths["nodes_article"]),
            "paper_usage": "商品节点规模",
        },
        {
            "entity_type": "attribute",
            "attr_type": "",
            "relation_type": "node",
            "count": len(graph_frames["nodes_attribute"]),
            "path": str(graph_paths["nodes_attribute"]),
            "paper_usage": "属性节点规模",
        },
        {
            "entity_type": "attribute",
            "attr_type": "",
            "relation_type": "hierarchy",
            "count": len(graph_frames["edges_attribute_hierarchy"]),
            "path": str(graph_paths["edges_attribute_hierarchy"]),
            "paper_usage": "属性层级边规模",
        },
    ]
    edge_frame = graph_frames["edges_article_attribute"]
    for attr_type, group in edge_frame.groupby("attr_type", sort=True):
        rows.append(
            {
                "entity_type": "article_attribute_edge",
                "attr_type": str(attr_type),
                "relation_type": "article_attribute",
                "count": len(group),
                "path": str(graph_paths["edges_article_attribute"]),
                "paper_usage": "商品-属性边规模",
            }
        )
    return rows


def build_trend_feature_summary_rows() -> list[dict[str, object]]:
    feature_specs = [
        ("level", "heat_t", "trend_model_samples", True, "当前周属性热度"),
        ("level", "share_t", "trend_model_samples", True, "当前周属性份额"),
        ("lag", "lag_1_heat", "trend_model_samples", True, "上一周属性热度"),
        ("lag", "lag_2_heat", "trend_model_samples", True, "前两周属性热度"),
        ("lag", "lag_4_heat", "trend_model_samples", True, "前四周属性热度"),
        ("growth", "growth_1w", "trend_model_samples", True, "一周热度变化"),
        ("growth", "growth_4w", "trend_model_samples", True, "四周热度变化"),
        ("history", "history_total_heat_t", "trend_model_samples", True, "历史累计热度"),
        (
            "history",
            "history_active_weeks_t",
            "trend_model_samples",
            True,
            "历史活跃周数",
        ),
        (
            "label",
            "target_growth",
            "trend_model_samples",
            False,
            "下一周趋势增长标签",
        ),
    ]
    return [
        {
            "feature_group": feature_group,
            "feature_name": feature_name,
            "source_table": source_table,
            "model_input": model_input,
            "description": description,
        }
        for feature_group, feature_name, source_table, model_input, description in feature_specs
    ]


def build_representative_trend_attributes(
    trend_view: pd.DataFrame,
    *,
    week_id: int | None = None,
    week_ids: list[int] | None = None,
    top_n: int = 8,
) -> pd.DataFrame:
    required_columns = {
        "week_id",
        "attr_type",
        "attr_value",
        "pred_target_growth",
        "heat_t",
        "is_trend_eligible_t",
    }
    missing_columns = sorted(required_columns - set(trend_view.columns))
    if missing_columns:
        raise ValueError(f"代表趋势属性缺少字段: {missing_columns}")
    if week_id is not None and week_ids is not None:
        raise ValueError("week_id 和 week_ids 不能同时传入")
    selected_weeks: list[int] | None = None
    if week_id is not None:
        selected_weeks = [int(week_id)]
    if week_ids is not None:
        selected_weeks = sorted({int(value) for value in week_ids})

    filtered = trend_view.loc[
        trend_view["is_trend_eligible_t"].astype(bool)
    ].copy()
    filtered["week_id"] = filtered["week_id"].astype(int)
    if selected_weeks is not None:
        filtered = filtered.loc[filtered["week_id"].isin(selected_weeks)].copy()
    filtered = filtered.sort_values(
        ["week_id", "pred_target_growth", "heat_t"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    per_week = filtered.groupby("week_id", group_keys=False, sort=False).head(top_n)
    return per_week.sort_values(
        ["pred_target_growth", "heat_t"],
        ascending=[False, False],
        kind="mergesort",
    ).loc[
        :,
        ["week_id", "attr_type", "attr_value", "pred_target_growth", "heat_t"],
    ]


def flatten_experiment_search_results(payload: dict[str, object]) -> pd.DataFrame:
    rows = []
    for rank, result in enumerate(payload.get("search_results", []), start=1):
        if not isinstance(result, dict):
            raise ValueError(f"search_results[{rank - 1}] 必须是对象")
        weights = _required_mapping(result, "weights", f"search_results[{rank - 1}]")
        metrics = _required_mapping(
            result,
            "valid_metrics",
            f"search_results[{rank - 1}]",
        )
        rows.append(
            _recommendation_experiment_row(
                section="search_results",
                rank=rank,
                method=str(result.get("method", "pop_similarity_trend")),
                split="valid",
                weights=weights,
                metrics=metrics,
            )
        )
    return pd.DataFrame(
        rows,
        columns=REPORT_TABLE_COLUMNS["recommendation_experiment_summary"],
    )


def flatten_recommendation_experiment_rows(
    payload: dict[str, object],
) -> list[dict[str, object]]:
    rows = flatten_experiment_search_results(payload).to_dict("records")
    best_weights = payload.get("best_weights", {})
    if not isinstance(best_weights, dict):
        raise ValueError("experiment.json best_weights 必须是对象")

    for rank, result in enumerate(payload.get("ablation", []), start=1):
        if not isinstance(result, dict):
            raise ValueError(f"ablation[{rank - 1}] 必须是对象")
        method = str(result.get("method", ""))
        weights: dict[str, object] = {}
        if method == "pop_similarity_trend":
            weights = best_weights
        rows.append(
            _recommendation_experiment_row(
                section="ablation",
                rank=rank,
                method=method,
                split=str(result.get("split", "test")),
                weights=weights,
                metrics=result,
                blank_missing_weights=method != "pop_similarity_trend",
            )
        )
    return rows


def _artifact_shape(path: Path) -> tuple[int, int]:
    if path.suffix == ".csv":
        dataframe = pd.read_csv(path)
        return len(dataframe), len(dataframe.columns)
    if path.suffix == ".parquet":
        dataframe = pd.read_parquet(path)
        return len(dataframe), len(dataframe.columns)
    if path.suffix == ".json":
        payload = read_json_object(path, artifact_name=path.name)
        return 1, len(payload)
    raise ValueError(f"不支持统计的报告 artifact 类型: {path}")


def _time_split_row(
    *,
    domain: str,
    split: str,
    dataframe: pd.DataFrame,
    user_count: int = 0,
) -> dict[str, object]:
    if "week_id" in dataframe.columns:
        week_ids = dataframe["week_id"].dropna().astype(int)
    elif {"cutoff_week", "label_week"} <= set(dataframe.columns):
        week_ids = pd.concat(
            [dataframe["cutoff_week"], dataframe["label_week"]],
            ignore_index=True,
        ).dropna().astype(int)
    else:
        raise ValueError("time split summary 缺少 week_id 或 cutoff_week/label_week")
    return {
        "domain": domain,
        "split": split,
        "week_start": int(week_ids.min()),
        "week_end": int(week_ids.max()),
        "week_count": int(week_ids.nunique()),
        "row_count": len(dataframe),
        "attribute_count": int(dataframe["attr_id"].nunique())
        if "attr_id" in dataframe.columns
        else 0,
        "user_count": int(user_count),
    }


def _recommendation_experiment_row(
    *,
    section: str,
    rank: int,
    method: str,
    split: str,
    weights: dict[str, object],
    metrics: dict[str, object],
    blank_missing_weights: bool = False,
) -> dict[str, object]:
    return {
        "section": section,
        "rank": rank,
        "method": method,
        "split": split,
        "pop_score": _score_value(weights, "pop_score", blank=blank_missing_weights),
        "sim_score": _score_value(weights, "sim_score", blank=blank_missing_weights),
        "trend_score": _score_value(
            weights,
            "trend_score",
            blank=blank_missing_weights,
        ),
        "recent_score": _score_value(
            weights,
            "recent_score",
            blank=blank_missing_weights,
        ),
        "map_at_12": float(metrics["map_at_12"]),
        "recall_at_12": float(metrics["recall_at_12"]),
        "hit_rate_at_12": float(metrics["hit_rate_at_12"]),
        "ndcg_at_12": float(metrics["ndcg_at_12"]),
        "coverage": float(metrics["coverage"]),
    }


def _score_value(
    weights: dict[str, object],
    key: str,
    *,
    blank: bool,
) -> float | str:
    if key not in weights:
        if blank:
            return ""
        raise ValueError(f"experiment.json 权重缺少字段: {key}")
    return float(weights[key])


def _required_mapping(
    payload: dict[str, object],
    key: str,
    source_name: str,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source_name} {key} 必须是对象")
    return value


def _build_trend_metric_rows(
    trend_metric_payloads: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    attr_type_rows: list[dict[str, object]] = []
    for payload in trend_metric_payloads:
        metric_rows.extend(flatten_trend_metrics(payload))
        attr_type_rows.extend(flatten_trend_metrics_by_attr_type(payload))
    return metric_rows, attr_type_rows


def _write_tables(
    report_table_rows: dict[str, list[dict[str, object]]],
    *,
    output_root: Path | None,
) -> tuple[list[str], dict[str, int]]:
    missing_tables = sorted(set(REPORT_TABLE_COLUMNS) - set(report_table_rows))
    if missing_tables:
        raise ValueError(f"报告表格缺少设计要求的表: {missing_tables}")
    output_paths: list[str] = []
    row_counts: dict[str, int] = {}
    for name, columns in REPORT_TABLE_COLUMNS.items():
        table = build_report_table(report_table_rows[name], table_name=name)
        written = write_report_table(
            table,
            columns=columns,
            output_paths=table_output_paths(name, output_root=output_root),
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
    best_weights: dict[str, float],
    *,
    trend_week: int,
    top_k: int,
    figure_formats: tuple[str, ...],
    output_root: Path | None,
) -> list[str]:
    figure_builders = {
        "data_pipeline": build_data_pipeline_figure(),
        "attribute_graph_schema": build_attribute_graph_schema_figure(),
        "trend_curve_examples": build_trend_curve_examples_figure(
            trend_view,
            week_id=trend_week,
            lookback_weeks=8,
            top_n=3,
        ),
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
            search_results,
            best_weights=best_weights,
        ),
    }
    output_paths: list[str] = []
    for name, figure in figure_builders.items():
        written = save_report_figure(
            figure,
            figure_output_paths(name, output_root=output_root),
            formats=figure_formats,
        )
        output_paths.extend(str(path) for path in written)
    return output_paths


def _write_cases(
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    user_profile: pd.DataFrame,
    article_attributes: pd.DataFrame,
    representative_trends: pd.DataFrame,
    *,
    case_count: int,
    output_root: Path | None,
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
            article_attributes=article_attributes,
            representative_trends=representative_trends,
        )
        case_id = f"case_{index:02d}"
        paths = case_study_output_paths(case_id, output_root=output_root)
        write_json_atomic(payload, paths["json"])
        write_text_atomic(render_case_markdown(payload), paths["markdown"])
        output_paths.extend(str(path) for path in paths.values())
        case_user_ids.append(str(payload["customer_id"]))
    return output_paths, case_user_ids
```

Each helper should return paths as strings for manifest recording. The runner must call `_write_figures(..., figure_formats=config.figure_formats, output_root=config.output_dir)` and use the returned paths directly for manifest `output_artifacts["figures"]`; do not recompute expected figure paths separately.

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

The default `--figure-format svg,png` writes 8 figures x 2 formats = 16 files. If a non-default format is requested, `figure_count` must be `8 * len(figure_formats)`. Exact table count depends on whether each CSV and Markdown path is counted separately. Manifest must record file paths and warnings.

Also run a focused format smoke:

```sh
uv run python src/17_export_paper_assets.py --figure-format svg --output-dir outputs/reports-svg-smoke
```

Expected: `figures=8`, every generated figure path ends with `.svg`, and no PNG files are written under `outputs/reports-svg-smoke/figures`.

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
