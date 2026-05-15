# Trend Graph Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated trend graph feature ablation experiment that creates explainable KG context features, runs five LightGBM group ablations, and writes all outputs only under `outputs/experiments/trend_graph_feature_ablation/`.

**Architecture:** Keep the default `fashion_trend.trend` training contract and `00-18` pipeline unchanged. Add a single non-mainline CLI `src/19_run_trend_graph_feature_ablation.py` that orchestrates experiment-only modules under `src/experiments/trend_graph_feature_ablation/`. Reuse stable readers, LightGBM config parsing, prediction validation, and trend metric computation, but control all write paths inside the experiment package.

**Tech Stack:** Python 3.10-3.12, pandas, numpy, LightGBM, pytest, existing `fashion_trend.foundation` IO helpers, existing trend evaluation metrics.

---

## Scope And Constraints

This plan implements the approved spec:

```text
docs/superpowers/specs/2026-05-15-trend-graph-feature-ablation-design.md
```

Hard constraints:

- Do not modify default `data/processed/features/trend_model_samples*.parquet` schema.
- Do not write to `outputs/models/lightgbm/`, `outputs/metrics/lightgbm/`, `outputs/reports/`, `outputs/defense_app/`, or `apps/defense_app/`.
- Do not add the experiment to the default `00-18` stable pipeline.
- Do not put new behavior into the default `fashion_trend.trend` training contract.
- All experiment outputs must stay under `outputs/experiments/trend_graph_feature_ablation/`.

## File Structure

Create these files:

```text
src/19_run_trend_graph_feature_ablation.py
src/experiments/__init__.py
src/experiments/trend_graph_feature_ablation/__init__.py
src/experiments/trend_graph_feature_ablation/paths.py
src/experiments/trend_graph_feature_ablation/contracts.py
src/experiments/trend_graph_feature_ablation/artifact_io.py
src/experiments/trend_graph_feature_ablation/build_features.py
src/experiments/trend_graph_feature_ablation/feature_groups.py
src/experiments/trend_graph_feature_ablation/train_runs.py
src/experiments/trend_graph_feature_ablation/evaluate.py
src/experiments/trend_graph_feature_ablation/summarize.py
src/experiments/trend_graph_feature_ablation/write_docs.py
src/experiments/trend_graph_feature_ablation/runner.py
tests/test_trend_graph_feature_ablation_artifacts.py
tests/test_trend_graph_feature_ablation_features.py
tests/test_trend_graph_feature_ablation_groups.py
tests/test_trend_graph_feature_ablation_runner.py
```

Modify these files:

```text
README.md
```

No other files should change unless a test exposes a narrow missing import boundary.

---

### Task 1: Paths, Contracts, And Write Guards

**Files:**
- Create: `src/experiments/__init__.py`
- Create: `src/experiments/trend_graph_feature_ablation/__init__.py`
- Create: `src/experiments/trend_graph_feature_ablation/paths.py`
- Create: `src/experiments/trend_graph_feature_ablation/contracts.py`
- Create: `src/experiments/trend_graph_feature_ablation/artifact_io.py`
- Test: `tests/test_trend_graph_feature_ablation_artifacts.py`

- [ ] **Step 1: Write failing tests for experiment paths and forbidden writes**

Add `tests/test_trend_graph_feature_ablation_artifacts.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.trend_graph_feature_ablation.artifact_io import (
    assert_experiment_write_path,
    build_input_hash_entry,
    digest_json_payload,
)
from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_VARIANTS,
    SCHEMA_VERSION,
)
from experiments.trend_graph_feature_ablation.paths import (
    EXPERIMENT_ROOT,
    FEATURE_GROUPS_PATH,
    run_dir,
)


def test_contracts_pin_schema_and_variant_order() -> None:
    assert SCHEMA_VERSION == "trend_graph_feature_ablation.v1"
    assert ABLATION_VARIANTS == (
        "no_graph",
        "current_coarse_graph",
        "full_enhanced",
        "wo_hierarchy_context",
        "wo_sibling_competition",
    )


def test_experiment_paths_stay_under_experiment_root() -> None:
    assert EXPERIMENT_ROOT.as_posix().endswith(
        "outputs/experiments/trend_graph_feature_ablation"
    )
    assert FEATURE_GROUPS_PATH.is_relative_to(EXPERIMENT_ROOT)
    assert run_dir("full_enhanced").is_relative_to(EXPERIMENT_ROOT)


@pytest.mark.parametrize(
    "path",
    [
        Path("data/processed/features/trend_model_samples.parquet"),
        Path("outputs/models/lightgbm/predictions.csv"),
        Path("outputs/metrics/lightgbm/trend_metrics.json"),
        Path("outputs/reports/manifest.json"),
        Path("outputs/defense_app/fashion_demo.sqlite"),
        Path("apps/defense_app/frontend/package.json"),
    ],
)
def test_write_guard_rejects_forbidden_paths(path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden experiment write path"):
        assert_experiment_write_path(path)


def test_write_guard_accepts_experiment_paths() -> None:
    assert_experiment_write_path(
        Path("outputs/experiments/trend_graph_feature_ablation/metrics_summary.csv")
    )


def test_write_guard_supports_injected_root_for_tmp_path(tmp_path: Path) -> None:
    assert_experiment_write_path(
        tmp_path / "runs" / "no_graph" / "predictions.csv",
        root=tmp_path,
    )


def test_build_input_hash_entry_records_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "outputs" / "models" / "lightgbm" / "params.json"

    entry = build_input_hash_entry(missing, required=False, row_count=None)

    assert entry["path"] == str(missing)
    assert entry["exists"] is False
    assert entry["hash"] is None
    assert entry["row_count"] is None


def test_digest_json_payload_is_stable() -> None:
    left = digest_json_payload({"b": [2, 1], "a": "x"})
    right = digest_json_payload({"a": "x", "b": [2, 1]})

    assert left == right
    assert len(left) == 64
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_artifacts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'`.

- [ ] **Step 3: Add package markers**

Create `src/experiments/__init__.py`:

```python
"""Experiment-only modules that are not part of the stable pipeline."""
```

Create `src/experiments/trend_graph_feature_ablation/__init__.py`:

```python
"""Independent trend graph feature ablation experiment."""
```

- [ ] **Step 4: Add experiment contracts**

Create `src/experiments/trend_graph_feature_ablation/contracts.py`:

```python
from __future__ import annotations

SCHEMA_VERSION = "trend_graph_feature_ablation.v1"

ABLATION_EXPERIMENT_ID = "trend_graph_feature_ablation"

ABLATION_VARIANTS = (
    "no_graph",
    "current_coarse_graph",
    "full_enhanced",
    "wo_hierarchy_context",
    "wo_sibling_competition",
)

FEATURE_GROUP_NAMES = (
    "base_numeric_non_graph",
    "categorical",
    "coarse_graph",
    "hierarchy_context",
    "sibling_competition",
    "light_structure",
)

ALL_SAMPLE_KEY_COLUMNS = ("week_id", "attr_id")
SPLIT_SAMPLE_KEY_COLUMNS = ("split", "week_id", "attr_id")
TARGET_COLUMNS = (
    "target_growth",
    "target_log_heat_t1",
    "target_rank_in_type_t1",
)
PREDICTION_COLUMNS = (
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
SUMMARY_COLUMNS = (
    "variant",
    "feature_count",
    "valid_ndcg_at_10",
    "valid_spearman",
    "valid_precision_at_10",
    "valid_recall_at_10",
    "valid_mae",
    "valid_rmse",
    "test_ndcg_at_10",
    "test_spearman",
    "test_precision_at_10",
    "test_recall_at_10",
    "test_mae",
    "test_rmse",
)
RUN_ARTIFACT_FILENAMES = (
    "predictions.csv",
    "metrics.json",
    "feature_importance.csv",
    "metadata.json",
    "params.json",
    "model.txt",
)
```

- [ ] **Step 5: Add experiment paths**

Create `src/experiments/trend_graph_feature_ablation/paths.py`:

```python
from __future__ import annotations

from pathlib import Path

from fashion_trend.foundation.paths import OUTPUT_DIR

EXPERIMENT_ROOT = OUTPUT_DIR / "experiments" / "trend_graph_feature_ablation"
FEATURES_DIR = EXPERIMENT_ROOT / "features"
RUNS_DIR = EXPERIMENT_ROOT / "runs"
STAGING_DIR = EXPERIMENT_ROOT / ".staging"

ENHANCED_SAMPLES_ALL_PATH = FEATURES_DIR / "enhanced_samples_all.parquet"
ENHANCED_SAMPLES_TRAIN_PATH = FEATURES_DIR / "enhanced_samples_train.parquet"
ENHANCED_SAMPLES_VALID_PATH = FEATURES_DIR / "enhanced_samples_valid.parquet"
ENHANCED_SAMPLES_TEST_PATH = FEATURES_DIR / "enhanced_samples_test.parquet"
FEATURE_GROUPS_PATH = FEATURES_DIR / "feature_groups.json"
FEATURE_SCHEMA_PATH = FEATURES_DIR / "feature_schema.json"
ROW_ALIGNMENT_CHECK_PATH = FEATURES_DIR / "row_alignment_check.json"
INPUT_HASHES_PATH = FEATURES_DIR / "input_hashes.json"

METRICS_SUMMARY_CSV_PATH = EXPERIMENT_ROOT / "metrics_summary.csv"
METRICS_SUMMARY_MD_PATH = EXPERIMENT_ROOT / "metrics_summary.md"
EXPERIMENT_DOC_PATH = EXPERIMENT_ROOT / "experiment.md"
MANIFEST_PATH = EXPERIMENT_ROOT / "manifest.json"


def run_dir(variant: str) -> Path:
    return RUNS_DIR / variant


def run_artifact_path(variant: str, filename: str) -> Path:
    return run_dir(variant) / filename


def staging_run_dir(run_token: str) -> Path:
    return STAGING_DIR / run_token
```

- [ ] **Step 6: Add digest, hash, and write guard helpers**

Create `src/experiments/trend_graph_feature_ablation/artifact_io.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.trend_graph_feature_ablation.paths import EXPERIMENT_ROOT

FORBIDDEN_WRITE_PATHS = (
    Path("data/processed/features/trend_model_samples.parquet"),
    Path("data/processed/features/trend_model_samples_train.parquet"),
    Path("data/processed/features/trend_model_samples_valid.parquet"),
    Path("data/processed/features/trend_model_samples_test.parquet"),
    Path("outputs/models/lightgbm"),
    Path("outputs/metrics/lightgbm"),
    Path("outputs/reports"),
    Path("outputs/defense_app"),
    Path("apps/defense_app"),
)


def digest_json_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_experiment_write_path(
    path: Path,
    *,
    root: Path = EXPERIMENT_ROOT,
) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"forbidden experiment write path outside root: {path}")
    if resolved_root != EXPERIMENT_ROOT.resolve():
        return
    for forbidden in FORBIDDEN_WRITE_PATHS:
        forbidden_resolved = forbidden.resolve()
        if resolved == forbidden_resolved or resolved.is_relative_to(forbidden_resolved):
            raise ValueError(f"forbidden experiment write path: {path}")


def build_input_hash_entry(
    path: Path,
    *,
    required: bool,
    row_count: int | None,
) -> dict[str, object]:
    exists = path.exists()
    if required and not exists:
        raise FileNotFoundError(f"实验输入不存在: {path}")
    if not exists:
        return {
            "path": str(path),
            "exists": False,
            "mtime": None,
            "size": None,
            "hash": None,
            "row_count": row_count,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "hash": digest_file(path),
        "row_count": row_count,
    }
```

- [ ] **Step 7: Run artifact tests**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```sh
git add src/experiments tests/test_trend_graph_feature_ablation_artifacts.py
git commit -m "feat(experiments): 添加图特征消融基础契约"
```

---

### Task 2: Feature Group Masks And Schema

**Files:**
- Create: `src/experiments/trend_graph_feature_ablation/feature_groups.py`
- Modify: `src/experiments/trend_graph_feature_ablation/contracts.py`
- Test: `tests/test_trend_graph_feature_ablation_groups.py`

- [ ] **Step 1: Write failing feature group tests**

Add `tests/test_trend_graph_feature_ablation_groups.py`:

```python
from __future__ import annotations

import pytest

from experiments.trend_graph_feature_ablation.feature_groups import (
    build_feature_groups_payload,
    build_feature_mask_digest,
    build_variant_feature_masks,
    validate_variant_masks,
)
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_CATEGORICAL_FEATURES,
    LIGHTGBM_NUMERIC_FEATURES,
)


def test_current_coarse_graph_matches_stable_lightgbm_features() -> None:
    masks = build_variant_feature_masks()
    current = masks["current_coarse_graph"]

    assert tuple(current["numeric_features"]) == LIGHTGBM_NUMERIC_FEATURES
    assert tuple(current["categorical_features"]) == LIGHTGBM_CATEGORICAL_FEATURES


def test_no_graph_excludes_coarse_and_kg_features() -> None:
    masks = build_variant_feature_masks()
    numeric = masks["no_graph"]["numeric_features"]

    assert "article_count" not in numeric
    assert "degree" not in numeric
    assert not any(feature.startswith("kg_") for feature in numeric)


def test_wo_hierarchy_keeps_light_structure() -> None:
    masks = build_variant_feature_masks()
    numeric = masks["wo_hierarchy_context"]["numeric_features"]

    assert "kg_parent_share_t_wavg" not in numeric
    assert "kg_parent_edge_weight_sum" in numeric
    assert "kg_has_parent" in numeric


def test_feature_groups_include_debug_rank_helper_as_not_in_mask() -> None:
    payload = build_feature_groups_payload()
    all_mask_features = set()
    for mask in payload["variants"].values():
        all_mask_features.update(mask["numeric_features"])
        all_mask_features.update(mask["categorical_features"])

    assert "rank_pct_t" not in all_mask_features
    assert not any(feature == "kg_rank_pct_t" for feature in all_mask_features)


def test_validate_variant_masks_rejects_target_columns() -> None:
    masks = build_variant_feature_masks()
    masks["no_graph"]["numeric_features"].append("target_growth")

    with pytest.raises(ValueError, match="forbidden feature"):
        validate_variant_masks(masks)


def test_feature_mask_digest_changes_with_feature_order() -> None:
    digest_a = build_feature_mask_digest(
        "variant",
        numeric_features=["a", "b"],
        categorical_features=["c"],
    )
    digest_b = build_feature_mask_digest(
        "variant",
        numeric_features=["b", "a"],
        categorical_features=["c"],
    )

    assert digest_a != digest_b
```

- [ ] **Step 2: Run feature group tests to verify failure**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_groups.py -q
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `feature_groups`.

- [ ] **Step 3: Add feature group definitions**

Create `src/experiments/trend_graph_feature_ablation/feature_groups.py`:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any

from experiments.trend_graph_feature_ablation.artifact_io import digest_json_payload
from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_VARIANTS,
    FEATURE_GROUP_NAMES,
    SCHEMA_VERSION,
    TARGET_COLUMNS,
)
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_CATEGORICAL_FEATURES,
    LIGHTGBM_NUMERIC_FEATURES,
)

IDENTIFIER_COLUMNS = {"attr_id", "attr_value", "split"}
FORBIDDEN_FEATURES = set(TARGET_COLUMNS) | IDENTIFIER_COLUMNS

COARSE_GRAPH_FEATURES = (
    "article_count",
    "is_core_attr",
    "parent_count",
    "child_count",
    "degree",
)
BASE_NUMERIC_NON_GRAPH_FEATURES = tuple(
    feature for feature in LIGHTGBM_NUMERIC_FEATURES if feature not in COARSE_GRAPH_FEATURES
)
HIERARCHY_CONTEXT_FEATURES = (
    "kg_parent_heat_t_wavg",
    "kg_parent_share_t_wavg",
    "kg_parent_growth_lag_1_wavg",
    "kg_parent_rank_pct_t_wavg",
    "kg_child_heat_t_wavg",
    "kg_child_share_t_wavg",
    "kg_child_growth_lag_1_wavg",
    "kg_child_rank_pct_t_wavg",
    "kg_self_parent_share_gap_t",
    "kg_self_parent_growth_gap_lag_1",
    "kg_self_child_share_gap_t",
    "kg_self_child_growth_gap_lag_1",
)
SIBLING_COMPETITION_FEATURES = (
    "kg_sibling_count",
    "kg_sibling_share_t_wavg",
    "kg_sibling_share_t_max",
    "kg_sibling_growth_lag_1_wavg",
    "kg_sibling_rank_pct_t_wavg",
    "kg_self_vs_sibling_share_gap_t",
    "kg_self_vs_sibling_growth_gap_lag_1",
    "kg_has_sibling",
)
LIGHT_STRUCTURE_FEATURES = (
    "kg_parent_edge_weight_sum",
    "kg_child_edge_weight_sum",
    "kg_parent_edge_weight_log",
    "kg_child_edge_weight_log",
    "kg_has_parent",
    "kg_has_child",
    "kg_is_root_attr",
    "kg_is_leaf_attr",
)


def build_feature_groups() -> dict[str, list[str]]:
    return {
        "base_numeric_non_graph": list(BASE_NUMERIC_NON_GRAPH_FEATURES),
        "categorical": list(LIGHTGBM_CATEGORICAL_FEATURES),
        "coarse_graph": list(COARSE_GRAPH_FEATURES),
        "hierarchy_context": list(HIERARCHY_CONTEXT_FEATURES),
        "sibling_competition": list(SIBLING_COMPETITION_FEATURES),
        "light_structure": list(LIGHT_STRUCTURE_FEATURES),
    }


def build_variant_feature_masks() -> dict[str, dict[str, list[str]]]:
    groups = build_feature_groups()
    masks = {
        "no_graph": {
            "numeric_features": groups["base_numeric_non_graph"].copy(),
            "categorical_features": groups["categorical"].copy(),
        },
        "current_coarse_graph": {
            "numeric_features": list(LIGHTGBM_NUMERIC_FEATURES),
            "categorical_features": groups["categorical"].copy(),
        },
        "full_enhanced": {
            "numeric_features": [
                *LIGHTGBM_NUMERIC_FEATURES,
                *groups["hierarchy_context"],
                *groups["sibling_competition"],
                *groups["light_structure"],
            ],
            "categorical_features": groups["categorical"].copy(),
        },
    }
    masks["wo_hierarchy_context"] = {
        "numeric_features": [
            feature
            for feature in masks["full_enhanced"]["numeric_features"]
            if feature not in groups["hierarchy_context"]
        ],
        "categorical_features": groups["categorical"].copy(),
    }
    masks["wo_sibling_competition"] = {
        "numeric_features": [
            feature
            for feature in masks["full_enhanced"]["numeric_features"]
            if feature not in groups["sibling_competition"]
        ],
        "categorical_features": groups["categorical"].copy(),
    }
    validate_variant_masks(masks)
    return deepcopy(masks)


def validate_variant_masks(masks: dict[str, dict[str, list[str]]]) -> None:
    if tuple(masks) != ABLATION_VARIANTS:
        raise ValueError("feature masks variant order does not match contract")
    all_known_features = {
        feature
        for features in build_feature_groups().values()
        for feature in features
    }
    for variant, mask in masks.items():
        numeric_features = list(mask["numeric_features"])
        categorical_features = list(mask["categorical_features"])
        for feature in [*numeric_features, *categorical_features]:
            if feature in FORBIDDEN_FEATURES:
                raise ValueError(f"forbidden feature in {variant}: {feature}")
            if feature not in all_known_features:
                raise ValueError(f"unknown feature in {variant}: {feature}")
    current = masks["current_coarse_graph"]
    if tuple(current["numeric_features"]) != LIGHTGBM_NUMERIC_FEATURES:
        raise ValueError("current_coarse_graph numeric features drifted from stable LightGBM")
    if tuple(current["categorical_features"]) != LIGHTGBM_CATEGORICAL_FEATURES:
        raise ValueError("current_coarse_graph categorical features drifted from stable LightGBM")


def build_feature_mask_digest(
    variant: str,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> str:
    return digest_json_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "variant": variant,
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
        }
    )


def build_feature_schema() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, features in build_feature_groups().items():
        if not group.startswith("base") and group != "categorical" and group != "coarse_graph":
            for feature in features:
                rows.append(
                    {
                        "feature": feature,
                        "group": group,
                        "dtype": "float64",
                        "dynamic": group in {"hierarchy_context", "sibling_competition"},
                        "uses_target_information": False,
                        "debug_only": False,
                    }
                )
    return rows


def build_feature_groups_payload() -> dict[str, object]:
    masks = build_variant_feature_masks()
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_groups": build_feature_groups(),
        "feature_schema": build_feature_schema(),
        "variants": masks,
        "feature_mask_digest": {
            variant: build_feature_mask_digest(
                variant,
                numeric_features=mask["numeric_features"],
                categorical_features=mask["categorical_features"],
            )
            for variant, mask in masks.items()
        },
    }
```

- [ ] **Step 4: Run feature group tests**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_groups.py -q
```

Expected: PASS.

- [ ] **Step 5: Run LightGBM stability test**

Run:

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_lightgbm_constants_are_stable tests/test_trend_graph_feature_ablation_groups.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```sh
git add src/experiments/trend_graph_feature_ablation/feature_groups.py tests/test_trend_graph_feature_ablation_groups.py
git commit -m "feat(experiments): 定义图特征消融分组"
```

---

### Task 3: Graph Context Feature Builder

**Files:**
- Create: `src/experiments/trend_graph_feature_ablation/build_features.py`
- Test: `tests/test_trend_graph_feature_ablation_features.py`

- [ ] **Step 1: Write failing tests for graph aggregation**

Add the first section of `tests/test_trend_graph_feature_ablation_features.py`:

```python
from __future__ import annotations

import math

import pandas as pd

from experiments.trend_graph_feature_ablation.build_features import (
    build_graph_context_features,
)


def sample_graph_samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": [4, 4, 4, 4],
            "attr_id": ["parent::A", "parent::B", "child::X", "child::Y"],
            "attr_type": ["parent", "parent", "child", "child"],
            "attr_value": ["A", "B", "X", "Y"],
            "heat_t": [10, 30, 20, 5],
            "share_t": [0.25, 0.75, 0.80, 0.20],
            "growth_lag_1": [0.10, 0.30, 0.50, -0.10],
            "rank_in_type_t": [2, 1, 1, 2],
            "target_growth": [0.0, 0.0, 0.0, 0.0],
            "target_log_heat_t1": [0.0, 0.0, 0.0, 0.0],
            "target_rank_in_type_t1": [1, 1, 1, 1],
        }
    )


def sample_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_attr_id": ["parent::A", "parent::B", "parent::A"],
            "child_attr_id": ["child::X", "child::X", "child::Y"],
            "parent_attr_type": ["parent", "parent", "parent"],
            "child_attr_type": ["child", "child", "child"],
            "relation_type": ["contains", "contains", "contains"],
            "edge_weight": [2.0, 6.0, 2.0],
        }
    )


def test_parent_context_uses_edge_weight_normalized_average() -> None:
    features = build_graph_context_features(sample_graph_samples(), sample_edges())
    child_x = features.set_index("attr_id").loc["child::X"]

    assert math.isclose(child_x["kg_parent_share_t_wavg"], 0.625)
    assert math.isclose(child_x["kg_parent_growth_lag_1_wavg"], 0.25)
    assert math.isclose(child_x["kg_self_parent_share_gap_t"], 0.175)
    assert math.isclose(child_x["kg_self_parent_growth_gap_lag_1"], 0.25)
    assert int(child_x["kg_has_parent"]) == 1
    assert int(child_x["kg_is_root_attr"]) == 0


def test_child_context_uses_edge_weight_normalized_average() -> None:
    features = build_graph_context_features(sample_graph_samples(), sample_edges())
    parent_a = features.set_index("attr_id").loc["parent::A"]

    assert math.isclose(parent_a["kg_child_share_t_wavg"], 0.50)
    assert math.isclose(parent_a["kg_self_child_share_gap_t"], -0.25)
    assert math.isclose(parent_a["kg_self_child_growth_gap_lag_1"], -0.20)
    assert int(parent_a["kg_has_child"]) == 1
    assert int(parent_a["kg_is_leaf_attr"]) == 0


def test_sibling_context_uses_fixed_shared_parent_formula() -> None:
    features = build_graph_context_features(sample_graph_samples(), sample_edges())
    child_x = features.set_index("attr_id").loc["child::X"]

    assert int(child_x["kg_has_sibling"]) == 1
    assert int(child_x["kg_sibling_count"]) == 1
    assert math.isclose(child_x["kg_sibling_share_t_wavg"], 0.20)
    assert math.isclose(child_x["kg_self_vs_sibling_share_gap_t"], 0.60)


def test_root_leaf_and_no_sibling_values_are_zero_filled() -> None:
    features = build_graph_context_features(sample_graph_samples(), sample_edges())
    parent_b = features.set_index("attr_id").loc["parent::B"]

    assert int(parent_b["kg_has_parent"]) == 0
    assert int(parent_b["kg_is_root_attr"]) == 1
    assert int(parent_b["kg_has_sibling"]) == 0
    assert parent_b["kg_sibling_share_t_wavg"] == 0
```

- [ ] **Step 2: Run feature tests to verify failure**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_features.py -q
```

Expected: FAIL with `ImportError` for `build_graph_context_features`.

- [ ] **Step 3: Implement graph context feature builder**

Create `src/experiments/trend_graph_feature_ablation/build_features.py` with the graph aggregation functions:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.trend_graph_feature_ablation.feature_groups import (
    HIERARCHY_CONTEXT_FEATURES,
    LIGHT_STRUCTURE_FEATURES,
    SIBLING_COMPETITION_FEATURES,
)
from fashion_trend.foundation.dataframe import validate_required_columns

GRAPH_CONTEXT_SOURCE_COLUMNS = (
    "week_id",
    "attr_id",
    "attr_type",
    "heat_t",
    "share_t",
    "growth_lag_1",
    "rank_in_type_t",
)
EDGE_COLUMNS = (
    "parent_attr_id",
    "child_attr_id",
    "edge_weight",
)


def build_graph_context_features(
    samples_all: pd.DataFrame,
    hierarchy_edges: pd.DataFrame,
) -> pd.DataFrame:
    validate_required_columns(samples_all, GRAPH_CONTEXT_SOURCE_COLUMNS, source_name="enhanced graph samples")
    validate_required_columns(hierarchy_edges, EDGE_COLUMNS, source_name="attribute hierarchy edges")
    base = _build_base_context_frame(samples_all)
    parent_features = _aggregate_neighbor_features(
        base,
        hierarchy_edges,
        neighbor_column="parent_attr_id",
        target_column="child_attr_id",
        prefix="kg_parent",
    )
    child_features = _aggregate_neighbor_features(
        base,
        hierarchy_edges,
        neighbor_column="child_attr_id",
        target_column="parent_attr_id",
        prefix="kg_child",
    )
    sibling_features = _aggregate_sibling_features(base, hierarchy_edges)
    structure = _build_light_structure_features(base, hierarchy_edges)
    features = base.loc[:, ["week_id", "attr_id"]].copy()
    for frame in (parent_features, child_features, sibling_features, structure):
        features = features.merge(frame, on=["week_id", "attr_id"], how="left")
    kg_columns = [
        *HIERARCHY_CONTEXT_FEATURES,
        *SIBLING_COMPETITION_FEATURES,
        *LIGHT_STRUCTURE_FEATURES,
    ]
    features[kg_columns] = features[kg_columns].fillna(0)
    self_state = base.loc[:, ["week_id", "attr_id", "share_t", "growth_lag_1"]]
    features = features.merge(self_state, on=["week_id", "attr_id"], how="left")
    features["kg_self_parent_share_gap_t"] = np.where(
        features["kg_has_parent"].eq(1),
        features["share_t"] - features["kg_parent_share_t_wavg"],
        0.0,
    )
    features["kg_self_parent_growth_gap_lag_1"] = np.where(
        features["kg_has_parent"].eq(1),
        features["growth_lag_1"] - features["kg_parent_growth_lag_1_wavg"],
        0.0,
    )
    features["kg_self_child_share_gap_t"] = np.where(
        features["kg_has_child"].eq(1),
        features["share_t"] - features["kg_child_share_t_wavg"],
        0.0,
    )
    features["kg_self_child_growth_gap_lag_1"] = np.where(
        features["kg_has_child"].eq(1),
        features["growth_lag_1"] - features["kg_child_growth_lag_1_wavg"],
        0.0,
    )
    features = features.drop(columns=["share_t", "growth_lag_1"])
    _validate_kg_features(features, kg_columns)
    return features


def _build_base_context_frame(samples_all: pd.DataFrame) -> pd.DataFrame:
    base = samples_all.loc[:, list(GRAPH_CONTEXT_SOURCE_COLUMNS)].copy()
    type_counts = base.groupby(["week_id", "attr_type"])["attr_id"].transform("count")
    denominator = (type_counts - 1).clip(lower=1)
    base["rank_pct_t"] = (pd.to_numeric(base["rank_in_type_t"]) - 1) / denominator
    return base


def _aggregate_neighbor_features(
    base: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    neighbor_column: str,
    target_column: str,
    prefix: str,
) -> pd.DataFrame:
    links = edges.loc[:, [neighbor_column, target_column, "edge_weight"]].rename(
        columns={
            neighbor_column: "neighbor_attr_id",
            target_column: "target_attr_id",
        }
    )
    links["edge_weight"] = pd.to_numeric(links["edge_weight"], errors="raise")
    neighbor_values = base.rename(columns={"attr_id": "neighbor_attr_id"})
    joined = links.merge(
        neighbor_values,
        on="neighbor_attr_id",
        how="inner",
    )
    value_columns = {
        "heat_t": f"{prefix}_heat_t_wavg",
        "share_t": f"{prefix}_share_t_wavg",
        "growth_lag_1": f"{prefix}_growth_lag_1_wavg",
        "rank_pct_t": f"{prefix}_rank_pct_t_wavg",
    }
    if joined.empty:
        return base.loc[:, ["week_id", "attr_id"]].assign(**{name: 0.0 for name in value_columns.values()})
    records = []
    for (week_id, target_attr_id), group in joined.groupby(
        ["week_id", "target_attr_id"],
        sort=False,
    ):
        weights = group["edge_weight"].astype(float)
        total = float(weights.sum())
        row = {"week_id": week_id, "attr_id": target_attr_id}
        for source, output in value_columns.items():
            row[output] = float((group[source].astype(float) * weights).sum() / total)
        records.append(row)
    return pd.DataFrame(records)


def _aggregate_sibling_features(base: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    current_edges = edges.loc[
        :, ["parent_attr_id", "child_attr_id", "edge_weight"]
    ].rename(
        columns={
            "child_attr_id": "current_attr_id",
            "edge_weight": "current_edge_weight",
        }
    )
    sibling_edges = edges.loc[
        :, ["parent_attr_id", "child_attr_id", "edge_weight"]
    ].rename(
        columns={
            "child_attr_id": "sibling_attr_id",
            "edge_weight": "sibling_edge_weight",
        }
    )
    sibling_links = current_edges.merge(sibling_edges, on="parent_attr_id", how="inner")
    sibling_links = sibling_links[
        sibling_links["current_attr_id"] != sibling_links["sibling_attr_id"]
    ].copy()
    sibling_links["raw_weight"] = (
        pd.to_numeric(sibling_links["current_edge_weight"], errors="raise")
        * pd.to_numeric(sibling_links["sibling_edge_weight"], errors="raise")
    )
    sibling_links = (
        sibling_links.groupby(["current_attr_id", "sibling_attr_id"], as_index=False)[
            "raw_weight"
        ]
        .sum()
    )
    sibling_values = base.rename(columns={"attr_id": "sibling_attr_id"})
    joined = sibling_links.merge(
        sibling_values,
        on="sibling_attr_id",
        how="inner",
    )
    if joined.empty:
        return base.loc[:, ["week_id", "attr_id"]].assign(
            kg_sibling_count=0,
            kg_sibling_share_t_wavg=0.0,
            kg_sibling_share_t_max=0.0,
            kg_sibling_growth_lag_1_wavg=0.0,
            kg_sibling_rank_pct_t_wavg=0.0,
            kg_self_vs_sibling_share_gap_t=0.0,
            kg_self_vs_sibling_growth_gap_lag_1=0.0,
            kg_has_sibling=0,
        )
    self_values = base.loc[:, ["week_id", "attr_id", "share_t", "growth_lag_1"]].rename(
        columns={"attr_id": "current_attr_id"}
    )
    records = []
    for (week_id, current_attr_id), group in joined.groupby(
        ["week_id", "current_attr_id"],
        sort=False,
    ):
        weights = group["raw_weight"].astype(float)
        total = float(weights.sum())
        sibling_share = float((group["share_t"].astype(float) * weights).sum() / total)
        sibling_growth = float((group["growth_lag_1"].astype(float) * weights).sum() / total)
        self_row = self_values[
            (self_values["week_id"] == week_id)
            & (self_values["current_attr_id"] == current_attr_id)
        ].iloc[0]
        records.append(
            {
                "week_id": week_id,
                "attr_id": current_attr_id,
                "kg_sibling_count": int(group["sibling_attr_id"].nunique()),
                "kg_sibling_share_t_wavg": sibling_share,
                "kg_sibling_share_t_max": float(group["share_t"].max()),
                "kg_sibling_growth_lag_1_wavg": sibling_growth,
                "kg_sibling_rank_pct_t_wavg": float((group["rank_pct_t"].astype(float) * weights).sum() / total),
                "kg_self_vs_sibling_share_gap_t": float(self_row["share_t"]) - sibling_share,
                "kg_self_vs_sibling_growth_gap_lag_1": float(self_row["growth_lag_1"]) - sibling_growth,
                "kg_has_sibling": 1,
            }
        )
    return pd.DataFrame(records)


def _build_light_structure_features(base: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    parent_weight = edges.groupby("child_attr_id")["edge_weight"].sum().rename("kg_parent_edge_weight_sum")
    child_weight = edges.groupby("parent_attr_id")["edge_weight"].sum().rename("kg_child_edge_weight_sum")
    structure = base.loc[:, ["week_id", "attr_id"]].copy()
    structure = structure.merge(parent_weight, left_on="attr_id", right_index=True, how="left")
    structure = structure.merge(child_weight, left_on="attr_id", right_index=True, how="left")
    structure[["kg_parent_edge_weight_sum", "kg_child_edge_weight_sum"]] = structure[
        ["kg_parent_edge_weight_sum", "kg_child_edge_weight_sum"]
    ].fillna(0.0)
    structure["kg_parent_edge_weight_log"] = np.log1p(structure["kg_parent_edge_weight_sum"])
    structure["kg_child_edge_weight_log"] = np.log1p(structure["kg_child_edge_weight_sum"])
    structure["kg_has_parent"] = structure["kg_parent_edge_weight_sum"].gt(0).astype("int64")
    structure["kg_has_child"] = structure["kg_child_edge_weight_sum"].gt(0).astype("int64")
    structure["kg_is_root_attr"] = 1 - structure["kg_has_parent"]
    structure["kg_is_leaf_attr"] = 1 - structure["kg_has_child"]
    return structure


def _validate_kg_features(features: pd.DataFrame, kg_columns: list[str]) -> None:
    values = features.loc[:, kg_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("enhanced kg features contain non-finite values")
```

- [ ] **Step 4: Run feature tests**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_features.py -q
```

Expected: PASS. If a merge suffix bug appears, fix only `build_features.py` until the four tests pass.

- [ ] **Step 5: Commit Task 3**

```sh
git add src/experiments/trend_graph_feature_ablation/build_features.py tests/test_trend_graph_feature_ablation_features.py
git commit -m "feat(experiments): 构建图谱上下文特征"
```

---

### Task 4: Enhanced Samples, Hashes, And Row Alignment

**Files:**
- Modify: `src/experiments/trend_graph_feature_ablation/build_features.py`
- Modify: `src/experiments/trend_graph_feature_ablation/artifact_io.py`
- Test: `tests/test_trend_graph_feature_ablation_features.py`

- [ ] **Step 1: Add failing tests for enhanced samples and alignment**

Append to `tests/test_trend_graph_feature_ablation_features.py`:

```python
from experiments.trend_graph_feature_ablation.build_features import (
    build_enhanced_sample_frames,
    build_row_alignment_check,
)


def sample_split_samples() -> dict[str, pd.DataFrame]:
    all_samples = sample_graph_samples().assign(
        article_count=1,
        is_core_attr=1,
        parent_count=0,
        child_count=0,
        degree=0,
    )
    train = all_samples.iloc[:2].copy()
    train.insert(0, "split", "train")
    valid = all_samples.iloc[2:3].copy()
    valid.insert(0, "split", "valid")
    test = all_samples.iloc[3:].copy()
    test.insert(0, "split", "test")
    return {"all": all_samples, "train": train, "valid": valid, "test": test}


def test_enhanced_samples_keep_keys_order_and_targets() -> None:
    samples = sample_split_samples()
    enhanced = build_enhanced_sample_frames(
        samples["all"],
        {"train": samples["train"], "valid": samples["valid"], "test": samples["test"]},
        sample_edges(),
    )

    assert "split" not in enhanced["all"].columns
    assert enhanced["all"][["week_id", "attr_id"]].equals(samples["all"][["week_id", "attr_id"]])
    assert enhanced["train"][["split", "week_id", "attr_id"]].equals(
        samples["train"][["split", "week_id", "attr_id"]]
    )
    assert enhanced["all"]["target_growth"].equals(samples["all"]["target_growth"])
    assert "rank_pct_t" not in enhanced["all"].columns
    assert any(column.startswith("kg_") for column in enhanced["all"].columns)


def test_row_alignment_check_fails_on_target_drift() -> None:
    samples = sample_split_samples()
    enhanced = build_enhanced_sample_frames(
        samples["all"],
        {"train": samples["train"], "valid": samples["valid"], "test": samples["test"]},
        sample_edges(),
    )
    enhanced["all"].loc[0, "target_growth"] = 99

    result = build_row_alignment_check(
        samples["all"],
        {"train": samples["train"], "valid": samples["valid"], "test": samples["test"]},
        enhanced,
    )

    assert result["all"]["passed"] is False
    assert result["passed"] is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_features.py -q
```

Expected: FAIL because `build_enhanced_sample_frames` is missing.

- [ ] **Step 3: Add DataFrame checksum helper**

Extend `artifact_io.py`:

```python
import pandas as pd


def digest_dataframe_columns(dataframe: pd.DataFrame, columns: list[str]) -> str:
    payload = dataframe.loc[:, columns].astype("string").fillna("<NA>")
    encoded = payload.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Add enhanced sample frame functions**

Append to `build_features.py`:

```python
from experiments.trend_graph_feature_ablation.artifact_io import digest_dataframe_columns
from experiments.trend_graph_feature_ablation.contracts import (
    ALL_SAMPLE_KEY_COLUMNS,
    SPLIT_SAMPLE_KEY_COLUMNS,
    TARGET_COLUMNS,
)


def build_enhanced_sample_frames(
    samples_all: pd.DataFrame,
    split_samples: dict[str, pd.DataFrame],
    hierarchy_edges: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    graph_features = build_graph_context_features(samples_all, hierarchy_edges)
    enhanced_all = samples_all.merge(graph_features, on=["week_id", "attr_id"], how="left")
    enhanced_frames = {"all": enhanced_all}
    for split_name, split_frame in split_samples.items():
        split_keys = split_frame.loc[:, list(SPLIT_SAMPLE_KEY_COLUMNS)]
        frame = split_frame.merge(
            graph_features,
            on=["week_id", "attr_id"],
            how="left",
        )
        frame = frame.loc[:, [*split_frame.columns, *[column for column in graph_features.columns if column.startswith("kg_")]]]
        if not frame.loc[:, list(SPLIT_SAMPLE_KEY_COLUMNS)].equals(split_keys):
            raise ValueError(f"enhanced {split_name} row alignment changed")
        enhanced_frames[split_name] = frame
    result = build_row_alignment_check(samples_all, split_samples, enhanced_frames)
    if not result["passed"]:
        raise ValueError("enhanced sample row alignment failed")
    return enhanced_frames


def build_row_alignment_check(
    samples_all: pd.DataFrame,
    split_samples: dict[str, pd.DataFrame],
    enhanced_frames: dict[str, pd.DataFrame],
) -> dict[str, object]:
    checks: dict[str, object] = {}
    checks["all"] = _alignment_entry(
        samples_all,
        enhanced_frames["all"],
        key_columns=list(ALL_SAMPLE_KEY_COLUMNS),
    )
    for split_name, split_frame in split_samples.items():
        checks[split_name] = _alignment_entry(
            split_frame,
            enhanced_frames[split_name],
            key_columns=list(SPLIT_SAMPLE_KEY_COLUMNS),
        )
    checks["passed"] = all(bool(checks[name]["passed"]) for name in checks if name != "passed")
    return checks


def _alignment_entry(
    original: pd.DataFrame,
    enhanced: pd.DataFrame,
    *,
    key_columns: list[str],
) -> dict[str, object]:
    target_columns = list(TARGET_COLUMNS)
    input_key_checksum = digest_dataframe_columns(original, key_columns)
    output_key_checksum = digest_dataframe_columns(enhanced, key_columns)
    input_target_checksum = digest_dataframe_columns(original, target_columns)
    output_target_checksum = digest_dataframe_columns(enhanced, target_columns)
    return {
        "input_rows": int(len(original)),
        "output_rows": int(len(enhanced)),
        "key_columns": key_columns,
        "input_key_checksum": input_key_checksum,
        "output_key_checksum": output_key_checksum,
        "input_target_checksum": input_target_checksum,
        "output_target_checksum": output_target_checksum,
        "order_matches": input_key_checksum == output_key_checksum,
        "target_matches": input_target_checksum == output_target_checksum,
        "passed": (
            len(original) == len(enhanced)
            and input_key_checksum == output_key_checksum
            and input_target_checksum == output_target_checksum
        ),
    }
```

- [ ] **Step 5: Run enhanced sample tests**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_features.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```sh
git add src/experiments/trend_graph_feature_ablation/build_features.py src/experiments/trend_graph_feature_ablation/artifact_io.py tests/test_trend_graph_feature_ablation_features.py
git commit -m "feat(experiments): 校验增强样本对齐"
```

---

### Task 5: Experiment Training Runs

**Files:**
- Create: `src/experiments/trend_graph_feature_ablation/train_runs.py`
- Test: `tests/test_trend_graph_feature_ablation_runner.py`

- [ ] **Step 1: Write failing train run tests with monkeypatch**

Add `tests/test_trend_graph_feature_ablation_runner.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.trend_graph_feature_ablation.train_runs import run_single_variant
from tests.trend_samples import sample_trend_model_samples_for_split


class FakeModel:
    best_iteration_ = 7
    best_score_ = {"valid_0": {"l2": 0.1}}

    def __init__(self, feature_names: list[str]):
        self._feature_names = feature_names
        self.booster_ = self

    def predict(self, features: pd.DataFrame, num_iteration: int | None = None):
        return features["growth_lag_1"].astype(float).to_numpy()

    def feature_name(self):
        return self._feature_names

    def feature_importance(self, importance_type: str):
        return [1 for _ in self._feature_names]

    def model_to_string(self):
        return "fake model"


def split_frames() -> dict[str, pd.DataFrame]:
    samples = sample_trend_model_samples_for_split()
    train = samples[samples["week_id"] < 12].copy()
    train.insert(0, "split", "train")
    valid = samples[(samples["week_id"] >= 12) & (samples["week_id"] < 16)].copy()
    valid.insert(0, "split", "valid")
    test = samples[samples["week_id"] >= 16].copy()
    test.insert(0, "split", "test")
    return {"train": train, "valid": valid, "test": test}


def test_run_single_variant_writes_experiment_artifacts(tmp_path: Path, monkeypatch) -> None:
    import experiments.trend_graph_feature_ablation.train_runs as train_runs

    def fake_fit(train_features, train_target, valid_features, valid_target, *, config):
        return FakeModel(train_features.columns.tolist())

    monkeypatch.setattr(train_runs, "_fit_lightgbm_model", fake_fit)

    output_dir = tmp_path / "runs" / "no_graph"
    result = run_single_variant(
        "no_graph",
        split_frames(),
        output_dir=output_dir,
        input_hashes={"items": []},
        experiment_root=tmp_path,
    )

    assert result["variant"] == "no_graph"
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "feature_importance.csv").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "params.json").exists()
    assert (output_dir / "model.txt").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["best_iteration"] == 7
    assert metadata["training_elapsed_seconds"] >= 0
    assert len(metadata["feature_mask_digest"]) == 64


def test_run_single_variant_rejects_unknown_valid_attr_type(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import experiments.trend_graph_feature_ablation.train_runs as train_runs

    def fake_fit(train_features, train_target, valid_features, valid_target, *, config):
        return FakeModel(train_features.columns.tolist())

    frames = split_frames()
    frames["valid"].loc[frames["valid"].index[0], "attr_type"] = "new_type"
    monkeypatch.setattr(train_runs, "_fit_lightgbm_model", fake_fit)

    with pytest.raises(ValueError, match="unknown attr_type"):
        run_single_variant(
            "no_graph",
            frames,
            output_dir=tmp_path / "runs" / "no_graph",
            input_hashes={"items": []},
            experiment_root=tmp_path,
        )
```

- [ ] **Step 2: Run test to verify failure**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_runner.py::test_run_single_variant_writes_experiment_artifacts -q
```

Expected: FAIL because `train_runs.py` is missing.

- [ ] **Step 3: Implement single variant training**

Create `src/experiments/trend_graph_feature_ablation/train_runs.py`:

```python
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pandas.api.types import CategoricalDtype

from experiments.trend_graph_feature_ablation.artifact_io import assert_experiment_write_path
from experiments.trend_graph_feature_ablation.feature_groups import (
    build_feature_mask_digest,
    build_variant_feature_masks,
)
from fashion_trend.foundation.io import (
    write_binary_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_EPSILON,
    LIGHTGBM_MODEL_NAME,
    _build_lightgbm_predictions,
    _fit_lightgbm_model,
    _read_best_iteration,
    build_feature_importance_frame,
)
from fashion_trend.trend.models.supervised.lightgbm_config import (
    resolve_lightgbm_config_from_stable_or_default,
)
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR


def run_single_variant(
    variant: str,
    split_frames: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    input_hashes: dict[str, object],
    experiment_root: Path | None = None,
) -> dict[str, object]:
    for filename in (
        "predictions.csv",
        "feature_importance.csv",
        "metadata.json",
        "params.json",
        "model.txt",
    ):
        if experiment_root is None:
            assert_experiment_write_path(output_dir / filename)
        else:
            assert_experiment_write_path(output_dir / filename, root=experiment_root)
    masks = build_variant_feature_masks()
    mask = masks[variant]
    config = resolve_lightgbm_config_from_stable_or_default(
        OUTPUT_MODELS_DIR / "lightgbm" / "params.json"
    )
    start = time.perf_counter()
    train_features, attr_type_categories = _select_features(
        split_frames["train"],
        mask,
        attr_type_categories=None,
    )
    valid_features, _ = _select_features(
        split_frames["valid"],
        mask,
        attr_type_categories=attr_type_categories,
    )
    train_target = split_frames["train"]["target_growth"].astype(float)
    valid_target = split_frames["valid"]["target_growth"].astype(float)
    model = _fit_lightgbm_model(
        train_features,
        train_target,
        valid_features,
        valid_target,
        config=config,
    )
    predictions = []
    for split_name in ("train", "valid", "test"):
        features, _ = _select_features(
            split_frames[split_name],
            mask,
            attr_type_categories=attr_type_categories,
        )
        pred = model.predict(features, num_iteration=_read_best_iteration(model))
        predictions.append(_build_lightgbm_predictions(split_frames[split_name], pred))
    prediction_frame = pd.concat(predictions, ignore_index=True)
    feature_importance = build_feature_importance_frame(model.booster_)
    elapsed = time.perf_counter() - start
    digest = build_feature_mask_digest(
        variant,
        numeric_features=mask["numeric_features"],
        categorical_features=mask["categorical_features"],
    )
    params = {
        "model_name": LIGHTGBM_MODEL_NAME,
        "epsilon": LIGHTGBM_EPSILON,
        "numeric_features": list(mask["numeric_features"]),
        "categorical_features": list(mask["categorical_features"]),
        "lightgbm_params": dict(config.lightgbm_params),
        "early_stopping": dict(config.early_stopping),
        "param_source": dict(config.param_source),
        "best_iteration": _read_best_iteration(model),
    }
    metadata = {
        "variant": variant,
        "feature_mask": mask,
        "feature_mask_digest": digest,
        "input_hashes": input_hashes,
        "best_iteration": _read_best_iteration(model),
        "training_elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
    }
    write_csv_atomic(prediction_frame, output_dir / "predictions.csv")
    write_csv_atomic(feature_importance, output_dir / "feature_importance.csv")
    write_json_atomic(metadata, output_dir / "metadata.json")
    write_json_atomic(params, output_dir / "params.json")
    write_binary_atomic(model.booster_.model_to_string().encode("utf-8"), output_dir / "model.txt")
    return metadata


def _select_features(
    samples: pd.DataFrame,
    mask: dict[str, list[str]],
    *,
    attr_type_categories: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    columns = [*mask["numeric_features"], *mask["categorical_features"]]
    frame = samples.loc[:, columns].copy()
    for column in mask["numeric_features"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if "attr_type" in frame.columns:
        attr_type = frame["attr_type"].astype(str)
        if attr_type_categories is None:
            categories = tuple(sorted(attr_type.unique()))
        else:
            categories = tuple(attr_type_categories)
            unknown = sorted(set(attr_type.unique()) - set(categories))
            if unknown:
                raise ValueError(f"unknown attr_type in experiment split: {unknown[:5]}")
        frame["attr_type"] = attr_type.astype(CategoricalDtype(categories=list(categories)))
        return frame, categories
    return frame, tuple()
```

- [ ] **Step 4: Run train run test**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_runner.py::test_run_single_variant_writes_experiment_artifacts -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```sh
git add src/experiments/trend_graph_feature_ablation/train_runs.py tests/test_trend_graph_feature_ablation_runner.py
git commit -m "feat(experiments): 运行图特征消融训练"
```

---

### Task 6: Evaluation And Summary

**Files:**
- Create: `src/experiments/trend_graph_feature_ablation/evaluate.py`
- Create: `src/experiments/trend_graph_feature_ablation/summarize.py`
- Test: `tests/test_trend_graph_feature_ablation_runner.py`

- [ ] **Step 1: Add failing tests for evaluation and summary**

Append to `tests/test_trend_graph_feature_ablation_runner.py`:

```python
from experiments.trend_graph_feature_ablation.evaluate import evaluate_variant_predictions
from experiments.trend_graph_feature_ablation.summarize import build_metrics_summary_frame


def test_evaluate_variant_predictions_writes_recall_at_10(tmp_path: Path) -> None:
    frames = split_frames()
    predictions = []
    for split_name, frame in frames.items():
        part = frame.loc[
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
        part.insert(4, "model_name", "lightgbm")
        part["pred_target_growth"] = part["target_growth"]
        part["pred_share_t1"] = part["share_t"]
        predictions.append(
            part.loc[
                :,
                [
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
                ],
            ]
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    output_path = tmp_path / "metrics.json"

    payload = evaluate_variant_predictions(
        "no_graph",
        prediction_frame,
        prediction_path=tmp_path / "predictions.csv",
        output_path=output_path,
        experiment_root=tmp_path,
    )

    assert output_path.exists()
    assert "recall_at_k" in payload["overall"]["valid"]
    assert "10" in payload["overall"]["valid"]["recall_at_k"]


def test_build_metrics_summary_frame_reads_recall_at_10() -> None:
    payloads = {
        "no_graph": {
            "overall": {
                "valid": {
                    "ndcg_at_k": {"10": 0.1},
                    "spearman": 0.2,
                    "precision_at_k": {"10": 0.3},
                    "recall_at_k": {"10": 0.4},
                    "mae": 0.5,
                    "rmse": 0.6,
                },
                "test": {
                    "ndcg_at_k": {"10": 0.7},
                    "spearman": 0.8,
                    "precision_at_k": {"10": 0.9},
                    "recall_at_k": {"10": 1.0},
                    "mae": 1.1,
                    "rmse": 1.2,
                },
            }
        }
    }
    metadata = {"no_graph": {"feature_mask": {"numeric_features": ["a"], "categorical_features": ["b"]}}}

    summary = build_metrics_summary_frame(payloads, metadata)

    assert summary.loc[0, "valid_recall_at_10"] == 0.4
    assert summary.loc[0, "test_recall_at_10"] == 1.0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_runner.py -q
```

Expected: FAIL because `evaluate.py` and `summarize.py` are missing.

- [ ] **Step 3: Implement evaluation**

Create `src/experiments/trend_graph_feature_ablation/evaluate.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.trend_graph_feature_ablation.artifact_io import assert_experiment_write_path
from fashion_trend.trend.evaluation.payloads import (
    build_trend_metrics_payload,
    write_trend_metrics,
)


def evaluate_variant_predictions(
    variant: str,
    predictions: pd.DataFrame,
    *,
    prediction_path: Path,
    output_path: Path,
    experiment_root: Path | None = None,
) -> dict[str, object]:
    if experiment_root is None:
        assert_experiment_write_path(output_path)
    else:
        assert_experiment_write_path(output_path, root=experiment_root)
    payload = build_trend_metrics_payload(
        predictions,
        model_name="lightgbm",
        prediction_path=prediction_path,
        output_path=output_path,
        run_id=variant,
    )
    write_trend_metrics(payload, output_path)
    return payload
```

- [ ] **Step 4: Implement summary**

Create `src/experiments/trend_graph_feature_ablation/summarize.py`:

```python
from __future__ import annotations

import pandas as pd

from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_VARIANTS,
    SUMMARY_COLUMNS,
)


def build_metrics_summary_frame(
    metrics_payloads: dict[str, dict[str, object]],
    metadata_payloads: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for variant in ABLATION_VARIANTS:
        metrics = metrics_payloads[variant]["overall"]
        metadata = metadata_payloads[variant]
        mask = metadata["feature_mask"]
        rows.append(
            {
                "variant": variant,
                "feature_count": len(mask["numeric_features"]) + len(mask["categorical_features"]),
                "valid_ndcg_at_10": _metric(metrics, "valid", "ndcg_at_k", "10"),
                "valid_spearman": _metric(metrics, "valid", "spearman"),
                "valid_precision_at_10": _metric(metrics, "valid", "precision_at_k", "10"),
                "valid_recall_at_10": _metric(metrics, "valid", "recall_at_k", "10"),
                "valid_mae": _metric(metrics, "valid", "mae"),
                "valid_rmse": _metric(metrics, "valid", "rmse"),
                "test_ndcg_at_10": _metric(metrics, "test", "ndcg_at_k", "10"),
                "test_spearman": _metric(metrics, "test", "spearman"),
                "test_precision_at_10": _metric(metrics, "test", "precision_at_k", "10"),
                "test_recall_at_10": _metric(metrics, "test", "recall_at_k", "10"),
                "test_mae": _metric(metrics, "test", "mae"),
                "test_rmse": _metric(metrics, "test", "rmse"),
            }
        )
    return pd.DataFrame(rows).loc[:, list(SUMMARY_COLUMNS)]


def render_metrics_summary_markdown(summary: pd.DataFrame) -> str:
    return summary.to_markdown(index=False) + "\n"


def _metric(metrics: dict[str, object], split: str, key: str, nested: str | None = None):
    value = metrics[split][key]
    if nested is not None:
        value = value[nested]
    return value
```

- [ ] **Step 5: Run evaluation and summary tests**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```sh
git add src/experiments/trend_graph_feature_ablation/evaluate.py src/experiments/trend_graph_feature_ablation/summarize.py tests/test_trend_graph_feature_ablation_runner.py
git commit -m "feat(experiments): 汇总图特征消融指标"
```

---

### Task 7: Experiment Runner, CLI, And Docs

**Files:**
- Create: `src/experiments/trend_graph_feature_ablation/write_docs.py`
- Create: `src/experiments/trend_graph_feature_ablation/runner.py`
- Create: `src/19_run_trend_graph_feature_ablation.py`
- Modify: `README.md`
- Test: `tests/test_trend_graph_feature_ablation_runner.py`

- [ ] **Step 1: Add failing CLI orchestration test**

Append to `tests/test_trend_graph_feature_ablation_runner.py`:

```python
from experiments.trend_graph_feature_ablation.write_docs import render_experiment_doc


def test_render_experiment_doc_states_non_stable_boundary() -> None:
    text = render_experiment_doc(
        summary_markdown="| variant | valid_ndcg_at_10 |\n| --- | --- |\n| no_graph | 0.1 |\n",
        command="uv run python src/19_run_trend_graph_feature_ablation.py",
    )

    assert "非 stable 独立实验" in text
    assert "outputs/models/lightgbm" in text
    assert "outputs/reports/manifest.json" in text
```

- [ ] **Step 2: Run test to verify failure**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_runner.py::test_render_experiment_doc_states_non_stable_boundary -q
```

Expected: FAIL because `write_docs.py` is missing.

- [ ] **Step 3: Implement docs renderer**

Create `src/experiments/trend_graph_feature_ablation/write_docs.py`:

```python
from __future__ import annotations


def render_experiment_doc(*, summary_markdown: str, command: str) -> str:
    return (
        "# 趋势图谱特征消融实验\n\n"
        "本文件由独立实验入口生成，记录图谱上下文特征组级消融结果。\n\n"
        "## 边界\n\n"
        "- 这是非 stable 独立实验。\n"
        "- 不覆盖 `outputs/models/lightgbm/`。\n"
        "- 不写入 `outputs/reports/manifest.json`。\n"
        "- 不改变 defense app 数据源。\n\n"
        "## 运行命令\n\n"
        f"```sh\n{command}\n```\n\n"
        "## 指标汇总\n\n"
        f"{summary_markdown}\n"
    )
```

- [ ] **Step 4: Implement runner orchestration**

Create `src/experiments/trend_graph_feature_ablation/runner.py`:

```python
from __future__ import annotations

import json
import uuid

import pandas as pd

from experiments.trend_graph_feature_ablation.artifact_io import (
    assert_experiment_write_path,
    build_input_hash_entry,
)
from experiments.trend_graph_feature_ablation.contracts import ABLATION_VARIANTS, SCHEMA_VERSION
from experiments.trend_graph_feature_ablation.evaluate import evaluate_variant_predictions
from experiments.trend_graph_feature_ablation.feature_groups import build_feature_groups_payload
from experiments.trend_graph_feature_ablation.paths import (
    EXPERIMENT_DOC_PATH,
    FEATURE_GROUPS_PATH,
    FEATURE_SCHEMA_PATH,
    INPUT_HASHES_PATH,
    MANIFEST_PATH,
    METRICS_SUMMARY_CSV_PATH,
    METRICS_SUMMARY_MD_PATH,
    ROW_ALIGNMENT_CHECK_PATH,
    run_artifact_path,
    run_dir,
    staging_run_dir,
)
from experiments.trend_graph_feature_ablation.summarize import (
    build_metrics_summary_frame,
    render_metrics_summary_markdown,
)
from experiments.trend_graph_feature_ablation.train_runs import run_single_variant
from experiments.trend_graph_feature_ablation.write_docs import render_experiment_doc
from fashion_trend.catalog.paths import (
    GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH,
    GRAPH_NODES_ATTRIBUTE_PATH,
)
from fashion_trend.catalog.readers import read_attribute_hierarchy_edges
from fashion_trend.foundation.io import (
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from fashion_trend.trend.paths import (
    OUTPUT_MODELS_DIR,
    TREND_MODEL_SAMPLES_PATH,
    TREND_MODEL_SAMPLES_TEST_PATH,
    TREND_MODEL_SAMPLES_TRAIN_PATH,
    TREND_MODEL_SAMPLES_VALID_PATH,
)
from experiments.trend_graph_feature_ablation.build_features import (
    build_enhanced_sample_frames,
    build_row_alignment_check,
)


def run_trend_graph_feature_ablation() -> dict[str, object]:
    command = "uv run python src/19_run_trend_graph_feature_ablation.py"
    token = uuid.uuid4().hex
    staging = staging_run_dir(token)
    split_frames = _read_split_frames()
    samples_all = pd.read_parquet(TREND_MODEL_SAMPLES_PATH)
    hierarchy_edges = read_attribute_hierarchy_edges(GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH)
    enhanced = build_enhanced_sample_frames(samples_all, split_frames, hierarchy_edges)
    row_alignment = build_row_alignment_check(samples_all, split_frames, enhanced)
    input_hashes = _build_input_hashes(samples_all, split_frames, hierarchy_edges)
    groups_payload = build_feature_groups_payload()

    for path in (
        FEATURE_GROUPS_PATH,
        FEATURE_SCHEMA_PATH,
        ROW_ALIGNMENT_CHECK_PATH,
        INPUT_HASHES_PATH,
        METRICS_SUMMARY_CSV_PATH,
        METRICS_SUMMARY_MD_PATH,
        EXPERIMENT_DOC_PATH,
        MANIFEST_PATH,
    ):
        assert_experiment_write_path(path)

    write_parquet_atomic(enhanced["all"], staging / "features" / "enhanced_samples_all.parquet")
    write_parquet_atomic(enhanced["train"], staging / "features" / "enhanced_samples_train.parquet")
    write_parquet_atomic(enhanced["valid"], staging / "features" / "enhanced_samples_valid.parquet")
    write_parquet_atomic(enhanced["test"], staging / "features" / "enhanced_samples_test.parquet")
    write_json_atomic(groups_payload, staging / "features" / "feature_groups.json")
    write_json_atomic({"schema_version": SCHEMA_VERSION, "features": groups_payload["feature_schema"]}, staging / "features" / "feature_schema.json")
    write_json_atomic(row_alignment, staging / "features" / "row_alignment_check.json")
    write_json_atomic(input_hashes, staging / "features" / "input_hashes.json")

    metrics_payloads: dict[str, dict[str, object]] = {}
    metadata_payloads: dict[str, dict[str, object]] = {}
    for variant in ABLATION_VARIANTS:
        metadata = run_single_variant(
            variant,
            {"train": enhanced["train"], "valid": enhanced["valid"], "test": enhanced["test"]},
            output_dir=staging / "runs" / variant,
            input_hashes=input_hashes,
        )
        predictions = pd.read_csv(staging / "runs" / variant / "predictions.csv")
        metrics = evaluate_variant_predictions(
            variant,
            predictions,
            prediction_path=run_artifact_path(variant, "predictions.csv"),
            output_path=staging / "runs" / variant / "metrics.json",
        )
        metrics_payloads[variant] = metrics
        metadata_payloads[variant] = metadata

    summary = build_metrics_summary_frame(metrics_payloads, metadata_payloads)
    summary_md = render_metrics_summary_markdown(summary)
    write_csv_atomic(summary, staging / "metrics_summary.csv")
    write_text_atomic(summary_md, staging / "metrics_summary.md")
    write_text_atomic(render_experiment_doc(summary_markdown=summary_md, command=command), staging / "experiment.md")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "input_hashes": input_hashes,
        "variants": list(ABLATION_VARIANTS),
        "metrics_summary_path": str(METRICS_SUMMARY_CSV_PATH),
        "warnings": [],
    }
    write_json_atomic(manifest, staging / "manifest.json")
    _publish_staging(staging)
    return manifest


def _read_split_frames() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(TREND_MODEL_SAMPLES_TRAIN_PATH),
        "valid": pd.read_parquet(TREND_MODEL_SAMPLES_VALID_PATH),
        "test": pd.read_parquet(TREND_MODEL_SAMPLES_TEST_PATH),
    }


def _build_input_hashes(
    samples_all: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    hierarchy_edges: pd.DataFrame,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "items": [
            build_input_hash_entry(TREND_MODEL_SAMPLES_PATH, required=True, row_count=len(samples_all)),
            build_input_hash_entry(TREND_MODEL_SAMPLES_TRAIN_PATH, required=True, row_count=len(split_frames["train"])),
            build_input_hash_entry(TREND_MODEL_SAMPLES_VALID_PATH, required=True, row_count=len(split_frames["valid"])),
            build_input_hash_entry(TREND_MODEL_SAMPLES_TEST_PATH, required=True, row_count=len(split_frames["test"])),
            build_input_hash_entry(GRAPH_NODES_ATTRIBUTE_PATH, required=True, row_count=None),
            build_input_hash_entry(GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH, required=True, row_count=len(hierarchy_edges)),
            build_input_hash_entry(OUTPUT_MODELS_DIR / "lightgbm" / "params.json", required=False, row_count=None),
        ],
    }


def _publish_staging(staging) -> None:
    publish_map = {
        staging / "features" / "enhanced_samples_all.parquet": "features/enhanced_samples_all.parquet",
        staging / "features" / "enhanced_samples_train.parquet": "features/enhanced_samples_train.parquet",
        staging / "features" / "enhanced_samples_valid.parquet": "features/enhanced_samples_valid.parquet",
        staging / "features" / "enhanced_samples_test.parquet": "features/enhanced_samples_test.parquet",
        staging / "features" / "feature_groups.json": "features/feature_groups.json",
        staging / "features" / "feature_schema.json": "features/feature_schema.json",
        staging / "features" / "row_alignment_check.json": "features/row_alignment_check.json",
        staging / "features" / "input_hashes.json": "features/input_hashes.json",
        staging / "metrics_summary.csv": "metrics_summary.csv",
        staging / "metrics_summary.md": "metrics_summary.md",
        staging / "experiment.md": "experiment.md",
        staging / "manifest.json": "manifest.json",
    }
    from experiments.trend_graph_feature_ablation.paths import EXPERIMENT_ROOT
    for source, relative in publish_map.items():
        destination = EXPERIMENT_ROOT / relative
        assert_experiment_write_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    for variant in ABLATION_VARIANTS:
        for filename in ("predictions.csv", "metrics.json", "feature_importance.csv", "metadata.json", "params.json", "model.txt"):
            source = staging / "runs" / variant / filename
            destination = run_dir(variant) / filename
            assert_experiment_write_path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
```

- [ ] **Step 5: Add CLI**

Create `src/19_run_trend_graph_feature_ablation.py`:

```python
from __future__ import annotations

from fashion_trend.foundation import logging as log
from experiments.trend_graph_feature_ablation.runner import (
    run_trend_graph_feature_ablation,
)

LOG_SOURCE = "trend-graph-feature-ablation"


def main() -> int:
    try:
        manifest = run_trend_graph_feature_ablation()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1
    log.info("趋势图谱特征消融实验完成", source=LOG_SOURCE)
    log.info(f"schema_version: {manifest['schema_version']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Update README with independent experiment section**

Add a short section near the LightGBM/reports experimental notes in `README.md`:

```markdown
### 19. 趋势图谱特征消融实验（非默认主流程）

`src/19_run_trend_graph_feature_ablation.py` 是独立补充实验入口，用于在不改变默认 `trend_model_samples*.parquet`、stable LightGBM、reports 和 defense app 的前提下，构建图谱上下文增强样本并运行组级消融。

运行命令：

```sh
uv run python src/19_run_trend_graph_feature_ablation.py
```

实验只写入：

```text
outputs/experiments/trend_graph_feature_ablation/
```

该入口不属于默认 `00-18` stable pipeline；结果如需进入论文 reports，应后续通过非默认 experimental 导出命令处理。
```

- [ ] **Step 7: Run focused tests and compile**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_*.py -q
uv run python -m compileall -q src
```

Expected: PASS.

- [ ] **Step 8: Commit Task 7**

```sh
git add src/19_run_trend_graph_feature_ablation.py src/experiments/trend_graph_feature_ablation/runner.py src/experiments/trend_graph_feature_ablation/write_docs.py README.md tests/test_trend_graph_feature_ablation_runner.py
git commit -m "feat(experiments): 编排趋势图特征消融实验"
```

---

### Task 8: Architecture Boundaries And Final Verification

**Files:**
- Modify: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: Add architecture boundary test for experiment isolation**

Add to `tests/test_architecture_boundaries.py`:

```python
def test_trend_graph_feature_ablation_does_not_modify_default_outputs() -> None:
    source_files = [
        Path("src/experiments/trend_graph_feature_ablation/runner.py"),
        Path("src/experiments/trend_graph_feature_ablation/train_runs.py"),
        Path("src/19_run_trend_graph_feature_ablation.py"),
    ]
    forbidden_write_strings = [
        "outputs/models/lightgbm",
        "outputs/metrics/lightgbm",
        "outputs/reports",
        "outputs/defense_app",
        "data/processed/features/trend_model_samples.parquet",
    ]
    for source_file in source_files:
        source = source_file.read_text(encoding="utf-8")
        for forbidden in forbidden_write_strings:
            assert forbidden not in source
```

- [ ] **Step 2: Run architecture and experiment tests**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_*.py tests/test_architecture_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full relevant verification**

Run:

```sh
uv run pytest tests/test_trend_graph_feature_ablation_*.py tests/test_trend_lightgbm.py tests/test_trend_evaluation.py tests/test_architecture_boundaries.py -q
uv run python -m compileall -q src
```

Expected: PASS.

- [ ] **Step 4: Run real artifact integration when inputs exist**

Run:

```sh
uv run python src/19_run_trend_graph_feature_ablation.py
```

Expected when input artifacts exist: exit code 0 and these files exist:

```text
outputs/experiments/trend_graph_feature_ablation/features/enhanced_samples_all.parquet
outputs/experiments/trend_graph_feature_ablation/features/feature_groups.json
outputs/experiments/trend_graph_feature_ablation/features/feature_schema.json
outputs/experiments/trend_graph_feature_ablation/features/row_alignment_check.json
outputs/experiments/trend_graph_feature_ablation/features/input_hashes.json
outputs/experiments/trend_graph_feature_ablation/metrics_summary.csv
outputs/experiments/trend_graph_feature_ablation/metrics_summary.md
outputs/experiments/trend_graph_feature_ablation/experiment.md
outputs/experiments/trend_graph_feature_ablation/manifest.json
```

Expected when input artifacts are absent in this worktree: exit code 1 with an error naming the missing artifact. Do not add fallback data.

- [ ] **Step 5: Inspect git status for generated outputs**

Run:

```sh
git status --short
```

Expected: only source/docs/test files are tracked. Generated `outputs/experiments/...` files should be ignored or untracked runtime artifacts and must not be staged.

- [ ] **Step 6: Commit final verification adjustments**

```sh
git add tests/test_architecture_boundaries.py
git commit -m "test: 保护图特征消融输出边界"
```

---

## Final Acceptance Checklist

- [ ] `uv run pytest tests/test_trend_graph_feature_ablation_*.py tests/test_architecture_boundaries.py -q` passes.
- [ ] `uv run python -m compileall -q src` passes.
- [ ] `src/19_run_trend_graph_feature_ablation.py` is the only CLI entry.
- [ ] `src/experiments/trend_graph_feature_ablation/` owns experiment logic.
- [ ] `current_coarse_graph` feature mask equals stable LightGBM feature constants.
- [ ] `rank_pct_t` is not written to enhanced samples.
- [ ] `predictions.csv` has the fixed trend prediction columns.
- [ ] `metrics.json` and `metrics_summary.csv/md` include `recall@10`.
- [ ] `metadata.json` includes `feature_mask_digest`, `best_iteration`, and `training_elapsed_seconds`.
- [ ] Write guards reject stable output and defense app paths.
- [ ] Generated experiment outputs are not committed.
- [ ] README describes the experiment as non-default and isolated.
