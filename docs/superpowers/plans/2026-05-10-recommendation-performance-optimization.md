# Recommendation Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将推荐 `12 -> 16` 全量生产流程从 GB 级 CSV 重复重建改造成带日志、Parquet 长表、freshness 门禁、feature cache 和精确 force 语义的可复用流水线。

**Architecture:** 保持编号脚本为薄 CLI，核心逻辑仍在 `src/fashion_trend/recommendation/`。先补观测，再迁移 artifact contract，然后统一 freshness，再引入 strategy/window 分区 feature cache，最后改造 `16` 的实验编排与 force 传播语义。

**Tech Stack:** Python 3.10-3.12、pandas、numpy、pyarrow、pytest、现有 `fashion_trend.foundation` IO/artifact helper；不新增 runtime 依赖。

---

## File Structure

- Modify: `src/fashion_trend/recommendation/paths.py`
  - 增加 `RECOMMEND_METADATA_PATH`、`FEATURES_DIR`、feature partition path、feature metadata path；将 method long item path 改为 `recommendation_items.parquet`，可选 CSV 使用独立 path。
- Modify: `src/fashion_trend/recommendation/fingerprints.py`
  - 扩展 artifact fingerprint helper，支持 path 存在性、size、mtime、schema/config/version payload。
- Create: `src/fashion_trend/recommendation/freshness.py`
  - 集中实现 input、candidate、feature cache、method output 的 freshness 校验和 stale 错误消息。
- Modify: `src/fashion_trend/recommendation/inputs.py`
  - `12` 写出 `data/processed/recommend/metadata.json`，记录 input artifacts、fingerprints、schema/version/config、row counts。
- Create: `src/fashion_trend/recommendation/perf.py`
  - 小型计时 helper，供 `12-16` 入口和 runner 记录 `stage=input_build rows=123 elapsed_seconds=0.5` 这类结构化日志。
- Modify: `src/fashion_trend/recommendation/outputs.py`
  - 默认写 `recommendation_items.parquet`，`RecommendationResultChunkWriter` 支持 chunk parquet 汇总和短表 CSV；CSV item export 不作为默认内部产物。
- Modify: `src/fashion_trend/recommendation/readers.py`
  - 读取 `recommendation_items.parquet`，继续读取 `recommendations.csv`，校验列、rank、Top-K、重复 key。
- Modify: `src/fashion_trend/recommendation/retrieval/candidates.py`
  - candidate metadata 增加 schema/version/config/row_counts；保持 source `top_n=12` 默认。
- Create: `src/fashion_trend/recommendation/features/cache.py`
  - 构建 `popularity_scores`、`recent_scores`、`similarity_scores`、`trend_scores`、`candidate_seen_flags`、`recommendable_pool` 的 strategy/window 分区 cache。
- Create: `src/fashion_trend/recommendation/features/__init__.py`
  - 导出 cache builder 和 cache reader。
- Modify: `src/fashion_trend/recommendation/ranking/features.py`
  - 复用已有 score 函数，并为 cache builder 提供按 window/strategy 计算的纯函数入口。
- Modify: `src/fashion_trend/recommendation/ranking/filters.py`
  - 提供 `build_candidate_seen_flags` 和基于 flags 的过滤函数。
- Modify: `src/fashion_trend/recommendation/runner.py`
  - method runner 改为按 window 读取 candidate + feature cache，写 Top-12 parquet 和短表 CSV，metadata 包含完整上游 artifact。
- Modify: `src/fashion_trend/recommendation/evaluation/runner.py`
  - 评价读取 `recommendable_pool` cache，避免每个 method 重新从 transactions 构建 pool。
- Modify: `src/fashion_trend/recommendation/experiments/runner.py`
  - 拆分 `--force-experiment`、`--force-method`、`--force-cache`、`--force-candidates`、`--force-rebuild-all` 语义；记录 reused/rebuilt/skipped 和阶段耗时。
- Modify: `src/12_build_recommendation_inputs.py`
- Modify: `src/13_build_recommend_candidates.py`
- Modify: `src/14_rerank_recommendations.py`
- Modify: `src/15_eval_recommendations.py`
- Modify: `src/16_run_recommendation_experiment.py`
  - 接入计时日志、metadata/freshness、feature cache、Parquet item output 和新的 force 参数。
- Modify: `tests/test_recommendation_inputs.py`
- Modify: `tests/test_recommendation_contracts.py`
- Modify: `tests/test_recommendation_ranking.py`
- Modify: `tests/test_recommendation_methods.py`
- Modify: `tests/test_recommendation_evaluation.py`
- Modify: `tests/test_recommendation_experiments.py`
- Create: `tests/test_recommendation_freshness.py`
- Create: `tests/test_recommendation_feature_cache.py`
  - 覆盖新 metadata、freshness、cache 分区、Parquet output、force stale 传播。
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`
- Modify: `AGENTS.md`
  - 同步推荐性能优化后的命令、artifact 契约和 force 语义。

Execution rule: 每个 Task 完成后运行该 Task 的聚焦测试和 `git diff --check`；每个 Task 独立 commit。不要提交 `data/`、`outputs/`、`.tmp`、`.part` 或真实推荐 artifact。

---

### Task 1: Instrumentation And Baseline Timing

**Files:**
- Create: `src/fashion_trend/recommendation/perf.py`
- Modify: `src/12_build_recommendation_inputs.py`
- Modify: `src/13_build_recommend_candidates.py`
- Modify: `src/14_rerank_recommendations.py`
- Modify: `src/15_eval_recommendations.py`
- Modify: `src/16_run_recommendation_experiment.py`
- Test: `tests/test_recommendation_experiments.py`

- [ ] **Step 1: Write failing tests for timing helper and experiment timing payload**

Append to `tests/test_recommendation_experiments.py`:

```python
from fashion_trend.recommendation.perf import StageTimer


def test_stage_timer_records_elapsed_and_rows() -> None:
    timer = StageTimer("feature_cache", rows=123, details={"name": "sim_score"})

    payload = timer.finish()

    assert payload["stage"] == "feature_cache"
    assert payload["rows"] == 123
    assert payload["name"] == "sim_score"
    assert payload["elapsed_seconds"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py::test_stage_timer_records_elapsed_and_rows -q
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `fashion_trend.recommendation.perf`.

- [ ] **Step 3: Implement timing helper**

Create `src/fashion_trend/recommendation/perf.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTimer:
    stage: str
    rows: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._started_at = time.perf_counter()

    def finish(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "elapsed_seconds": time.perf_counter() - self._started_at,
            **self.details,
        }
        if self.rows is not None:
            payload["rows"] = int(self.rows)
        return payload


def format_stage_log(payload: dict[str, Any]) -> str:
    parts = [f"stage={payload['stage']}"]
    for key, value in payload.items():
        if key == "stage":
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)
```

- [ ] **Step 4: Add timing logs to CLI scripts**

In each numbered script, wrap the main work with `StageTimer` and log `format_stage_log(payload)` via existing `log.info`. Example for `src/12_build_recommendation_inputs.py`:

```python
from fashion_trend.recommendation.perf import StageTimer, format_stage_log


timer = StageTimer("input_build")
artifacts = build_and_write_recommendation_inputs(
    transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
    article_attributes=read_article_attribute_edges(
        GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
    ),
    trend_predictions=read_trend_model_predictions(
        OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv"
    ),
)
payload = timer.finish()
log.info(format_stage_log({**payload, "rows": len(artifacts.target_users)}), source=LOG_SOURCE)
```

Use stage names:

```text
input_build
candidate_build
method
evaluation
experiment
```

- [ ] **Step 5: Verify tests and formatting**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py::test_stage_timer_records_elapsed_and_rows -q
uv run python -m compileall -q src
git diff --check
```

Expected: pytest PASS, compileall exits 0, diff check has no output.

- [ ] **Step 6: Commit**

```sh
git add src/fashion_trend/recommendation/perf.py src/12_build_recommendation_inputs.py src/13_build_recommend_candidates.py src/14_rerank_recommendations.py src/15_eval_recommendations.py src/16_run_recommendation_experiment.py tests/test_recommendation_experiments.py
git commit -m "feat(recommendation): 添加推荐阶段耗时日志"
```

---

### Task 2: Parquet Method Output Contract

**Files:**
- Modify: `src/fashion_trend/recommendation/paths.py`
- Modify: `src/fashion_trend/recommendation/outputs.py`
- Modify: `src/fashion_trend/recommendation/readers.py`
- Modify: `src/fashion_trend/recommendation/runner.py`
- Modify: `tests/test_recommendation_contracts.py`
- Modify: `tests/test_recommendation_methods.py`

- [ ] **Step 1: Write failing tests for Parquet item output**

Append to `tests/test_recommendation_methods.py`:

```python
def test_method_output_paths_use_parquet_items() -> None:
    paths = recommendation_paths.method_output_paths("pop_similarity")

    assert paths.recommendation_items.name == "recommendation_items.parquet"
    assert paths.recommendation_items_csv.name == "recommendation_items.csv"


def test_chunk_writer_writes_items_parquet_by_default(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "pop_similarity"
    monkeypatch.setattr(
        recommendation_paths,
        "method_output_paths",
        lambda method: recommendation_paths.RecommendationOutputPaths(
            output_dir=output_dir,
            recommendations=output_dir / "recommendations.csv",
            recommendation_items=output_dir / "recommendation_items.parquet",
            recommendation_items_csv=output_dir / "recommendation_items.csv",
            params=output_dir / "params.json",
            metadata=output_dir / "metadata.json",
            metrics=output_dir / "metrics.json",
        ),
    )
    from fashion_trend.recommendation.outputs import write_recommendation_result
    from fashion_trend.recommendation.methods.base import RecommendationResult

    result = RecommendationResult(
        recommendations=pd.DataFrame(columns=list(RECOMMENDATIONS_COLUMNS)),
        recommendation_items=pd.DataFrame(columns=list(RECOMMENDATION_ITEMS_COLUMNS)),
        params={"method": "pop_similarity"},
        metadata={"method": "pop_similarity"},
    )

    write_recommendation_result(result)

    assert (output_dir / "recommendations.csv").exists()
    assert (output_dir / "recommendation_items.parquet").exists()
    assert not (output_dir / "recommendation_items.csv").exists()
```

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```sh
uv run pytest tests/test_recommendation_methods.py::test_method_output_paths_use_parquet_items tests/test_recommendation_methods.py::test_chunk_writer_writes_items_parquet_by_default -q
```

Expected: FAIL because `RecommendationOutputPaths` does not yet include `recommendation_items_csv` and still points long item output to CSV.

- [ ] **Step 3: Update method output path dataclass**

Modify `src/fashion_trend/recommendation/paths.py`:

```python
@dataclass(frozen=True)
class RecommendationOutputPaths:
    output_dir: Path
    recommendations: Path
    recommendation_items: Path
    recommendation_items_csv: Path
    params: Path
    metadata: Path
    metrics: Path
```

Update `method_output_paths`:

```python
return RecommendationOutputPaths(
    output_dir=output_dir,
    recommendations=output_dir / "recommendations.csv",
    recommendation_items=output_dir / "recommendation_items.parquet",
    recommendation_items_csv=output_dir / "recommendation_items.csv",
    params=output_dir / "params.json",
    metadata=output_dir / "metadata.json",
    metrics=output_dir / "metrics.json",
)
```

- [ ] **Step 4: Update writer to write Parquet items**

Modify `src/fashion_trend/recommendation/outputs.py`:

```python
from fashion_trend.foundation.io import write_parquet_atomic


def write_recommendation_result(result: RecommendationResult) -> None:
    output_paths = method_output_paths(str(result.params["method"]))
    write_csv_atomic(result.recommendations, output_paths.recommendations)
    write_parquet_atomic(result.recommendation_items, output_paths.recommendation_items)
    write_json_atomic(result.params, output_paths.params)
    write_json_atomic(result.metadata, output_paths.metadata)
```

For `RecommendationResultChunkWriter`, collect item chunks per window and publish one parquet file:

```python
self._item_chunks: list[pd.DataFrame] = []

def write_chunk(self, result: RecommendationResult) -> None:
    if not self._started:
        raise RuntimeError("chunk writer has not been opened")
    _append_csv_rows(result.recommendations, self.recommendations_tmp)
    self._item_chunks.append(result.recommendation_items)

def publish(self) -> None:
    if not self._started:
        raise RuntimeError("chunk writer has not been opened")
    self.recommendations_tmp.replace(self.output_paths.recommendations)
    items = (
        pd.concat(self._item_chunks, ignore_index=True)
        if self._item_chunks
        else pd.DataFrame(columns=list(RECOMMENDATION_ITEMS_COLUMNS))
    )
    write_parquet_atomic(items, self.output_paths.recommendation_items)
```

Keep CSV chunking for `recommendations.csv`.

- [ ] **Step 5: Update readers and validation**

In `src/fashion_trend/recommendation/readers.py`, make `read_recommendation_items` read parquet:

```python
def read_recommendation_items(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    validate_columns(frame, RECOMMENDATION_ITEMS_COLUMNS, "recommendation_items")
    reject_duplicate_key(frame, RECOMMENDATION_ITEMS_KEY_COLUMNS, "recommendation_items")
    if not frame.empty and not frame["rank"].between(1, RECOMMENDATION_TOP_K).all():
        raise ValueError("recommendation_items rank out of Top-K range")
    return frame.loc[:, list(RECOMMENDATION_ITEMS_COLUMNS)]
```

- [ ] **Step 6: Verify method and contract tests**

Run:

```sh
uv run pytest tests/test_recommendation_contracts.py tests/test_recommendation_methods.py -q
uv run python -m compileall -q src
git diff --check
```

Expected: tests PASS, compileall exits 0, diff check has no output.

- [ ] **Step 7: Commit**

```sh
git add src/fashion_trend/recommendation/paths.py src/fashion_trend/recommendation/outputs.py src/fashion_trend/recommendation/readers.py src/fashion_trend/recommendation/runner.py tests/test_recommendation_contracts.py tests/test_recommendation_methods.py
git commit -m "feat(recommendation): 使用 Parquet 保存推荐明细"
```

---

### Task 3: Input Metadata And Unified Freshness

**Files:**
- Modify: `src/fashion_trend/recommendation/paths.py`
- Modify: `src/fashion_trend/recommendation/fingerprints.py`
- Create: `src/fashion_trend/recommendation/freshness.py`
- Modify: `src/fashion_trend/recommendation/inputs.py`
- Modify: `src/fashion_trend/recommendation/retrieval/candidates.py`
- Modify: `src/fashion_trend/recommendation/experiments/runner.py`
- Create: `tests/test_recommendation_freshness.py`
- Modify: `tests/test_recommendation_inputs.py`
- Modify: `tests/test_recommendation_experiments.py`

- [ ] **Step 1: Write failing tests for input metadata and stale checks**

Create `tests/test_recommendation_freshness.py`:

```python
from __future__ import annotations

import json

import pytest

from fashion_trend.recommendation.fingerprints import build_input_fingerprints
from fashion_trend.recommendation.freshness import (
    assert_fresh_metadata,
    build_artifact_metadata,
)


def test_build_artifact_metadata_records_versions_config_and_rows(tmp_path) -> None:
    artifact = tmp_path / "input.parquet"
    artifact.write_text("payload", encoding="utf-8")

    metadata = build_artifact_metadata(
        name="recommendation_inputs",
        input_artifacts={"input": str(artifact)},
        output_artifacts={"time_windows": str(artifact)},
        schema_version=1,
        algorithm_version="recommendation-inputs-v1",
        config={"top_k": 12},
        row_counts={"time_windows": 2},
    )

    assert metadata["schema_version"] == 1
    assert metadata["algorithm_version"] == "recommendation-inputs-v1"
    assert metadata["config"] == {"top_k": 12}
    assert metadata["row_counts"] == {"time_windows": 2}
    assert metadata["input_fingerprints"] == build_input_fingerprints({"input": str(artifact)})


def test_assert_fresh_metadata_rejects_changed_candidate_fingerprint(tmp_path) -> None:
    candidate = tmp_path / "candidate_items.parquet"
    candidate.write_text("old", encoding="utf-8")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            build_artifact_metadata(
                name="method",
                input_artifacts={"candidate_items": str(candidate)},
                output_artifacts={},
                schema_version=1,
                algorithm_version="method-v1",
                config={"method": "pop_similarity"},
                row_counts={},
            )
        ),
        encoding="utf-8",
    )
    candidate.write_text("new payload", encoding="utf-8")

    with pytest.raises(RuntimeError, match="input_fingerprints changed"):
        assert_fresh_metadata(
            metadata_path,
            expected_input_artifacts={"candidate_items": str(candidate)},
            expected_output_artifacts={},
            expected_schema_version=1,
            expected_algorithm_version="method-v1",
            expected_config={"method": "pop_similarity"},
            stale_message=lambda reason: f"stale: {reason}",
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```sh
uv run pytest tests/test_recommendation_freshness.py -q
```

Expected: FAIL because `freshness.py` does not exist.

- [ ] **Step 3: Implement metadata and freshness helpers**

Create `src/fashion_trend/recommendation/freshness.py`:

```python
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fashion_trend.recommendation.fingerprints import build_input_fingerprints


def build_artifact_metadata(
    *,
    name: str,
    input_artifacts: dict[str, str],
    output_artifacts: dict[str, str],
    schema_version: int,
    algorithm_version: str,
    config: dict[str, Any],
    row_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "name": name,
        "input_artifacts": dict(input_artifacts),
        "input_fingerprints": build_input_fingerprints(input_artifacts),
        "output_artifacts": dict(output_artifacts),
        "schema_version": schema_version,
        "algorithm_version": algorithm_version,
        "config": dict(config),
        "row_counts": {key: int(value) for key, value in row_counts.items()},
    }


def assert_fresh_metadata(
    metadata_path: Path,
    *,
    expected_input_artifacts: dict[str, str],
    expected_output_artifacts: dict[str, str],
    expected_schema_version: int,
    expected_algorithm_version: str,
    expected_config: dict[str, Any],
    stale_message: Callable[[str], str],
) -> None:
    if not metadata_path.exists():
        raise RuntimeError(stale_message("metadata.json is missing"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = {
        "input_artifacts": dict(expected_input_artifacts),
        "input_fingerprints": build_input_fingerprints(expected_input_artifacts),
        "output_artifacts": dict(expected_output_artifacts),
        "schema_version": expected_schema_version,
        "algorithm_version": expected_algorithm_version,
        "config": dict(expected_config),
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise RuntimeError(stale_message(f"{key} changed"))
```

- [ ] **Step 4: Add input metadata path and writer**

In `src/fashion_trend/recommendation/paths.py`:

```python
RECOMMEND_METADATA_PATH = RECOMMEND_DIR / "metadata.json"
```

In `src/fashion_trend/recommendation/inputs.py`, after writing four parquet inputs:

```python
from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.freshness import build_artifact_metadata
from fashion_trend.recommendation.paths import RECOMMEND_METADATA_PATH

write_json_atomic(
    build_artifact_metadata(
        name="recommendation_inputs",
        input_artifacts={},
        output_artifacts={
            "time_windows": str(TIME_WINDOWS_PATH),
            "target_users": str(TARGET_USERS_PATH),
            "evaluation_labels": str(EVALUATION_LABELS_PATH),
            "user_profile": str(USER_PROFILE_PATH),
        },
        schema_version=1,
        algorithm_version="recommendation-inputs-v1",
        config={"profile_top_attributes": RECOMMENDATION_PROFILE_TOP_ATTRIBUTES},
        row_counts={
            "time_windows": len(windows),
            "target_users": len(target_users),
            "evaluation_labels": len(labels),
            "user_profile": len(profile),
        },
    ),
    RECOMMEND_METADATA_PATH,
)
```

- [ ] **Step 5: Update candidate and method freshness callers**

Replace direct metadata comparisons in `src/fashion_trend/recommendation/experiments/runner.py` with `assert_fresh_metadata`. For method output, expected `output_artifacts` must include:

```python
{
    "recommendations": str(output_paths.recommendations),
    "recommendation_items": str(output_paths.recommendation_items),
    "params": str(output_paths.params),
    "metadata": str(output_paths.metadata),
}
```

Require all files to exist before reuse:

```python
for required_path in (
    output_paths.recommendations,
    output_paths.recommendation_items,
    output_paths.params,
    output_paths.metadata,
):
    if not required_path.exists():
        raise RuntimeError(_stale_output_message(method_name, f"{required_path.name} is missing"))
```

- [ ] **Step 6: Verify freshness tests**

Run:

```sh
uv run pytest tests/test_recommendation_freshness.py tests/test_recommendation_inputs.py tests/test_recommendation_experiments.py -q
uv run python -m compileall -q src
git diff --check
```

Expected: tests PASS, compileall exits 0, diff check has no output.

- [ ] **Step 7: Commit**

```sh
git add src/fashion_trend/recommendation/paths.py src/fashion_trend/recommendation/fingerprints.py src/fashion_trend/recommendation/freshness.py src/fashion_trend/recommendation/inputs.py src/fashion_trend/recommendation/retrieval/candidates.py src/fashion_trend/recommendation/experiments/runner.py tests/test_recommendation_freshness.py tests/test_recommendation_inputs.py tests/test_recommendation_experiments.py
git commit -m "feat(recommendation): 统一推荐产物新鲜度校验"
```

---

### Task 4: Strategy And Window Feature Cache

**Files:**
- Modify: `src/fashion_trend/recommendation/paths.py`
- Create: `src/fashion_trend/recommendation/features/__init__.py`
- Create: `src/fashion_trend/recommendation/features/cache.py`
- Modify: `src/fashion_trend/recommendation/ranking/features.py`
- Modify: `src/fashion_trend/recommendation/ranking/filters.py`
- Create: `tests/test_recommendation_feature_cache.py`
- Modify: `tests/test_recommendation_ranking.py`

- [ ] **Step 1: Write failing cache path and candidate seen tests**

Create `tests/test_recommendation_feature_cache.py`:

```python
from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.features.cache import (
    FEATURE_NAMES,
    build_candidate_seen_flags,
)
from fashion_trend.recommendation.paths import feature_cache_partition_path


def test_feature_cache_partition_path_is_strategy_window_scoped() -> None:
    path = feature_cache_partition_path(
        "candidate_seen_flags",
        strategy="default",
        split="valid",
        cutoff_week=10,
    )

    assert path.as_posix().endswith(
        "data/processed/recommend/features/candidate_seen_flags/strategy=default/split=valid/cutoff_week=10/part.parquet"
    )
    assert "candidate_seen_flags" in FEATURE_NAMES


def test_candidate_seen_flags_only_contains_seen_candidate_pairs() -> None:
    candidates = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "strategy": ["default", "default"],
            "customer_id": ["u1", "u1"],
            "article_id": ["a1", "a2"],
        }
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "article_id": ["a1", "a3", "a4"],
            "week_id": [10, 9, 11],
        }
    )

    flags = build_candidate_seen_flags(candidates, transactions)

    assert flags.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "strategy": "default",
            "customer_id": "u1",
            "article_id": "a1",
            "seen": True,
        }
    ]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```sh
uv run pytest tests/test_recommendation_feature_cache.py -q
```

Expected: FAIL because `features.cache` and `feature_cache_partition_path` do not exist.

- [ ] **Step 3: Add feature cache paths**

In `src/fashion_trend/recommendation/paths.py`:

```python
FEATURES_DIR = RECOMMEND_DIR / "features"
FEATURE_CACHE_METADATA_PATH = FEATURES_DIR / "metadata.json"


def feature_cache_partition_path(
    feature_name: str,
    *,
    strategy: str,
    split: str,
    cutoff_week: int,
) -> Path:
    _validate_recommendation_path_segment(feature_name, "feature_name")
    _validate_recommendation_path_segment(strategy, "strategy")
    _validate_recommendation_path_segment(split, "split")
    return (
        FEATURES_DIR
        / feature_name
        / f"strategy={strategy}"
        / f"split={split}"
        / f"cutoff_week={int(cutoff_week)}"
        / "part.parquet"
    )
```

- [ ] **Step 4: Implement candidate-scoped seen flags**

Create `src/fashion_trend/recommendation/features/cache.py` with the first function:

```python
from __future__ import annotations

import pandas as pd

FEATURE_NAMES = (
    "popularity_scores",
    "recent_scores",
    "similarity_scores",
    "trend_scores",
    "candidate_seen_flags",
    "recommendable_pool",
)
WINDOW_COLUMNS = ["split", "cutoff_week", "label_week"]


def build_candidate_seen_flags(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [
        *WINDOW_COLUMNS,
        "strategy",
        "customer_id",
        "article_id",
        "seen",
    ]
    if candidates.empty or transactions.empty:
        return pd.DataFrame(columns=output_columns)
    candidate_frame = candidates.copy()
    candidate_frame["customer_id"] = candidate_frame["customer_id"].astype(str)
    candidate_frame["article_id"] = candidate_frame["article_id"].astype(str)
    transaction_frame = transactions.copy()
    transaction_frame["customer_id"] = transaction_frame["customer_id"].astype(str)
    transaction_frame["article_id"] = transaction_frame["article_id"].astype(str)

    frames: list[pd.DataFrame] = []
    for window in candidate_frame[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        window_candidates = _frame_for_window(candidate_frame, window)
        history = transaction_frame.loc[
            pd.to_numeric(transaction_frame["week_id"], errors="raise")
            <= int(window["cutoff_week"]),
            ["customer_id", "article_id"],
        ].drop_duplicates()
        matched = window_candidates.merge(
            history,
            on=["customer_id", "article_id"],
            how="inner",
        )
        if matched.empty:
            continue
        matched["seen"] = True
        frames.append(matched.loc[:, output_columns])
    if not frames:
        return pd.DataFrame(columns=output_columns)
    return pd.concat(frames, ignore_index=True).drop_duplicates(output_columns[:-1])


def _frame_for_window(frame: pd.DataFrame, window: dict[str, object]) -> pd.DataFrame:
    return frame.loc[
        (frame["split"] == window["split"])
        & (frame["cutoff_week"] == window["cutoff_week"])
        & (frame["label_week"] == window["label_week"])
    ].copy()
```

Export in `src/fashion_trend/recommendation/features/__init__.py`:

```python
from fashion_trend.recommendation.features.cache import (
    FEATURE_NAMES,
    build_candidate_seen_flags,
)

__all__ = ["FEATURE_NAMES", "build_candidate_seen_flags"]
```

- [ ] **Step 5: Add cache builder for all feature partitions**

Extend `cache.py` with `build_and_write_feature_cache_for_strategy`:

```python
def build_and_write_feature_cache_for_strategy(
    *,
    strategy: str,
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame,
    trend_predictions: pd.DataFrame,
    input_artifacts: dict[str, str],
) -> dict[str, object]:
    manifest: dict[str, object] = {"strategy": strategy, "partitions": [], "row_counts": {}}
    for window in candidates[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        partition_candidates = _frame_for_window(candidates, window)
        feature_frame = build_ranking_features(
            partition_candidates,
            transactions,
            article_attributes,
            user_profile,
            trend_predictions,
        )
        seen_flags = build_candidate_seen_flags(partition_candidates, transactions)
        for feature_name, frame in {
            "similarity_scores": feature_frame.loc[:, [*WINDOW_COLUMNS, "strategy", "customer_id", "article_id", "sim_score"]],
            "popularity_scores": feature_frame.loc[:, [*WINDOW_COLUMNS, "strategy", "article_id", "pop_score"]].drop_duplicates(),
            "recent_scores": feature_frame.loc[:, [*WINDOW_COLUMNS, "strategy", "article_id", "recent_score"]].drop_duplicates(),
            "trend_scores": feature_frame.loc[:, [*WINDOW_COLUMNS, "strategy", "article_id", "trend_score"]].drop_duplicates(),
            "candidate_seen_flags": seen_flags,
        }.items():
            output_path = feature_cache_partition_path(
                feature_name,
                strategy=strategy,
                split=str(window["split"]),
                cutoff_week=int(window["cutoff_week"]),
            )
            write_parquet_atomic(frame, output_path)
            manifest["partitions"].append(str(output_path))
            manifest["row_counts"][str(output_path)] = int(len(frame))
    return manifest
```

Use `write_parquet_atomic`, `build_ranking_features`, and `feature_cache_partition_path`.

- [ ] **Step 6: Verify cache tests**

Run:

```sh
uv run pytest tests/test_recommendation_feature_cache.py tests/test_recommendation_ranking.py -q
uv run python -m compileall -q src
git diff --check
```

Expected: tests PASS, compileall exits 0, diff check has no output.

- [ ] **Step 7: Commit**

```sh
git add src/fashion_trend/recommendation/paths.py src/fashion_trend/recommendation/features src/fashion_trend/recommendation/ranking/features.py src/fashion_trend/recommendation/ranking/filters.py tests/test_recommendation_feature_cache.py tests/test_recommendation_ranking.py
git commit -m "feat(recommendation): 增加分区特征缓存"
```

---

### Task 5: Method Runner Uses Cache And Writes Fresh Metadata

**Files:**
- Modify: `src/fashion_trend/recommendation/runner.py`
- Modify: `src/fashion_trend/recommendation/methods/base.py`
- Modify: `src/fashion_trend/recommendation/ranking/filters.py`
- Modify: `src/14_rerank_recommendations.py`
- Modify: `tests/test_recommendation_methods.py`
- Modify: `tests/test_recommendation_freshness.py`

- [ ] **Step 1: Write failing tests for method metadata upstream chain**

Append to `tests/test_recommendation_methods.py`:

```python
def test_method_metadata_includes_candidate_and_feature_cache_artifacts(tmp_path, monkeypatch) -> None:
    candidate_path = tmp_path / "candidate_items.parquet"
    candidate_metadata = tmp_path / "candidate_metadata.json"
    cache_metadata = tmp_path / "feature_metadata.json"
    for path in (candidate_path, candidate_metadata, cache_metadata):
        path.write_text("x", encoding="utf-8")

    metadata = {
        "input_artifacts": {
            "recommendation_inputs": "data/processed/recommend/metadata.json",
            "candidate_items": str(candidate_path),
            "candidate_metadata": str(candidate_metadata),
            "feature_cache_metadata": str(cache_metadata),
        }
    }

    assert "candidate_items" in metadata["input_artifacts"]
    assert "candidate_metadata" in metadata["input_artifacts"]
    assert "feature_cache_metadata" in metadata["input_artifacts"]
```

This is a scaffold test. Replace the manual `metadata` construction with a call to the implemented runner metadata helper in Step 4.

- [ ] **Step 2: Run test**

Run:

```sh
uv run pytest tests/test_recommendation_methods.py::test_method_metadata_includes_candidate_and_feature_cache_artifacts -q
```

Expected: PASS as scaffold; after Step 4 it must assert real helper output.

- [ ] **Step 3: Add method input artifact helper**

In `src/fashion_trend/recommendation/runner.py` add:

```python
def method_input_artifacts(
    *,
    base_input_paths: dict[str, str],
    candidate_items: str | None,
    candidate_metadata: str | None,
    feature_cache_metadata: str | None,
    feature_partitions: list[str],
) -> dict[str, str]:
    artifacts = dict(base_input_paths)
    if candidate_items is not None:
        artifacts["candidate_items"] = candidate_items
    if candidate_metadata is not None:
        artifacts["candidate_metadata"] = candidate_metadata
    if feature_cache_metadata is not None:
        artifacts["feature_cache_metadata"] = feature_cache_metadata
    for index, path in enumerate(feature_partitions):
        artifacts[f"feature_partition_{index:04d}"] = path
    return artifacts
```

- [ ] **Step 4: Replace scaffold test with helper assertion**

Update `test_method_metadata_includes_candidate_and_feature_cache_artifacts`:

```python
from fashion_trend.recommendation.runner import method_input_artifacts


def test_method_metadata_includes_candidate_and_feature_cache_artifacts(tmp_path) -> None:
    candidate_path = tmp_path / "candidate_items.parquet"
    candidate_metadata = tmp_path / "candidate_metadata.json"
    cache_metadata = tmp_path / "feature_metadata.json"
    partition = tmp_path / "features" / "part.parquet"

    artifacts = method_input_artifacts(
        base_input_paths={"recommendation_inputs": "data/processed/recommend/metadata.json"},
        candidate_items=str(candidate_path),
        candidate_metadata=str(candidate_metadata),
        feature_cache_metadata=str(cache_metadata),
        feature_partitions=[str(partition)],
    )

    assert artifacts == {
        "recommendation_inputs": "data/processed/recommend/metadata.json",
        "candidate_items": str(candidate_path),
        "candidate_metadata": str(candidate_metadata),
        "feature_cache_metadata": str(cache_metadata),
        "feature_partition_0000": str(partition),
    }
```

- [ ] **Step 5: Wire runner metadata and cache consumption**

Modify `_base_metadata` to include:

```python
from fashion_trend.recommendation.freshness import build_artifact_metadata

metadata = build_artifact_metadata(
    name=f"recommendation_method:{method_name}",
    input_artifacts=dict(input_paths or {}),
    output_artifacts={
        "recommendations": str(method_output_paths(method_name).recommendations),
        "recommendation_items": str(method_output_paths(method_name).recommendation_items),
        "params": str(method_output_paths(method_name).params),
        "metadata": str(method_output_paths(method_name).metadata),
    },
    schema_version=1,
    algorithm_version="recommendation-method-v1",
    config={
        "method": method_name,
        "top_k": RECOMMENDATION_TOP_K,
        "required_features": list(required_features),
    },
    row_counts={},
)
```

Keep `window_config` and `trend_score_config` as additional metadata keys.

- [ ] **Step 6: Update `14` CLI input paths**

In `src/14_rerank_recommendations.py`, include:

```python
from fashion_trend.recommendation.paths import FEATURE_CACHE_METADATA_PATH, RECOMMEND_METADATA_PATH

input_paths = {
    "recommendation_inputs": str(RECOMMEND_METADATA_PATH),
    "weekly_transactions": str(WEEKLY_TRANSACTIONS_PATH),
    "article_attributes": str(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
    "time_windows": str(TIME_WINDOWS_PATH),
    "target_users": str(TARGET_USERS_PATH),
}
if candidate_path is not None:
    input_paths["candidate_items"] = str(candidate_path)
    input_paths["candidate_metadata"] = str(candidate_path.with_name("metadata.json"))
if FEATURE_CACHE_METADATA_PATH.exists():
    input_paths["feature_cache_metadata"] = str(FEATURE_CACHE_METADATA_PATH)
```

- [ ] **Step 7: Verify method tests**

Run:

```sh
uv run pytest tests/test_recommendation_methods.py tests/test_recommendation_freshness.py -q
uv run python -m compileall -q src
git diff --check
```

Expected: tests PASS, compileall exits 0, diff check has no output.

- [ ] **Step 8: Commit**

```sh
git add src/fashion_trend/recommendation/runner.py src/fashion_trend/recommendation/methods/base.py src/fashion_trend/recommendation/ranking/filters.py src/14_rerank_recommendations.py tests/test_recommendation_methods.py tests/test_recommendation_freshness.py
git commit -m "feat(recommendation): 记录方法产物完整输入链路"
```

---

### Task 6: Evaluation Reuses Recommendable Pool Cache

**Files:**
- Modify: `src/fashion_trend/recommendation/features/cache.py`
- Modify: `src/fashion_trend/recommendation/evaluation/runner.py`
- Modify: `src/15_eval_recommendations.py`
- Modify: `src/fashion_trend/recommendation/experiments/runner.py`
- Modify: `tests/test_recommendation_evaluation.py`
- Modify: `tests/test_recommendation_feature_cache.py`

- [ ] **Step 1: Write failing recommendable pool cache test**

Append to `tests/test_recommendation_feature_cache.py`:

```python
from fashion_trend.recommendation.features.cache import build_recommendable_pool


def test_recommendable_pool_uses_cutoff_history_only() -> None:
    windows = pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])
    transactions = pd.DataFrame(
        {
            "article_id": ["a1", "a2"],
            "week_id": [10, 11],
        }
    )

    pool = build_recommendable_pool(transactions, windows)

    assert pool.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "article_id": "a1",
        }
    ]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```sh
uv run pytest tests/test_recommendation_feature_cache.py::test_recommendable_pool_uses_cutoff_history_only -q
```

Expected: FAIL because `build_recommendable_pool` does not exist in `features.cache`.

- [ ] **Step 3: Move recommendable pool logic into cache module**

Add to `src/fashion_trend/recommendation/features/cache.py`:

```python
def build_recommendable_pool(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        active = (
            transactions.loc[
                transactions["week_id"] <= int(window.cutoff_week),
                ["article_id"],
            ]
            .drop_duplicates()
            .copy()
        )
        active["article_id"] = active["article_id"].astype("string")
        active = active.assign(
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
            label_week=int(window.label_week),
        )
        frames.append(active.loc[:, ["split", "cutoff_week", "label_week", "article_id"]])
    if not frames:
        return pd.DataFrame(columns=["split", "cutoff_week", "label_week", "article_id"])
    return pd.concat(frames, ignore_index=True)
```

Keep `build_recommendable_pool_for_windows` in `evaluation.runner` as a compatibility wrapper that calls this function.

- [ ] **Step 4: Update experiment evaluation to read cached pool**

In `src/fashion_trend/recommendation/experiments/runner.py`, replace repeated `build_recommendable_pool_for_windows(context.transactions, inputs.time_windows)` with:

```python
recommendable_pool = ensure_or_build_recommendable_pool_cache(context, inputs, force=force_cache)
```

Implement `ensure_or_build_recommendable_pool_cache` in the same module or import from `features.cache`, returning a DataFrame read from the feature cache partition for the needed split/window.

- [ ] **Step 5: Verify evaluation tests**

Run:

```sh
uv run pytest tests/test_recommendation_evaluation.py tests/test_recommendation_feature_cache.py tests/test_recommendation_experiments.py -q
uv run python -m compileall -q src
git diff --check
```

Expected: tests PASS, compileall exits 0, diff check has no output.

- [ ] **Step 6: Commit**

```sh
git add src/fashion_trend/recommendation/features/cache.py src/fashion_trend/recommendation/evaluation/runner.py src/fashion_trend/recommendation/experiments/runner.py src/15_eval_recommendations.py tests/test_recommendation_evaluation.py tests/test_recommendation_feature_cache.py
git commit -m "feat(recommendation): 复用推荐评价候选池缓存"
```

---

### Task 7: Experiment Force Semantics And Stale Propagation

**Files:**
- Modify: `src/fashion_trend/recommendation/experiments/runner.py`
- Modify: `src/16_run_recommendation_experiment.py`
- Modify: `tests/test_recommendation_experiments.py`

- [ ] **Step 1: Write failing tests for new force options**

Append to `tests/test_recommendation_experiments.py`:

```python
def test_experiment_runner_exposes_explicit_force_switches() -> None:
    signature = inspect.signature(run_recommendation_experiment)

    assert "force_experiment" in signature.parameters
    assert "force_methods" in signature.parameters
    assert "force_cache" in signature.parameters
    assert "force_candidates" in signature.parameters
    assert "force_rebuild_all" in signature.parameters


def test_force_cache_marks_method_outputs_stale() -> None:
    from fashion_trend.recommendation.experiments.runner import should_rebuild_method

    decision = should_rebuild_method(
        method_name="pop_similarity",
        stale_reason=None,
        force_methods=(),
        force_cache=True,
        force_candidates=False,
        force_rebuild_all=False,
    )

    assert decision.rebuild is True
    assert decision.reason == "force-cache"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py::test_experiment_runner_exposes_explicit_force_switches tests/test_recommendation_experiments.py::test_force_cache_marks_method_outputs_stale -q
```

Expected: FAIL because the new signature and `should_rebuild_method` do not exist.

- [ ] **Step 3: Add rebuild decision helper**

In `src/fashion_trend/recommendation/experiments/runner.py`:

```python
@dataclass(frozen=True)
class RebuildDecision:
    rebuild: bool
    reason: str


def should_rebuild_method(
    *,
    method_name: str,
    stale_reason: str | None,
    force_methods: tuple[str, ...],
    force_cache: bool,
    force_candidates: bool,
    force_rebuild_all: bool,
) -> RebuildDecision:
    if force_rebuild_all:
        return RebuildDecision(True, "force-rebuild-all")
    if method_name in force_methods:
        return RebuildDecision(True, "force-method")
    if force_candidates:
        return RebuildDecision(True, "force-candidates")
    if force_cache:
        return RebuildDecision(True, "force-cache")
    if stale_reason is not None:
        return RebuildDecision(True, stale_reason)
    return RebuildDecision(False, "fresh")
```

- [ ] **Step 4: Update experiment runner signature**

Change `run_recommendation_experiment` signature:

```python
def run_recommendation_experiment(
    context: RecommendationExperimentContext,
    experiment_id: str = "main",
    force_experiment: bool = False,
    force_methods: tuple[str, ...] = (),
    force_cache: bool = False,
    force_candidates: bool = False,
    force_rebuild_all: bool = False,
) -> dict[str, Any]:
```

Map old `force=True` callers only in CLI compatibility layer; inside runner use explicit switches.

- [ ] **Step 5: Update CLI arguments**

In `src/16_run_recommendation_experiment.py`:

```python
parser.add_argument("--force-experiment", action="store_true")
parser.add_argument("--force-method", action="append", default=[])
parser.add_argument("--force-cache", action="store_true")
parser.add_argument("--force-candidates", action="store_true")
parser.add_argument("--force-rebuild-all", action="store_true")
parser.add_argument(
    "--force",
    action="store_true",
    help="Deprecated alias for --force-experiment.",
)
```

Call runner:

```python
payload = run_recommendation_experiment(
    experiment_id=args.experiment,
    force_experiment=args.force_experiment or args.force,
    force_methods=tuple(args.force_method),
    force_cache=args.force_cache,
    force_candidates=args.force_candidates,
    force_rebuild_all=args.force_rebuild_all,
    context=RecommendationExperimentContext(
        transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
        article_attributes=read_article_attribute_edges(
            GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
        ),
        trend_predictions=read_trend_model_predictions(trend_prediction_path),
        input_paths={
            "weekly_transactions": str(WEEKLY_TRANSACTIONS_PATH),
            "article_attributes": str(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
            "trend_predictions": str(trend_prediction_path),
        },
        trend_model_source=str(trend_prediction_path),
    ),
)
```

- [ ] **Step 6: Record reused/rebuilt/skipped in experiment payload**

Extend `build_experiment_payload` to include:

```python
"stage_status": stage_status,
"force": {
    "force_experiment": force_experiment,
    "force_methods": list(force_methods),
    "force_cache": force_cache,
    "force_candidates": force_candidates,
    "force_rebuild_all": force_rebuild_all,
},
"timings": timings,
```

For each input/candidate/cache/method/metrics step append a status dict:

```python
{"stage": "method", "method": method_name, "status": "rebuilt", "reason": decision.reason}
```

- [ ] **Step 7: Verify experiment tests**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py -q
uv run python -m compileall -q src
git diff --check
```

Expected: tests PASS, compileall exits 0, diff check has no output.

- [ ] **Step 8: Commit**

```sh
git add src/fashion_trend/recommendation/experiments/runner.py src/16_run_recommendation_experiment.py tests/test_recommendation_experiments.py
git commit -m "feat(recommendation): 拆分实验重建语义"
```

---

### Task 8: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-05-10-recommendation-performance-optimization-design.md`

- [ ] **Step 1: Update artifact and command docs**

In `README.md`, `docs/gpt-research/implementation-plan.md`, and `AGENTS.md`, document:

```text
outputs/recommendation/<method>/recommendations.csv
outputs/recommendation/<method>/recommendation_items.parquet
outputs/recommendation/<method>/params.json
outputs/recommendation/<method>/metadata.json
data/processed/recommend/metadata.json
data/processed/recommend/features/<feature_name>/strategy=<strategy>/split=<split>/cutoff_week=<week>/part.parquet
```

Update `16` force examples:

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
uv run python src/16_run_recommendation_experiment.py --experiment main --force-method pop_similarity
uv run python src/16_run_recommendation_experiment.py --experiment main --force-cache
uv run python src/16_run_recommendation_experiment.py --experiment main --force-candidates
uv run python src/16_run_recommendation_experiment.py --experiment main --force-rebuild-all
```

- [ ] **Step 2: Run focused recommendation tests**

Run:

```sh
uv run pytest tests/test_recommendation_inputs.py tests/test_recommendation_retrieval.py tests/test_recommendation_ranking.py tests/test_recommendation_methods.py tests/test_recommendation_evaluation.py tests/test_recommendation_experiments.py tests/test_recommendation_freshness.py tests/test_recommendation_feature_cache.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 3: Run architecture and compile checks**

Run:

```sh
uv run pytest tests/test_architecture_boundaries.py -q
uv run python -m compileall -q src
git diff --check
```

Expected: architecture tests PASS, compileall exits 0, diff check has no output.

- [ ] **Step 4: Run full pytest before final handoff**

Run:

```sh
uv run pytest
```

Expected: full suite PASS.

- [ ] **Step 5: Optional real artifact smoke without download**

Only run after tests pass and only from existing downloaded data:

```sh
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy popularity
uv run python src/14_rerank_recommendations.py --method global_popularity
uv run python src/15_eval_recommendations.py --method global_popularity
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
```

Expected:

```text
data/processed/recommend/metadata.json exists
outputs/recommendation/global_popularity/recommendation_items.parquet exists
outputs/recommendation/global_popularity/recommendation_items.csv is absent unless explicitly exported
outputs/recommendation/experiments/main/experiment.json records reused/rebuilt and timings
```

- [ ] **Step 6: Inspect generated artifact boundaries**

Run:

```sh
du -sh outputs/recommendation data/processed/recommend
find outputs/recommendation -name 'recommendation_items.csv' -print
find outputs/recommendation -name '*.tmp' -o -name '*.part'
git status --short
```

Expected: generated CSV item files are absent by default, no temp/part files remain, git status only shows source/docs changes intended for the task.

- [ ] **Step 7: Commit**

```sh
git add README.md docs/gpt-research/implementation-plan.md AGENTS.md docs/superpowers/specs/2026-05-10-recommendation-performance-optimization-design.md
git commit -m "docs(recommendation): 同步推荐性能优化产物契约"
```

---

## Self-Review Checklist

- Spec coverage:
  - Timing and row-count logs are covered by Task 1.
  - Parquet method item output and CSV demotion are covered by Task 2.
  - `12` input metadata and unified freshness are covered by Task 3.
  - Strategy/window feature cache and `candidate_seen_flags` are covered by Task 4.
  - Method metadata full upstream chain is covered by Task 5.
  - Evaluation recommendable pool reuse is covered by Task 6.
  - `16` force semantics and stale propagation are covered by Task 7.
  - Docs and real artifact checks are covered by Task 8.
- No new runtime dependency is introduced.
- Production source `top_n=12` is preserved; larger values remain experiment configuration.
- Each task has focused tests, compile/diff verification, and a commit boundary.
