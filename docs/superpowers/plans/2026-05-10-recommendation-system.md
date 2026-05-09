# Recommendation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现趋势预测完成后的轻量 Top-12 推荐系统，跑通 `12 -> 13 -> 14 -> 15 -> 16` 的推荐输入、候选召回、重排序、评价和实验闭环。

**Architecture:** 推荐域采用“方法层 + 能力层 + runner 层”。`methods/` 只声明方法组合，`retrieval/`、`ranking/`、`evaluation/`、`experiments/` 提供底层能力，编号脚本只做 CLI 编排和日志。

**Tech Stack:** Python 3.10-3.12、pandas、numpy、pyarrow、pytest、现有 `fashion_trend.foundation` IO/artifact helper、上游 transactions/catalog/trend 公开 reader 与契约。

---

## File Structure

- Modify: `src/fashion_trend/recommendation/contracts.py`
  - 固定 Top-K、method、strategy、split、score feature、artifact 列、payload 必需字段、核心属性类型和趋势属性权重。
- Modify: `src/fashion_trend/recommendation/paths.py`
  - 提供 recommendation 中间产物、strategy-scoped candidates、method-scoped outputs、experiment-scoped runs 的安全路径构造。
- Modify: `src/fashion_trend/recommendation/readers.py`
  - 读取并严格校验 recommendation artifacts，拒绝缺列、重复键、不安全路径和不匹配的 method/strategy。
- Create: `src/fashion_trend/recommendation/outputs.py`
  - 写出 `recommendations.csv`、`recommendation_items.csv`、`params.json`、`metadata.json`、`metrics.json`。
- Create: `src/fashion_trend/recommendation/time_windows.py`
  - 生成 valid/test 推荐窗口，集中校验 `cutoff_week < label_week` 和 `label_week == cutoff_week + 1`。
- Create: `src/fashion_trend/recommendation/inputs.py`
  - 生成 `target_users.parquet`、`evaluation_labels.parquet`、`user_profile.parquet`。
- Create: `src/fashion_trend/recommendation/retrieval/popularity.py`
  - 生成累计热门、近期热门候选源。
- Create: `src/fashion_trend/recommendation/retrieval/attributes.py`
  - 基于用户属性画像和商品属性边生成 similarity 候选源。
- Create: `src/fashion_trend/recommendation/retrieval/trend.py`
  - 基于 stable LightGBM `pred_target_growth` 和核心属性生成 trend 候选源。
- Create: `src/fashion_trend/recommendation/retrieval/candidates.py`
  - 合并候选源，写出 `data/processed/recommend/candidates/<strategy>/candidate_items.parquet`。
- Create: `src/fashion_trend/recommendation/ranking/features.py`
  - 计算 `pop_score`、`recent_score`、`sim_score`、`trend_score`，并按 spec 作用域归一化。
- Create: `src/fashion_trend/recommendation/ranking/scoring.py`
  - 执行线性加权和稳定排序。
- Create: `src/fashion_trend/recommendation/ranking/filters.py`
  - 实现 `exclude_seen` 历史已购商品过滤，保证所有 method 共用同一规则。
- Create: `src/fashion_trend/recommendation/ranking/weights.py`
  - 校验权重非负、feature 完整、权重和为 1。
- Create: `src/fashion_trend/recommendation/methods/base.py`
  - 定义 `RecommendationMethod`、`RecommendationContext`、`RecommendationResult`。
- Create: `src/fashion_trend/recommendation/methods/baselines/global_popularity.py`
- Create: `src/fashion_trend/recommendation/methods/baselines/recent_popularity.py`
- Create: `src/fashion_trend/recommendation/methods/baselines/attribute_similarity.py`
- Create: `src/fashion_trend/recommendation/methods/baselines/pop_similarity.py`
  - 实现不依赖趋势预测的 baseline 和无趋势主对照。
- Create: `src/fashion_trend/recommendation/methods/trend_aware/pop_similarity_trend.py`
  - 实现趋势感知主方法。
- Create: `src/fashion_trend/recommendation/registry.py`
  - 注册 `global_popularity`、`recent_popularity`、`attribute_similarity`、`pop_similarity`、`pop_similarity_trend`。
- Create: `src/fashion_trend/recommendation/runner.py`
  - 读取 method、构建 context、执行推荐并写出 method-scoped 产物。
- Create: `src/fashion_trend/recommendation/evaluation/metrics.py`
  - 计算 MAP@12、Recall@12、HitRate@12、NDCG@12、Coverage。
- Create: `src/fashion_trend/recommendation/evaluation/payloads.py`
  - 构建严格 metrics JSON payload。
- Create: `src/fashion_trend/recommendation/evaluation/runner.py`
  - 读取 method 输出、target users 和 labels，执行单方法评价。
- Create: `src/fashion_trend/recommendation/experiments/grid_search.py`
  - valid 权重网格搜索，默认内存评估。
- Create: `src/fashion_trend/recommendation/experiments/ablation.py`
  - 汇总 baseline、主方法和消融指标。
- Create: `src/fashion_trend/recommendation/experiments/runner.py`
  - 编排主实验，生成 experiment-scoped artifact，不覆盖中间 run 的 stable method 目录。
- Create: `src/12_build_recommendation_inputs.py`
- Create: `src/13_build_recommend_candidates.py`
- Create: `src/14_rerank_recommendations.py`
- Create: `src/15_eval_recommendations.py`
- Create: `src/16_run_recommendation_experiment.py`
  - 新增编号 CLI，保持薄编排层。
- Create: `tests/test_recommendation_time_windows.py`
- Create: `tests/test_recommendation_inputs.py`
- Create: `tests/test_recommendation_retrieval.py`
- Create: `tests/test_recommendation_ranking.py`
- Create: `tests/test_recommendation_methods.py`
- Create: `tests/test_recommendation_evaluation.py`
- Create: `tests/test_recommendation_experiments.py`
  - 按能力边界测试 recommendation。
- Modify: `tests/test_architecture_boundaries.py`
  - 保持 recommendation 只读上游公开接口，必要时补充禁止导入测试。
- Modify: `README.md`
  - 同步推荐阶段状态、命令和 artifact 路径。
- Modify: `docs/gpt-research/implementation-plan.md`
  - 将历史 `12_build_user_profile.py`、扁平输出路径和 `16_make_reports.py` 描述更新为本设计的推荐阶段入口。

每个任务末尾包含 commit 命令。只有在执行阶段用户明确授权提交时才运行 commit；否则把 commit 命令作为阶段检查点。

Architecture boundary rule for this plan: modules under `src/fashion_trend/recommendation/` do not import upstream path constants such as `WEEKLY_TRANSACTIONS_PATH`, `GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH` or `OUTPUT_MODELS_DIR`. Numbered CLI scripts may load upstream artifacts through public readers and pass DataFrames into recommendation package functions.

---

### Task 1: Contracts, Paths, Readers

**Files:**
- Modify: `src/fashion_trend/recommendation/contracts.py`
- Modify: `src/fashion_trend/recommendation/paths.py`
- Modify: `src/fashion_trend/recommendation/readers.py`
- Create: `tests/test_recommendation_contracts.py`

- [ ] **Step 1: Write failing tests for constants, safe paths and strict readers**

Create `tests/test_recommendation_contracts.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.recommendation import contracts, paths
from fashion_trend.recommendation.readers import (
    read_candidate_items,
    read_recommendation_items,
    read_recommendations,
    read_time_windows,
)


def test_recommendation_contracts_define_public_constants() -> None:
    assert contracts.RECOMMENDATION_TOP_K == 12
    assert contracts.RECOMMENDATION_ARTICLE_ID_DTYPE == "string"
    assert contracts.RECOMMENDATION_METHODS == (
        "global_popularity",
        "recent_popularity",
        "attribute_similarity",
        "pop_similarity",
        "pop_similarity_trend",
    )
    assert contracts.RECOMMENDATION_CANDIDATE_STRATEGIES == (
        "popularity",
        "similarity",
        "trend_union",
        "default",
    )
    assert contracts.RECOMMENDATION_SCORE_COLUMNS == (
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
    )
    assert contracts.RECOMMENDATION_CORE_ATTR_TYPES == (
        "product_type_name",
        "colour_group_name",
        "garment_group_name",
        "product_group_name",
        "graphical_appearance_name",
    )
    assert contracts.RECOMMENDATION_TREND_ATTR_WEIGHTS == {
        "product_type_name": 0.35,
        "colour_group_name": 0.25,
        "garment_group_name": 0.20,
        "product_group_name": 0.10,
        "graphical_appearance_name": 0.10,
    }
    assert contracts.TIME_WINDOW_KEY_COLUMNS == ("split", "cutoff_week", "label_week")
    assert contracts.TARGET_USER_KEY_COLUMNS == ("split", "cutoff_week", "label_week", "customer_id")
    assert contracts.EVALUATION_LABEL_KEY_COLUMNS == (
        "split",
        "cutoff_week",
        "label_week",
        "customer_id",
        "article_id",
    )
    assert contracts.CANDIDATE_ITEM_KEY_COLUMNS == (
        "split",
        "cutoff_week",
        "label_week",
        "strategy",
        "customer_id",
        "article_id",
    )
    assert contracts.RECOMMENDATIONS_KEY_COLUMNS == (
        "customer_id",
        "split",
        "cutoff_week",
        "label_week",
        "method",
    )
    assert contracts.RECOMMENDATION_ITEMS_KEY_COLUMNS == (
        "customer_id",
        "split",
        "cutoff_week",
        "label_week",
        "method",
        "article_id",
    )


def test_candidate_path_is_strategy_scoped() -> None:
    assert paths.candidate_items_path("default").as_posix().endswith(
        "data/processed/recommend/candidates/default/candidate_items.parquet"
    )


def test_method_output_paths_are_method_scoped() -> None:
    bundle = paths.method_output_paths("pop_similarity_trend")

    assert bundle.recommendations.as_posix().endswith(
        "outputs/recommendation/pop_similarity_trend/recommendations.csv"
    )
    assert bundle.recommendation_items.name == "recommendation_items.csv"
    assert bundle.params.name == "params.json"
    assert bundle.metadata.name == "metadata.json"
    assert bundle.metrics.name == "metrics.json"


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b"])
def test_path_builders_reject_unsafe_segments(bad: str) -> None:
    with pytest.raises(ValueError, match="安全"):
        paths.candidate_items_path(bad)
    with pytest.raises(ValueError, match="安全"):
        paths.method_output_paths(bad)
    with pytest.raises(ValueError, match="安全"):
        paths.experiment_dir(bad)


def test_csv_readers_preserve_leading_zero_ids(tmp_path) -> None:
    method_dir = tmp_path / "pop_similarity_trend"
    method_dir.mkdir()
    items_path = method_dir / "recommendation_items.csv"
    recommendations_path = method_dir / "recommendations.csv"
    pd.DataFrame(
        [
            {
                "customer_id": "000000000001",
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "method": "pop_similarity_trend",
                "article_id": "0000000001",
                "rank": 1,
                "score": 1.0,
                "pop_score": 1.0,
                "sim_score": 0.0,
                "trend_score": 0.0,
                "recent_score": 0.0,
                "candidate_sources": "popularity",
            }
        ]
    ).to_csv(items_path, index=False)
    pd.DataFrame(
        [
            {
                "customer_id": "000000000001",
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "method": "pop_similarity_trend",
                "prediction": "0000000001 0000000002",
            }
        ]
    ).to_csv(recommendations_path, index=False)

    items = read_recommendation_items(items_path)
    recommendations = read_recommendations(recommendations_path)

    assert items.loc[0, "customer_id"] == "000000000001"
    assert items.loc[0, "article_id"] == "0000000001"
    assert str(items["customer_id"].dtype) == "string"
    assert str(items["article_id"].dtype) == "string"
    assert recommendations.loc[0, "customer_id"] == "000000000001"
    assert recommendations.loc[0, "prediction"] == "0000000001 0000000002"
    assert str(recommendations["prediction"].dtype) == "string"


def test_reader_rejects_duplicate_unique_keys(tmp_path) -> None:
    path = tmp_path / "time_windows.parquet"
    pd.DataFrame(
        [
            {"split": "valid", "cutoff_week": 10, "label_week": 11},
            {"split": "valid", "cutoff_week": 10, "label_week": 11},
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="重复键"):
        read_time_windows(path)


def test_candidate_reader_rejects_strategy_path_mismatch(tmp_path) -> None:
    candidate_dir = tmp_path / "candidates" / "similarity"
    candidate_dir.mkdir(parents=True)
    path = candidate_dir / "candidate_items.parquet"
    pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "strategy": "default",
                "customer_id": "c1",
                "article_id": "0000000001",
                "candidate_sources": "similarity",
                "primary_source": "similarity",
                "best_source_rank": 1,
            }
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="strategy"):
        read_candidate_items(path)


def test_recommendations_reader_rejects_method_path_mismatch(tmp_path) -> None:
    method_dir = tmp_path / "recent_popularity"
    method_dir.mkdir()
    path = method_dir / "recommendations.csv"
    pd.DataFrame(
        [
            {
                "customer_id": "c1",
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "method": "global_popularity",
                "prediction": "0000000001",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="method"):
        read_recommendations(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
uv run pytest tests/test_recommendation_contracts.py -q
```

Expected: FAIL because path builders, method constants, strategy constants and strict readers are not implemented yet.

- [ ] **Step 3: Implement contracts and path builders**

Update `src/fashion_trend/recommendation/contracts.py` with these public constants:

```python
RECOMMENDATION_TOP_K = 12
RECOMMENDATION_ARTICLE_ID_DTYPE = "string"

VALID_RECOMMENDATION_SPLITS = ("valid", "test")

RECOMMENDATION_METHODS = (
    "global_popularity",
    "recent_popularity",
    "attribute_similarity",
    "pop_similarity",
    "pop_similarity_trend",
)

RECOMMENDATION_CANDIDATE_STRATEGIES = (
    "popularity",
    "similarity",
    "trend_union",
    "default",
)

RECOMMENDATION_SCORE_COLUMNS = (
    "pop_score",
    "sim_score",
    "trend_score",
    "recent_score",
)

RECOMMENDATION_CORE_ATTR_TYPES = (
    "product_type_name",
    "colour_group_name",
    "garment_group_name",
    "product_group_name",
    "graphical_appearance_name",
)

RECOMMENDATION_TREND_ATTR_WEIGHTS = {
    "product_type_name": 0.35,
    "colour_group_name": 0.25,
    "garment_group_name": 0.20,
    "product_group_name": 0.10,
    "graphical_appearance_name": 0.10,
}

TIME_WINDOW_COLUMNS = ("split", "cutoff_week", "label_week")
TARGET_USER_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "history_purchase_count",
    "label_purchase_count",
)
EVALUATION_LABEL_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "article_id",
)
USER_PROFILE_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "preference_score",
    "purchase_count",
    "last_purchase_week",
)
CANDIDATE_ITEM_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "strategy",
    "customer_id",
    "article_id",
    "candidate_sources",
    "primary_source",
    "best_source_rank",
)
RECOMMENDATIONS_COLUMNS = (
    "customer_id",
    "split",
    "cutoff_week",
    "label_week",
    "method",
    "prediction",
)
RECOMMENDATION_ITEMS_COLUMNS = (
    "customer_id",
    "split",
    "cutoff_week",
    "label_week",
    "method",
    "article_id",
    "rank",
    "score",
    "pop_score",
    "sim_score",
    "trend_score",
    "recent_score",
    "candidate_sources",
)

TIME_WINDOW_KEY_COLUMNS = ("split", "cutoff_week", "label_week")
TARGET_USER_KEY_COLUMNS = ("split", "cutoff_week", "label_week", "customer_id")
EVALUATION_LABEL_KEY_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "article_id",
)
USER_PROFILE_KEY_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "attr_id",
    "attr_type",
    "attr_value",
)
CANDIDATE_ITEM_KEY_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "strategy",
    "customer_id",
    "article_id",
)
RECOMMENDATIONS_KEY_COLUMNS = ("customer_id", "split", "cutoff_week", "label_week", "method")
RECOMMENDATION_ITEMS_KEY_COLUMNS = (
    "customer_id",
    "split",
    "cutoff_week",
    "label_week",
    "method",
    "article_id",
)

RECOMMENDATION_TEXT_COLUMNS = (
    "split",
    "customer_id",
    "article_id",
    "prediction",
    "strategy",
    "method",
    "candidate_sources",
    "primary_source",
    "attr_type",
    "attr_value",
)
```

Update `src/fashion_trend/recommendation/paths.py` to expose:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.foundation.paths import OUTPUT_DIR, PROCESSED_DIR

RECOMMEND_DIR = PROCESSED_DIR / "recommend"
OUTPUT_RECOMMENDATION_DIR = OUTPUT_DIR / "recommendation"

TIME_WINDOWS_PATH = RECOMMEND_DIR / "time_windows.parquet"
TARGET_USERS_PATH = RECOMMEND_DIR / "target_users.parquet"
EVALUATION_LABELS_PATH = RECOMMEND_DIR / "evaluation_labels.parquet"
USER_PROFILE_PATH = RECOMMEND_DIR / "user_profile.parquet"
CANDIDATES_DIR = RECOMMEND_DIR / "candidates"
EXPERIMENTS_DIR = OUTPUT_RECOMMENDATION_DIR / "experiments"


@dataclass(frozen=True)
class RecommendationOutputPaths:
    output_dir: Path
    recommendations: Path
    recommendation_items: Path
    params: Path
    metadata: Path
    metrics: Path


def candidate_items_path(strategy: str) -> Path:
    validate_safe_path_segment(strategy, "candidate strategy")
    return CANDIDATES_DIR / strategy / "candidate_items.parquet"


def method_output_paths(method: str) -> RecommendationOutputPaths:
    validate_safe_path_segment(method, "recommendation method")
    output_dir = OUTPUT_RECOMMENDATION_DIR / method
    return RecommendationOutputPaths(
        output_dir=output_dir,
        recommendations=output_dir / "recommendations.csv",
        recommendation_items=output_dir / "recommendation_items.csv",
        params=output_dir / "params.json",
        metadata=output_dir / "metadata.json",
        metrics=output_dir / "metrics.json",
    )


def experiment_dir(experiment_id: str) -> Path:
    validate_safe_path_segment(experiment_id, "experiment_id")
    return EXPERIMENTS_DIR / experiment_id


def experiment_run_dir(experiment_id: str, run_id: str) -> Path:
    validate_safe_path_segment(run_id, "experiment run_id")
    return experiment_dir(experiment_id) / "runs" / run_id
```

- [ ] **Step 4: Implement reader validation helpers**

Add these helpers to `readers.py`:

```python
from pathlib import Path


def validate_columns(dataframe: pd.DataFrame, expected_columns: Sequence[str], artifact_name: str) -> None:
    actual_columns = tuple(dataframe.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            f"{artifact_name} 列契约不匹配: expected={expected_columns}, actual={actual_columns}"
        )


def reject_duplicate_key(dataframe: pd.DataFrame, key_columns: Sequence[str], artifact_name: str) -> None:
    duplicated = dataframe.duplicated(list(key_columns), keep=False)
    if duplicated.any():
        sample = dataframe.loc[duplicated, list(key_columns)].head(3).to_dict("records")
        raise ValueError(f"{artifact_name} 存在重复键: {sample}")


def text_dtypes_for_columns(columns: Sequence[str]) -> dict[str, str]:
    return {column: "string" for column in columns if column in RECOMMENDATION_TEXT_COLUMNS}


def read_csv_artifact(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=text_dtypes_for_columns(expected_columns), keep_default_na=False)


def coerce_article_id_string(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in RECOMMENDATION_TEXT_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("string")
    return result


def validate_path_value_matches(dataframe: pd.DataFrame, column: str, expected: str, artifact_name: str) -> None:
    actual_values = set(dataframe[column].dropna().astype(str))
    if actual_values != {expected}:
        raise ValueError(
            f"{artifact_name} {column} 与路径不一致: expected={expected}, actual={sorted(actual_values)}"
        )
```

Reader rules:

- CSV readers must call `pd.read_csv(..., dtype=...)` at read time through `read_csv_artifact()`. Do not rely on reading first and then casting, because pandas may already have dropped leading zeroes from `article_id`, `customer_id` or `prediction`.
- Parquet readers may read first and then call `coerce_article_id_string()` because parquet preserves typed values.
- `coerce_article_id_string()` is a post-read guard for mixed sources, not the primary CSV protection.
- Candidate readers must infer the expected strategy from `path.parent.name` and reject rows where `strategy` differs.
- Recommendation and recommendation-item readers must infer the expected method from `path.parent.name` and reject rows where `method` differs.
- Every strict reader must call `validate_columns()` and `reject_duplicate_key()` using the key constants from `contracts.py`.

Example CSV reader:

```python
def read_recommendation_items(path: Path) -> pd.DataFrame:
    dataframe = read_csv_artifact(path, RECOMMENDATION_ITEMS_COLUMNS)
    dataframe = coerce_article_id_string(dataframe)
    validate_columns(dataframe, RECOMMENDATION_ITEMS_COLUMNS, "recommendation_items.csv")
    reject_duplicate_key(dataframe, RECOMMENDATION_ITEMS_KEY_COLUMNS, "recommendation_items.csv")
    validate_path_value_matches(dataframe, "method", path.parent.name, "recommendation_items.csv")
    return dataframe
```

Then add strict readers for time windows, target users, evaluation labels, user profile, candidate items, recommendations and recommendation items. Each reader must use the key constants from `contracts.py`; do not hard-code unique-key column lists inside reader functions.

- [ ] **Step 5: Run tests and commit**

Run:

```sh
uv run pytest tests/test_recommendation_contracts.py tests/test_architecture_boundaries.py -q
```

Expected: PASS.

Commit:

```sh
git add src/fashion_trend/recommendation/contracts.py src/fashion_trend/recommendation/paths.py src/fashion_trend/recommendation/readers.py tests/test_recommendation_contracts.py
git commit -m "feat(recommendation): 定义推荐产物契约"
```

---

### Task 2: Time Windows and Recommendation Inputs

**Files:**
- Create: `src/fashion_trend/recommendation/time_windows.py`
- Create: `src/fashion_trend/recommendation/inputs.py`
- Create: `src/12_build_recommendation_inputs.py`
- Create: `tests/test_recommendation_time_windows.py`
- Create: `tests/test_recommendation_inputs.py`

- [ ] **Step 1: Write failing tests for windows**

Create `tests/test_recommendation_time_windows.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.recommendation.time_windows import build_recommendation_windows


def test_build_recommendation_windows_uses_cutoff_week() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["valid", "valid", "test"],
            "week_id": [104, 105, 106],
            "attr_type": ["product_type_name"] * 3,
            "attr_id": [1, 2, 3],
            "attr_value": ["A", "B", "C"],
            "pred_target_growth": [0.1, 0.2, 0.3],
            "pred_share_t1": [0.3, 0.3, 0.4],
        }
    )

    windows = build_recommendation_windows(predictions)

    assert windows.to_dict("records") == [
        {"split": "valid", "cutoff_week": 104, "label_week": 105},
        {"split": "valid", "cutoff_week": 105, "label_week": 106},
        {"split": "test", "cutoff_week": 106, "label_week": 107},
    ]


def test_build_recommendation_windows_requires_valid_and_test() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["valid"],
            "week_id": [104],
            "attr_type": ["product_type_name"],
            "attr_id": [1],
            "attr_value": ["A"],
            "pred_target_growth": [0.1],
            "pred_share_t1": [1.0],
        }
    )

    with pytest.raises(ValueError, match="test"):
        build_recommendation_windows(predictions)
```

- [ ] **Step 2: Write failing tests for target users, labels and profile**

Create `tests/test_recommendation_inputs.py`:

```python
from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.inputs import (
    build_evaluation_labels,
    build_target_users,
    build_user_profile,
)


def test_target_users_require_history_and_label_purchase() -> None:
    windows = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2", "u3"],
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "week_id": [9, 11, 11, 9],
        }
    )

    target_users = build_target_users(transactions, windows)

    assert target_users.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "history_purchase_count": 1,
            "label_purchase_count": 1,
        }
    ]


def test_evaluation_labels_deduplicate_articles_per_user_window() -> None:
    windows = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 2,
            }
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "article_id": ["0000000002", "0000000002", "0000000003"],
            "week_id": [11, 11, 10],
        }
    )

    labels = build_evaluation_labels(transactions, windows, target_users)

    assert labels.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "customer_id": "u1",
            "article_id": "0000000002",
        }
    ]


def test_user_profile_uses_history_before_or_at_cutoff_only() -> None:
    windows = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 2,
                "label_purchase_count": 1,
            }
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "article_id": ["0000000001", "0000000002", "0000000003"],
            "week_id": [8, 10, 11],
        }
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002", "0000000003"],
            "attr_id": [101, 102, 103],
            "attr_type": ["product_type_name"] * 3,
            "attr_value": ["Dress", "Shirt", "Shoes"],
        }
    )

    profile = build_user_profile(transactions, article_attributes, windows, target_users)

    assert profile["article_id"].tolist() == [] if "article_id" in profile else True
    assert set(profile["attr_value"]) == {"Dress", "Shirt"}
    assert "Shoes" not in set(profile["attr_value"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```sh
uv run pytest tests/test_recommendation_time_windows.py tests/test_recommendation_inputs.py -q
```

Expected: FAIL because `time_windows.py` and `inputs.py` do not exist.

- [ ] **Step 4: Implement windows and input builders**

Create `time_windows.py` with:

```python
def build_recommendation_windows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build recommendation windows from stable trend predictions."""
    required = {"split", "week_id"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"趋势预测缺少列: {sorted(missing)}")
    windows = (
        predictions.loc[predictions["split"].isin(VALID_RECOMMENDATION_SPLITS), ["split", "week_id"]]
        .drop_duplicates()
        .rename(columns={"week_id": "cutoff_week"})
        .assign(label_week=lambda frame: frame["cutoff_week"].astype(int) + 1)
        .sort_values(["split", "cutoff_week"])
        .reset_index(drop=True)
    )
    validate_recommendation_windows(windows)
    return windows.loc[:, list(TIME_WINDOW_COLUMNS)]
```

Create `inputs.py` with public functions:

```python
@dataclass(frozen=True)
class RecommendationInputArtifacts:
    time_windows: pd.DataFrame
    target_users: pd.DataFrame
    evaluation_labels: pd.DataFrame
    user_profile: pd.DataFrame


def build_target_users(transactions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        history = transactions.loc[transactions["week_id"] <= window.cutoff_week]
        labels = transactions.loc[transactions["week_id"] == window.label_week]
        history_counts = history.groupby("customer_id").size().rename("history_purchase_count")
        label_counts = labels.groupby("customer_id").size().rename("label_purchase_count")
        eligible = (
            pd.concat([history_counts, label_counts], axis=1)
            .dropna()
            .reset_index()
            .assign(split=window.split, cutoff_week=window.cutoff_week, label_week=window.label_week)
        )
        frames.append(eligible)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TARGET_USER_COLUMNS)
    return result.loc[:, list(TARGET_USER_COLUMNS)]


def build_evaluation_labels(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        labels = transactions.loc[transactions["week_id"] == window.label_week]
        eligible = target_users.loc[
            (target_users["split"] == window.split)
            & (target_users["cutoff_week"] == window.cutoff_week)
            & (target_users["label_week"] == window.label_week)
        ]
        merged = labels.merge(eligible[["customer_id"]], on="customer_id", how="inner")
        frames.append(
            merged.assign(split=window.split, cutoff_week=window.cutoff_week, label_week=window.label_week)
        )
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVALUATION_LABEL_COLUMNS)
    return result.loc[:, list(EVALUATION_LABEL_COLUMNS)].drop_duplicates()


def build_user_profile(
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        eligible = target_users.loc[
            (target_users["split"] == window.split)
            & (target_users["cutoff_week"] == window.cutoff_week)
            & (target_users["label_week"] == window.label_week)
        ]
        history = transactions.loc[transactions["week_id"] <= window.cutoff_week]
        history = history.merge(eligible[["customer_id"]], on="customer_id", how="inner")
        profile = history.merge(article_attributes, on="article_id", how="inner")
        profile = profile.groupby(["customer_id", "attr_id", "attr_type", "attr_value"], as_index=False).agg(
            purchase_count=("article_id", "size"),
            last_purchase_week=("week_id", "max"),
        )
        total = profile.groupby("customer_id")["purchase_count"].transform("sum")
        frames.append(
            profile.assign(
                split=window.split,
                cutoff_week=window.cutoff_week,
                label_week=window.label_week,
                preference_score=profile["purchase_count"] / total,
            )
        )
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=USER_PROFILE_COLUMNS)
    return result.loc[:, list(USER_PROFILE_COLUMNS)]


def build_and_write_recommendation_inputs(
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    trend_predictions: pd.DataFrame,
) -> RecommendationInputArtifacts:
    windows = build_recommendation_windows(trend_predictions)
    target_users = build_target_users(transactions, windows)
    labels = build_evaluation_labels(transactions, windows, target_users)
    profile = build_user_profile(transactions, article_attributes, windows, target_users)
    write_parquet_atomic(windows, TIME_WINDOWS_PATH)
    write_parquet_atomic(target_users, TARGET_USERS_PATH)
    write_parquet_atomic(labels, EVALUATION_LABELS_PATH)
    write_parquet_atomic(profile, USER_PROFILE_PATH)
    return RecommendationInputArtifacts(windows, target_users, labels, profile)
```

Implementation rules:

- `build_target_users()` counts history with `week_id <= cutoff_week` and labels with `week_id == label_week`.
- `build_evaluation_labels()` inner-joins `target_users` and deduplicates by `split + cutoff_week + label_week + customer_id + article_id`.
- `build_user_profile()` only uses transactions with `week_id <= cutoff_week`.
- `preference_score = purchase_count / sum(purchase_count per customer window)`.
- Output columns must exactly match `contracts.py`.

- [ ] **Step 5: Add CLI 12**

Create `src/12_build_recommendation_inputs.py`:

```python
from __future__ import annotations

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.foundation.logging import configure_logging, get_logger
from fashion_trend.recommendation.inputs import build_and_write_recommendation_inputs
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.readers import read_trend_model_predictions

LOGGER = get_logger(__name__)


def main() -> None:
    configure_logging()
    artifacts = build_and_write_recommendation_inputs(
        transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
        article_attributes=read_article_attribute_edges(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
        trend_predictions=read_trend_model_predictions(OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv"),
    )
    LOGGER.info("recommendation inputs written: %s", artifacts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests and commit**

Run:

```sh
uv run pytest tests/test_recommendation_time_windows.py tests/test_recommendation_inputs.py -q
uv run python -m compileall -q src
```

Expected: PASS.

Commit:

```sh
git add src/12_build_recommendation_inputs.py src/fashion_trend/recommendation/time_windows.py src/fashion_trend/recommendation/inputs.py tests/test_recommendation_time_windows.py tests/test_recommendation_inputs.py
git commit -m "feat(recommendation): 构建推荐输入产物"
```

---

### Task 3: Retrieval Strategies and Candidates

**Files:**
- Create: `src/fashion_trend/recommendation/retrieval/popularity.py`
- Create: `src/fashion_trend/recommendation/retrieval/attributes.py`
- Create: `src/fashion_trend/recommendation/retrieval/trend.py`
- Create: `src/fashion_trend/recommendation/retrieval/candidates.py`
- Create: `src/13_build_recommend_candidates.py`
- Create: `tests/test_recommendation_retrieval.py`

- [ ] **Step 1: Write failing retrieval tests**

Create `tests/test_recommendation_retrieval.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.recommendation.retrieval.candidates import (
    build_candidate_items,
    build_source_frames_for_frames,
    validate_candidate_strategy,
)
from fashion_trend.recommendation.retrieval.popularity import build_popularity_candidates


def sample_window() -> pd.DataFrame:
    return pd.DataFrame([{"split": "valid", "cutoff_week": 10, "label_week": 11}])


def sample_targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            }
        ]
    )


def test_popularity_candidates_ignore_label_week_transactions() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3"],
            "article_id": ["0000000001", "0000000002", "0000000999"],
            "week_id": [8, 10, 11],
        }
    )

    candidates = build_popularity_candidates(transactions, sample_window(), sample_targets(), top_n=10)

    assert set(candidates["article_id"]) == {"0000000001", "0000000002"}
    assert "0000000999" not in set(candidates["article_id"])


def test_default_candidates_merge_sources_with_best_rank() -> None:
    popularity = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "article_id": "0000000001",
                "source": "popularity",
                "source_rank": 2,
            }
        ]
    )
    similarity = popularity.assign(source="similarity", source_rank=1)
    trend = popularity.assign(source="trend", source_rank=3)

    candidates = build_candidate_items(
        strategy="default",
        source_frames=[popularity, similarity, trend],
    )

    assert candidates.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "strategy": "default",
            "customer_id": "u1",
            "article_id": "0000000001",
            "candidate_sources": "popularity|similarity|trend",
            "primary_source": "similarity",
            "best_source_rank": 1,
        }
    ]


def test_trend_union_requires_predictions() -> None:
    with pytest.raises(FileNotFoundError):
        build_candidate_items(strategy="trend_union", source_frames=[])


def test_unknown_strategy_fails_in_domain_layer() -> None:
    with pytest.raises(ValueError, match="未知候选 strategy"):
        validate_candidate_strategy("missing")
    with pytest.raises(ValueError, match="未知候选 strategy"):
        build_candidate_items(strategy="missing", source_frames=[])


def test_popularity_strategy_does_not_require_profile_or_trend_predictions() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "article_id": ["0000000001", "0000000002"],
            "week_id": [8, 10],
        }
    )

    frames = build_source_frames_for_frames(
        strategy="popularity",
        transactions=transactions,
        article_attributes=None,
        trend_predictions=None,
        windows=sample_window(),
        target_users=sample_targets(),
        user_profile=None,
    )

    assert len(frames) == 1
    assert set(frames[0]["source"]) == {"popularity"}


def test_trend_strategy_requires_trend_predictions() -> None:
    with pytest.raises(FileNotFoundError, match="trend predictions"):
        build_source_frames_for_frames(
            strategy="trend_union",
            transactions=pd.DataFrame(),
            article_attributes=pd.DataFrame(),
            trend_predictions=None,
            windows=sample_window(),
            target_users=sample_targets(),
            user_profile=None,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
uv run pytest tests/test_recommendation_retrieval.py -q
```

Expected: FAIL because retrieval modules do not exist.

- [ ] **Step 3: Implement candidate source builders**

Implement public functions:

```python
def build_popularity_candidates(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return one popularity candidate source frame per target user and window."""


def build_recent_popularity_candidates(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
    recent_weeks: int = 4,
) -> pd.DataFrame:
    """Return recent popularity candidates using cutoff-bounded history only."""


def build_attribute_similarity_candidates(
    user_profile: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return user-specific article candidates ranked by attribute preference match."""


def build_trend_candidates(
    predictions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return trend-driven candidates using predictions where week_id equals cutoff_week."""
```

All source builders return:

```text
split
cutoff_week
label_week
customer_id
article_id
source
source_rank
```

Rules:

- Use only `week_id <= cutoff_week` for popularity and recent popularity.
- Use only `target_users` as the user universe.
- Trend candidates read `predictions.week_id == cutoff_week`.
- Trend candidates use only `RECOMMENDATION_CORE_ATTR_TYPES`.
- Source rank is 1-based and stable by score desc, article_id asc.

- [ ] **Step 4: Implement strategy combiner and writer**

Implement:

```python
def validate_candidate_strategy(strategy: str) -> None:
    if strategy not in RECOMMENDATION_CANDIDATE_STRATEGIES:
        choices = ", ".join(RECOMMENDATION_CANDIDATE_STRATEGIES)
        raise ValueError(f"未知候选 strategy: {strategy}. 可用 strategy: {choices}")


def build_candidate_items(strategy: str, source_frames: list[pd.DataFrame]) -> pd.DataFrame:
    validate_candidate_strategy(strategy)
    source_order = {"popularity": 0, "similarity": 1, "trend": 2}
    if strategy == "trend_union" and not source_frames:
        raise FileNotFoundError("trend_union strategy requires trend source candidates")
    if not source_frames:
        return pd.DataFrame(columns=CANDIDATE_ITEM_COLUMNS)
    sources = pd.concat(source_frames, ignore_index=True)
    sources = sources.sort_values(["source_rank", "source"], key=lambda column: column.map(source_order) if column.name == "source" else column)
    grouped = sources.groupby(["split", "cutoff_week", "label_week", "customer_id", "article_id"], sort=False)
    result = grouped.agg(
        candidate_sources=("source", lambda values: "|".join(sorted(set(values), key=source_order.__getitem__))),
        primary_source=("source", "first"),
        best_source_rank=("source_rank", "min"),
    ).reset_index()
    result.insert(3, "strategy", strategy)
    return result.loc[:, list(CANDIDATE_ITEM_COLUMNS)]

def build_and_write_candidate_items(
    strategy: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    user_profile: pd.DataFrame | None,
) -> Path:
    source_frames = build_source_frames_for_frames(
        strategy,
        transactions,
        article_attributes,
        trend_predictions,
        windows,
        target_users,
        user_profile,
    )
    candidates = build_candidate_items(strategy, source_frames)
    output_path = candidate_items_path(strategy)
    write_parquet_atomic(candidates, output_path)
    return output_path


def build_source_frames_for_frames(
    strategy: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    user_profile: pd.DataFrame | None,
) -> list[pd.DataFrame]:
    validate_candidate_strategy(strategy)
    frames: list[pd.DataFrame] = []
    if strategy in {"popularity", "default"}:
        frames.append(build_recent_popularity_candidates(transactions, windows, target_users, top_n=200))
    if strategy in {"similarity", "default"}:
        if user_profile is None or article_attributes is None:
            raise FileNotFoundError("similarity strategy requires user profile and article attributes")
        frames.append(build_attribute_similarity_candidates(user_profile, article_attributes, windows, target_users, top_n=200))
    if strategy in {"trend_union", "default"}:
        if trend_predictions is None or article_attributes is None:
            raise FileNotFoundError("trend strategy requires trend predictions and article attributes")
        frames.append(build_trend_candidates(trend_predictions, article_attributes, windows, target_users, top_n=200))
    return frames
```

Merge rules:

- `candidate_sources` is a `|`-joined string ordered as `popularity`, `similarity`, `trend`.
- `primary_source` is the source with the smallest `source_rank`; ties use the same source order.
- `best_source_rank` is the minimum `source_rank`.
- Output columns exactly match `CANDIDATE_ITEM_COLUMNS`.
- The `strategy` column must equal the path segment used in `candidate_items_path(strategy)`.

- [ ] **Step 5: Add CLI 13**

Create `src/13_build_recommend_candidates.py`:

```python
from __future__ import annotations

import argparse

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.foundation.logging import configure_logging, get_logger
from fashion_trend.recommendation.contracts import RECOMMENDATION_CANDIDATE_STRATEGIES
from fashion_trend.recommendation.paths import TARGET_USERS_PATH, TIME_WINDOWS_PATH, USER_PROFILE_PATH
from fashion_trend.recommendation.readers import read_target_users, read_time_windows, read_user_profile
from fashion_trend.recommendation.retrieval.candidates import build_and_write_candidate_items
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.readers import read_trend_model_predictions

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=RECOMMENDATION_CANDIDATE_STRATEGIES, default="default")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    article_attributes = None
    user_profile = None
    trend_predictions = None
    if args.strategy in {"similarity", "trend_union", "default"}:
        article_attributes = read_article_attribute_edges(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH)
    if args.strategy in {"similarity", "default"}:
        user_profile = read_user_profile(USER_PROFILE_PATH)
    if args.strategy in {"trend_union", "default"}:
        trend_predictions = read_trend_model_predictions(OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv")
    output_path = build_and_write_candidate_items(
        strategy=args.strategy,
        transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
        article_attributes=article_attributes,
        trend_predictions=trend_predictions,
        windows=read_time_windows(TIME_WINDOWS_PATH),
        target_users=read_target_users(TARGET_USERS_PATH),
        user_profile=user_profile,
    )
    LOGGER.info("recommendation candidates written: %s", output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests and commit**

Run:

```sh
uv run pytest tests/test_recommendation_retrieval.py tests/test_recommendation_contracts.py -q
uv run python -m compileall -q src
```

Expected: PASS.

Commit:

```sh
git add src/13_build_recommend_candidates.py src/fashion_trend/recommendation/retrieval tests/test_recommendation_retrieval.py
git commit -m "feat(recommendation): 实现候选召回策略"
```

---

### Task 4: Ranking Features and Weights

**Files:**
- Create: `src/fashion_trend/recommendation/ranking/features.py`
- Create: `src/fashion_trend/recommendation/ranking/scoring.py`
- Create: `src/fashion_trend/recommendation/ranking/filters.py`
- Create: `src/fashion_trend/recommendation/ranking/weights.py`
- Create: `tests/test_recommendation_ranking.py`

- [ ] **Step 1: Write failing ranking tests**

Create `tests/test_recommendation_ranking.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.recommendation.ranking.features import minmax_normalize_by_group
from fashion_trend.recommendation.ranking.filters import filter_seen_items
from fashion_trend.recommendation.ranking.scoring import rank_candidate_items
from fashion_trend.recommendation.ranking.weights import validate_score_weights


def test_minmax_constant_group_fills_zero() -> None:
    frame = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "value": [5.0, 5.0],
        }
    )

    result = minmax_normalize_by_group(
        frame,
        value_column="value",
        output_column="score",
        group_columns=("split", "cutoff_week", "label_week"),
    )

    assert result["score"].tolist() == [0.0, 0.0]


def test_validate_score_weights_rejects_invalid_sum() -> None:
    with pytest.raises(ValueError, match="sum"):
        validate_score_weights(
            {"pop_score": 0.5, "sim_score": 0.5, "trend_score": 0.5, "recent_score": 0.0},
            required_features=("pop_score", "sim_score", "trend_score", "recent_score"),
        )


def test_rank_candidate_items_uses_stable_tie_break() -> None:
    candidates = pd.DataFrame(
        {
            "customer_id": ["u1", "u1"],
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "method": ["pop_similarity", "pop_similarity"],
            "article_id": ["0000000020", "0000000010"],
            "pop_score": [0.5, 0.5],
            "sim_score": [0.5, 0.5],
            "trend_score": [0.0, 0.0],
            "recent_score": [0.0, 0.0],
            "candidate_sources": ["popularity", "popularity"],
        }
    )

    ranked = rank_candidate_items(
        candidates,
        weights={"pop_score": 0.5, "sim_score": 0.5, "trend_score": 0.0, "recent_score": 0.0},
        top_k=12,
    )

    assert ranked["article_id"].tolist() == ["0000000010", "0000000020"]
    assert ranked["rank"].tolist() == [1, 2]


def test_filter_seen_items_uses_history_at_or_before_cutoff() -> None:
    candidates = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "split": ["valid", "valid", "valid"],
            "cutoff_week": [10, 10, 10],
            "label_week": [11, 11, 11],
            "article_id": ["0000000001", "0000000002", "0000000003"],
        }
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u1"],
            "article_id": ["0000000001", "0000000003"],
            "week_id": [10, 11],
        }
    )

    filtered = filter_seen_items(candidates, transactions)

    assert filtered["article_id"].tolist() == ["0000000002", "0000000003"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
uv run pytest tests/test_recommendation_ranking.py -q
```

Expected: FAIL because ranking modules do not exist.

- [ ] **Step 3: Implement normalization and feature builders**

Implement in `features.py`:

```python
def minmax_normalize_by_group(
    dataframe: pd.DataFrame,
    value_column: str,
    output_column: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Normalize one numeric column within groups; constant groups become 0.0."""
    result = dataframe.copy()
    grouped = result.groupby(list(group_columns))[value_column]
    min_value = grouped.transform("min")
    max_value = grouped.transform("max")
    denominator = max_value - min_value
    result[output_column] = np.where(denominator == 0, 0.0, (result[value_column] - min_value) / denominator)
    if not np.isfinite(result[output_column]).all():
        raise ValueError(f"{output_column} contains non-finite values")
    return result

def build_ranking_features(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    feature_frame = candidates.copy()
    feature_frame = add_pop_score(feature_frame, transactions)
    feature_frame = add_recent_score(feature_frame, transactions)
    feature_frame = add_sim_score(feature_frame, article_attributes, user_profile)
    feature_frame = add_trend_score(feature_frame, article_attributes, trend_predictions)
    return feature_frame


def add_pop_score(candidates: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative popularity normalized by split, cutoff_week and label_week."""


def add_recent_score(candidates: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Add recent popularity normalized by split, cutoff_week and label_week."""


def add_sim_score(
    candidates: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add user-specific similarity normalized by user and window."""


def add_trend_score(
    candidates: pd.DataFrame,
    article_attributes: pd.DataFrame,
    trend_predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add article trend score; non-trend methods receive explicit 0.0 values."""
```

Required scopes:

- `pop_score`: `split + cutoff_week + label_week`.
- `recent_score`: `split + cutoff_week + label_week`.
- `sim_score`: `split + cutoff_week + label_week + customer_id`.
- `trend_score`: attributes first use `split + cutoff_week + attr_type`; article-level score uses `split + cutoff_week + label_week`.

- [ ] **Step 4: Implement seen-item filtering**

Implement in `filters.py`:

```python
def filter_seen_items(items: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Remove items purchased by the same user at or before each cutoff week."""
    frames: list[pd.DataFrame] = []
    for window in items[["split", "cutoff_week", "label_week"]].drop_duplicates().itertuples(index=False):
        window_items = items.loc[
            (items["split"] == window.split)
            & (items["cutoff_week"] == window.cutoff_week)
            & (items["label_week"] == window.label_week)
        ]
        seen = transactions.loc[
            transactions["week_id"] <= window.cutoff_week,
            ["customer_id", "article_id"],
        ].drop_duplicates()
        filtered = window_items.merge(
            seen.assign(_seen=True),
            on=["customer_id", "article_id"],
            how="left",
        )
        frames.append(filtered.loc[filtered["_seen"].isna(), window_items.columns])
    return pd.concat(frames, ignore_index=True) if frames else items.copy()
```

Every method runner must call this helper when `context.exclude_seen` is true, including global popularity, recent popularity, attribute similarity, `pop_similarity` and `pop_similarity_trend`. The selected value must be recorded in `params.json`.

- [ ] **Step 5: Implement weights and scoring**

Implement in `weights.py`:

```python
def validate_score_weights(weights: dict[str, float], required_features: Sequence[str]) -> dict[str, float]:
    required = set(required_features)
    actual = set(weights)
    if actual != required:
        raise ValueError(f"score weights keys mismatch: expected={sorted(required)}, actual={sorted(actual)}")
    for feature, value in weights.items():
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"invalid weight for {feature}: {value}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"score weights sum must be 1.0, got {total}")
    return dict(weights)
```

Validation rules:

- Reject missing features.
- Reject extra features.
- Reject negative values.
- Reject non-finite values.
- Reject sum not equal to 1.0 within `abs(sum - 1.0) <= 1e-9`.
- Do not auto-normalize.

Implement in `scoring.py`:

```python
def rank_candidate_items(feature_frame: pd.DataFrame, weights: dict[str, float], top_k: int) -> pd.DataFrame:
    result = feature_frame.copy()
    result["score"] = sum(result[feature] * weight for feature, weight in weights.items())
    result = result.sort_values(
        ["customer_id", "split", "cutoff_week", "label_week", "score", "article_id"],
        ascending=[True, True, True, True, False, True],
    )
    result["rank"] = result.groupby(["customer_id", "split", "cutoff_week", "label_week"]).cumcount() + 1
    return result.loc[result["rank"] <= top_k].reset_index(drop=True)
```

Ordering:

```text
score desc
article_id asc
```

- [ ] **Step 6: Run tests and commit**

Run:

```sh
uv run pytest tests/test_recommendation_ranking.py -q
uv run python -m compileall -q src
```

Expected: PASS.

Commit:

```sh
git add src/fashion_trend/recommendation/ranking tests/test_recommendation_ranking.py
git commit -m "feat(recommendation): 实现推荐排序特征"
```

---

### Task 5: Baseline Methods and Method Registry

**Files:**
- Create: `src/fashion_trend/recommendation/methods/base.py`
- Create: `src/fashion_trend/recommendation/methods/baselines/global_popularity.py`
- Create: `src/fashion_trend/recommendation/methods/baselines/recent_popularity.py`
- Create: `src/fashion_trend/recommendation/methods/baselines/attribute_similarity.py`
- Create: `src/fashion_trend/recommendation/methods/baselines/pop_similarity.py`
- Create: `src/fashion_trend/recommendation/registry.py`
- Create: `src/fashion_trend/recommendation/outputs.py`
- Create: `src/fashion_trend/recommendation/runner.py`
- Create: `src/14_rerank_recommendations.py`
- Create: `tests/test_recommendation_methods.py`

- [ ] **Step 1: Write failing registry and baseline tests**

Create `tests/test_recommendation_methods.py`:

```python
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from fashion_trend.recommendation.contracts import (
    CANDIDATE_ITEM_COLUMNS,
    RECOMMENDATION_ITEMS_COLUMNS,
    RECOMMENDATIONS_COLUMNS,
    USER_PROFILE_COLUMNS,
)
from fashion_trend.recommendation.methods.base import RecommendationContext
from fashion_trend.recommendation.registry import get_recommendation_method


def test_registry_lists_unknown_method_choices() -> None:
    with pytest.raises(ValueError, match="global_popularity.*pop_similarity"):
        get_recommendation_method("missing")


def test_global_popularity_does_not_require_profile_or_candidates() -> None:
    method = get_recommendation_method("global_popularity")

    assert method.name == "global_popularity"
    assert method.default_candidate_strategy is None
    assert method.required_features == ("pop_score",)


def test_recent_popularity_does_not_require_profile_or_candidates() -> None:
    method = get_recommendation_method("recent_popularity")

    assert method.name == "recent_popularity"
    assert method.default_candidate_strategy is None
    assert method.required_features == ("recent_score",)


def test_pop_similarity_uses_default_candidates_without_trend_score() -> None:
    method = get_recommendation_method("pop_similarity")

    assert method.default_candidate_strategy == "default"
    assert method.required_features == ("pop_score", "sim_score", "recent_score")
    assert method.default_weights == {
        "pop_score": 0.45,
        "sim_score": 0.45,
        "recent_score": 0.10,
    }


def sample_method_context(
    *,
    method_name: str = "global_popularity",
    exclude_seen: bool = True,
    user_profile: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
) -> RecommendationContext:
    windows = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            }
        ]
    )
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3", "u4"],
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "week_id": [9, 9, 10, 10],
        }
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "attr_id": [101, 102, 101, 103],
            "attr_type": ["product_type_name"] * 4,
            "attr_value": ["Dress", "Shirt", "Dress", "Shoes"],
        }
    )
    if user_profile is None:
        user_profile = pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "customer_id": "u1",
                    "attr_id": 101,
                    "attr_type": "product_type_name",
                    "attr_value": "Dress",
                    "preference_score": 1.0,
                    "purchase_count": 1,
                    "last_purchase_week": 9,
                }
            ],
            columns=list(USER_PROFILE_COLUMNS),
        )
    if candidates is None:
        candidates = pd.DataFrame(
            [
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "strategy": "default",
                    "customer_id": "u1",
                    "article_id": "0000000003",
                    "candidate_sources": "popularity|similarity",
                    "primary_source": "similarity",
                    "best_source_rank": 1,
                },
                {
                    "split": "valid",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "strategy": "default",
                    "customer_id": "u1",
                    "article_id": "0000000001",
                    "candidate_sources": "popularity",
                    "primary_source": "popularity",
                    "best_source_rank": 2,
                },
            ],
            columns=list(CANDIDATE_ITEM_COLUMNS),
        )
    return RecommendationContext(
        method=method_name,
        top_k=12,
        exclude_seen=exclude_seen,
        transactions=transactions,
        article_attributes=article_attributes,
        windows=windows,
        target_users=target_users,
        candidates=candidates,
        user_profile=user_profile,
        trend_predictions=None,
    )


def assert_method_result_shape(result, method_name: str) -> None:
    assert tuple(result.recommendations.columns) == RECOMMENDATIONS_COLUMNS
    assert tuple(result.recommendation_items.columns) == RECOMMENDATION_ITEMS_COLUMNS
    assert set(result.recommendations["method"]) == {method_name}
    assert set(result.recommendation_items["method"]) == {method_name}
    assert result.recommendation_items["rank"].between(1, 12).all()
    assert result.params["method"] == method_name


@pytest.mark.parametrize("method_name", ["global_popularity", "recent_popularity"])
def test_popularity_baseline_methods_build_recommendations_without_profile_or_candidates(method_name: str) -> None:
    method = get_recommendation_method(method_name)
    context = sample_method_context(method_name=method_name, user_profile=None, candidates=None)

    result = method.build_recommendations(context)

    assert_method_result_shape(result, method_name)
    assert "0000000001" not in set(result.recommendation_items["article_id"])
    assert result.params["exclude_seen"] is True


def test_attribute_similarity_falls_back_when_profile_is_empty() -> None:
    method = get_recommendation_method("attribute_similarity")
    empty_profile = pd.DataFrame(columns=list(USER_PROFILE_COLUMNS))
    context = sample_method_context(method_name="attribute_similarity", user_profile=empty_profile)

    result = method.build_recommendations(context)

    assert_method_result_shape(result, "attribute_similarity")
    assert result.metadata["fallback_user_count"] == 1


def test_pop_similarity_method_builds_recommendations_without_trend_predictions() -> None:
    method = get_recommendation_method("pop_similarity")
    context = sample_method_context(method_name="pop_similarity")

    result = method.build_recommendations(context)

    assert_method_result_shape(result, "pop_similarity")
    assert "trend_score" in result.recommendation_items.columns
    assert result.recommendation_items["trend_score"].eq(0.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
uv run pytest tests/test_recommendation_methods.py -q
```

Expected: FAIL because method registry and method modules do not exist.

- [ ] **Step 3: Implement base protocol and registry**

Create `methods/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class RecommendationContext:
    method: str
    top_k: int
    exclude_seen: bool
    transactions: pd.DataFrame
    article_attributes: pd.DataFrame
    windows: pd.DataFrame
    target_users: pd.DataFrame
    candidates: pd.DataFrame | None = None
    user_profile: pd.DataFrame | None = None
    trend_predictions: pd.DataFrame | None = None
    weights: dict[str, float] | None = None


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: pd.DataFrame
    recommendation_items: pd.DataFrame
    params: dict[str, object]
    metadata: dict[str, object]


class RecommendationMethod(Protocol):
    name: str
    method_type: str
    default_candidate_strategy: str | None
    default_weights: dict[str, float]
    required_features: Sequence[str]

    def build_recommendations(self, context: RecommendationContext) -> RecommendationResult:
        raise NotImplementedError
```

Create `registry.py`:

```python
RECOMMENDATION_METHOD_REGISTRY = {
    "global_popularity": GlobalPopularityMethod(),
    "recent_popularity": RecentPopularityMethod(),
    "attribute_similarity": AttributeSimilarityMethod(),
    "pop_similarity": PopSimilarityMethod(),
}


def get_recommendation_method(name: str) -> RecommendationMethod:
    validate_safe_path_segment(name, "recommendation method")
    try:
        return RECOMMENDATION_METHOD_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(RECOMMENDATION_METHOD_REGISTRY))
        raise ValueError(f"未知推荐 method: {name}. 可用 method: {choices}") from exc
```

Task 5 deliberately registers baseline methods only. `pop_similarity_trend` is added to the registry in Task 6 after `PopSimilarityTrendMethod` exists, so Task 5 imports remain runnable.

- [ ] **Step 4: Implement baseline method classes**

Implement classes with these properties:

```python
GlobalPopularityMethod.name = "global_popularity"
GlobalPopularityMethod.default_candidate_strategy = None
GlobalPopularityMethod.required_features = ("pop_score",)
GlobalPopularityMethod.default_weights = {"pop_score": 1.0}

RecentPopularityMethod.name = "recent_popularity"
RecentPopularityMethod.default_candidate_strategy = None
RecentPopularityMethod.required_features = ("recent_score",)
RecentPopularityMethod.default_weights = {"recent_score": 1.0}

AttributeSimilarityMethod.name = "attribute_similarity"
AttributeSimilarityMethod.default_candidate_strategy = "similarity"
AttributeSimilarityMethod.required_features = ("sim_score",)
AttributeSimilarityMethod.default_weights = {"sim_score": 1.0}

PopSimilarityMethod.name = "pop_similarity"
PopSimilarityMethod.default_candidate_strategy = "default"
PopSimilarityMethod.required_features = ("pop_score", "sim_score", "recent_score")
PopSimilarityMethod.default_weights = {"pop_score": 0.45, "sim_score": 0.45, "recent_score": 0.10}
```

Baseline behavior:

- `global_popularity` and `recent_popularity` can build candidates internally from transactions and target users.
- `attribute_similarity` falls back to recent popularity when a user profile is empty; metadata records `fallback_user_count`.
- `pop_similarity` uses default candidates and does not require trend predictions.
- All baseline methods apply `filter_seen_items()` after candidate/feature construction when `context.exclude_seen` is true.

- [ ] **Step 5: Implement output writer and CLI 14 for baselines**

Create `outputs.py`:

```python
def build_recommendations_csv(recommendation_items: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = recommendation_items.loc[recommendation_items["rank"] <= top_k].sort_values(
        ["customer_id", "split", "cutoff_week", "label_week", "rank"]
    )
    predictions = ranked.groupby(["customer_id", "split", "cutoff_week", "label_week", "method"])["article_id"].apply(
        lambda values: " ".join(str(value) for value in values)
    )
    return predictions.reset_index(name="prediction").loc[:, list(RECOMMENDATIONS_COLUMNS)]

def write_recommendation_result(result: RecommendationResult) -> None:
    output_paths = method_output_paths(str(result.params["method"]))
    write_csv_atomic(result.recommendations, output_paths.recommendations)
    write_csv_atomic(result.recommendation_items, output_paths.recommendation_items)
    write_json_atomic(result.params, output_paths.params)
    write_json_atomic(result.metadata, output_paths.metadata)
```

Create `runner.py`:

```python
def run_recommendation_method(
    method_name: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    user_profile: pd.DataFrame | None = None,
    trend_predictions: pd.DataFrame | None = None,
    exclude_seen: bool = True,
    weights: dict[str, float] | None = None,
) -> RecommendationResult:
    method = get_recommendation_method(method_name)
    context = RecommendationContext(
        method=method_name,
        top_k=RECOMMENDATION_TOP_K,
        exclude_seen=exclude_seen,
        transactions=transactions,
        article_attributes=article_attributes,
        windows=windows,
        target_users=target_users,
        candidates=candidates,
        user_profile=user_profile,
        trend_predictions=trend_predictions,
        weights=weights,
    )
    result = method.build_recommendations(context)
    write_recommendation_result(result)
    return result
```

Create `src/14_rerank_recommendations.py`:

```python
from __future__ import annotations

import argparse

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.foundation.logging import configure_logging, get_logger
from fashion_trend.recommendation.paths import (
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    USER_PROFILE_PATH,
    candidate_items_path,
)
from fashion_trend.recommendation.readers import (
    read_candidate_items,
    read_target_users,
    read_time_windows,
    read_user_profile,
)
from fashion_trend.recommendation.contracts import RECOMMENDATION_METHODS
from fashion_trend.recommendation.registry import get_recommendation_method
from fashion_trend.recommendation.runner import run_recommendation_method
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.readers import read_trend_model_predictions

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=RECOMMENDATION_METHODS, required=True)
    parser.add_argument("--exclude-seen", action="store_true", default=True)
    parser.add_argument("--include-seen", action="store_false", dest="exclude_seen")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    method = get_recommendation_method(args.method)
    candidates = None
    if method.default_candidate_strategy is not None:
        candidate_path = candidate_items_path(method.default_candidate_strategy)
        candidates = read_candidate_items(candidate_path)
    trend_predictions = None
    if method.name == "pop_similarity_trend":
        trend_predictions = read_trend_model_predictions(OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv")
    result = run_recommendation_method(
        method_name=args.method,
        transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
        article_attributes=read_article_attribute_edges(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
        windows=read_time_windows(TIME_WINDOWS_PATH),
        target_users=read_target_users(TARGET_USERS_PATH),
        candidates=candidates,
        user_profile=read_user_profile(USER_PROFILE_PATH) if USER_PROFILE_PATH.exists() else None,
        trend_predictions=trend_predictions,
        exclude_seen=args.exclude_seen,
    )
    LOGGER.info("recommendations written for method=%s rows=%s", args.method, len(result.recommendations))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests and commit**

Run:

```sh
uv run pytest tests/test_recommendation_methods.py tests/test_recommendation_ranking.py -q
uv run python -m compileall -q src
```

Expected: PASS.

Commit:

```sh
git add src/14_rerank_recommendations.py src/fashion_trend/recommendation/methods src/fashion_trend/recommendation/registry.py src/fashion_trend/recommendation/outputs.py src/fashion_trend/recommendation/runner.py tests/test_recommendation_methods.py
git commit -m "feat(recommendation): 接入推荐 baseline 方法"
```

---

### Task 6: Trend-Aware Main Method

**Files:**
- Create: `src/fashion_trend/recommendation/methods/trend_aware/pop_similarity_trend.py`
- Modify: `src/fashion_trend/recommendation/registry.py`
- Modify: `src/fashion_trend/recommendation/ranking/features.py`
- Modify: `tests/test_recommendation_methods.py`
- Modify: `tests/test_recommendation_ranking.py`

- [ ] **Step 1: Write failing trend-aware tests**

Append to `tests/test_recommendation_methods.py`:

```python
def test_pop_similarity_trend_uses_default_candidates_and_trend_score() -> None:
    method = get_recommendation_method("pop_similarity_trend")

    assert method.default_candidate_strategy == "default"
    assert method.required_features == (
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
    )
    assert method.default_weights == {
        "pop_score": 0.35,
        "sim_score": 0.35,
        "trend_score": 0.25,
        "recent_score": 0.05,
    }


def test_pop_similarity_trend_method_builds_recommendations_with_trend_predictions() -> None:
    method = get_recommendation_method("pop_similarity_trend")
    context = sample_method_context()
    predictions = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "week_id": [10, 10],
            "attr_type": ["product_type_name", "product_type_name"],
            "attr_id": [101, 102],
            "attr_value": ["Dress", "Shirt"],
            "pred_target_growth": [2.0, 1.0],
            "pred_share_t1": [0.6, 0.4],
        }
    )

    result = method.build_recommendations(
        replace(context, method="pop_similarity_trend", trend_predictions=predictions)
    )

    assert_method_result_shape(result, "pop_similarity_trend")
    assert result.recommendation_items["trend_score"].max() > 0.0


def test_pop_similarity_trend_method_requires_trend_predictions() -> None:
    method = get_recommendation_method("pop_similarity_trend")
    context = sample_method_context()

    with pytest.raises(FileNotFoundError, match="trend predictions"):
        method.build_recommendations(replace(context, method="pop_similarity_trend", trend_predictions=None))
```

Append to `tests/test_recommendation_ranking.py`:

```python
from fashion_trend.recommendation.ranking.features import build_article_trend_scores


def test_trend_score_uses_prediction_week_equal_cutoff_week() -> None:
    windows = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )
    predictions = pd.DataFrame(
        {
            "split": ["valid", "valid", "valid", "valid"],
            "week_id": [10, 10, 11, 11],
            "attr_type": ["product_type_name"] * 4,
            "attr_id": [101, 102, 101, 102],
            "attr_value": ["Dress", "Shirt", "Dress", "Shirt"],
            "pred_target_growth": [10.0, 1.0, 1.0, 10.0],
            "pred_share_t1": [0.6, 0.4, 0.4, 0.6],
        }
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001"],
            "attr_id": [101],
            "attr_type": ["product_type_name"],
            "attr_value": ["Dress"],
        }
    )

    scores = build_article_trend_scores(predictions, article_attributes, windows)

    assert scores.to_dict("records") == [
        {
            "split": "valid",
            "cutoff_week": 10,
            "label_week": 11,
            "article_id": "0000000001",
            "trend_score": 1.0,
        }
    ]


def test_trend_score_renormalizes_weights_for_matched_attribute_types() -> None:
    windows = pd.DataFrame(
        [{"split": "valid", "cutoff_week": 10, "label_week": 11}]
    )
    predictions = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "week_id": [10, 10],
            "attr_type": ["product_type_name", "product_type_name"],
            "attr_id": [101, 102],
            "attr_value": ["Dress", "Shirt"],
            "pred_target_growth": [2.0, 1.0],
            "pred_share_t1": [0.6, 0.4],
        }
    )
    article_attributes = pd.DataFrame(
        {
            "article_id": ["0000000001"],
            "attr_id": [101],
            "attr_type": ["product_type_name"],
            "attr_value": ["Dress"],
        }
    )

    scores = build_article_trend_scores(predictions, article_attributes, windows)

    assert scores.loc[0, "trend_score"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
uv run pytest tests/test_recommendation_methods.py::test_pop_similarity_trend_uses_default_candidates_and_trend_score tests/test_recommendation_ranking.py::test_trend_score_uses_prediction_week_equal_cutoff_week -q
```

Expected: FAIL because trend-aware method and trend score builder are not implemented.

- [ ] **Step 3: Implement trend score feature**

Implement `build_article_trend_scores()` in `ranking/features.py`:

```python
def build_article_trend_scores(
    predictions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    trend_rows = windows.merge(predictions, left_on=["split", "cutoff_week"], right_on=["split", "week_id"], how="inner")
    trend_rows = trend_rows.loc[trend_rows["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES)]
    trend_rows = minmax_normalize_by_group(
        trend_rows,
        value_column="pred_target_growth",
        output_column="attr_trend_score",
        group_columns=["split", "cutoff_week", "attr_type"],
    )
    mapped = trend_rows.merge(article_attributes, on=["attr_id", "attr_type", "attr_value"], how="inner")
    mapped["attr_weight"] = mapped["attr_type"].map(RECOMMENDATION_TREND_ATTR_WEIGHTS).astype(float)
    mapped["weighted_score"] = mapped["attr_trend_score"] * mapped["attr_weight"]
    grouped = mapped.groupby(["split", "cutoff_week", "label_week", "article_id"], as_index=False).agg(
        weighted_score=("weighted_score", "sum"),
        matched_weight=("attr_weight", "sum"),
    )
    grouped["trend_score"] = np.where(grouped["matched_weight"] > 0, grouped["weighted_score"] / grouped["matched_weight"], 0.0)
    return grouped.loc[:, ["split", "cutoff_week", "label_week", "article_id", "trend_score"]]
```

Rules:

- Join predictions on `predictions.week_id == cutoff_week`.
- Reject any attempt to join `predictions.week_id == label_week`.
- Use `pred_target_growth`, not `pred_share_t1`.
- Normalize within `split + cutoff_week + attr_type`.
- Filter `attr_type` to `RECOMMENDATION_CORE_ATTR_TYPES`.
- Aggregate to article with `RECOMMENDATION_TREND_ATTR_WEIGHTS`.
- When an article only matches a subset of core attribute types, divide by that article/window's matched weight sum so weights are renormalized over matched attributes.
- Missing trend attributes become `trend_score = 0.0`.
- Constant normalization groups become `0.0`.

- [ ] **Step 4: Implement `PopSimilarityTrendMethod`**

Create `methods/trend_aware/pop_similarity_trend.py` with:

```python
class PopSimilarityTrendMethod:
    name = "pop_similarity_trend"
    method_type = "trend_aware"
    default_candidate_strategy = "default"
    default_weights = {
        "pop_score": 0.35,
        "sim_score": 0.35,
        "trend_score": 0.25,
        "recent_score": 0.05,
    }
    required_features = ("pop_score", "sim_score", "trend_score", "recent_score")
```

Then update `registry.py`:

```python
RECOMMENDATION_METHOD_REGISTRY = {
    "global_popularity": GlobalPopularityMethod(),
    "recent_popularity": RecentPopularityMethod(),
    "attribute_similarity": AttributeSimilarityMethod(),
    "pop_similarity": PopSimilarityMethod(),
    "pop_similarity_trend": PopSimilarityTrendMethod(),
}
```

Behavior:

- Missing stable LightGBM predictions fail with a clear error.
- Candidate strategy defaults to `default`.
- Recommendation outputs include all four score columns.
- Metadata records trend prediction path and `trend_score` configuration.

- [ ] **Step 5: Run tests and commit**

Run:

```sh
uv run pytest tests/test_recommendation_methods.py tests/test_recommendation_ranking.py tests/test_recommendation_retrieval.py -q
uv run python -m compileall -q src
```

Expected: PASS.

Commit:

```sh
git add src/fashion_trend/recommendation/methods/trend_aware src/fashion_trend/recommendation/registry.py src/fashion_trend/recommendation/ranking/features.py tests/test_recommendation_methods.py tests/test_recommendation_ranking.py
git commit -m "feat(recommendation): 实现趋势感知重排序方法"
```

---

### Task 7: Evaluation Metrics and Single-Method Runner

**Files:**
- Create: `src/fashion_trend/recommendation/evaluation/metrics.py`
- Create: `src/fashion_trend/recommendation/evaluation/payloads.py`
- Create: `src/fashion_trend/recommendation/evaluation/runner.py`
- Create: `src/15_eval_recommendations.py`
- Create: `tests/test_recommendation_evaluation.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_recommendation_evaluation.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from fashion_trend.recommendation.evaluation.metrics import evaluate_recommendations


def test_missing_recommendation_user_scores_zero_by_default() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u2",
                "history_purchase_count": 1,
                "label_purchase_count": 1,
            },
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "customer_id": ["u1", "u2"],
            "article_id": ["0000000001", "0000000002"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "method": ["recent_popularity"],
            "prediction": ["0000000001 0000000003 0000000004"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "article_id": ["0000000001"],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["user_count"] == 2
    assert metrics["valid"]["missing_recommendation_user_count"] == 1
    assert metrics["valid"]["hit_rate_at_12"] == 0.5
    assert metrics["valid"]["recall_at_12"] == 0.5


def test_missing_recommendation_user_fails_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="missing"):
        evaluate_recommendations(
            pd.DataFrame(columns=["customer_id", "split", "cutoff_week", "label_week", "method", "prediction"]),
            pd.DataFrame(
                [
                    {
                        "split": "valid",
                        "cutoff_week": 10,
                        "label_week": 11,
                        "customer_id": "u1",
                        "history_purchase_count": 1,
                        "label_purchase_count": 1,
                    }
                ]
            ),
            pd.DataFrame(
                {
                    "split": ["valid"],
                    "cutoff_week": [10],
                    "label_week": [11],
                    "customer_id": ["u1"],
                    "article_id": ["0000000001"],
                }
            ),
            pd.DataFrame(
                {
                    "split": ["valid"],
                    "cutoff_week": [10],
                    "label_week": [11],
                    "article_id": ["0000000001"],
                }
            ),
            top_k=12,
            strict_missing_users=True,
        )


def test_ranking_metrics_use_exact_relevant_sets() -> None:
    target_users = pd.DataFrame(
        [
            {
                "split": "valid",
                "cutoff_week": 10,
                "label_week": 11,
                "customer_id": "u1",
                "history_purchase_count": 1,
                "label_purchase_count": 2,
            }
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 10],
            "label_week": [11, 11],
            "customer_id": ["u1", "u1"],
            "article_id": ["0000000001", "0000000003"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1"],
            "split": ["valid"],
            "cutoff_week": [10],
            "label_week": [11],
            "method": ["pop_similarity_trend"],
            "prediction": ["0000000001 0000000002 0000000003"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"] * 4,
            "cutoff_week": [10] * 4,
            "label_week": [11] * 4,
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["map_at_12"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert metrics["valid"]["recall_at_12"] == 1.0
    assert metrics["valid"]["hit_rate_at_12"] == 1.0
    assert metrics["valid"]["ndcg_at_12"] == pytest.approx((1.0 + 0.5) / (1.0 + 1.0 / 1.584962500721156))
    assert metrics["valid"]["coverage"] == 0.75


def test_coverage_is_computed_per_window_before_split_average() -> None:
    target_users = pd.DataFrame(
        [
            {"split": "valid", "cutoff_week": 10, "label_week": 11, "customer_id": "u1", "history_purchase_count": 1, "label_purchase_count": 1},
            {"split": "valid", "cutoff_week": 20, "label_week": 21, "customer_id": "u2", "history_purchase_count": 1, "label_purchase_count": 1},
        ]
    )
    labels = pd.DataFrame(
        {
            "split": ["valid", "valid"],
            "cutoff_week": [10, 20],
            "label_week": [11, 21],
            "customer_id": ["u1", "u2"],
            "article_id": ["0000000001", "0000000005"],
        }
    )
    recommendations = pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "split": ["valid", "valid"],
            "cutoff_week": [10, 20],
            "label_week": [11, 21],
            "method": ["recent_popularity", "recent_popularity"],
            "prediction": ["0000000001 0000000002", "0000000005"],
        }
    )
    recommendable_pool = pd.DataFrame(
        {
            "split": ["valid"] * 6,
            "cutoff_week": [10, 10, 10, 10, 20, 20],
            "label_week": [11, 11, 11, 11, 21, 21],
            "article_id": ["0000000001", "0000000002", "0000000003", "0000000004", "0000000005", "0000000006"],
        }
    )

    metrics = evaluate_recommendations(
        recommendations,
        target_users,
        labels,
        recommendable_pool,
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["coverage_by_window"] == [
        {"cutoff_week": 10, "label_week": 11, "coverage": 0.5},
        {"cutoff_week": 20, "label_week": 21, "coverage": 0.5},
    ]
    assert metrics["valid"]["coverage"] == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
uv run pytest tests/test_recommendation_evaluation.py -q
```

Expected: FAIL because evaluation modules do not exist.

- [ ] **Step 3: Implement metrics**

Implement:

```python
def parse_prediction_items(prediction: str, top_k: int) -> list[str]:
    items = [item for item in prediction.split() if item]
    if len(items) > top_k:
        raise ValueError(f"prediction contains more than {top_k} items")
    if len(set(items)) != len(items):
        raise ValueError("prediction contains duplicate article_id values")
    return items


def apk(predicted: list[str], relevant: set[str], top_k: int) -> float:
    hits = 0
    score = 0.0
    for index, article_id in enumerate(predicted[:top_k], start=1):
        if article_id in relevant:
            hits += 1
            score += hits / index
    return score / min(len(relevant), top_k) if relevant else 0.0


def recall_at_k(predicted: list[str], relevant: set[str], top_k: int) -> float:
    return len(set(predicted[:top_k]) & relevant) / len(relevant) if relevant else 0.0


def hit_rate_at_k(predicted: list[str], relevant: set[str], top_k: int) -> float:
    return float(bool(set(predicted[:top_k]) & relevant))


def ndcg_at_k(predicted: list[str], relevant: set[str], top_k: int) -> float:
    dcg = sum(1.0 / math.log2(index + 1) for index, article_id in enumerate(predicted[:top_k], start=1) if article_id in relevant)
    ideal_hits = min(len(relevant), top_k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
```

Metric rules:

- Evaluation denominator is `target_users.parquet`.
- Relevant set is deduplicated `evaluation_labels.parquet`.
- Missing eligible user recommendation scores 0 unless strict mode is enabled.
- Empty eligible user set fails.
- Duplicate articles inside a prediction row fail.
- More than 12 articles inside a prediction row fail.
- `article_id` stays string end-to-end; parsing `prediction` must not cast to `int`, because real H&M IDs contain leading zeroes.
- Coverage is computed per window, then averaged per split.

- [ ] **Step 4: Implement payloads, runner and CLI 15**

Create `payloads.py` with:

```python
def build_recommendation_metrics_payload(
    method: str,
    metrics_by_split: dict[str, dict[str, object]],
    input_paths: dict[str, str],
) -> dict[str, object]:
    payload = {
        "method": method,
        "metrics": metrics_by_split,
        "input_paths": input_paths,
    }
    ensure_finite_json_payload(payload)
    return payload


def ensure_finite_json_payload(payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, allow_nan=False)
    json.loads(encoded)
```

Create `runner.py` with no imports from upstream `transactions.paths`, `catalog.paths` or `trend.paths`:

```python
def run_recommendation_evaluation(
    method: str,
    recommendations: pd.DataFrame,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
    recommendable_pool: pd.DataFrame,
    input_paths: dict[str, str],
    strict_missing_users: bool = False,
) -> dict[str, object]:
    output_paths = method_output_paths(method)
    metrics = evaluate_recommendations(recommendations, target_users, labels, recommendable_pool, RECOMMENDATION_TOP_K, strict_missing_users)
    payload = build_recommendation_metrics_payload(method, metrics, input_paths)
    write_json_atomic(payload, output_paths.metrics)
    return payload


def build_recommendable_pool_for_windows(transactions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        active = transactions.loc[transactions["week_id"] <= window.cutoff_week, ["article_id"]].drop_duplicates()
        frames.append(active.assign(split=window.split, cutoff_week=window.cutoff_week, label_week=window.label_week))
    return pd.concat(frames, ignore_index=True)


def input_paths_for_method(method: str) -> dict[str, str]:
    paths = method_output_paths(method)
    return {"recommendations": str(paths.recommendations)}
```

Create `src/15_eval_recommendations.py`:

```python
from __future__ import annotations

import argparse

from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.foundation.logging import configure_logging, get_logger
from fashion_trend.recommendation.contracts import RECOMMENDATION_METHODS
from fashion_trend.recommendation.paths import (
    EVALUATION_LABELS_PATH,
    TARGET_USERS_PATH,
    TIME_WINDOWS_PATH,
    method_output_paths,
)
from fashion_trend.recommendation.readers import (
    read_evaluation_labels,
    read_recommendations,
    read_target_users,
    read_time_windows,
)
from fashion_trend.recommendation.evaluation.runner import (
    build_recommendable_pool_for_windows,
    run_recommendation_evaluation,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=RECOMMENDATION_METHODS, required=True)
    parser.add_argument("--strict-missing-users", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    output_paths = method_output_paths(args.method)
    input_paths = {
        "recommendations": str(output_paths.recommendations),
        "target_users": str(TARGET_USERS_PATH),
        "evaluation_labels": str(EVALUATION_LABELS_PATH),
        "time_windows": str(TIME_WINDOWS_PATH),
        "weekly_transactions": str(WEEKLY_TRANSACTIONS_PATH),
    }
    payload = run_recommendation_evaluation(
        method=args.method,
        recommendations=read_recommendations(output_paths.recommendations),
        target_users=read_target_users(TARGET_USERS_PATH),
        labels=read_evaluation_labels(EVALUATION_LABELS_PATH),
        recommendable_pool=build_recommendable_pool_for_windows(
            read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
            read_time_windows(TIME_WINDOWS_PATH),
        ),
        input_paths=input_paths,
        strict_missing_users=args.strict_missing_users,
    )
    LOGGER.info("recommendation metrics written for method=%s splits=%s", args.method, sorted(payload["metrics"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests and commit**

Run:

```sh
uv run pytest tests/test_recommendation_evaluation.py tests/test_recommendation_methods.py -q
uv run python -m compileall -q src
```

Expected: PASS.

Commit:

```sh
git add src/15_eval_recommendations.py src/fashion_trend/recommendation/evaluation tests/test_recommendation_evaluation.py
git commit -m "feat(recommendation): 实现推荐评价指标"
```

---

### Task 8: Experiment Runner and Ablations

**Files:**
- Create: `src/fashion_trend/recommendation/experiments/grid_search.py`
- Create: `src/fashion_trend/recommendation/experiments/ablation.py`
- Create: `src/fashion_trend/recommendation/experiments/runner.py`
- Create: `src/16_run_recommendation_experiment.py`
- Create: `tests/test_recommendation_experiments.py`

- [ ] **Step 1: Write failing experiment tests**

Create `tests/test_recommendation_experiments.py`:

```python
from __future__ import annotations

import re

import pytest

from fashion_trend.recommendation.experiments.grid_search import iter_weight_grid, select_best_weights
from fashion_trend.recommendation.experiments.runner import (
    candidate_strategy_for_method,
    generate_experiment_run_id,
)
from fashion_trend.recommendation.paths import experiment_run_dir


def test_weight_grid_contains_only_valid_normalized_weights() -> None:
    weights = list(iter_weight_grid())

    assert weights
    assert all(abs(sum(item.values()) - 1.0) <= 1e-9 for item in weights)
    assert all(all(value >= 0.0 for value in item.values()) for item in weights)
    assert {
        "pop_score": 0.4,
        "sim_score": 0.3,
        "trend_score": 0.2,
        "recent_score": 0.1,
    } in weights


def test_select_best_weights_uses_stable_grid_order_for_ties() -> None:
    results = [
        {
            "grid_index": 1,
            "weights": {"pop_score": 0.3, "sim_score": 0.4, "trend_score": 0.2, "recent_score": 0.1},
            "valid_metrics": {"map_at_12": 0.25},
        },
        {
            "grid_index": 0,
            "weights": {"pop_score": 0.4, "sim_score": 0.3, "trend_score": 0.2, "recent_score": 0.1},
            "valid_metrics": {"map_at_12": 0.25},
        },
    ]

    assert select_best_weights(results) == {
        "pop_score": 0.4,
        "sim_score": 0.3,
        "trend_score": 0.2,
        "recent_score": 0.1,
    }


def test_generated_run_id_is_safe_path_segment() -> None:
    run_id = generate_experiment_run_id()

    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", run_id)
    assert experiment_run_dir("main", run_id).as_posix().endswith(f"/experiments/main/runs/{run_id}")


def test_experiment_uses_method_default_candidate_strategy() -> None:
    assert candidate_strategy_for_method("global_popularity") is None
    assert candidate_strategy_for_method("recent_popularity") is None
    assert candidate_strategy_for_method("attribute_similarity") == "similarity"
    assert candidate_strategy_for_method("pop_similarity") == "default"
    assert candidate_strategy_for_method("pop_similarity_trend") == "default"


@pytest.mark.parametrize("bad", ["", ".", "..", "main/evil", "main\\evil"])
def test_experiment_id_rejects_unsafe_path_segments(bad: str) -> None:
    with pytest.raises(ValueError, match="安全"):
        experiment_run_dir(bad, "20260510-120000-1234abcd")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py -q
```

Expected: FAIL because experiment modules do not exist.

- [ ] **Step 3: Implement grid search and run IDs**

Implement in `grid_search.py`:

```python
def iter_weight_grid() -> list[dict[str, float]]:
    grid: list[dict[str, float]] = []
    for pop_score in POP_VALUES:
        for sim_score in SIM_VALUES:
            for trend_score in TREND_VALUES:
                for recent_score in RECENT_VALUES:
                    weights = {
                        "pop_score": pop_score,
                        "sim_score": sim_score,
                        "trend_score": trend_score,
                        "recent_score": recent_score,
                    }
                    if abs(sum(weights.values()) - 1.0) <= 1e-9:
                        grid.append(weights)
    return grid

def select_best_weights(results: list[dict[str, object]], metric_name: str = "map_at_12") -> dict[str, float]:
    if not results:
        raise ValueError("grid search results are empty")
    best = min(results, key=lambda item: (-item["valid_metrics"][metric_name], item["grid_index"]))
    return dict(best["weights"])
```

Grid values:

```python
POP_VALUES = (0.2, 0.3, 0.4)
SIM_VALUES = (0.2, 0.3, 0.4)
TREND_VALUES = (0.1, 0.2, 0.3)
RECENT_VALUES = (0.0, 0.05, 0.1)
```

Only emit combinations where the weight sum is exactly 1 within tolerance.

Implement in `runner.py`:

```python
@dataclass(frozen=True)
class RecommendationExperimentContext:
    transactions: pd.DataFrame
    article_attributes: pd.DataFrame
    trend_predictions: pd.DataFrame


def generate_experiment_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(4)
    run_id = f"{timestamp}-{suffix}"
    validate_safe_path_segment(run_id, "experiment run_id")
    return run_id

def run_recommendation_experiment(
    context: RecommendationExperimentContext,
    experiment_id: str = "main",
) -> dict[str, object]:
    validate_safe_path_segment(experiment_id, "experiment_id")
    inputs = ensure_or_build_recommendation_inputs(context)
    baseline_payloads = run_baseline_methods(context, inputs)
    default_candidates = ensure_or_build_candidates_for_method("pop_similarity_trend", context, inputs)
    if default_candidates is None:
        raise ValueError("pop_similarity_trend requires a candidate strategy")
    search_results = evaluate_weight_grid_on_valid(iter_weight_grid(), context, inputs, default_candidates)
    best_weights = select_best_weights(search_results)
    trend_payload = publish_trend_method_with_weights(best_weights, context, inputs, default_candidates)
    experiment_payload = build_experiment_payload(experiment_id, baseline_payloads, search_results, trend_payload)
    write_json_atomic(experiment_payload, experiment_dir(experiment_id) / "experiment.json")
    return experiment_payload


def ensure_or_build_recommendation_inputs(context: RecommendationExperimentContext) -> RecommendationInputArtifacts:
    if all(path.exists() for path in (TIME_WINDOWS_PATH, TARGET_USERS_PATH, EVALUATION_LABELS_PATH, USER_PROFILE_PATH)):
        return RecommendationInputArtifacts(
            time_windows=read_time_windows(TIME_WINDOWS_PATH),
            target_users=read_target_users(TARGET_USERS_PATH),
            evaluation_labels=read_evaluation_labels(EVALUATION_LABELS_PATH),
            user_profile=read_user_profile(USER_PROFILE_PATH),
        )
    return build_and_write_recommendation_inputs(
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        trend_predictions=context.trend_predictions,
    )


def ensure_or_build_candidate_items(
    strategy: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
) -> pd.DataFrame:
    path = candidate_items_path(strategy)
    if path.exists():
        return read_candidate_items(path)
    build_and_write_candidate_items(
        strategy=strategy,
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        trend_predictions=context.trend_predictions,
        windows=inputs.time_windows,
        target_users=inputs.target_users,
        user_profile=inputs.user_profile,
    )
    return read_candidate_items(path)


def candidate_strategy_for_method(method_name: str) -> str | None:
    return get_recommendation_method(method_name).default_candidate_strategy


def ensure_or_build_candidates_for_method(
    method_name: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
) -> pd.DataFrame | None:
    strategy = candidate_strategy_for_method(method_name)
    if strategy is None:
        return None
    return ensure_or_build_candidate_items(strategy, context, inputs)


def run_baseline_methods(
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for method in ("global_popularity", "recent_popularity", "attribute_similarity", "pop_similarity"):
        candidates = ensure_or_build_candidates_for_method(method, context, inputs)
        result = run_recommendation_method(
            method,
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=inputs.time_windows,
            target_users=inputs.target_users,
            candidates=candidates,
            user_profile=inputs.user_profile,
            trend_predictions=None,
        )
        payloads.append(evaluate_result_for_experiment(method, result, context, inputs))
    return payloads


def evaluate_weight_grid_on_valid(
    weight_grid: list[dict[str, float]],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> list[dict[str, object]]:
    return [
        evaluate_one_weight_run_on_valid(grid_index, weights, context, inputs, candidates)
        for grid_index, weights in enumerate(weight_grid)
    ]


def evaluate_one_weight_run_on_valid(
    grid_index: int,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    run_id = generate_experiment_run_id()
    result = build_recommendation_result_in_memory("pop_similarity_trend", weights, "valid", context, inputs, candidates)
    metrics = evaluate_recommendations(
        result.recommendations,
        inputs.target_users,
        inputs.evaluation_labels,
        build_recommendable_pool_for_windows(
            context.transactions,
            inputs.time_windows,
        ),
        top_k=RECOMMENDATION_TOP_K,
        strict_missing_users=False,
    )
    return {"run_id": run_id, "grid_index": grid_index, "weights": weights, "valid_metrics": metrics["valid"]}


def build_recommendation_result_in_memory(
    method_name: str,
    weights: dict[str, float],
    split_filter: str,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> RecommendationResult:
    result = get_recommendation_method(method_name).build_recommendations(
        RecommendationContext(
            method=method_name,
            top_k=RECOMMENDATION_TOP_K,
            exclude_seen=True,
            transactions=context.transactions,
            article_attributes=context.article_attributes,
            windows=inputs.time_windows,
            target_users=inputs.target_users,
            candidates=candidates,
            user_profile=inputs.user_profile,
            trend_predictions=context.trend_predictions,
            weights=weights,
        )
    )
    recommendations = result.recommendations.loc[result.recommendations["split"] == split_filter].reset_index(drop=True)
    items = result.recommendation_items.loc[result.recommendation_items["split"] == split_filter].reset_index(drop=True)
    return RecommendationResult(recommendations, items, result.params, result.metadata)


def publish_trend_method_with_weights(
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    result = run_recommendation_method(
        "pop_similarity_trend",
        transactions=context.transactions,
        article_attributes=context.article_attributes,
        windows=inputs.time_windows,
        target_users=inputs.target_users,
        candidates=candidates,
        user_profile=inputs.user_profile,
        trend_predictions=context.trend_predictions,
        weights=weights,
    )
    return evaluate_result_for_experiment("pop_similarity_trend", result, context, inputs)


def evaluate_result_for_experiment(
    method: str,
    result: RecommendationResult,
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
) -> dict[str, object]:
    return run_recommendation_evaluation(
        method=method,
        recommendations=result.recommendations,
        target_users=inputs.target_users,
        labels=inputs.evaluation_labels,
        recommendable_pool=build_recommendable_pool_for_windows(context.transactions, inputs.time_windows),
        input_paths={"experiment": "in_memory"},
        strict_missing_users=False,
    )


def build_experiment_payload(
    experiment_id: str,
    baseline_payloads: list[dict[str, object]],
    search_results: list[dict[str, object]],
    trend_payload: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "experiment_path": str(experiment_dir(experiment_id) / "experiment.json"),
        "best_weights": select_best_weights(search_results),
        "search_results": search_results,
        "ablation": build_ablation_summary([*baseline_payloads, trend_payload]),
    }
```

Run ID format:

```text
YYYYMMDD-HHMMSS-<8hex>
```

- [ ] **Step 4: Implement ablation summary and CLI 16**

Implement in `ablation.py`:

```python
def build_ablation_summary(metrics_payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in metrics_payloads:
        method = str(payload["method"])
        for split, metrics in payload["metrics"].items():
            rows.append({"method": method, "split": split, **metrics})
    return sorted(rows, key=lambda row: (row["split"], row["method"]))
```

Create `src/16_run_recommendation_experiment.py`:

```python
from __future__ import annotations

import argparse

from fashion_trend.catalog.paths import GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.foundation.logging import configure_logging, get_logger
from fashion_trend.recommendation.experiments.runner import (
    RecommendationExperimentContext,
    run_recommendation_experiment,
)
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
from fashion_trend.transactions.readers import read_weekly_transactions
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.readers import read_trend_model_predictions

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="main")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    payload = run_recommendation_experiment(
        experiment_id=args.experiment,
        context=RecommendationExperimentContext(
            transactions=read_weekly_transactions(WEEKLY_TRANSACTIONS_PATH),
            article_attributes=read_article_attribute_edges(GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH),
            trend_predictions=read_trend_model_predictions(OUTPUT_MODELS_DIR / "lightgbm" / "predictions.csv"),
        ),
    )
    LOGGER.info("recommendation experiment written: %s", payload["experiment_path"])


if __name__ == "__main__":
    main()
```

Experiment rules:

- `16` is a complete orchestration entry: it reads existing recommendation inputs/default candidates when present, and builds them from the supplied upstream DataFrames when missing.
- Weight search evaluates valid only.
- Test uses valid-selected fixed weights.
- Intermediate search runs stay in memory by default.
- Saved debug runs go under `outputs/recommendation/experiments/<experiment_id>/runs/<run_id>/`.
- Stable method directories are written only for final published method outputs.

- [ ] **Step 5: Run tests and commit**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py tests/test_recommendation_evaluation.py -q
uv run python -m compileall -q src
```

Expected: PASS.

Commit:

```sh
git add src/16_run_recommendation_experiment.py src/fashion_trend/recommendation/experiments tests/test_recommendation_experiments.py
git commit -m "feat(recommendation): 编排推荐实验"
```

---

### Task 9: Integration Tests and Architecture Guardrails

**Files:**
- Modify: `tests/test_recommendation_inputs.py`
- Modify: `tests/test_recommendation_retrieval.py`
- Modify: `tests/test_recommendation_methods.py`
- Modify: `tests/test_recommendation_evaluation.py`
- Modify: `tests/test_recommendation_experiments.py`
- Modify: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: Add small fixture integration test for `12 -> 13 -> 14 -> 15`**

Add `from fashion_trend.recommendation.experiments.runner import candidate_strategy_for_method` to the `tests/test_recommendation_methods.py` imports, then add one in-memory integration test that uses temporary output paths or pure functions:

```python
def test_recommendation_pipeline_small_fixture_runs_without_leakage(tmp_path) -> None:
    transactions = make_small_transactions()
    article_attributes = make_small_article_attributes()
    predictions = make_small_trend_predictions()

    windows = build_recommendation_windows(predictions)
    target_users = build_target_users(transactions, windows)
    labels = build_evaluation_labels(transactions, windows, target_users)
    profile = build_user_profile(transactions, article_attributes, windows, target_users)

    source_frames = build_source_frames_for_frames(
        strategy="default",
        transactions=transactions,
        article_attributes=article_attributes,
        trend_predictions=predictions,
        windows=windows,
        target_users=target_users,
        user_profile=profile,
    )
    candidates = build_candidate_items(strategy="default", source_frames=source_frames)
    result = get_recommendation_method("pop_similarity_trend").build_recommendations(
        RecommendationContext(
            method="pop_similarity_trend",
            top_k=12,
            exclude_seen=True,
            transactions=transactions,
            article_attributes=article_attributes,
            windows=windows,
            target_users=target_users,
            candidates=candidates,
            user_profile=profile,
            trend_predictions=predictions,
            weights={"pop_score": 0.35, "sim_score": 0.35, "trend_score": 0.25, "recent_score": 0.05},
        )
    )
    metrics = evaluate_recommendations(
        result.recommendations,
        target_users,
        labels,
        build_recommendable_pool_for_windows(transactions, windows),
        top_k=12,
        strict_missing_users=False,
    )

    assert metrics["valid"]["user_count"] > 0
    assert result.recommendation_items["rank"].max() <= 12
    assert not result.recommendation_items.duplicated(
        ["customer_id", "split", "cutoff_week", "label_week", "article_id"]
    ).any()


def build_candidates_for_registered_method(
    method_name: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    predictions: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame | None:
    strategy = candidate_strategy_for_method(method_name)
    if strategy is None:
        return None
    return build_candidate_items(
        strategy=strategy,
        source_frames=build_source_frames_for_frames(
            strategy=strategy,
            transactions=transactions,
            article_attributes=article_attributes,
            trend_predictions=predictions,
            windows=windows,
            target_users=target_users,
            user_profile=profile,
        ),
    )


@pytest.mark.parametrize(
    "method_name",
    ["global_popularity", "recent_popularity", "attribute_similarity", "pop_similarity", "pop_similarity_trend"],
)
def test_each_registered_method_builds_recommendations_on_small_fixture(method_name: str) -> None:
    transactions = make_small_transactions()
    article_attributes = make_small_article_attributes()
    predictions = make_small_trend_predictions()
    windows = build_recommendation_windows(predictions)
    target_users = build_target_users(transactions, windows)
    profile = build_user_profile(transactions, article_attributes, windows, target_users)
    candidates = build_candidates_for_registered_method(
        method_name,
        transactions,
        article_attributes,
        predictions,
        windows,
        target_users,
        profile,
    )

    result = get_recommendation_method(method_name).build_recommendations(
        RecommendationContext(
            method=method_name,
            top_k=12,
            exclude_seen=True,
            transactions=transactions,
            article_attributes=article_attributes,
            windows=windows,
            target_users=target_users,
            candidates=candidates,
            user_profile=profile,
            trend_predictions=predictions if method_name == "pop_similarity_trend" else None,
            weights=None,
        )
    )

    assert set(result.recommendations["method"]) == {method_name}
    assert set(result.recommendation_items["method"]) == {method_name}
    assert result.recommendation_items["rank"].between(1, 12).all()
    assert not result.recommendation_items.duplicated(
        ["customer_id", "split", "cutoff_week", "label_week", "article_id"]
    ).any()
```

`make_small_transactions()`、`make_small_article_attributes()` 和 `make_small_trend_predictions()` are local test fixture helpers. `build_source_frames_for_frames()` is a production helper that accepts in-memory frames and is reused by `build_and_write_candidate_items()` after numbered CLI scripts load real inputs.

- [ ] **Step 2: Add architecture tests**

Extend `tests/test_architecture_boundaries.py` with a targeted check that recommendation does not import implementation-only trend modules:

```python
def test_recommendation_does_not_import_trend_training_or_models() -> None:
    assert_package_does_not_import(
        "recommendation",
        {
            "fashion_trend.trend.training",
            "fashion_trend.trend.evaluation.runner",
            "fashion_trend.trend.models",
            "fashion_trend.catalog.graph.builders",
        },
    )
```

- [ ] **Step 3: Run integration and architecture tests**

Run:

```sh
uv run pytest tests/test_recommendation_*.py tests/test_architecture_boundaries.py -q
uv run python -m compileall -q src
```

Expected: PASS.

- [ ] **Step 4: Commit**

```sh
git add tests/test_recommendation_*.py tests/test_architecture_boundaries.py
git commit -m "test(recommendation): 覆盖推荐闭环和架构边界"
```

---

### Task 10: Documentation and Real Artifact Validation

**Files:**
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`

- [ ] **Step 1: Update README status and commands**

Update README so it states:

```text
推荐模块已实现轻量离线实验版，入口为:
src/12_build_recommendation_inputs.py
src/13_build_recommend_candidates.py
src/14_rerank_recommendations.py
src/15_eval_recommendations.py
src/16_run_recommendation_experiment.py
```

Update artifact paths to:

```text
data/processed/recommend/time_windows.parquet
data/processed/recommend/target_users.parquet
data/processed/recommend/evaluation_labels.parquet
data/processed/recommend/user_profile.parquet
data/processed/recommend/candidates/<strategy>/candidate_items.parquet
outputs/recommendation/<method>/recommendations.csv
outputs/recommendation/<method>/recommendation_items.csv
outputs/recommendation/<method>/params.json
outputs/recommendation/<method>/metadata.json
outputs/recommendation/<method>/metrics.json
outputs/recommendation/experiments/<experiment_id>/experiment.json
```

- [ ] **Step 2: Update implementation plan drift**

Update `docs/gpt-research/implementation-plan.md` so historical recommended entries become:

```text
12_build_recommendation_inputs.py
13_build_recommend_candidates.py --strategy <strategy>
14_rerank_recommendations.py --method <method>
15_eval_recommendations.py --method <method>
16_run_recommendation_experiment.py --experiment main
```

Replace old flat paths:

```text
data/processed/recommend/candidate_items.parquet
outputs/recommendation/recommendation_result.csv
outputs/recommendation/recommendation_metrics.json
```

with the strategy-scoped and method-scoped paths from Step 1.

- [ ] **Step 3: Run full recommendation validation on real artifacts**

Run:

```sh
uv run pytest tests/test_recommendation_*.py tests/test_architecture_boundaries.py -q
uv run python -m compileall -q src
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy default
uv run python src/13_build_recommend_candidates.py --strategy similarity
uv run python src/14_rerank_recommendations.py --method global_popularity
uv run python src/14_rerank_recommendations.py --method recent_popularity
uv run python src/14_rerank_recommendations.py --method attribute_similarity
uv run python src/14_rerank_recommendations.py --method pop_similarity
uv run python src/14_rerank_recommendations.py --method pop_similarity_trend
uv run python src/15_eval_recommendations.py --method pop_similarity_trend
uv run python src/16_run_recommendation_experiment.py --experiment main
```

Expected:

- Tests pass.
- Compileall exits 0.
- `data/processed/recommend/` contains windows, target users, labels, profile and `default`/`similarity` candidates.
- Each method output directory contains `recommendations.csv`, `recommendation_items.csv`, `params.json`, `metadata.json`.
- `outputs/recommendation/pop_similarity_trend/metrics.json` exists after `15`.
- `outputs/recommendation/<method>/metrics.json` exists for every method participating in `16`.
- `outputs/recommendation/experiments/main/experiment.json` exists after `16`.

- [ ] **Step 4: Inspect output contracts**

Run:

```sh
uv run python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("outputs/recommendation")
for method in [
    "global_popularity",
    "recent_popularity",
    "attribute_similarity",
    "pop_similarity",
    "pop_similarity_trend",
]:
    items = pd.read_csv(
        root / method / "recommendation_items.csv",
        dtype={
            "customer_id": "string",
            "article_id": "string",
            "method": "string",
            "candidate_sources": "string",
        },
    )
    assert items["rank"].between(1, 12).all(), method
    assert not items.duplicated(["customer_id", "split", "cutoff_week", "label_week", "article_id"]).any(), method
    assert items[["score", "pop_score", "sim_score", "trend_score", "recent_score"]].notna().all().all(), method
    assert (items[["pop_score", "sim_score", "trend_score", "recent_score"]] >= 0).all().all(), method
    assert (items[["pop_score", "sim_score", "trend_score", "recent_score"]] <= 1).all().all(), method
    with (root / method / "metadata.json").open(encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["method"] == method
    with (root / method / "metrics.json").open(encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert metrics["method"] == method
    assert "metrics" in metrics

with (root / "experiments" / "main" / "experiment.json").open(encoding="utf-8") as fh:
    experiment = json.load(fh)
assert experiment["experiment_id"] == "main"
print("recommendation artifact audit passed")
PY
```

Expected: prints `recommendation artifact audit passed`.

- [ ] **Step 5: Final diff and commit**

Run:

```sh
git diff --check
git status --short
```

Expected: only source, tests and docs for recommendation are modified.

Commit:

```sh
git add README.md docs/gpt-research/implementation-plan.md
git commit -m "docs(recommendation): 同步推荐系统实现说明"
```

---

## Self-Review Checklist

- [ ] Spec coverage: contracts, time windows, target users, evaluation labels, strategy-scoped candidates, method-scoped outputs, trend score off-by-one, normalization constant groups, metrics denominator, grid search isolation and path safety are each mapped to a task.
- [ ] Placeholder scan: the plan contains no unfinished-marker text and no unspecified edge-case steps.
- [ ] Type consistency: `method`, `strategy`, `split`, `cutoff_week`, `label_week`, `customer_id`, `article_id`, `prediction`, score column names and artifact filenames match the design spec.
- [ ] Boundary consistency: `recommendation` uses upstream public readers and contracts only; no imports from trend training, trend models, trend evaluation runner or catalog graph builders.
- [ ] Validation consistency: implementation ends with focused recommendation tests, architecture tests, compileall, real artifact CLI run and output contract audit.
