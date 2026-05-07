# 趋势标签与趋势训练样本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先把 `attribute_week_heat.csv` 调整为完整属性-周面板，再生成趋势标签表 `attribute_week_target.csv` 和趋势训练样本表 `trend_model_samples.parquet`。

**Architecture:** 继续把可测试的数据逻辑放在 `src/fashion_trend/trend.py`，脚本只负责编排路径、日志、校验、写出和退出码。完整面板由 `article_week_sales.week_id` 和 `nodes_attribute.attr_id` 构造，阶段 5 的标签与样本都消费这张完整事实表，避免不同阶段各自补 0。样本表只使用当前周和历史周特征，`t+1` 信息只出现在目标字段。

**Tech Stack:** Python 3.10-3.12，`pandas`，`numpy`，`pyarrow`，标准库 `csv` / `pathlib` / `unittest` / `tempfile`，现有 `fashion_trend.config.PATH` 和 `fashion_trend.log`。

---

## 文件结构

- Modify: `src/fashion_trend/trend.py`
  - 增加属性节点读取、完整属性-周面板构造、趋势标签构造、趋势样本构造、Parquet 写出和对应校验函数。
- Modify: `src/06_compute_attribute_week_heat.py`
  - 额外读取 `nodes_attribute.csv`，调用完整面板版本的热度构造与校验。
- Create: `src/07_build_trend_targets.py`
  - 从完整属性热度表生成 `attribute_week_target.csv`。
- Create: `src/08_build_trend_model_samples.py`
  - 从完整热度表、标签表、属性节点和层级边生成 `trend_model_samples.parquet`。
- Modify: `src/fashion_trend/config.py`
  - 增加 `FEATURES_DIR`、`trend_attribute_week_target` 和 `features_trend_model_samples` 路径。
- Modify: `tests/test_trend.py`
  - 扩展完整面板、标签、样本、读写和校验测试。
- Modify: `README.md`
  - 更新阶段状态、流水线命令、输出字段和完整面板口径。

## Task 1: 调整属性周热度为完整面板

**Files:**
- Modify: `src/fashion_trend/trend.py`
- Modify: `src/06_compute_attribute_week_heat.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: 在测试中加入属性节点样例**

Modify `tests/test_trend.py` near existing sample helpers:

```python
def sample_attribute_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "attr_id": [
                "colour_group_name::Black",
                "colour_group_name::White",
                "colour_group_name::Blue",
                "product_type_name::Vest top",
                "product_type_name::Bra",
                "product_type_name::Dress",
            ],
            "attr_type": [
                "colour_group_name",
                "colour_group_name",
                "colour_group_name",
                "product_type_name",
                "product_type_name",
                "product_type_name",
            ],
            "attr_value": ["Black", "White", "Blue", "Vest top", "Bra", "Dress"],
            "attr_node_id": [
                "colour_group_name::Black",
                "colour_group_name::White",
                "colour_group_name::Blue",
                "product_type_name::Vest top",
                "product_type_name::Bra",
                "product_type_name::Dress",
            ],
            "article_count": [2, 1, 0, 2, 1, 0],
            "is_core_attr": [1, 1, 1, 1, 1, 1],
            "level": ["child", "child", "child", "child", "child", "child"],
        }
    )
```

- [ ] **Step 2: 写失败测试，完整面板会补齐零热度属性**

Modify `tests/test_trend.py` imports:

```python
from fashion_trend.catalog.graph import read_article_attribute_edges, read_attribute_nodes
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.transactions.weekly import read_weekly_transactions
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.schema import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
)
```

Update existing `sample_attribute_week_heat` helper:

```python
def sample_attribute_week_heat() -> pd.DataFrame:
    return build_attribute_week_heat_frame(
        sample_attribute_article_week_sales(),
        sample_article_attribute_edges(),
        sample_attribute_nodes(),
    )
```

Add test:

```python
def test_build_attribute_week_heat_frame_builds_complete_attribute_week_panel(
    self,
) -> None:
    heat = build_attribute_week_heat_frame(
        sample_attribute_article_week_sales(),
        sample_article_attribute_edges(),
        sample_attribute_nodes(),
    )

    self.assertEqual(len(heat), 12)
    self.assertEqual(heat.columns.tolist(), list(ATTRIBUTE_WEEK_HEAT_COLUMNS))
    self.assertEqual(set(heat["week_id"]), {0, 1})
    self.assertEqual(set(heat["attr_id"]), set(sample_attribute_nodes()["attr_id"]))

    zero_row = heat[
        (heat["week_id"] == 0)
        & (heat["attr_id"] == "colour_group_name::Blue")
    ].iloc[0]
    self.assertEqual(int(zero_row["heat_cnt"]), 0)
    self.assertEqual(float(zero_row["heat_share"]), 0.0)
    self.assertEqual(float(zero_row["log_heat"]), 0.0)
```

- [ ] **Step 3: 运行测试，确认旧签名失败**

Run:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.AttributeWeekHeatFrameTests.test_build_attribute_week_heat_frame_builds_complete_attribute_week_panel -v
```

Expected:

```text
TypeError: build_attribute_week_heat_frame() takes 2 positional arguments but 3 were given
```

- [ ] **Step 4: 在 `trend.py` 增加属性节点常量、读取和校验**

Modify `src/fashion_trend/trend.py` after `ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS`:

```python
ATTRIBUTE_NODE_HEAT_COLUMNS: tuple[str, ...] = (
    "attr_id",
    "attr_type",
    "attr_value",
    "article_count",
    "is_core_attr",
)

ATTRIBUTE_NODE_HEAT_DTYPES: dict[str, str] = {
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "article_count": "int64",
    "is_core_attr": "int64",
}
```

Add functions near existing readers and validators:

```python
def validate_attribute_nodes_for_heat(attribute_nodes: pd.DataFrame) -> None:
    validate_required_columns(
        attribute_nodes.columns.tolist(),
        ATTRIBUTE_NODE_HEAT_COLUMNS,
        source_name="属性节点表",
    )
    validate_no_missing_values(
        attribute_nodes,
        ATTRIBUTE_NODE_HEAT_COLUMNS,
        source_name="属性节点表",
    )
    validate_unique_key(attribute_nodes, ["attr_id"], source_name="属性节点表")
    validate_non_negative_values(
        attribute_nodes,
        ["article_count"],
        source_name="属性节点表",
    )
    invalid_core_flags = sorted(set(attribute_nodes["is_core_attr"]) - {0, 1})
    if invalid_core_flags:
        raise ValueError("属性节点表存在非法 is_core_attr: " + str(invalid_core_flags))


def read_attribute_nodes(attribute_nodes_path: Path) -> pd.DataFrame:
    if not attribute_nodes_path.exists():
        raise FileNotFoundError(f"属性节点表不存在: {attribute_nodes_path}")

    try:
        header = pd.read_csv(attribute_nodes_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性节点表: {attribute_nodes_path}") from exc

    missing_columns = sorted(set(ATTRIBUTE_NODE_HEAT_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "属性节点表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_nodes_path}"
        )

    try:
        return pd.read_csv(
            attribute_nodes_path,
            usecols=list(ATTRIBUTE_NODE_HEAT_COLUMNS),
            dtype=ATTRIBUTE_NODE_HEAT_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性节点表: {attribute_nodes_path}") from exc
```

- [ ] **Step 5: 修改完整面板构造逻辑**

Replace `build_attribute_week_heat_frame` in `src/fashion_trend/trend.py` with:

```python
def build_attribute_week_heat_frame(
    article_week_sales: pd.DataFrame,
    article_attribute_edges: pd.DataFrame,
    attribute_nodes: pd.DataFrame,
) -> pd.DataFrame:
    validate_article_week_sales(article_week_sales)
    validate_article_attribute_edges_for_heat(article_attribute_edges)
    validate_attribute_nodes_for_heat(attribute_nodes)
    validate_all_sales_articles_have_attribute_edges(
        article_week_sales,
        article_attribute_edges,
    )

    edge_attr_ids = set(article_attribute_edges["attr_id"].astype("string"))
    node_attr_ids = set(attribute_nodes["attr_id"].astype("string"))
    missing_node_attr_ids = sorted(edge_attr_ids - node_attr_ids)
    if missing_node_attr_ids:
        examples = ", ".join(missing_node_attr_ids[:5])
        raise ValueError(
            f"商品-属性边表存在 {len(missing_node_attr_ids)} 个 attr_id "
            f"无法映射到属性节点，例如: {examples}"
        )

    normalized_sales = article_week_sales.loc[
        :, ["week_id", "article_id", "sales_cnt"]
    ].copy()
    normalized_sales["article_id"] = normalized_sales["article_id"].astype("string")

    normalized_edges = article_attribute_edges.loc[
        :, list(ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS)
    ].copy()
    normalized_edges["article_id"] = normalized_edges["article_id"].astype("string")
    normalized_edges["attr_id"] = normalized_edges["attr_id"].astype("string")
    normalized_edges["attr_type"] = normalized_edges["attr_type"].astype("string")
    normalized_edges["attr_value"] = normalized_edges["attr_value"].astype("string")

    joined = normalized_sales.merge(normalized_edges, on="article_id", how="inner")
    observed_heat = (
        joined.groupby(["week_id", "attr_id"], as_index=False)["sales_cnt"]
        .sum()
        .rename(columns={"sales_cnt": "heat_cnt"})
    )

    weeks = pd.DataFrame({"week_id": sorted(normalized_sales["week_id"].unique())})
    attributes = attribute_nodes.loc[
        :, ["attr_id", "attr_type", "attr_value"]
    ].copy()
    panel = weeks.merge(attributes, how="cross")
    heat = panel.merge(observed_heat, on=["week_id", "attr_id"], how="left")
    heat["heat_cnt"] = heat["heat_cnt"].fillna(0).astype("int64")
    heat["type_total_heat"] = heat.groupby(["week_id", "attr_type"])[
        "heat_cnt"
    ].transform("sum")
    heat["type_total_heat"] = heat["type_total_heat"].astype("int64")
    heat["heat_share"] = np.where(
        heat["type_total_heat"] > 0,
        heat["heat_cnt"] / heat["type_total_heat"],
        0.0,
    )
    heat["log_heat"] = np.log1p(heat["heat_cnt"])

    heat = heat.sort_values(
        ["week_id", "attr_type", "heat_cnt", "attr_id"],
        ascending=[True, True, False, True],
        ignore_index=True,
    )
    heat["rank_in_type"] = (
        heat.groupby(["week_id", "attr_type"]).cumcount().add(1).astype("int64")
    )

    return heat.loc[:, list(ATTRIBUTE_WEEK_HEAT_COLUMNS)].sort_values(
        ["week_id", "attr_type", "rank_in_type", "attr_id"],
        ignore_index=True,
    )
```

- [ ] **Step 6: 放宽并增强属性热度校验**

Modify `validate_attribute_week_heat` signature and body in `src/fashion_trend/trend.py`:

```python
def validate_attribute_week_heat(
    attribute_week_heat: pd.DataFrame,
    expected_week_ids: Sequence[int] | None = None,
    expected_attribute_nodes: pd.DataFrame | None = None,
) -> None:
    validate_required_columns(
        attribute_week_heat.columns.tolist(),
        ATTRIBUTE_WEEK_HEAT_COLUMNS,
        source_name="属性周热度表",
    )
    validate_no_missing_values(
        attribute_week_heat,
        ATTRIBUTE_WEEK_HEAT_COLUMNS,
        source_name="属性周热度表",
    )
    validate_unique_key(
        attribute_week_heat,
        ["week_id", "attr_id"],
        source_name="属性周热度表",
    )
    validate_non_negative_values(
        attribute_week_heat,
        ["heat_cnt", "type_total_heat", "heat_share", "log_heat"],
        source_name="属性周热度表",
    )
    validate_positive_values(
        attribute_week_heat,
        ["rank_in_type"],
        source_name="属性周热度表",
    )

    if expected_week_ids is not None:
        actual_week_ids = set(attribute_week_heat["week_id"].astype("int64"))
        expected_week_id_set = set(int(week_id) for week_id in expected_week_ids)
        if actual_week_ids != expected_week_id_set:
            raise ValueError("属性周热度表输出周集合与预期不一致。")

    if expected_attribute_nodes is not None:
        validate_attribute_nodes_for_heat(expected_attribute_nodes)
        actual_attr_ids = set(attribute_week_heat["attr_id"].astype("string"))
        expected_attr_ids = set(expected_attribute_nodes["attr_id"].astype("string"))
        if actual_attr_ids != expected_attr_ids:
            raise ValueError("属性周热度表输出属性集合与属性节点表不一致。")
        if expected_week_ids is not None:
            expected_rows = len(expected_week_ids) * len(expected_attribute_nodes)
            if len(attribute_week_heat) != expected_rows:
                raise ValueError(
                    f"属性周热度表行数应为 {expected_rows}，实际为 {len(attribute_week_heat)}。"
                )
```

Keep the existing checks for `type_total_heat`, `heat_share`, `log_heat`, and ranks, but change share total logic to:

```python
    share_totals = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "heat_share"
    ].sum()
    type_totals = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "type_total_heat"
    ].first()
    invalid_positive_share_totals = share_totals[
        (type_totals > 0) & ~np.isclose(share_totals, 1.0, atol=1e-9, rtol=0)
    ]
    invalid_zero_share_totals = share_totals[
        (type_totals == 0) & ~np.isclose(share_totals, 0.0, atol=1e-9, rtol=0)
    ]
    if not invalid_positive_share_totals.empty or not invalid_zero_share_totals.empty:
        raise ValueError("属性周热度表存在 week_id + attr_type 占比和不符合总热度状态的分组。")
```

- [ ] **Step 7: 修改 `06_compute_attribute_week_heat.py` 读取属性节点**

Modify imports:

```python
from fashion_trend.catalog.graph import read_article_attribute_edges, read_attribute_nodes
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.trend.article_sales import (
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
```

Modify `compute_attribute_week_heat` before building heat:

```python
    log.info(f"输入属性节点表: {PATH['graph_nodes_attribute']}", source=LOG_SOURCE)
    attribute_nodes = read_attribute_nodes(PATH["graph_nodes_attribute"])
    validate_attribute_nodes_for_heat(attribute_nodes)

    attribute_week_heat = build_attribute_week_heat_frame(
        article_week_sales,
        article_attribute_edges,
        attribute_nodes,
    )
    validate_attribute_week_heat(
        attribute_week_heat,
        expected_week_ids=sorted(article_week_sales["week_id"].unique()),
        expected_attribute_nodes=attribute_nodes,
    )
```

- [ ] **Step 8: 运行完整面板相关测试**

Run:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.AttributeWeekHeatFrameTests -v
```

Expected:

```text
OK
```

- [ ] **Step 9: Commit checkpoint**

Only run this commit step after user has explicitly authorized committing implementation changes:

```sh
git add src/fashion_trend/trend.py src/06_compute_attribute_week_heat.py tests/test_trend.py
git commit -m "fix(trend): 生成完整属性周热度面板"
```

## Task 2: 新增趋势标签表

**Files:**
- Modify: `src/fashion_trend/trend.py`
- Create: `src/07_build_trend_targets.py`
- Modify: `src/fashion_trend/config.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: 写趋势标签字段常量和失败测试**

Modify imports in `tests/test_trend.py`:

```python
from fashion_trend.catalog.graph import read_article_attribute_edges, read_attribute_nodes
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.transactions.weekly import read_weekly_transactions
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.schema import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
)
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    validate_attribute_week_target,
)
```

Add tests:

```python
class AttributeWeekTargetFrameTests(unittest.TestCase):
    def test_build_attribute_week_target_frame_calculates_next_week_targets(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())

        self.assertEqual(target.columns.tolist(), list(ATTRIBUTE_WEEK_TARGET_COLUMNS))
        self.assertEqual(len(target), 6)
        self.assertEqual(set(target["week_id"]), {0})

        black = target[target["attr_id"] == "colour_group_name::Black"].iloc[0]
        self.assertEqual(int(black["heat_t"]), 2)
        self.assertEqual(int(black["heat_t1"]), 1)
        self.assertTrue(math.isclose(float(black["share_t"]), 2 / 3))
        self.assertTrue(math.isclose(float(black["share_t1"]), 1.0))
        self.assertTrue(
            math.isclose(
                float(black["target_growth"]),
                math.log((1.0 + 1e-6) / ((2 / 3) + 1e-6)),
            )
        )
        self.assertTrue(
            math.isclose(float(black["target_log_heat_t1"]), math.log1p(1))
        )
        self.assertEqual(int(black["target_rank_in_type_t1"]), 1)

    def test_validate_attribute_week_target_rejects_inconsistent_growth(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[0, "target_growth"] = 999.0

        with self.assertRaisesRegex(ValueError, "target_growth"):
            validate_attribute_week_target(target)
```

- [ ] **Step 2: 运行测试，确认常量或函数缺失**

Run:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.AttributeWeekTargetFrameTests -v
```

Expected:

```text
ImportError: cannot import name 'ATTRIBUTE_WEEK_TARGET_COLUMNS'
```

- [ ] **Step 3: 在 `trend.py` 实现趋势标签逻辑**

Add constant:

```python
ATTRIBUTE_WEEK_TARGET_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "heat_t",
    "heat_t1",
    "share_t",
    "share_t1",
    "rank_in_type_t",
    "target_log_heat_t1",
    "target_growth",
    "target_rank_in_type_t1",
)
```

Add functions:

```python
def build_attribute_week_target_frame(
    attribute_week_heat: pd.DataFrame,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    validate_attribute_week_heat(attribute_week_heat)
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数。")

    current = attribute_week_heat.loc[
        :,
        [
            "week_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "heat_cnt",
            "heat_share",
            "rank_in_type",
        ],
    ].rename(
        columns={
            "heat_cnt": "heat_t",
            "heat_share": "share_t",
            "rank_in_type": "rank_in_type_t",
        }
    )
    next_week = attribute_week_heat.loc[
        :, ["week_id", "attr_id", "heat_cnt", "heat_share", "log_heat", "rank_in_type"]
    ].copy()
    next_week["week_id"] = next_week["week_id"] - 1
    next_week = next_week.rename(
        columns={
            "heat_cnt": "heat_t1",
            "heat_share": "share_t1",
            "log_heat": "target_log_heat_t1",
            "rank_in_type": "target_rank_in_type_t1",
        }
    )

    target = current.merge(next_week, on=["week_id", "attr_id"], how="inner")
    target["target_growth"] = np.log(
        (target["share_t1"] + epsilon) / (target["share_t"] + epsilon)
    )
    target = target.loc[:, list(ATTRIBUTE_WEEK_TARGET_COLUMNS)].sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return target


def validate_attribute_week_target(
    attribute_week_target: pd.DataFrame,
    expected_week_count: int | None = None,
    expected_attribute_count: int | None = None,
    epsilon: float = 1e-6,
) -> None:
    validate_required_columns(
        attribute_week_target.columns.tolist(),
        ATTRIBUTE_WEEK_TARGET_COLUMNS,
        source_name="属性趋势标签表",
    )
    validate_no_missing_values(
        attribute_week_target,
        ATTRIBUTE_WEEK_TARGET_COLUMNS,
        source_name="属性趋势标签表",
    )
    validate_unique_key(
        attribute_week_target,
        ["week_id", "attr_id"],
        source_name="属性趋势标签表",
    )
    validate_non_negative_values(
        attribute_week_target,
        ["heat_t", "heat_t1", "share_t", "share_t1", "target_log_heat_t1"],
        source_name="属性趋势标签表",
    )
    validate_positive_values(
        attribute_week_target,
        ["rank_in_type_t", "target_rank_in_type_t1"],
        source_name="属性趋势标签表",
    )
    if expected_week_count is not None and expected_attribute_count is not None:
        expected_rows = (expected_week_count - 1) * expected_attribute_count
        if len(attribute_week_target) != expected_rows:
            raise ValueError(
                f"属性趋势标签表行数应为 {expected_rows}，实际为 {len(attribute_week_target)}。"
            )
    if (attribute_week_target[["share_t", "share_t1"]] > 1).any().any():
        raise ValueError("属性趋势标签表存在 share 大于 1 的记录。")
    numeric_targets = attribute_week_target[
        ["target_growth", "target_log_heat_t1"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_targets).all():
        raise ValueError("属性趋势标签表存在非有限目标字段。")
    expected_growth = np.log(
        (attribute_week_target["share_t1"] + epsilon)
        / (attribute_week_target["share_t"] + epsilon)
    )
    if not np.allclose(
        attribute_week_target["target_growth"].to_numpy(dtype=float),
        expected_growth.to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("属性趋势标签表存在 target_growth 与公式不一致。")
    expected_log_heat_t1 = np.log1p(attribute_week_target["heat_t1"])
    if not np.allclose(
        attribute_week_target["target_log_heat_t1"].to_numpy(dtype=float),
        expected_log_heat_t1.to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("属性趋势标签表存在 target_log_heat_t1 与公式不一致。")
```

- [ ] **Step 4: 更新配置路径**

Modify `src/fashion_trend/config.py`:

```python
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
TREND_DIR = PROCESSED_DIR / "trend"
FEATURES_DIR = PROCESSED_DIR / "features"
```

Add to `PATH`:

```python
    "trend_attribute_week_target": TREND_DIR / "attribute_week_target.csv",
    "features_trend_model_samples": FEATURES_DIR / "trend_model_samples.parquet",
```

- [ ] **Step 5: 创建趋势标签脚本**

Create `src/07_build_trend_targets.py`:

```python
from __future__ import annotations

from fashion_trend import log
from fashion_trend.config import PATH
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.trend.attribute_heat import (
    read_attribute_week_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    validate_attribute_week_target,
)

LOG_SOURCE = "trend-targets"


def build_trend_targets() -> dict[str, int]:
    log.info(f"输入属性周热度表: {PATH['trend_attribute_week_heat']}", source=LOG_SOURCE)
    attribute_week_heat = read_attribute_week_heat(PATH["trend_attribute_week_heat"])
    validate_attribute_week_heat(attribute_week_heat)

    attribute_week_target = build_attribute_week_target_frame(attribute_week_heat)
    validate_attribute_week_target(
        attribute_week_target,
        expected_week_count=int(attribute_week_heat["week_id"].nunique()),
        expected_attribute_count=int(attribute_week_heat["attr_id"].nunique()),
    )
    write_csv_atomic(attribute_week_target, PATH["trend_attribute_week_target"])

    return {
        "rows": len(attribute_week_target),
        "weeks": int(attribute_week_target["week_id"].nunique()),
        "attributes": int(attribute_week_target["attr_id"].nunique()),
    }


def main() -> int:
    try:
        stats = build_trend_targets()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"趋势标签行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖当前周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {stats['attributes']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {PATH['trend_attribute_week_target']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Also add `read_attribute_week_heat` to `trend.py`:

```python
ATTRIBUTE_WEEK_HEAT_DTYPES: dict[str, str] = {
    "week_id": "int64",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "heat_cnt": "int64",
    "type_total_heat": "int64",
    "heat_share": "float64",
    "log_heat": "float64",
    "rank_in_type": "int64",
}


def read_attribute_week_heat(attribute_week_heat_path: Path) -> pd.DataFrame:
    if not attribute_week_heat_path.exists():
        raise FileNotFoundError(f"属性周热度表不存在: {attribute_week_heat_path}")

    try:
        header = pd.read_csv(attribute_week_heat_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性周热度表: {attribute_week_heat_path}") from exc

    missing_columns = sorted(set(ATTRIBUTE_WEEK_HEAT_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "属性周热度表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_week_heat_path}"
        )

    try:
        return pd.read_csv(
            attribute_week_heat_path,
            usecols=list(ATTRIBUTE_WEEK_HEAT_COLUMNS),
            dtype=ATTRIBUTE_WEEK_HEAT_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性周热度表: {attribute_week_heat_path}") from exc
```

- [ ] **Step 6: 运行趋势标签测试**

Run:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.AttributeWeekTargetFrameTests -v
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit checkpoint**

Only run this commit step after user has explicitly authorized committing implementation changes:

```sh
git add src/fashion_trend/trend.py src/fashion_trend/config.py src/07_build_trend_targets.py tests/test_trend.py
git commit -m "feat(trend): 构建属性趋势标签"
```

## Task 3: 新增趋势训练样本表

**Files:**
- Modify: `src/fashion_trend/trend.py`
- Create: `src/08_build_trend_model_samples.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: 写样本表失败测试**

Modify imports in `tests/test_trend.py`:

```python
from fashion_trend.catalog.graph import read_article_attribute_edges, read_attribute_nodes
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.transactions.weekly import read_weekly_transactions
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.samples import (
    build_attribute_graph_features_frame,
    build_trend_model_samples_frame,
    validate_trend_model_samples,
)
from fashion_trend.trend.schema import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
)
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    validate_attribute_week_target,
)
```

Add helpers:

```python
def sample_long_attribute_week_heat() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for week_id, black_heat, white_heat, blue_heat in [
        (0, 2, 1, 0),
        (1, 1, 0, 0),
        (2, 3, 1, 0),
        (3, 4, 1, 0),
        (4, 8, 2, 0),
        (5, 4, 4, 2),
    ]:
        colour_total = black_heat + white_heat + blue_heat
        for attr_id, attr_value, heat_cnt in [
            ("colour_group_name::Black", "Black", black_heat),
            ("colour_group_name::White", "White", white_heat),
            ("colour_group_name::Blue", "Blue", blue_heat),
        ]:
            records.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": "colour_group_name",
                    "attr_value": attr_value,
                    "heat_cnt": heat_cnt,
                    "type_total_heat": colour_total,
                    "heat_share": heat_cnt / colour_total if colour_total else 0.0,
                    "log_heat": math.log1p(heat_cnt),
                    "rank_in_type": 1,
                }
            )
        product_total = 1
        for attr_id, attr_value, heat_cnt in [
            ("product_type_name::Vest top", "Vest top", 1),
            ("product_type_name::Bra", "Bra", 0),
            ("product_type_name::Dress", "Dress", 0),
        ]:
            records.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": "product_type_name",
                    "attr_value": attr_value,
                    "heat_cnt": heat_cnt,
                    "type_total_heat": product_total,
                    "heat_share": heat_cnt / product_total,
                    "log_heat": math.log1p(heat_cnt),
                    "rank_in_type": 1,
                }
            )
    heat = pd.DataFrame(records).sort_values(
        ["week_id", "attr_type", "heat_cnt", "attr_id"],
        ascending=[True, True, False, True],
        ignore_index=True,
    )
    heat["rank_in_type"] = (
        heat.groupby(["week_id", "attr_type"]).cumcount().add(1).astype("int64")
    )
    return heat.loc[:, list(ATTRIBUTE_WEEK_HEAT_COLUMNS)]


def sample_attribute_hierarchy_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_attr_id": [
                "colour_group_name::Black",
                "product_type_name::Vest top",
            ],
            "child_attr_id": [
                "colour_group_name::White",
                "product_type_name::Bra",
            ],
            "parent_attr_type": ["colour_group_name", "product_type_name"],
            "child_attr_type": ["colour_group_name", "product_type_name"],
            "relation_type": ["test_contains_colour", "test_contains_type"],
            "edge_weight": [2, 1],
        }
    )
```

Add tests:

```python
class TrendModelSamplesFrameTests(unittest.TestCase):
    def test_build_trend_model_samples_frame_uses_lags_and_targets(self) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat)
        samples = build_trend_model_samples_frame(
            heat,
            target,
            sample_attribute_nodes(),
            sample_attribute_hierarchy_edges(),
        )

        self.assertEqual(samples.columns.tolist(), list(TREND_MODEL_SAMPLE_COLUMNS))
        self.assertEqual(set(samples["week_id"]), {4})

        black = samples[samples["attr_id"] == "colour_group_name::Black"].iloc[0]
        self.assertEqual(int(black["heat_t"]), 8)
        self.assertEqual(int(black["heat_lag_1"]), 4)
        self.assertEqual(int(black["heat_lag_4"]), 2)
        self.assertTrue(math.isclose(float(black["heat_ma_4"]), (1 + 3 + 4 + 8) / 4))
        self.assertTrue(
            math.isclose(
                float(black["growth_lag_1"]),
                math.log((black["share_t"] + 1e-6) / (black["share_lag_1"] + 1e-6)),
            )
        )
        self.assertEqual(int(black["child_count"]), 1)
        self.assertEqual(int(black["parent_count"]), 0)
        self.assertEqual(int(black["degree"]), 1)
        self.assertEqual(int(black["history_total_heat_t"]), 18)
        self.assertEqual(int(black["history_active_weeks_t"]), 5)
        self.assertFalse(bool(black["is_trend_eligible_t"]))
        self.assertIn("target_growth", samples.columns)

    def test_validate_trend_model_samples_rejects_missing_target(self) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat).drop(columns=["target_growth"])

        with self.assertRaisesRegex(ValueError, "target_growth"):
            build_trend_model_samples_frame(
                heat,
                target,
                sample_attribute_nodes(),
                sample_attribute_hierarchy_edges(),
            )
```

- [ ] **Step 2: 运行测试，确认样本函数缺失**

Run:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.TrendModelSamplesFrameTests -v
```

Expected:

```text
ImportError: cannot import name 'TREND_MODEL_SAMPLE_COLUMNS'
```

- [ ] **Step 3: 在 `trend.py` 增加层级边和样本字段常量**

Add constants:

```python
ATTRIBUTE_HIERARCHY_EDGE_COLUMNS: tuple[str, ...] = (
    "parent_attr_id",
    "child_attr_id",
    "parent_attr_type",
    "child_attr_type",
    "relation_type",
    "edge_weight",
)

ATTRIBUTE_HIERARCHY_EDGE_DTYPES: dict[str, str] = {
    "parent_attr_id": "string",
    "child_attr_id": "string",
    "parent_attr_type": "string",
    "child_attr_type": "string",
    "relation_type": "string",
    "edge_weight": "int64",
}

TREND_MODEL_SAMPLE_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
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
    "target_growth",
    "target_log_heat_t1",
    "target_rank_in_type_t1",
)
```

- [ ] **Step 4: 实现层级边读取和图特征**

Add functions:

```python
def read_attribute_hierarchy_edges(attribute_hierarchy_edges_path: Path) -> pd.DataFrame:
    if not attribute_hierarchy_edges_path.exists():
        raise FileNotFoundError(f"属性层级边表不存在: {attribute_hierarchy_edges_path}")

    try:
        header = pd.read_csv(attribute_hierarchy_edges_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性层级边表: {attribute_hierarchy_edges_path}") from exc

    missing_columns = sorted(set(ATTRIBUTE_HIERARCHY_EDGE_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "属性层级边表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_hierarchy_edges_path}"
        )

    try:
        return pd.read_csv(
            attribute_hierarchy_edges_path,
            usecols=list(ATTRIBUTE_HIERARCHY_EDGE_COLUMNS),
            dtype=ATTRIBUTE_HIERARCHY_EDGE_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性层级边表: {attribute_hierarchy_edges_path}") from exc


def build_attribute_graph_features_frame(
    attribute_nodes: pd.DataFrame,
    attribute_hierarchy_edges: pd.DataFrame,
) -> pd.DataFrame:
    validate_attribute_nodes_for_heat(attribute_nodes)
    validate_required_columns(
        attribute_hierarchy_edges.columns.tolist(),
        ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
        source_name="属性层级边表",
    )
    validate_no_missing_values(
        attribute_hierarchy_edges,
        ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
        source_name="属性层级边表",
    )

    features = attribute_nodes.loc[
        :, ["attr_id", "article_count", "is_core_attr"]
    ].copy()
    parent_counts = (
        attribute_hierarchy_edges.groupby("child_attr_id")
        .size()
        .rename("parent_count")
        .reset_index()
        .rename(columns={"child_attr_id": "attr_id"})
    )
    child_counts = (
        attribute_hierarchy_edges.groupby("parent_attr_id")
        .size()
        .rename("child_count")
        .reset_index()
        .rename(columns={"parent_attr_id": "attr_id"})
    )
    features = features.merge(parent_counts, on="attr_id", how="left")
    features = features.merge(child_counts, on="attr_id", how="left")
    features[["parent_count", "child_count"]] = features[
        ["parent_count", "child_count"]
    ].fillna(0).astype("int64")
    features["degree"] = features["parent_count"] + features["child_count"]
    return features
```

- [ ] **Step 5: 实现趋势样本构造和校验**

Add functions:

```python
def build_trend_model_samples_frame(
    attribute_week_heat: pd.DataFrame,
    attribute_week_target: pd.DataFrame,
    attribute_nodes: pd.DataFrame,
    attribute_hierarchy_edges: pd.DataFrame,
    min_lag_weeks: int = 4,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    validate_attribute_week_heat(attribute_week_heat)
    validate_attribute_week_target(attribute_week_target)
    validate_attribute_nodes_for_heat(attribute_nodes)
    if min_lag_weeks < 1:
        raise ValueError("min_lag_weeks 必须大于等于 1。")

    base = attribute_week_heat.sort_values(["attr_id", "week_id"]).copy()
    base = base.rename(
        columns={
            "heat_cnt": "heat_t",
            "heat_share": "share_t",
            "log_heat": "log_heat_t",
            "rank_in_type": "rank_in_type_t",
        }
    )
    grouped = base.groupby("attr_id", sort=False)
    for lag in range(1, min_lag_weeks + 1):
        base[f"heat_lag_{lag}"] = grouped["heat_t"].shift(lag)
        base[f"share_lag_{lag}"] = grouped["share_t"].shift(lag)

    base["growth_lag_1"] = np.log(
        (base["share_t"] + epsilon) / (base["share_lag_1"] + epsilon)
    )
    base["growth_lag_2"] = np.log(
        (base["share_lag_1"] + epsilon) / (base["share_lag_2"] + epsilon)
    )
    base["acc_lag_1"] = base["growth_lag_1"] - base["growth_lag_2"]

    rolling = grouped[["heat_t", "share_t"]].rolling(
        window=min_lag_weeks,
        min_periods=min_lag_weeks,
    )
    base["heat_ma_4"] = rolling["heat_t"].mean().reset_index(level=0, drop=True)
    base["share_ma_4"] = rolling["share_t"].mean().reset_index(level=0, drop=True)
    base["share_std_4"] = (
        grouped["share_t"]
        .rolling(window=min_lag_weeks, min_periods=min_lag_weeks)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    base["share_max_4"] = rolling["share_t"].max().reset_index(level=0, drop=True)
    base["share_min_4"] = rolling["share_t"].min().reset_index(level=0, drop=True)

    base["history_total_heat_t"] = grouped["heat_t"].cumsum()
    base["history_active_weeks_t"] = (
        base["heat_t"].gt(0).astype("int64").groupby(base["attr_id"]).cumsum()
    )
    base["is_trend_eligible_t"] = (
        (base["history_total_heat_t"] >= 100)
        & (base["history_active_weeks_t"] >= 8)
    )
    base["week_index"] = base["week_id"]
    base["week_mod_52"] = base["week_id"] % 52

    graph_features = build_attribute_graph_features_frame(
        attribute_nodes,
        attribute_hierarchy_edges,
    )
    samples = base.merge(graph_features, on="attr_id", how="left")
    samples = samples.merge(
        attribute_week_target.loc[
            :,
            [
                "week_id",
                "attr_id",
                "target_growth",
                "target_log_heat_t1",
                "target_rank_in_type_t1",
            ],
        ],
        on=["week_id", "attr_id"],
        how="inner",
    )
    samples = samples[samples["week_id"] >= min_lag_weeks].copy()
    samples = samples.loc[:, list(TREND_MODEL_SAMPLE_COLUMNS)].sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    validate_trend_model_samples(samples)
    return samples


def validate_trend_model_samples(trend_model_samples: pd.DataFrame) -> None:
    validate_required_columns(
        trend_model_samples.columns.tolist(),
        TREND_MODEL_SAMPLE_COLUMNS,
        source_name="趋势训练样本表",
    )
    validate_no_missing_values(
        trend_model_samples,
        TREND_MODEL_SAMPLE_COLUMNS,
        source_name="趋势训练样本表",
    )
    validate_unique_key(
        trend_model_samples,
        ["week_id", "attr_id"],
        source_name="趋势训练样本表",
    )
    numeric_values = trend_model_samples.drop(columns=["attr_id", "attr_type", "attr_value"])
    if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
        raise ValueError("趋势训练样本表存在非有限数值。")
```

- [ ] **Step 6: 使用共享 Parquet 写出**

Current code uses `fashion_trend.foundation.io.write_parquet_atomic`:

```python
write_parquet_atomic(dataframe, output_path)
        remove_file_if_exists(tmp_output_path)
        raise
```

- [ ] **Step 7: 创建趋势样本脚本**

Create `src/08_build_trend_model_samples.py`:

```python
from __future__ import annotations

from fashion_trend import log
from fashion_trend.config import PATH
from fashion_trend.catalog.graph import read_attribute_nodes
from fashion_trend.foundation.io import write_parquet_atomic
from fashion_trend.trend.attribute_heat import read_attribute_week_heat
from fashion_trend.trend.samples import (
    build_trend_model_samples_frame,
    read_attribute_hierarchy_edges,
    validate_trend_model_samples,
)
from fashion_trend.trend.targets import read_attribute_week_target

LOG_SOURCE = "trend-model-samples"


def build_trend_model_samples() -> dict[str, int]:
    log.info(f"输入属性周热度表: {PATH['trend_attribute_week_heat']}", source=LOG_SOURCE)
    attribute_week_heat = read_attribute_week_heat(PATH["trend_attribute_week_heat"])

    log.info(f"输入趋势标签表: {PATH['trend_attribute_week_target']}", source=LOG_SOURCE)
    attribute_week_target = read_attribute_week_target(PATH["trend_attribute_week_target"])

    log.info(f"输入属性节点表: {PATH['graph_nodes_attribute']}", source=LOG_SOURCE)
    attribute_nodes = read_attribute_nodes(PATH["graph_nodes_attribute"])

    log.info(
        f"输入属性层级边表: {PATH['graph_edges_attribute_hierarchy']}",
        source=LOG_SOURCE,
    )
    attribute_hierarchy_edges = read_attribute_hierarchy_edges(
        PATH["graph_edges_attribute_hierarchy"]
    )

    samples = build_trend_model_samples_frame(
        attribute_week_heat,
        attribute_week_target,
        attribute_nodes,
        attribute_hierarchy_edges,
    )
    validate_trend_model_samples(samples)
    write_parquet_atomic(samples, PATH["features_trend_model_samples"])

    return {
        "rows": len(samples),
        "weeks": int(samples["week_id"].nunique()),
        "attributes": int(samples["attr_id"].nunique()),
    }


def main() -> int:
    try:
        stats = build_trend_model_samples()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"趋势训练样本行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖样本周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {stats['attributes']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {PATH['features_trend_model_samples']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Also add `read_attribute_week_target` to `trend.py`:

```python
ATTRIBUTE_WEEK_TARGET_DTYPES: dict[str, str] = {
    "week_id": "int64",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "heat_t": "int64",
    "heat_t1": "int64",
    "share_t": "float64",
    "share_t1": "float64",
    "rank_in_type_t": "int64",
    "target_log_heat_t1": "float64",
    "target_growth": "float64",
    "target_rank_in_type_t1": "int64",
}


def read_attribute_week_target(attribute_week_target_path: Path) -> pd.DataFrame:
    if not attribute_week_target_path.exists():
        raise FileNotFoundError(f"属性趋势标签表不存在: {attribute_week_target_path}")

    try:
        header = pd.read_csv(attribute_week_target_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性趋势标签表: {attribute_week_target_path}") from exc

    missing_columns = sorted(set(ATTRIBUTE_WEEK_TARGET_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "属性趋势标签表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_week_target_path}"
        )

    try:
        return pd.read_csv(
            attribute_week_target_path,
            usecols=list(ATTRIBUTE_WEEK_TARGET_COLUMNS),
            dtype=ATTRIBUTE_WEEK_TARGET_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性趋势标签表: {attribute_week_target_path}") from exc
```

- [ ] **Step 8: 运行趋势样本测试**

Run:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.TrendModelSamplesFrameTests -v
```

Expected:

```text
OK
```

- [ ] **Step 9: Commit checkpoint**

Only run this commit step after user has explicitly authorized committing implementation changes:

```sh
git add src/fashion_trend/trend.py src/08_build_trend_model_samples.py tests/test_trend.py
git commit -m "feat(trend): 构建趋势训练样本"
```

## Task 4: 文档与真实流水线验证

**Files:**
- Modify: `README.md`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: 更新 README 流水线命令和状态**

Modify the preprocessing command block in `README.md` to include:

```sh
uv run python src/07_build_trend_targets.py
uv run python src/08_build_trend_model_samples.py
```

Modify the stage table so:

```text
趋势标签 | 已实现 | attribute_week_target.csv
趋势样本 | 已实现 | trend_model_samples.parquet
模型训练、推荐评价 | 尚未实现 | 后续模型和推荐结果
```

- [ ] **Step 2: 增加 README 字段说明**

Add a short section after `attribute_week_heat.csv` describing:

```text
attribute_week_heat.csv 现在是完整属性-周面板：每个 week_id 都覆盖 nodes_attribute.csv 中的全部 attr_id。
没有观测购买的属性-周显式保留 heat_cnt = 0、heat_share = 0、log_heat = 0。
```

Add sections for:

```text
data/processed/trend/attribute_week_target.csv
data/processed/features/trend_model_samples.parquet
```

Include the target formula:

```text
target_growth = log((share_t1 + 1e-6) / (share_t + 1e-6))
```

- [ ] **Step 3: 运行单元测试全集**

Run:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 4: 重新生成真实产物**

Run in order:

```sh
PYTHONPATH=src .venv/bin/python src/06_compute_attribute_week_heat.py
PYTHONPATH=src .venv/bin/python src/07_build_trend_targets.py
PYTHONPATH=src .venv/bin/python src/08_build_trend_model_samples.py
```

Expected:

```text
three commands exit 0
```

- [ ] **Step 5: 抽查真实产物行数和边界**

Run:

```sh
PYTHONPATH=src .venv/bin/python -c "import pandas as pd; h=pd.read_csv('data/processed/trend/attribute_week_heat.csv'); t=pd.read_csv('data/processed/trend/attribute_week_target.csv'); s=pd.read_parquet('data/processed/features/trend_model_samples.parquet'); print({'heat_rows': len(h), 'heat_weeks': h.week_id.nunique(), 'heat_attrs': h.attr_id.nunique(), 'target_rows': len(t), 'sample_rows': len(s), 'sample_min_week': int(s.week_id.min()), 'sample_max_week': int(s.week_id.max()), 'heat_missing': int(h.isna().sum().sum()), 'target_missing': int(t.isna().sum().sum()), 'sample_missing': int(s.isna().sum().sum())})"
```

Expected on the current data shape:

```text
{'heat_rows': 62160, 'heat_weeks': 105, 'heat_attrs': 592, 'target_rows': 61568, 'sample_rows': 59200, 'sample_min_week': 4, 'sample_max_week': 103, 'heat_missing': 0, 'target_missing': 0, 'sample_missing': 0}
```

- [ ] **Step 6: 检查 diff 范围**

Run:

```sh
git status --short
git diff --stat
```

Expected changed paths:

```text
README.md
src/06_compute_attribute_week_heat.py
src/07_build_trend_targets.py
src/08_build_trend_model_samples.py
src/fashion_trend/config.py
src/fashion_trend/trend.py
tests/test_trend.py
data/processed/trend/attribute_week_heat.csv
data/processed/trend/attribute_week_target.csv
data/processed/features/trend_model_samples.parquet
```

- [ ] **Step 7: Final commit checkpoint**

Only run this commit step after user has explicitly authorized committing implementation changes and generated data:

```sh
git add README.md src/06_compute_attribute_week_heat.py src/07_build_trend_targets.py src/08_build_trend_model_samples.py src/fashion_trend/config.py src/fashion_trend/trend.py tests/test_trend.py data/processed/trend/attribute_week_heat.csv data/processed/trend/attribute_week_target.csv data/processed/features/trend_model_samples.parquet
git commit -m "feat(trend): 生成趋势标签和训练样本"
```

## Self-Review

- Spec coverage: 本计划覆盖完整属性-周面板、趋势标签表、趋势训练样本表、配置路径、脚本、测试、README 和真实产物验证。
- Completion-marker scan: 计划未发现未完成标记或空泛步骤。
- Type consistency: 字段名与设计文档一致；热度表、标签表和样本表都以 `week_id + attr_id` 为唯一键；`t+1` 只在目标字段中出现。
