# articles 清洗与属性图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 `articles.csv` 先生成 `data/interim/articles_clean_mvp.csv` 和 `data/interim/articles_clean.csv`，再基于 `articles_clean.csv` 构建 4 张静态属性图 CSV 表。

**Architecture:** 新增 `fashion_trend.articles` 作为可测试的业务模块，CLI 脚本只负责调用模块函数和日志输出。第一阶段清洗原始 articles 字段并写入 interim；第二阶段只读取 `articles_clean.csv`，生成商品节点、属性节点、商品-属性边和属性层级边。

**Tech Stack:** Python 标准库 `unittest`、`tempfile`、`pathlib`，`pandas`，现有 `fashion_trend.config.PATH` 和 `fashion_trend.log`。

---

## 文件结构

- 创建：`src/fashion_trend/articles.py`
  - 负责字段常量、清洗校验、clean 表生成、属性图 DataFrame 构建、图表写出。
- 创建：`src/03_clean_articles.py`
  - CLI 入口，读取 `PATH["raw_articles"]`，写出两份 `data/interim` 中间表。
- 创建：`src/04_build_attribute_graph.py`
  - CLI 入口，读取 `PATH["interim_articles_clean"]`，写出 4 张图表。
- 修改：`src/fashion_trend/config.py`
  - 增加 `PATH` 中的 clean 表路径和 graph 表路径。
- 创建：`tests/test_articles_clean.py`
  - 覆盖清洗字段、列顺序、缺失字段、缺失值和文件写出。
- 创建：`tests/test_attribute_graph.py`
  - 覆盖 4 张属性图表的核心结构、边权重和引用完整性。

## Task 1: 清洗字段常量与 DataFrame 清洗逻辑

**Files:**
- Create: `src/fashion_trend/articles.py`
- Create: `tests/test_articles_clean.py`

- [ ] **Step 1: 写失败测试，锁定 clean_mvp 和 clean 的列集合**

写入 `tests/test_articles_clean.py`：

```python
from __future__ import annotations

import unittest

import pandas as pd

from fashion_trend.articles import (
    CLEAN_ARTICLE_COLUMNS,
    MVP_ARTICLE_COLUMNS,
    build_clean_article_frames,
    validate_required_columns,
)


def sample_raw_articles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["0108775015", "0108775044"],
            "product_code": ["0108775", "0108775"],
            "prod_name": ["Strap top", "Strap top"],
            "product_group_name": ["Garment Upper body", "Garment Upper body"],
            "product_type_name": ["Vest top", "Vest top"],
            "garment_group_name": ["Jersey Basic", "Jersey Basic"],
            "colour_group_name": ["Black", "White"],
            "graphical_appearance_name": ["Solid", "Solid"],
            "perceived_colour_master_name": ["Black", "White"],
            "index_group_name": ["Ladieswear", "Ladieswear"],
            "index_name": ["Ladieswear", "Ladieswear"],
            "section_name": ["Womens Everyday Basics", "Womens Everyday Basics"],
            "department_name": ["Jersey Basic", "Jersey Basic"],
            "detail_desc": ["Ignored text", "Ignored text"],
        }
    )


class CleanArticleFrameTests(unittest.TestCase):
    def test_build_clean_article_frames_returns_fixed_columns(self) -> None:
        mvp_articles, clean_articles = build_clean_article_frames(sample_raw_articles())

        self.assertEqual(list(mvp_articles.columns), list(MVP_ARTICLE_COLUMNS))
        self.assertEqual(list(clean_articles.columns), list(CLEAN_ARTICLE_COLUMNS))
        self.assertEqual(len(mvp_articles), 2)
        self.assertEqual(len(clean_articles), 2)
        self.assertEqual(mvp_articles["article_id"].tolist(), ["0108775015", "0108775044"])
        self.assertNotIn("detail_desc", mvp_articles.columns)
        self.assertNotIn("detail_desc", clean_articles.columns)

    def test_build_clean_article_frames_casts_identifier_columns_to_string(self) -> None:
        raw_articles = sample_raw_articles()
        raw_articles["article_id"] = pd.Series(["0108775015", "0108775044"], dtype="string")
        raw_articles["product_code"] = pd.Series(["0108775", "0108775"], dtype="string")

        mvp_articles, clean_articles = build_clean_article_frames(raw_articles)

        self.assertEqual(mvp_articles["article_id"].dtype.name, "string")
        self.assertEqual(mvp_articles["product_code"].dtype.name, "string")
        self.assertEqual(clean_articles["article_id"].dtype.name, "string")
        self.assertEqual(clean_articles["product_code"].dtype.name, "string")

    def test_validate_required_columns_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少必要字段: product_type_name"):
            validate_required_columns(
                ["article_id", "product_group_name"],
                ["article_id", "product_group_name", "product_type_name"],
                source_name="测试 articles 表",
            )

    def test_build_clean_article_frames_rejects_missing_values(self) -> None:
        raw_articles = sample_raw_articles()
        raw_articles.loc[0, "colour_group_name"] = pd.NA

        with self.assertRaisesRegex(ValueError, "colour_group_name"):
            build_clean_article_frames(raw_articles)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认因为模块缺失而失败**

Run:

```bash
PYTHONPATH=src uv run python -m unittest tests.test_articles_clean -v
```

Expected:

```text
ModuleNotFoundError: No module named 'fashion_trend.articles'
```

- [ ] **Step 3: 实现最小清洗逻辑**

创建 `src/fashion_trend/articles.py`：

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

ARTICLE_ID_COLUMN = "article_id"
PRODUCT_CODE_COLUMN = "product_code"

CORE_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "product_group_name",
    "product_type_name",
    "garment_group_name",
    "colour_group_name",
    "graphical_appearance_name",
)

HIERARCHY_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "perceived_colour_master_name",
    "index_group_name",
    "index_name",
    "section_name",
    "department_name",
)

MVP_ARTICLE_COLUMNS: tuple[str, ...] = (
    "article_id",
    "product_code",
    "prod_name",
    *CORE_ATTRIBUTE_COLUMNS,
)

CLEAN_ARTICLE_COLUMNS: tuple[str, ...] = (
    *MVP_ARTICLE_COLUMNS,
    *HIERARCHY_ATTRIBUTE_COLUMNS,
)


def validate_required_columns(
    actual_columns: Sequence[str],
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(actual_columns))
    if missing_columns:
        raise ValueError(f"{source_name} 缺少必要字段: " + ", ".join(missing_columns))


def validate_no_missing_values(
    articles: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = [
        column for column in required_columns if int(articles[column].isna().sum()) > 0
    ]
    if missing_columns:
        raise ValueError(f"{source_name} 存在缺失值字段: " + ", ".join(missing_columns))


def normalize_article_identifiers(articles: pd.DataFrame) -> pd.DataFrame:
    normalized = articles.copy()
    normalized[ARTICLE_ID_COLUMN] = normalized[ARTICLE_ID_COLUMN].astype("string")
    normalized[PRODUCT_CODE_COLUMN] = normalized[PRODUCT_CODE_COLUMN].astype("string")
    return normalized


def build_clean_article_frames(raw_articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_required_columns(
        raw_articles.columns.tolist(),
        CLEAN_ARTICLE_COLUMNS,
        source_name="原始 articles.csv",
    )
    validate_no_missing_values(
        raw_articles,
        CLEAN_ARTICLE_COLUMNS,
        source_name="原始 articles.csv",
    )

    normalized_articles = normalize_article_identifiers(raw_articles)
    mvp_articles = normalized_articles.loc[:, list(MVP_ARTICLE_COLUMNS)].copy()
    clean_articles = normalized_articles.loc[:, list(CLEAN_ARTICLE_COLUMNS)].copy()
    return mvp_articles, clean_articles
```

- [ ] **Step 4: 运行测试，确认通过**

Run:

```bash
PYTHONPATH=src uv run python -m unittest tests.test_articles_clean -v
```

Expected:

```text
Ran 4 tests

OK
```

- [ ] **Step 5: 提交**

```bash
git add src/fashion_trend/articles.py tests/test_articles_clean.py
git commit -m "test: cover articles cleaning frames"
```

## Task 2: clean_mvp 与 clean 文件写出

**Files:**
- Modify: `src/fashion_trend/articles.py`
- Modify: `src/fashion_trend/config.py`
- Create: `src/03_clean_articles.py`
- Modify: `tests/test_articles_clean.py`

- [ ] **Step 1: 写失败测试，锁定 CSV 文件读写行为**

追加到 `tests/test_articles_clean.py`：

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from fashion_trend.articles import clean_articles_file


class CleanArticleFileTests(unittest.TestCase):
    def test_clean_articles_file_writes_mvp_and_clean_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path = tmp_path / "articles.csv"
            mvp_output_path = tmp_path / "articles_clean_mvp.csv"
            clean_output_path = tmp_path / "articles_clean.csv"
            sample_raw_articles().to_csv(raw_path, index=False)

            row_count = clean_articles_file(
                raw_articles_path=raw_path,
                mvp_output_path=mvp_output_path,
                clean_output_path=clean_output_path,
            )

            self.assertEqual(row_count, 2)
            mvp_articles = pd.read_csv(mvp_output_path, dtype={"article_id": "string"})
            clean_articles = pd.read_csv(clean_output_path, dtype={"article_id": "string"})
            self.assertEqual(list(mvp_articles.columns), list(MVP_ARTICLE_COLUMNS))
            self.assertEqual(list(clean_articles.columns), list(CLEAN_ARTICLE_COLUMNS))
            self.assertEqual(mvp_articles["article_id"].tolist(), ["0108775015", "0108775044"])
            self.assertEqual(clean_articles["article_id"].tolist(), ["0108775015", "0108775044"])

    def test_clean_articles_file_fails_when_input_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "原始商品文件不存在"):
                clean_articles_file(
                    raw_articles_path=tmp_path / "missing.csv",
                    mvp_output_path=tmp_path / "articles_clean_mvp.csv",
                    clean_output_path=tmp_path / "articles_clean.csv",
                )
```

- [ ] **Step 2: 运行测试，确认因为函数缺失而失败**

Run:

```bash
PYTHONPATH=src uv run python -m unittest tests.test_articles_clean -v
```

Expected:

```text
ImportError: cannot import name 'clean_articles_file'
```

- [ ] **Step 3: 实现文件读写函数**

追加到 `src/fashion_trend/articles.py`：

```python
def read_articles_csv(raw_articles_path: Path) -> pd.DataFrame:
    if not raw_articles_path.exists():
        raise FileNotFoundError(f"原始商品文件不存在: {raw_articles_path}")

    return pd.read_csv(
        raw_articles_path,
        usecols=list(CLEAN_ARTICLE_COLUMNS),
        dtype={
            "article_id": "string",
            "product_code": "string",
        },
    )


def write_csv_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(tmp_output_path, index=False)
    tmp_output_path.replace(output_path)


def clean_articles_file(
    raw_articles_path: Path,
    mvp_output_path: Path,
    clean_output_path: Path,
) -> int:
    raw_articles = read_articles_csv(raw_articles_path)
    mvp_articles, clean_articles = build_clean_article_frames(raw_articles)

    write_csv_atomically(mvp_articles, mvp_output_path)
    write_csv_atomically(clean_articles, clean_output_path)

    if len(mvp_articles) != len(clean_articles):
        raise RuntimeError(
            f"clean_mvp 与 clean 行数不一致: {len(mvp_articles)} != {len(clean_articles)}"
        )
    if set(mvp_articles["article_id"]) != set(clean_articles["article_id"]):
        raise RuntimeError("clean_mvp 与 clean 的 article_id 集合不一致。")

    return len(clean_articles)
```

- [ ] **Step 4: 增加路径配置和 CLI 脚本**

修改 `src/fashion_trend/config.py` 的 `PATH`：

```python
PATH = {
    # ---------------- Raw H&M data ----------------
    "raw_transactions": RAW_HM_DIR / "transactions_train.csv",
    "raw_articles": RAW_HM_DIR / "articles.csv",
    "raw_customers": RAW_HM_DIR / "customers.csv",
    # ---------------- Interim data ----------------
    "interim_transactions_weekly": INTERIM_DIR / "transactions_train_weekly.parquet",
    "interim_articles_clean_mvp": INTERIM_DIR / "articles_clean_mvp.csv",
    "interim_articles_clean": INTERIM_DIR / "articles_clean.csv",
}
```

创建 `src/03_clean_articles.py`：

```python
from __future__ import annotations

from fashion_trend import log
from fashion_trend.articles import clean_articles_file
from fashion_trend.config import PATH

LOG_SOURCE = "clean-articles"


def main() -> int:
    try:
        row_count = clean_articles_file(
            raw_articles_path=PATH["raw_articles"],
            mvp_output_path=PATH["interim_articles_clean_mvp"],
            clean_output_path=PATH["interim_articles_clean"],
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"已写出商品中间表行数: {row_count:,}", source=LOG_SOURCE)
    log.info(f"MVP 输出文件: {PATH['interim_articles_clean_mvp']}", source=LOG_SOURCE)
    log.info(f"稳妥版输出文件: {PATH['interim_articles_clean']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行测试，确认通过**

Run:

```bash
PYTHONPATH=src uv run python -m unittest tests.test_articles_clean -v
```

Expected:

```text
Ran 6 tests

OK
```

- [ ] **Step 6: 提交**

```bash
git add src/fashion_trend/articles.py src/fashion_trend/config.py src/03_clean_articles.py tests/test_articles_clean.py
git commit -m "feat: write clean articles interim files"
```

## Task 3: 属性图 DataFrame 构建逻辑

**Files:**
- Modify: `src/fashion_trend/articles.py`
- Create: `tests/test_attribute_graph.py`

- [ ] **Step 1: 写失败测试，锁定 4 张图表的核心结构**

写入 `tests/test_attribute_graph.py`：

```python
from __future__ import annotations

import unittest

import pandas as pd

from fashion_trend.articles import (
    ATTRIBUTE_COLUMNS,
    build_article_attribute_edges,
    build_article_nodes,
    build_attribute_hierarchy_edges,
    build_attribute_nodes,
)


def sample_clean_articles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["0108775015", "0108775044", "0110065001"],
            "product_code": ["0108775", "0108775", "0110065"],
            "prod_name": ["Strap top", "Strap top", "Bra"],
            "product_group_name": ["Garment Upper body", "Garment Upper body", "Underwear"],
            "product_type_name": ["Vest top", "T-shirt", "Bra"],
            "garment_group_name": ["Jersey Basic", "Jersey Basic", "Under-, Nightwear"],
            "colour_group_name": ["Black", "Black", "White"],
            "graphical_appearance_name": ["Solid", "Stripe", "Solid"],
            "perceived_colour_master_name": ["Black", "Black", "White"],
            "index_group_name": ["Ladieswear", "Ladieswear", "Ladieswear"],
            "index_name": ["Ladieswear", "Ladieswear", "Lingeries/Tights"],
            "section_name": [
                "Womens Everyday Basics",
                "Womens Everyday Basics",
                "Womens Lingerie",
            ],
            "department_name": ["Jersey Basic", "Jersey Basic", "Clean Lingerie"],
        }
    )


class AttributeGraphBuilderTests(unittest.TestCase):
    def test_build_article_nodes_returns_one_node_per_article(self) -> None:
        nodes_article = build_article_nodes(sample_clean_articles())

        self.assertEqual(
            nodes_article.columns.tolist(),
            ["article_id", "article_node_id", "product_code", "prod_name"],
        )
        self.assertEqual(len(nodes_article), 3)
        self.assertEqual(nodes_article.loc[0, "article_node_id"], "article_0108775015")

    def test_build_attribute_nodes_counts_articles_and_marks_core_fields(self) -> None:
        nodes_attribute = build_attribute_nodes(sample_clean_articles())

        black_node = nodes_attribute.set_index("attr_id").loc["colour_group_name::Black"]
        self.assertEqual(int(black_node["article_count"]), 2)
        self.assertEqual(int(black_node["is_core_attr"]), 1)
        self.assertEqual(black_node["level"], "child")

        index_node = nodes_attribute.set_index("attr_id").loc["index_name::Ladieswear"]
        self.assertEqual(int(index_node["is_core_attr"]), 0)
        self.assertEqual(index_node["level"], "parent_child")

    def test_build_article_attribute_edges_returns_one_edge_per_article_attribute(self) -> None:
        edges = build_article_attribute_edges(sample_clean_articles())

        self.assertEqual(len(edges), 3 * len(ATTRIBUTE_COLUMNS))
        first_edge = edges[
            (edges["article_id"] == "0108775015")
            & (edges["attr_id"] == "product_group_name::Garment Upper body")
        ].iloc[0]
        self.assertEqual(first_edge["article_node_id"], "article_0108775015")
        self.assertEqual(first_edge["edge_type"], "has_product_group")
        self.assertEqual(float(first_edge["edge_weight"]), 1.0)

    def test_build_attribute_hierarchy_edges_counts_parent_child_cooccurrence(self) -> None:
        hierarchy_edges = build_attribute_hierarchy_edges(sample_clean_articles())

        colour_edge = hierarchy_edges[
            (
                hierarchy_edges["parent_attr_id"]
                == "perceived_colour_master_name::Black"
            )
            & (hierarchy_edges["child_attr_id"] == "colour_group_name::Black")
        ].iloc[0]
        self.assertEqual(colour_edge["relation_type"], "colour_master_contains_colour")
        self.assertEqual(int(colour_edge["edge_weight"]), 2)

        section_edge = hierarchy_edges[
            (hierarchy_edges["parent_attr_id"] == "section_name::Womens Everyday Basics")
            & (hierarchy_edges["child_attr_id"] == "department_name::Jersey Basic")
        ].iloc[0]
        self.assertEqual(section_edge["relation_type"], "section_contains_department")
        self.assertEqual(int(section_edge["edge_weight"]), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认因为函数缺失而失败**

Run:

```bash
PYTHONPATH=src uv run python -m unittest tests.test_attribute_graph -v
```

Expected:

```text
ImportError: cannot import name 'ATTRIBUTE_COLUMNS'
```

- [ ] **Step 3: 实现图构建函数**

追加到 `src/fashion_trend/articles.py`：

```python
ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    *CORE_ATTRIBUTE_COLUMNS,
    *HIERARCHY_ATTRIBUTE_COLUMNS,
)

LEVEL_BY_ATTRIBUTE: dict[str, str] = {
    "product_group_name": "parent",
    "product_type_name": "child",
    "perceived_colour_master_name": "parent",
    "colour_group_name": "child",
    "index_group_name": "parent",
    "index_name": "parent_child",
    "section_name": "parent_child",
    "department_name": "child",
    "garment_group_name": "flat",
    "graphical_appearance_name": "flat",
}

HIERARCHY_RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("product_group_name", "product_type_name", "product_group_contains_type"),
    ("perceived_colour_master_name", "colour_group_name", "colour_master_contains_colour"),
    ("index_group_name", "index_name", "index_group_contains_index"),
    ("index_name", "section_name", "index_contains_section"),
    ("section_name", "department_name", "section_contains_department"),
)


def make_attr_id(attr_type: str, attr_value: str) -> str:
    return f"{attr_type}::{attr_value}"


def make_article_node_id(article_id: str) -> str:
    return f"article_{article_id}"


def make_edge_type(attr_type: str) -> str:
    return "has_" + attr_type.removesuffix("_name")


def build_article_nodes(clean_articles: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        clean_articles.columns.tolist(),
        ["article_id", "product_code", "prod_name"],
        source_name="articles_clean.csv",
    )
    article_nodes = clean_articles.loc[
        :, ["article_id", "product_code", "prod_name"]
    ].copy()
    article_nodes["article_id"] = article_nodes["article_id"].astype("string")
    article_nodes.insert(
        1,
        "article_node_id",
        article_nodes["article_id"].map(make_article_node_id),
    )
    return article_nodes[["article_id", "article_node_id", "product_code", "prod_name"]]


def build_attribute_nodes(clean_articles: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        clean_articles.columns.tolist(),
        ATTRIBUTE_COLUMNS,
        source_name="articles_clean.csv",
    )

    frames: list[pd.DataFrame] = []
    for attr_type in ATTRIBUTE_COLUMNS:
        counts = (
            clean_articles.groupby(attr_type, dropna=False)
            .size()
            .reset_index(name="article_count")
            .rename(columns={attr_type: "attr_value"})
        )
        counts["attr_type"] = attr_type
        frames.append(counts)

    attribute_nodes = pd.concat(frames, ignore_index=True)
    attribute_nodes["attr_value"] = attribute_nodes["attr_value"].astype("string")
    attribute_nodes["attr_id"] = attribute_nodes.apply(
        lambda row: make_attr_id(row["attr_type"], row["attr_value"]),
        axis=1,
    )
    attribute_nodes["attr_node_id"] = attribute_nodes["attr_id"]
    attribute_nodes["is_core_attr"] = attribute_nodes["attr_type"].isin(
        CORE_ATTRIBUTE_COLUMNS
    ).astype("int8")
    attribute_nodes["level"] = attribute_nodes["attr_type"].map(LEVEL_BY_ATTRIBUTE)
    return attribute_nodes[
        [
            "attr_id",
            "attr_type",
            "attr_value",
            "attr_node_id",
            "article_count",
            "is_core_attr",
            "level",
        ]
    ].sort_values(["attr_type", "attr_value"], ignore_index=True)


def build_article_attribute_edges(clean_articles: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        clean_articles.columns.tolist(),
        ["article_id", *ATTRIBUTE_COLUMNS],
        source_name="articles_clean.csv",
    )

    edges: list[pd.DataFrame] = []
    for attr_type in ATTRIBUTE_COLUMNS:
        edge_frame = clean_articles.loc[:, ["article_id", attr_type]].copy()
        edge_frame = edge_frame.rename(columns={attr_type: "attr_value"})
        edge_frame["article_id"] = edge_frame["article_id"].astype("string")
        edge_frame["attr_value"] = edge_frame["attr_value"].astype("string")
        edge_frame["article_node_id"] = edge_frame["article_id"].map(make_article_node_id)
        edge_frame["attr_type"] = attr_type
        edge_frame["attr_id"] = edge_frame["attr_value"].map(
            lambda value: make_attr_id(attr_type, value)
        )
        edge_frame["edge_type"] = make_edge_type(attr_type)
        edge_frame["edge_weight"] = 1.0
        edges.append(edge_frame)

    return pd.concat(edges, ignore_index=True)[
        [
            "article_id",
            "article_node_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "edge_type",
            "edge_weight",
        ]
    ]


def build_attribute_hierarchy_edges(clean_articles: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        clean_articles.columns.tolist(),
        [column for relation in HIERARCHY_RELATIONS for column in relation[:2]],
        source_name="articles_clean.csv",
    )

    hierarchy_edges: list[pd.DataFrame] = []
    for parent_attr_type, child_attr_type, relation_type in HIERARCHY_RELATIONS:
        relation_counts = (
            clean_articles.groupby([parent_attr_type, child_attr_type], dropna=False)
            .size()
            .reset_index(name="edge_weight")
        )
        relation_counts["parent_attr_type"] = parent_attr_type
        relation_counts["child_attr_type"] = child_attr_type
        relation_counts["parent_attr_id"] = relation_counts[parent_attr_type].map(
            lambda value: make_attr_id(parent_attr_type, str(value))
        )
        relation_counts["child_attr_id"] = relation_counts[child_attr_type].map(
            lambda value: make_attr_id(child_attr_type, str(value))
        )
        relation_counts["relation_type"] = relation_type
        hierarchy_edges.append(relation_counts)

    return pd.concat(hierarchy_edges, ignore_index=True)[
        [
            "parent_attr_id",
            "child_attr_id",
            "parent_attr_type",
            "child_attr_type",
            "relation_type",
            "edge_weight",
        ]
    ].sort_values(
        ["parent_attr_type", "child_attr_type", "parent_attr_id", "child_attr_id"],
        ignore_index=True,
    )
```

- [ ] **Step 4: 运行属性图测试，确认通过**

Run:

```bash
PYTHONPATH=src uv run python -m unittest tests.test_attribute_graph -v
```

Expected:

```text
Ran 4 tests

OK
```

- [ ] **Step 5: 运行全部测试，确认通过**

Run:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

Expected:

```text
Ran 10 tests

OK
```

- [ ] **Step 6: 提交**

```bash
git add src/fashion_trend/articles.py tests/test_attribute_graph.py
git commit -m "feat: build attribute graph tables"
```

## Task 4: 属性图 CSV 写出与 CLI 脚本

**Files:**
- Modify: `src/fashion_trend/articles.py`
- Modify: `src/fashion_trend/config.py`
- Create: `src/04_build_attribute_graph.py`
- Modify: `tests/test_attribute_graph.py`

- [ ] **Step 1: 写失败测试，锁定图表文件写出**

追加到 `tests/test_attribute_graph.py`：

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from fashion_trend.articles import build_attribute_graph_files


class AttributeGraphFileTests(unittest.TestCase):
    def test_build_attribute_graph_files_writes_all_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            clean_articles_path = tmp_path / "articles_clean.csv"
            output_dir = tmp_path / "graph"
            sample_clean_articles().to_csv(clean_articles_path, index=False)

            output_counts = build_attribute_graph_files(
                clean_articles_path=clean_articles_path,
                graph_dir=output_dir,
            )

            self.assertEqual(output_counts["nodes_article"], 3)
            self.assertEqual(output_counts["edges_article_attribute"], 3 * len(ATTRIBUTE_COLUMNS))
            self.assertTrue((output_dir / "nodes_article.csv").exists())
            self.assertTrue((output_dir / "nodes_attribute.csv").exists())
            self.assertTrue((output_dir / "edges_article_attribute.csv").exists())
            self.assertTrue((output_dir / "edges_attribute_hierarchy.csv").exists())

            nodes_attribute = pd.read_csv(output_dir / "nodes_attribute.csv")
            edges = pd.read_csv(output_dir / "edges_article_attribute.csv")
            self.assertTrue(set(edges["attr_id"]).issubset(set(nodes_attribute["attr_id"])))

    def test_build_attribute_graph_files_fails_when_clean_input_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "商品 clean 文件不存在"):
                build_attribute_graph_files(
                    clean_articles_path=tmp_path / "articles_clean.csv",
                    graph_dir=tmp_path / "graph",
                )
```

- [ ] **Step 2: 运行测试，确认因为函数缺失而失败**

Run:

```bash
PYTHONPATH=src uv run python -m unittest tests.test_attribute_graph -v
```

Expected:

```text
ImportError: cannot import name 'build_attribute_graph_files'
```

- [ ] **Step 3: 实现图表写出和引用校验**

追加到 `src/fashion_trend/articles.py`：

```python
GRAPH_OUTPUT_FILENAMES: dict[str, str] = {
    "nodes_article": "nodes_article.csv",
    "nodes_attribute": "nodes_attribute.csv",
    "edges_article_attribute": "edges_article_attribute.csv",
    "edges_attribute_hierarchy": "edges_attribute_hierarchy.csv",
}


def read_clean_articles(clean_articles_path: Path) -> pd.DataFrame:
    if not clean_articles_path.exists():
        raise FileNotFoundError(f"商品 clean 文件不存在: {clean_articles_path}")
    return pd.read_csv(
        clean_articles_path,
        dtype={
            "article_id": "string",
            "product_code": "string",
        },
    )


def validate_graph_references(
    nodes_article: pd.DataFrame,
    nodes_attribute: pd.DataFrame,
    edges_article_attribute: pd.DataFrame,
    edges_attribute_hierarchy: pd.DataFrame,
) -> None:
    article_node_ids = set(nodes_article["article_node_id"])
    attr_ids = set(nodes_attribute["attr_id"])

    missing_article_nodes = set(edges_article_attribute["article_node_id"]) - article_node_ids
    if missing_article_nodes:
        raise RuntimeError("商品-属性边引用了不存在的商品节点。")

    missing_attr_nodes = set(edges_article_attribute["attr_id"]) - attr_ids
    if missing_attr_nodes:
        raise RuntimeError("商品-属性边引用了不存在的属性节点。")

    missing_parent_nodes = set(edges_attribute_hierarchy["parent_attr_id"]) - attr_ids
    missing_child_nodes = set(edges_attribute_hierarchy["child_attr_id"]) - attr_ids
    if missing_parent_nodes or missing_child_nodes:
        raise RuntimeError("属性层级边引用了不存在的属性节点。")


def build_attribute_graph_frames(
    clean_articles: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    nodes_article = build_article_nodes(clean_articles)
    nodes_attribute = build_attribute_nodes(clean_articles)
    edges_article_attribute = build_article_attribute_edges(clean_articles)
    edges_attribute_hierarchy = build_attribute_hierarchy_edges(clean_articles)
    validate_graph_references(
        nodes_article,
        nodes_attribute,
        edges_article_attribute,
        edges_attribute_hierarchy,
    )
    return {
        "nodes_article": nodes_article,
        "nodes_attribute": nodes_attribute,
        "edges_article_attribute": edges_article_attribute,
        "edges_attribute_hierarchy": edges_attribute_hierarchy,
    }


def build_attribute_graph_files(
    clean_articles_path: Path,
    graph_dir: Path,
) -> dict[str, int]:
    clean_articles = read_clean_articles(clean_articles_path)
    graph_frames = build_attribute_graph_frames(clean_articles)
    graph_dir.mkdir(parents=True, exist_ok=True)

    output_counts: dict[str, int] = {}
    for graph_name, graph_frame in graph_frames.items():
        output_path = graph_dir / GRAPH_OUTPUT_FILENAMES[graph_name]
        write_csv_atomically(graph_frame, output_path)
        output_counts[graph_name] = len(graph_frame)

    return output_counts
```

- [ ] **Step 4: 增加 graph 路径配置和 CLI 脚本**

修改 `src/fashion_trend/config.py`，增加 graph 路径：

```python
PATH = {
    # ---------------- Raw H&M data ----------------
    "raw_transactions": RAW_HM_DIR / "transactions_train.csv",
    "raw_articles": RAW_HM_DIR / "articles.csv",
    "raw_customers": RAW_HM_DIR / "customers.csv",
    # ---------------- Interim data ----------------
    "interim_transactions_weekly": INTERIM_DIR / "transactions_train_weekly.parquet",
    "interim_articles_clean_mvp": INTERIM_DIR / "articles_clean_mvp.csv",
    "interim_articles_clean": INTERIM_DIR / "articles_clean.csv",
    # ---------------- Processed graph data ----------------
    "graph_nodes_article": GRAPH_DIR / "nodes_article.csv",
    "graph_nodes_attribute": GRAPH_DIR / "nodes_attribute.csv",
    "graph_edges_article_attribute": GRAPH_DIR / "edges_article_attribute.csv",
    "graph_edges_attribute_hierarchy": GRAPH_DIR / "edges_attribute_hierarchy.csv",
}
```

创建 `src/04_build_attribute_graph.py`：

```python
from __future__ import annotations

from fashion_trend import log
from fashion_trend.articles import build_attribute_graph_files
from fashion_trend.config import GRAPH_DIR, PATH

LOG_SOURCE = "attribute-graph"


def main() -> int:
    try:
        output_counts = build_attribute_graph_files(
            clean_articles_path=PATH["interim_articles_clean"],
            graph_dir=GRAPH_DIR,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    for output_name, row_count in output_counts.items():
        log.info(f"{output_name}: {row_count:,} 行", source=LOG_SOURCE)
    log.info(f"属性图输出目录: {GRAPH_DIR}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行测试，确认通过**

Run:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

Expected:

```text
Ran 12 tests

OK
```

- [ ] **Step 6: 提交**

```bash
git add src/fashion_trend/articles.py src/fashion_trend/config.py src/04_build_attribute_graph.py tests/test_attribute_graph.py
git commit -m "feat: write attribute graph outputs"
```

## Task 5: 真实数据冒烟验证

**Files:**
- 本任务不创建源码文件。
- `data/interim/` 和 `data/processed/graph/` 下的生成结果是本地数据产物。

- [ ] **Step 1: 运行 articles 清洗脚本**

Run:

```bash
uv run python src/03_clean_articles.py
```

Expected:

```text
[clean-articles] 已写出商品中间表行数: 105,542
[clean-articles] MVP 输出文件: /Users/ghstlnx/Workspace/Fashion/data/interim/articles_clean_mvp.csv
[clean-articles] 稳妥版输出文件: /Users/ghstlnx/Workspace/Fashion/data/interim/articles_clean.csv
```

- [ ] **Step 2: 校验中间表行数和列数**

Run:

```bash
uv run python -c "import pandas as pd; m=pd.read_csv('data/interim/articles_clean_mvp.csv', dtype={'article_id':'string'}); c=pd.read_csv('data/interim/articles_clean.csv', dtype={'article_id':'string'}); print(len(m), len(c)); print(len(m.columns), len(c.columns)); print(m['article_id'].head().tolist())"
```

Expected:

```text
105542 105542
8 13
['0108775015', '0108775044', '0108775051', '0110065001', '0110065002']
```

- [ ] **Step 3: 运行属性图构建脚本**

Run:

```bash
uv run python src/04_build_attribute_graph.py
```

Expected:

```text
[attribute-graph] nodes_article: 105,542 行
[attribute-graph] nodes_attribute: 592 行
[attribute-graph] edges_article_attribute: 1,055,420 行
[attribute-graph] edges_attribute_hierarchy: 658 行
[attribute-graph] 属性图输出目录: /Users/ghstlnx/Workspace/Fashion/data/processed/graph
```

- [ ] **Step 4: 校验图表引用完整性**

Run:

```bash
uv run python -c "import pandas as pd; na=pd.read_csv('data/processed/graph/nodes_article.csv'); nt=pd.read_csv('data/processed/graph/nodes_attribute.csv'); ea=pd.read_csv('data/processed/graph/edges_article_attribute.csv'); eh=pd.read_csv('data/processed/graph/edges_attribute_hierarchy.csv'); print(len(na), len(nt), len(ea), len(eh)); print(set(ea['article_node_id']).issubset(set(na['article_node_id']))); print(set(ea['attr_id']).issubset(set(nt['attr_id']))); print(set(eh['parent_attr_id']).issubset(set(nt['attr_id'])) and set(eh['child_attr_id']).issubset(set(nt['attr_id'])))"
```

Expected:

```text
105542 592 1055420 658
True
True
True
```

- [ ] **Step 5: 确认工作区状态**

Run:

```bash
git status --short
```

Expected:

```text
```

如果生成的 `data/` 文件出现在 `git status` 中，不要提交它们，除非仓库明确要求跟踪数据产物。

## 自查

- 规格覆盖：计划覆盖了 `articles_clean_mvp.csv`、`articles_clean.csv`、4 张 graph CSV、清洗校验、图边引用校验和真实数据冒烟验证。
- 占位符检查：本计划没有占位实现、未定义函数名或后续补齐说明。
- 类型一致性：所有测试和实现步骤使用同一组常量名与函数名：`MVP_ARTICLE_COLUMNS`、`CLEAN_ARTICLE_COLUMNS`、`ATTRIBUTE_COLUMNS`、`build_clean_article_frames`、`clean_articles_file`、`build_attribute_graph_files`。
