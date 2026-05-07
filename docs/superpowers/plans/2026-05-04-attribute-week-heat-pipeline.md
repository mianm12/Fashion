# 商品周销量与属性周热度 Implementation Plan

> Note: This historical plan predates the domain-driven module migration. Current code must use concrete module imports; `fashion_trend.trend` is only a package marker.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按开发顺序先生成 `article_week_sales.csv`，再基于它和商品-属性边生成 `attribute_week_heat.csv`。

**Architecture:** 新增 `src/fashion_trend/trend.py` 承载可测试的趋势聚合逻辑，两个顶层脚本只负责按清楚顺序编排路径、读取、校验、聚合、写出和日志。`05_compute_article_week_sales.py` 只产出商品周销量，`06_compute_attribute_week_heat.py` 只消费商品周销量和属性边并产出属性热度，两个阶段可以独立重跑。

**Tech Stack:** Python 3.10-3.12，`pandas`，`numpy`，`pyarrow`，标准库 `csv` / `pathlib` / `unittest` / `tempfile`，现有 `fashion_trend.config.PATH` 和 `fashion_trend.log`。

---

## 文件结构

- Create: `src/fashion_trend/trend.py`
  - 负责趋势聚合相关字段常量、输入读取、DataFrame 校验、商品周销量聚合、属性周热度聚合、CSV 写出。
- Create: `src/05_compute_article_week_sales.py`
  - 顶层阶段脚本，读取 `PATH["interim_transactions_weekly"]`，写出 `PATH["trend_article_week_sales"]`。
- Create: `src/06_compute_attribute_week_heat.py`
  - 顶层阶段脚本，读取 `PATH["trend_article_week_sales"]` 和 `PATH["graph_edges_article_attribute"]`，写出 `PATH["trend_attribute_week_heat"]`。
- Modify: `src/fashion_trend/config.py`
  - 增加 `TREND_DIR` 和两条 trend 输出路径。
- Create: `tests/test_trend.py`
  - 覆盖商品周销量、属性周热度、输入校验、CSV 写出。
- Modify: `README.md`
  - 增加阶段 4/5 的运行命令和输出文件说明，和新增脚本编号保持一致。

## Task 1: 商品周销量 DataFrame 逻辑

**Files:**
- Create: `src/fashion_trend/trend.py`
- Create: `tests/test_trend.py`

- [ ] **Step 1: 写失败测试，锁定商品周销量聚合行为**

Create `tests/test_trend.py`:

```python
from __future__ import annotations

import math
import unittest

import pandas as pd

from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    validate_article_week_sales,
)
from fashion_trend.trend.schema import ARTICLE_WEEK_SALES_COLUMNS


def sample_weekly_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": [0, 0, 0, 1],
            "article_id": ["0108775015", "0108775015", "0110065001", "0108775015"],
            "customer_id": ["customer_1", "customer_2", "customer_1", "customer_1"],
            "price": [0.10, 0.20, 0.30, 0.40],
        }
    )


class ArticleWeekSalesFrameTests(unittest.TestCase):
    def test_build_article_week_sales_frame_aggregates_sales_by_week_and_article(
        self,
    ) -> None:
        sales = build_article_week_sales_frame(sample_weekly_transactions())

        self.assertEqual(sales.columns.tolist(), list(ARTICLE_WEEK_SALES_COLUMNS))
        self.assertEqual(
            sales[["week_id", "article_id", "sales_cnt", "sales_user_cnt"]].to_dict(
                "records"
            ),
            [
                {
                    "week_id": 0,
                    "article_id": "0108775015",
                    "sales_cnt": 2,
                    "sales_user_cnt": 2,
                },
                {
                    "week_id": 0,
                    "article_id": "0110065001",
                    "sales_cnt": 1,
                    "sales_user_cnt": 1,
                },
                {
                    "week_id": 1,
                    "article_id": "0108775015",
                    "sales_cnt": 1,
                    "sales_user_cnt": 1,
                },
            ],
        )
        self.assertTrue(math.isclose(float(sales.loc[0, "sales_amount"]), 0.30))
        self.assertTrue(math.isclose(float(sales.loc[1, "sales_amount"]), 0.30))
        self.assertTrue(math.isclose(float(sales.loc[2, "sales_amount"]), 0.40))

    def test_build_article_week_sales_frame_preserves_article_id_as_string(self) -> None:
        sales = build_article_week_sales_frame(sample_weekly_transactions())

        self.assertEqual(sales["article_id"].dtype.name, "string")
        self.assertEqual(sales.loc[0, "article_id"], "0108775015")

    def test_build_article_week_sales_frame_rejects_missing_required_values(self) -> None:
        transactions = sample_weekly_transactions()
        transactions.loc[0, "customer_id"] = pd.NA

        with self.assertRaisesRegex(ValueError, "customer_id"):
            build_article_week_sales_frame(transactions)

    def test_build_article_week_sales_frame_rejects_negative_price(self) -> None:
        transactions = sample_weekly_transactions()
        transactions.loc[0, "price"] = -0.01

        with self.assertRaisesRegex(ValueError, "price"):
            build_article_week_sales_frame(transactions)

    def test_validate_article_week_sales_rejects_duplicate_week_article(self) -> None:
        sales = pd.DataFrame(
            {
                "week_id": [0, 0],
                "article_id": ["0108775015", "0108775015"],
                "sales_cnt": [1, 1],
                "sales_user_cnt": [1, 1],
                "sales_amount": [0.10, 0.20],
            }
        )

        with self.assertRaisesRegex(ValueError, "week_id, article_id"):
            validate_article_week_sales(sales)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认因为模块缺失而失败**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend -v
```

Expected:

```text
ModuleNotFoundError: No module named 'fashion_trend.trend'
```

- [ ] **Step 3: 实现商品周销量最小逻辑**

Create `src/fashion_trend/trend.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

WEEKLY_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "customer_id",
    "price",
)

ARTICLE_WEEK_SALES_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "sales_cnt",
    "sales_user_cnt",
    "sales_amount",
)

ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS: tuple[str, ...] = (
    "article_id",
    "attr_id",
    "attr_type",
    "attr_value",
)

ATTRIBUTE_WEEK_HEAT_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "heat_cnt",
    "type_total_heat",
    "heat_share",
    "log_heat",
    "rank_in_type",
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
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = [
        column for column in required_columns if int(dataframe[column].isna().sum()) > 0
    ]
    if missing_columns:
        raise ValueError(f"{source_name} 存在缺失值字段: " + ", ".join(missing_columns))


def validate_unique_key(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str],
    source_name: str,
) -> None:
    duplicate_mask = dataframe.duplicated(subset=list(key_columns), keep=False)
    if duplicate_mask.any():
        raise ValueError(f"{source_name} 存在重复字段值: " + ", ".join(key_columns))


def validate_non_negative_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    invalid_columns = [column for column in columns if (dataframe[column] < 0).any()]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在负值字段: " + ", ".join(invalid_columns))


def validate_positive_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    invalid_columns = [column for column in columns if (dataframe[column] <= 0).any()]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在非正值字段: " + ", ".join(invalid_columns))


def build_article_week_sales_frame(weekly_transactions: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        weekly_transactions.columns.tolist(),
        WEEKLY_TRANSACTION_COLUMNS,
        source_name="周级交易表",
    )
    validate_no_missing_values(
        weekly_transactions,
        WEEKLY_TRANSACTION_COLUMNS,
        source_name="周级交易表",
    )
    validate_non_negative_values(
        weekly_transactions,
        ["price"],
        source_name="周级交易表",
    )

    normalized_transactions = weekly_transactions.loc[
        :, list(WEEKLY_TRANSACTION_COLUMNS)
    ].copy()
    normalized_transactions["article_id"] = normalized_transactions["article_id"].astype(
        "string"
    )

    sales = (
        normalized_transactions.groupby(["week_id", "article_id"], as_index=False)
        .agg(
            sales_cnt=("article_id", "size"),
            sales_user_cnt=("customer_id", "nunique"),
            sales_amount=("price", "sum"),
        )
        .sort_values(["week_id", "article_id"], ignore_index=True)
    )
    sales["sales_cnt"] = sales["sales_cnt"].astype("int64")
    sales["sales_user_cnt"] = sales["sales_user_cnt"].astype("int64")

    return sales.loc[:, list(ARTICLE_WEEK_SALES_COLUMNS)]


def validate_article_week_sales(article_week_sales: pd.DataFrame) -> None:
    validate_required_columns(
        article_week_sales.columns.tolist(),
        ARTICLE_WEEK_SALES_COLUMNS,
        source_name="商品周销量表",
    )
    validate_no_missing_values(
        article_week_sales,
        ARTICLE_WEEK_SALES_COLUMNS,
        source_name="商品周销量表",
    )
    validate_unique_key(
        article_week_sales,
        ["week_id", "article_id"],
        source_name="商品周销量表",
    )
    validate_positive_values(
        article_week_sales,
        ["sales_cnt", "sales_user_cnt"],
        source_name="商品周销量表",
    )
    validate_non_negative_values(
        article_week_sales,
        ["sales_amount"],
        source_name="商品周销量表",
    )
```

- [ ] **Step 4: 运行商品周销量测试，确认通过**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.ArticleWeekSalesFrameTests -v
```

Expected:

```text
Ran 5 tests

OK
```

- [ ] **Step 5: 提交商品周销量 DataFrame 逻辑**

Run:

```bash
git add src/fashion_trend/trend.py tests/test_trend.py
git commit -m "feat(trend): aggregate article week sales"
```

Expected:

```text
[codex/preprocess-data <hash>] feat(trend): aggregate article week sales
```

## Task 2: 商品周销量文件写出与顶层脚本

**Files:**
- Modify: `src/fashion_trend/trend.py`
- Modify: `src/fashion_trend/config.py`
- Create: `src/05_compute_article_week_sales.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: 追加失败测试，锁定 CSV 写出格式**

Append to `tests/test_trend.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from fashion_trend.foundation.io import write_csv_atomic


class TrendCsvWriteTests(unittest.TestCase):
    def test_write_trend_csv_quotes_all_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "attribute_week_heat.csv"
            dataframe = pd.DataFrame(
                {
                    "week_id": [0],
                    "attr_id": ["garment_group_name::Under-, Nightwear"],
                    "attr_type": ["garment_group_name"],
                    "attr_value": ["Under-, Nightwear"],
                    "heat_cnt": [2],
                    "type_total_heat": [2],
                    "heat_share": [1.0],
                    "log_heat": [1.0986122886681098],
                    "rank_in_type": [1],
                }
            )

            write_csv_atomic(dataframe, output_path)

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                lines[0],
                '"week_id","attr_id","attr_type","attr_value","heat_cnt","type_total_heat","heat_share","log_heat","rank_in_type"',
            )
            self.assertIn('"garment_group_name::Under-, Nightwear"', lines[1])
            self.assertIn('"Under-, Nightwear"', lines[1])
            self.assertFalse(output_path.with_suffix(".csv.tmp").exists())
```

- [ ] **Step 2: 运行新增测试，确认因为函数缺失而失败**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.TrendCsvWriteTests -v
```

Expected:

```text
ImportError: cannot import name 'write_trend_csv'
```

- [ ] **Step 3: 实现 CSV 写出函数**

Append to `src/fashion_trend/trend.py`:

```python
def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_trend_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_csv(tmp_output_path, index=False, quoting=csv.QUOTE_ALL)
        tmp_output_path.replace(output_path)
    except Exception:
        remove_file_if_exists(tmp_output_path)
        raise
```

- [ ] **Step 4: 修改配置，增加 trend 输出路径**

Update `src/fashion_trend/config.py` so the processed path section contains:

```python
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
TREND_DIR = PROCESSED_DIR / "trend"
```

Update `PATH` so it includes these entries after the graph paths:

```python
    # ---------------- Processed trend data ----------------
    "trend_article_week_sales": TREND_DIR / "article_week_sales.csv",
    "trend_attribute_week_heat": TREND_DIR / "attribute_week_heat.csv",
```

- [ ] **Step 5: 创建商品周销量顶层脚本**

Create `src/05_compute_article_week_sales.py`:

```python
from __future__ import annotations

from fashion_trend import log
from fashion_trend.config import PATH
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.transactions.weekly import read_weekly_transactions
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    validate_article_week_sales,
)

LOG_SOURCE = "article-week-sales"


def compute_article_week_sales() -> dict[str, int]:
    log.info(f"输入周级交易表: {PATH['interim_transactions_weekly']}", source=LOG_SOURCE)
    weekly_transactions = read_weekly_transactions(
        PATH["interim_transactions_weekly"]
    )
    article_week_sales = build_article_week_sales_frame(weekly_transactions)
    validate_article_week_sales(article_week_sales)
    write_csv_atomic(article_week_sales, PATH["trend_article_week_sales"])

    return {
        "rows": len(article_week_sales),
        "weeks": int(article_week_sales["week_id"].nunique()),
        "articles": int(article_week_sales["article_id"].nunique()),
    }


def main() -> int:
    try:
        stats = compute_article_week_sales()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"商品周销量行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖商品数: {stats['articles']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {PATH['trend_article_week_sales']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 实现周级交易读取函数**

Current ownership places this reader in `src/fashion_trend/transactions/weekly.py`, not in the trend package:

```python
def read_weekly_transactions(weekly_transactions_path: Path) -> pd.DataFrame:
    if not weekly_transactions_path.exists():
        raise FileNotFoundError(f"周级交易表不存在: {weekly_transactions_path}")

    try:
        return pd.read_parquet(
            weekly_transactions_path,
            columns=list(WEEKLY_TRANSACTION_COLUMNS),
        )
    except ValueError as exc:
        raise ValueError(
            f"周级交易表缺少必要字段: {weekly_transactions_path}"
        ) from exc
```

- [ ] **Step 7: 运行相关测试和脚本语法检查**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.ArticleWeekSalesFrameTests tests.test_trend.TrendCsvWriteTests -v
PYTHONPATH=src .venv/bin/python -m py_compile src/05_compute_article_week_sales.py src/fashion_trend/trend.py src/fashion_trend/config.py
```

Expected:

```text
Ran 6 tests

OK
```

`py_compile` exits with code 0 and no output.

- [ ] **Step 8: 提交商品周销量脚本与配置**

Run:

```bash
git add src/fashion_trend/trend.py src/fashion_trend/config.py src/05_compute_article_week_sales.py tests/test_trend.py
git commit -m "feat(trend): write article week sales file"
```

Expected:

```text
[codex/preprocess-data <hash>] feat(trend): write article week sales file
```

## Task 3: 属性周热度 DataFrame 逻辑

**Files:**
- Modify: `src/fashion_trend/trend.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: 追加失败测试，锁定属性周热度计算**

Append to `tests/test_trend.py`:

```python
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_article_attribute_edges_for_heat,
)
from fashion_trend.trend.schema import ATTRIBUTE_WEEK_HEAT_COLUMNS


def sample_article_week_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": [0, 0, 1],
            "article_id": ["0108775015", "0110065001", "0108775015"],
            "sales_cnt": [2, 1, 1],
            "sales_user_cnt": [2, 1, 1],
            "sales_amount": [0.30, 0.30, 0.40],
        }
    )


def sample_article_attribute_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [
                "0108775015",
                "0108775015",
                "0110065001",
                "0110065001",
            ],
            "article_node_id": [
                "article_0108775015",
                "article_0108775015",
                "article_0110065001",
                "article_0110065001",
            ],
            "attr_id": [
                "colour_group_name::Black",
                "product_type_name::Vest top",
                "colour_group_name::White",
                "product_type_name::Bra",
            ],
            "attr_type": [
                "colour_group_name",
                "product_type_name",
                "colour_group_name",
                "product_type_name",
            ],
            "attr_value": ["Black", "Vest top", "White", "Bra"],
            "edge_type": [
                "has_colour_group",
                "has_product_type",
                "has_colour_group",
                "has_product_type",
            ],
            "edge_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )


class AttributeWeekHeatFrameTests(unittest.TestCase):
    def test_build_attribute_week_heat_frame_calculates_heat_metrics(self) -> None:
        heat = build_attribute_week_heat_frame(
            sample_article_week_sales(),
            sample_article_attribute_edges(),
        )

        self.assertEqual(heat.columns.tolist(), list(ATTRIBUTE_WEEK_HEAT_COLUMNS))

        week0_colour = heat[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ].sort_values("rank_in_type")
        self.assertEqual(
            week0_colour[
                ["attr_id", "heat_cnt", "type_total_heat", "rank_in_type"]
            ].to_dict("records"),
            [
                {
                    "attr_id": "colour_group_name::Black",
                    "heat_cnt": 2,
                    "type_total_heat": 3,
                    "rank_in_type": 1,
                },
                {
                    "attr_id": "colour_group_name::White",
                    "heat_cnt": 1,
                    "type_total_heat": 3,
                    "rank_in_type": 2,
                },
            ],
        )
        self.assertTrue(math.isclose(float(week0_colour.iloc[0]["heat_share"]), 2 / 3))
        self.assertTrue(math.isclose(float(week0_colour.iloc[1]["heat_share"]), 1 / 3))
        self.assertTrue(
            math.isclose(float(week0_colour.iloc[0]["log_heat"]), math.log1p(2))
        )

    def test_build_attribute_week_heat_frame_rejects_unmapped_sales_articles(self) -> None:
        sales = sample_article_week_sales()
        sales.loc[len(sales)] = [0, "0999999999", 1, 1, 0.10]

        with self.assertRaisesRegex(ValueError, "无法映射到属性边"):
            build_attribute_week_heat_frame(sales, sample_article_attribute_edges())

    def test_validate_article_attribute_edges_for_heat_rejects_duplicate_edges(self) -> None:
        edges = sample_article_attribute_edges()
        edges.loc[len(edges)] = edges.loc[0]

        with self.assertRaisesRegex(ValueError, "article_id, attr_id"):
            validate_article_attribute_edges_for_heat(edges)
```

- [ ] **Step 2: 运行属性周热度测试，确认因为函数缺失而失败**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.AttributeWeekHeatFrameTests -v
```

Expected:

```text
ImportError: cannot import name 'build_attribute_week_heat_frame'
```

- [ ] **Step 3: 实现属性边校验和属性热度计算**

Append to `src/fashion_trend/trend.py`:

```python
def validate_article_attribute_edges_for_heat(
    article_attribute_edges: pd.DataFrame,
) -> None:
    validate_required_columns(
        article_attribute_edges.columns.tolist(),
        ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS,
        source_name="商品-属性边表",
    )
    validate_no_missing_values(
        article_attribute_edges,
        ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS,
        source_name="商品-属性边表",
    )
    validate_unique_key(
        article_attribute_edges,
        ["article_id", "attr_id"],
        source_name="商品-属性边表",
    )


def validate_all_sales_articles_have_attribute_edges(
    article_week_sales: pd.DataFrame,
    article_attribute_edges: pd.DataFrame,
) -> None:
    sales_article_ids = set(article_week_sales["article_id"].astype("string"))
    edge_article_ids = set(article_attribute_edges["article_id"].astype("string"))
    missing_article_ids = sorted(sales_article_ids - edge_article_ids)
    if missing_article_ids:
        examples = ", ".join(missing_article_ids[:5])
        raise ValueError(
            f"商品周销量表存在 {len(missing_article_ids)} 个 article_id "
            f"无法映射到属性边，例如: {examples}"
        )


def build_attribute_week_heat_frame(
    article_week_sales: pd.DataFrame,
    article_attribute_edges: pd.DataFrame,
) -> pd.DataFrame:
    validate_article_week_sales(article_week_sales)
    validate_article_attribute_edges_for_heat(article_attribute_edges)
    validate_all_sales_articles_have_attribute_edges(
        article_week_sales,
        article_attribute_edges,
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

    joined = normalized_sales.merge(
        normalized_edges,
        on="article_id",
        how="inner",
    )
    heat = (
        joined.groupby(["week_id", "attr_id", "attr_type", "attr_value"], as_index=False)[
            "sales_cnt"
        ]
        .sum()
        .rename(columns={"sales_cnt": "heat_cnt"})
    )
    heat["heat_cnt"] = heat["heat_cnt"].astype("int64")
    heat["type_total_heat"] = heat.groupby(["week_id", "attr_type"])[
        "heat_cnt"
    ].transform("sum")
    heat["type_total_heat"] = heat["type_total_heat"].astype("int64")
    heat["heat_share"] = heat["heat_cnt"] / heat["type_total_heat"]
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

- [ ] **Step 4: 实现属性热度输出校验**

Append to `src/fashion_trend/trend.py`:

```python
def validate_attribute_week_heat(attribute_week_heat: pd.DataFrame) -> None:
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
    validate_positive_values(
        attribute_week_heat,
        ["heat_cnt", "type_total_heat", "heat_share", "rank_in_type"],
        source_name="属性周热度表",
    )

    if (attribute_week_heat["type_total_heat"] < attribute_week_heat["heat_cnt"]).any():
        raise ValueError("属性周热度表存在 type_total_heat 小于 heat_cnt 的记录。")
    if (attribute_week_heat["heat_share"] > 1).any():
        raise ValueError("属性周热度表存在 heat_share 大于 1 的记录。")

    share_totals = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "heat_share"
    ].sum()
    invalid_share_totals = share_totals[~np.isclose(share_totals, 1.0, atol=1e-9)]
    if not invalid_share_totals.empty:
        raise ValueError("属性周热度表存在 week_id + attr_type 占比和不等于 1 的分组。")

    rank_counts = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "rank_in_type"
    ].nunique()
    row_counts = attribute_week_heat.groupby(["week_id", "attr_type"]).size()
    if not rank_counts.equals(row_counts):
        raise ValueError("属性周热度表存在重复 rank_in_type。")
    min_ranks = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "rank_in_type"
    ].min()
    if (min_ranks != 1).any():
        raise ValueError("属性周热度表存在 rank_in_type 未从 1 开始的分组。")
```

- [ ] **Step 5: 运行属性热度测试，确认通过**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend.AttributeWeekHeatFrameTests -v
```

Expected:

```text
Ran 3 tests

OK
```

- [ ] **Step 6: 提交属性周热度 DataFrame 逻辑**

Run:

```bash
git add src/fashion_trend/trend.py tests/test_trend.py
git commit -m "feat(trend): compute attribute week heat"
```

Expected:

```text
[codex/preprocess-data <hash>] feat(trend): compute attribute week heat
```

## Task 4: 属性周热度文件读取与顶层脚本

**Files:**
- Modify: `src/fashion_trend/trend.py`
- Create: `src/06_compute_attribute_week_heat.py`

- [ ] **Step 1: 实现商品周销量和商品-属性边读取函数**

Current ownership keeps article sales reading in `trend/article_sales.py` and moves catalog graph readers to `catalog/graph.py`:

```python
def read_article_week_sales(article_week_sales_path: Path) -> pd.DataFrame:
    if not article_week_sales_path.exists():
        raise FileNotFoundError(f"商品周销量表不存在: {article_week_sales_path}")

    return pd.read_csv(
        article_week_sales_path,
        usecols=list(ARTICLE_WEEK_SALES_COLUMNS),
        dtype={"article_id": "string"},
    )


def read_article_attribute_edges(article_attribute_edges_path: Path) -> pd.DataFrame:
    if not article_attribute_edges_path.exists():
        raise FileNotFoundError(f"商品-属性边表不存在: {article_attribute_edges_path}")

    return pd.read_csv(
        article_attribute_edges_path,
        usecols=list(ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS),
        dtype={
            "article_id": "string",
            "attr_id": "string",
            "attr_type": "string",
            "attr_value": "string",
        },
    )
```

- [ ] **Step 2: 创建属性周热度顶层脚本**

Create `src/06_compute_attribute_week_heat.py`:

```python
from __future__ import annotations

from fashion_trend import log
from fashion_trend.config import PATH
from fashion_trend.catalog.graph import read_article_attribute_edges, read_attribute_nodes
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.trend.article_sales import (
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_all_sales_articles_have_attribute_edges,
    validate_article_attribute_edges_for_heat,
    validate_attribute_edge_node_metadata_consistency,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)

LOG_SOURCE = "attribute-week-heat"


def compute_attribute_week_heat() -> dict[str, int]:
    log.info(f"输入商品周销量表: {PATH['trend_article_week_sales']}", source=LOG_SOURCE)
    article_week_sales = read_article_week_sales(PATH["trend_article_week_sales"])
    validate_article_week_sales(article_week_sales)

    log.info(
        f"输入商品-属性边表: {PATH['graph_edges_article_attribute']}",
        source=LOG_SOURCE,
    )
    article_attribute_edges = read_article_attribute_edges(
        PATH["graph_edges_article_attribute"]
    )
    validate_article_attribute_edges_for_heat(article_attribute_edges)

    attribute_nodes = read_attribute_nodes(PATH["graph_nodes_attribute"])
    validate_attribute_nodes_for_heat(attribute_nodes)
    validate_all_sales_articles_have_attribute_edges(
        article_week_sales,
        article_attribute_edges,
    )
    validate_attribute_edge_node_metadata_consistency(
        article_attribute_edges,
        attribute_nodes,
    )

    attribute_week_heat = build_attribute_week_heat_frame(
        article_week_sales,
        article_attribute_edges,
        attribute_nodes,
    )
    validate_attribute_week_heat(attribute_week_heat)
    write_csv_atomic(attribute_week_heat, PATH["trend_attribute_week_heat"])

    return {
        "rows": len(attribute_week_heat),
        "weeks": int(attribute_week_heat["week_id"].nunique()),
        "attr_types": int(attribute_week_heat["attr_type"].nunique()),
        "attributes": int(attribute_week_heat["attr_id"].nunique()),
    }


def main() -> int:
    try:
        stats = compute_attribute_week_heat()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"属性周热度行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性类型数: {stats['attr_types']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {stats['attributes']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {PATH['trend_attribute_week_heat']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 运行全量单元测试和脚本语法检查**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend -v
PYTHONPATH=src .venv/bin/python -m py_compile src/05_compute_article_week_sales.py src/06_compute_attribute_week_heat.py src/fashion_trend/trend.py src/fashion_trend/config.py
```

Expected:

```text
Ran 9 tests

OK
```

`py_compile` exits with code 0 and no output.

- [ ] **Step 4: 提交属性热度脚本**

Run:

```bash
git add src/fashion_trend/trend.py src/06_compute_attribute_week_heat.py
git commit -m "feat(trend): write attribute week heat file"
```

Expected:

```text
[codex/preprocess-data <hash>] feat(trend): write attribute week heat file
```

## Task 5: README 阶段说明

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在 README 的数据预处理部分追加商品周销量说明**

Insert this section after the current articles / 属性图相关说明:

````markdown
### 3. article_week_sales.csv

商品周销量表基于 `data/interim/transactions_train_weekly.parquet` 生成，按 `week_id + article_id` 聚合每个商品在每周的购买次数、购买用户数和销售额。

输出文件:

```sh
data/processed/trend/article_week_sales.csv
```

运行命令:

```sh
PYTHONPATH=src .venv/bin/python src/05_compute_article_week_sales.py
```

输出字段:

| 字段 | 说明 |
| :--- | :--- |
| `week_id` | 周编号 |
| `article_id` | 商品 ID，保留前导 0 |
| `sales_cnt` | 该商品本周购买次数 |
| `sales_user_cnt` | 该商品本周购买用户数 |
| `sales_amount` | 该商品本周销售额 |
````

- [ ] **Step 2: 在 README 追加属性周热度说明**

Insert this section immediately after the article sales section:

````markdown
### 4. attribute_week_heat.csv

属性周热度表基于 `article_week_sales.csv` 和 `data/processed/graph/edges_article_attribute.csv` 生成，将商品周销量映射到商品关联的属性节点。

输出文件:

```sh
data/processed/trend/attribute_week_heat.csv
```

运行命令:

```sh
PYTHONPATH=src .venv/bin/python src/06_compute_attribute_week_heat.py
```

输出字段:

| 字段 | 说明 |
| :--- | :--- |
| `week_id` | 周编号 |
| `attr_id` | 属性 ID |
| `attr_type` | 属性类型 |
| `attr_value` | 属性取值 |
| `heat_cnt` | 属性原始热度 |
| `type_total_heat` | 同一属性类型在本周的总热度 |
| `heat_share` | 属性在同类型中的热度占比 |
| `log_heat` | `log1p(heat_cnt)` |
| `rank_in_type` | 同一周、同一属性类型内的热度排名 |
````

- [ ] **Step 3: 运行 README 相关检查**

Run:

```bash
rg -n "05_compute_article_week_sales|06_compute_attribute_week_heat|attribute_week_heat|article_week_sales" README.md
git diff --check
```

Expected:

```text
README.md:<line>:PYTHONPATH=src .venv/bin/python src/05_compute_article_week_sales.py
README.md:<line>:PYTHONPATH=src .venv/bin/python src/06_compute_attribute_week_heat.py
```

`git diff --check` exits with code 0 and no output.

- [ ] **Step 4: 提交 README 更新**

Run:

```bash
git add README.md
git commit -m "docs(trend): document attribute heat pipeline"
```

Expected:

```text
[codex/preprocess-data <hash>] docs(trend): document attribute heat pipeline
```

## Task 6: 全量验证与真实数据试跑

**Files:**
- No source edits expected.

- [ ] **Step 1: 运行全量测试**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

The output includes `tests.test_trend`, `tests.test_articles_clean`, and `tests.test_attribute_graph`.

- [ ] **Step 2: 运行商品周销量真实数据脚本**

Run:

```bash
PYTHONPATH=src .venv/bin/python src/05_compute_article_week_sales.py
```

Expected:

```text
[INFO] [article-week-sales] 输入周级交易表: /Users/ghstlnx/Workspace/Fashion/data/interim/transactions_train_weekly.parquet
[INFO] [article-week-sales] 商品周销量行数: <non-zero row count>
[INFO] [article-week-sales] 覆盖周数: <non-zero week count>
[INFO] [article-week-sales] 覆盖商品数: <non-zero article count>
[INFO] [article-week-sales] 输出文件: /Users/ghstlnx/Workspace/Fashion/data/processed/trend/article_week_sales.csv
```

- [ ] **Step 3: 运行属性周热度真实数据脚本**

Run:

```bash
PYTHONPATH=src .venv/bin/python src/06_compute_attribute_week_heat.py
```

Expected:

```text
[INFO] [attribute-week-heat] 输入商品周销量表: /Users/ghstlnx/Workspace/Fashion/data/processed/trend/article_week_sales.csv
[INFO] [attribute-week-heat] 输入商品-属性边表: /Users/ghstlnx/Workspace/Fashion/data/processed/graph/edges_article_attribute.csv
[INFO] [attribute-week-heat] 属性周热度行数: <non-zero row count>
[INFO] [attribute-week-heat] 覆盖周数: <non-zero week count>
[INFO] [attribute-week-heat] 覆盖属性类型数: 10
[INFO] [attribute-week-heat] 覆盖属性节点数: <non-zero attribute count>
[INFO] [attribute-week-heat] 输出文件: /Users/ghstlnx/Workspace/Fashion/data/processed/trend/attribute_week_heat.csv
```

- [ ] **Step 4: 抽查输出文件表头**

Run:

```bash
head -1 data/processed/trend/article_week_sales.csv
head -1 data/processed/trend/attribute_week_heat.csv
```

Expected:

```text
"week_id","article_id","sales_cnt","sales_user_cnt","sales_amount"
"week_id","attr_id","attr_type","attr_value","heat_cnt","type_total_heat","heat_share","log_heat","rank_in_type"
```

- [ ] **Step 5: 查看最终 diff**

Run:

```bash
git status --short
git diff --stat
```

Expected:

```text
 M README.md
 M src/fashion_trend/config.py
?? src/05_compute_article_week_sales.py
?? src/06_compute_attribute_week_heat.py
?? src/fashion_trend/trend.py
?? tests/test_trend.py
```

If all task commits were made as written, `git status --short` is empty instead.
