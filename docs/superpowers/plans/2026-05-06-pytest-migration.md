# pytest 迁移实施计划

> Note: This historical plan predates the domain-driven module migration. Current test examples must use concrete module imports; `fashion_trend.trend` is only a package marker.

> **给 agentic workers：** 必须使用子技能：推荐 `superpowers:subagent-driven-development`，也可以使用 `superpowers:executing-plans`，逐任务执行本计划。步骤使用 checkbox（`- [ ]`）语法追踪进度。

**目标：** 将测试套件迁移到 pytest，并按照项目数据流水线拆分过大的趋势测试文件，同时不改变生产代码行为。

**架构：** 先引入 pytest 作为正式测试入口，确认未迁移的 unittest 风格测试也能被 pytest 收集并通过；再迁移两个较小测试文件；最后按 README 和研究实施方案中的阶段拆分 `tests/test_trend.py`。只有跨多个新测试文件复用的样本 builder 才移动到共享位置；每个任务都以聚焦验证和一次 commit 收口。

**技术栈：** Python 3.10-3.12、uv、pytest、pandas、pyarrow、现有 `fashion_trend` 模块。

---

## 文件结构

目标测试目录结构：

```text
tests/
    __init__.py
    trend_samples.py
    test_articles_clean.py
    test_attribute_graph.py
    test_trend_article_sales.py
    test_trend_attribute_heat.py
    test_trend_targets.py
    test_trend_samples.py
    test_trend_splits.py
    test_trend_training.py
    test_trend_evaluation.py
```

文件职责边界：

| 文件 | 职责 |
| --- | --- |
| `pyproject.toml` | 声明 pytest dev 依赖，并配置测试发现与 `src` 导入路径 |
| `uv.lock` | 锁定 pytest 及其传递 dev 依赖 |
| `README.md` | 将正式验证命令切换为 `uv run pytest` |
| `tests/__init__.py` | 让共享测试 helper 可以通过 `tests.trend_samples` 显式导入 |
| `tests/trend_samples.py` | 放置多个拆分后的趋势测试文件复用的样本 builder |
| `tests/test_articles_clean.py` | articles 清洗、缺失值、重复 ID、双输出写出回滚 |
| `tests/test_attribute_graph.py` | 属性节点、商品-属性边、属性层级边、图文件写出与回滚 |
| `tests/test_trend_article_sales.py` | 商品周销量、周交易读取、商品周销量读取、趋势 CSV 写出 |
| `tests/test_trend_attribute_heat.py` | 商品属性边读取、属性节点校验、属性周热度完整面板与派生字段 |
| `tests/test_trend_targets.py` | 属性周趋势目标计算、增长公式和异常输入 |
| `tests/test_trend_samples.py` | 趋势样本 lag、移动窗口、图特征、目标合入与 stale target 防护 |
| `tests/test_trend_splits.py` | train/valid/test 时间切分、split 读取、metadata、JSON/Parquet 写出 |
| `tests/test_trend_training.py` | baseline trainer、registry、训练 runner、训练输出契约和训练 CLI |
| `tests/test_trend_evaluation.py` | 预测读取、评价输入校验、指标计算、payload、写出边界和评价 CLI |

当前 `tests/test_trend.py` 到目标文件的映射：

| 当前类 | 目标文件 |
| --- | --- |
| `ArticleWeekSalesFrameTests` | `tests/test_trend_article_sales.py` |
| `TrendCsvWriteTests` | `tests/test_trend_article_sales.py` |
| `AttributeWeekHeatFrameTests` | `tests/test_trend_attribute_heat.py` |
| `AttributeWeekTargetFrameTests` | `tests/test_trend_targets.py` |
| `TrendModelSamplesFrameTests` | `tests/test_trend_samples.py` |
| `TrendModelSplitFrameTests` | `tests/test_trend_splits.py` |
| `TrendModelSplitWriteTests` | `tests/test_trend_splits.py` |
| `LastWeekBaselineTests` | `tests/test_trend_training.py` |
| `TrendEvaluationTests` | `tests/test_trend_evaluation.py` |

---

### 任务 1：引入 pytest 测试入口和正式命令

**文件：**
- 修改：`pyproject.toml`
- 修改：`uv.lock`
- 修改：`README.md`

- [ ] **步骤 1：确认干净起点**

运行：

```sh
git status --short --branch
```

期望：当前分支为 `codex/pytest-migration`，并且没有未提交文件。

- [ ] **步骤 2：添加 pytest dev 依赖**

运行：

```sh
uv add --dev pytest
```

期望：`pyproject.toml` 的 `[dependency-groups].dev` 增加 `pytest`，`uv.lock` 增加 pytest 及 uv 解析出的传递依赖，例如 `iniconfig`、`packaging`、`pluggy`、`pygments`。

`pyproject.toml` 的 dev 依赖应类似：

```toml
[dependency-groups]
dev = [
    "isort>=8.0.1",
    "pytest>=8.0.0",
]
```

- [ ] **步骤 3：添加 pytest 配置**

在 `pyproject.toml` 中加入：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

保留已有的 `[tool.setuptools.packages.find]`，不要改动包发现配置。

- [ ] **步骤 4：用 pytest 跑未迁移的 unittest 风格测试**

运行：

```sh
uv run pytest
```

期望：pytest 能收集并通过当前 147 个测试。pytest 版本不同会导致 header 略有差异，但结果必须是全部通过。

- [ ] **步骤 5：更新 README 验证命令**

将 README 验证部分从：

````markdown
当前测试使用标准库 `unittest`，不依赖真实 H&M 数据：

```sh
uv run python -m unittest discover -s tests -v
```
````

改为：

````markdown
当前测试使用 `pytest`，不依赖真实 H&M 数据：

```sh
uv run pytest
```
````

保留下方已有的测试覆盖说明列表。

- [ ] **步骤 6：确认正式文档不再宣传 unittest discover**

运行：

```sh
rg -n "当前测试使用标准库|unittest discover|uv run python -m unittest discover" README.md docs/gpt-research/implementation-plan.md
```

期望：无匹配。

- [ ] **步骤 7：全量运行 pytest**

运行：

```sh
uv run pytest
```

期望：147 个测试全部通过。

- [ ] **步骤 8：检查 diff**

运行：

```sh
git diff -- pyproject.toml uv.lock README.md
```

期望：diff 只包含 pytest 依赖/配置、lockfile 解析结果、README 验证命令。

- [ ] **步骤 9：提交**

运行：

```sh
git add pyproject.toml uv.lock README.md
git commit -m "test: 引入 pytest 测试入口"
```

期望：commit 成功。

---

### 任务 2：将 articles 清洗测试迁移到 pytest idiom

**文件：**
- 修改：`tests/test_articles_clean.py`

- [ ] **步骤 1：更新 import**

替换：

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
```

为：

```python
from pathlib import Path

import pytest
```

保留已有的 `pandas` 和 `fashion_trend.articles` import。

- [ ] **步骤 2：转换测试类**

替换：

```python
class CleanArticleFrameTests(unittest.TestCase):
```

为：

```python
class TestCleanArticleFrame:
```

替换：

```python
class CleanArticleFileTests(unittest.TestCase):
```

为：

```python
class TestCleanArticleFile:
```

- [ ] **步骤 3：转换 frame 断言**

在 `TestCleanArticleFrame` 中将 unittest 断言改为裸 `assert`。示例：

```python
assert list(mvp_articles.columns) == list(MVP_ARTICLE_COLUMNS)
assert list(clean_articles.columns) == list(CLEAN_ARTICLE_COLUMNS)
assert len(mvp_articles) == 2
assert len(clean_articles) == 2
assert mvp_articles["article_id"].tolist() == ["0108775015", "0108775044"]
assert "detail_desc" not in mvp_articles.columns
assert "detail_desc" not in clean_articles.columns
```

异常断言使用：

```python
with pytest.raises(ValueError, match="缺少必要字段: product_type_name"):
    validate_required_columns(
        ["article_id", "product_group_name"],
        ["article_id", "product_group_name", "product_type_name"],
        source_name="测试 articles 表",
    )
```

- [ ] **步骤 4：将文件测试改为 `tmp_path`**

把当前使用 `TemporaryDirectory()` 的测试改成接收 `tmp_path: Path`。示例：

```python
def test_clean_articles_file_writes_mvp_and_clean_outputs(tmp_path: Path) -> None:
    raw_path = tmp_path / "articles.csv"
    mvp_output_path = tmp_path / "articles_clean_mvp.csv"
    clean_output_path = tmp_path / "articles_clean.csv"
    sample_raw_articles().to_csv(raw_path, index=False)

    row_count = clean_articles_file(
        raw_articles_path=raw_path,
        mvp_output_path=mvp_output_path,
        clean_output_path=clean_output_path,
    )

    assert row_count == 2
```

该文件中所有文件系统测试都使用同样模式。

- [ ] **步骤 5：确认没有 unittest 残留**

运行：

```sh
rg -n "unittest|TestCase|self\\.assert|TemporaryDirectory" tests/test_articles_clean.py
```

期望：无匹配。

- [ ] **步骤 6：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_articles_clean.py -q
```

期望：9 个测试通过。

- [ ] **步骤 7：运行全量 pytest**

运行：

```sh
uv run pytest
```

期望：全部测试通过。

- [ ] **步骤 8：检查 diff 并提交**

运行：

```sh
git diff -- tests/test_articles_clean.py
git add tests/test_articles_clean.py
git commit -m "test: 迁移 articles 清洗测试到 pytest"
```

期望：commit 成功。

---

### 任务 3：将属性图测试迁移到 pytest idiom

**文件：**
- 修改：`tests/test_attribute_graph.py`

- [ ] **步骤 1：更新 import**

替换：

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
```

为：

```python
from pathlib import Path

import pytest
```

保留已有的 pandas 和 `fashion_trend.articles` import。

- [ ] **步骤 2：转换测试类**

替换：

```python
class AttributeGraphBuilderTests(unittest.TestCase):
```

为：

```python
class TestAttributeGraphBuilder:
```

替换：

```python
class AttributeGraphFileTests(unittest.TestCase):
```

为：

```python
class TestAttributeGraphFile:
```

- [ ] **步骤 3：将 builder subTest 循环改成 parametrize**

将 `test_graph_builders_reject_missing_attribute_values` 中的 `self.subTest` 循环替换为：

```python
@pytest.mark.parametrize(
    "builder",
    [
        build_attribute_nodes,
        build_article_attribute_edges,
        build_attribute_hierarchy_edges,
    ],
)
def test_graph_builders_reject_missing_attribute_values(builder) -> None:
    clean_articles = sample_clean_articles()
    clean_articles.loc[0, "colour_group_name"] = pd.NA

    with pytest.raises(ValueError, match="colour_group_name"):
        builder(clean_articles)
```

- [ ] **步骤 4：将文件测试改为 `tmp_path`**

文件测试接收 `tmp_path: Path`。示例：

```python
def test_build_attribute_graph_files_writes_all_outputs(tmp_path: Path) -> None:
    clean_articles_path = tmp_path / "articles_clean.csv"
    output_dir = tmp_path / "graph"
    sample_clean_articles().to_csv(clean_articles_path, index=False)

    output_counts = build_attribute_graph_files(
        clean_articles_path=clean_articles_path,
        graph_dir=output_dir,
    )

    assert output_counts["nodes_article"] == 3
```

文件存在与回滚检查使用直接断言：

```python
assert (output_dir / "nodes_article.csv").exists()
assert list(output_dir.glob("*.tmp")) == []
assert list(output_dir.glob("*.bak")) == []
```

- [ ] **步骤 5：确认没有 unittest 残留**

运行：

```sh
rg -n "unittest|TestCase|self\\.assert|TemporaryDirectory|subTest" tests/test_attribute_graph.py
```

期望：无匹配。

- [ ] **步骤 6：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_attribute_graph.py -q
```

期望：`tests/test_attribute_graph.py` 中所有测试通过。由于原 `subTest` 会被参数化，pytest 报告的 case 数可能多于迁移前。

- [ ] **步骤 7：运行全量 pytest**

运行：

```sh
uv run pytest
```

期望：全部测试通过。

- [ ] **步骤 8：检查 diff 并提交**

运行：

```sh
git diff -- tests/test_attribute_graph.py
git add tests/test_attribute_graph.py
git commit -m "test: 迁移属性图测试到 pytest"
```

期望：commit 成功。

---

### 任务 4：提取趋势测试共享样本

**文件：**
- 新建：`tests/__init__.py`
- 新建：`tests/trend_samples.py`
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：创建共享样本模块**

创建空的 `tests/__init__.py`，再新建 `tests/trend_samples.py` 并写入：

```python
from __future__ import annotations

import pandas as pd
```

- [ ] **步骤 2：原样移动跨文件样本 builder**

将以下函数从 `tests/test_trend.py` 移动到 `tests/trend_samples.py`：

```python
def sample_article_attribute_edges() -> pd.DataFrame:
def sample_attribute_nodes() -> pd.DataFrame:
def sample_attribute_week_heat() -> pd.DataFrame:
def sample_long_attribute_week_heat() -> pd.DataFrame:
def sample_attribute_hierarchy_edges() -> pd.DataFrame:
def sample_trend_model_samples_for_split() -> pd.DataFrame:
def sample_trend_predictions_for_evaluation() -> pd.DataFrame:
```

移动时保持函数体不变，再处理调用方。

- [ ] **步骤 3：在 `tests/test_trend.py` 中导入共享样本**

在 `tests/test_trend.py` 的项目 import 后添加：

```python
from tests.trend_samples import (
    sample_article_attribute_edges,
    sample_attribute_hierarchy_edges,
    sample_attribute_nodes,
    sample_attribute_week_heat,
    sample_long_attribute_week_heat,
    sample_trend_model_samples_for_split,
    sample_trend_predictions_for_evaluation,
)
```

这样在真正拆分前，当前单体测试文件仍能通过。

- [ ] **步骤 4：确认重复定义已从 `tests/test_trend.py` 移除**

运行：

```sh
rg -n "^def sample_(article_attribute_edges|attribute_nodes|attribute_week_heat|long_attribute_week_heat|attribute_hierarchy_edges|trend_model_samples_for_split|trend_predictions_for_evaluation)" tests/test_trend.py
```

期望：无匹配。

- [ ] **步骤 5：运行趋势测试**

运行：

```sh
uv run pytest tests/test_trend.py -q
```

期望：128 个测试通过。

- [ ] **步骤 6：运行全量 pytest**

运行：

```sh
uv run pytest
```

期望：全部测试通过。

- [ ] **步骤 7：检查 diff 并提交**

运行：

```sh
git diff -- tests/__init__.py tests/trend_samples.py tests/test_trend.py
git add tests/__init__.py tests/trend_samples.py tests/test_trend.py
git commit -m "test: 提取趋势测试共享样本"
```

期望：commit 成功。

---

### 任务 5：拆分商品周销量和趋势 CSV 写出测试

**文件：**
- 新建：`tests/test_trend_article_sales.py`
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：创建目标文件 import**

创建 `tests/test_trend_article_sales.py`，写入迁移后测试需要的 import：

```python
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.transactions.weekly import read_weekly_transactions
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.schema import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
)
```

- [ ] **步骤 2：移动局部样本 builder**

将以下函数从 `tests/test_trend.py` 移动到 `tests/test_trend_article_sales.py`：

```python
def sample_weekly_transactions() -> pd.DataFrame:
def sample_article_week_sales() -> pd.DataFrame:
```

函数体保持不变。

- [ ] **步骤 3：移动并重命名测试类**

将以下类从 `tests/test_trend.py` 移到 `tests/test_trend_article_sales.py`：

```python
class ArticleWeekSalesFrameTests(unittest.TestCase):
class TrendCsvWriteTests(unittest.TestCase):
```

重命名为：

```python
class TestArticleWeekSalesFrame:
class TestTrendCsvWrite:
```

- [ ] **步骤 4：转换临时目录、异常断言和普通断言**

文件测试使用 `tmp_path: Path`，异常断言使用 `pytest.raises`。示例：

```python
def test_read_weekly_transactions_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="周级交易表不存在"):
        read_weekly_transactions(missing_path)
```

普通断言使用：

```python
assert sales.columns.tolist() == list(ARTICLE_WEEK_SALES_COLUMNS)
assert math.isclose(float(sales.loc[0, "sales_amount"]), 0.30)
assert not output_path.with_suffix(".csv.tmp").exists()
```

- [ ] **步骤 5：确认 moved definitions 已从 `tests/test_trend.py` 移除**

运行：

```sh
rg -n "sample_weekly_transactions|sample_article_week_sales|ArticleWeekSalesFrameTests|TrendCsvWriteTests" tests/test_trend.py
```

期望：无匹配。

- [ ] **步骤 6：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_article_sales.py tests/test_trend.py -q
```

期望：拆分出的商品周销量测试和剩余趋势测试通过。

- [ ] **步骤 7：运行全量 pytest**

运行：

```sh
uv run pytest
```

期望：全部测试通过，收集到的测试语义覆盖点不少于 147 个。

- [ ] **步骤 8：检查 diff 并提交**

运行：

```sh
git diff -- tests/test_trend_article_sales.py tests/test_trend.py
git add tests/test_trend_article_sales.py tests/test_trend.py
git commit -m "test: 拆分商品周销量趋势测试"
```

期望：commit 成功。

---

### 任务 6：拆分属性周热度测试

**文件：**
- 新建：`tests/test_trend_attribute_heat.py`
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：创建目标文件 import**

创建 `tests/test_trend_attribute_heat.py`，写入：

```python
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from tests.trend_samples import (
    sample_article_attribute_edges,
    sample_attribute_nodes,
)
from fashion_trend.catalog.graph import read_article_attribute_edges
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_article_attribute_edges_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.schema import ATTRIBUTE_WEEK_HEAT_COLUMNS
```

- [ ] **步骤 2：移动该文件专用样本 builder**

将以下函数从 `tests/test_trend.py` 移到 `tests/test_trend_attribute_heat.py`：

```python
def sample_attribute_article_week_sales() -> pd.DataFrame:
```

函数体保持不变。

- [ ] **步骤 3：移动并重命名测试类**

移动：

```python
class AttributeWeekHeatFrameTests(unittest.TestCase):
```

重命名为：

```python
class TestAttributeWeekHeatFrame:
```

- [ ] **步骤 4：将 subTest 循环改为 parametrize**

对于负数热度列检查，使用：

```python
@pytest.mark.parametrize("column", ["heat_cnt", "type_total_heat", "heat_share", "log_heat"])
def test_validate_attribute_week_heat_rejects_negative_heat_values(column: str) -> None:
    heat = sample_attribute_week_heat()
    heat.loc[0, column] = -1

    with pytest.raises(ValueError, match=column):
        validate_attribute_week_heat(
            heat,
            expected_week_ids=sorted(heat["week_id"].unique()),
            expected_attr_ids=sorted(heat["attr_id"].unique()),
        )
```

如果现有测试使用的数字列集合不同，以现有测试为准。

- [ ] **步骤 5：转换 tmp path 和断言**

示例：

```python
def test_read_article_attribute_edges_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="商品-属性边表不存在"):
        read_article_attribute_edges(missing_path)
```

普通断言示例：

```python
assert len(heat) == 12
assert heat.columns.tolist() == list(ATTRIBUTE_WEEK_HEAT_COLUMNS)
assert set(heat["attr_id"]) == set(sample_attribute_nodes()["attr_id"])
```

- [ ] **步骤 6：确认 moved definitions 已从 `tests/test_trend.py` 移除**

运行：

```sh
rg -n "sample_attribute_article_week_sales|AttributeWeekHeatFrameTests" tests/test_trend.py
```

期望：无匹配。

- [ ] **步骤 7：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_attribute_heat.py tests/test_trend.py -q
```

期望：拆分出的属性周热度测试和剩余趋势测试通过。

- [ ] **步骤 8：运行全量 pytest 并提交**

运行：

```sh
uv run pytest
git diff -- tests/test_trend_attribute_heat.py tests/test_trend.py
git add tests/test_trend_attribute_heat.py tests/test_trend.py
git commit -m "test: 拆分属性周热度测试"
```

期望：pytest 先通过，然后 commit 成功。

---

### 任务 7：拆分趋势标签测试

**文件：**
- 新建：`tests/test_trend_targets.py`
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：创建目标文件 import**

创建 `tests/test_trend_targets.py`，写入：

```python
from __future__ import annotations

import math

import pytest

from tests.trend_samples import sample_attribute_week_heat
from fashion_trend.trend.schema import ATTRIBUTE_WEEK_TARGET_COLUMNS
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    validate_attribute_week_target,
)
```

- [ ] **步骤 2：移动并重命名测试类**

移动：

```python
class AttributeWeekTargetFrameTests(unittest.TestCase):
```

重命名为：

```python
class TestAttributeWeekTargetFrame:
```

- [ ] **步骤 3：转换断言和异常检查**

使用裸 `assert` 和 `pytest.raises`：

```python
target = build_attribute_week_target_frame(sample_attribute_week_heat())

assert target.columns.tolist() == list(ATTRIBUTE_WEEK_TARGET_COLUMNS)
assert len(target) == 6
assert set(target["week_id"]) == {0}

with pytest.raises(ValueError, match="target_growth"):
    validate_attribute_week_target(bad_target)
```

现有 `math.isclose` 比较保留，只把 `self.assertTrue(...)` 改为 `assert ...`。

- [ ] **步骤 4：确认 moved class 已从 `tests/test_trend.py` 移除**

运行：

```sh
rg -n "AttributeWeekTargetFrameTests" tests/test_trend.py
```

期望：无匹配。

- [ ] **步骤 5：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_targets.py tests/test_trend.py -q
```

期望：拆分出的趋势标签测试和剩余趋势测试通过。

- [ ] **步骤 6：运行全量 pytest 并提交**

运行：

```sh
uv run pytest
git diff -- tests/test_trend_targets.py tests/test_trend.py
git add tests/test_trend_targets.py tests/test_trend.py
git commit -m "test: 拆分趋势标签测试"
```

期望：pytest 先通过，然后 commit 成功。

---

### 任务 8：拆分趋势样本测试

**文件：**
- 新建：`tests/test_trend_samples.py`
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：创建目标文件 import**

创建 `tests/test_trend_samples.py`，写入：

```python
from __future__ import annotations

import math

import pandas as pd
import pytest

from tests.trend_samples import (
    sample_attribute_hierarchy_edges,
    sample_attribute_nodes,
    sample_long_attribute_week_heat,
)
from fashion_trend.trend.samples import (
    build_attribute_graph_features_frame,
    build_trend_model_samples_frame,
    validate_trend_model_samples,
)
from fashion_trend.trend.schema import TREND_MODEL_SAMPLE_COLUMNS
```

- [ ] **步骤 2：移动并重命名测试类**

移动：

```python
class TrendModelSamplesFrameTests(unittest.TestCase):
```

重命名为：

```python
class TestTrendModelSamplesFrame:
```

- [ ] **步骤 3：将 edge_weight subTest 改为 parametrize**

使用：

```python
@pytest.mark.parametrize("edge_weight", [0, -1])
def test_build_attribute_graph_features_frame_rejects_non_positive_edge_weight(
    edge_weight: int,
) -> None:
    edges = sample_attribute_hierarchy_edges()
    edges.loc[0, "edge_weight"] = edge_weight

    with pytest.raises(ValueError, match="edge_weight"):
        build_attribute_graph_features_frame(sample_attribute_nodes(), edges)
```

- [ ] **步骤 4：转换样本断言**

使用：

```python
assert samples.columns.tolist() == list(TREND_MODEL_SAMPLE_COLUMNS)
assert set(samples["week_id"]) == {4}
assert int(black["heat_t"]) == 8
assert math.isclose(float(black["heat_ma_4"]), (1 + 3 + 4 + 8) / 4)
assert "target_growth" in samples.columns
```

异常检查使用：

```python
with pytest.raises(ValueError, match="target_growth"):
    validate_trend_model_samples(samples_without_target)
```

- [ ] **步骤 5：确认 moved class 已从 `tests/test_trend.py` 移除**

运行：

```sh
rg -n "TrendModelSamplesFrameTests" tests/test_trend.py
```

期望：无匹配。

- [ ] **步骤 6：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_samples.py tests/test_trend.py -q
```

期望：拆分出的趋势样本测试和剩余趋势测试通过。

- [ ] **步骤 7：运行全量 pytest 并提交**

运行：

```sh
uv run pytest
git diff -- tests/test_trend_samples.py tests/test_trend.py
git add tests/test_trend_samples.py tests/test_trend.py
git commit -m "test: 拆分趋势样本测试"
```

期望：pytest 先通过，然后 commit 成功。

---

### 任务 9：拆分趋势切分测试

**文件：**
- 新建：`tests/test_trend_splits.py`
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：创建目标文件 import**

创建 `tests/test_trend_splits.py`，写入：

```python
from __future__ import annotations

from pathlib import Path

import pytest

from tests.trend_samples import sample_trend_model_samples_for_split
from fashion_trend.foundation.io import write_json_atomic, write_parquet_atomic
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_COLUMNS
from fashion_trend.trend.splits import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    read_trend_model_split,
)
```

- [ ] **步骤 2：移动并重命名测试类**

移动：

```python
class TrendModelSplitFrameTests(unittest.TestCase):
class TrendModelSplitWriteTests(unittest.TestCase):
```

重命名为：

```python
class TestTrendModelSplitFrame:
class TestTrendModelSplitWrite:
```

- [ ] **步骤 3：转换 tmp path 和断言**

示例：

```python
def test_read_trend_model_split_preserves_columns_for_legal_parquet(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.parquet"
    split_frame = build_trend_model_split_frames(sample_trend_model_samples_for_split())[
        "train"
    ]
    write_parquet_atomic(split_frame, split_path)

    split = read_trend_model_split(split_path)

    assert split.columns.tolist() == list(TREND_MODEL_SPLIT_COLUMNS)
    assert set(split["split"]) == {"train"}
```

非法 split 和重复 key 检查使用 `pytest.raises`。

- [ ] **步骤 4：确认 moved classes 已从 `tests/test_trend.py` 移除**

运行：

```sh
rg -n "TrendModelSplitFrameTests|TrendModelSplitWriteTests" tests/test_trend.py
```

期望：无匹配。

- [ ] **步骤 5：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_splits.py tests/test_trend.py -q
```

期望：拆分出的趋势切分测试和剩余趋势测试通过。

- [ ] **步骤 6：运行全量 pytest 并提交**

运行：

```sh
uv run pytest
git diff -- tests/test_trend_splits.py tests/test_trend.py
git add tests/test_trend_splits.py tests/test_trend.py
git commit -m "test: 拆分趋势切分测试"
```

期望：pytest 先通过，然后 commit 成功。

---

### 任务 10：拆分趋势训练测试

**文件：**
- 新建：`tests/test_trend_training.py`
- 修改：`tests/test_trend.py`

- [ ] **步骤 1：创建目标文件 import**

创建 `tests/test_trend_training.py`，写入：

```python
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tests.trend_samples import sample_trend_model_samples_for_split
from fashion_trend.config import OUTPUT_MODELS_DIR
from fashion_trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendArtifact,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.models.last_week import (
    LAST_WEEK_MODEL_NAME,
    LAST_WEEK_PARAMS,
    LastWeekTrainer,
    predict_last_week,
)
from fashion_trend.models.moving_average import (
    MOVING_AVERAGE_GROWTH_LAGS,
    MOVING_AVERAGE_MODEL_NAME,
    MOVING_AVERAGE_PARAMS,
    MovingAverageTrainer,
    predict_moving_average,
)
from fashion_trend.models.registry import (
    UnknownTrendModelError,
    get_trend_model_trainer,
    list_trend_model_names,
)
from fashion_trend.training import (
    build_trend_train_metadata,
    derive_trend_model_output_paths,
    run_trend_model_training,
    validate_trend_train_result,
    write_trend_model_outputs,
)
from fashion_trend.foundation.io import write_parquet_atomic
from fashion_trend.trend.predictions import validate_trend_model_predictions
from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
from fashion_trend.trend.splits import build_trend_model_split_frames
```

- [ ] **步骤 2：移动并重命名测试类**

移动：

```python
class LastWeekBaselineTests(unittest.TestCase):
```

重命名为：

```python
class TestTrendTraining:
```

- [ ] **步骤 3：转换参数化负向值测试**

非有限增长 lag 测试改为：

```python
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_predict_moving_average_rejects_non_finite_growth_lag(
    bad_value: float,
) -> None:
    split_frames = build_trend_model_split_frames(sample_trend_model_samples_for_split())
    samples = split_frames["train"].copy()
    samples.loc[0, "growth_lag_1"] = bad_value

    with pytest.raises(ValueError, match="非有限|增长 lag"):
        predict_moving_average(samples)
```

unsafe artifact path 测试改为：

```python
@pytest.mark.parametrize("unsafe_path", ["../escape.csv", "/tmp/escape.csv"])
def test_write_trend_model_outputs_rejects_unsafe_artifact_path_before_writing(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    paths = derive_trend_model_output_paths("last_week", output_root=tmp_path)
    result = TrendTrainResult(
        model_name=LAST_WEEK_MODEL_NAME,
        model_type=MODEL_TYPE_BASELINE,
        predictions=predict_last_week(
            build_trend_model_split_frames(sample_trend_model_samples_for_split())["train"]
        ),
        params=LAST_WEEK_PARAMS,
        artifacts=(TrendArtifact(relative_path=unsafe_path, kind="json", payload={}),),
    )

    with pytest.raises(ValueError, match="artifact"):
        write_trend_model_outputs(result, paths)
```

转换时保留现有测试里的 payload、路径和失败前不写文件的检查语义。

- [ ] **步骤 4：转换 CLI 测试但不改变生产行为**

保留：

```python
train_model = importlib.import_module("10_train_trend_model")
```

断言改为：

```python
assert train_model.main(["--unknown"]) == 2
assert train_model.main(["--model", "unknown_model"]) == 1
```

如果当前测试 monkeypatch 了模块全局变量，保留同样的 monkeypatch 方式，只转换断言写法。

- [ ] **步骤 5：确认 moved class 已从 `tests/test_trend.py` 移除**

运行：

```sh
rg -n "LastWeekBaselineTests" tests/test_trend.py
```

期望：无匹配。

- [ ] **步骤 6：运行聚焦测试**

运行：

```sh
uv run pytest tests/test_trend_training.py tests/test_trend.py -q
```

期望：拆分出的趋势训练测试和剩余趋势测试通过。

- [ ] **步骤 7：运行全量 pytest 并提交**

运行：

```sh
uv run pytest
git diff -- tests/test_trend_training.py tests/test_trend.py
git add tests/test_trend_training.py tests/test_trend.py
git commit -m "test: 拆分趋势训练测试"
```

期望：pytest 先通过，然后 commit 成功。

---

### 任务 11：拆分趋势评价测试并删除单体文件

**文件：**
- 新建：`tests/test_trend_evaluation.py`
- 删除：`tests/test_trend.py`

- [ ] **步骤 1：创建目标文件 import**

创建 `tests/test_trend_evaluation.py`，写入：

```python
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import pytest

from tests.trend_samples import (
    sample_trend_model_samples_for_split,
    sample_trend_predictions_for_evaluation,
)
from fashion_trend.evaluation import (
    build_trend_metrics_payload,
    compute_trend_group_metrics,
    compute_trend_metrics,
    derive_trend_metric_output_paths,
    read_trend_model_predictions,
    run_trend_model_evaluation,
    validate_trend_model_predictions_for_evaluation,
    write_trend_metrics,
)
from fashion_trend.models.moving_average import MOVING_AVERAGE_MODEL_NAME
from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
```

- [ ] **步骤 2：移动并重命名评价测试类**

移动：

```python
class TrendEvaluationTests(unittest.TestCase):
```

重命名为：

```python
class TestTrendEvaluation:
```

- [ ] **步骤 3：将 model_name subTest 改为 parametrize**

使用：

```python
@pytest.mark.parametrize("model_name", ["../escape", "/tmp/escape"])
def test_derive_trend_metric_output_paths_rejects_unsafe_model_name(
    model_name: str,
) -> None:
    with pytest.raises(ValueError, match="model_name|模型名"):
        derive_trend_metric_output_paths(model_name)
```

如果现有测试中的 unsafe model name 不同，以现有测试为准。

- [ ] **步骤 4：转换 tmp path、指标断言和 CLI 断言**

示例：

```python
def test_read_trend_model_predictions_preserves_contract_columns(
    tmp_path: Path,
) -> None:
    predictions = sample_trend_predictions_for_evaluation()
    prediction_path = tmp_path / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    loaded = read_trend_model_predictions(prediction_path)

    assert loaded.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
    assert len(loaded) == len(predictions)
```

指标精度断言使用：

```python
assert math.isclose(metrics["mae"], 0.5666666667, rel_tol=1e-9)
assert metrics["precision_at_k"]["2"] == 0.5
```

CLI 测试继续使用 `importlib.import_module("11_eval_trend_model")`，退出码检查改为裸 `assert`。

- [ ] **步骤 5：删除 `tests/test_trend.py`**

移动完 `TrendEvaluationTests` 后，`tests/test_trend.py` 不应再包含测试类或局部 helper，删除该文件。

运行：

```sh
test ! -e tests/test_trend.py
```

期望：命令退出码为 0。

- [ ] **步骤 6：运行评价测试**

运行：

```sh
uv run pytest tests/test_trend_evaluation.py -q
```

期望：拆分出的评价测试通过。

- [ ] **步骤 7：运行全量 pytest**

运行：

```sh
uv run pytest
```

期望：全部测试通过，收集到的测试语义覆盖点不少于 147 个。

- [ ] **步骤 8：提交**

运行：

```sh
git diff -- tests/test_trend_evaluation.py tests/test_trend.py
git add tests/test_trend_evaluation.py
git rm tests/test_trend.py
git commit -m "test: 拆分趋势评价测试"
```

期望：commit 成功。

---

### 任务 12：最终清理与验证

**文件：**
- 修改：`README.md`，仅当最终验证说明需要收口措辞时修改
- 修改：`docs/superpowers/specs/2026-05-06-pytest-migration-design.md`，仅当实施发现已批准 spec 必须修正时修改

- [ ] **步骤 1：检查 active tests 中是否还有 unittest 残留**

运行：

```sh
rg -n "unittest|TestCase|self\\.assert|TemporaryDirectory|subTest" tests
```

期望：无匹配。

- [ ] **步骤 2：检查正式文档是否还有旧测试命令**

运行：

```sh
rg -n "当前测试使用标准库|unittest discover|uv run python -m unittest discover" README.md docs/gpt-research/implementation-plan.md
```

期望：无匹配。

- [ ] **步骤 3：确认目标测试文件布局**

运行：

```sh
find tests -maxdepth 1 -type f | sort
```

期望输出包含：

```text
tests/__init__.py
tests/test_articles_clean.py
tests/test_attribute_graph.py
tests/test_trend_article_sales.py
tests/test_trend_attribute_heat.py
tests/test_trend_evaluation.py
tests/test_trend_samples.py
tests/test_trend_splits.py
tests/test_trend_targets.py
tests/test_trend_training.py
tests/trend_samples.py
```

期望输出不包含 `tests/test_trend.py`。

- [ ] **步骤 4：运行全量 pytest**

运行：

```sh
uv run pytest
```

期望：全部测试通过，收集到的测试语义覆盖点不少于 147 个。

- [ ] **步骤 5：运行语法编译检查**

运行：

```sh
uv run python -m py_compile \
  tests/__init__.py \
  tests/trend_samples.py \
  tests/test_articles_clean.py \
  tests/test_attribute_graph.py \
  tests/test_trend_article_sales.py \
  tests/test_trend_attribute_heat.py \
  tests/test_trend_targets.py \
  tests/test_trend_samples.py \
  tests/test_trend_splits.py \
  tests/test_trend_training.py \
  tests/test_trend_evaluation.py
```

期望：命令成功退出且无输出。

- [ ] **步骤 6：检查整个分支 diff**

运行：

```sh
git diff master...HEAD --stat
git diff master...HEAD -- pyproject.toml README.md tests docs/superpowers/specs docs/superpowers/plans
```

期望：diff 范围只包含 pytest 依赖/配置、README 验证说明、已提交的 spec/plan 文档，以及测试迁移/拆分文件。

- [ ] **步骤 7：如有最终清理改动则提交**

如果步骤 1 或步骤 2 需要文档或测试清理，提交：

```sh
git add README.md docs/superpowers/specs/2026-05-06-pytest-migration-design.md tests
git commit -m "test: 收口 pytest 迁移验证"
```

期望：只有存在清理改动时才提交；没有改动时不要创建空提交。

---

## 执行备注

- 手动编辑使用 `apply_patch`。
- 不使用生成式 codemod 或一次性重写脚本。
- 如果 `uv run ...` 失败并出现 `Failed to initialize cache at /Users/ghstlnx/.cache/uv`，用已批准的提权 `uv run` 权限重跑同一命令。该问题按本地权限问题处理，不视为测试失败。
- 每次 commit 前，先运行对应任务的 pytest 命令，检查 `git diff`，并且只提交该任务列出的文件。
- 不通过弱化断言、添加 skip、隐藏异常来让 pytest 通过。
