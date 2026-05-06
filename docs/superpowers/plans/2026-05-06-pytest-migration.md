# pytest Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the test suite to pytest and split the oversized trend tests along the project data pipeline without changing production behavior.

**Architecture:** Add pytest as the official test runner first, then convert the two small test files, then split `tests/test_trend.py` by README and research-plan stages. Shared sample builders move only when multiple new trend test files need them; each task ends with a focused pytest run and a commit.

**Tech Stack:** Python 3.10-3.12, uv, pytest, pandas, pyarrow, existing `fashion_trend` modules.

---

## File Structure

Target test layout:

```text
tests/
    conftest.py
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

Implementation boundaries:

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Declare pytest dev dependency and configure test discovery/import path |
| `uv.lock` | Lock pytest and its transitive dev dependencies |
| `README.md` | Switch the official verification command to `uv run pytest` |
| `tests/conftest.py` | Cross-file trend sample builders used by multiple split test files |
| `tests/test_articles_clean.py` | pytest idiom for article cleaning and two-output write rollback |
| `tests/test_attribute_graph.py` | pytest idiom for attribute graph builders and graph file rollback |
| `tests/test_trend_article_sales.py` | Article week sales, trend CSV write helper, and related readers |
| `tests/test_trend_attribute_heat.py` | Attribute edge/node readers and attribute week heat frame validation |
| `tests/test_trend_targets.py` | Attribute week target calculation and validation |
| `tests/test_trend_samples.py` | Trend sample lag/window/graph/target merge behavior |
| `tests/test_trend_splits.py` | Trend train/valid/test split frames, split readers, JSON/Parquet helpers |
| `tests/test_trend_training.py` | Baseline trainers, registry, training runner, training CLI |
| `tests/test_trend_evaluation.py` | Evaluation prediction readers, metrics, payload, writer, evaluation CLI |

Current `tests/test_trend.py` class-to-file mapping:

| Current class | Target file |
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

### Task 1: Add pytest Runner and Official Command

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`

- [ ] **Step 1: Confirm clean starting point**

Run:

```sh
git status --short --branch
```

Expected: branch is `codex/pytest-migration` and no uncommitted files before starting the task.

- [ ] **Step 2: Add pytest to the dev dependency group**

Run:

```sh
uv add --dev pytest
```

Expected: `pyproject.toml` gains `pytest` under `[dependency-groups].dev`, and `uv.lock` gains pytest plus transitive packages such as `iniconfig`, `packaging`, `pluggy`, and `pygments` if uv resolves them.

Expected `pyproject.toml` shape:

```toml
[dependency-groups]
dev = [
    "isort>=8.0.1",
    "pytest>=8.0.0",
]
```

- [ ] **Step 3: Add pytest configuration**

Edit `pyproject.toml` so the bottom includes:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Keep existing `[tool.setuptools.packages.find]` unchanged.

- [ ] **Step 4: Run pytest against the unchanged unittest-style suite**

Run:

```sh
uv run pytest
```

Expected: pytest collects and passes the existing 147 tests. The exact header can vary by pytest version, but the run must end with all tests passed.

- [ ] **Step 5: Update README verification command**

Change the verification section from:

````markdown
当前测试使用标准库 `unittest`，不依赖真实 H&M 数据：

```sh
uv run python -m unittest discover -s tests -v
```
````

to:

````markdown
当前测试使用 `pytest`，不依赖真实 H&M 数据：

```sh
uv run pytest
```
````

Keep the existing coverage bullet list below the command.

- [ ] **Step 6: Verify documentation no longer advertises unittest as the current command**

Run:

```sh
rg -n "当前测试使用标准库|unittest discover|uv run python -m unittest discover" README.md docs/gpt-research/implementation-plan.md
```

Expected: no matches.

- [ ] **Step 7: Run full pytest**

Run:

```sh
uv run pytest
```

Expected: all 147 tests pass.

- [ ] **Step 8: Review diff**

Run:

```sh
git diff -- pyproject.toml uv.lock README.md
```

Expected: only pytest dependency/config, lockfile dependency resolution, and README verification command changed.

- [ ] **Step 9: Commit**

Run:

```sh
git add pyproject.toml uv.lock README.md
git commit -m "test: 引入 pytest 测试入口"
```

Expected: commit succeeds.

---

### Task 2: Convert Article Cleaning Tests to pytest Idiom

**Files:**
- Modify: `tests/test_articles_clean.py`

- [ ] **Step 1: Update imports**

Replace:

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
```

with:

```python
from pathlib import Path

import pytest
```

Keep the existing `pandas` and `fashion_trend.articles` imports.

- [ ] **Step 2: Convert test classes**

Replace:

```python
class CleanArticleFrameTests(unittest.TestCase):
```

with:

```python
class TestCleanArticleFrame:
```

Replace:

```python
class CleanArticleFileTests(unittest.TestCase):
```

with:

```python
class TestCleanArticleFile:
```

- [ ] **Step 3: Convert frame assertions**

In `TestCleanArticleFrame`, convert assertions directly. For example:

```python
assert list(mvp_articles.columns) == list(MVP_ARTICLE_COLUMNS)
assert list(clean_articles.columns) == list(CLEAN_ARTICLE_COLUMNS)
assert len(mvp_articles) == 2
assert len(clean_articles) == 2
assert mvp_articles["article_id"].tolist() == ["0108775015", "0108775044"]
assert "detail_desc" not in mvp_articles.columns
assert "detail_desc" not in clean_articles.columns
```

For exception assertions, use:

```python
with pytest.raises(ValueError, match="缺少必要字段: product_type_name"):
    validate_required_columns(
        ["article_id", "product_group_name"],
        ["article_id", "product_group_name", "product_type_name"],
        source_name="测试 articles 表",
    )
```

- [ ] **Step 4: Convert file tests to `tmp_path`**

Change tests that currently open `TemporaryDirectory()` to accept `tmp_path: Path`. For example:

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

Apply the same `tmp_path` pattern to all file tests in this file.

- [ ] **Step 5: Remove unittest references**

Run:

```sh
rg -n "unittest|TestCase|self\\.assert|TemporaryDirectory" tests/test_articles_clean.py
```

Expected: no matches.

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_articles_clean.py -q
```

Expected: 9 tests pass.

- [ ] **Step 7: Run full pytest**

Run:

```sh
uv run pytest
```

Expected: all tests pass.

- [ ] **Step 8: Review diff and commit**

Run:

```sh
git diff -- tests/test_articles_clean.py
git add tests/test_articles_clean.py
git commit -m "test: 迁移 articles 清洗测试到 pytest"
```

Expected: commit succeeds.

---

### Task 3: Convert Attribute Graph Tests to pytest Idiom

**Files:**
- Modify: `tests/test_attribute_graph.py`

- [ ] **Step 1: Update imports**

Replace:

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
```

with:

```python
from pathlib import Path

import pytest
```

Keep the existing pandas and `fashion_trend.articles` imports.

- [ ] **Step 2: Convert test classes**

Replace:

```python
class AttributeGraphBuilderTests(unittest.TestCase):
```

with:

```python
class TestAttributeGraphBuilder:
```

Replace:

```python
class AttributeGraphFileTests(unittest.TestCase):
```

with:

```python
class TestAttributeGraphFile:
```

- [ ] **Step 3: Convert the builder subTest loop to parametrize**

Replace the loop in `test_graph_builders_reject_missing_attribute_values` with:

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

- [ ] **Step 4: Convert file tests to `tmp_path`**

Change file tests to accept `tmp_path: Path`. Example:

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

Use direct assertions for file existence and rollback checks:

```python
assert (output_dir / "nodes_article.csv").exists()
assert list(output_dir.glob("*.tmp")) == []
assert list(output_dir.glob("*.bak")) == []
```

- [ ] **Step 5: Remove unittest references**

Run:

```sh
rg -n "unittest|TestCase|self\\.assert|TemporaryDirectory|subTest" tests/test_attribute_graph.py
```

Expected: no matches.

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_attribute_graph.py -q
```

Expected: all tests in `tests/test_attribute_graph.py` pass. pytest may report more than 10 cases because the former `subTest` loop is parameterized.

- [ ] **Step 7: Run full pytest**

Run:

```sh
uv run pytest
```

Expected: all tests pass.

- [ ] **Step 8: Review diff and commit**

Run:

```sh
git diff -- tests/test_attribute_graph.py
git add tests/test_attribute_graph.py
git commit -m "test: 迁移属性图测试到 pytest"
```

Expected: commit succeeds.

---

### Task 4: Create Shared Trend Test Samples

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Create `tests/conftest.py` with shared imports**

Create the file with this header:

```python
from __future__ import annotations

import pandas as pd
```

- [ ] **Step 2: Move cross-file sample builders unchanged**

Move these function definitions from `tests/test_trend.py` into `tests/conftest.py`:

```python
def sample_article_attribute_edges() -> pd.DataFrame:
def sample_attribute_nodes() -> pd.DataFrame:
def sample_attribute_week_heat() -> pd.DataFrame:
def sample_long_attribute_week_heat() -> pd.DataFrame:
def sample_attribute_hierarchy_edges() -> pd.DataFrame:
def sample_trend_model_samples_for_split() -> pd.DataFrame:
def sample_trend_predictions_for_evaluation() -> pd.DataFrame:
```

Keep each function body exactly as it is before changing call sites.

- [ ] **Step 3: Import shared sample builders in `tests/test_trend.py`**

Add this import below the existing project imports in `tests/test_trend.py`:

```python
from conftest import (
    sample_article_attribute_edges,
    sample_attribute_hierarchy_edges,
    sample_attribute_nodes,
    sample_attribute_week_heat,
    sample_long_attribute_week_heat,
    sample_trend_model_samples_for_split,
    sample_trend_predictions_for_evaluation,
)
```

This keeps the current monolithic test file passing before splitting begins.

- [ ] **Step 4: Verify duplicate definitions are gone from `tests/test_trend.py`**

Run:

```sh
rg -n "^def sample_(article_attribute_edges|attribute_nodes|attribute_week_heat|long_attribute_week_heat|attribute_hierarchy_edges|trend_model_samples_for_split|trend_predictions_for_evaluation)" tests/test_trend.py
```

Expected: no matches.

- [ ] **Step 5: Run focused trend tests**

Run:

```sh
uv run pytest tests/test_trend.py -q
```

Expected: 128 tests pass.

- [ ] **Step 6: Run full pytest**

Run:

```sh
uv run pytest
```

Expected: all tests pass.

- [ ] **Step 7: Review diff and commit**

Run:

```sh
git diff -- tests/conftest.py tests/test_trend.py
git add tests/conftest.py tests/test_trend.py
git commit -m "test: 提取趋势测试共享样本"
```

Expected: commit succeeds.

---

### Task 5: Split Article Sales and CSV Writer Tests

**Files:**
- Create: `tests/test_trend_article_sales.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Create target file imports**

Create `tests/test_trend_article_sales.py` with imports needed by the moved tests:

```python
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.trend import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    build_article_week_sales_frame,
    read_article_week_sales,
    read_weekly_transactions,
    validate_article_week_sales,
    write_trend_csv,
)
```

- [ ] **Step 2: Move local sample builders**

Move these function definitions from `tests/test_trend.py` into `tests/test_trend_article_sales.py`:

```python
def sample_weekly_transactions() -> pd.DataFrame:
def sample_article_week_sales() -> pd.DataFrame:
```

Keep function bodies unchanged.

- [ ] **Step 3: Move current test classes into the target file**

Move these class bodies from `tests/test_trend.py` into `tests/test_trend_article_sales.py`:

```python
class ArticleWeekSalesFrameTests(unittest.TestCase):
class TrendCsvWriteTests(unittest.TestCase):
```

Then rename them:

```python
class TestArticleWeekSalesFrame:
class TestTrendCsvWrite:
```

- [ ] **Step 4: Convert tmp directory and assertions**

Use `tmp_path: Path` for file tests and pytest raises for exceptions. Example conversion:

```python
def test_read_weekly_transactions_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="周级交易表不存在"):
        read_weekly_transactions(missing_path)
```

Use direct assertions:

```python
assert sales.columns.tolist() == list(ARTICLE_WEEK_SALES_COLUMNS)
assert math.isclose(float(sales.loc[0, "sales_amount"]), 0.30)
assert not output_path.with_suffix(".csv.tmp").exists()
```

- [ ] **Step 5: Remove moved definitions from `tests/test_trend.py`**

Ensure these names no longer exist in `tests/test_trend.py`:

```sh
rg -n "sample_weekly_transactions|sample_article_week_sales|ArticleWeekSalesFrameTests|TrendCsvWriteTests" tests/test_trend.py
```

Expected: no matches.

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_article_sales.py tests/test_trend.py -q
```

Expected: the moved article sales tests and remaining trend tests pass.

- [ ] **Step 7: Run full pytest**

Run:

```sh
uv run pytest
```

Expected: all tests pass and total collected tests remains at least 147.

- [ ] **Step 8: Review diff and commit**

Run:

```sh
git diff -- tests/test_trend_article_sales.py tests/test_trend.py
git add tests/test_trend_article_sales.py tests/test_trend.py
git commit -m "test: 拆分商品周销量趋势测试"
```

Expected: commit succeeds.

---

### Task 6: Split Attribute Heat Tests

**Files:**
- Create: `tests/test_trend_attribute_heat.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Create target file imports**

Create `tests/test_trend_attribute_heat.py` with:

```python
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from conftest import (
    sample_article_attribute_edges,
    sample_attribute_nodes,
)
from fashion_trend.trend import (
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    build_attribute_week_heat_frame,
    read_article_attribute_edges,
    validate_article_attribute_edges_for_heat,
    validate_attribute_week_heat,
)
```

- [ ] **Step 2: Move required local sample builder**

Move this function from `tests/test_trend.py` into `tests/test_trend_attribute_heat.py`:

```python
def sample_attribute_article_week_sales() -> pd.DataFrame:
```

Keep its body unchanged.

- [ ] **Step 3: Move and rename the test class**

Move:

```python
class AttributeWeekHeatFrameTests(unittest.TestCase):
```

Rename to:

```python
class TestAttributeWeekHeatFrame:
```

- [ ] **Step 4: Convert subTest loops to parametrize**

For negative heat column checks, use:

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

Use the current expected column list if the existing test uses a different set of numeric columns.

- [ ] **Step 5: Convert tmp paths and assertions**

Use:

```python
def test_read_article_attribute_edges_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="商品-属性边表不存在"):
        read_article_attribute_edges(missing_path)
```

Use direct assertions:

```python
assert len(heat) == 12
assert heat.columns.tolist() == list(ATTRIBUTE_WEEK_HEAT_COLUMNS)
assert set(heat["attr_id"]) == set(sample_attribute_nodes()["attr_id"])
```

- [ ] **Step 6: Remove moved definitions from `tests/test_trend.py`**

Run:

```sh
rg -n "sample_attribute_article_week_sales|AttributeWeekHeatFrameTests" tests/test_trend.py
```

Expected: no matches.

- [ ] **Step 7: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_attribute_heat.py tests/test_trend.py -q
```

Expected: moved attribute heat tests and remaining trend tests pass.

- [ ] **Step 8: Run full pytest and commit**

Run:

```sh
uv run pytest
git diff -- tests/test_trend_attribute_heat.py tests/test_trend.py
git add tests/test_trend_attribute_heat.py tests/test_trend.py
git commit -m "test: 拆分属性周热度测试"
```

Expected: pytest passes before commit, then commit succeeds.

---

### Task 7: Split Trend Target Tests

**Files:**
- Create: `tests/test_trend_targets.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Create target file imports**

Create `tests/test_trend_targets.py` with:

```python
from __future__ import annotations

import math

import pytest

from conftest import sample_attribute_week_heat
from fashion_trend.trend import (
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    build_attribute_week_target_frame,
    validate_attribute_week_target,
)
```

- [ ] **Step 2: Move and rename target class**

Move:

```python
class AttributeWeekTargetFrameTests(unittest.TestCase):
```

Rename to:

```python
class TestAttributeWeekTargetFrame:
```

- [ ] **Step 3: Convert assertions and raises**

Use direct assertions and pytest raises:

```python
target = build_attribute_week_target_frame(sample_attribute_week_heat())

assert target.columns.tolist() == list(ATTRIBUTE_WEEK_TARGET_COLUMNS)
assert len(target) == 6
assert set(target["week_id"]) == {0}

with pytest.raises(ValueError, match="target_growth"):
    validate_attribute_week_target(bad_target)
```

Keep the current `math.isclose` comparisons, changing only `self.assertTrue(...)` to `assert ...`.

- [ ] **Step 4: Remove moved class from `tests/test_trend.py`**

Run:

```sh
rg -n "AttributeWeekTargetFrameTests" tests/test_trend.py
```

Expected: no matches.

- [ ] **Step 5: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_targets.py tests/test_trend.py -q
```

Expected: moved target tests and remaining trend tests pass.

- [ ] **Step 6: Run full pytest and commit**

Run:

```sh
uv run pytest
git diff -- tests/test_trend_targets.py tests/test_trend.py
git add tests/test_trend_targets.py tests/test_trend.py
git commit -m "test: 拆分趋势标签测试"
```

Expected: pytest passes before commit, then commit succeeds.

---

### Task 8: Split Trend Sample Tests

**Files:**
- Create: `tests/test_trend_samples.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Create target file imports**

Create `tests/test_trend_samples.py` with:

```python
from __future__ import annotations

import math

import pandas as pd
import pytest

from conftest import (
    sample_attribute_hierarchy_edges,
    sample_attribute_nodes,
    sample_long_attribute_week_heat,
)
from fashion_trend.trend import (
    TREND_MODEL_SAMPLE_COLUMNS,
    build_attribute_graph_features_frame,
    build_trend_model_samples_frame,
    validate_trend_model_samples,
)
```

- [ ] **Step 2: Move and rename samples class**

Move:

```python
class TrendModelSamplesFrameTests(unittest.TestCase):
```

Rename to:

```python
class TestTrendModelSamplesFrame:
```

- [ ] **Step 3: Convert edge-weight subTest to parametrize**

Use:

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

- [ ] **Step 4: Convert sample assertions**

Use direct assertions:

```python
assert samples.columns.tolist() == list(TREND_MODEL_SAMPLE_COLUMNS)
assert set(samples["week_id"]) == {4}
assert int(black["heat_t"]) == 8
assert math.isclose(float(black["heat_ma_4"]), (1 + 3 + 4 + 8) / 4)
assert "target_growth" in samples.columns
```

Use pytest raises:

```python
with pytest.raises(ValueError, match="target_growth"):
    validate_trend_model_samples(samples_without_target)
```

- [ ] **Step 5: Remove moved class from `tests/test_trend.py`**

Run:

```sh
rg -n "TrendModelSamplesFrameTests" tests/test_trend.py
```

Expected: no matches.

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_samples.py tests/test_trend.py -q
```

Expected: moved sample tests and remaining trend tests pass.

- [ ] **Step 7: Run full pytest and commit**

Run:

```sh
uv run pytest
git diff -- tests/test_trend_samples.py tests/test_trend.py
git add tests/test_trend_samples.py tests/test_trend.py
git commit -m "test: 拆分趋势样本测试"
```

Expected: pytest passes before commit, then commit succeeds.

---

### Task 9: Split Trend Split Tests

**Files:**
- Create: `tests/test_trend_splits.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Create target file imports**

Create `tests/test_trend_splits.py` with:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import sample_trend_model_samples_for_split
from fashion_trend.trend import (
    TREND_MODEL_SPLIT_COLUMNS,
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    read_trend_model_split,
    write_json,
    write_trend_parquet,
)
```

- [ ] **Step 2: Move and rename split classes**

Move:

```python
class TrendModelSplitFrameTests(unittest.TestCase):
class TrendModelSplitWriteTests(unittest.TestCase):
```

Rename to:

```python
class TestTrendModelSplitFrame:
class TestTrendModelSplitWrite:
```

- [ ] **Step 3: Convert tmp paths and assertions**

Use:

```python
def test_read_trend_model_split_preserves_columns_for_legal_parquet(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.parquet"
    split_frame = build_trend_model_split_frames(sample_trend_model_samples_for_split())[
        "train"
    ]
    write_trend_parquet(split_frame, split_path)

    split = read_trend_model_split(split_path)

    assert split.columns.tolist() == list(TREND_MODEL_SPLIT_COLUMNS)
    assert set(split["split"]) == {"train"}
```

Use pytest raises for invalid split and duplicate key checks.

- [ ] **Step 4: Remove moved classes from `tests/test_trend.py`**

Run:

```sh
rg -n "TrendModelSplitFrameTests|TrendModelSplitWriteTests" tests/test_trend.py
```

Expected: no matches.

- [ ] **Step 5: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_splits.py tests/test_trend.py -q
```

Expected: moved split tests and remaining trend tests pass.

- [ ] **Step 6: Run full pytest and commit**

Run:

```sh
uv run pytest
git diff -- tests/test_trend_splits.py tests/test_trend.py
git add tests/test_trend_splits.py tests/test_trend.py
git commit -m "test: 拆分趋势切分测试"
```

Expected: pytest passes before commit, then commit succeeds.

---

### Task 10: Split Trend Training Tests

**Files:**
- Create: `tests/test_trend_training.py`
- Modify: `tests/test_trend.py`

- [ ] **Step 1: Create target file imports**

Create `tests/test_trend_training.py` with:

```python
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import sample_trend_model_samples_for_split
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
from fashion_trend.trend import (
    TREND_MODEL_PREDICTION_COLUMNS,
    build_trend_model_split_frames,
    validate_trend_model_predictions,
    write_trend_parquet,
)
```

- [ ] **Step 2: Move and rename training class**

Move:

```python
class LastWeekBaselineTests(unittest.TestCase):
```

Rename to:

```python
class TestTrendTraining:
```

- [ ] **Step 3: Convert parameterized negative value tests**

Convert the non-finite growth lag loop to:

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

Convert unsafe artifact path checks to:

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

Keep the current payload and path expectations from the existing tests when converting.

- [ ] **Step 4: Convert CLI tests without changing production behavior**

Keep `importlib.import_module("10_train_trend_model")`. Convert assertions:

```python
train_model = importlib.import_module("10_train_trend_model")

assert train_model.main(["--unknown"]) == 2
assert train_model.main(["--model", "unknown_model"]) == 1
```

When tests monkeypatch module globals, keep the same monkeypatching mechanism already used in the current file and change only assertion style.

- [ ] **Step 5: Remove moved class from `tests/test_trend.py`**

Run:

```sh
rg -n "LastWeekBaselineTests" tests/test_trend.py
```

Expected: no matches.

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_training.py tests/test_trend.py -q
```

Expected: moved training tests and remaining trend tests pass.

- [ ] **Step 7: Run full pytest and commit**

Run:

```sh
uv run pytest
git diff -- tests/test_trend_training.py tests/test_trend.py
git add tests/test_trend_training.py tests/test_trend.py
git commit -m "test: 拆分趋势训练测试"
```

Expected: pytest passes before commit, then commit succeeds.

---

### Task 11: Split Trend Evaluation Tests and Remove Monolithic File

**Files:**
- Create: `tests/test_trend_evaluation.py`
- Delete: `tests/test_trend.py`

- [ ] **Step 1: Create target file imports**

Create `tests/test_trend_evaluation.py` with:

```python
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import pytest

from conftest import sample_trend_predictions_for_evaluation
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
from fashion_trend.trend import TREND_MODEL_PREDICTION_COLUMNS
```

- [ ] **Step 2: Move and rename evaluation class**

Move:

```python
class TrendEvaluationTests(unittest.TestCase):
```

Rename to:

```python
class TestTrendEvaluation:
```

- [ ] **Step 3: Convert model-name subTest to parametrize**

Use:

```python
@pytest.mark.parametrize("model_name", ["../escape", "/tmp/escape"])
def test_derive_trend_metric_output_paths_rejects_unsafe_model_name(
    model_name: str,
) -> None:
    with pytest.raises(ValueError, match="model_name|模型名"):
        derive_trend_metric_output_paths(model_name)
```

Use the exact unsafe model names currently present in the test if they differ.

- [ ] **Step 4: Convert tmp paths, metrics assertions, and CLI assertions**

Use:

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

Keep metric precision assertions as direct `math.isclose` checks:

```python
assert math.isclose(metrics["mae"], 0.5666666667, rel_tol=1e-9)
assert metrics["precision_at_k"]["2"] == 0.5
```

Keep `importlib.import_module("11_eval_trend_model")` for CLI tests and convert exit-code checks to direct assertions.

- [ ] **Step 5: Delete `tests/test_trend.py`**

After moving `TrendEvaluationTests`, `tests/test_trend.py` should contain no test classes or local helpers. Delete the file.

Run:

```sh
test ! -e tests/test_trend.py
```

Expected: command exits with status 0.

- [ ] **Step 6: Run evaluation tests**

Run:

```sh
uv run pytest tests/test_trend_evaluation.py -q
```

Expected: moved evaluation tests pass.

- [ ] **Step 7: Run full pytest**

Run:

```sh
uv run pytest
```

Expected: all tests pass and total collected tests remains at least 147.

- [ ] **Step 8: Commit**

Run:

```sh
git diff -- tests/test_trend_evaluation.py tests/test_trend.py
git add tests/test_trend_evaluation.py
git rm tests/test_trend.py
git commit -m "test: 拆分趋势评价测试"
```

Expected: commit succeeds.

---

### Task 12: Final pytest Cleanup and Verification

**Files:**
- Modify: `README.md` only if the final verification section needs wording cleanup
- Modify: `docs/superpowers/specs/2026-05-06-pytest-migration-design.md` only if implementation uncovered a required correction to the approved spec

- [ ] **Step 1: Check for unittest leftovers in active tests**

Run:

```sh
rg -n "unittest|TestCase|self\\.assert|TemporaryDirectory|subTest" tests
```

Expected: no matches.

- [ ] **Step 2: Check formal docs for old test command**

Run:

```sh
rg -n "当前测试使用标准库|unittest discover|uv run python -m unittest discover" README.md docs/gpt-research/implementation-plan.md
```

Expected: no matches.

- [ ] **Step 3: Confirm target test file layout**

Run:

```sh
find tests -maxdepth 1 -type f | sort
```

Expected output includes:

```text
tests/conftest.py
tests/test_articles_clean.py
tests/test_attribute_graph.py
tests/test_trend_article_sales.py
tests/test_trend_attribute_heat.py
tests/test_trend_evaluation.py
tests/test_trend_samples.py
tests/test_trend_splits.py
tests/test_trend_targets.py
tests/test_trend_training.py
```

Expected output does not include `tests/test_trend.py`.

- [ ] **Step 4: Run full pytest**

Run:

```sh
uv run pytest
```

Expected: all tests pass and total collected tests is at least 147.

- [ ] **Step 5: Run syntax compile**

Run:

```sh
uv run python -m py_compile \
  tests/conftest.py \
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

Expected: command exits successfully with no output.

- [ ] **Step 6: Review full branch diff**

Run:

```sh
git diff master...HEAD --stat
git diff master...HEAD -- pyproject.toml README.md tests docs/superpowers/specs docs/superpowers/plans
```

Expected: diff is limited to pytest dependency/config, README verification docs, committed spec/plan docs, and test migration/split files.

- [ ] **Step 7: Commit final cleanup if needed**

If Step 1 or Step 2 required documentation or cleanup edits, commit them:

```sh
git add README.md docs/superpowers/specs/2026-05-06-pytest-migration-design.md tests
git commit -m "test: 收口 pytest 迁移验证"
```

Expected: commit succeeds only when there were cleanup edits. If no files changed, do not create an empty commit.

---

## Execution Notes

- Use `apply_patch` for manual edits.
- Do not use generated codemods or one-off rewrite scripts.
- If `uv run ...` fails with `Failed to initialize cache at /Users/ghstlnx/.cache/uv`, rerun the same command with approved elevated `uv run` permissions. Treat that as a local permission issue, not as a test failure.
- Before every commit, run the task-specific pytest command, inspect `git diff`, and commit only the files listed in that task.
- Do not weaken assertions, add skips, or hide exceptions to make pytest pass.
