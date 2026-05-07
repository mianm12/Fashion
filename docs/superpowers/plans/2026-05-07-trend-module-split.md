# 趋势模块拆分实施计划

> **给 agentic workers：** 必须使用子技能 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐步执行本计划。所有步骤使用 checkbox（`- [ ]`）语法跟踪状态。

**目标：** 将过大的 `src/fashion_trend/trend.py` 拆成职责清晰的 `fashion_trend.trend` 子包，同时保持行为不变和旧导入路径兼容。

**架构：** 先把 `fashion_trend.trend` 从单文件模块转换为包，并用 `trend/__init__.py` 作为兼容 facade。随后把共享列契约、通用校验、IO、各阶段构建逻辑、时间切分和预测契约拆到专门子模块，最后把生产代码导入迁移到这些子模块。

**技术栈：** Python 3.12、pandas、numpy、pyarrow、pytest、uv、git。

---

## 文件结构

新增文件：

```text
src/fashion_trend/trend/__init__.py
src/fashion_trend/trend/schema.py
src/fashion_trend/trend/validation.py
src/fashion_trend/trend/io.py
src/fashion_trend/trend/article_sales.py
src/fashion_trend/trend/attribute_heat.py
src/fashion_trend/trend/targets.py
src/fashion_trend/trend/samples.py
src/fashion_trend/trend/splits.py
src/fashion_trend/trend/predictions.py
tests/test_trend_package_compat.py
```

包结构就位后移除旧大文件：

```text
src/fashion_trend/trend.py
```

迁移这些生产代码导入：

```text
src/05_compute_article_week_sales.py
src/06_compute_attribute_week_heat.py
src/07_build_trend_targets.py
src/08_build_trend_model_samples.py
src/09_split_trend_model_samples.py
src/10_train_trend_model.py
src/fashion_trend/training.py
src/fashion_trend/evaluation.py
src/fashion_trend/models/last_week.py
src/fashion_trend/models/moving_average.py
README.md
```

`tests/test_trend_*.py` 继续按流水线阶段组织。只有为了覆盖新模块直接导入时，才调整测试导入。

## 任务 1：把 `trend.py` 转成包 facade

**文件：**
- 新增：`tests/test_trend_package_compat.py`
- 移动：`src/fashion_trend/trend.py` -> `src/fashion_trend/trend/__init__.py`

- [ ] **步骤 1：写一个失败的包兼容测试**

创建 `tests/test_trend_package_compat.py`：

```python
from __future__ import annotations

import importlib
from pathlib import Path


def test_trend_entrypoint_is_package_facade() -> None:
    trend = importlib.import_module("fashion_trend.trend")

    assert Path(trend.__file__).name == "__init__.py"
    assert hasattr(trend, "build_article_week_sales_frame")
    assert hasattr(trend, "TREND_MODEL_PREDICTION_COLUMNS")
```

- [ ] **步骤 2：运行测试，确认当前会失败**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py -q
```

预期：失败，因为当前 `fashion_trend.trend.__file__` 仍指向 `trend.py`。

- [ ] **步骤 3：把现有模块移动为包入口**

在仓库根目录运行：

```sh
git mv src/fashion_trend/trend.py src/fashion_trend/_trend_compat.py
mkdir -p src/fashion_trend/trend
git mv src/fashion_trend/_trend_compat.py src/fashion_trend/trend/__init__.py
```

本步骤不修改任何函数体。

- [ ] **步骤 4：验证包兼容和既有行为**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py tests/test_trend_article_sales.py -q
```

预期：通过。

- [ ] **步骤 5：提交包转换**

运行：

```sh
git status --short
git diff --stat
git add src/fashion_trend/trend/__init__.py tests/test_trend_package_compat.py
git commit -m "refactor(trend): 转为趋势子包"
```

预期：提交只包含文件移动和新的兼容测试。

## 任务 2：拆分 schema 和通用 validation

**文件：**
- 新增：`src/fashion_trend/trend/schema.py`
- 新增：`src/fashion_trend/trend/validation.py`
- 修改：`src/fashion_trend/trend/__init__.py`
- 修改：`tests/test_trend_package_compat.py`

- [ ] **步骤 1：补直接导入 schema 与 validation 的测试**

追加到 `tests/test_trend_package_compat.py`：

```python
import pandas as pd
import pytest


def test_trend_schema_module_exports_core_contracts() -> None:
    from fashion_trend.trend.schema import (
        ARTICLE_WEEK_SALES_COLUMNS,
        TREND_MODEL_PREDICTION_COLUMNS,
        TREND_MODEL_SPLIT_VALUES,
    )

    assert ARTICLE_WEEK_SALES_COLUMNS == (
        "week_id",
        "article_id",
        "sales_cnt",
        "sales_user_cnt",
        "sales_amount",
    )
    assert TREND_MODEL_SPLIT_VALUES == ("train", "valid", "test")
    assert TREND_MODEL_PREDICTION_COLUMNS[:6] == (
        "week_id",
        "attr_id",
        "attr_type",
        "attr_value",
        "model_name",
        "split",
    )


def test_trend_validation_module_rejects_missing_columns() -> None:
    from fashion_trend.trend.validation import validate_required_columns

    with pytest.raises(ValueError, match="missing_col"):
        validate_required_columns(
            pd.DataFrame({"present": [1]}).columns.tolist(),
            ("present", "missing_col"),
            source_name="测试表",
        )
```

- [ ] **步骤 2：运行直接导入测试，确认当前会失败**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py::test_trend_schema_module_exports_core_contracts tests/test_trend_package_compat.py::test_trend_validation_module_rejects_missing_columns -q
```

预期：失败，并报 `ModuleNotFoundError: fashion_trend.trend.schema`。

- [ ] **步骤 3：创建 `schema.py` 并移动现有常量**

把以下定义从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/schema.py`，保持值和顺序不变：

```python
WEEKLY_TRANSACTION_COLUMNS
ARTICLE_WEEK_SALES_COLUMNS
ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS
ATTRIBUTE_WEEK_HEAT_COLUMNS
ATTRIBUTE_WEEK_TARGET_COLUMNS
ATTRIBUTE_HIERARCHY_EDGE_COLUMNS
TREND_MODEL_SAMPLE_COLUMNS
TREND_MODEL_SPLIT_VALUES
TREND_MODEL_PREDICTION_COLUMNS
TREND_MODEL_PRED_SHARE_GROUP_COLUMNS
TREND_MODEL_SHARE_TOLERANCE
TREND_MODEL_SPLIT_COLUMNS
ARTICLE_WEEK_SALES_DTYPES
ATTRIBUTE_WEEK_HEAT_DTYPES
ATTRIBUTE_WEEK_TARGET_DTYPES
ATTRIBUTE_HIERARCHY_EDGE_DTYPES
ARTICLE_ATTRIBUTE_EDGE_HEAT_DTYPES
ATTRIBUTE_NODE_HEAT_COLUMNS
ATTRIBUTE_NODE_HEAT_DTYPES
```

`schema.py` 只需要这个文件头：

```python
from __future__ import annotations
```

- [ ] **步骤 4：创建 `validation.py` 并移动通用校验函数**

把以下函数从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/validation.py`，不改行为：

```python
validate_required_columns
validate_no_missing_values
validate_unique_key
validate_non_negative_values
validate_positive_values
```

模块头部使用：

```python
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
```

- [ ] **步骤 5：在 `trend/__init__.py` re-export schema 和 validation**

在 `src/fashion_trend/trend/__init__.py` 顶部 `from __future__ import annotations` 后加入：

```python
from fashion_trend.trend.schema import (
    ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS,
    ARTICLE_ATTRIBUTE_EDGE_HEAT_DTYPES,
    ARTICLE_WEEK_SALES_COLUMNS,
    ARTICLE_WEEK_SALES_DTYPES,
    ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
    ATTRIBUTE_HIERARCHY_EDGE_DTYPES,
    ATTRIBUTE_NODE_HEAT_COLUMNS,
    ATTRIBUTE_NODE_HEAT_DTYPES,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_DTYPES,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_DTYPES,
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_PRED_SHARE_GROUP_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
    TREND_MODEL_SHARE_TOLERANCE,
    TREND_MODEL_SPLIT_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
    WEEKLY_TRANSACTION_COLUMNS,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
```

删除 `trend/__init__.py` 中已移动的重复常量和通用校验函数定义。

- [ ] **步骤 6：确认 `trend/__init__.py` 剩余函数仍能引用这些符号**

`trend/__init__.py` 里尚未移动的函数会继续使用这些常量和校验函数。确认它们通过步骤 5 的 import 正常解析，不新增本地重复定义。

- [ ] **步骤 7：验证 schema 和 validation 拆分**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py tests/test_trend_article_sales.py tests/test_trend_attribute_heat.py -q
```

预期：通过。

- [ ] **步骤 8：提交 schema 和 validation 拆分**

运行：

```sh
git status --short
git diff --stat
git add src/fashion_trend/trend/__init__.py src/fashion_trend/trend/schema.py src/fashion_trend/trend/validation.py tests/test_trend_package_compat.py
git commit -m "refactor(trend): 拆分共享契约与校验"
```

预期：提交只包含 schema、validation、facade 和包兼容测试改动。

## 任务 3：拆分 IO 和商品周销量阶段

**文件：**
- 新增：`src/fashion_trend/trend/io.py`
- 新增：`src/fashion_trend/trend/article_sales.py`
- 修改：`src/fashion_trend/trend/__init__.py`
- 修改：`tests/test_trend_package_compat.py`

- [ ] **步骤 1：补 IO 和商品周销量模块的直接导入测试**

追加到 `tests/test_trend_package_compat.py`：

```python
def test_article_sales_and_io_modules_export_stage_api() -> None:
    from fashion_trend.trend.article_sales import (
        build_article_week_sales_frame,
        read_article_week_sales,
        read_weekly_transactions,
        validate_article_week_sales,
    )
    from fashion_trend.trend.io import write_json, write_trend_csv, write_trend_parquet

    assert callable(read_weekly_transactions)
    assert callable(build_article_week_sales_frame)
    assert callable(validate_article_week_sales)
    assert callable(read_article_week_sales)
    assert callable(write_json)
    assert callable(write_trend_csv)
    assert callable(write_trend_parquet)
```

- [ ] **步骤 2：运行直接导入测试，确认当前会失败**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py::test_article_sales_and_io_modules_export_stage_api -q
```

预期：失败，并报 `ModuleNotFoundError: fashion_trend.trend.article_sales`。

- [ ] **步骤 3：创建 `io.py`**

把以下函数从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/io.py`：

```python
remove_file_if_exists
write_json
write_trend_csv
write_trend_parquet
```

模块头部使用：

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
```

保留现有临时文件写入和失败清理行为。

- [ ] **步骤 4：创建 `article_sales.py`**

把以下函数从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/article_sales.py`：

```python
read_weekly_transactions
build_article_week_sales_frame
validate_article_week_sales
read_article_week_sales
```

模块头部和导入使用：

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fashion_trend.trend.schema import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ARTICLE_WEEK_SALES_DTYPES,
    WEEKLY_TRANSACTION_COLUMNS,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
```

- [ ] **步骤 5：在 `trend/__init__.py` re-export IO 和商品周销量 API**

加入：

```python
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    read_weekly_transactions,
    validate_article_week_sales,
)
from fashion_trend.trend.io import (
    remove_file_if_exists,
    write_json,
    write_trend_csv,
    write_trend_parquet,
)
```

删除 `trend/__init__.py` 中已移动的重复函数。

- [ ] **步骤 6：验证 IO 和商品周销量拆分**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py tests/test_trend_article_sales.py tests/test_trend_splits.py -q
```

预期：通过。

- [ ] **步骤 7：提交 IO 和商品周销量拆分**

运行：

```sh
git status --short
git diff --stat
git add src/fashion_trend/trend/__init__.py src/fashion_trend/trend/io.py src/fashion_trend/trend/article_sales.py tests/test_trend_package_compat.py
git commit -m "refactor(trend): 拆分IO与商品周销量"
```

预期：提交只包含 IO、商品周销量、facade 和直接导入测试改动。

## 任务 4：拆分属性热度和趋势标签阶段

**文件：**
- 新增：`src/fashion_trend/trend/attribute_heat.py`
- 新增：`src/fashion_trend/trend/targets.py`
- 修改：`src/fashion_trend/trend/__init__.py`
- 修改：`tests/test_trend_package_compat.py`

- [ ] **步骤 1：补热度和标签模块的直接导入测试**

追加到 `tests/test_trend_package_compat.py`：

```python
def test_heat_and_target_modules_export_stage_api() -> None:
    from fashion_trend.trend.attribute_heat import (
        build_attribute_week_heat_frame,
        read_article_attribute_edges,
        read_attribute_nodes,
        read_attribute_week_heat,
        validate_attribute_week_heat,
    )
    from fashion_trend.trend.targets import (
        build_attribute_week_target_frame,
        read_attribute_week_target,
        validate_attribute_week_target,
        validate_attribute_week_target_matches_heat,
    )

    assert callable(read_article_attribute_edges)
    assert callable(read_attribute_nodes)
    assert callable(read_attribute_week_heat)
    assert callable(build_attribute_week_heat_frame)
    assert callable(validate_attribute_week_heat)
    assert callable(read_attribute_week_target)
    assert callable(build_attribute_week_target_frame)
    assert callable(validate_attribute_week_target)
    assert callable(validate_attribute_week_target_matches_heat)
```

- [ ] **步骤 2：运行直接导入测试，确认当前会失败**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py::test_heat_and_target_modules_export_stage_api -q
```

预期：失败，并报 `ModuleNotFoundError: fashion_trend.trend.attribute_heat`。

- [ ] **步骤 3：创建 `attribute_heat.py`**

把以下函数从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/attribute_heat.py`：

```python
validate_article_attribute_edges_for_heat
validate_all_sales_articles_have_attribute_edges
read_attribute_week_heat
read_article_attribute_edges
validate_attribute_nodes_for_heat
read_attribute_nodes
validate_attribute_edge_node_metadata_consistency
build_attribute_week_heat_frame
validate_attribute_week_heat
```

模块头部和导入使用：

```python
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fashion_trend.trend.article_sales import validate_article_week_sales
from fashion_trend.trend.schema import (
    ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS,
    ARTICLE_ATTRIBUTE_EDGE_HEAT_DTYPES,
    ATTRIBUTE_NODE_HEAT_COLUMNS,
    ATTRIBUTE_NODE_HEAT_DTYPES,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_DTYPES,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
```

- [ ] **步骤 4：创建 `targets.py`**

把以下函数从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/targets.py`：

```python
read_attribute_week_target
build_attribute_week_target_frame
validate_attribute_week_target
validate_attribute_week_target_matches_heat
```

模块头部和导入使用：

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fashion_trend.trend.attribute_heat import validate_attribute_week_heat
from fashion_trend.trend.schema import (
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_DTYPES,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
```

- [ ] **步骤 5：在 `trend/__init__.py` re-export 热度和标签 API**

加入：

```python
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    read_article_attribute_edges,
    read_attribute_nodes,
    read_attribute_week_heat,
    validate_all_sales_articles_have_attribute_edges,
    validate_article_attribute_edges_for_heat,
    validate_attribute_edge_node_metadata_consistency,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    read_attribute_week_target,
    validate_attribute_week_target,
    validate_attribute_week_target_matches_heat,
)
```

删除 `trend/__init__.py` 中已移动的重复函数。

- [ ] **步骤 6：验证热度和标签拆分**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py tests/test_trend_attribute_heat.py tests/test_trend_targets.py tests/test_trend_samples.py -q
```

预期：通过。

- [ ] **步骤 7：提交热度和标签拆分**

运行：

```sh
git status --short
git diff --stat
git add src/fashion_trend/trend/__init__.py src/fashion_trend/trend/attribute_heat.py src/fashion_trend/trend/targets.py tests/test_trend_package_compat.py
git commit -m "refactor(trend): 拆分热度与标签阶段"
```

预期：提交只包含热度、标签、facade 和直接导入测试改动。

## 任务 5：拆分样本和时间切分阶段

**文件：**
- 新增：`src/fashion_trend/trend/samples.py`
- 新增：`src/fashion_trend/trend/splits.py`
- 修改：`src/fashion_trend/trend/__init__.py`
- 修改：`tests/test_trend_package_compat.py`

- [ ] **步骤 1：补样本和 split 模块的直接导入测试**

追加到 `tests/test_trend_package_compat.py`：

```python
def test_sample_and_split_modules_export_stage_api() -> None:
    from fashion_trend.trend.samples import (
        build_attribute_graph_features_frame,
        build_trend_model_samples_frame,
        read_attribute_hierarchy_edges,
        validate_trend_model_samples,
    )
    from fashion_trend.trend.splits import (
        build_trend_model_split_frames,
        build_trend_model_split_metadata,
        read_trend_model_split,
        validate_trend_model_split_frame,
        validate_trend_model_split_frames,
    )

    assert callable(read_attribute_hierarchy_edges)
    assert callable(build_attribute_graph_features_frame)
    assert callable(build_trend_model_samples_frame)
    assert callable(validate_trend_model_samples)
    assert callable(build_trend_model_split_frames)
    assert callable(validate_trend_model_split_frames)
    assert callable(validate_trend_model_split_frame)
    assert callable(build_trend_model_split_metadata)
    assert callable(read_trend_model_split)
```

- [ ] **步骤 2：运行直接导入测试，确认当前会失败**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py::test_sample_and_split_modules_export_stage_api -q
```

预期：失败，并报 `ModuleNotFoundError: fashion_trend.trend.samples`。

- [ ] **步骤 3：创建 `samples.py`**

把以下函数从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/samples.py`：

```python
read_attribute_hierarchy_edges
build_attribute_graph_features_frame
build_trend_model_samples_frame
validate_trend_model_samples
```

模块头部和导入使用：

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fashion_trend.trend.attribute_heat import (
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.schema import (
    ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
    ATTRIBUTE_HIERARCHY_EDGE_DTYPES,
    TREND_MODEL_SAMPLE_COLUMNS,
)
from fashion_trend.trend.targets import validate_attribute_week_target_matches_heat
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
```

- [ ] **步骤 4：创建 `splits.py`**

把以下函数从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/splits.py`：

```python
build_trend_model_split_frames
validate_trend_model_split_frames
validate_trend_model_split_frame
build_trend_model_split_metadata
read_trend_model_split
```

模块头部和导入使用：

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion_trend.trend.samples import validate_trend_model_samples
from fashion_trend.trend.schema import (
    TREND_MODEL_SPLIT_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_key,
)
```

- [ ] **步骤 5：在 `trend/__init__.py` re-export 样本和 split API**

加入：

```python
from fashion_trend.trend.samples import (
    build_attribute_graph_features_frame,
    build_trend_model_samples_frame,
    read_attribute_hierarchy_edges,
    validate_trend_model_samples,
)
from fashion_trend.trend.splits import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    read_trend_model_split,
    validate_trend_model_split_frame,
    validate_trend_model_split_frames,
)
```

删除 `trend/__init__.py` 中已移动的重复函数。

- [ ] **步骤 6：验证样本和 split 拆分**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py tests/test_trend_samples.py tests/test_trend_splits.py tests/test_trend_training.py -q
```

预期：通过。

- [ ] **步骤 7：提交样本和时间切分拆分**

运行：

```sh
git status --short
git diff --stat
git add src/fashion_trend/trend/__init__.py src/fashion_trend/trend/samples.py src/fashion_trend/trend/splits.py tests/test_trend_package_compat.py
git commit -m "refactor(trend): 拆分样本与时间切分"
```

预期：提交只包含样本、split、facade 和直接导入测试改动。

## 任务 6：拆分预测契约

**文件：**
- 新增：`src/fashion_trend/trend/predictions.py`
- 修改：`src/fashion_trend/trend/__init__.py`
- 修改：`tests/test_trend_package_compat.py`

- [ ] **步骤 1：补预测契约模块的直接导入测试**

追加到 `tests/test_trend_package_compat.py`：

```python
def test_prediction_module_exports_contract_api() -> None:
    from fashion_trend.trend.predictions import (
        derive_normalized_pred_share_t1,
        validate_pred_share_t1_distribution,
        validate_trend_model_predictions,
    )

    assert callable(validate_trend_model_predictions)
    assert callable(derive_normalized_pred_share_t1)
    assert callable(validate_pred_share_t1_distribution)
```

- [ ] **步骤 2：运行直接导入测试，确认当前会失败**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py::test_prediction_module_exports_contract_api -q
```

预期：失败，并报 `ModuleNotFoundError: fashion_trend.trend.predictions`。

- [ ] **步骤 3：创建 `predictions.py`**

把以下函数从 `trend/__init__.py` 移动到 `src/fashion_trend/trend/predictions.py`：

```python
validate_trend_model_predictions
derive_normalized_pred_share_t1
validate_pred_share_t1_distribution
```

模块头部和导入使用：

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_PRED_SHARE_GROUP_COLUMNS,
    TREND_MODEL_SHARE_TOLERANCE,
    TREND_MODEL_SPLIT_VALUES,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_key,
)
```

- [ ] **步骤 4：在 `trend/__init__.py` re-export 预测 API**

加入：

```python
from fashion_trend.trend.predictions import (
    derive_normalized_pred_share_t1,
    validate_pred_share_t1_distribution,
    validate_trend_model_predictions,
)
```

删除 `trend/__init__.py` 中已移动的重复函数。

- [ ] **步骤 5：确认 facade 只做 re-export**

打开 `src/fashion_trend/trend/__init__.py`。它应只包含：

```python
from __future__ import annotations
```

以及来自 `fashion_trend.trend.*` 的 import block。它不应包含 `def `、`class `、DataFrame builder 逻辑、文件写入逻辑或校验函数体。

- [ ] **步骤 6：验证预测契约拆分**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py tests/test_trend_training.py tests/test_trend_evaluation.py -q
```

预期：通过。

- [ ] **步骤 7：提交预测契约拆分**

运行：

```sh
git status --short
git diff --stat
git add src/fashion_trend/trend/__init__.py src/fashion_trend/trend/predictions.py tests/test_trend_package_compat.py
git commit -m "refactor(trend): 拆分预测契约"
```

预期：提交只包含预测契约、facade 和直接导入测试改动。

## 任务 7：迁移生产代码导入并补文档边界

**文件：**
- 修改：`src/05_compute_article_week_sales.py`
- 修改：`src/06_compute_attribute_week_heat.py`
- 修改：`src/07_build_trend_targets.py`
- 修改：`src/08_build_trend_model_samples.py`
- 修改：`src/09_split_trend_model_samples.py`
- 修改：`src/10_train_trend_model.py`
- 修改：`src/fashion_trend/training.py`
- 修改：`src/fashion_trend/evaluation.py`
- 修改：`src/fashion_trend/models/last_week.py`
- 修改：`src/fashion_trend/models/moving_average.py`
- 修改：`README.md`

- [ ] **步骤 1：替换 article sales CLI 导入**

在 `src/05_compute_article_week_sales.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_weekly_transactions,
    validate_article_week_sales,
)
from fashion_trend.trend.io import write_trend_csv
```

- [ ] **步骤 2：替换 attribute heat CLI 导入**

在 `src/06_compute_attribute_week_heat.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.article_sales import (
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    read_article_attribute_edges,
    read_attribute_nodes,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.io import write_trend_csv
```

- [ ] **步骤 3：替换 target CLI 导入**

在 `src/07_build_trend_targets.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.attribute_heat import (
    read_attribute_nodes,
    read_attribute_week_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.io import write_trend_csv
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    validate_attribute_week_target,
)
```

- [ ] **步骤 4：替换 sample CLI 导入**

在 `src/08_build_trend_model_samples.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.attribute_heat import (
    read_attribute_nodes,
    read_attribute_week_heat,
)
from fashion_trend.trend.io import write_trend_parquet
from fashion_trend.trend.samples import (
    build_trend_model_samples_frame,
    read_attribute_hierarchy_edges,
    validate_trend_model_samples,
)
from fashion_trend.trend.targets import read_attribute_week_target
```

- [ ] **步骤 5：替换 split CLI 导入**

在 `src/09_split_trend_model_samples.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.io import write_json, write_trend_parquet
from fashion_trend.trend.splits import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    validate_trend_model_split_frames,
)
```

- [ ] **步骤 6：替换训练 CLI 导入**

在 `src/10_train_trend_model.py` 中，从 schema 导入 split 常量：

```python
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_VALUES
```

- [ ] **步骤 7：替换 training runner 导入**

在 `src/fashion_trend/training.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.io import remove_file_if_exists, write_json, write_trend_csv
from fashion_trend.trend.predictions import validate_trend_model_predictions
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_VALUES
from fashion_trend.trend.splits import read_trend_model_split
```

- [ ] **步骤 8：替换 evaluation 导入**

在 `src/fashion_trend/evaluation.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.io import write_json
from fashion_trend.trend.predictions import validate_pred_share_t1_distribution
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_key,
)
```

- [ ] **步骤 9：替换 baseline 模型导入**

在 `src/fashion_trend/models/last_week.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.predictions import (
    derive_normalized_pred_share_t1,
    validate_trend_model_predictions,
)
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)
from fashion_trend.trend.validation import validate_required_columns
```

在 `src/fashion_trend/models/moving_average.py` 中，把 `from fashion_trend.trend import (...)` block 替换为：

```python
from fashion_trend.trend.predictions import (
    derive_normalized_pred_share_t1,
    validate_trend_model_predictions,
)
from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)
from fashion_trend.trend.validation import validate_required_columns
```

- [ ] **步骤 10：在 README 记录新实现边界**

在 `README.md` 的 `## 验证` 前加入：

```markdown
趋势共享实现位于 `src/fashion_trend/trend/` 子包。`article_sales.py`、`attribute_heat.py`、`targets.py`、`samples.py`、`splits.py` 和 `predictions.py` 分别对应当前趋势流水线阶段与训练/评价共享契约；`trend/__init__.py` 只保留旧导入兼容。
```

- [ ] **步骤 11：验证生产代码不再从 facade 导入**

运行：

```sh
rg -n "from fashion_trend\.trend import" src
```

预期：无输出。

- [ ] **步骤 12：运行生产导入相关测试**

运行：

```sh
uv run pytest tests/test_trend_package_compat.py tests/test_trend_training.py tests/test_trend_evaluation.py -q
```

预期：通过。

- [ ] **步骤 13：提交生产导入迁移**

运行：

```sh
git status --short
git diff --stat
git add src/05_compute_article_week_sales.py src/06_compute_attribute_week_heat.py src/07_build_trend_targets.py src/08_build_trend_model_samples.py src/09_split_trend_model_samples.py src/10_train_trend_model.py src/fashion_trend/training.py src/fashion_trend/evaluation.py src/fashion_trend/models/last_week.py src/fashion_trend/models/moving_average.py README.md
git commit -m "refactor(trend): 迁移生产导入边界"
```

预期：提交包含生产导入改写和 README 一段说明。

## 任务 8：最终验证

**文件：**
- 验证：任务 1-7 涉及的所有文件

- [ ] **步骤 1：编译新包和入口脚本**

运行：

```sh
uv run python -m py_compile \
  src/fashion_trend/trend/__init__.py \
  src/fashion_trend/trend/schema.py \
  src/fashion_trend/trend/validation.py \
  src/fashion_trend/trend/io.py \
  src/fashion_trend/trend/article_sales.py \
  src/fashion_trend/trend/attribute_heat.py \
  src/fashion_trend/trend/targets.py \
  src/fashion_trend/trend/samples.py \
  src/fashion_trend/trend/splits.py \
  src/fashion_trend/trend/predictions.py \
  src/05_compute_article_week_sales.py \
  src/06_compute_attribute_week_heat.py \
  src/07_build_trend_targets.py \
  src/08_build_trend_model_samples.py \
  src/09_split_trend_model_samples.py \
  src/10_train_trend_model.py \
  src/11_eval_trend_model.py
```

预期：命令退出码为 0，没有语法错误。

- [ ] **步骤 2：运行全量测试**

运行：

```sh
uv run pytest
```

预期：全部测试通过。

- [ ] **步骤 3：确认 facade 只被兼容场景使用**

运行：

```sh
rg -n "from fashion_trend\.trend import" src tests
```

预期：`src/` 下无匹配；`tests/` 下只有明确验证兼容入口的测试可以保留匹配。

- [ ] **步骤 4：确认旧大文件已不存在**

运行：

```sh
test ! -f src/fashion_trend/trend.py
test -d src/fashion_trend/trend
find src/fashion_trend/trend -maxdepth 1 -type f | sort
```

预期：`trend.py` 不存在，`trend/` 目录存在，输出列出拆分后的包文件。

- [ ] **步骤 5：检查 diff 干净度**

运行：

```sh
git diff --check
git status --short
```

预期：`git diff --check` 退出码为 0。完成任务提交后，`git status --short` 不再显示未提交实现改动。
