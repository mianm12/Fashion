# 业务域驱动代码架构实施计划

> **给 agentic workers：** 必须使用子技能：执行本计划时使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐任务推进。步骤使用 checkbox（`- [ ]`）语法用于跟踪。

**目标：** 将项目内部组织方式从脚本编号驱动迁移为业务域驱动，同时保持编号 CLI 命令和既有产物契约稳定。

**架构：** 编号脚本继续作为可读的业务流程索引。业务包成为计算事实来源。`foundation` 是唯一共享基础层，并且必须保持无业务语义。

**技术栈：** Python 3.10-3.12、pandas、numpy、pyarrow、pytest、isort、setuptools 从 `src` 发现包。

---

## 范围检查

本计划只实施架构迁移，不新增 LightGBM，不实现推荐重排序，不实现报告生成逻辑，也不修改现有算法。计划会建立 `recommendation` 和 `reports` 的包边界，以便后续约束依赖方向，但不会在本轮实现它们的业务功能。

迁移拆成多个可独立提交的阶段：

1. 添加架构边界测试和业务包骨架。
2. 提取 `foundation`。
3. 迁移 `datasets`、`transactions` 和 `catalog`。
4. 收敛确定性 `trend` 流水线。
5. 将趋势模型、训练和评价移动到 `trend` 域内。
6. 刷新编号 CLI，使其成为业务流程索引。
7. 同步文档并执行完整验证。

## 目标文件结构

### 新增或移动的包

- `src/fashion_trend/foundation/`
  - `__init__.py`：包标记。
  - `paths.py`：项目路径、artifact 目录、`PATH`、切分常量、competition slug。
  - `logging.py`：当前根包 `log.py` 中的日志工具。
  - `io.py`：通用原子 JSON/CSV/Parquet/binary 写入和文件删除。
  - `dataframe.py`：通用 DataFrame 校验原语。
  - `artifacts.py`：趋势训练和评价使用的 artifact 路径与 model name 安全检查。
- `src/fashion_trend/datasets/`
  - `__init__.py`：包标记。
  - `download.py`：从 `00_download_data.py` 下沉的 Kaggle 下载、安全解压、跳过下载逻辑。
  - `profile.py`：供 `01_data_check.py` 使用的原始数据存在性和基础 profile 辅助函数。
- `src/fashion_trend/transactions/`
  - `__init__.py`：包标记。
  - `weekly.py`：从 `02_build_weekly_transactions.py` 下沉的周级交易表流水线。
- `src/fashion_trend/catalog/`
  - `__init__.py`：包标记。
  - `articles.py`：从根包 `articles.py` 拆出的商品清洗和商品标识规范化。
  - `graph.py`：从根包 `articles.py` 拆出的属性图节点、边构建和发布逻辑。
- `src/fashion_trend/trend/`
  - 保留聚焦的确定性模块：`schema.py`、`article_sales.py`、`attribute_heat.py`、`targets.py`、`samples.py`、`splits.py`、`predictions.py`。
  - 移除 `trend/__init__.py` 的兼容 re-export 行为，只保留简短包 docstring，不再作为 facade API。
  - 将实验层模块移入该包：
    - `trend/models/base.py`
    - `trend/models/last_week.py`
    - `trend/models/moving_average.py`
    - `trend/models/registry.py`
    - `trend/training.py`
    - `trend/evaluation.py`
- `src/fashion_trend/recommendation/__init__.py`：推荐域包标记，用于依赖边界约束。
- `src/fashion_trend/reports/__init__.py`：报告域包标记，用于依赖边界约束。

### 删除的历史根包模块

- 采用 `foundation.paths` 后删除 `src/fashion_trend/config.py`。
- 采用 `foundation.logging` 后删除 `src/fashion_trend/log.py`。
- 删除没有实际行为的 `src/fashion_trend/data_loader.py`。
- 拆分到 `catalog` 后删除根包 `src/fashion_trend/articles.py`。
- 移动到 `trend/training.py` 后删除根包 `src/fashion_trend/training.py`。
- 移动到 `trend/evaluation.py` 后删除根包 `src/fashion_trend/evaluation.py`。
- 移动到 `trend/models/` 后删除根包 `src/fashion_trend/models/`。

### 测试

- 新增 `tests/test_architecture_boundaries.py`，覆盖依赖规则和历史根包模块移除。
- 更新现有测试，改为直接导入业务域模块：
  - `tests/test_articles_clean.py`
  - `tests/test_attribute_graph.py`
  - `tests/test_trend_article_sales.py`
  - `tests/test_trend_attribute_heat.py`
  - `tests/test_trend_targets.py`
  - `tests/test_trend_samples.py`
  - `tests/test_trend_splits.py`
  - `tests/test_trend_training.py`
  - `tests/test_trend_evaluation.py`
- 架构目标由 `tests/test_architecture_boundaries.py` 覆盖；不保留趋势包聚合导出入口兼容测试。

### 文档

- 更新 `README.md`，说明业务域驱动的内部组织方式，同时保持用户运行命令不变。
- 更新 `docs/gpt-research/implementation-plan.md` 中涉及内部模块路径的描述。
- 仅当实现发现设计矛盾时，才更新 `docs/superpowers/specs/2026-05-07-domain-driven-code-architecture-design.md`。

## Task 1：添加架构边界测试和包骨架

**Files:**
- Create: `tests/test_architecture_boundaries.py`
- Create: `src/fashion_trend/foundation/__init__.py`
- Create: `src/fashion_trend/datasets/__init__.py`
- Create: `src/fashion_trend/transactions/__init__.py`
- Create: `src/fashion_trend/catalog/__init__.py`
- Create: `src/fashion_trend/recommendation/__init__.py`
- Create: `src/fashion_trend/reports/__init__.py`

- [ ] **Step 1：写入失败的架构测试**

创建 `tests/test_architecture_boundaries.py`：

```python
from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "fashion_trend"

BUSINESS_DOMAINS = {
    "datasets",
    "transactions",
    "catalog",
    "trend",
    "recommendation",
    "reports",
}

HISTORICAL_ROOT_MODULES = {
    "articles.py",
    "config.py",
    "data_loader.py",
    "evaluation.py",
    "log.py",
    "training.py",
}


def iter_python_files(package_name: str) -> list[Path]:
    package_path = PACKAGE_ROOT / package_name
    assert package_path.exists(), f"package missing: fashion_trend.{package_name}"
    return sorted(
        path
        for path in package_path.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def assert_package_does_not_import(
    package_name: str,
    forbidden_modules: set[str],
) -> None:
    offenders: list[str] = []
    for path in iter_python_files(package_name):
        for module_name in imported_modules(path):
            for forbidden in forbidden_modules:
                if module_name == forbidden or module_name.startswith(forbidden + "."):
                    relative_path = path.relative_to(PACKAGE_ROOT.parents[0])
                    offenders.append(f"{relative_path}: {module_name}")
    assert not offenders, "\n".join(offenders)


def test_foundation_has_no_business_domain_imports() -> None:
    forbidden = {f"fashion_trend.{name}" for name in BUSINESS_DOMAINS}
    assert_package_does_not_import("foundation", forbidden)


def test_catalog_does_not_depend_on_trend_or_recommendation() -> None:
    assert_package_does_not_import(
        "catalog",
        {"fashion_trend.trend", "fashion_trend.recommendation"},
    )


def test_transactions_does_not_depend_on_catalog_trend_or_recommendation() -> None:
    assert_package_does_not_import(
        "transactions",
        {
            "fashion_trend.catalog",
            "fashion_trend.trend",
            "fashion_trend.recommendation",
        },
    )


def test_trend_does_not_depend_on_recommendation_or_reports() -> None:
    assert_package_does_not_import(
        "trend",
        {"fashion_trend.recommendation", "fashion_trend.reports"},
    )


def test_recommendation_does_not_depend_on_trend_model_internals() -> None:
    assert_package_does_not_import(
        "recommendation",
        {"fashion_trend.trend.models"},
    )


def test_historical_root_modules_are_removed() -> None:
    existing = sorted(
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_file() and path.name in HISTORICAL_ROOT_MODULES
    )
    assert existing == []
```

- [ ] **Step 2：运行新测试，确认当前结构会失败**

Run:

```sh
uv run pytest tests/test_architecture_boundaries.py -q
```

Expected: 失败信息应提到缺失的包和仍存在的历史根包模块。这是本轮迁移的 red test。

- [ ] **Step 3：添加包骨架**

每个包标记文件写入短 docstring：

```python
"""Domain package for the Fashion trend project."""
```

适用于：

```text
src/fashion_trend/foundation/__init__.py
src/fashion_trend/datasets/__init__.py
src/fashion_trend/transactions/__init__.py
src/fashion_trend/catalog/__init__.py
src/fashion_trend/recommendation/__init__.py
src/fashion_trend/reports/__init__.py
```

- [ ] **Step 4：再次运行架构测试**

Run:

```sh
uv run pytest tests/test_architecture_boundaries.py -q
```

Expected: 只剩历史根包模块和尚未迁移依赖导致的失败。

- [ ] **Step 5：提交**

Run:

```sh
git add tests/test_architecture_boundaries.py src/fashion_trend/foundation/__init__.py src/fashion_trend/datasets/__init__.py src/fashion_trend/transactions/__init__.py src/fashion_trend/catalog/__init__.py src/fashion_trend/recommendation/__init__.py src/fashion_trend/reports/__init__.py
git commit -m "test: 添加领域架构边界测试"
```

## Task 2：提取 foundation

**Files:**
- Move: `src/fashion_trend/config.py` -> `src/fashion_trend/foundation/paths.py`
- Move: `src/fashion_trend/log.py` -> `src/fashion_trend/foundation/logging.py`
- Create: `src/fashion_trend/foundation/io.py`
- Create: `src/fashion_trend/foundation/dataframe.py`
- Create: `src/fashion_trend/foundation/artifacts.py`
- Modify imports in all `src/` and `tests/` files that reference `fashion_trend.config`, `fashion_trend.log`, `fashion_trend.trend.io`, or `fashion_trend.trend.validation`.

- [ ] **Step 1：移动路径和日志模块**

Run:

```sh
git mv src/fashion_trend/config.py src/fashion_trend/foundation/paths.py
git mv src/fashion_trend/log.py src/fashion_trend/foundation/logging.py
```

- [ ] **Step 2：更新路径和日志 import**

手动替换这些 import：

```python
from fashion_trend.config import PATH
from fashion_trend.config import GRAPH_DIR, PATH
from fashion_trend.config import DEFAULT_COMPETITION, RAW_DIR
from fashion_trend.config import OUTPUT_METRICS_DIR, OUTPUT_MODELS_DIR
from fashion_trend import log
```

替换为：

```python
from fashion_trend.foundation.paths import PATH
from fashion_trend.foundation.paths import GRAPH_DIR, PATH
from fashion_trend.foundation.paths import DEFAULT_COMPETITION, RAW_DIR
from fashion_trend.foundation.paths import OUTPUT_METRICS_DIR, OUTPUT_MODELS_DIR
from fashion_trend.foundation import logging as log
```

- [ ] **Step 3：创建通用 DataFrame 校验模块**

创建 `src/fashion_trend/foundation/dataframe.py`，从 `src/fashion_trend/trend/validation.py` 移入通用函数，保持行为不变：

```python
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"{source_name} 缺少必要字段: {', '.join(missing_columns)}")


def validate_no_missing_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    missing_counts = dataframe[list(columns)].isna().sum()
    invalid_columns = [
        f"{column}={int(count)}"
        for column, count in missing_counts.items()
        if int(count) > 0
    ]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在缺失值: {', '.join(invalid_columns)}")


def validate_unique_key(
    dataframe: pd.DataFrame,
    key_columns: Sequence[str],
    source_name: str,
) -> None:
    duplicate_count = int(dataframe.duplicated(list(key_columns)).sum())
    if duplicate_count > 0:
        key_names = ", ".join(key_columns)
        raise ValueError(f"{source_name} 存在重复键 {key_names}: {duplicate_count} 行")


def validate_non_negative_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    invalid_columns = [
        column
        for column in columns
        if bool((dataframe[column] < 0).any())
    ]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在负数: {', '.join(invalid_columns)}")


def validate_positive_values(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    invalid_columns = [
        column
        for column in columns
        if bool((dataframe[column] <= 0).any())
    ]
    if invalid_columns:
        raise ValueError(f"{source_name} 存在非正数: {', '.join(invalid_columns)}")
```

- [ ] **Step 4：创建通用 foundation IO**

创建 `src/fashion_trend/foundation/io.py`，使用通用命名，并保留当前 `trend/io.py` 的原子写入行为：

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


def remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def write_json_atomic(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)


def write_csv_atomic(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_csv(tmp_path, index=False, quoting=csv.QUOTE_ALL)
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)


def write_parquet_atomic(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_parquet(tmp_path, index=False)
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)


def write_binary_atomic(payload: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)
```

- [ ] **Step 5：创建 artifact 安全辅助模块**

创建 `src/fashion_trend/foundation/artifacts.py`：

```python
from __future__ import annotations

from pathlib import Path


def validate_safe_path_segment(segment: str, source_name: str) -> None:
    if not segment:
        raise ValueError(f"{source_name} 不能为空。")
    if segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise ValueError(f"{source_name} 不是安全的路径片段: {segment}")


def validate_output_parent_dirs(parent_path: Path, output_dir: Path) -> None:
    parent_path = parent_path.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(parent_path):
        raise ValueError(f"输出目录不在允许范围内: {output_dir}")
```

- [ ] **Step 6：更新 trend 模块，使其使用 foundation 校验和 IO**

在当前 trend 文件中，将：

```python
from fashion_trend.trend.validation import validate_required_columns
from fashion_trend.trend.io import write_json, write_trend_csv, write_trend_parquet
```

替换为：

```python
from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.foundation.io import write_csv_atomic, write_json_atomic, write_parquet_atomic
```

并将调用：

```python
write_json(payload, path)
write_trend_csv(frame, path)
write_trend_parquet(frame, path)
```

替换为：

```python
write_json_atomic(payload, path)
write_csv_atomic(frame, path)
write_parquet_atomic(frame, path)
```

- [ ] **Step 7：删除旧通用模块**

所有 import 都迁移完成后，删除：

```text
src/fashion_trend/trend/io.py
src/fashion_trend/trend/validation.py
```

- [ ] **Step 8：验证 foundation 边界**

Run:

```sh
uv run pytest tests/test_architecture_boundaries.py -q
uv run pytest tests/test_trend_article_sales.py tests/test_trend_attribute_heat.py tests/test_trend_targets.py tests/test_trend_samples.py tests/test_trend_splits.py -q
uv run python -m py_compile src/fashion_trend/foundation/paths.py src/fashion_trend/foundation/logging.py src/fashion_trend/foundation/io.py src/fashion_trend/foundation/dataframe.py src/fashion_trend/foundation/artifacts.py
```

Expected: 趋势阶段测试通过；架构测试仍可能因尚未迁移的历史模块失败。

- [ ] **Step 9：提交**

Run:

```sh
git add src tests
git commit -m "refactor: 提取无业务基础层"
```

## Task 3：迁移 datasets、transactions 和 catalog

**Files:**
- Create: `src/fashion_trend/datasets/download.py`
- Create: `src/fashion_trend/datasets/profile.py`
- Move script logic from: `src/00_download_data.py`
- Move script logic from: `src/02_build_weekly_transactions.py`
- Split: `src/fashion_trend/articles.py` -> `src/fashion_trend/catalog/articles.py` and `src/fashion_trend/catalog/graph.py`
- Modify: `src/00_download_data.py`
- Modify: `src/01_data_check.py`
- Modify: `src/02_build_weekly_transactions.py`
- Modify: `src/03_clean_articles.py`
- Modify: `src/04_build_attribute_graph.py`
- Modify tests for article cleaning and graph construction.

- [ ] **Step 1：提取数据下载逻辑**

将这些函数和类型别名从 `src/00_download_data.py` 移到 `src/fashion_trend/datasets/download.py`：

```text
Downloader
competition_target_dir
should_skip_download
extract_zip_files
kagglehub_competition_download
download_competition
```

`src/00_download_data.py` 中保留 `parse_args()` 和 `main()`。

移动后，脚本 import 应为：

```python
from fashion_trend.datasets.download import download_competition
from fashion_trend.foundation.paths import DEFAULT_COMPETITION, RAW_DIR
```

- [ ] **Step 2：添加 raw profile 模块，明确 data-check ownership**

创建 `src/fashion_trend/datasets/profile.py`：

```python
from __future__ import annotations

from pathlib import Path


RAW_FILE_NAMES = (
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
)


def validate_raw_dataset_files(raw_dataset_dir: Path) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for file_name in RAW_FILE_NAMES:
        csv_path = raw_dataset_dir / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"原始数据文件不存在: {csv_path}")
        with csv_path.open("rb") as handle:
            line_count = sum(1 for _ in handle)
        row_counts[file_name] = max(line_count - 1, 0)
    return row_counts
```

这能让 `datasets` ownership 明确，即使当前 `01_data_check.py` 的行为仍然较轻量。

- [ ] **Step 3：让 `01_data_check.py` 成为可读的 datasets 流程索引**

将 `src/01_data_check.py` 替换为：

```python
from __future__ import annotations

from fashion_trend.datasets.profile import validate_raw_dataset_files
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.paths import RAW_HM_DIR


LOG_SOURCE = "data-check"


def main() -> int:
    try:
        log.info(f"检查原始数据目录: {RAW_HM_DIR}", source=LOG_SOURCE)
        row_counts = validate_raw_dataset_files(RAW_HM_DIR)
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    for file_name, row_count in row_counts.items():
        log.info(f"{file_name}: {row_count:,} 行", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4：移动周级交易逻辑**

将 `src/02_build_weekly_transactions.py` 中除 `main()` 之外的所有常量和函数移动到 `src/fashion_trend/transactions/weekly.py`。

`src/02_build_weekly_transactions.py` 只保留这些 import：

```python
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.paths import PATH
from fashion_trend.transactions.weekly import build_weekly_transactions
```

保持脚本 `main()` 可读：

```python
def main() -> int:
    try:
        log.info(f"输入文件: {PATH['raw_transactions']}", source=LOG_SOURCE)
        build_weekly_transactions(
            raw_transactions_path=PATH["raw_transactions"],
            weekly_transactions_path=PATH["interim_transactions_weekly"],
        )
        log.info(f"输出文件: {PATH['interim_transactions_weekly']}", source=LOG_SOURCE)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1
    return 0
```

- [ ] **Step 5：拆分 catalog 商品清洗和属性图构建**

将商品清洗函数移动到 `src/fashion_trend/catalog/articles.py`：

```text
ARTICLE_ID_COLUMN
PRODUCT_CODE_COLUMN
validate_required_columns
validate_no_missing_values
validate_unique_values
normalize_article_identifiers
build_clean_article_frames
read_articles_csv
clean_articles_file
restore_mvp_output
```

将属性图相关函数移动到 `src/fashion_trend/catalog/graph.py`：

```text
make_attr_id
make_article_node_id
make_edge_type
build_article_nodes
build_attribute_nodes
build_article_attribute_edges
build_attribute_hierarchy_edges
read_clean_articles
validate_graph_references
cleanup_graph_publish_files
rollback_graph_outputs
publish_graph_frames
build_attribute_graph_frames
build_attribute_graph_files
```

将通用 CSV/文件 helper 替换为 foundation 调用：

```python
from fashion_trend.foundation.io import remove_file_if_exists, write_csv_atomic
```

旧代码中单个 CSV 写入的位置使用 `write_csv_atomic(frame, output_path)`。图发布的 rollback helper 保留在 `catalog.graph`，因为它属于属性图发布语义。

- [ ] **Step 6：更新 catalog 脚本**

`src/03_clean_articles.py` 应导入：

```python
from fashion_trend.catalog.articles import clean_articles_file
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.paths import PATH
```

`src/04_build_attribute_graph.py` 应导入：

```python
from fashion_trend.catalog.graph import build_attribute_graph_files
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.paths import GRAPH_DIR, PATH
```

- [ ] **Step 7：更新 catalog 测试**

将测试中的：

```python
from fashion_trend.articles import ...
```

替换为：

```python
from fashion_trend.catalog.articles import ...
from fashion_trend.catalog.graph import ...
```

- [ ] **Step 8：删除根包 article/data-loader 模块**

删除：

```text
src/fashion_trend/articles.py
src/fashion_trend/data_loader.py
```

- [ ] **Step 9：验证 datasets、transactions 和 catalog**

Run:

```sh
uv run pytest tests/test_articles_clean.py tests/test_attribute_graph.py tests/test_architecture_boundaries.py -q
uv run python -m py_compile src/00_download_data.py src/01_data_check.py src/02_build_weekly_transactions.py src/03_clean_articles.py src/04_build_attribute_graph.py src/fashion_trend/datasets/download.py src/fashion_trend/datasets/profile.py src/fashion_trend/transactions/weekly.py src/fashion_trend/catalog/articles.py src/fashion_trend/catalog/graph.py
```

Expected: article 和 graph 测试通过；架构测试可能仍因 Task 5 前的趋势实验层根包模块失败。

- [ ] **Step 10：提交**

Run:

```sh
git add src tests
git commit -m "refactor: 迁移数据交易和商品目录域"
```

## Task 4：收敛确定性趋势流水线边界

**Files:**
- Modify: `src/fashion_trend/trend/article_sales.py`
- Modify: `src/fashion_trend/trend/attribute_heat.py`
- Modify: `src/fashion_trend/trend/targets.py`
- Modify: `src/fashion_trend/trend/samples.py`
- Modify: `src/fashion_trend/trend/splits.py`
- Modify: `src/fashion_trend/trend/predictions.py`
- Modify: `src/fashion_trend/trend/__init__.py`
- Modify: `src/05_compute_article_week_sales.py`
- Modify: `src/06_compute_attribute_week_heat.py`
- Modify: `src/07_build_trend_targets.py`
- Modify: `src/08_build_trend_model_samples.py`
- Modify: `src/09_split_trend_model_samples.py`
- Modify trend tests.

- [ ] **Step 1：移动周级交易读取 ownership**

如果 `read_weekly_transactions()` 仍在 `trend/article_sales.py`，将它移动到 `src/fashion_trend/transactions/weekly.py`，因为它读取的是稳定交易产物。不要在 `trend/article_sales.py` 中导入或重新导出该 reader，避免恢复旧 reader 泄漏。

然后在 `src/05_compute_article_week_sales.py` 中直接导入：

```python
from fashion_trend.transactions.weekly import read_weekly_transactions
```

`trend/article_sales.py` 只保留 `build_article_week_sales_frame()`、`validate_article_week_sales()` 和商品周销量读取/校验等 article sales 阶段逻辑。

- [ ] **Step 2：把 catalog 图读取从 trend heat 中移出**

将这些 reader 从 `trend/attribute_heat.py` 移动到 `catalog/graph.py`：

```text
read_article_attribute_edges
read_attribute_nodes
```

这些 heat-specific validator 保留在 `trend/attribute_heat.py`：

```text
validate_article_attribute_edges_for_heat
validate_all_sales_articles_have_attribute_edges
validate_attribute_nodes_for_heat
validate_attribute_edge_node_metadata_consistency
build_attribute_week_heat_frame
validate_attribute_week_heat
read_attribute_week_heat
```

更新 `src/06_compute_attribute_week_heat.py`：

```python
from fashion_trend.catalog.graph import read_article_attribute_edges, read_attribute_nodes
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_all_sales_articles_have_attribute_edges,
    validate_attribute_edge_node_metadata_consistency,
    validate_attribute_week_heat,
)
```

- [ ] **Step 3：移除 trend package facade exports**

将 `src/fashion_trend/trend/__init__.py` 替换为：

```python
"""Trend domain package."""
```

如仍存在趋势包聚合导出入口兼容测试，应删除；该导入形式不再支持，`tests/test_architecture_boundaries.py` 已覆盖架构目标。

- [ ] **Step 4：更新确定性趋势测试，改为直接导入**

每个趋势测试都应从具体模块导入，例如：

```python
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import build_attribute_week_heat_frame
from fashion_trend.trend.targets import build_attribute_week_target_frame
from fashion_trend.trend.samples import build_trend_model_samples_frame
from fashion_trend.trend.splits import build_trend_model_split_frames
```

- [ ] **Step 5：验证确定性趋势流水线**

Run:

```sh
uv run pytest tests/test_trend_article_sales.py tests/test_trend_attribute_heat.py tests/test_trend_targets.py tests/test_trend_samples.py tests/test_trend_splits.py tests/test_architecture_boundaries.py -q
uv run python -m py_compile src/fashion_trend/trend/article_sales.py src/fashion_trend/trend/attribute_heat.py src/fashion_trend/trend/targets.py src/fashion_trend/trend/samples.py src/fashion_trend/trend/splits.py src/fashion_trend/trend/predictions.py src/05_compute_article_week_sales.py src/06_compute_attribute_week_heat.py src/07_build_trend_targets.py src/08_build_trend_model_samples.py src/09_split_trend_model_samples.py
```

Expected: 确定性趋势测试通过；如果根包趋势实验模块尚未迁移，架构测试仍可能因此失败。

- [ ] **Step 6：提交**

Run:

```sh
git add src tests
git commit -m "refactor: 收敛趋势确定性流水线边界"
```

## Task 5：将趋势模型、训练和评价移动到 trend 域内

**Files:**
- Move: `src/fashion_trend/models/` -> `src/fashion_trend/trend/models/`
- Move: `src/fashion_trend/training.py` -> `src/fashion_trend/trend/training.py`
- Move: `src/fashion_trend/evaluation.py` -> `src/fashion_trend/trend/evaluation.py`
- Modify: `src/10_train_trend_model.py`
- Modify: `src/11_eval_trend_model.py`
- Modify: `tests/test_trend_training.py`
- Modify: `tests/test_trend_evaluation.py`

- [ ] **Step 1：移动趋势实验层模块**

Run:

```sh
mkdir -p src/fashion_trend/trend/models
git mv src/fashion_trend/models/base.py src/fashion_trend/trend/models/base.py
git mv src/fashion_trend/models/last_week.py src/fashion_trend/trend/models/last_week.py
git mv src/fashion_trend/models/moving_average.py src/fashion_trend/trend/models/moving_average.py
git mv src/fashion_trend/models/registry.py src/fashion_trend/trend/models/registry.py
git mv src/fashion_trend/models/__init__.py src/fashion_trend/trend/models/__init__.py
git mv src/fashion_trend/training.py src/fashion_trend/trend/training.py
git mv src/fashion_trend/evaluation.py src/fashion_trend/trend/evaluation.py
```

- [ ] **Step 2：更新趋势模型 import**

将：

```python
from fashion_trend.models.base import ...
from fashion_trend.models.last_week import ...
from fashion_trend.models.moving_average import ...
from fashion_trend.models.registry import ...
from fashion_trend.training import run_trend_model_training
from fashion_trend.evaluation import run_trend_model_evaluation
```

替换为：

```python
from fashion_trend.trend.models.base import ...
from fashion_trend.trend.models.last_week import ...
from fashion_trend.trend.models.moving_average import ...
from fashion_trend.trend.models.registry import ...
from fashion_trend.trend.training import run_trend_model_training
from fashion_trend.trend.evaluation import run_trend_model_evaluation
```

- [ ] **Step 3：使用 foundation artifact 安全 helper**

在 `trend/training.py` 和 `trend/evaluation.py` 中，用 foundation helper 替换重复的私有 model-name/path 检查：

```python
from fashion_trend.foundation.artifacts import (
    validate_output_parent_dirs,
    validate_safe_path_segment,
)
```

调用：

```python
validate_safe_path_segment(model_name, "model_name")
```

以及：

```python
validate_output_parent_dirs(OUTPUT_MODELS_DIR, output_dir)
```

或：

```python
validate_output_parent_dirs(OUTPUT_METRICS_DIR, output_dir)
```

兼容目标：不安全的 `model_name` 仍必须抛出 `ValueError`。

- [ ] **Step 4：删除根包 models 目录**

所有 import 更新完成、测试也改为从 `fashion_trend.trend.models` 导入后，删除空根目录：

```sh
rmdir src/fashion_trend/models
```

- [ ] **Step 5：验证训练和评价**

Run:

```sh
uv run pytest tests/test_trend_training.py tests/test_trend_evaluation.py tests/test_architecture_boundaries.py -q
uv run python -m py_compile src/10_train_trend_model.py src/11_eval_trend_model.py src/fashion_trend/trend/training.py src/fashion_trend/trend/evaluation.py src/fashion_trend/trend/models/base.py src/fashion_trend/trend/models/last_week.py src/fashion_trend/trend/models/moving_average.py src/fashion_trend/trend/models/registry.py
```

Expected: 训练/评价测试通过；架构边界测试在依赖方向和历史根包模块移除方面通过。

- [ ] **Step 6：提交**

Run:

```sh
git add src tests
git commit -m "refactor: 迁移趋势实验层"
```

## Task 6：刷新编号 CLI，使其成为流程索引

**Files:**
- Modify: `src/00_download_data.py`
- Modify: `src/01_data_check.py`
- Modify: `src/02_build_weekly_transactions.py`
- Modify: `src/03_clean_articles.py`
- Modify: `src/04_build_attribute_graph.py`
- Modify: `src/05_compute_article_week_sales.py`
- Modify: `src/06_compute_attribute_week_heat.py`
- Modify: `src/07_build_trend_targets.py`
- Modify: `src/08_build_trend_model_samples.py`
- Modify: `src/09_split_trend_model_samples.py`
- Modify: `src/10_train_trend_model.py`
- Modify: `src/11_eval_trend_model.py`

- [ ] **Step 1：审查每个编号脚本是否保留可读流程**

每个脚本中，`main()` 或顶层阶段函数应呈现这种形态：

```python
def main(...) -> int:
    try:
        log.info("输入文件或输入目录: ...", source=LOG_SOURCE)
        ...
        log.info("关键处理步骤: ...", source=LOG_SOURCE)
        ...
        log.info("输出文件或输出目录: ...", source=LOG_SOURCE)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1
    return 0
```

实际日志应命名真实产物，例如 `attribute_week_heat.csv`、`trend_model_samples.parquet` 或 `outputs/models/<model>/predictions.csv`。

- [ ] **Step 2：CLI 参数解析继续留在编号脚本**

`00_download_data.py`、`10_train_trend_model.py` 和 `11_eval_trend_model.py` 的 `parse_args()` 保留在脚本中。业务包接收已解析的值，不接收 argparse namespace。

`src/10_train_trend_model.py` 应导入：

```python
from fashion_trend.foundation import logging as log
from fashion_trend.trend.models.registry import UnknownTrendModelError
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_VALUES
from fashion_trend.trend.training import run_trend_model_training
```

`src/11_eval_trend_model.py` 应导入：

```python
from fashion_trend.foundation import logging as log
from fashion_trend.trend.evaluation import run_trend_model_evaluation
```

- [ ] **Step 3：确认编号脚本不是计算事实来源**

搜索编号脚本中的重 pandas 逻辑：

```sh
rg -n "groupby|merge|rolling|to_parquet|to_csv|pd\\." src/0*.py src/1*.py
```

Expected: 编号脚本中不再有重计算 pandas 用法。`01_data_check.py` 可以调用 `datasets.profile`，但不应直接做业务 transform。

- [ ] **Step 4：验证 CLI 可编译**

Run:

```sh
uv run python -m py_compile src/00_download_data.py src/01_data_check.py src/02_build_weekly_transactions.py src/03_clean_articles.py src/04_build_attribute_graph.py src/05_compute_article_week_sales.py src/06_compute_attribute_week_heat.py src/07_build_trend_targets.py src/08_build_trend_model_samples.py src/09_split_trend_model_samples.py src/10_train_trend_model.py src/11_eval_trend_model.py
```

Expected: 所有脚本编译通过。

- [ ] **Step 5：提交**

Run:

```sh
git add src
git commit -m "refactor: 刷新编号脚本流程索引"
```

## Task 7：同步文档并运行完整验证

**Files:**
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`
- Modify: `docs/superpowers/specs/2026-05-07-domain-driven-code-architecture-design.md` only if implementation contradicts the spec.

- [ ] **Step 1：更新 README 架构说明**

在 `README.md` 的实现位置说明中写明：

```markdown
项目内部代码按业务域组织在 `src/fashion_trend/` 下：

- `foundation/`：路径、日志、原子写入、通用校验和 artifact 安全。
- `datasets/`：原始数据下载、解压和基础检查。
- `transactions/`：周级交易表和交易窗口。
- `catalog/`：商品表清洗和静态属性图。
- `trend/`：属性热度、标签、样本、时间切分、趋势模型训练和趋势评价。
- `recommendation/`：候选、重排序、Top-12 和推荐评价。
- `reports/`：图表、表格和案例导出。

`src/00_*.py` 到 `src/16_*.py` 仍是用户运行入口；脚本保留高层流程索引，计算事实位于业务包。
```

保持现有用户命令不变。

- [ ] **Step 2：更新 implementation-plan 中的模块路径引用**

在 `docs/gpt-research/implementation-plan.md` 中，将旧内部路径：

```text
src/fashion_trend/articles.py
src/fashion_trend/training.py
src/fashion_trend/evaluation.py
src/fashion_trend/models/
```

替换为：

```text
src/fashion_trend/catalog/
src/fashion_trend/trend/training.py
src/fashion_trend/trend/evaluation.py
src/fashion_trend/trend/models/
```

编号脚本名称和 artifact 路径保持不变。

- [ ] **Step 3：运行完整测试**

Run:

```sh
uv run pytest
```

Expected: 全部测试通过。

- [ ] **Step 4：运行 src 全量编译验证**

Run:

```sh
uv run python -m py_compile $(find src -name '*.py' -not -path '*/__pycache__/*' | sort)
```

Expected: 命令成功退出。

- [ ] **Step 5：检查历史 import 残留**

Run:

```sh
rg -n "fashion_trend\\.(articles|config|data_loader|evaluation|log|models|training)\\b|from fashion_trend\\.trend import" src tests README.md docs/gpt-research/implementation-plan.md
```

Expected: 无匹配。

- [ ] **Step 6：运行现有产物契约 CLI 烟测**

运行最能发现行为漂移的命令：

```sh
uv run python src/05_compute_article_week_sales.py
uv run python src/06_compute_attribute_week_heat.py
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

Expected:

```text
data/processed/trend/article_week_sales.csv exists
data/processed/trend/attribute_week_heat.csv exists
outputs/models/moving_average/predictions.csv exists
outputs/models/moving_average/metadata.json exists
outputs/models/moving_average/params.json exists
outputs/metrics/moving_average/trend_metrics.json exists
```

- [ ] **Step 7：运行 artifact shape 检查**

Run:

```sh
uv run python - <<'PY'
from pathlib import Path

import pandas as pd

checks = {
    "article_week_sales": Path("data/processed/trend/article_week_sales.csv"),
    "attribute_week_heat": Path("data/processed/trend/attribute_week_heat.csv"),
    "moving_average_predictions": Path("outputs/models/moving_average/predictions.csv"),
}

for name, path in checks.items():
    frame = pd.read_csv(path)
    print(name, len(frame), list(frame.columns))
PY
```

Expected:

```text
article_week_sales has columns ['week_id', 'article_id', 'sales_cnt', 'sales_user_cnt', 'sales_amount']
attribute_week_heat has columns ['week_id', 'attr_id', 'attr_type', 'attr_value', 'heat_cnt', 'type_total_heat', 'heat_share', 'log_heat', 'rank_in_type']
moving_average_predictions has the same columns as TREND_MODEL_PREDICTION_COLUMNS
```

- [ ] **Step 8：检查 git diff 范围**

Run:

```sh
git diff --stat
git diff --check
```

Expected: 只包含计划内代码、测试和文档改动；没有 whitespace errors。

- [ ] **Step 9：提交**

Run:

```sh
git add README.md docs/gpt-research/implementation-plan.md docs/superpowers/specs/2026-05-07-domain-driven-code-architecture-design.md
git commit -m "docs: 同步业务域架构说明"
```

## 最终验收

满足以下条件时，本次迁移完成：

- `uv run pytest` 通过。
- `src` 全量 `py_compile` 通过。
- `tests/test_architecture_boundaries.py` 通过。
- `rg` 找不到已移除根包模块 import，也找不到旧 `fashion_trend.trend` facade import。
- 编号 CLI 仍能编译，且用户命令名称稳定。
- 既有 artifact 路径保持稳定：
  - `data/processed/trend/article_week_sales.csv`
  - `data/processed/trend/attribute_week_heat.csv`
  - `data/processed/trend/attribute_week_target.csv`
  - `data/processed/features/trend_model_samples.parquet`
  - `outputs/models/<model>/predictions.csv`
  - `outputs/metrics/<model>/trend_metrics.json`
- README 和 implementation plan 已描述新的业务域驱动内部组织。
