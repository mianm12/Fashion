# 业务域驱动代码架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在第一轮业务域迁移已完成的基础上，冻结目标态依赖边界，迁出业务路径 ownership，建立跨领域只读 public surface，并继续拆分 `catalog` 与 `trend` 内部大模块。

**Architecture:** 编号 CLI 继续作为 `read -> build -> validate -> write` 的可读流程索引，业务计算事实留在 `src/fashion_trend/<domain>/`。`foundation` 只保留无业务语义基础能力；`recommendation` 和 `reports` 只能消费上游 public read-only allowlist。

**Tech Stack:** Python 3.10-3.12、pandas、numpy、pyarrow、pytest、uv、setuptools `src` layout。

---

## Scope Check

本计划替换旧的第一轮迁移计划。当前基线已经具备：

- `foundation/`、`datasets/`、`transactions/`、`catalog/`、`trend/`、`recommendation/`、`reports/` 包。
- 历史根包模块 `articles.py`、`config.py`、`data_loader.py`、`evaluation.py`、`log.py`、`training.py` 和根包 `models/` 已移除。
- 编号 CLI 已直接导入业务域模块。

本计划不新增 LightGBM，不实现推荐算法，不改变现有算法公式，不改变既有产物路径或 schema。每个任务只做结构、边界、路径 ownership 或文档同步；涉及真实产物的任务必须用现有 CLI 或测试证明产物契约未漂移。

## Execution Rules

- 每个任务独立提交；不要把多个任务攒成一个大 diff。
- 每个任务先写或更新能锁住目标边界的测试，再改实现。
- 每个任务提交前运行本任务列出的最小验证命令。
- 触及核心产物读写路径时，额外跑对应 CLI 烟测或明确说明本地缺少真实数据。
- 不使用批量替换、codemod 或大范围格式化；逐文件人工审阅修改。
- 将现有 `.py` 文件替换成同名目录包时，必须在同一个补丁和同一个提交里删除旧文件并添加新目录文件；不要先创建同名目录，也不要提交中间不可导入状态。
- 开始 Task 2 前，如果本地具备真实上游数据，必须先用当前实现生成并保存 artifact baseline；后续最终验证要和该 baseline 做精确对比，不能只检查非空行数。

### Artifact Drift Baseline

Before starting Task 2, check whether these upstream files exist:

```text
data/interim/articles_clean.csv
data/interim/transactions_train_weekly.parquet
data/processed/features/trend_model_samples_train.parquet
data/processed/features/trend_model_samples_valid.parquet
data/processed/features/trend_model_samples_test.parquet
```

If any file is missing, record the exact missing path in the task report and use the same reason when skipping real-data artifact comparisons later.

If all files exist, run the current implementation before structural edits:

```sh
uv run python src/04_build_attribute_graph.py
uv run python src/05_compute_article_week_sales.py
uv run python src/06_compute_attribute_week_heat.py
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

Then capture the baseline summary:

```sh
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path

import pandas as pd

BASELINE_PATH = Path("/tmp/fashion_domain_arch_artifact_baseline.json")
TREND_METRICS_KEYS = {
    "model_name",
    "prediction_path",
    "output_path",
    "evaluated_splits",
    "ranking",
    "overall",
    "by_attr_type",
    "groups",
}
CSV_ARTIFACTS = {
    "graph_nodes_article": Path("data/processed/graph/nodes_article.csv"),
    "graph_nodes_attribute": Path("data/processed/graph/nodes_attribute.csv"),
    "graph_edges_article_attribute": Path(
        "data/processed/graph/edges_article_attribute.csv"
    ),
    "graph_edges_attribute_hierarchy": Path(
        "data/processed/graph/edges_attribute_hierarchy.csv"
    ),
    "article_week_sales": Path("data/processed/trend/article_week_sales.csv"),
    "attribute_week_heat": Path("data/processed/trend/attribute_week_heat.csv"),
    "moving_average_predictions": Path(
        "outputs/models/moving_average/predictions.csv"
    ),
}
JSON_ARTIFACTS = {
    "moving_average_metadata": Path("outputs/models/moving_average/metadata.json"),
    "moving_average_params": Path("outputs/models/moving_average/params.json"),
    "moving_average_metrics": Path(
        "outputs/metrics/moving_average/trend_metrics.json"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary: dict[str, object] = {}
for name, path in CSV_ARTIFACTS.items():
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    summary[name] = {
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
    }
    if "attr_type" in frame.columns:
        summary[name]["rows_by_attr_type"] = {
            str(key): int(value)
            for key, value in frame["attr_type"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        }
    if "week_id" in frame.columns:
        week_ids = pd.to_numeric(frame["week_id"], errors="raise")
        summary[name]["week_id"] = {
            "min": int(week_ids.min()),
            "max": int(week_ids.max()),
            "nunique": int(week_ids.nunique()),
        }

for name, path in JSON_ARTIFACTS.items():
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if name == "moving_average_metrics":
        missing = sorted(TREND_METRICS_KEYS - set(payload))
        if missing:
            raise AssertionError(f"trend_metrics.json missing keys: {missing}")
    summary[name] = {
        "path": str(path),
        "sha256": sha256(path),
        "keys": sorted(payload),
    }

BASELINE_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(f"wrote {BASELINE_PATH}")
PY
```

Expected: the baseline command exits 0 and writes `/tmp/fashion_domain_arch_artifact_baseline.json`. Do not regenerate this baseline after refactors; final verification compares current artifacts to this pre-migration file.

## Target File Map

### Architecture Tests

- Modify: `tests/test_architecture_boundaries.py`
  - 继续检测历史根包不回流。
  - 新增 `recommendation` 上游 allowlist 检查。
  - 新增 `reports` 上游 allowlist 检查。
  - 新增 `foundation.paths` 导出白名单检查。

### Domain Paths

- Modify: `src/fashion_trend/foundation/paths.py`
  - 最终只保留 `PROJECT_ROOT`、`DATA_DIR`、`RAW_DIR`、`INTERIM_DIR`、`PROCESSED_DIR`、`OUTPUT_DIR`。
- Create: `src/fashion_trend/datasets/paths.py`
  - H&M competition slug、raw H&M 根目录和 raw CSV 路径。
- Create: `src/fashion_trend/transactions/paths.py`
  - 周级交易表路径。
- Create: `src/fashion_trend/catalog/paths.py`
  - 清洗商品表和属性图产物路径。
- Create: `src/fashion_trend/trend/paths.py`
  - article sales、heat、target、samples、split、model output、metrics output 路径和 split 窗口配置。
- Create: `src/fashion_trend/recommendation/paths.py`
  - 推荐中间表、推荐结果和推荐评价输出路径。
- Create: `src/fashion_trend/reports/paths.py`
  - figures、tables、case studies 和 report export 路径。
- Modify: `src/00_download_data.py` through `src/11_eval_trend_model.py`
  - 从领域 path 模块导入本阶段路径，不再导入业务路径自 `foundation.paths`。
- Modify: `src/fashion_trend/datasets/download.py`
- Modify: `src/fashion_trend/trend/training.py`
- Modify: `src/fashion_trend/trend/evaluation.py`
- Modify: tests importing `foundation.paths` business constants.

### Public Read-Only Surface

- Create: `src/fashion_trend/transactions/contracts.py`
- Create: `src/fashion_trend/transactions/readers.py`
- Create: `src/fashion_trend/catalog/contracts.py`
- Create: `src/fashion_trend/catalog/readers.py`
- Create: `src/fashion_trend/trend/readers.py`
- Create: `src/fashion_trend/recommendation/contracts.py`
- Create: `src/fashion_trend/recommendation/readers.py`
- Modify: trend CLI/scripts that read catalog or transaction stable artifacts to import `transactions.readers` / `catalog.readers` where they are crossing domain boundaries.

### Catalog Split

- Delete after replacement: `src/fashion_trend/catalog/graph.py`
- Create: `src/fashion_trend/catalog/graph/__init__.py`
- Create: `src/fashion_trend/catalog/graph/schema.py`
- Create: `src/fashion_trend/catalog/graph/builders.py`
- Create: `src/fashion_trend/catalog/graph/publishing.py`
- Use: `src/fashion_trend/catalog/readers.py` for public graph readers.
- Modify: `src/04_build_attribute_graph.py`
- Modify: `tests/test_attribute_graph.py`

### Trend Split

- Replace file modules with focused subpackages only when tests are already locked:
  - `src/fashion_trend/trend/attribute_heat.py` -> `src/fashion_trend/trend/heat/`
  - `src/fashion_trend/trend/targets.py` -> `src/fashion_trend/trend/labels/`
  - `src/fashion_trend/trend/samples.py` -> `src/fashion_trend/trend/features/`
  - `src/fashion_trend/trend/splits.py` -> `src/fashion_trend/trend/splits/`
- Split experiments:
  - `src/fashion_trend/trend/models/last_week.py` -> `src/fashion_trend/trend/models/baselines/last_week.py`
  - `src/fashion_trend/trend/models/moving_average.py` -> `src/fashion_trend/trend/models/baselines/moving_average.py`
  - keep `src/fashion_trend/trend/models/supervised/` available for future LightGBM without implementing it in this plan.
  - `src/fashion_trend/trend/training.py` -> `src/fashion_trend/trend/training/runner.py` and `outputs.py`
  - `src/fashion_trend/trend/evaluation.py` -> `src/fashion_trend/trend/evaluation/metrics.py`, `payloads.py`, and `runner.py`

### Documentation

- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`
- Modify only if implementation reveals a contradiction: `docs/superpowers/specs/2026-05-07-domain-driven-code-architecture-design.md`

## Task 1: Harden Architecture Boundary Tests

**Files:**

- Modify: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: Add public allowlist helpers**

Add these constants near the existing import-boundary constants:

```python
RECOMMENDATION_PUBLIC_UPSTREAM_IMPORTS = {
    "fashion_trend.transactions.contracts",
    "fashion_trend.transactions.readers",
    "fashion_trend.catalog.contracts",
    "fashion_trend.catalog.readers",
    "fashion_trend.trend.schema",
    "fashion_trend.trend.predictions",
    "fashion_trend.trend.readers",
}

REPORTS_PUBLIC_IMPORTS = {
    "fashion_trend.transactions.contracts",
    "fashion_trend.transactions.readers",
    "fashion_trend.catalog.contracts",
    "fashion_trend.catalog.readers",
    "fashion_trend.trend.schema",
    "fashion_trend.trend.predictions",
    "fashion_trend.trend.readers",
    "fashion_trend.recommendation.contracts",
    "fashion_trend.recommendation.readers",
}
```

Add these helpers below `assert_package_does_not_import()`:

```python
def package_upstream_import_offenders(
    paths: list[Path],
    upstream_roots: set[str],
    allowed_modules: set[str],
) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        for module_name in sorted(imported_modules(path)):
            matched_root = next(
                (
                    root
                    for root in upstream_roots
                    if module_name == root or module_name.startswith(root + ".")
                ),
                None,
            )
            if matched_root is None:
                continue
            is_allowed = any(
                module_name == allowed
                or module_name.startswith(allowed + ".")
                for allowed in allowed_modules
            )
            if not is_allowed:
                try:
                    display_path = path.relative_to(PACKAGE_ROOT.parents[0])
                except ValueError:
                    display_path = path
                offenders.append(f"{display_path}: {module_name}")
    return offenders


def assert_package_imports_only_allowed_upstream(
    package_name: str,
    upstream_roots: set[str],
    allowed_modules: set[str],
) -> None:
    offenders = package_upstream_import_offenders(
        iter_python_files(package_name),
        upstream_roots,
        allowed_modules,
    )
    assert offenders == []
```

- [ ] **Step 2: Replace broad denylist tests with allowlist tests**

Replace `test_recommendation_does_not_depend_on_trend_model_internals()` with:

```python
def test_recommendation_imports_only_public_upstream_surfaces() -> None:
    assert_package_imports_only_allowed_upstream(
        "recommendation",
        {
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
        },
        RECOMMENDATION_PUBLIC_UPSTREAM_IMPORTS,
    )
```

Replace `test_reports_does_not_depend_on_core_computation_domains()` with:

```python
def test_reports_imports_only_public_read_only_surfaces() -> None:
    assert_package_imports_only_allowed_upstream(
        "reports",
        {
            "fashion_trend.datasets",
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
            "fashion_trend.recommendation",
        },
        REPORTS_PUBLIC_IMPORTS,
    )
```

- [ ] **Step 3: Add regression tests for allowlist behavior**

Add tests using `tmp_path` so a future worker cannot accidentally weaken the helper:

```python
def test_allowlist_rejects_recommendation_importing_catalog_graph(tmp_path) -> None:
    package_root = tmp_path / "src" / "fashion_trend"
    module_path = package_root / "recommendation" / "ranker.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "from fashion_trend.catalog.graph import read_attribute_nodes\n",
        encoding="utf-8",
    )

    offenders = package_upstream_import_offenders(
        [module_path],
        {"fashion_trend.catalog"},
        {"fashion_trend.catalog.readers"},
    )

    assert offenders == [
        f"{module_path}: fashion_trend.catalog.graph",
        f"{module_path}: fashion_trend.catalog.graph.read_attribute_nodes",
    ]
```

- [ ] **Step 4: Run boundary tests**

Run:

```sh
uv run pytest tests/test_architecture_boundaries.py -q
```

Expected: PASS. `recommendation/` and `reports/` are still empty, so the new allowlist tests should pass while guarding future imports.

- [ ] **Step 5: Commit**

```sh
git add tests/test_architecture_boundaries.py
git commit -m "test: 收紧业务域公开依赖边界"
```

## Task 2: Create Domain Path Modules

**Files:**

- Create: `src/fashion_trend/datasets/paths.py`
- Create: `src/fashion_trend/transactions/paths.py`
- Create: `src/fashion_trend/catalog/paths.py`
- Create: `src/fashion_trend/trend/paths.py`
- Create: `src/fashion_trend/recommendation/paths.py`
- Create: `src/fashion_trend/reports/paths.py`
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
- Modify: `src/fashion_trend/datasets/download.py`
- Modify: `src/fashion_trend/trend/training.py`
- Modify: `src/fashion_trend/trend/evaluation.py`
- Modify: tests that import business path constants.

- [ ] **Step 1: Add path modules without removing old exports**

Create `src/fashion_trend/datasets/paths.py`:

```python
from pathlib import Path

from fashion_trend.foundation.paths import RAW_DIR

DEFAULT_COMPETITION = "h-and-m-personalized-fashion-recommendations"
RAW_HM_DIR = RAW_DIR / DEFAULT_COMPETITION
RAW_TRANSACTIONS_PATH = RAW_HM_DIR / "transactions_train.csv"
RAW_ARTICLES_PATH = RAW_HM_DIR / "articles.csv"
RAW_CUSTOMERS_PATH = RAW_HM_DIR / "customers.csv"
```

Create `src/fashion_trend/transactions/paths.py`:

```python
from pathlib import Path

from fashion_trend.foundation.paths import INTERIM_DIR

WEEKLY_TRANSACTIONS_PATH = INTERIM_DIR / "transactions_train_weekly.parquet"
```

Create `src/fashion_trend/catalog/paths.py`:

```python
from pathlib import Path

from fashion_trend.foundation.paths import INTERIM_DIR, PROCESSED_DIR

GRAPH_DIR = PROCESSED_DIR / "graph"
ARTICLES_CLEAN_MVP_PATH = INTERIM_DIR / "articles_clean_mvp.csv"
ARTICLES_CLEAN_PATH = INTERIM_DIR / "articles_clean.csv"
GRAPH_NODES_ARTICLE_PATH = GRAPH_DIR / "nodes_article.csv"
GRAPH_NODES_ATTRIBUTE_PATH = GRAPH_DIR / "nodes_attribute.csv"
GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH = GRAPH_DIR / "edges_article_attribute.csv"
GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH = GRAPH_DIR / "edges_attribute_hierarchy.csv"
```

Create `src/fashion_trend/trend/paths.py`:

```python
from pathlib import Path

from fashion_trend.foundation.paths import OUTPUT_DIR, PROCESSED_DIR

TREND_DIR = PROCESSED_DIR / "trend"
FEATURES_DIR = PROCESSED_DIR / "features"
OUTPUT_MODELS_DIR = OUTPUT_DIR / "models"
OUTPUT_METRICS_DIR = OUTPUT_DIR / "metrics"

TREND_ARTICLE_WEEK_SALES_PATH = TREND_DIR / "article_week_sales.csv"
TREND_ATTRIBUTE_WEEK_HEAT_PATH = TREND_DIR / "attribute_week_heat.csv"
TREND_ATTRIBUTE_WEEK_TARGET_PATH = TREND_DIR / "attribute_week_target.csv"
TREND_MODEL_SAMPLES_PATH = FEATURES_DIR / "trend_model_samples.parquet"
TREND_MODEL_SAMPLES_TRAIN_PATH = FEATURES_DIR / "trend_model_samples_train.parquet"
TREND_MODEL_SAMPLES_VALID_PATH = FEATURES_DIR / "trend_model_samples_valid.parquet"
TREND_MODEL_SAMPLES_TEST_PATH = FEATURES_DIR / "trend_model_samples_test.parquet"
TREND_MODEL_SAMPLES_SPLIT_METADATA_PATH = (
    FEATURES_DIR / "trend_model_samples_split_metadata.json"
)
TREND_SPLIT_VALID_WEEKS = 8
TREND_SPLIT_TEST_WEEKS = 8
```

Create `src/fashion_trend/recommendation/paths.py`:

```python
from pathlib import Path

from fashion_trend.foundation.paths import OUTPUT_DIR, PROCESSED_DIR

RECOMMEND_DIR = PROCESSED_DIR / "recommend"
RECOMMEND_FEATURES_DIR = RECOMMEND_DIR / "features"
OUTPUT_RECOMMENDATION_DIR = OUTPUT_DIR / "recommendation"

USER_PROFILE_PATH = RECOMMEND_DIR / "user_profile.parquet"
RECOMMEND_CANDIDATES_PATH = RECOMMEND_DIR / "candidate_items.parquet"
RECOMMENDATION_RESULT_PATH = OUTPUT_RECOMMENDATION_DIR / "recommendation_result.csv"
RECOMMENDATION_METRICS_PATH = OUTPUT_RECOMMENDATION_DIR / "recommendation_metrics.json"
```

Create `src/fashion_trend/reports/paths.py`:

```python
from pathlib import Path

from fashion_trend.foundation.paths import OUTPUT_DIR

OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_FIGURES_DIR = OUTPUT_REPORTS_DIR / "figures"
OUTPUT_TABLES_DIR = OUTPUT_REPORTS_DIR / "tables"
OUTPUT_CASE_STUDIES_DIR = OUTPUT_REPORTS_DIR / "case_studies"
```

- [ ] **Step 2: Migrate import sites from `PATH[...]` to domain constants**

Replace each `PATH[...]` access with the explicit domain constant. Examples:

```python
# src/02_build_weekly_transactions.py
from fashion_trend.datasets.paths import RAW_TRANSACTIONS_PATH
from fashion_trend.transactions.paths import WEEKLY_TRANSACTIONS_PATH
```

```python
# src/06_compute_attribute_week_heat.py
from fashion_trend.catalog.paths import (
    GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH,
    GRAPH_NODES_ATTRIBUTE_PATH,
)
from fashion_trend.trend.paths import (
    TREND_ARTICLE_WEEK_SALES_PATH,
    TREND_ATTRIBUTE_WEEK_HEAT_PATH,
)
```

```python
# src/fashion_trend/trend/training.py
from fashion_trend.trend.paths import (
    OUTPUT_MODELS_DIR,
    TREND_MODEL_SAMPLES_TEST_PATH,
    TREND_MODEL_SAMPLES_TRAIN_PATH,
    TREND_MODEL_SAMPLES_VALID_PATH,
)
```

`default_trend_model_input_paths()` should return:

```python
return {
    "train": TREND_MODEL_SAMPLES_TRAIN_PATH,
    "valid": TREND_MODEL_SAMPLES_VALID_PATH,
    "test": TREND_MODEL_SAMPLES_TEST_PATH,
}
```

- [ ] **Step 3: Update tests importing path constants**

Move imports such as:

```python
from fashion_trend.foundation.paths import OUTPUT_MODELS_DIR
```

to:

```python
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
```

- [ ] **Step 4: Verify no production or test file still consumes global `PATH`**

Run:

```sh
rg -n "PATH\\[|from fashion_trend\\.foundation\\.paths import .*PATH|OUTPUT_MODELS_DIR|OUTPUT_METRICS_DIR|GRAPH_DIR|RAW_HM_DIR|DEFAULT_COMPETITION|TREND_SPLIT" src tests
```

Expected: no `PATH[` references. Business constants may still appear only in their new owning domain path modules and their intended import sites.

- [ ] **Step 5: Run focused tests**

Run:

```sh
uv run pytest tests/test_architecture_boundaries.py tests/test_trend_training.py tests/test_trend_evaluation.py tests/test_attribute_graph.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```sh
git add src tests
git commit -m "refactor: 迁移业务路径到领域模块"
```

## Task 3: Retire Business Exports from `foundation.paths`

**Files:**

- Modify: `src/fashion_trend/foundation/paths.py`
- Modify: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: Add the `foundation.paths` export whitelist test**

Add to `tests/test_architecture_boundaries.py`:

```python
FOUNDATION_PATH_ALLOWED_EXPORTS = {
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "OUTPUT_DIR",
}


def test_foundation_paths_exports_only_project_roots() -> None:
    import fashion_trend.foundation.paths as paths

    exported_names = {
        name
        for name in vars(paths)
        if name.isupper() and not name.startswith("_")
    }
    assert exported_names == FOUNDATION_PATH_ALLOWED_EXPORTS
```

- [ ] **Step 2: Shrink `foundation.paths`**

Replace `src/fashion_trend/foundation/paths.py` with:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
```

- [ ] **Step 3: Run boundary and path import scans**

Run:

```sh
uv run pytest tests/test_architecture_boundaries.py -q
rg -n "PATH\\[|DEFAULT_COMPETITION|RAW_HM_DIR|GRAPH_DIR|TREND_DIR|FEATURES_DIR|OUTPUT_MODELS_DIR|OUTPUT_METRICS_DIR|OUTPUT_FIGURES_DIR|OUTPUT_REPORTS_DIR|TREND_SPLIT" src/fashion_trend/foundation src tests
```

Expected:

- pytest PASS.
- `rg` only finds business path names in domain path modules, scripts importing those domain constants, tests that intentionally assert domain path behavior, or documentation strings inside tests. It must not find business exports in `src/fashion_trend/foundation/paths.py`.

- [ ] **Step 4: Run full tests**

Run:

```sh
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```sh
git add src/fashion_trend/foundation/paths.py tests/test_architecture_boundaries.py src tests
git commit -m "refactor: 退役 foundation 业务路径"
```

## Task 4: Establish Public `contracts` and `readers`

**Files:**

- Create: `src/fashion_trend/transactions/contracts.py`
- Create: `src/fashion_trend/transactions/readers.py`
- Create: `src/fashion_trend/catalog/contracts.py`
- Create: `src/fashion_trend/catalog/readers.py`
- Create: `src/fashion_trend/trend/readers.py`
- Create: `src/fashion_trend/recommendation/contracts.py`
- Create: `src/fashion_trend/recommendation/readers.py`
- Modify: `src/fashion_trend/transactions/weekly.py`
- Modify: `src/fashion_trend/catalog/graph.py`
- Modify: `src/fashion_trend/trend/schema.py`
- Modify: `src/fashion_trend/trend/article_sales.py`
- Modify: `src/fashion_trend/trend/attribute_heat.py`
- Modify: `src/fashion_trend/trend/samples.py`
- Modify: `src/05_compute_article_week_sales.py`
- Modify: `src/06_compute_attribute_week_heat.py`
- Modify: `src/07_build_trend_targets.py`
- Modify: `src/08_build_trend_model_samples.py`
- Modify: tests that import reader functions from internal build modules.

- [ ] **Step 1: Move stable transaction contracts**

Create `transactions/contracts.py`:

```python
WEEKLY_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "customer_id",
    "price",
)
```

This is the same stable weekly transaction contract currently defined in `trend.schema`. After creating it:

```python
# src/fashion_trend/trend/article_sales.py
from fashion_trend.transactions.contracts import WEEKLY_TRANSACTION_COLUMNS
```

Remove `WEEKLY_TRANSACTION_COLUMNS` from `src/fashion_trend/trend/schema.py`. The trend domain may consume the transaction public contract, but it must not keep a duplicate transaction schema.

Create `transactions/readers.py` with the current `read_weekly_transactions()` implementation. Update `transactions/weekly.py` to import `WEEKLY_TRANSACTION_COLUMNS` from `transactions.contracts` and remove its old reader-only column constant.

- [ ] **Step 2: Move stable catalog graph contracts and readers**

Create `catalog/contracts.py` with:

```python
ARTICLE_ATTRIBUTE_EDGE_COLUMNS: tuple[str, ...] = (
    "article_id",
    "attr_id",
    "attr_type",
    "attr_value",
)

ARTICLE_ATTRIBUTE_EDGE_DTYPES: dict[str, str] = {
    "article_id": "string",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
}

ATTRIBUTE_NODE_COLUMNS: tuple[str, ...] = (
    "attr_id",
    "attr_type",
    "attr_value",
    "article_count",
    "is_core_attr",
)

ATTRIBUTE_NODE_DTYPES: dict[str, str] = {
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "article_count": "int64",
    "is_core_attr": "int64",
}

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
```

These replace the upstream catalog contracts currently duplicated in `trend.schema`:

```text
ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS
ARTICLE_ATTRIBUTE_EDGE_HEAT_DTYPES
ATTRIBUTE_NODE_HEAT_COLUMNS
ATTRIBUTE_NODE_HEAT_DTYPES
ATTRIBUTE_HIERARCHY_EDGE_COLUMNS
ATTRIBUTE_HIERARCHY_EDGE_DTYPES
```

After creating `catalog.contracts`, update trend internals to consume the upstream public contracts:

```python
# src/fashion_trend/trend/attribute_heat.py
from fashion_trend.catalog.contracts import (
    ARTICLE_ATTRIBUTE_EDGE_COLUMNS,
    ATTRIBUTE_NODE_COLUMNS,
)

# src/fashion_trend/trend/samples.py
from fashion_trend.catalog.contracts import ATTRIBUTE_HIERARCHY_EDGE_COLUMNS
```

Then remove the duplicated upstream catalog contract names from `src/fashion_trend/trend/schema.py`.

Create `catalog/readers.py` by moving these functions out of `catalog.graph`:

```python
read_article_attribute_edges()
read_attribute_nodes()
read_attribute_hierarchy_edges()
```

Keep the same exception messages and dtype behavior.

- [ ] **Step 3: Add trend readers without wrapping internal builders**

Create `trend/readers.py` and move or re-export stable read-only functions:

```python
from __future__ import annotations

import json
from pathlib import Path

from fashion_trend.trend.article_sales import read_article_week_sales
from fashion_trend.trend.attribute_heat import read_attribute_week_heat
from fashion_trend.trend.evaluation import read_trend_model_predictions
from fashion_trend.trend.splits import read_trend_model_split
from fashion_trend.trend.targets import read_attribute_week_target
from fashion_trend.trend.schema import TREND_METRICS_PAYLOAD_REQUIRED_KEYS


def read_trend_metrics(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"趋势评价指标文件不存在: {metrics_path}")
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取趋势评价指标文件: {metrics_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"趋势评价指标文件必须是 JSON object: {metrics_path}")
    missing_keys = sorted(set(TREND_METRICS_PAYLOAD_REQUIRED_KEYS) - set(payload))
    if missing_keys:
        raise ValueError(
            "趋势评价指标文件缺少必要字段: "
            + ", ".join(missing_keys)
            + f"。文件: {metrics_path}"
        )
    return payload
```

Do not expose `build_*`, `validate_*`, `run_*`, `compute_*`, `train_*`, or `evaluate_*` functions from `trend.readers`.

Add the public metrics payload contract to `trend.schema`:

```python
TREND_METRICS_PAYLOAD_REQUIRED_KEYS: tuple[str, ...] = (
    "model_name",
    "prediction_path",
    "output_path",
    "evaluated_splits",
    "ranking",
    "overall",
    "by_attr_type",
    "groups",
)
```

`reports` must use `trend.readers.read_trend_metrics()` for `trend_metrics.json`; it must not import `trend.evaluation` or duplicate JSON parsing.

- [ ] **Step 4: Add empty recommendation public modules**

Create `recommendation/contracts.py`:

```python
RECOMMENDATION_TOP_K = 12
```

Create `recommendation/readers.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_recommendation_result(result_path: Path) -> pd.DataFrame:
    if not result_path.exists():
        raise FileNotFoundError(f"推荐结果文件不存在: {result_path}")
    return pd.read_csv(result_path)
```

This reader is intentionally minimal and read-only. It must not build candidates or compute metrics.

- [ ] **Step 5: Update cross-domain imports**

Use public readers at domain boundaries:

```python
# src/05_compute_article_week_sales.py
from fashion_trend.transactions.readers import read_weekly_transactions
```

```python
# src/06_compute_attribute_week_heat.py
from fashion_trend.catalog.readers import (
    read_article_attribute_edges,
    read_attribute_nodes,
)
```

```python
# src/08_build_trend_model_samples.py
from fashion_trend.catalog.readers import (
    read_attribute_hierarchy_edges,
    read_attribute_nodes,
)
```

- [ ] **Step 6: Update tests**

Reader tests should import from public reader modules:

```python
from fashion_trend.catalog.readers import read_article_attribute_edges
from fashion_trend.transactions.readers import read_weekly_transactions
```

Builder tests should continue to import builder functions from their implementation modules until Task 5 splits `catalog.graph`.

- [ ] **Step 7: Verify**

Run:

```sh
uv run pytest tests/test_architecture_boundaries.py tests/test_trend_article_sales.py tests/test_trend_attribute_heat.py tests/test_trend_samples.py tests/test_attribute_graph.py -q
rg -n "WEEKLY_TRANSACTION_COLUMNS|ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS|ARTICLE_ATTRIBUTE_EDGE_HEAT_DTYPES|ATTRIBUTE_NODE_HEAT_COLUMNS|ATTRIBUTE_NODE_HEAT_DTYPES|ATTRIBUTE_HIERARCHY_EDGE_COLUMNS|ATTRIBUTE_HIERARCHY_EDGE_DTYPES" src/fashion_trend/trend/schema.py
```

Expected:

- pytest PASS.
- `rg` returns no matches for upstream transaction/catalog contract names in `trend.schema.py` (exit code 1 with no output is acceptable here); those names now live in `transactions.contracts` and `catalog.contracts`.

- [ ] **Step 8: Commit**

```sh
git add src tests
git commit -m "refactor: 建立跨领域只读接口"
```

## Task 5: Split `catalog` Graph Responsibilities

**Files:**

- Delete in the same patch before adding the directory: `src/fashion_trend/catalog/graph.py`
- Create: `src/fashion_trend/catalog/graph/__init__.py`
- Create: `src/fashion_trend/catalog/graph/schema.py`
- Create: `src/fashion_trend/catalog/graph/builders.py`
- Create: `src/fashion_trend/catalog/graph/publishing.py`
- Modify: `src/fashion_trend/catalog/readers.py`
- Modify: `src/04_build_attribute_graph.py`
- Modify: `tests/test_attribute_graph.py`
- Modify: `tests/test_architecture_boundaries.py` only if its fixture still imports the old file module.

- [ ] **Step 1: Replace `catalog/graph.py` with `catalog/graph/` in one patch**

`src/fashion_trend/catalog/graph.py` already exists, so `src/fashion_trend/catalog/graph/` cannot be created until the file path is removed. Perform this as one atomic edit:

```text
Delete File: src/fashion_trend/catalog/graph.py
Add File: src/fashion_trend/catalog/graph/__init__.py
Add File: src/fashion_trend/catalog/graph/schema.py
Add File: src/fashion_trend/catalog/graph/builders.py
Add File: src/fashion_trend/catalog/graph/publishing.py
```

Do not commit after deleting `graph.py` but before adding the package files and updating imports.

- [ ] **Step 2: Move graph constants to `graph/schema.py`**

Move these definitions from `catalog.graph`:

```python
LEVEL_BY_ATTRIBUTE
HIERARCHY_RELATIONS
GRAPH_OUTPUT_FILENAMES
make_attr_id()
make_article_node_id()
make_edge_type()
```

`graph/schema.py` must not import pandas or write files.

- [ ] **Step 3: Move build functions to `graph/builders.py`**

Move:

```python
build_article_nodes()
build_attribute_nodes()
build_article_attribute_edges()
build_attribute_hierarchy_edges()
validate_graph_references()
build_attribute_graph_frames()
```

Keep `build_attribute_graph_frames()` returning the same four keys:

```python
{
    "nodes_article": nodes_article,
    "nodes_attribute": nodes_attribute,
    "edges_article_attribute": edges_article_attribute,
    "edges_attribute_hierarchy": edges_attribute_hierarchy,
}
```

- [ ] **Step 4: Move publishing functions to `graph/publishing.py`**

Move:

```python
cleanup_graph_publish_files()
rollback_graph_outputs()
write_graph_frame_temp()
publish_graph_frames()
```

Keep rollback semantics and CSV quoting exactly as current tests assert.

- [ ] **Step 5: Keep the CLI entry function public through `graph/__init__.py`**

Implement `build_attribute_graph_files()` in `graph/__init__.py`:

```python
from pathlib import Path

from fashion_trend.catalog.graph.builders import build_attribute_graph_frames
from fashion_trend.catalog.graph.publishing import publish_graph_frames
from fashion_trend.catalog.readers import read_clean_articles


def build_attribute_graph_files(clean_articles_path: Path, graph_dir: Path) -> dict[str, int]:
    clean_articles = read_clean_articles(clean_articles_path)
    graph_frames = build_attribute_graph_frames(clean_articles)
    graph_dir.mkdir(parents=True, exist_ok=True)
    publish_graph_frames(graph_frames, graph_dir)
    return {
        graph_name: len(graph_frame)
        for graph_name, graph_frame in graph_frames.items()
    }
```

- [ ] **Step 6: Move `read_clean_articles()` to `catalog/readers.py`**

`catalog/readers.py` owns read-only access to catalog stable artifacts. Keep `read_clean_articles()` behavior unchanged.

- [ ] **Step 7: Update tests and imports**

Use these imports:

```python
from fashion_trend.catalog.graph import build_attribute_graph_files
from fashion_trend.catalog.graph.builders import (
    build_article_attribute_edges,
    build_article_nodes,
    build_attribute_graph_frames,
    build_attribute_hierarchy_edges,
    build_attribute_nodes,
)
from fashion_trend.catalog.graph.publishing import publish_graph_frames
```

- [ ] **Step 8: Verify**

Run:

```sh
uv run pytest tests/test_attribute_graph.py tests/test_architecture_boundaries.py -q
uv run python -m py_compile src/fashion_trend/catalog/graph/__init__.py src/fashion_trend/catalog/graph/schema.py src/fashion_trend/catalog/graph/builders.py src/fashion_trend/catalog/graph/publishing.py src/fashion_trend/catalog/readers.py
```

Expected: PASS and compile succeeds.

- [ ] **Step 9: Commit**

```sh
git add src/fashion_trend/catalog src/04_build_attribute_graph.py tests/test_attribute_graph.py tests/test_architecture_boundaries.py
git commit -m "refactor: 拆分 catalog 图构建职责"
```

## Task 6: Split Trend Deterministic Pipeline

**Files:**

- Create: `src/fashion_trend/trend/heat/__init__.py`
- Create: `src/fashion_trend/trend/heat/article_sales.py`
- Create: `src/fashion_trend/trend/heat/attribute_heat.py`
- Create: `src/fashion_trend/trend/labels/__init__.py`
- Create: `src/fashion_trend/trend/labels/targets.py`
- Create: `src/fashion_trend/trend/features/__init__.py`
- Create: `src/fashion_trend/trend/features/samples.py`
- Delete in the same patch before adding the directory: `src/fashion_trend/trend/splits.py`
- Create: `src/fashion_trend/trend/splits/__init__.py`
- Create: `src/fashion_trend/trend/splits/time_split.py`
- Delete after imports are migrated: old file modules `article_sales.py`, `attribute_heat.py`, `targets.py`, `samples.py`
- Modify: `src/fashion_trend/trend/readers.py`
- Modify: `src/05_compute_article_week_sales.py`
- Modify: `src/06_compute_attribute_week_heat.py`
- Modify: `src/07_build_trend_targets.py`
- Modify: `src/08_build_trend_model_samples.py`
- Modify: `src/09_split_trend_model_samples.py`
- Modify: trend tests importing deterministic modules.

- [ ] **Step 1: Move article sales and attribute heat into `trend/heat/`**

Move the current contents as:

```text
src/fashion_trend/trend/article_sales.py -> src/fashion_trend/trend/heat/article_sales.py
src/fashion_trend/trend/attribute_heat.py -> src/fashion_trend/trend/heat/attribute_heat.py
```

Update CLI imports to:

```python
from fashion_trend.trend.heat.article_sales import (
    build_article_week_sales_frame,
    validate_article_week_sales,
)
from fashion_trend.trend.heat.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_attribute_week_heat,
)
```

- [ ] **Step 2: Move labels and features**

Move:

```text
src/fashion_trend/trend/targets.py -> src/fashion_trend/trend/labels/targets.py
src/fashion_trend/trend/samples.py -> src/fashion_trend/trend/features/samples.py
```

Update CLI imports accordingly.

- [ ] **Step 3: Replace `trend/splits.py` with `trend/splits/` in one patch**

`src/fashion_trend/trend/splits.py` already exists, so `src/fashion_trend/trend/splits/` cannot be created while the file exists. Perform this as one atomic edit:

```text
Delete File: src/fashion_trend/trend/splits.py
Add File: src/fashion_trend/trend/splits/__init__.py
Add File: src/fashion_trend/trend/splits/time_split.py
```

Move the old `splits.py` contents into `splits/time_split.py`.

Expose only the functions used by CLI/training through `trend/splits/__init__.py`:

```python
from fashion_trend.trend.splits.time_split import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    read_trend_model_split,
    validate_trend_model_split_frame,
    validate_trend_model_split_frames,
)

__all__ = [
    "build_trend_model_split_frames",
    "build_trend_model_split_metadata",
    "read_trend_model_split",
    "validate_trend_model_split_frame",
    "validate_trend_model_split_frames",
]
```

- [ ] **Step 4: Update `trend.readers`**

After moving modules, `trend/readers.py` should import read-only functions from the new locations:

```python
from fashion_trend.trend.heat.article_sales import read_article_week_sales
from fashion_trend.trend.heat.attribute_heat import read_attribute_week_heat
from fashion_trend.trend.labels.targets import read_attribute_week_target
from fashion_trend.trend.splits import read_trend_model_split
```

Do not expose builders through `trend.readers`.

- [ ] **Step 5: Verify no code uses old module paths**

Run:

```sh
test ! -f src/fashion_trend/trend/article_sales.py
test ! -f src/fashion_trend/trend/attribute_heat.py
test ! -f src/fashion_trend/trend/targets.py
test ! -f src/fashion_trend/trend/samples.py
test ! -f src/fashion_trend/trend/splits.py
rg -n "fashion_trend\\.trend\\.(article_sales|attribute_heat|targets|samples)" src tests
rg -n "fashion_trend\\.trend\\.splits\\.time_split" src/fashion_trend/recommendation src/fashion_trend/reports
```

Expected:

- all `test ! -f` checks exit 0.
- first `rg` returns no matches; old file-module paths are gone.
- second `rg` returns no matches in `recommendation` or `reports`; package-level imports from `fashion_trend.trend.splits` remain allowed for CLI, trend internals, and tests.

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_article_sales.py tests/test_trend_attribute_heat.py tests/test_trend_targets.py tests/test_trend_samples.py tests/test_trend_splits.py tests/test_architecture_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```sh
git add src/fashion_trend/trend src/05_compute_article_week_sales.py src/06_compute_attribute_week_heat.py src/07_build_trend_targets.py src/08_build_trend_model_samples.py src/09_split_trend_model_samples.py tests
git commit -m "refactor: 收敛趋势确定性流水线"
```

## Task 7: Split Trend Experiments

**Files:**

- Create: `src/fashion_trend/trend/models/baselines/__init__.py`
- Create: `src/fashion_trend/trend/models/baselines/last_week.py`
- Create: `src/fashion_trend/trend/models/baselines/moving_average.py`
- Create: `src/fashion_trend/trend/models/supervised/__init__.py`
- Delete after imports are migrated: `src/fashion_trend/trend/models/last_week.py`
- Delete after imports are migrated: `src/fashion_trend/trend/models/moving_average.py`
- Modify: `src/fashion_trend/trend/models/registry.py`
- Delete in the same patch before adding the directory: `src/fashion_trend/trend/training.py`
- Create: `src/fashion_trend/trend/training/__init__.py`
- Create: `src/fashion_trend/trend/training/outputs.py`
- Create: `src/fashion_trend/trend/training/runner.py`
- Delete in the same patch before adding the directory: `src/fashion_trend/trend/evaluation.py`
- Create: `src/fashion_trend/trend/evaluation/__init__.py`
- Create: `src/fashion_trend/trend/evaluation/metrics.py`
- Create: `src/fashion_trend/trend/evaluation/payloads.py`
- Create: `src/fashion_trend/trend/evaluation/runner.py`
- Modify: `src/10_train_trend_model.py`
- Modify: `src/11_eval_trend_model.py`
- Modify: `src/fashion_trend/trend/readers.py`
- Modify: `tests/test_trend_training.py`
- Modify: `tests/test_trend_evaluation.py`

- [ ] **Step 1: Move baseline models**

Move files:

```text
trend/models/last_week.py -> trend/models/baselines/last_week.py
trend/models/moving_average.py -> trend/models/baselines/moving_average.py
```

Update `registry.py` imports:

```python
from fashion_trend.trend.models.baselines.last_week import (
    LAST_WEEK_MODEL_NAME,
    LastWeekTrainer,
)
from fashion_trend.trend.models.baselines.moving_average import (
    MOVING_AVERAGE_MODEL_NAME,
    MovingAverageTrainer,
)
```

Create `models/supervised/__init__.py` with:

```python
"""Supervised trend model implementations."""
```

- [ ] **Step 2: Replace `trend/training.py` with `trend/training/` in one patch**

`src/fashion_trend/trend/training.py` already exists, so create the package and delete the file in one atomic edit:

```text
Delete File: src/fashion_trend/trend/training.py
Add File: src/fashion_trend/trend/training/__init__.py
Add File: src/fashion_trend/trend/training/outputs.py
Add File: src/fashion_trend/trend/training/runner.py
```

Do not commit after deleting `training.py` but before adding the package files and updating imports.

- [ ] **Step 3: Split training outputs from runner**

Move these functions to `trend/training/outputs.py`:

```python
derive_trend_model_output_paths()
validate_trend_train_result()
build_trend_train_metadata()
write_trend_model_outputs()
```

Move private output helpers used only by those functions into `outputs.py` as well.

Move these functions to `trend/training/runner.py`:

```python
default_trend_model_input_paths()
read_trend_model_split_frames()
run_trend_model_training()
```

`runner.py` imports output helpers from `outputs.py`.

`trend/training/__init__.py` exports the CLI/public training entry points:

```python
from fashion_trend.trend.training.outputs import (
    build_trend_train_metadata,
    derive_trend_model_output_paths,
    validate_trend_train_result,
    write_trend_model_outputs,
)
from fashion_trend.trend.training.runner import run_trend_model_training

__all__ = [
    "build_trend_train_metadata",
    "derive_trend_model_output_paths",
    "run_trend_model_training",
    "validate_trend_train_result",
    "write_trend_model_outputs",
]
```

- [ ] **Step 4: Replace `trend/evaluation.py` with `trend/evaluation/` in one patch**

`src/fashion_trend/trend/evaluation.py` already exists, so create the package and delete the file in one atomic edit:

```text
Delete File: src/fashion_trend/trend/evaluation.py
Add File: src/fashion_trend/trend/evaluation/__init__.py
Add File: src/fashion_trend/trend/evaluation/metrics.py
Add File: src/fashion_trend/trend/evaluation/payloads.py
Add File: src/fashion_trend/trend/evaluation/runner.py
```

Do not commit after deleting `evaluation.py` but before adding the package files and updating imports.

- [ ] **Step 5: Split evaluation metrics, payloads, and runner**

Move these functions and constants to `trend/evaluation/metrics.py`:

```python
TREND_EVALUATION_SPLITS
TREND_EVALUATION_K_VALUES
TREND_EVALUATION_GROUP_COLUMNS
TREND_EVALUATION_TARGET_COLUMN
TREND_EVALUATION_PREDICTION_COLUMN
compute_trend_group_metrics()
compute_trend_metrics()
```

Move metric private helpers to `metrics.py`.

Move these functions to `trend/evaluation/payloads.py`:

```python
derive_trend_metric_output_paths()
read_trend_model_predictions()
validate_trend_model_predictions_for_evaluation()
build_trend_metrics_payload()
write_trend_metrics()
```

Move `run_trend_model_evaluation()` to `trend/evaluation/runner.py`.

`trend/evaluation/__init__.py` exports the stable public functions currently imported by CLI and tests.

- [ ] **Step 6: Update imports**

CLI remains stable:

```python
from fashion_trend.trend.training import run_trend_model_training
from fashion_trend.trend.evaluation import run_trend_model_evaluation
```

Tests can import focused internals where they are testing internals:

```python
from fashion_trend.trend.training.outputs import derive_trend_model_output_paths
from fashion_trend.trend.evaluation.metrics import compute_trend_group_metrics
```

- [ ] **Step 7: Verify old experiment paths are gone**

Run:

```sh
test ! -f src/fashion_trend/trend/models/last_week.py
test ! -f src/fashion_trend/trend/models/moving_average.py
test ! -f src/fashion_trend/trend/training.py
test ! -f src/fashion_trend/trend/evaluation.py
rg -n "trend\\.models\\.(last_week|moving_average)" src tests
rg -n "fashion_trend\\.trend\\.training\\.(runner|outputs)|fashion_trend\\.trend\\.evaluation\\.(metrics|payloads|runner)" src/10_train_trend_model.py src/11_eval_trend_model.py src/fashion_trend/recommendation src/fashion_trend/reports
```

Expected:

- all `test ! -f` checks exit 0.
- first `rg` returns no stale model file imports.
- second `rg` returns no CLI, recommendation, or reports imports that bypass the package public entrypoints. CLI may still import `run_trend_model_training` and `run_trend_model_evaluation` from `fashion_trend.trend.training` / `fashion_trend.trend.evaluation` package `__init__.py`; tests may import focused internals when they are testing those internals.

- [ ] **Step 8: Run focused and full tests**

Run:

```sh
uv run pytest tests/test_trend_training.py tests/test_trend_evaluation.py tests/test_architecture_boundaries.py -q
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```sh
git add src/fashion_trend/trend src/10_train_trend_model.py src/11_eval_trend_model.py tests/test_trend_training.py tests/test_trend_evaluation.py tests/test_architecture_boundaries.py
git commit -m "refactor: 拆分趋势实验层职责"
```

## Task 8: Lock Recommendation and Reports Boundaries, Then Sync Docs

**Files:**

- Modify: `src/fashion_trend/recommendation/contracts.py`
- Modify: `src/fashion_trend/recommendation/readers.py`
- Modify: `src/fashion_trend/reports/paths.py`
- Modify: `tests/test_architecture_boundaries.py`
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`

- [ ] **Step 1: Add explicit no-core-computation import fixtures**

Add this regression test to prove core computation imports fail the allowlist without importing those modules at runtime:

```python
def test_allowlist_rejects_core_computation_imports(tmp_path) -> None:
    package_root = tmp_path / "src" / "fashion_trend"
    module_path = package_root / "reports" / "summary.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "\n".join(
            [
                "from fashion_trend.transactions.weekly import build_weekly_transactions",
                "from fashion_trend.catalog.graph.builders import build_attribute_nodes",
                "from fashion_trend.trend.features.samples import build_trend_model_samples_frame",
                "from fashion_trend.trend.training import run_trend_model_training",
                "from fashion_trend.trend.evaluation import run_trend_model_evaluation",
            ]
        ),
        encoding="utf-8",
    )

    offenders = package_upstream_import_offenders(
        [module_path],
        {
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
        },
        REPORTS_PUBLIC_IMPORTS,
    )

    assert offenders == [
        f"{module_path}: fashion_trend.catalog.graph.builders",
        f"{module_path}: fashion_trend.catalog.graph.builders.build_attribute_nodes",
        f"{module_path}: fashion_trend.transactions.weekly",
        f"{module_path}: fashion_trend.transactions.weekly.build_weekly_transactions",
        f"{module_path}: fashion_trend.trend.evaluation",
        f"{module_path}: fashion_trend.trend.evaluation.run_trend_model_evaluation",
        f"{module_path}: fashion_trend.trend.features.samples",
        f"{module_path}: fashion_trend.trend.features.samples.build_trend_model_samples_frame",
        f"{module_path}: fashion_trend.trend.training",
        f"{module_path}: fashion_trend.trend.training.run_trend_model_training",
    ]
```

- [ ] **Step 2: Keep recommendation public modules read-only**

`recommendation/contracts.py` may define stable output column names and `RECOMMENDATION_TOP_K`. It must not import `transactions`, `catalog`, or `trend`.

`recommendation/readers.py` may import `recommendation.contracts` and pandas only. It must not import upstream domains.

- [ ] **Step 3: Keep reports path ownership under reports**

Ensure `reports/paths.py` owns only `outputs/reports/` descendants:

```python
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_FIGURES_DIR = OUTPUT_REPORTS_DIR / "figures"
OUTPUT_TABLES_DIR = OUTPUT_REPORTS_DIR / "tables"
OUTPUT_CASE_STUDIES_DIR = OUTPUT_REPORTS_DIR / "case_studies"
```

- [ ] **Step 4: Sync README architecture notes**

Update README so it says:

```text
默认路径根常量位于 foundation.paths；数据集、交易、catalog、trend、recommendation 和 reports 的业务路径由各自领域的 paths.py 持有。
```

Also ensure README keeps these user commands stable:

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model moving_average
```

- [ ] **Step 5: Sync `docs/gpt-research/implementation-plan.md`**

Correct these known drifts:

```text
LightGBM 通过 src/10_train_trend_model.py --model lightgbm 运行，不新增 12_train_lightgbm_trend_model.py。
12_build_user_profile.py 到 15_eval_recommendations.py 属于 recommendation。
16_make_reports.py 覆盖 figures、tables、case studies；不再使用 16_make_figures.py 作为目标命名。
推荐特征写入 data/processed/recommend/ 或其子目录，不写入 data/processed/features/。
```

- [ ] **Step 6: Run documentation and architecture scans**

Run:

```sh
rg -n "12_train_lightgbm|16_make_figures|data/processed/features/.+recommend|from fashion_trend\\.trend import" README.md docs/gpt-research/implementation-plan.md src tests
uv run pytest tests/test_architecture_boundaries.py -q
```

Expected:

- `rg` finds no stale target naming except explanatory historical references that explicitly say they are obsolete.
- architecture tests PASS.

- [ ] **Step 7: Run final verification**

Run:

```sh
uv run pytest -q
uv run python -m compileall -q src
git diff --check
```

Expected: tests PASS, compile succeeds, diff check has no output.

- [ ] **Step 8: Run real artifact smoke tests when inputs exist**

If the local workspace has the required upstream data files, run the artifact-preserving smoke chain:

```sh
uv run python src/04_build_attribute_graph.py
uv run python src/05_compute_article_week_sales.py
uv run python src/06_compute_attribute_week_heat.py
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

Expected:

```text
data/processed/graph/nodes_article.csv exists
data/processed/graph/nodes_attribute.csv exists
data/processed/graph/edges_article_attribute.csv exists
data/processed/graph/edges_attribute_hierarchy.csv exists
data/processed/trend/article_week_sales.csv exists
data/processed/trend/attribute_week_heat.csv exists
outputs/models/moving_average/predictions.csv exists
outputs/models/moving_average/metadata.json exists
outputs/models/moving_average/params.json exists
outputs/metrics/moving_average/trend_metrics.json exists
```

If any required upstream data is missing, do not treat the smoke test as passing. Record the exact missing file path and the reason the CLI smoke chain was skipped.

- [ ] **Step 9: Check artifact schemas, row counts, distributions, and checksums**

After the smoke chain runs, compare current artifact summaries against the pre-migration baseline captured before Task 2:

```sh
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path

import pandas as pd

BASELINE_PATH = Path("/tmp/fashion_domain_arch_artifact_baseline.json")
if not BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"pre-migration artifact baseline missing: {BASELINE_PATH}"
    )

TREND_METRICS_KEYS = {
    "model_name",
    "prediction_path",
    "output_path",
    "evaluated_splits",
    "ranking",
    "overall",
    "by_attr_type",
    "groups",
}
CSV_ARTIFACTS = {
    "graph_nodes_article": Path("data/processed/graph/nodes_article.csv"),
    "graph_nodes_attribute": Path("data/processed/graph/nodes_attribute.csv"),
    "graph_edges_article_attribute": Path(
        "data/processed/graph/edges_article_attribute.csv"
    ),
    "graph_edges_attribute_hierarchy": Path(
        "data/processed/graph/edges_attribute_hierarchy.csv"
    ),
    "article_week_sales": Path("data/processed/trend/article_week_sales.csv"),
    "attribute_week_heat": Path("data/processed/trend/attribute_week_heat.csv"),
    "moving_average_predictions": Path(
        "outputs/models/moving_average/predictions.csv"
    ),
}
JSON_ARTIFACTS = {
    "moving_average_metadata": Path("outputs/models/moving_average/metadata.json"),
    "moving_average_params": Path("outputs/models/moving_average/params.json"),
    "moving_average_metrics": Path(
        "outputs/metrics/moving_average/trend_metrics.json"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_summary() -> dict[str, object]:
    summary: dict[str, object] = {}
    for name, path in CSV_ARTIFACTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.empty:
            raise AssertionError(f"{name} is empty")
        summary[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
        }
        if "attr_type" in frame.columns:
            summary[name]["rows_by_attr_type"] = {
                str(key): int(value)
                for key, value in frame["attr_type"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            }
        if "week_id" in frame.columns:
            week_ids = pd.to_numeric(frame["week_id"], errors="raise")
            summary[name]["week_id"] = {
                "min": int(week_ids.min()),
                "max": int(week_ids.max()),
                "nunique": int(week_ids.nunique()),
            }
    for name, path in JSON_ARTIFACTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "moving_average_metrics":
            missing = sorted(TREND_METRICS_KEYS - set(payload))
            if missing:
                raise AssertionError(f"trend_metrics.json missing keys: {missing}")
            if payload["model_name"] != "moving_average":
                raise AssertionError("trend_metrics.json model_name changed")
        summary[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "keys": sorted(payload),
        }
    return summary


baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
current = collect_summary()
if current != baseline:
    changed = sorted(
        name
        for name in set(current) | set(baseline)
        if current.get(name) != baseline.get(name)
    )
    raise AssertionError("artifact drift detected: " + ", ".join(changed))
for name in sorted(current):
    entry = current[name]
    if "rows" in entry:
        print(name, entry["rows"], entry["sha256"])
    else:
        print(name, entry["sha256"])
PY
```

Expected: command exits 0 and prints exact baseline-matching row counts and checksums. If real data is unavailable and Step 8 was skipped, skip this check with the same explicit missing-file reason. If the baseline file is missing, the architecture work is not complete; recreate it only from the pre-migration commit or rerun this task from a clean baseline.

- [ ] **Step 10: Commit**

```sh
git add src tests README.md docs/gpt-research/implementation-plan.md
git commit -m "docs: 同步业务域架构实施边界"
```

## Final Acceptance

The architecture work is complete only when all checks below are true:

- `tests/test_architecture_boundaries.py` uses allowlist checks for `recommendation` and `reports`.
- `recommendation` can import upstream only through `transactions.contracts/readers`, `catalog.contracts/readers`, `trend.schema`, `trend.predictions`, and `trend.readers`.
- `reports` can import domains only through public `contracts` / `readers` / explicit trend allowlist.
- `foundation.paths` exports only `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`, and `OUTPUT_DIR`.
- No runtime code imports global `PATH`.
- No business path or business configuration remains in `foundation.paths`.
- `catalog.graph` is split into schema, builders, publishing, and public readers.
- Trend deterministic pipeline is split by heat, labels, features, and splits.
- Trend experiments are split into model registry/baselines/supervised, training runner/outputs, and evaluation metrics/payloads/runner.
- CLI commands `src/00_*.py` through `src/11_*.py` keep their user-facing behavior and remain readable flow indexes.
- Existing output paths and schemas are unchanged.
- `README.md` and `docs/gpt-research/implementation-plan.md` match the final architecture.
- `uv run pytest -q` passes.
- `uv run python -m compileall -q src` passes.
- Real-data CLI smoke tests for `04`, `05`, `06`, `10 --model moving_average`, and `11 --model moving_average` pass, or the final report records the exact missing upstream data paths that prevented them from running.
- Artifact schema, row count, key distribution, and checksum comparisons match the pre-migration baseline for graph CSVs, `article_week_sales.csv`, `attribute_week_heat.csv`, `outputs/models/moving_average/predictions.csv`, model metadata/params, and `outputs/metrics/moving_average/trend_metrics.json`; if real data is unavailable, they are skipped with the same explicit missing-data reason.
- `git diff --check` passes before each commit.
