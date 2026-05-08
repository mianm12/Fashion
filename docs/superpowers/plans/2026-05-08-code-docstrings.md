# 代码注释与文档字符串补齐实施计划

> **给执行代理的要求：** 实施本计划时，必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，并按任务逐项执行。所有步骤使用复选框语法跟踪状态。

**目标：** 为 `src/` 和 `tests/` 补齐可维护的中文文档字符串与必要注释，让项目数据流、函数契约、列契约和复杂边界更容易理解。

**架构：** 本计划只增强代码可读性，不改变领域分层、业务逻辑、测试断言、产物路径或列顺序。实施顺序按业务域推进：先基础层和上游领域，再趋势领域，再编号命令行入口，最后补测试辅助函数；每个阶段都用语法检查、相关测试和人工差异审查证明没有行为变化。

**技术栈：** Python 3.10-3.12、pandas、numpy、pyarrow、pytest、uv、`src` 布局。

---

## 范围检查

设计文档 `docs/superpowers/specs/2026-05-08-code-docstrings-design.md` 只覆盖一个子项目：为现有 Python 源码和测试补齐注释。它不包含多个独立系统，不需要拆成多份计划。

本计划不新增测试、不改业务逻辑、不改 README、不引入依赖、不运行批量重写工具。测试和验证只用于证明注释改动没有破坏现有行为。

## 执行规则

- 每个任务只修改任务列出的文件。
- 使用逐文件人工编辑，不使用自动注释生成、批量替换、全仓格式化或一次性脚本。
- 不修改导入、函数签名、常量值、列顺序、pandas 表达式、异常类型、错误文案、测试断言或产物路径。
- 简单函数使用一句中文文档字符串；复杂函数使用结构化中文文档字符串。
- 私有辅助函数只在路径安全、原子写入、回滚、指标折现、抽象语法树导入解析、滚动窗口或滞后特征等逻辑不直观时补说明。
- 普通 `test_*` 方法不机械添加文档字符串；测试名已经清楚的测试不改。
- 每个任务完成后运行任务内的最小验证命令，并检查差异只包含文档字符串、注释和必要空行。
- 不在执行中主动提交。若用户在执行阶段明确授权提交，则按任务边界提交；否则保留工作区变更供审查。

## 文档字符串风格

简单函数使用一行中文文档字符串：

```python
def read_trend_metrics(metrics_path: Path) -> dict[str, object]:
    """读取趋势评价 JSON 产物。"""
```

复杂函数使用结构化中文文档字符串：

```python
def build_trend_model_samples_frame(
    attribute_week_heat: pd.DataFrame,
    attribute_week_target: pd.DataFrame,
    attribute_nodes: pd.DataFrame,
    attribute_hierarchy_edges: pd.DataFrame,
    min_lag_weeks: int = 4,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """构建属性级趋势模型训练样本。

    参数:
        attribute_week_heat: 完整 week_id x attr_id 属性周热度面板。
        attribute_week_target: 与热度面板对齐的下一周趋势标签。
        attribute_nodes: 属性节点表，用于补充属性图静态特征。
        attribute_hierarchy_edges: 属性层级边表，用于计算父子度数特征。
        min_lag_weeks: 允许产出样本的最小历史周数。
        epsilon: 计算增长率时使用的平滑项，必须为正数。

    返回:
        按 `TREND_MODEL_SAMPLE_COLUMNS` 排列的趋势训练样本表。

    异常:
        ValueError: 输入表缺少字段、目标键缺失、数值非有限或样本契约不成立。

    边界:
        特征窗口固定为 4 周；`min_lag_weeks` 只控制最早采样周，不改变特征宽度。
    """
```

列契约常量使用上方短注释解释对应产物：

```python
# `trend_model_samples.parquet` 的稳定列契约，训练和切分阶段共享。
TREND_MODEL_SAMPLE_COLUMNS: tuple[str, ...] = (...)
```

数据类和协议使用类文档字符串解释消费方和字段稳定性：

```python
@dataclass(frozen=True)
class TrendTrainResult:
    """单个趋势模型训练器返回给通用运行器的标准训练结果。"""
```

## 目标文件映射

### 任务 1：基础层和上游领域

修改：

- `src/fashion_trend/foundation/artifacts.py`
- `src/fashion_trend/foundation/dataframe.py`
- `src/fashion_trend/foundation/io.py`
- `src/fashion_trend/foundation/logging.py`
- `src/fashion_trend/foundation/paths.py`
- `src/fashion_trend/datasets/download.py`
- `src/fashion_trend/datasets/paths.py`
- `src/fashion_trend/datasets/profile.py`
- `src/fashion_trend/transactions/contracts.py`
- `src/fashion_trend/transactions/paths.py`
- `src/fashion_trend/transactions/readers.py`
- `src/fashion_trend/transactions/weekly.py`
- `src/fashion_trend/catalog/articles.py`
- `src/fashion_trend/catalog/contracts.py`
- `src/fashion_trend/catalog/paths.py`
- `src/fashion_trend/catalog/readers.py`
- `src/fashion_trend/catalog/graph/__init__.py`
- `src/fashion_trend/catalog/graph/builders.py`
- `src/fashion_trend/catalog/graph/publishing.py`
- `src/fashion_trend/catalog/graph/schema.py`

### 任务 2：趋势确定性流水线

修改：

- `src/fashion_trend/trend/schema.py`
- `src/fashion_trend/trend/paths.py`
- `src/fashion_trend/trend/readers.py`
- `src/fashion_trend/trend/heat/article_sales.py`
- `src/fashion_trend/trend/heat/attribute_heat.py`
- `src/fashion_trend/trend/labels/targets.py`
- `src/fashion_trend/trend/features/samples.py`
- `src/fashion_trend/trend/splits/time_split.py`
- `src/fashion_trend/trend/predictions.py`

### 任务 3：趋势模型训练与评价

修改：

- `src/fashion_trend/trend/models/base.py`
- `src/fashion_trend/trend/models/registry.py`
- `src/fashion_trend/trend/models/baselines/last_week.py`
- `src/fashion_trend/trend/models/baselines/moving_average.py`
- `src/fashion_trend/trend/training/runner.py`
- `src/fashion_trend/trend/training/outputs.py`
- `src/fashion_trend/trend/evaluation/metrics.py`
- `src/fashion_trend/trend/evaluation/payloads.py`
- `src/fashion_trend/trend/evaluation/runner.py`

### 任务 4：编号命令行入口与轻量未扩展领域

修改：

- `src/00_download_data.py`
- `src/01_data_check.py`
- `src/02_build_weekly_transactions.py`
- `src/03_clean_articles.py`
- `src/04_build_attribute_graph.py`
- `src/05_compute_article_week_sales.py`
- `src/06_compute_attribute_week_heat.py`
- `src/07_build_trend_targets.py`
- `src/08_build_trend_model_samples.py`
- `src/09_split_trend_model_samples.py`
- `src/10_train_trend_model.py`
- `src/11_eval_trend_model.py`
- `src/fashion_trend/__init__.py`
- `src/fashion_trend/recommendation/contracts.py`
- `src/fashion_trend/recommendation/paths.py`
- `src/fashion_trend/recommendation/readers.py`
- `src/fashion_trend/reports/paths.py`

### 任务 5：测试辅助函数与全量验证

修改：

- `tests/__init__.py`
- `tests/trend_samples.py`
- `tests/test_architecture_boundaries.py`
- `tests/test_trend_training.py`
- `tests/test_trend_evaluation.py`
- `tests/test_attribute_graph.py`
- `tests/test_articles_clean.py`
- `tests/test_trend_article_sales.py`
- `tests/test_trend_attribute_heat.py`
- `tests/test_trend_samples.py`
- `tests/test_trend_splits.py`
- `tests/test_trend_targets.py`
- `tests/test_foundation_artifacts.py`

## 任务 1：基础层和上游领域文档字符串

**文件：**

- 修改：`src/fashion_trend/foundation/artifacts.py`
- 修改：`src/fashion_trend/foundation/dataframe.py`
- 修改：`src/fashion_trend/foundation/io.py`
- 修改：`src/fashion_trend/foundation/logging.py`
- 修改：`src/fashion_trend/foundation/paths.py`
- 修改：`src/fashion_trend/datasets/download.py`
- 修改：`src/fashion_trend/datasets/paths.py`
- 修改：`src/fashion_trend/datasets/profile.py`
- 修改：`src/fashion_trend/transactions/contracts.py`
- 修改：`src/fashion_trend/transactions/paths.py`
- 修改：`src/fashion_trend/transactions/readers.py`
- 修改：`src/fashion_trend/transactions/weekly.py`
- 修改：`src/fashion_trend/catalog/articles.py`
- 修改：`src/fashion_trend/catalog/contracts.py`
- 修改：`src/fashion_trend/catalog/paths.py`
- 修改：`src/fashion_trend/catalog/readers.py`
- 修改：`src/fashion_trend/catalog/graph/__init__.py`
- 修改：`src/fashion_trend/catalog/graph/builders.py`
- 修改：`src/fashion_trend/catalog/graph/publishing.py`
- 修改：`src/fashion_trend/catalog/graph/schema.py`
- 测试：`tests/test_foundation_artifacts.py`
- 测试：`tests/test_articles_clean.py`
- 测试：`tests/test_attribute_graph.py`
- 测试：`tests/test_trend_article_sales.py`
- 测试：`tests/test_trend_attribute_heat.py`
- 测试：`tests/test_trend_samples.py`

- [ ] **步骤 1：查看当前公开函数和常量**

运行：

```sh
rg -n "^(def|class) |^[A-Z][A-Z0-9_]+\\s*[:=]" src/fashion_trend/foundation src/fashion_trend/datasets src/fashion_trend/transactions src/fashion_trend/catalog
```

预期：命令只列出四个上游领域内的文件。把输出作为审查清单，不用该命令生成文档字符串。

- [ ] **步骤 2：补基础层说明**

只编辑：

```text
src/fashion_trend/foundation/artifacts.py
src/fashion_trend/foundation/dataframe.py
src/fashion_trend/foundation/io.py
src/fashion_trend/foundation/logging.py
src/fashion_trend/foundation/paths.py
```

覆盖要求：

- 给 `validate_safe_path_segment()` 和 `validate_output_parent_dirs()` 添加一行文档字符串，说明路径穿越防护和产物根目录约束。
- 给 `foundation/dataframe.py` 中所有公开校验函数添加一行文档字符串，说明各自维护的不变量。
- 给 `foundation/io.py` 中的原子写入辅助函数添加一行文档字符串；`write_json_atomic()`、`write_csv_atomic()`、`write_parquet_atomic()` 和 `write_binary_atomic()` 需要说明会先创建父目录再原子替换。
- 保留 `foundation/logging.py` 已有文档字符串；只有在返回值或输出流语义不清楚时才收紧表述。
- 在 `foundation/paths.py` 的 `PROJECT_ROOT`、`DATA_DIR`、`RAW_DIR`、`INTERIM_DIR`、`PROCESSED_DIR` 和 `OUTPUT_DIR` 上方添加短注释，说明这些是无业务语义的项目根路径。

禁止修改路径表达式、环境变量名、输出流选择、异常类型或 JSON 排序行为。

- [ ] **步骤 3：补数据集和交易领域说明**

只编辑：

```text
src/fashion_trend/datasets/download.py
src/fashion_trend/datasets/paths.py
src/fashion_trend/datasets/profile.py
src/fashion_trend/transactions/contracts.py
src/fashion_trend/transactions/paths.py
src/fashion_trend/transactions/readers.py
src/fashion_trend/transactions/weekly.py
```

覆盖要求：

- 在 `DEFAULT_COMPETITION`、原始数据路径常量、`WEEKLY_TRANSACTIONS_PATH` 和 `WEEKLY_TRANSACTION_COLUMNS` 上方添加注释，说明它们对应哪个稳定产物或原始文件。
- 给 `validate_raw_dataset_files()` 和 `datasets/download.py` 中所有公开函数添加文档字符串；复杂函数需要说明跳过下载、强制下载和 zip 路径逃逸防护。
- 保留 `transactions/weekly.py` 已有文档字符串，但把 `write_weekly_transactions()` 和 `build_weekly_transactions()` 扩展成结构化文档字符串，因为它们定义分块读取、日期范围扫描、周编号派生和输出写入。
- 给 `read_weekly_transactions()` 添加一行文档字符串，说明它保留周级交易表列契约。

禁止修改 CSV 类型映射、`week_id` 公式、分块大小、目标路径或 KaggleHub 调用参数。

- [ ] **步骤 4：补商品目录和属性图说明**

只编辑：

```text
src/fashion_trend/catalog/articles.py
src/fashion_trend/catalog/contracts.py
src/fashion_trend/catalog/paths.py
src/fashion_trend/catalog/readers.py
src/fashion_trend/catalog/graph/__init__.py
src/fashion_trend/catalog/graph/builders.py
src/fashion_trend/catalog/graph/publishing.py
src/fashion_trend/catalog/graph/schema.py
```

覆盖要求：

- 在商品和属性图列契约常量上方添加注释，说明它们描述哪个 CSV 产物。
- 给 `build_article_nodes()`、`build_attribute_nodes()`、`build_article_attribute_edges()` 和图标识符辅助函数添加一行文档字符串。
- 给 `build_attribute_hierarchy_edges()`、`validate_graph_references()`、`build_attribute_graph_frames()`、`publish_graph_frames()` 和 `clean_articles_file()` 添加结构化文档字符串。
- 在 `catalog/articles.py` 中说明 MVP 产物和完整清洗产物一起发布，回滚逻辑保护已替换的 MVP 产物。
- 在属性图发布辅助函数中说明备份和回滚意图，不改变替换顺序。

禁止修改图关系定义、输出文件名、CSV 引号行为、清理行为或回滚语义。

- [ ] **步骤 5：验证任务 1**

运行：

```sh
uv run python -m compileall \
  src/fashion_trend/foundation \
  src/fashion_trend/datasets \
  src/fashion_trend/transactions \
  src/fashion_trend/catalog
```

预期：命令退出码为 0。

运行：

```sh
uv run pytest \
  tests/test_foundation_artifacts.py \
  tests/test_articles_clean.py \
  tests/test_attribute_graph.py \
  tests/test_trend_article_sales.py \
  tests/test_trend_attribute_heat.py \
  tests/test_trend_samples.py
```

预期：所选测试全部通过。

- [ ] **步骤 6：审查任务 1 差异**

运行：

```sh
git diff -- src/fashion_trend/foundation src/fashion_trend/datasets src/fashion_trend/transactions src/fashion_trend/catalog
git diff --check
```

预期：差异只包含文档字符串、注释和必要空行；`git diff --check` 退出码为 0。

## 任务 2：趋势确定性流水线文档字符串

**文件：**

- 修改：`src/fashion_trend/trend/schema.py`
- 修改：`src/fashion_trend/trend/paths.py`
- 修改：`src/fashion_trend/trend/readers.py`
- 修改：`src/fashion_trend/trend/heat/article_sales.py`
- 修改：`src/fashion_trend/trend/heat/attribute_heat.py`
- 修改：`src/fashion_trend/trend/labels/targets.py`
- 修改：`src/fashion_trend/trend/features/samples.py`
- 修改：`src/fashion_trend/trend/splits/time_split.py`
- 修改：`src/fashion_trend/trend/predictions.py`
- 测试：`tests/test_trend_article_sales.py`
- 测试：`tests/test_trend_attribute_heat.py`
- 测试：`tests/test_trend_targets.py`
- 测试：`tests/test_trend_samples.py`
- 测试：`tests/test_trend_splits.py`
- 测试：`tests/test_trend_training.py`
- 测试：`tests/test_trend_evaluation.py`

- [ ] **步骤 1：查看趋势流水线目标**

运行：

```sh
rg -n "^(def|class) |^[A-Z][A-Z0-9_]+\\s*[:=]" src/fashion_trend/trend/schema.py src/fashion_trend/trend/paths.py src/fashion_trend/trend/readers.py src/fashion_trend/trend/heat src/fashion_trend/trend/labels src/fashion_trend/trend/features src/fashion_trend/trend/splits src/fashion_trend/trend/predictions.py
```

预期：命令只列出趋势领域的列契约、路径、读取器、热度、标签、特征、切分和预测文件。

- [ ] **步骤 2：补列契约、路径和读取器说明**

只编辑：

```text
src/fashion_trend/trend/schema.py
src/fashion_trend/trend/paths.py
src/fashion_trend/trend/readers.py
```

覆盖要求：

- 在 `trend/schema.py` 中每个公开列契约上方添加注释，说明它稳定哪个产物或接口。
- 在类型映射上方添加注释，说明读取器用它们保留标识符和数值类型。
- 在 `trend/paths.py` 的趋势路径常量上方添加注释，说明对应流水线阶段和输出根目录。
- 给 `trend/readers.py` 中公开读取器添加一行文档字符串，说明读取哪个 JSON 或表格产物。

禁止修改元组值、元组顺序、类型字符串、路径表达式、切分窗口或模型输出根目录常量。

- [ ] **步骤 3：补热度和标签说明**

只编辑：

```text
src/fashion_trend/trend/heat/article_sales.py
src/fashion_trend/trend/heat/attribute_heat.py
src/fashion_trend/trend/labels/targets.py
```

覆盖要求：

- 给 `build_article_week_sales_frame()`、`build_attribute_week_heat_frame()`、`validate_attribute_week_heat()`、`build_attribute_week_target_frame()`、`validate_attribute_week_target()` 和 `validate_attribute_week_target_matches_heat()` 添加结构化文档字符串。
- 给简单读取器和校验器添加一行文档字符串。
- `build_attribute_week_heat_frame()` 需要说明完整 `week_id x attr_id` 面板、零填充行为、按属性类型计算占比、`log1p` 和排名排序键。
- 标签构造需要说明 `t -> t+1` 位移、增长率平滑公式和最后一周排除。
- 只在分组变换、面板交叉连接或排名校验不易读时添加短行内注释。

禁止修改数值容差、`epsilon` 默认值、排序键、排名逻辑、校验文案或读取器类型选择。

- [ ] **步骤 4：补特征、切分和预测说明**

只编辑：

```text
src/fashion_trend/trend/features/samples.py
src/fashion_trend/trend/splits/time_split.py
src/fashion_trend/trend/predictions.py
```

覆盖要求：

- 给 `build_attribute_graph_features_frame()`、`build_trend_model_samples_frame()`、`validate_trend_model_samples()`、`build_trend_model_split_frames()`、`validate_trend_model_split_frames()`、`build_trend_model_split_metadata()`、`validate_trend_model_predictions()`、`derive_normalized_pred_share_t1()` 和 `validate_pred_share_t1_distribution()` 添加结构化文档字符串。
- 样本构造需要说明固定 4 周特征窗口、`min_lag_weeks` 边界、陈旧目标防护、图度数特征、历史活跃资格和最终列契约。
- 时间切分需要说明训练、验证、测试周范围，以及验证和测试按时间留出的要求。
- 预测校验需要说明 `pred_share_t1` 按 `split + week_id + attr_type` 归一化，而 `pred_target_growth` 是趋势指标的主要输入。

禁止修改特征公式、合并方式、切分赋值、校验严格性、归一化容差或预测列顺序。

- [ ] **步骤 5：验证任务 2**

运行：

```sh
uv run python -m compileall \
  src/fashion_trend/trend/schema.py \
  src/fashion_trend/trend/paths.py \
  src/fashion_trend/trend/readers.py \
  src/fashion_trend/trend/heat \
  src/fashion_trend/trend/labels \
  src/fashion_trend/trend/features \
  src/fashion_trend/trend/splits \
  src/fashion_trend/trend/predictions.py
```

预期：命令退出码为 0。

运行：

```sh
uv run pytest \
  tests/test_trend_article_sales.py \
  tests/test_trend_attribute_heat.py \
  tests/test_trend_targets.py \
  tests/test_trend_samples.py \
  tests/test_trend_splits.py \
  tests/test_trend_training.py \
  tests/test_trend_evaluation.py
```

预期：所选测试全部通过。

- [ ] **步骤 6：审查任务 2 差异**

运行：

```sh
git diff -- src/fashion_trend/trend/schema.py src/fashion_trend/trend/paths.py src/fashion_trend/trend/readers.py src/fashion_trend/trend/heat src/fashion_trend/trend/labels src/fashion_trend/trend/features src/fashion_trend/trend/splits src/fashion_trend/trend/predictions.py
git diff --check
```

预期：差异只包含文档字符串、注释和必要空行；`git diff --check` 退出码为 0。

## 任务 3：趋势模型训练与评价文档字符串

**文件：**

- 修改：`src/fashion_trend/trend/models/base.py`
- 修改：`src/fashion_trend/trend/models/registry.py`
- 修改：`src/fashion_trend/trend/models/baselines/last_week.py`
- 修改：`src/fashion_trend/trend/models/baselines/moving_average.py`
- 修改：`src/fashion_trend/trend/training/runner.py`
- 修改：`src/fashion_trend/trend/training/outputs.py`
- 修改：`src/fashion_trend/trend/evaluation/metrics.py`
- 修改：`src/fashion_trend/trend/evaluation/payloads.py`
- 修改：`src/fashion_trend/trend/evaluation/runner.py`
- 测试：`tests/test_trend_training.py`
- 测试：`tests/test_trend_evaluation.py`

- [ ] **步骤 1：查看模型、训练和评价目标**

运行：

```sh
rg -n "^(def|class) |@dataclass|Protocol|^[A-Z][A-Z0-9_]+\\s*[:=]" src/fashion_trend/trend/models src/fashion_trend/trend/training src/fashion_trend/trend/evaluation
```

预期：命令列出数据类、训练器协议、注册表、基线训练器、运行器辅助函数、输出辅助函数、指标、载荷和评价运行器。

- [ ] **步骤 2：补模型接口和基线说明**

只编辑：

```text
src/fashion_trend/trend/models/base.py
src/fashion_trend/trend/models/registry.py
src/fashion_trend/trend/models/baselines/last_week.py
src/fashion_trend/trend/models/baselines/moving_average.py
```

覆盖要求：

- 给 `TrendArtifact`、`TrendTrainContext`、`TrendTrainResult` 和 `TrendModelTrainer` 添加类文档字符串。
- 在 `MODEL_TYPE_BASELINE`、`MODEL_TYPE_SUPERVISED` 和 `KNOWN_MODEL_TYPES` 上方添加注释，说明运行器校验用途。
- 给注册表辅助函数和 `UnknownTrendModelError` 添加一行文档字符串。
- 给 `predict_last_week()` 和 `predict_moving_average()` 添加结构化文档字符串，说明公式和必要滞后列。
- 给 `LastWeekTrainer` 和 `MovingAverageTrainer` 添加类文档字符串，说明它们为通用运行器产出标准 `TrendTrainResult`。
- 只在校验有限滞后值或复制可变参数时，给私有辅助函数添加一行文档字符串。

禁止修改模型名、公式、参数载荷、注册映射、可变参数复制行为或错误文案。

- [ ] **步骤 3：补训练运行器和输出说明**

只编辑：

```text
src/fashion_trend/trend/training/runner.py
src/fashion_trend/trend/training/outputs.py
```

覆盖要求：

- 给 `default_trend_model_input_paths()` 和 `read_trend_model_split_frames()` 添加一行文档字符串。
- 给 `run_trend_model_training()` 添加结构化文档字符串，说明切分读取、训练器查找、输出路径派生、元数据构建和写出阶段。
- 给 `validate_trend_train_result()`、`build_trend_train_metadata()` 和 `write_trend_model_outputs()` 添加结构化文档字符串。
- 给 `outputs.py` 中的私有辅助函数添加一行文档字符串，说明产物路径安全、整数周编号校验、严格 JSON 校验、暂存输出构造、目标路径校验、载荷写入、回滚和备份清理。
- 只在能帮助理解两阶段发布流程时，在 `write_trend_model_outputs()` 的暂存发布或回滚前添加短注释。

禁止修改暂存目录名、UUID 使用、备份替换顺序、JSON 序列化选项、输出文件名或回滚行为。

- [ ] **步骤 4：补评价指标和载荷说明**

只编辑：

```text
src/fashion_trend/trend/evaluation/metrics.py
src/fashion_trend/trend/evaluation/payloads.py
src/fashion_trend/trend/evaluation/runner.py
```

覆盖要求：

- 给 `compute_trend_group_metrics()`、`compute_trend_metrics()`、`build_trend_metrics_payload()`、`validate_trend_model_predictions_for_evaluation()` 和 `run_trend_model_evaluation()` 添加结构化文档字符串。
- 给私有指标辅助函数添加一行文档字符串，包括 K 值校验、热门属性提取、Spearman 空值回退、NDCG 空值回退、折现增益、记录汇总、可空均值和 JSON 浮点转换。
- 说明指标只聚合验证集和测试集；训练集预测保留在模型输出中，但不进入趋势评价。
- 说明写出 `trend_metrics.json` 前会做严格 JSON 校验。

禁止修改排名排序、有效 K 逻辑、空值处理、载荷键、评价切分集合或 JSON 写出行为。

- [ ] **步骤 5：验证任务 3**

运行：

```sh
uv run python -m compileall \
  src/fashion_trend/trend/models \
  src/fashion_trend/trend/training \
  src/fashion_trend/trend/evaluation
```

预期：命令退出码为 0。

运行：

```sh
uv run pytest tests/test_trend_training.py tests/test_trend_evaluation.py
```

预期：两个测试文件全部通过。

- [ ] **步骤 6：审查任务 3 差异**

运行：

```sh
git diff -- src/fashion_trend/trend/models src/fashion_trend/trend/training src/fashion_trend/trend/evaluation
git diff --check
```

预期：差异只包含文档字符串、注释和必要空行；`git diff --check` 退出码为 0。

## 任务 4：编号命令行入口与轻量未扩展领域文档字符串

**文件：**

- 修改：`src/00_download_data.py`
- 修改：`src/01_data_check.py`
- 修改：`src/02_build_weekly_transactions.py`
- 修改：`src/03_clean_articles.py`
- 修改：`src/04_build_attribute_graph.py`
- 修改：`src/05_compute_article_week_sales.py`
- 修改：`src/06_compute_attribute_week_heat.py`
- 修改：`src/07_build_trend_targets.py`
- 修改：`src/08_build_trend_model_samples.py`
- 修改：`src/09_split_trend_model_samples.py`
- 修改：`src/10_train_trend_model.py`
- 修改：`src/11_eval_trend_model.py`
- 修改：`src/fashion_trend/__init__.py`
- 修改：`src/fashion_trend/recommendation/contracts.py`
- 修改：`src/fashion_trend/recommendation/paths.py`
- 修改：`src/fashion_trend/recommendation/readers.py`
- 修改：`src/fashion_trend/reports/paths.py`
- 测试：`tests/test_trend_training.py`
- 测试：`tests/test_trend_evaluation.py`
- 测试：`tests/test_architecture_boundaries.py`

- [ ] **步骤 1：查看命令行入口目标**

运行：

```sh
rg -n "^(def|class) |^[A-Z][A-Z0-9_]+\\s*[:=]" src/00_download_data.py src/01_data_check.py src/02_build_weekly_transactions.py src/03_clean_articles.py src/04_build_attribute_graph.py src/05_compute_article_week_sales.py src/06_compute_attribute_week_heat.py src/07_build_trend_targets.py src/08_build_trend_model_samples.py src/09_split_trend_model_samples.py src/10_train_trend_model.py src/11_eval_trend_model.py src/fashion_trend/recommendation src/fashion_trend/reports
```

预期：命令列出命令行入口的 `main()`、编排函数，以及轻量推荐和报告领域的常量或读取器。

- [ ] **步骤 2：补编号命令行入口说明**

只编辑：

```text
src/00_download_data.py
src/01_data_check.py
src/02_build_weekly_transactions.py
src/03_clean_articles.py
src/04_build_attribute_graph.py
src/05_compute_article_week_sales.py
src/06_compute_attribute_week_heat.py
src/07_build_trend_targets.py
src/08_build_trend_model_samples.py
src/09_split_trend_model_samples.py
src/10_train_trend_model.py
src/11_eval_trend_model.py
```

覆盖要求：

- 保留已经解释参数解析或入口行为的现有文档字符串。
- 给缺少说明的 `main()` 添加一行文档字符串，说明脚本是哪个阶段入口，并指出稳定输出产物。
- 给 `compute_article_week_sales()`、`compute_attribute_week_heat()`、`build_trend_targets()`、`build_trend_model_samples()` 和 `split_trend_model_samples()` 等编排辅助函数添加结构化文档字符串，因为它们展示读取、构建、校验、写出、汇总的流程。
- `src/10_train_trend_model.py` 和 `src/11_eval_trend_model.py` 需要保留参数用法错误返回 2、领域错误返回 1 的说明。

禁止修改参数选项、返回码、日志文本、输出摘要或命令入口行为。

- [ ] **步骤 3：补包说明和轻量领域注释**

只编辑：

```text
src/fashion_trend/__init__.py
src/fashion_trend/recommendation/contracts.py
src/fashion_trend/recommendation/paths.py
src/fashion_trend/recommendation/readers.py
src/fashion_trend/reports/paths.py
```

覆盖要求：

- 只有当通用包文档字符串有误导性时才替换为简洁领域包说明。
- 在 `RECOMMENDATION_TOP_K` 上方添加注释，说明它是当前 Top-N 推荐契约。
- 在推荐和报告路径常量上方添加注释，说明它们是尚未扩展的推荐和报告阶段输出位置。
- 给 `read_recommendation_result()` 添加一行文档字符串，说明它是面向推荐产物的读取器。

禁止添加行为、导入、新文件或尚未落地的算法承诺。

- [ ] **步骤 4：验证任务 4**

运行：

```sh
uv run python -m compileall \
  src/00_download_data.py \
  src/01_data_check.py \
  src/02_build_weekly_transactions.py \
  src/03_clean_articles.py \
  src/04_build_attribute_graph.py \
  src/05_compute_article_week_sales.py \
  src/06_compute_attribute_week_heat.py \
  src/07_build_trend_targets.py \
  src/08_build_trend_model_samples.py \
  src/09_split_trend_model_samples.py \
  src/10_train_trend_model.py \
  src/11_eval_trend_model.py \
  src/fashion_trend/__init__.py \
  src/fashion_trend/recommendation \
  src/fashion_trend/reports
```

预期：命令退出码为 0。

运行：

```sh
uv run pytest tests/test_trend_training.py tests/test_trend_evaluation.py tests/test_architecture_boundaries.py
```

预期：所选测试全部通过。

- [ ] **步骤 5：审查任务 4 差异**

运行：

```sh
git diff -- src/00_download_data.py src/01_data_check.py src/02_build_weekly_transactions.py src/03_clean_articles.py src/04_build_attribute_graph.py src/05_compute_article_week_sales.py src/06_compute_attribute_week_heat.py src/07_build_trend_targets.py src/08_build_trend_model_samples.py src/09_split_trend_model_samples.py src/10_train_trend_model.py src/11_eval_trend_model.py src/fashion_trend/__init__.py src/fashion_trend/recommendation src/fashion_trend/reports
git diff --check
```

预期：差异只包含文档字符串、注释和必要空行；`git diff --check` 退出码为 0。

## 任务 5：测试辅助函数说明与全量验证

**文件：**

- 修改：`tests/__init__.py`
- 修改：`tests/trend_samples.py`
- 修改：`tests/test_architecture_boundaries.py`
- 修改：`tests/test_trend_training.py`
- 修改：`tests/test_trend_evaluation.py`
- 修改：`tests/test_attribute_graph.py`
- 修改：`tests/test_articles_clean.py`
- 修改：`tests/test_trend_article_sales.py`
- 修改：`tests/test_trend_attribute_heat.py`
- 修改：`tests/test_trend_samples.py`
- 修改：`tests/test_trend_splits.py`
- 修改：`tests/test_trend_targets.py`
- 修改：`tests/test_foundation_artifacts.py`
- 测试：完整测试套件

- [ ] **步骤 1：查看测试辅助函数目标**

运行：

```sh
rg -n "^(def|class) |^    def _|^def _" tests
```

预期：命令列出辅助函数、测试类和测试方法。只有辅助函数和复杂测试基础设施需要文档字符串。

- [ ] **步骤 2：补共享样本和架构辅助函数说明**

只编辑：

```text
tests/__init__.py
tests/trend_samples.py
tests/test_architecture_boundaries.py
```

覆盖要求：

- 给 `tests/trend_samples.py` 中每个样本构造函数添加一行文档字符串，说明样本代表哪个流水线阶段或契约。
- 给 `sample_long_attribute_week_heat()`、`sample_trend_model_samples_for_split()` 和 `sample_trend_predictions_for_evaluation()` 添加结构化文档字符串，因为它们编码多周和多切分场景。
- 给 `tests/test_architecture_boundaries.py` 中的抽象语法树辅助函数添加一行文档字符串。
- 给 `imported_modules()`、`imported_from_modules()`、`trend_facade_import_offenders()` 和 `package_upstream_import_offenders()` 添加结构化文档字符串，说明静态导入解析和白名单匹配。
- 保持测试断言不变。

不要给普通测试方法添加文档字符串，除非该方法包含从名称无法理解的辅助逻辑。

- [ ] **步骤 3：给大型测试文件选择性补充辅助说明**

只编辑：

```text
tests/test_trend_training.py
tests/test_trend_evaluation.py
tests/test_attribute_graph.py
tests/test_articles_clean.py
tests/test_trend_article_sales.py
tests/test_trend_attribute_heat.py
tests/test_trend_samples.py
tests/test_trend_splits.py
tests/test_trend_targets.py
tests/test_foundation_artifacts.py
```

覆盖要求：

- 只给 `_expected_normalized_pred_share()`、`_assert_pred_share_t1_distribution()` 这类私有辅助函数添加文档字符串。
- 只有当复杂 monkeypatch 或回滚场景从测试名无法看懂时，才在设置代码前添加短注释。
- 不给行为已经由名称表达清楚的测试类或测试方法添加重复文档字符串。

禁止修改夹具、monkeypatch 目标、期望字典、断言顺序、临时路径或异常匹配。

- [ ] **步骤 4：运行全量验证**

运行：

```sh
uv run python -m compileall src tests
```

预期：命令退出码为 0。

运行：

```sh
uv run pytest
```

预期：完整测试套件通过。

运行：

```sh
git diff --check
```

预期：命令退出码为 0。

- [ ] **步骤 5：审查最终差异是否存在行为变化**

运行：

```sh
git diff --stat
git diff -- src tests
```

预期：

- 差异统计只包含 `src/` 和 `tests/` 下的 Python 文件。
- 源码差异只包含文档字符串、注释和必要空行。
- 没有导入路径、函数签名、常量值、元组顺序、pandas 表达式、异常文案、测试断言或输出路径变化。

- [ ] **步骤 6：在明确授权后按需提交**

如果用户在执行阶段明确授权提交，则完整验证通过后提交：

```sh
git add src tests
git commit -m "docs: 补齐代码注释和文档字符串"
```

预期：提交成功，并且只包含已验证的文档字符串和注释变更。若没有提交授权，跳过本步骤，并报告验证结果和工作区差异状态。
