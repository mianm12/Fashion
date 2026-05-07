# 业务域驱动代码架构设计

## 范围

本设计用于把项目代码组织方式从“脚本编号驱动”升级为“业务域驱动”。编号脚本 `src/00_*.py` 到 `src/16_*.py` 继续作为用户运行入口和业务流程索引，但真正的计算、schema、校验、模型、评价和推荐逻辑应进入 `src/fashion_trend/` 下的业务包。

本次设计围绕实施计划主线组织代码：

```text
属性图 -> 属性周热度 -> 趋势预测 -> 轻量推荐闭环
```

架构目标不是保留内部 Python import 兼容，而是找出更清晰、更适合后续 LightGBM、推荐模块、消融实验和展示页面扩展的组织方式。唯一必须稳定的是：

- 编号 CLI 的用户入口和大致阶段边界。
- 既有数据产物、模型产物和指标产物的路径与 schema 契约。

内部 Python import 路径允许破坏性迁移，旧 facade 和历史散落模块允许删除。

## 非目标

本设计不实现代码迁移，不新增模型，不实现 LightGBM，不实现推荐模块，不改变现有算法公式，不改变现有产物路径或 schema。

本设计也不要求业务域内部机械拆成 `builders.py`、`readers.py`、`validators.py`、`writers.py` 这种固定模板。域内模块应按真实业务对象、算法职责、复用程度和复杂度决定。

## 已确认原则

### 编号脚本是业务流程索引

编号脚本应能让人快速浏览本阶段的大致数据流处理过程。脚本中保留高层步骤，例如：

```text
读取输入 -> 调用领域能力 -> 关键校验 -> 写出稳定产物 -> 打印摘要
```

脚本可以展示输入来自哪里、调用哪些业务函数、关键校验点在哪里、输出写到哪里，以及本阶段与上下游产物的关系。

脚本不应承载 pandas groupby、merge、rolling 细节，不定义 schema 常量，不写模型公式，不写复杂指标计算，不实现原子写入，不保存跨阶段共享 helper。

### 业务包是计算事实来源

所有实际业务计算、业务 schema、业务校验、模型训练、趋势评价、推荐候选、推荐打分和推荐评价都应进入 `src/fashion_trend/<domain>/`。

编号脚本只从领域包导入高层、语义清晰的函数，不作为兼容层，也不 re-export 业务函数。

### foundation 只承载无业务语义的基础能力

`foundation` 是唯一共享基础层，不能依赖任何业务模块。它只能放路径、日志、原子写入、通用 DataFrame 校验、JSON/CSV/Parquet 安全读写、artifact path 安全检查等稳定基础能力。

商品、交易、趋势、推荐等业务语义不能进入 `foundation`。

## 备选方案

### 方案一：极薄 CLI

编号脚本只调用一个 `run_xxx_pipeline()`。这种方式最干净，但脚本无法展示大致处理流程，不符合“编号脚本可浏览数据流”的目标。

### 方案二：可读编排 CLI

编号脚本保留 `read -> build -> validate -> write` 级别的高层流程，业务细节下沉到领域包。这是推荐方案。它同时保留用户入口的可读性和业务包的可维护性。

### 方案三：脚本内保留业务骨架

编号脚本中保留部分 transform 逻辑。短期直观，但长期会重新滑回脚本编号驱动，难以支撑 LightGBM、推荐模块和消融实验扩展。

本设计采用方案二。

## 目标包结构

```text
src/fashion_trend/
  foundation/
  datasets/
  transactions/
  catalog/
  trend/
  recommendation/
  reports/
```

依赖方向固定为：

```text
foundation
  <- datasets
  <- transactions
  <- catalog
  <- trend
  <- recommendation
  <- reports / CLI / future app
```

更具体地说：

- `foundation` 不依赖任何业务域。
- `datasets` 只依赖 `foundation`。
- `transactions` 依赖 `foundation`，可以消费 raw dataset。
- `catalog` 依赖 `foundation`，可以消费 raw articles。
- `trend` 依赖 `foundation`，并消费 `transactions` 和 `catalog` 的稳定产物或公开读取接口。
- `recommendation` 依赖 `foundation`，并消费 `transactions`、`catalog` 和 `trend` 的稳定预测产物。
- `recommendation` 不依赖 `trend.models.*` 或训练内部实现。
- `reports` 和未来 app 只读稳定产物，不参与核心计算。

## 业务域职责

### foundation

无业务语义基础层：

- 项目路径和 artifact 根目录定义。
- 日志工具。
- 原子写入。
- CSV、Parquet、JSON 的通用安全读写。
- 通用 DataFrame 校验原语，例如必需列、缺失值、唯一键、非负数、有限数值。
- artifact path 安全检查，例如 model name path segment 校验。

业务 schema、业务公式和业务错误语义不进入 `foundation`。

### datasets

原始数据层：

- 下载 Kaggle 数据。
- 安全解压。
- 原始文件存在性检查。
- 基础 profile，例如数据行数、日期范围、字段概览。

`datasets` 只确认 raw dataset 是否可用，不做商品清洗、交易聚合、趋势特征或推荐逻辑。

### transactions

交易域：

- 读取 raw transactions。
- 构建周级交易表。
- 维护 `week_id`、交易时间窗口、用户购买历史等稳定交易产物。
- 提供趋势和推荐可复用的交易读取接口。

`transactions` 不知道属性图如何构建，不知道趋势模型如何训练，也不知道推荐如何打分。

### catalog

商品目录域：

- 清洗 `articles.csv`。
- 构建 article 节点。
- 构建 attribute 节点。
- 构建 article-attribute 边。
- 构建 attribute hierarchy 边。
- 提供商品属性图读取接口。

`catalog` 不知道销量、趋势模型或推荐模型存在。商品属性图是静态业务事实，趋势和推荐只能消费它的稳定产物或公开读取接口。

### trend

趋势域：

- 构建 `article_week_sales.csv`。
- 构建 `attribute_week_heat.csv`。
- 构建 `attribute_week_target.csv`。
- 构建 `trend_model_samples.parquet`。
- 执行 train/valid/test 时间切分。
- 管理趋势模型接口、baseline、LightGBM 和后续模型。
- 管理趋势训练 artifact。
- 管理趋势预测契约。
- 管理趋势评价指标与 `trend_metrics.json`。

`article_week_sales` 归入 `trend`，不是 `transactions`。它虽然来自周交易表，但它的业务目的明确是属性热度的上游信号，而不是通用交易基础表。

现有根包下的 `models/`、`training.py`、`evaluation.py` 应迁入 `trend` 域内，例如：

```text
trend/
  heat/
  samples/
  models/
  training/
  evaluation/
```

具体文件和子包不按固定模板预设，实施时按职责复杂度确定。

### recommendation

推荐域：

- 构建候选商品。
- 构建用户历史属性偏好。
- 消费趋势预测结果并映射为推荐趋势分。
- 执行趋势感知重排序。
- 生成 Top-12 推荐结果。
- 计算推荐评价指标。

推荐模块是实施计划中的辅助任务，用趋势预测结果展示应用价值。它不追求复杂推荐系统性能，不依赖趋势模型内部训练实现，只消费稳定预测 artifact 或趋势域公开的预测读取契约。

### reports

报告域：

- 生成趋势曲线图。
- 生成模型指标表。
- 生成推荐案例。
- 导出论文或展示需要的 figures、tables、case studies。

`reports` 只读 `data/processed` 与 `outputs` 下的稳定产物，不参与核心计算，也不把展示需求反向写入业务域。

## 编号 CLI 映射

编号脚本继续保留当前阶段边界：

```text
00_download_data.py                 datasets
01_data_check.py                    datasets
02_build_weekly_transactions.py     transactions
03_clean_articles.py                catalog
04_build_attribute_graph.py         catalog
05_compute_article_week_sales.py    trend
06_compute_attribute_week_heat.py   trend
07_build_trend_targets.py           trend
08_build_trend_model_samples.py     trend
09_split_trend_model_samples.py     trend
10_train_trend_model.py             trend
11_eval_trend_model.py              trend
12_train_lightgbm_trend_model.py    trend, 后续新增或调整
13_build_recommend_candidates.py    recommendation
14_rerank_recommendations.py        recommendation
15_eval_recommendations.py          recommendation
16_make_reports.py                  reports
```

`03_clean_articles.py` 和 `04_build_attribute_graph.py` 分开保留：二者同属 `catalog`，但一个是商品表清洗，一个是静态属性图构建。

`05_compute_article_week_sales.py` 归入 `trend`：它是属性周热度的直接上游，不是通用交易域基础产物。

## CLI 编排形态

编号脚本应保持可读流程。例如 `06_compute_attribute_week_heat.py` 可以呈现为：

```text
解析参数和路径
读取商品周销量
读取属性图节点和商品-属性边
构建属性周热度
校验属性周热度
写出 data/processed/trend/attribute_week_heat.csv
打印摘要
```

这类脚本中的函数调用应来自对应领域包，函数名表达业务语义，而不是暴露实现细节。

不要求所有领域都提供同名 `readers.py`、`builders.py`、`validators.py`、`writers.py`。简单阶段可以一个文件承载，例如 `transactions/weekly.py` 或 `catalog/articles.py`。复杂阶段再拆子包，例如 `trend/heat/`、`trend/models/`、`recommendation/rerank/`。

## 产物 ownership

```text
data/processed/basic/           datasets
data/processed/transactions/    transactions
data/processed/graph/           catalog
data/processed/trend/           trend deterministic tables
data/processed/features/        trend/recommendation feature tables, 文件级明确 ownership
data/processed/recommend/       recommendation intermediate tables
outputs/models/<model>/         trend model artifacts
outputs/metrics/<model>/        trend metrics
outputs/recommendation/         recommendation results and metrics
outputs/reports/                reports figures/tables/cases
```

`data/processed` 放可复现的中间数据和特征表。`outputs` 放实验、模型、指标、报告和展示产物。

趋势预测 `predictions.csv` 继续属于 `outputs/models/<model>/predictions.csv`，不挪进 `data/processed/trend/`。趋势评价继续写入 `outputs/metrics/<model>/trend_metrics.json`。

`data/processed/features/` 不能变成跨域垃圾桶。趋势样本、推荐特征等文件必须有明确文件 ownership；多个领域不能随意互写同一文件。

## 迁移策略

迁移可以破坏内部 import，但不应一次性混入行为修改。建议按阶段推进，每阶段都能独立验证。

### 阶段 1：foundation

提取路径、日志、原子写入、通用校验、artifact path 安全检查。不引入业务语义。

### 阶段 2：catalog 与 transactions

迁移商品清洗、属性图构建和周交易逻辑，建立 `catalog` 和 `transactions` 的稳定公开函数。

### 阶段 3：trend deterministic pipeline

迁移 `article_sales`、`attribute_heat`、`targets`、`samples`、`splits`。保持 heat、target、sample、split 产物契约不变。

### 阶段 4：trend experiments

迁移 `models`、`training`、`evaluation`。保持 `--model` CLI、`outputs/models/<model>/` 和 `outputs/metrics/<model>/` 不变。

### 阶段 5：recommendation 与 reports

新增推荐和报告域。推荐模块作为后续新增业务，不混进前四个结构迁移阶段。

## 测试与验证

每个迁移阶段至少运行：

```sh
uv run pytest
uv run python -m py_compile <本阶段改动的 src 文件>
```

涉及真实产物契约的阶段，应补充相关 CLI 烟测，例如：

```sh
uv run python src/05_compute_article_week_sales.py
uv run python src/06_compute_attribute_week_heat.py
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

如果迁移触及核心构建逻辑，即使目标只是 import 和结构调整，也应通过行数、schema、关键字段分布或 checksum 检查确认产物没有漂移。

## 验收标准

架构重组完成后应满足：

- 根包不再有 `articles.py`、`data_loader.py`、`training.py`、`evaluation.py` 这类历史散落模块。
- `foundation` 不依赖任何业务域。
- `catalog` 不依赖 `trend` 或 `recommendation`。
- `trend` 不依赖 `recommendation` 或 `reports`。
- `recommendation` 不依赖 `trend.models.*`，只消费稳定预测契约。
- 编号 CLI 保持可运行、可读，且用户命令不变。
- 编号 CLI 能展示高层数据流，但不承载业务计算事实。
- 既有产物路径和 schema 不变。
- `README.md` 和 `docs/gpt-research/implementation-plan.md` 的架构描述与实际代码一致。
- 测试通过，必要 CLI 验证通过。
- 删除旧 facade 和旧 import 后，仓库内不再残留历史路径引用。

## 风险与控制

最大风险是把结构迁移和行为修改混在一起，导致产物漂移但难以定位。控制方式是按阶段迁移、每阶段检查 diff、运行测试和必要 CLI，并在触及核心产物时做 schema 或 checksum 对比。

第二个风险是 `foundation` 变成业务工具垃圾桶。控制方式是禁止任何业务字段名、业务公式、业务对象进入 `foundation`。

第三个风险是 `data/processed/features/` ownership 模糊。控制方式是在文档和代码中明确每个 feature 文件归属哪个领域，避免多个领域共同写同一个产物。

第四个风险是编号 CLI 过度变薄，丧失流程索引价值。控制方式是把 `read -> build -> validate -> write` 的高层步骤保留在脚本里，但把计算事实留在业务包。
