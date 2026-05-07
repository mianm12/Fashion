# 业务域驱动代码架构设计

## 范围

本设计用于把项目代码组织方式从“脚本编号驱动”升级为“业务域驱动”。编号脚本 `src/00_*.py` 到 `src/16_*.py` 继续留在 `src/` 中，作为用户运行入口和业务流程索引；真正的计算、schema、校验、模型、评价、推荐和报告逻辑应进入 `src/fashion_trend/` 下的业务包。

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

## 当前基线

当前仓库已经完成第一轮业务域迁移：

- `src/fashion_trend/foundation/`、`datasets/`、`transactions/`、`catalog/`、`trend/`、`recommendation/`、`reports/` 已存在。
- 历史根包模块 `articles.py`、`config.py`、`data_loader.py`、`training.py`、`evaluation.py`、`log.py` 和根包 `models/` 已移除。
- 编号 CLI 已开始直接导入业务域模块。
- 仍存在需要继续收敛的目标态差距：`foundation.paths` 仍持有业务路径，`recommendation` / `reports` 的跨域公开面还没有白名单测试，现有实施计划仍从第一轮迁移开始，已经落后于当前设计。

因此，本设计是“第一轮迁移后的目标态修订”，不是从零开始的包迁移说明。后续实施必须重写 `docs/superpowers/plans/2026-05-07-domain-driven-code-architecture.md`，不能继续按旧计划直接执行。

## 已确认原则

本设计采用“可读编排 CLI + 业务域内部职责拆分”的方案。编号入口暂时不迁出 `src/`，也不追求把每个编号脚本压缩成单行 runner；它们继续承担流程索引职责。

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

允许依赖关系是一个有向无环图，不是线性链式依赖。固定规则如下：

```text
domain          may import
foundation      stdlib / third-party only
datasets        foundation
transactions    foundation
catalog         foundation
trend           foundation, transactions, catalog
recommendation  foundation, transactions, catalog, trend prediction contracts/readers allowlist only
reports         foundation, domain public read-only contracts/readers allowlist only
CLI / app        public APIs of the domains they orchestrate
```

更具体地说：

- `foundation` 不依赖任何业务域。
- `datasets` 只依赖 `foundation`，并拥有原始数据路径、下载和 raw profile。
- `transactions` 依赖 `foundation`，可以消费 raw dataset，但不能依赖 `catalog`、`trend`、`recommendation` 或 `reports`。
- `catalog` 依赖 `foundation`，可以消费 raw articles，但不能依赖 `transactions`、`trend`、`recommendation` 或 `reports`。
- `trend` 依赖 `foundation`，并消费 `transactions` 和 `catalog` 的稳定产物或公开读取接口；它不能依赖 `recommendation` 或 `reports`。
- `recommendation` 依赖 `foundation`，并消费 `transactions`、`catalog` 和 `trend` 的稳定公开契约。
- `reports` 可以复用各领域公开的只读 contract 和 reader，不参与核心计算，也不依赖训练、评价、候选构建、重排序或推荐评价实现。
- 编号 CLI 和未来 app 可以调用被编排领域的公开 API，但不反向成为业务域依赖。

## 公共接口分层

跨模块接口分为三层，边界测试应按这三层执行。

### CLI 编排接口

编号 CLI 可以调用本阶段领域的高层业务函数，例如：

- `datasets.download.download_competition`
- `transactions.weekly.build_weekly_transactions`
- `catalog.articles.clean_articles_file`
- `catalog.graph.build_attribute_graph_files`
- `trend.article_sales.build_article_week_sales_frame`
- `trend.attribute_heat.build_attribute_week_heat_frame`
- `trend.training.run_trend_model_training`
- `trend.evaluation.run_trend_model_evaluation`

这些接口服务 CLI 编排，不等于跨领域公共读取面。`recommendation` 和 `reports` 不能因为 CLI 可调用某个函数，就依赖该函数的内部实现模块。

### 跨领域只读接口

跨领域只读 contract 和 reader 必须有可识别的模块边界。新领域优先命名为：

- `fashion_trend.<domain>.contracts` 或 `fashion_trend.<domain>.contracts.*`
- `fashion_trend.<domain>.readers` 或 `fashion_trend.<domain>.readers.*`

现有趋势域不新增 `trend/contracts/` 聚合层，避免把已有清晰模块再次包一层。趋势域公开白名单固定为：

- `fashion_trend.trend.schema`
- `fashion_trend.trend.predictions`
- `fashion_trend.trend.readers` 或 `fashion_trend.trend.readers.*`

`recommendation` 的上游公开白名单固定为：

- `fashion_trend.transactions.contracts` 或 `fashion_trend.transactions.contracts.*`
- `fashion_trend.transactions.readers` 或 `fashion_trend.transactions.readers.*`
- `fashion_trend.catalog.contracts` 或 `fashion_trend.catalog.contracts.*`
- `fashion_trend.catalog.readers` 或 `fashion_trend.catalog.readers.*`
- `fashion_trend.trend.schema`
- `fashion_trend.trend.predictions`
- `fashion_trend.trend.readers` 或 `fashion_trend.trend.readers.*`

这些模块只能定义产物 schema、payload contract、只读解析、只读校验和展示所需的轻量派生字段。不写文件，不训练模型，不构建候选，不重排序，不计算核心评价指标。

### 内部实现接口

除 CLI 编排接口和跨领域只读接口外，其余模块默认是领域内部实现。其他领域不能直接导入这些模块。尤其是：

- `recommendation` 不能导入 `fashion_trend.transactions.weekly`、`catalog.articles`、`catalog.graph` 或后续 `catalog.graph.*` 这类上游构建实现；它只能通过 `transactions`、`catalog` 和 `trend` 的公开只读白名单消费稳定产物。
- `recommendation` 不能导入 `fashion_trend.trend.article_sales`、`trend.attribute_heat`、`trend.targets`、`trend.samples`、`trend.splits`、`trend.training`、`trend.evaluation`、`trend.models` 或后续 `trend.heat`、`trend.labels`、`trend.features`。
- `reports` 不能导入任何领域的核心计算实现，例如 `datasets.download`、`transactions.weekly`、`catalog.graph` 的构建/发布实现、`trend.training`、`trend.evaluation`、`recommendation.candidates`、`recommendation.rerank`、`recommendation.evaluation`。
- `reports` 可以导入公开只读模块，例如 `catalog.readers`、`catalog.contracts`、`trend.schema`、`trend.predictions`、`trend.readers`、`recommendation.readers`、`recommendation.contracts`。

架构边界测试必须按 allowlist 执行：对 `recommendation` 和 `reports` 的每个 `fashion_trend.<domain>` import，先判定是否命中允许的 public read-only 模块；未命中则失败。不能只禁止 `trend.models`、`trend.training`、`trend.evaluation` 这几个显眼模块，也不能放行 `recommendation` 直接导入 `transactions.weekly` 或 `catalog.graph`。

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

目标态下，`foundation.paths` 只允许保留这些无业务语义的根路径：

- `PROJECT_ROOT`
- `DATA_DIR`
- `RAW_DIR`
- `INTERIM_DIR`
- `PROCESSED_DIR`
- `OUTPUT_DIR`

`foundation.paths` 可以提供通用路径安全 helper，例如“确认输出路径在某个根目录下”，但不能定义具体业务产物路径。

目标态下，下列内容不得继续留在 `foundation.paths`：

- 全局 `PATH` 字典。
- H&M 竞赛 slug 和 `RAW_HM_DIR` 这类数据集特定路径。
- `GRAPH_DIR`、`TREND_DIR`、`FEATURES_DIR` 这类领域目录。
- `OUTPUT_MODELS_DIR`、`OUTPUT_METRICS_DIR`、`OUTPUT_FIGURES_DIR`、`OUTPUT_REPORTS_DIR` 这类实验或报告目录。
- `TREND_SPLIT_VALID_WEEKS`、`TREND_SPLIT_TEST_WEEKS` 这类趋势业务配置。
- 任意 `trend_*`、`features_trend_*`、`graph_*`、`recommend_*` 文件路径。

这些路径应迁入对应领域：

- `datasets.paths`：竞赛 slug、raw H&M 根目录、raw transactions/articles/customers 路径。
- `transactions.paths`：周级交易表路径。
- `catalog.paths`：商品清洗产物、属性图节点/边产物路径。
- `trend.paths`：article sales、heat、target、samples、split、model output、metrics output 路径。
- `recommendation.paths`：用户画像、候选、重排序结果、推荐评价、推荐特征路径。
- `reports.paths`：figures、tables、case studies、报告导出路径。

全局 `PATH` 字典必须退役。领域内可以根据需要提供小范围 path dataclass 或语义常量，但不能再用跨领域字符串 key 共享所有路径。

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

趋势域内部继续按业务对象拆分。目标结构示例：

```text
trend/
  heat/
  labels/
  features/
  splits/
  models/
    base.py
    registry.py
    baselines/
      last_week.py
      moving_average.py
    supervised/
      lightgbm.py
  training/
    runner.py
    outputs.py
  evaluation/
    metrics.py
    payloads.py
    runner.py
  readers.py
```

具体文件和子包不按固定模板预设，实施时按职责复杂度确定。

`schema.py` 与 `predictions.py` 是趋势域对外稳定契约，继续定义预测表 schema、metric payload 需要的字段和预测契约校验。`readers.py` 或 `readers/` 只提供读取稳定产物的函数，供 recommendation 和 reports 复用。趋势域内部的 `heat/`、`labels/`、`features/`、`splits/`、`models/`、`training/`、`evaluation/` 不作为 recommendation 或 reports 的公共依赖面。

`models/base.py` 定义模型训练协议、训练上下文、训练结果和 artifact 契约。`models/registry.py` 是模型名到 trainer 的唯一映射入口，外部训练 runner 只通过 registry 获取模型，不直接 import 具体模型实现。

`models/baselines/` 存放确定性或轻量 baseline，例如 `last_week` 和 `moving_average`。`models/supervised/` 存放需要拟合的监督学习模型，例如后续 LightGBM。这个分层只影响内部代码组织，不改变模型名、`--model` CLI、预测表 schema 或 `outputs/models/<model>/` 产物路径。

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

`reports` 读取 `data/processed` 与 `outputs` 下的稳定产物。它可以复用 `transactions`、`catalog`、`trend`、`recommendation` 暴露的只读 schema、contract 和 reader，避免重复解析产物格式；但这些公开入口必须位于 `contracts` / `readers` 这类白名单模块中。`reports` 不能依赖训练 runner、评价指标计算、推荐候选构建、推荐重排序等核心计算实现，也不能把展示需求反向写入业务域。

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
10_train_trend_model.py             trend, 统一训练入口，包括 --model lightgbm
11_eval_trend_model.py              trend
12_build_user_profile.py            recommendation
13_build_recommend_candidates.py    recommendation
14_rerank_recommendations.py        recommendation
15_eval_recommendations.py          recommendation
16_make_reports.py                  reports, 覆盖 figures/tables/cases
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
data/processed/features/        existing trend feature contract only
data/processed/recommend/       recommendation intermediate tables and features
outputs/models/<model>/         trend model artifacts
outputs/metrics/<model>/        trend metrics
outputs/recommendation/         recommendation results and metrics
outputs/reports/                reports figures/tables/cases
```

`data/processed` 放可复现的中间数据和特征表。`outputs` 放实验、模型、指标、报告和展示产物。

趋势预测 `predictions.csv` 继续属于 `outputs/models/<model>/predictions.csv`，不挪进 `data/processed/trend/`。趋势评价继续写入 `outputs/metrics/<model>/trend_metrics.json`。

`data/processed/features/` 保留给已存在的趋势训练样本契约，例如 `trend_model_samples.parquet` 和 split parquet。新增推荐特征不再写入该目录，应进入 `data/processed/recommend/` 或 `data/processed/recommend/features/`。多个领域不能共同写同一文件。

## 迁移策略

当前项目已经完成第一轮业务域迁移。后续迁移重点不是再次创建包骨架或从根包搬文件，而是冻结目标边界、迁出路径 ownership、建立跨领域只读接口，并继续拆分现有大模块。

迁移可以破坏内部 import，但不应一次性混入行为修改。每个阶段都必须能独立验证。现有 `docs/superpowers/plans/2026-05-07-domain-driven-code-architecture.md` 已经落后于本设计，不能直接执行；进入实现前应先按本设计重写实施计划。

### 阶段 0：重写实施计划

先替换现有实施计划，使 plan 从当前代码基线出发。新的 plan 不应再包含“创建业务包骨架”“迁移根包 `articles.py` / `training.py` / `evaluation.py` / `models/`”这类已完成任务。

新的 plan 应拆成这些可独立提交的阶段：

1. 架构边界测试目标态化。
2. 业务路径 ownership 迁移。
3. 跨领域 readers/contracts 建立。
4. `catalog` 图构建职责拆分。
5. `trend` deterministic pipeline 收敛。
6. `trend` experiments 收敛。
7. `recommendation` 与 `reports` 新业务边界落地。
8. README 与 `docs/gpt-research/implementation-plan.md` 同步。

### 阶段 1：架构边界测试目标态化

保留并强化架构边界测试，确保历史根模块不回流，`foundation` 不引入业务语义，编号 CLI 仍作为流程索引保留在 `src/` 中。

同时把架构边界测试调整为目标态规则：

- `recommendation` 对 `transactions`、`catalog` 和 `trend` 的 import 必须全部命中上游公开白名单；其中 `trend` 只能导入 `trend.schema`、`trend.predictions`、`trend.readers` 这类趋势预测契约模块。
- `reports` 只允许导入领域公开 `contracts` / `readers` 白名单模块，以及显式白名单内的 `trend.schema`、`trend.predictions`、`trend.readers`。
- `recommendation` 和 `reports` 继续禁止导入训练、评价、样本构建、切分、候选构建、重排序等计算实现。
- `foundation.paths` 中业务产物路径迁出后，测试应按允许项白名单防止新的业务路径回流到 `foundation`。
- `reports` 对各领域的 import 必须全部命中公开 `contracts` / `readers` / 显式 allowlist。

这一步应先写失败测试，再按后续阶段逐步修正代码结构。不能只用 denylist 禁止 `trend.models`、`trend.training`、`trend.evaluation`，因为这会漏掉 `trend.samples`、`trend.splits`、`trend.attribute_heat`、`trend.targets` 等内部流水线，也会漏掉 `recommendation` 直接依赖 `transactions.weekly` 或 `catalog.graph` 的情况。

### 阶段 2：迁移业务路径 ownership

先建立领域路径模块或契约，再迁移 import，不和业务逻辑重组混在一起：

- `foundation.paths` 只保留允许项白名单中的根路径。
- `datasets.paths` 持有 H&M raw 路径和竞赛 slug。
- `transactions.paths` 持有周级交易表路径。
- `catalog.paths` 持有商品清洗与属性图产物路径。
- `trend.paths` 持有 article sales、heat、target、samples、split、model output 和 metrics output 路径。
- `recommendation.paths` 持有用户画像、候选、重排序结果、推荐评价和推荐特征路径。
- `reports.paths` 持有 figures、tables、cases 和报告导出路径。
- 编号 CLI 只从领域路径模块读取本阶段业务路径。

该阶段只迁移路径 ownership 和 import，不改变产物实际落盘路径。迁移完成后：

- 根包和业务包内不再导入全局 `PATH`。
- `foundation.paths` 不再导出 `PATH`。
- `foundation.paths` 不再导出业务目录或业务文件路径。
- 架构边界测试按允许项白名单检查 `foundation.paths`。

### 阶段 3：建立跨领域 readers/contracts

先建立最小 public read-only surface，再让后续 `recommendation` 和 `reports` 只依赖这些入口：

- `transactions.readers` / `transactions.contracts`：周级交易表只读契约。
- `catalog.readers` / `catalog.contracts`：商品属性图节点和边的只读契约。
- `trend.readers`：趋势稳定产物、预测产物和 metrics payload 的只读入口。
- `trend.schema` 与 `trend.predictions` 保持为趋势预测公开契约。
- 后续新增 `recommendation.readers` / `recommendation.contracts`，供 reports 读取推荐结果和评价。

这些 reader 只能读取和校验稳定产物，不能构建产物、写文件、训练模型、计算核心指标或执行重排序。

### 阶段 4：拆分 catalog 图构建职责

在 `catalog` 域内拆分属性图 schema、节点/边构建、读取校验、发布回滚职责。保持 `03_clean_articles.py`、`04_build_attribute_graph.py`、图节点表和图边表产物契约不变。

建议目标结构：

```text
catalog/
  articles.py
  graph/
    schema.py
    builders.py
    readers.py
    publishing.py
```

若实施时发现拆成子包会制造过度碎片，可以保留文件级拆分；但 `catalog/graph.py` 不应继续同时持有 schema、构建、读取、校验、发布和回滚。

### 阶段 5：收敛 trend deterministic pipeline

按 `heat/`、`labels/`、`features/`、`splits/` 等真实业务对象收敛趋势确定性流水线。保持 article sales、heat、target、sample、split 产物路径和 schema 不变。

建议目标结构：

```text
trend/
  schema.py
  predictions.py
  readers.py
  article_sales.py
  heat/
  labels/
  features/
  splits/
```

`trend.readers` 是跨领域公开读取入口；`heat/`、`labels/`、`features/`、`splits/` 是趋势域内部实现，不允许 recommendation 或 reports 直接依赖。

### 阶段 6：trend experiments

先把 `trend/models/` 整理为 `base.py`、`registry.py`、`baselines/` 和 `supervised/`。`last_week`、`moving_average` 移入 `baselines/`，后续 LightGBM 放入 `supervised/`；registry 继续提供唯一模型发现入口。LightGBM 训练不新增专用编号脚本，统一通过 `src/10_train_trend_model.py --model lightgbm` 运行。

然后拆分 `trend/training.py` 与 `trend/evaluation.py` 的内部职责。训练侧优先拆出 runner、metadata/output payload、artifact 发布与回滚；评价侧优先拆出 metrics、payloads、runner。保持 `--model` CLI、模型名、预测表 schema、`outputs/models/<model>/` 和 `outputs/metrics/<model>/` 不变。

建议目标结构：

```text
trend/
  models/
    base.py
    registry.py
    baselines/
    supervised/
  training/
    runner.py
    outputs.py
  evaluation/
    metrics.py
    payloads.py
    runner.py
```

### 阶段 7：recommendation 与 reports

在已有包边界内新增推荐和报告业务。推荐模块作为后续新增业务，不混进前四个结构迁移阶段；报告域覆盖 figures、tables 和 case studies，不只生成单一图片脚本。

`recommendation` 只能通过 public read-only surface 消费上游稳定产物。`reports` 只能通过 public read-only surface 消费各领域产物。它们都不能反向依赖训练、评价、候选构建或重排序实现。

### 阶段 8：文档与编号契约同步

更新 `README.md` 与 `docs/gpt-research/implementation-plan.md`，确保架构描述、编号脚本和产物口径与目标态一致：

- LightGBM 通过 `src/10_train_trend_model.py --model lightgbm` 运行，不新增 `12_train_lightgbm_trend_model.py`。
- LightGBM 产物继续落在 `outputs/models/lightgbm/` 与 `outputs/metrics/lightgbm/`。
- `12_build_user_profile.py` 到 `15_eval_recommendations.py` 归属 recommendation。
- `16_make_reports.py` 覆盖 figures、tables 和 case studies；旧的 `16_make_figures.py` 只作为历史命名引用清理。
- 推荐特征路径写入 `data/processed/recommend/` 或其子目录，不写入 `data/processed/features/`。

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

- 根包不再有 `articles.py`、`config.py`、`data_loader.py`、`evaluation.py`、`log.py`、`training.py` 这类历史散落模块，也不再有根包 `models/`。
- `foundation` 不依赖任何业务域。
- `foundation.paths` 只导出允许项白名单中的根路径，不导出全局 `PATH` 字典、业务目录、业务文件路径、业务配置或竞赛 slug。
- `catalog` 不依赖 `trend` 或 `recommendation`。
- `trend` 不依赖 `recommendation` 或 `reports`。
- `recommendation` 对 `transactions`、`catalog` 和 `trend` 的依赖只命中 public read-only allowlist；不依赖周交易构建、商品清洗、属性图构建、趋势确定性流水线、切分、训练、评价或模型内部实现。
- `reports` 对各领域的依赖只命中公开 `contracts` / `readers` / 显式 allowlist；不依赖下载、构建、训练、评价、候选、重排序等计算实现。
- 架构边界测试使用 allowlist 检查 `recommendation` 和 `reports`，不是只用少量 denylist。
- `trend/models/` 按 `base.py`、`registry.py`、`baselines/`、`supervised/` 管理模型实现。
- `catalog/graph.py` 的 schema、构建、读取校验、发布回滚职责已拆分或明确隔离。
- `trend/evaluation.py` 的 metrics、payload、runner 职责已拆分或明确隔离。
- `trend/training.py` 的 runner、outputs、artifact 发布回滚职责已拆分或明确隔离。
- 编号 CLI 保持可运行、可读，且用户命令不变。
- 编号 CLI 能展示高层数据流，但不承载业务计算事实。
- 既有产物路径和 schema 不变。
- `README.md` 和 `docs/gpt-research/implementation-plan.md` 的架构描述与实际代码一致。
- `docs/superpowers/plans/2026-05-07-domain-driven-code-architecture.md` 已按本设计重写，不再执行过期的第一轮迁移计划。
- 测试通过，必要 CLI 验证通过。
- 删除旧 facade 和旧 import 后，仓库内不再残留历史路径引用。

## 风险与控制

最大风险是把结构迁移和行为修改混在一起，导致产物漂移但难以定位。控制方式是按阶段迁移、每阶段检查 diff、运行测试和必要 CLI，并在触及核心产物时做 schema 或 checksum 对比。

第二个风险是 `foundation` 变成业务工具垃圾桶。控制方式是禁止任何业务字段名、业务公式、业务对象进入 `foundation`。

第三个风险是 `data/processed/features/` ownership 模糊。控制方式是把该目录限定为现有趋势样本契约，新增推荐特征进入 `data/processed/recommend/` 或其子目录，避免多个领域共同写同一个产物。

第四个风险是编号 CLI 过度变薄，丧失流程索引价值。控制方式是把 `read -> build -> validate -> write` 的高层步骤保留在脚本里，但把计算事实留在业务包。

第五个风险是边界测试过宽或过窄。过宽会让 `recommendation` 悄悄依赖趋势内部流水线；过窄会让 `reports` 无法复用稳定 reader，迫使展示代码重复解析产物。控制方式是对 `recommendation` 和 `reports` 使用 allowlist，而不是只写 denylist。

第六个风险是继续执行过期计划。控制方式是先重写 plan，再实施；plan 的第一阶段必须从当前仓库状态出发，而不是从包骨架创建开始。
