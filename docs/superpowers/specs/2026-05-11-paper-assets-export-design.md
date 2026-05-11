# 论文图表与案例导出设计

## 背景

当前项目已经完成 H&M 原始交易数据到属性周级趋势预测，再到轻量 Top-N 推荐实验的主要闭环。`docs/gpt-research/project-status-summary.md` 已经整理了论文素材现状，README 也明确 `reports/` 仍是预留层，图表、表格和案例导出尚未实现。

本设计补齐最终论文所需的核心图表、表格和案例导出。目标是先服务中文论文正文的严谨表达，再让同一批素材可复用于答辩展示。导出阶段只读取已经发布的稳定 artifact，不重新训练趋势模型，不重跑推荐方法，不把本次工作扩大成新的推荐实验阶段。

## 已确认决策

- 范围采用“核心论文包 + 分析增强”。
- 每张图同时导出 SVG 和 PNG。
- 表格同时导出 CSV 和 Markdown。
- 图表语言采用双语偏中文：标题、图例和说明使用中文，模型名、指标名、字段名保留英文。
- 新增 `matplotlib` 作为 reports 阶段绘图依赖；后续只有在明确需要更复杂图布局时才评估其他依赖。
- 第一版导出 3 个可复现筛选出来的用户推荐案例。
- 采用 `reports` 领域包 + 单一编号导出入口，不使用一次性 notebook 或把图表逻辑塞回 trend / recommendation。

## 目标

- 固化最终论文需要的核心图表、表格和案例清单。
- 提供可重复运行的导出入口：`uv run python src/17_export_paper_assets.py`。
- 将输出统一写入 `outputs/reports/`，并生成 manifest 记录输入、输出、参数、warnings 和案例选择结果。
- 复用现有 stable artifact 和公开 reader / contract，保持 reports 只读汇总层边界。
- 通过测试覆盖 loader 校验、表格生成、图文件生成、案例筛选和 manifest 契约。

## 非目标

- 不新增趋势模型、推荐方法、深度推荐或在线服务。
- 不补跑完整严格推荐消融，例如独立 `w/o Recent`，也不把每个 trend 权重组合的 test 指标作为本轮必须新增 artifact。
- 不提交生成的 `outputs/reports/` 图表、表格或案例。
- 不实现交互式 dashboard、网页展示、Streamlit 或 notebook 工作流。
- 不默认导出 `recommendation_items.csv`，案例读取以 `recommendation_items.parquet` 为准。

## 架构

新增报告导出阶段作为只读汇总层：

```text
src/17_export_paper_assets.py
    -> fashion_trend.reports.runner
        -> loaders
        -> tables
        -> figures
        -> cases
        -> manifest
```

`src/17_export_paper_assets.py` 是薄编排入口，只负责解析参数、调用 reports runner、输出摘要和返回稳定退出码。核心读取、转换、绘图、案例筛选和 manifest 写入都位于 `src/fashion_trend/reports/`。

`reports` 允许消费的上游公开面保持为架构测试白名单内的只读接口：

```text
fashion_trend.transactions.contracts
fashion_trend.transactions.readers
fashion_trend.catalog.contracts
fashion_trend.catalog.readers
fashion_trend.trend.schema
fashion_trend.trend.predictions
fashion_trend.trend.readers
fashion_trend.recommendation.contracts
fashion_trend.recommendation.readers
```

`reports` 不能导入趋势训练 runner、趋势评价 runner、趋势模型实现、推荐候选构建、推荐重排序、推荐实验 runner 或 catalog graph builder。展示需求不能反向污染核心业务域。

## 模块设计

目标结构：

```text
src/fashion_trend/reports/
  __init__.py
  paths.py
  loaders.py
  tables.py
  figures.py
  cases.py
  manifest.py
  runner.py
```

### `loaders.py`

负责读取稳定 artifact，并做统一输入校验：

- 检查路径存在且不是目录。
- 校验 CSV / parquet / JSON schema。
- 保留 `article_id`、`customer_id`、`prediction` 的字符串语义，尤其不能丢失前导 0。
- 读取趋势 metrics、LightGBM feature importance、LightGBM predictions、趋势样本、属性图节点边表、推荐 metrics、推荐 experiment、推荐长表和推荐输入表。
- 对缺失 artifact 给出明确上游命令提示。

### `tables.py`

负责生成论文表格数据，输出 CSV 和 Markdown：

| 表格文件前缀 | 内容 |
| --- | --- |
| `data_artifact_summary` | 原始数据、处理后数据、模型与推荐 artifact 规模 |
| `time_split_summary` | 趋势 train/valid/test 与推荐 valid/test 窗口 |
| `attribute_graph_summary` | 属性图节点、商品-属性边、属性层级边统计 |
| `trend_feature_summary` | 趋势样本特征清单与 LightGBM 特征组说明 |
| `trend_model_metrics` | `last_week`、`previous_growth`、`moving_average`、`lightgbm` 的 valid/test 指标 |
| `trend_metrics_by_attr_type` | LightGBM 或模型对比的属性类型分组指标 |
| `recommendation_method_metrics` | 五个推荐方法的 valid/test 指标 |
| `recommendation_experiment_summary` | 主实验权重、grid search 摘要和当前 experiment artifact 支持的消融 |

Markdown 表格用于论文粘贴和人工审阅，CSV 用于复核和后续制图。

### `figures.py`

负责使用 `matplotlib` 导出 SVG + PNG。第一版固定 8 张图：

| 图文件前缀 | 图表内容 | 主要数据来源 |
| --- | --- | --- |
| `data_pipeline` | 原始数据到趋势预测和推荐应用的流程图 | 代码内固定流程定义 |
| `attribute_graph_schema` | 商品节点、属性节点、商品-属性边、层级边示意 | 属性图统计和固定结构定义 |
| `trend_curve_examples` | 典型属性最近 8 周真实热度、预测增长和预测份额曲线 | LightGBM predictions / trend samples |
| `lightgbm_feature_importance` | LightGBM Top-N 特征重要性 | `outputs/models/lightgbm/feature_importance.csv` |
| `trend_model_metrics` | 趋势模型 MAE / Spearman / NDCG@10 对比 | `outputs/metrics/<model>/trend_metrics.json` |
| `recommendation_method_metrics` | 推荐方法 MAP@12 / Recall@12 / NDCG@12 对比 | `outputs/recommendation/<method>/metrics.json` |
| `topk_trend_attributes` | test `week_id=103` 的颜色、品类、图案 Top-K 趋势榜 | LightGBM predictions |
| `recommendation_weight_analysis` | `trend_score` 权重与 valid 指标、主实验权重构成 | `outputs/recommendation/experiments/main/experiment.json` |

图表默认使用中文标题和说明，保留 `LightGBM`、`NDCG@10`、`NDCG@12`、`MAP@12`、`Recall@12` 等英文指标名。SVG 是论文排版主格式，PNG 是答辩和快速预览格式。

### `cases.py`

负责筛选并导出 3 个用户推荐案例。默认读取 `pop_similarity_trend` 的 test split 推荐长表，筛选规则必须可复现：

1. 只考虑 test split。
2. 优先选择命中数高的用户窗口。
3. 用户历史偏好属性至少包含清晰的 `graphical_appearance_name`、`product_group_name` 或 `colour_group_name`。
4. 推荐项必须包含完整的 `pop_score`、`sim_score`、`trend_score`、`recent_score` 和 `candidate_sources`。
5. 同一案例内 Top-12 商品不能重复，且 `article_id` 字符串语义必须保留。

每个案例输出：

- 用户 ID、split、cutoff week、label week。
- 用户历史偏好属性 Top-N。
- 代表性趋势属性。
- Top-12 推荐商品、命中标记、商品属性和分数分解。
- 简短案例解读。

输出格式为 JSON + Markdown。JSON 便于复核，Markdown 便于论文和答辩整理。

### `manifest.py` 与 `runner.py`

`runner.py` 负责编排完整导出，`manifest.py` 负责构造和校验 manifest。`outputs/reports/manifest.json` 至少包含：

- `generated_at`
- `parameters`
- `input_artifacts`
- `output_artifacts`
- `row_counts`
- `figure_count`
- `table_count`
- `case_count`
- `case_user_ids`
- `warnings`
- `schema_version`

manifest 是 reports 阶段的审计入口。只要导出成功，manifest 必须能说明每个输出来自哪些输入。

## 输出目录契约

第一版输出目录固定为：

```text
outputs/reports/
  figures/
    *.svg
    *.png
  tables/
    *.csv
    *.md
  case_studies/
    *.json
    *.md
  manifest.json
```

`outputs/reports/` 是运行时产物目录，不提交到 git。后续论文文档只引用这些产物路径，不把二进制图片纳入源码提交。

## 数据流与指标口径

导出流程从稳定 artifact 开始，不重跑上游阶段：

- 趋势模型对比读取四个 `outputs/metrics/<model>/trend_metrics.json`。
- LightGBM 特征重要性读取 `outputs/models/lightgbm/feature_importance.csv`。
- 趋势曲线和 Top-K 趋势榜读取 `outputs/models/lightgbm/predictions.csv`，必要时结合趋势样本列。
- 推荐方法对比读取五个 `outputs/recommendation/<method>/metrics.json`。
- 推荐权重和消融摘要读取 `outputs/recommendation/experiments/main/experiment.json`。
- 推荐案例读取 `outputs/recommendation/pop_similarity_trend/recommendation_items.parquet` 和推荐输入 / 商品属性公开 reader。

口径约束：

- 趋势模型选择解释优先看 valid，最终报告同时展示 valid 和 test。
- test 指标只能作为最终报告，不能在图表叙述中暗示用于调参选择。
- 推荐主方法应表述为 valid 最优、test 接近强 `recent_popularity` baseline，并显著优于不含趋势分的 `pop_similarity`。
- `topk_trend_attributes` 默认取 test `week_id=103`，过滤条件为 `is_trend_eligible_t = 1`、`heat_t >= 20`、`history_total_heat_t >= 100`、`history_active_weeks_t >= 8`。
- 数据流程图和属性层次图是论文示意图，不伪装成全量网络布局图；精确规模由表格承载。

## CLI 设计

默认入口：

```sh
uv run python src/17_export_paper_assets.py
```

建议参数：

```text
--case-count 3
--top-k 10
--trend-week 103
--figure-format svg,png
--output-dir outputs/reports
```

默认参数应覆盖本设计的论文第一版需求。参数只控制 reports 导出，不触发上游 artifact 重建。

## 错误处理

关键问题 fail-fast：

- 必需输入路径缺失。
- JSON / CSV / parquet 无法读取。
- schema 或必需字段缺失。
- 指标字段缺失或不是有限数值。
- ID dtype 被破坏。
- `recommendation_items.parquet` 缺失或被 CSV 替代。
- 可复现规则下不足 3 个案例。
- SVG / PNG / CSV / Markdown / JSON 输出缺失或为空。

错误信息必须包含路径和建议上游命令。例如缺少 `outputs/models/lightgbm/predictions.csv` 时，应提示先运行 LightGBM 训练和评价链路，而不是在 reports 内添加 fallback。

非阻塞情况写入 manifest `warnings`：

- 当前 experiment artifact 不包含严格 `w/o Recent` 独立消融。
- 当前 grid search 只有 valid 指标，没有每组权重的 test 指标。
- 工作区存在历史 `recommendation_items.csv`，但 reports 未使用。
- 某个可选分析图的数据点少于预期，但核心图表和案例仍完整。

## 测试策略

### 单元测试

新增 `tests/test_reports_*.py`，覆盖：

- loader 缺失路径、schema 错误、dtype 保留和非法 JSON。
- 表格生成的列、行数和 Markdown 输出。
- figure 导出同时生成 SVG 和 PNG，且文件非空。
- case selector 能选出 3 个合法案例，并在案例不足时失败。
- manifest payload 必需字段、输入输出路径记录和 warnings。

测试使用小型 fixture，不依赖真实 H&M 数据集。

### 集成 smoke

用小型 fixture 运行 reports runner，确认：

- `figures/`、`tables/`、`case_studies/` 和 `manifest.json` 都生成。
- SVG / PNG / CSV / Markdown / JSON 文件非空。
- manifest 的输出清单与实际文件一致。

### 真实产物验证

实现完成后运行：

```sh
uv run python src/17_export_paper_assets.py
uv run pytest tests/test_reports_*.py tests/test_architecture_boundaries.py
uv run python -m compileall -q src
uv run black --check src tests
uv run isort --check-only src tests
git diff --check
```

真实导出后还要检查：

- `outputs/reports/figures/` 有 8 个图表前缀，且每个都有 SVG 和 PNG。
- `outputs/reports/tables/` 有 CSV 和 Markdown。
- `outputs/reports/case_studies/` 有 3 个案例 JSON 和 3 个案例 Markdown。
- `manifest.json` 记录完整输入、输出、warnings 和案例用户。

## 文档同步

实现时需要同步：

- `README.md`：补充 reports 阶段入口、输出路径和图表/案例用途。
- `docs/gpt-research/implementation-plan.md`：将 reports 目录结构中的图表和表格从建议状态更新为 as-built 行为。
- `docs/gpt-research/project-status-summary.md`：可在导出完成后记录最终图表和案例文件清单。

文档不得把 reports 描述为在线展示系统，也不得暗示推荐主方法在 test 上全面超过 `recent_popularity`。

## 验收标准

- `src/17_export_paper_assets.py` 能在现有稳定 artifact 存在时一次性导出全部论文素材。
- 第一版生成 8 张图，每张都有 SVG 和 PNG。
- 第一版生成表格 CSV 和 Markdown，覆盖数据规模、属性图、切分、趋势指标、推荐指标、实验权重和消融摘要。
- 第一版生成 3 个用户推荐案例，案例可由规则复现。
- `outputs/reports/manifest.json` 能审计输入、输出、参数、warnings 和案例选择。
- reports 架构边界测试通过，不依赖上游计算实现。
- 新增 tests 和现有架构验证通过。
