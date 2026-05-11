# 论文图表与案例导出设计

## 背景

当前项目已经完成 H&M 原始交易数据到属性周级趋势预测，再到轻量 Top-N 推荐实验的主要闭环。`docs/gpt-research/project-status-summary.md` 已经整理了论文素材现状，README 也明确 `reports/` 仍是预留层，图表、表格和案例导出尚未实现。

本设计补齐最终论文所需的核心图表、表格和案例导出。目标是先服务中文论文正文的严谨表达，再让同一批素材可复用于答辩展示。导出阶段只读取已经发布的稳定 artifact，不重新训练趋势模型，不重跑推荐方法，不把本次工作扩大成新的推荐实验阶段。

## 已确认决策

- 范围采用“核心论文包 + 分析增强”。
- 每张图同时导出 SVG 和 PNG。
- 表格同时导出 CSV 和 Markdown。
- 图表语言采用双语偏中文：标题、图例和说明使用中文，模型名、指标名、字段名保留英文。
- 新增 `matplotlib` 作为 reports 阶段唯一新增依赖；Markdown 表格使用项目内简单 pipe-table writer，不依赖 `tabulate` 或 `pandas.DataFrame.to_markdown()`。
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
- 为趋势曲线和 Top-K 趋势榜构建 `LightGBM predictions + trend_model_samples` 的 1:1 join 视图。
- 对缺失 artifact 给出明确上游命令提示。

趋势预测与样本 join 契约：

- join key 固定为 `week_id + attr_id + attr_type + attr_value`。
- `outputs/models/lightgbm/predictions.csv` 与 `data/processed/features/trend_model_samples.parquet` 在 join key 上都必须唯一。
- join 后行数必须等于 LightGBM predictions 行数，不能出现 left-only 或 right-only 预测行。
- join 后必须保留 `split`、`share_t`、`pred_share_t1`、`target_growth`、`pred_target_growth`、`target_rank_in_type_t1`，并从样本表补充 `heat_t`、`history_total_heat_t`、`history_active_weeks_t`、`is_trend_eligible_t`。
- 如果 predictions 和 samples 中同名字段语义重复，例如 `share_t` 或 `target_growth`，loader 必须校验数值一致后只暴露一个标准列，不能静默采用任意一侧。

### `tables.py`

负责生成论文表格数据，输出 CSV 和 Markdown：

| 表格文件前缀 | 排序 | 必需列 |
| --- | --- | --- |
| `data_artifact_summary` | `section, artifact` | `section`, `artifact`, `path`, `row_count`, `column_count`, `paper_usage` |
| `time_split_summary` | `domain, split, week_start` | `domain`, `split`, `week_start`, `week_end`, `week_count`, `row_count`, `attribute_count`, `user_count` |
| `attribute_graph_summary` | `entity_type, attr_type, relation_type` | `entity_type`, `attr_type`, `relation_type`, `count`, `path`, `paper_usage` |
| `trend_feature_summary` | `feature_group, feature_name` | `feature_group`, `feature_name`, `source_table`, `model_input`, `description` |
| `trend_model_metrics` | `model_name, split` | `model_name`, `split`, `mae`, `rmse`, `spearman`, `ndcg_at_10`, `precision_at_10`, `recall_at_10`, `run_id` |
| `trend_metrics_by_attr_type` | `model_name, split, attr_type` | `model_name`, `split`, `attr_type`, `mae`, `rmse`, `spearman`, `ndcg_at_10`, `precision_at_10`, `recall_at_10` |
| `recommendation_method_metrics` | `method, split` | `method`, `split`, `map_at_12`, `recall_at_12`, `hit_rate_at_12`, `ndcg_at_12`, `coverage`, `user_count`, `missing_recommendation_user_count` |
| `recommendation_experiment_summary` | `section, rank, split, method` | `section`, `rank`, `method`, `split`, `pop_score`, `sim_score`, `trend_score`, `recent_score`, `map_at_12`, `recall_at_12`, `hit_rate_at_12`, `ndcg_at_12`, `coverage` |

Markdown 表格用于论文粘贴和人工审阅，CSV 用于复核和后续制图。

表格输出约束：

- CSV 和 Markdown 使用同一 DataFrame 和同一列顺序生成。
- 数值列保留原始浮点精度到 CSV；Markdown 可以按论文展示需要格式化到固定小数位。
- 缺少某张表的必需列必须失败，不能由实现临时省略列。
- `recommendation_experiment_summary` 只能总结当前 `experiment.json` 已保存的 valid grid 与 ablation，不虚构不存在的 test grid 指标。
- Markdown 不能调用 `DataFrame.to_markdown()`，因为它隐式依赖当前项目未声明的 `tabulate`。
- 实现项目内 `write_markdown_table()` 或等价 helper，只支持本阶段需要的 GitHub Flavored Markdown pipe table：表头、分隔行、字符串转义、空值显示、数值格式化和稳定列顺序。
- 该 helper 必须对单元格中的换行和 `|` 做安全转义，避免生成无法复制到论文草稿中的破损表格。
- 如果后续决定使用 `tabulate`，必须作为单独依赖决策写入 spec / plan，并更新 `pyproject.toml` 与 `uv.lock`；本设计第一版不引入。

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
| `topk_trend_attributes` | test `week_id=103` 的颜色、品类、图案 Top-K 趋势榜 | LightGBM predictions + trend samples join view |
| `recommendation_weight_analysis` | `trend_score` 权重与 valid 指标、主实验权重构成 | `outputs/recommendation/experiments/main/experiment.json` |

图表默认使用中文标题和说明，保留 `LightGBM`、`NDCG@10`、`NDCG@12`、`MAP@12`、`Recall@12` 等英文指标名。SVG 是论文排版主格式，PNG 是答辩和快速预览格式。

## 绘图运行环境

实现阶段需要把 `matplotlib` 加入 `pyproject.toml` 的 runtime dependencies，并通过 `uv sync` 更新锁文件。reports 导出入口启动时必须初始化绘图环境：

- 使用非交互式后端，例如 `Agg`。
- 设置候选 CJK 字体栈：`PingFang SC`、`Heiti SC`、`Songti SC`、`Noto Sans CJK SC`、`Microsoft YaHei`、`SimHei`、`Arial Unicode MS`。
- 设置 `axes.unicode_minus = False`，避免负号在中文字体环境下显示异常。
- 默认要求至少找到一个可用 CJK 字体；找不到时 fail-fast，错误信息说明如何安装字体或改用英文标签。
- 如果后续实现提供 `--allow-missing-cjk-font` 之类显式参数，才允许降级为 manifest warning；默认论文导出不允许悄悄生成缺字图。

测试需要包含一个最小中文图渲染 smoke：生成含中文标题、英文指标名和负数坐标轴的 SVG + PNG，并确认文件非空。该 smoke 不需要验证字体视觉效果，但要覆盖字体发现、`unicode_minus` 配置和双格式写出链路。

除 `matplotlib` 外，本阶段不新增 `tabulate`、`seaborn`、`plotly`、`altair`、`networkx` 或 `Pillow`。如果后续某张图确实需要额外依赖，必须先更新设计或实施计划，说明用途和影响范围。

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
- 趋势曲线和 Top-K 趋势榜读取 `outputs/models/lightgbm/predictions.csv` 与 `data/processed/features/trend_model_samples.parquet` 的 1:1 join 视图。
- 推荐方法对比读取五个 `outputs/recommendation/<method>/metrics.json`。
- 推荐权重和消融摘要读取 `outputs/recommendation/experiments/main/experiment.json`。
- 推荐案例读取 `outputs/recommendation/pop_similarity_trend/recommendation_items.parquet` 和推荐输入 / 商品属性公开 reader。

口径约束：

- 趋势模型选择解释优先看 valid，最终报告同时展示 valid 和 test。
- test 指标只能作为最终报告，不能在图表叙述中暗示用于调参选择。
- 推荐主方法应表述为 valid 最优、test 接近强 `recent_popularity` baseline，并显著优于不含趋势分的 `pop_similarity`。
- `topk_trend_attributes` 默认取 test `week_id=103`，过滤字段来自 predictions + samples join 视图；过滤条件为 `is_trend_eligible_t = 1`、`heat_t >= 20`、`history_total_heat_t >= 100`、`history_active_weeks_t >= 8`。
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
- LightGBM predictions 与 trend samples 无法按 join key 做 1:1 对齐。
- 缺少可用 CJK 字体，导致中文图表无法可靠渲染。
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
- LightGBM predictions 与 trend samples join key 重复、缺失或字段不一致时失败。
- 表格生成的列、行数和 Markdown 输出。
- Markdown pipe-table writer 覆盖 `|`、换行、空值、数值格式化和列顺序，不依赖 `tabulate`。
- figure 导出同时生成 SVG 和 PNG，且文件非空。
- 绘图环境 smoke 覆盖中文标题、英文指标名、负数坐标轴、CJK 字体发现和 `axes.unicode_minus = False`。
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
