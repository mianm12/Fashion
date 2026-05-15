# 趋势图谱特征增强与组级消融设计

## 范围

本设计为当前较粗的图结构特征新增一组可解释的属性知识图谱上下文特征，并通过独立实验验证这些特征对 LightGBM 趋势预测的贡献。

本轮只做独立实验，不改变默认主流程：

- 不改变 LightGBM 主模型类型。
- 不引入知识图谱嵌入、图神经网络、向量召回或在线推荐服务。
- 不修改默认 `data/processed/features/trend_model_samples*.parquet` schema。
- 不覆盖 `outputs/models/lightgbm/`、`outputs/reports/`、`outputs/reports/manifest.json` 或 defense app 数据源。
- 不把本实验加入 `00-18` 默认 stable pipeline。

实验唯一入口为：

```sh
uv run python src/19_run_trend_graph_feature_ablation.py
```

`19` 号脚本仅表示它发生在默认 `00-18` 主流程之后，是非主线独立补充实验。所有实验产物只写入：

```text
outputs/experiments/trend_graph_feature_ablation/
```

后续如果结果稳定且论文决定正式采用，再通过单独的非默认导出命令把汇总表复制到 `outputs/reports/experimental/`。本设计不把该复制动作纳入默认 reports 导出。

## 设计目标

当前趋势样本里的图特征只有：

```text
article_count
is_core_attr
parent_count
child_count
degree
```

这组特征能说明节点静态强度，但不能表达属性知识图谱中的上下位语义传播，也不能表达同一父属性下子属性之间的竞争状态。

本轮新增特征聚焦三类信息：

1. 父子层级上下文：刻画直接父属性和直接子属性的热度、份额和近期增长状态。
2. 同级竞争关系：刻画同父属性下 sibling 属性相对于当前属性的竞争状态。
3. 轻量结构强度：补充加权度、边权强度和根/叶子/sibling 标记。

实验比较五个组级消融版本：

```text
no_graph
current_coarse_graph
full_enhanced
wo_hierarchy_context
wo_sibling_competition
```

所有版本保持同一 resolved LightGBM 参数、同一 train/valid/test 时间切分和同一评价口径。差异只来自 feature mask。

主比较指标为 valid/test 的：

```text
NDCG@10
Spearman
Precision@10
```

MAE 和 RMSE 保留为辅助诊断，不作为本实验的主要结论依据。

## 架构边界

新增入口：

```text
src/19_run_trend_graph_feature_ablation.py
```

该脚本只负责 CLI、日志和阶段编排，不承载核心实验逻辑。

实验专用模块放在：

```text
src/experiments/trend_graph_feature_ablation/
```

建议模块职责：

| 模块 | 职责 |
| --- | --- |
| `paths.py` | 实验根目录、features 目录、runs 目录和 summary/doc 路径 |
| `contracts.py` | 版本名、固定文件名、主键列、目标列、指标列、schema version |
| `build_features.py` | 构建增强样本，写出 enhanced samples、row alignment、input hashes |
| `feature_groups.py` | 定义特征组、feature schema 和五个组级 feature mask |
| `train_runs.py` | 使用固定参数和 feature mask 训练五个 LightGBM 实验 run |
| `evaluate.py` | 按趋势评价口径计算每个 run 的 valid/test 指标 |
| `summarize.py` | 生成 `metrics_summary.csv` 和 `metrics_summary.md` |
| `write_docs.py` | 生成 `experiment.md` 和实验 `manifest.json` |

边界约束：

- 实验模块不放入 `fashion_trend.trend` 默认训练契约。
- 实验模块不调用会写入 `outputs/models/lightgbm/` 的默认训练 runner。
- 实验模块不调用会写入 `outputs/metrics/lightgbm/` 的默认评价 runner。
- 实验模块可以复用已有 reader、路径常量、LightGBM 参数解析、预测标准列和趋势评价指标计算等稳定工具，但输出路径必须由实验模块显式控制。
- `src/19_run_trend_graph_feature_ablation.py` 不应被 README 的默认全链路命令列为必须步骤，只能列在独立实验说明中。

实验 schema version 固定为：

```text
trend_graph_feature_ablation.v1
```

## 输入数据

实验读取默认主流程已经生成的稳定输入：

```text
data/processed/features/trend_model_samples.parquet
data/processed/features/trend_model_samples_train.parquet
data/processed/features/trend_model_samples_valid.parquet
data/processed/features/trend_model_samples_test.parquet
data/processed/graph/nodes_attribute.csv
data/processed/graph/edges_attribute_hierarchy.csv
```

输入缺失时直接失败，不做 fallback，不重建上游数据。

增强特征只读取当前周 `t` 及以前在默认趋势样本中已经存在的字段，例如：

```text
heat_t
share_t
log_heat_t
rank_in_type_t
growth_lag_1
growth_lag_2
acc_lag_1
heat_ma_4
share_ma_4
history_total_heat_t
history_active_weeks_t
```

增强特征不得读取、聚合或派生任何目标周 `t+1` 信息，包括：

```text
target_growth
target_log_heat_t1
target_rank_in_type_t1
heat_t1
share_t1
```

默认样本中的目标列只用于 row alignment 校验和训练标签，不参与图谱上下文特征构造。

## 图关系口径

图关系只使用属性层级图中的直接一跳关系：

```text
edges_attribute_hierarchy.csv
```

一个属性节点存在多个父节点或多个子节点时，全部保留，不强行简化为树结构。

父子动态上下文聚合规则：

- 使用直接父节点和直接子节点。
- 按 `edge_weight` 归一化加权。
- 多父、多子全部参与聚合。
- 缺父节点时父上下文字段补 0。
- 缺子节点时子上下文字段补 0。

结构强度规则：

- 使用原始 `edge_weight` 求和表示加权连接强度。
- 同时提供 `log1p(edge_weight_sum)` 作为稳定数值尺度。
- 不采用最大父节点简化。
- 不采用等权聚合。

sibling 规则：

- sibling 是与当前节点共享至少一个直接父节点的其他子节点。
- 多父节点时，所有父节点贡献都保留。
- sibling 加权可由父边权和 sibling 边权组合得到，并归一化为加权均值。
- 没有 sibling 时 sibling 动态字段补 0，并通过 `kg_has_sibling` 标记解释。

## 增强特征组

增强样本以默认 all/train/valid/test 样本为行基准，新增 `kg_*` 特征。行数、排序、主键和目标列必须与默认样本严格一致。

特征组固定为：

```text
base_numeric_non_graph
categorical
coarse_graph
hierarchy_context
sibling_competition
light_structure
```

### base_numeric_non_graph

默认 LightGBM 中的非图数值特征。该组排除所有粗图结构特征和新增 `kg_*` 特征。

示例字段：

```text
heat_t
share_t
log_heat_t
rank_in_type_t
heat_lag_1
heat_lag_2
heat_lag_3
heat_lag_4
share_lag_1
share_lag_2
share_lag_3
share_lag_4
growth_lag_1
growth_lag_2
acc_lag_1
heat_ma_4
share_ma_4
share_std_4
share_max_4
share_min_4
history_total_heat_t
history_active_weeks_t
is_trend_eligible_t
week_index
week_mod_52
```

### categorical

默认 LightGBM 分类特征：

```text
attr_type
```

该组在所有消融版本中保留，并使用 train split 固化 category levels。valid/test 出现 train 未见过的 `attr_type` 时必须失败。

### coarse_graph

当前粗粒度图结构版本：

```text
article_count
is_core_attr
parent_count
child_count
degree
```

`current_coarse_graph` 等价于默认当前粗图结构版本，不包含任何新增图谱上下文特征。

### hierarchy_context

父子层级动态上下文特征。该组只表达动态父子上下文，不包含根/叶子、has parent/child 或边权强度等静态结构标记。

建议字段：

```text
kg_parent_heat_t_wavg
kg_parent_share_t_wavg
kg_parent_growth_lag_1_wavg
kg_parent_rank_pct_t_wavg
kg_child_heat_t_wavg
kg_child_share_t_wavg
kg_child_growth_lag_1_wavg
kg_child_rank_pct_t_wavg
kg_self_parent_share_gap_t
kg_self_parent_growth_gap_lag_1
kg_self_child_share_gap_t
kg_self_child_growth_gap_lag_1
```

排名类聚合统一使用 `rank_pct` 口径，不直接聚合原始 rank。建议口径为同一 `week_id + attr_type` 内：

```text
rank_pct_t = (rank_in_type_t - 1) / max(type_attr_count - 1, 1)
```

这样数值越小代表排名越靠前，且不同属性类型之间更可比。

### sibling_competition

同父属性下的 sibling 竞争特征。该组表达当前属性相对于同父其他子属性的竞争状态。

建议字段：

```text
kg_sibling_count
kg_sibling_share_t_wavg
kg_sibling_share_t_max
kg_sibling_growth_lag_1_wavg
kg_sibling_rank_pct_t_wavg
kg_self_vs_sibling_share_gap_t
kg_self_vs_sibling_growth_gap_lag_1
kg_has_sibling
```

无 sibling 时动态字段补 0，`kg_has_sibling = 0`。

### light_structure

少量静态结构强度和节点角色标记。该组用于补充结构强度，不承担动态父子上下文含义。

建议字段：

```text
kg_parent_edge_weight_sum
kg_child_edge_weight_sum
kg_parent_edge_weight_log
kg_child_edge_weight_log
kg_has_parent
kg_has_child
kg_is_root_attr
kg_is_leaf_attr
```

其中：

```text
kg_parent_edge_weight_log = log1p(kg_parent_edge_weight_sum)
kg_child_edge_weight_log = log1p(kg_child_edge_weight_sum)
kg_is_root_attr = 1 - kg_has_parent
kg_is_leaf_attr = 1 - kg_has_child
```

## 组级消融版本

五个版本固定为：

| 版本 | 特征组 |
| --- | --- |
| `no_graph` | `base_numeric_non_graph + categorical` |
| `current_coarse_graph` | `base_numeric_non_graph + categorical + coarse_graph` |
| `full_enhanced` | `base_numeric_non_graph + categorical + coarse_graph + hierarchy_context + sibling_competition + light_structure` |
| `wo_hierarchy_context` | `full_enhanced - hierarchy_context` |
| `wo_sibling_competition` | `full_enhanced - sibling_competition` |

`wo_hierarchy_context` 只表示去除动态父子层级上下文，不表示去除所有层级结构标记。因此 `light_structure` 仍然保留。

本轮不做单项特征级消融，不扩展到父热度、子聚合、同级排名、同级均值、加权度等逐项消融。

## 输出契约

实验输出目录：

```text
outputs/experiments/trend_graph_feature_ablation/
```

完整结构：

```text
outputs/experiments/trend_graph_feature_ablation/
  experiment.md
  manifest.json
  metrics_summary.csv
  metrics_summary.md
  features/
    enhanced_samples_all.parquet
    enhanced_samples_train.parquet
    enhanced_samples_valid.parquet
    enhanced_samples_test.parquet
    feature_groups.json
    feature_schema.json
    row_alignment_check.json
    input_hashes.json
  runs/
    no_graph/
      predictions.csv
      metrics.json
      feature_importance.csv
      metadata.json
      params.json
      model.txt
    current_coarse_graph/
      predictions.csv
      metrics.json
      feature_importance.csv
      metadata.json
      params.json
      model.txt
    full_enhanced/
      predictions.csv
      metrics.json
      feature_importance.csv
      metadata.json
      params.json
      model.txt
    wo_hierarchy_context/
      predictions.csv
      metrics.json
      feature_importance.csv
      metadata.json
      params.json
      model.txt
    wo_sibling_competition/
      predictions.csv
      metrics.json
      feature_importance.csv
      metadata.json
      params.json
      model.txt
```

### Enhanced Samples

`enhanced_samples_*.parquet` 包含默认样本原列和新增 `kg_*` 特征。

写出前必须校验：

- all/train/valid/test 行数与默认样本一致。
- all/train/valid/test 排序与默认样本一致。
- split 样本的 `split + week_id + attr_id` 主键与默认 split 样本严格一致。
- all 样本的 `week_id + attr_id` 主键与默认 all 样本严格一致。
- 目标列 `target_growth`、`target_log_heat_t1`、`target_rank_in_type_t1` 与默认样本逐行一致。
- 新增数值特征无缺失且全部为有限值。
- 新增特征名统一使用 `kg_` 前缀。

### feature_groups.json

记录：

- 六个 feature group 的字段清单。
- 五个 ablation variant 的最终 feature mask。
- 每个 mask 的 numeric/categorical 字段。
- schema version。
- 生成时间和输入 hash 摘要。

校验规则：

- 每个 variant 至少包含 `base_numeric_non_graph` 和 `categorical`。
- 每个 variant 不包含目标列、标识列或 split 列。
- 每个 variant 不包含未知字段。
- `current_coarse_graph` 不包含新增 `kg_*` 特征。
- `no_graph` 不包含 `coarse_graph` 或新增 `kg_*` 特征。

### feature_schema.json

记录每个新增 `kg_*` 特征的：

- 字段名。
- 所属 feature group。
- dtype。
- 是否动态特征。
- 是否使用目标信息，必须全部为 `false`。
- 缺失父/子/sibling 时的补 0 规则。
- 聚合口径，例如 edge-weight normalized weighted average。

### row_alignment_check.json

记录 all/train/valid/test 的：

- 输入行数。
- 输出行数。
- 主键 checksum。
- 目标列 checksum。
- 排序是否一致。
- 是否通过严格对齐。

checksum 使用 SHA-256，对当前 DataFrame 顺序下的主键列和目标列进行稳定序列化后计算。该 checksum 用于发现行顺序、主键或目标值漂移。

### input_hashes.json

记录输入 artifact 的：

- path。
- mtime。
- size。
- hash。
- row count。

文件 hash 使用 SHA-256。Parquet 和 CSV 均按文件字节计算 hash，同时记录 row count 作为人类审查字段。

覆盖输入：

```text
trend_model_samples.parquet
trend_model_samples_train.parquet
trend_model_samples_valid.parquet
trend_model_samples_test.parquet
nodes_attribute.csv
edges_attribute_hierarchy.csv
```

### Run Artifacts

每个 run 目录保存：

| 文件 | 说明 |
| --- | --- |
| `predictions.csv` | 标准趋势预测列，覆盖 train/valid/test |
| `metrics.json` | valid/test 趋势评价指标 |
| `feature_importance.csv` | LightGBM split/gain importance |
| `metadata.json` | variant、feature mask、输入 hash、输出路径、训练摘要 |
| `params.json` | 实际 resolved LightGBM 参数和 early stopping 配置 |
| `model.txt` | LightGBM booster 文本模型 |

`params.json` 必须记录实际 resolved 参数，而不是只记录源码默认或用户输入片段。

### metrics_summary

`metrics_summary.csv` 和 `metrics_summary.md` 一行一个 variant，至少包含：

```text
variant
feature_count
valid_ndcg_at_10
valid_spearman
valid_precision_at_10
valid_mae
valid_rmse
test_ndcg_at_10
test_spearman
test_precision_at_10
test_mae
test_rmse
```

summary 同时展示 valid/test，避免只按 test 指标选择或解释结论。

## Runner 流程

`src/19_run_trend_graph_feature_ablation.py` 对外保持一条命令，内部按六个阶段执行：

1. `build_features`
2. `feature_groups`
3. `train_runs`
4. `evaluate`
5. `summarize`
6. `write_docs`

### build_features

读取默认 all/train/valid/test 样本和图 artifact。

构造步骤：

1. 从默认 all 样本计算 `rank_pct_t`。
2. 从 `edges_attribute_hierarchy.csv` 构建 parent、child 和 sibling 一跳邻接关系。
3. 在每个 `week_id` 内把邻居关系连接到对应属性节点的当前周特征。
4. 按 edge weight 归一化聚合父、子、sibling 动态特征。
5. 补齐无父、无子、无 sibling 的 0 值和结构标记。
6. 左连接回默认 all 样本。
7. 按默认 split 主键切出 enhanced train/valid/test。
8. 写出 enhanced samples、row alignment 和 input hashes。

### feature_groups

生成 feature groups、feature schema 和五个消融 feature mask。

该阶段必须明确区分：

- `base_numeric_non_graph`
- `categorical`
- `coarse_graph`
- `hierarchy_context`
- `sibling_competition`
- `light_structure`

### train_runs

按固定顺序训练：

```text
no_graph
current_coarse_graph
full_enhanced
wo_hierarchy_context
wo_sibling_competition
```

训练约束：

- 五个版本使用同一 resolved LightGBM 参数。
- 五个版本使用同一 train/valid/test split。
- 五个版本使用同一 early stopping 配置。
- 五个版本使用同一预测归一化口径。
- 差异只来自 feature mask。

实验训练不写 `outputs/models/lightgbm/`，不写 `outputs/models/lightgbm/runs/`。

### evaluate

对每个 run 的 `predictions.csv` 计算 valid/test 指标。

评价口径与现有趋势评价保持一致，但输出路径由实验模块控制，只写 `runs/<variant>/metrics.json`。

主指标：

```text
ndcg@10
spearman
precision@10
```

辅助指标：

```text
mae
rmse
```

### summarize

读取五个 run 的 `metrics.json`、`metadata.json` 和 `params.json`，生成 `metrics_summary.csv` 和 `metrics_summary.md`。

summary 不写入默认 reports，不进入 `outputs/reports/manifest.json`。

### write_docs

生成 `experiment.md` 和 `manifest.json`。

`experiment.md` 至少说明：

- 实验目的。
- 非 stable 边界。
- 输入 artifact。
- 特征组定义。
- 五个消融版本。
- 运行命令。
- 指标表。
- 论文使用注意事项。

`manifest.json` 记录：

- schema version。
- command。
- input hashes。
- output artifacts。
- variant list。
- feature groups path。
- metrics summary path。
- warnings。

## 错误处理

- 输入 artifact 缺失时直接失败。
- 输入字段缺失、主键重复、目标列不一致或 row alignment 失败时直接失败。
- 新增 `kg_*` 特征存在缺失值或非有限值时直接失败。
- feature mask 包含未知列、目标列、标识列或 split 列时直接失败。
- 任一 run 训练失败时整个脚本返回非零。
- 任一 run 评价失败时整个脚本返回非零。
- 失败时不生成伪完整 `metrics_summary.csv/md`。
- 输出目录存在时可以覆盖实验目录内同名产物，但不得删除或覆盖实验目录外文件。
- 写文件应使用原子写入或先写临时路径后替换，避免半成品被误读为完整结果。

## 测试计划

新增测试文件建议：

```text
tests/test_trend_graph_feature_ablation_features.py
tests/test_trend_graph_feature_ablation_runner.py
```

测试覆盖：

- 父节点加权均值支持多父节点。
- 子节点加权均值支持多子节点。
- 缺父节点补 0，并设置 `kg_has_parent = 0`、`kg_is_root_attr = 1`。
- 缺子节点补 0，并设置 `kg_has_child = 0`、`kg_is_leaf_attr = 1`。
- 无 sibling 时补 0，并设置 `kg_has_sibling = 0`。
- sibling 聚合排除当前节点自身。
- rank 聚合使用 `rank_pct`，不直接聚合原始 rank。
- `row_alignment_check.json` 能发现行数变化、排序变化、主键变化和目标列变化。
- `feature_groups.json` 中五个 variant 的 mask 与设计一致。
- `no_graph` 不包含粗图结构和 `kg_*` 特征。
- `current_coarse_graph` 不包含新增 `kg_*` 特征。
- `wo_hierarchy_context` 只移除 `hierarchy_context`，保留 `light_structure`。
- 训练阶段使用同一 resolved params。
- 训练阶段不写 `outputs/models/lightgbm/`。
- 评价和 summary 只写实验目录。
- `src/19_run_trend_graph_feature_ablation.py` 任一阶段失败时返回非零。

聚焦验证命令：

```sh
uv run pytest tests/test_trend_graph_feature_ablation_*.py tests/test_architecture_boundaries.py
uv run python -m compileall -q src
```

真实 artifact 可用时的集成验收命令：

```sh
uv run python src/19_run_trend_graph_feature_ablation.py
```

集成验收检查：

- `outputs/experiments/trend_graph_feature_ablation/features/enhanced_samples_all.parquet` 存在。
- all/train/valid/test enhanced samples 全部存在。
- `feature_groups.json`、`feature_schema.json`、`row_alignment_check.json`、`input_hashes.json` 全部存在。
- 五个 `runs/<variant>/` 目录全部存在。
- 每个 run 都包含 `predictions.csv`、`metrics.json`、`feature_importance.csv`、`metadata.json`、`params.json`、`model.txt`。
- `metrics_summary.csv` 和 `metrics_summary.md` 存在。
- `metrics_summary` 同时包含 valid/test 的 `ndcg@10`、`spearman`、`precision@10`。
- `outputs/models/lightgbm/`、`outputs/reports/` 和 defense app 数据源没有被本实验写入。

## 文档同步

实现本设计时需要同步：

- `README.md`：只在独立实验章节说明 19 号入口，不把它加入默认主流程。
- `docs/gpt-research/project-status-summary.md`：如论文采用本实验结果，再补充独立实验口径和结果摘要。
- `docs/gpt-research/defense-stable-roadmap.md`：如该 roadmap 仍作为答辩收口依据，可将“父属性热度/同级均值缺口”更新为独立实验状态。

默认 reports 和 defense app 文档不需要更新，除非后续显式决定让实验结果进入非默认 reports experimental 导出。

## 验收标准

本设计完成后的验收标准：

- 默认主流程 `00-18` 行为不变。
- 默认 `trend_model_samples*.parquet` schema 不变。
- 默认 stable LightGBM 输出不变。
- 默认 reports manifest、figures、tables 不变。
- defense app 数据源不变。
- 独立实验目录完整记录增强样本、特征组、input hashes、row alignment、五个 run、summary 和实验说明。
- 五个组级消融版本使用同一参数、同一 split、同一评价口径。
- 指标比较以 valid/test 的 NDCG@10、Spearman、Precision@10 为主。
- 实验结果可用于论文中的图谱特征贡献分析，但在正式采用前不替代默认主模型结果。
