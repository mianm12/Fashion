# 推荐消融实验补强设计

## 背景

当前项目已经完成属性周级趋势预测、LightGBM 主模型、轻量 Top-N 推荐实验、论文素材导出和本地答辩展示应用。推荐实验当前已有 5 个方法在 valid/test 上的对比：

- `global_popularity`
- `recent_popularity`
- `attribute_similarity`
- `pop_similarity`
- `pop_similarity_trend`

这些结果已经能说明趋势感知主方法相对不含趋势分的融合 baseline 有增益，但现有 `outputs/recommendation/experiments/main/experiment.json` 仍缺少更适合论文表述的命名消融行，尤其是严格的 `w/o Recent`。`outputs/reports/manifest.json` 当前也会记录“缺少严格 w/o Recent 消融行”的 warning。

本设计补强推荐层消融实验。目标是提供可复现、可审计、可直接进入论文表格的推荐消融 artifact，而不是扩大到新的趋势模型训练阶段。

## 已确认决策

- 采用“方案 2：推荐层完整补强，有界执行”。
- 本轮只做推荐层消融，不重训 LightGBM，不新增趋势模型特征消融。
- 保留现有 `pop_similarity_trend` 作为 Full Model。
- 保留 `pop_similarity` 作为 `w/o Trend in Rec` 的稳定口径。
- 新增 `w/o Similarity` 与 `w/o Recent` 的命名消融，并为每个版本记录 valid/test 指标。
- 补充少量代表性 `trend_score` 权重组合的 valid/test 指标。
- 实验变体只写入 `outputs/recommendation/experiments/main/experiment.json`，不注册为正式推荐 method，也不写入 `outputs/recommendation/<method>/` stable 目录。

## 与原实施方案消融项的关系

`docs/gpt-research/implementation-plan.md` 曾列出以下消融项。本轮只覆盖其中推荐阶段相关部分：

| 原计划项 | 本轮状态 | 说明 |
| --- | --- | --- |
| `Full Model` | 实现 | 对应 `pop_similarity_trend` 的 best weights。 |
| `w/o Trend in Rec` | 实现 | 对应推荐阶段去掉趋势分，稳定口径为 `pop_similarity`。 |
| `w/o Graph` | 不实现 | 属于趋势模型特征消融，需要重训 LightGBM 变体。 |
| `w/o Growth` | 不实现 | 属于趋势模型特征消融，需要按特征组训练和评价。 |
| `w/o Rank` | 不实现 | 属于趋势模型特征消融，影响 `rank_in_type_t` 等趋势样本特征。 |

`w/o Graph`、`w/o Growth`、`w/o Rank` 可以作为后续独立趋势模型消融设计处理。它们不能在本轮推荐消融中被描述为已实现。

## 目标

- 在 `experiment.json` 中新增可直接写论文的命名推荐消融结果。
- 为 `w/o Similarity`、`w/o Recent` 和代表性 `trend_score` 权重补齐 valid/test 指标。
- 让 reports 导出的 `recommendation_experiment_summary.{csv,md}` 能覆盖命名消融和权重分析。
- 移除 reports manifest 中“缺少严格 w/o Recent 消融行”的 blocker warning。
- 保持推荐实验边界：不污染 method registry，不新增 stable method 输出目录，不重跑趋势训练。

## 非目标

- 不实现 `w/o Graph`、`w/o Growth`、`w/o Rank` 等趋势模型特征消融。
- 不新增父属性热度、同级属性均值、图片特征、深度推荐或在线推荐服务。
- 不改变 `pop_similarity_trend` 的 stable method 输出语义。
- 不把 valid-only grid search 伪装成 test grid search。
- 不提交 `outputs/recommendation/`、`outputs/reports/` 或其他运行时产物。

## 推荐实验架构

现有入口保持不变：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main
```

实验编排仍由 `fashion_trend.recommendation.experiments.runner` 承担。新增逻辑应放在 `experiments` 层，复用已有 ranking feature cache、candidate、evaluation labels 和 recommendable pool：

```text
src/16_run_recommendation_experiment.py
    -> recommendation.experiments.runner
        -> run baseline methods
        -> valid grid search
        -> publish full pop_similarity_trend method
        -> build named ablation
        -> build trend weight analysis
        -> write experiment.json
```

核心原则：

- `Full Model` 是正式方法输出，继续发布到 `outputs/recommendation/pop_similarity_trend/`。
- 其他命名消融变体只在实验内构建和评价，不发布为正式 method。
- 所有实验变体必须使用相同时间窗口、目标用户、候选集、特征缓存和评价标签。
- valid 用于调参和选择 best weights；test 只用于最终评价，不反向参与权重选择。

## 命名消融设计

以当前 best full weights 为基准：

```text
Full Model: pop=0.2, sim=0.2, trend=0.1, recent=0.5
```

命名消融固定为：

| 名称 | 口径 | 权重或方法 |
| --- | --- | --- |
| `Full Model` | 趋势感知融合主方法 | `pop_similarity_trend`, `pop=0.2, sim=0.2, trend=0.1, recent=0.5` |
| `w/o Trend in Rec` | 推荐阶段去掉趋势分 | 读取已有 `pop_similarity` valid/test 指标 |
| `w/o Similarity` | 去掉用户属性相似分 | `pop=0.4, sim=0.0, trend=0.1, recent=0.5` |
| `w/o Recent` | 去掉近期热门分 | `pop=0.5, sim=0.3, trend=0.2, recent=0.0` |
| `Recent Only` | 强近期热门 baseline | 读取已有 `recent_popularity` valid/test 指标 |

`w/o Similarity` 使用当前 grid 中已有且靠前的 `sim_score=0` 代表组合。`w/o Recent` 当前 grid 中没有，需要作为固定命名消融新增评价。该权重保留 `sim_score` 和 `trend_score`，用于回答去掉近期热门后，属性相似与趋势分还能贡献多少。

`w/o Trend in Rec` 不再额外制造第二种“把 Full Model 的 trend 置零再重归一化”的口径。当前项目已经有稳定的 `pop_similarity` baseline，继续用它作为不含趋势分的主对照，可以减少论文解释歧义。

## trend weight 分析

补充代表性 trend 权重分析，覆盖：

```text
trend_score = 0.0 / 0.1 / 0.2 / 0.3 / 0.4
```

每个 `trend_score` 优先选用当前 grid 中该 trend 权重下 valid `NDCG@12` 最好的组合，并补齐 test 指标。当前 grid search 仍只记录 valid 指标，新增的 `trend_weight_analysis` 才记录代表组合的 valid/test 指标。

权重分析用于论文说明 trend weight 对推荐指标的影响，不参与 best weights 重新选择，不改变 Full Model 的 stable 权重。

## experiment.json 契约

保留现有字段：

- `experiment_id`
- `experiment_path`
- `best_weights`
- `search_results`
- `ablation`
- `stage_status`
- `force`
- `timings`

新增字段：

```json
{
  "named_ablation": [
    {
      "name": "Full Model",
      "method": "pop_similarity_trend",
      "weights": {
        "pop_score": 0.2,
        "sim_score": 0.2,
        "trend_score": 0.1,
        "recent_score": 0.5
      },
      "metrics": {
        "valid": {},
        "test": {}
      }
    }
  ],
  "trend_weight_analysis": [
    {
      "trend_score": 0.1,
      "weights": {
        "pop_score": 0.2,
        "sim_score": 0.2,
        "trend_score": 0.1,
        "recent_score": 0.5
      },
      "metrics": {
        "valid": {},
        "test": {}
      }
    }
  ]
}
```

约束：

- `named_ablation` 必须包含 5 个确认名称，且每个名称都必须有 valid/test 指标。
- `w/o Recent` 的 `recent_score` 必须为 `0.0`。
- `w/o Similarity` 的 `sim_score` 必须为 `0.0`。
- `trend_weight_analysis` 必须覆盖 5 个代表 `trend_score` 值。
- 所有指标必须是有限数值，不能输出空成功结果。
- `method` 可记录实际构建方法名，但 `name` 是论文展示用名称。

## reports 集成

reports 仍只读取稳定 artifact，不运行推荐实验。`src/fashion_trend/reports/runner.py` 需要扩展 `recommendation_experiment_summary` 的展开逻辑：

- `search_results` 继续作为 `section=search_results` 行，仍然只包含 valid 指标。
- `ablation` 继续作为 `section=ablation` 行，保留方法级对比。
- 新增 `named_ablation` 展开为 `section=named_ablation` 行。
- 新增 `trend_weight_analysis` 展开为 `section=trend_weight_analysis` 行。

表格列可以继续复用现有 `RECOMMENDATION_EXPERIMENT_SUMMARY_COLUMNS`：

```text
section, rank, method, split,
pop_score, sim_score, trend_score, recent_score,
map_at_12, recall_at_12, hit_rate_at_12, ndcg_at_12, coverage
```

其中 `named_ablation` 的 `method` 列写展示名称，例如 `Full Model` 或 `w/o Recent`。如果后续需要同时保留底层 method，可作为另一个设计补充新增列；本轮不扩表结构，避免影响既有 reports 表格契约过大。

manifest warning 规则：

- 如果 `named_ablation` 中存在 `w/o Recent` 且 valid/test 指标完整，不再输出“缺少严格 w/o Recent 消融行”。
- 如果 `search_results` 仍然只有 valid 指标，继续保留 valid-only grid search warning。这是调参语义提示，不是 blocker。

## 错误处理

必须显式失败的情况：

- `best_weights` 缺失或不是合法权重。
- 命名消融权重不是有限非负数，或权重和不是 1。
- `w/o Recent` 没有 `recent_score=0`。
- `w/o Similarity` 没有 `sim_score=0`。
- 任一命名消融缺少 valid 或 test 指标。
- `trend_weight_analysis` 没有覆盖 0.0、0.1、0.2、0.3、0.4。
- 实验变体读取了不匹配 split 的 target users、labels 或 candidates。
- 评价阶段没有有效用户，却试图输出空成功指标。

错误信息必须包含具体实验名称、缺失字段或非法权重，便于定位。

## force 与缓存语义

保持当前 force 开关语义：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
uv run python src/16_run_recommendation_experiment.py --experiment main --force-method pop_similarity
uv run python src/16_run_recommendation_experiment.py --experiment main --force-cache
uv run python src/16_run_recommendation_experiment.py --experiment main --force-candidates
uv run python src/16_run_recommendation_experiment.py --experiment main --force-rebuild-all
```

命名消融和 trend weight 分析是 `experiment.json` 的派生内容。只要 `run_recommendation_experiment()` 执行，就应重新计算并写入当前 payload。它们不需要新增独立 force 开关。

如果 ranking feature、候选构建或推荐输入契约发生变化，真实 artifact 验证应使用 `--force-rebuild-all`。如果只调整实验 payload 展开或 reports 读取逻辑，`--force-experiment` 足够。

## 测试设计

推荐实验测试：

- `tests/test_recommendation_experiments.py` 验证 `named_ablation` 包含固定 5 个名称。
- 验证每个命名消融都有 valid/test 指标。
- 验证 `w/o Recent` 的 `recent_score=0`。
- 验证 `w/o Similarity` 的 `sim_score=0`。
- 验证 `trend_weight_analysis` 覆盖 0.0、0.1、0.2、0.3、0.4，且每组有 valid/test。
- 验证这些变体不会进入 recommendation method registry。
- 验证 payload 构建不会写入新的 stable method 输出目录。

reports 测试：

- `tests/test_reports_runner.py` 验证新字段能展开到 `recommendation_experiment_summary`。
- 验证存在严格 `w/o Recent` 后不再产生 blocker warning。
- 验证 valid-only grid search warning 仍保留。
- `tests/test_reports_tables.py` 验证表格列顺序和排序仍稳定。

架构测试：

- `tests/test_architecture_boundaries.py` 确认 reports 不导入 recommendation experiment runner，仍只读取 experiment artifact。
- recommendation experiments 层不能反向依赖 reports、presentation 或 defense app。

## 验证命令

实现后的最小验证：

```sh
uv run pytest tests/test_recommendation_experiments.py tests/test_reports_runner.py tests/test_reports_tables.py tests/test_architecture_boundaries.py
uv run python -m compileall -q src
```

真实 artifact 验证：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
uv run python src/17_export_paper_assets.py
```

如果实现触及缓存、候选或 ranking feature 语义，真实重建升级为：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main --force-rebuild-all
uv run python src/17_export_paper_assets.py
```

真实验证后需要检查：

- `outputs/recommendation/experiments/main/experiment.json` 含 `named_ablation` 和 `trend_weight_analysis`。
- `named_ablation` 中 5 个版本都有 valid/test 指标。
- `trend_weight_analysis` 覆盖 5 个 trend 权重。
- `outputs/reports/tables/recommendation_experiment_summary.{csv,md}` 可直接支撑论文消融表。
- `outputs/reports/manifest.json` 不再包含“缺少严格 w/o Recent 消融行”。

## 文档同步

实现后需要同步：

- `README.md`：更新推荐实验和消融说明，说明 Full Model、`w/o Trend in Rec`、`w/o Similarity`、`w/o Recent` 和 trend weight 分析。
- `docs/gpt-research/implementation-plan.md`：明确本轮实现的是推荐层消融；`w/o Graph`、`w/o Growth`、`w/o Rank` 仍是趋势模型特征消融，不在本轮。
- `docs/gpt-research/project-status-summary.md` 或当前 roadmap：真实 artifact 重建后更新最新指标和缺口状态。

文档不得把本轮结果写成趋势模型特征消融已完成，也不得把推荐主方法描述为全面超过 `recent_popularity`。

## 验收标准

- 新设计只改变推荐实验 payload、reports 展开和相关文档，不新增正式推荐 method。
- `experiment.json` 有命名消融行，每个命名版本都有 valid/test 指标。
- `trend_weight_analysis` 有代表权重的 valid/test 指标。
- reports 表格能展开新字段并保持稳定列契约。
- manifest 不再报告缺少严格 `w/o Recent`。
- 聚焦测试、架构测试和编译检查通过。
- 真实 artifact 验证跑通后，论文可直接引用推荐消融表。
