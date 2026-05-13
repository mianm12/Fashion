# 答辩稳妥版差距分析与推进方案

更新时间：2026-05-13

## 1. 总体判断

当前项目已经不是“主线最小闭环待实现”状态，而是已经进入答辩收口阶段。`docs/gpt-research/implementation-plan.md` 中定义的主线已经跑通到趋势预测、轻量 Top-N 推荐、论文素材导出和本地答辩展示应用：

```text
H&M 原始数据
  -> 商品属性层次图
  -> 属性周热度
  -> LightGBM 趋势预测
  -> 趋势感知 Top-N 推荐
  -> reports 论文素材
  -> SQLite + FastAPI + Vue 本地答辩展示
```

下一步不应继续扩大模型范围。答辩稳妥版的核心目标应收敛为：

1. 锁定可复现的稳定 artifact。
2. 消除或明确记录当前仍影响论文说服力的证据缺口。
3. 保持论文、展示应用和真实实现的口径一致。
4. 准备一条现场可控、失败风险低的演示路径。

结论：建议采用“最小补强 + 全链路验收 + 答辩脚本固化”的路线，而不是继续引入复杂推荐模型、深度学习、图数据库或在线服务。

## 2. 与答辩稳妥版设计的对比

| `implementation-plan.md` 中的答辩稳妥版要求 | 当前状态 | 证据与说明 | 推进判断 |
| --- | --- | --- | --- |
| 完整属性层级边 | 已实现 | `data/processed/graph/edges_attribute_hierarchy.csv` 已作为图 artifact；reports 和 defense app 均可消费属性图。 | 已满足，保持当前口径。 |
| 父属性热度特征、同级属性均值特征 | 部分满足 | 当前趋势样本使用 `parent_count`、`child_count`、`degree`、`article_count`、`is_core_attr` 等图结构特征，并未实现父节点热度或同级热度均值。 | 不建议临近答辩强行扩模型。论文中应写成“图结构特征”，不要写成“父属性热度/同级热度特征”。如必须写这两类特征，应作为单独小任务实现和重评估。 |
| LightGBM 特征重要性 | 已实现 | `outputs/models/lightgbm/feature_importance.csv` 和 reports 图表 `lightgbm_feature_importance.{svg,png}` 已存在。 | 已满足。 |
| 按属性类型分组评价 | 已实现 | reports 表格包含 `trend_metrics_by_attr_type.{csv,md}`；现状文档已记录各属性类型评价口径。 | 已满足。 |
| 推荐消融实验 | 已满足推荐层稳妥版 | `outputs/recommendation/experiments/main/experiment.json` 已包含 `named_ablation` 和 `trend_bucket_best_by_valid`；reports 已展开为 `recommendation_experiment_summary`，manifest 不再提示缺少严格 `w/o Recent`。 | 推荐层命名消融已补齐。后续如继续做，应只考虑趋势模型特征消融，不再扩推荐模型。 |
| 趋势曲线可视化 | 已实现 | reports 已导出 `trend_curve_examples.{svg,png}` 和 `topk_trend_attributes.{svg,png}`；defense app 有趋势看板和属性详情页。 | 已满足。 |
| 用户推荐解释 | 已实现 | reports 已导出 3 个案例；SQLite 展示库当前有 50 个 demo users 和 600 条 Top-12 推荐项。 | 已满足。答辩时使用预置 demo 用户，不做任意用户在线推荐承诺。 |
| 趋势看板 / 属性详情 / 属性图展示 / 推荐展示 / 推荐理由页面 | 已实现 | `apps/defense_app/frontend/` 已包含五类视图，后端只读 SQLite，前端只调用 `/api`。 | 已满足，后续重点是浏览器走查和演示脚本。 |
| 商品图片展示 | 未作为主线实现 | 原计划把商品图片列为有余力增强，且只用于网页展示。当前推荐解释以商品名、属性和分数组件为主。 | 低优先级。除非答辩视觉效果明显不足，否则不要为了图片引入大体积资产和路径风险。 |

## 3. 推荐推进策略

### 方案 A：当前基础上只做文档冻结

只更新论文表述、整理现有图表和演示页面，不再继续扩代码或补新实验。

优点是最快；风险是展示库、论文截图和最终报告素材如果未重新验收，可能与最新推荐消融 artifact 不完全同步。

### 方案 B：最小补强后冻结，已采用

只补影响答辩说服力的最小缺口：严格推荐消融、最终 artifact 重建、全链路验证、演示脚本。其他模型增强全部不做。

这是当前已采用的稳妥路线。推荐层严格消融和 reports 已补强，后续重点是展示库同步、浏览器走查和答辩脚本固化。

### 方案 C：继续扩模型

继续实现父属性热度、同级热度、按属性类型单独训练、图片展示或更强推荐召回。

不推荐作为当前主线。除非论文已经明确承诺这些内容，否则临近答辩继续扩模型会增加重跑、调参、文档漂移和现场演示风险。

## 4. 推进任务

### P0：冻结答辩口径

目标：先把论文和答辩中的边界说清楚，避免后续所有材料出现过度表述。

应固定的表述：

- 本项目核心是属性级周趋势预测，不是复现 H&M Kaggle 高分推荐方案。
- 推荐模块是趋势预测结果的应用验证层，不是生产推荐系统。
- defense app 是只读本地展示应用，不训练模型、不重跑推荐、不提供在线推荐服务。
- 当前 LightGBM 使用的是历史热度、增长率、图结构、历史活跃度和时间特征；不要写成已经使用父属性热度或同级属性均值。

验收标准：

- `README.md`、`docs/gpt-research/project-status-summary.md`、论文正文和答辩 PPT 口径一致。
- 没有“在线服务”“生产推荐平台”“深度推荐模型”“全面超过强 baseline”这类不准确表述。

### P1：严格推荐消融，已完成

目标：把推荐实验从“已有方法对比”补强成更适合论文答辩的命名消融。

当前已实现：

- 保留现有 `Full Model = pop_similarity_trend`。
- 严格 `w/o Trend in Rec`、严格 `w/o Similarity`、严格 `w/o Recent` 均从当前 `best_weights` 动态 drop-and-renormalize，并记录 valid/test 指标。
- `Recent Only` 和 `Pop + Similarity baseline` 作为稳定 method baseline 展示，`selection_split=not_applicable`。
- `trend_bucket_best_by_valid` 记录 `trend_score=0.0/0.1/0.2/0.3/0.4` 下按 valid NDCG@12 选择的代表组合及 valid/test 指标；它不是固定其他权重的单因素 sweep。

验收结果：

- `outputs/recommendation/experiments/main/experiment.json` 中有命名消融行，且每个命名版本都有 valid/test 指标。
- reports 导出的 `recommendation_experiment_summary.{csv,md}` 可直接支撑论文消融表。
- `outputs/reports/manifest.json` 不再出现“缺少严格 w/o Recent 消融行”这类 blocker warning；仍保留 grid valid-only warning，论文中应只把 grid search 写作 valid 调参依据。

### P2：重建最终稳定 artifact

目标：在补强后重新生成论文素材和展示库，确保所有下游结果来自同一批稳定 artifact。

已验证命令：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
uv run python src/17_export_paper_assets.py
```

展示库如需在答辩前同步重建，再运行：

```sh
uv run python src/18_build_defense_app_db.py
sqlite3 outputs/defense_app/fashion_demo.sqlite "pragma integrity_check;"
```

如推荐候选、feature cache 或权重网格配置发生变化，再使用更强的推荐重建开关：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main --force-rebuild-all
```

验收标准：

- reports 至少包含 16 个 figure 文件、16 个 table 文件、3 个 case study 和 `manifest.json`。
- SQLite 展示库完整性检查返回 `ok`。
- 展示库 demo users 在 20-50 个之间，且推荐项数量等于 `demo_users * 12`。
- `app_metadata.source_artifacts` 记录的 source artifact 指向最终重建后的稳定产物。

### P3：全链路验证

目标：确认“代码、artifact、API、前端构建、展示库”都处在可交付状态。

建议命令：

```sh
uv run pytest tests/test_presentation_*.py tests/test_architecture_boundaries.py
uv run --group app pytest apps/defense_app/backend/tests
uv run python -m compileall -q src
uv run --group app python -m compileall -q apps/defense_app/backend
cd apps/defense_app/frontend
npm run typecheck
npm run build
```

验收标准：

- presentation、architecture、backend API 测试通过。
- Python 编译检查通过。
- 前端 typecheck 和 build 通过。
- 如 Vite 仅出现 chunk size warning，可记录为非阻塞；如出现类型错误、API 字段不匹配或页面构建失败，必须先修复。

### P4：浏览器视觉 QA 与演示脚本

目标：把“可以运行”推进到“现场可控展示”。

建议固定一条演示路径：

1. 趋势看板：展示四类核心属性趋势榜，说明趋势预测任务。
2. 属性详情：打开一个代表性属性，展示 8 周热度曲线和预测值。
3. 属性图展示：从属性或商品进入商品-属性连接图，说明属性图设计。
4. 推荐展示：选择预置 demo user，展示 Top-12 推荐。
5. 推荐理由：点开一个推荐商品，展示用户画像、商品属性、趋势匹配和 score breakdown。

验收标准：

- 后端 `/docs` 可打开。
- 前端五个路由均可在 1440x900 或更宽桌面视口完整展示。
- 不依赖现场输入随机 `customer_id`。
- 准备一组固定 demo case 和备用截图，防止现场网络、依赖或浏览器状态干扰。

### P5：最终文档同步

目标：让项目文档、论文素材和答辩材料都指向同一事实。

需要同步的文件：

- `docs/gpt-research/project-status-summary.md`
- `docs/gpt-research/implementation-plan.md`
- `README.md`
- `apps/defense_app/README.md`
- 论文正文和答辩 PPT 中的实验表、图表和案例截图

验收标准：

- 所有指标取自最终重建后的 `outputs/metrics/`、`outputs/recommendation/` 和 `outputs/reports/`。
- 论文只引用已存在、可复现、可解释的图表和表格。
- 文档明确记录仍未做的内容，例如深度推荐、图片特征、父属性热度、同级热度或生产化部署。

## 5. Go / No-Go 门禁

可以进入答辩稳妥版的条件：

- 趋势预测主结论稳定：LightGBM 在 valid/test 上显著优于简单趋势 baseline。
- 推荐结论克制：趋势分显著优于不含趋势分的融合模型，但不夸大为全面击败 `recent_popularity`。
- reports、SQLite、后端、前端全部可重建或可验证。
- 演示路径依赖预置 demo users，不依赖随机用户输入。
- 所有文档和答辩材料都承认推荐指标绝对值偏低、近期热门 baseline 很强、项目不是生产推荐系统。

必须暂缓的情况：

- 最终 artifact 和文档指标不一致。
- `recommendation_items.parquet`、`evaluation_labels.parquet` 或 `user_profile.parquet` 无法支撑推荐解释。
- defense app 需要现场重跑长耗时推荐实验才能展示。
- 前端页面、后端 API 或 SQLite schema 存在未解释的构建/测试失败。
- 论文或 PPT 写了当前实现没有支持的能力。

## 6. 推荐执行顺序

建议按下面顺序推进：

1. P1 已完成，推荐层严格命名消融已固化。
2. 继续做 P2 和 P3，按需重建展示库并验证最终 artifact。
3. 然后做 P4，形成演示路径、固定 demo case 和备用截图。
4. 最后做 P5，同步论文、README、项目现状文档和答辩材料。

如果时间不足，最低可接受路线是使用当前已生成 reports 和推荐消融表，但必须把推荐主结论收缩为“趋势分相对不含趋势分的融合模型有增益，接近但未全面超过强近期热门 baseline”。
