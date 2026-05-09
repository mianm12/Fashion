# 推荐系统设计

## 范围

本设计覆盖趋势预测完成后的轻量 Top-N 推荐系统。目标是把已完成的趋势预测结果用于推荐重排序，并通过离线实验说明趋势分对推荐结果的贡献。

当前项目已经完成到：

```text
周级交易表
商品属性层次图
属性周热度
趋势标签和样本
三类 baseline
LightGBM 主模型
趋势评价
```

推荐系统从这些稳定产物继续向下游推进：

```text
weekly transactions
+ article-attribute edges
+ clean articles
+ stable LightGBM predictions
    -> 推荐时间窗口
    -> 用户属性画像
    -> 候选召回
    -> 推荐重排序
    -> 单方法评价
    -> 权重搜索、test 固定评价和消融实验
```

本轮不实现深度推荐模型、双塔模型、LightGCN、外部向量库、在线服务、Streamlit 展示或新的运行时依赖。推荐模块是论文中的应用验证层，不把推荐系统本身做成新的主任务。

## 设计结论

采用“方法层 + 能力层 + runner 层”的结构。推荐方法通过 registry 暴露，底层能力按 retrieval、ranking、evaluation、experiments 分层，避免把候选召回、权重搜索和消融实验堆进几个平铺大文件。

推荐包目标结构：

```text
src/fashion_trend/recommendation/
  contracts.py
  paths.py
  readers.py
  registry.py
  outputs.py
  time_windows.py

  methods/
    base.py
    baselines/
      global_popularity.py
      recent_popularity.py
      attribute_similarity.py
      pop_similarity.py
    trend_aware/
      pop_similarity_trend.py

  retrieval/
    popularity.py
    attributes.py
    trend.py
    candidates.py

  ranking/
    features.py
    scoring.py
    weights.py

  evaluation/
    metrics.py
    payloads.py
    runner.py

  experiments/
    grid_search.py
    ablation.py
    runner.py
```

推荐域不新增 `schema.py`。推荐 artifact 列契约、Top-K、method 名称、split 名称、strategy 名称、ranking feature 名称、核心属性类型、trend score 属性权重和 metrics payload 必需字段统一放在 `contracts.py`。`paths.py` 只放路径，`readers.py` 只放读取和严格校验。

`recommendation.contracts` 固定推荐阶段使用的核心属性类型：

```text
RECOMMENDATION_CORE_ATTR_TYPES = (
  "product_type_name",
  "colour_group_name",
  "garment_group_name",
  "product_group_name",
  "graphical_appearance_name",
)
```

推荐模块不能导入 `fashion_trend.catalog.articles` 里的实现常量来判断核心属性类型。若后续需要把核心属性类型变成跨域公开契约，应提升到 `catalog.contracts`，再通过架构测试白名单显式允许。

## 架构边界

`recommendation` 可以只读使用这些上游公开接口：

```text
fashion_trend.transactions.contracts
fashion_trend.transactions.readers
fashion_trend.catalog.contracts
fashion_trend.catalog.readers
fashion_trend.trend.schema
fashion_trend.trend.predictions
fashion_trend.trend.readers
```

`recommendation` 不应依赖：

```text
fashion_trend.trend.training
fashion_trend.trend.evaluation.runner
fashion_trend.trend.models
fashion_trend.catalog.graph.builders
```

趋势模型训练、趋势评价和属性图构建是上游阶段，推荐系统只消费它们的公开产物和 reader。编号脚本仍然是薄编排层，业务事实进入 `src/fashion_trend/recommendation/`。

## 时间窗口

推荐系统必须显式区分 `cutoff_week` 和 `label_week`。这套逻辑集中放在 `recommendation/time_windows.py`，不散落到 retrieval、ranking 或 evaluation 中。

核心语义：

```text
cutoff_week = t
label_week = t + 1
trend predictions use predictions.week_id == cutoff_week
recommendation ground truth uses transactions.week_id == label_week
```

趋势预测表中的 `week_id` 表示“用 week t 的特征预测 t+1”。因此为 `label_week` 生成推荐时，趋势分必须读取 `predictions.week_id == cutoff_week`，不能直接读取 `week_id == label_week`。

`time_windows.py` 负责：

- 从 stable trend predictions 或 split metadata 中生成 valid/test 推荐窗口。
- 校验 `cutoff_week < label_week`，默认要求 `label_week == cutoff_week + 1`。
- 输出包含 `split`、`cutoff_week`、`label_week` 的窗口表。
- 拒绝重复窗口、非法 split、缺失 valid/test 或窗口越界。

推荐阶段的 `split` 以 cutoff week 所属趋势预测 split 为准。评价真实购买只在 `label_week` 上读取交易。

## 防泄漏规则

所有推荐中间表和输出都必须携带：

```text
split
cutoff_week
label_week
```

规则：

- 用户画像只允许读取 `week_id <= cutoff_week` 的交易。
- 近期热门、属性相似、趋势候选和 ranking feature 都只允许读取 `week_id <= cutoff_week` 的历史事实。
- 推荐评价 ground truth 只允许读取 `week_id == label_week` 的交易。
- 趋势分只允许读取 `predictions.week_id == cutoff_week` 的预测。
- 用 `label_week` 选择评价用户是允许的，但不能把 `label_week` 的购买用于候选、画像或打分特征。
- 第一版只面向 eligible evaluation users 生成离线推荐：这些用户在 `label_week` 有真实购买，且 `cutoff_week` 前有历史记录。
- `12`、`13`、`14` 可以读取 `target_users.parquet` 中的用户集合，但不能读取 `evaluation_labels.parquet` 中的 `article_id` 作为候选、画像或打分特征。
- `15` 评价时以 `target_users.parquet` 为完整 eligible user set。缺少某个 eligible 用户的推荐结果时，默认按空推荐计 0 分；如果配置为 strict 模式，则直接失败。不能把“推荐结果中存在该用户”作为评价用户筛选条件。

## CLI 契约

新增入口：

```text
src/12_build_recommendation_inputs.py
src/13_build_recommend_candidates.py
src/14_rerank_recommendations.py
src/15_eval_recommendations.py
src/16_run_recommendation_experiment.py
```

`12_build_recommendation_inputs.py` 是推荐阶段的输入构建入口，生成推荐窗口、eligible target users、evaluation labels 和可复用用户属性画像。它写出：

```text
data/processed/recommend/time_windows.parquet
data/processed/recommend/target_users.parquet
data/processed/recommend/evaluation_labels.parquet
data/processed/recommend/user_profile.parquet
```

`user_profile.parquet` 服务 attribute similarity 类方法，但不是所有 method 的必经步骤。`global_popularity` 和 `recent_popularity` 不依赖用户画像。

`13_build_recommend_candidates.py` 使用候选 strategy，不使用推荐 method：

```sh
uv run python src/13_build_recommend_candidates.py --strategy default
uv run python src/13_build_recommend_candidates.py --strategy popularity
uv run python src/13_build_recommend_candidates.py --strategy similarity
uv run python src/13_build_recommend_candidates.py --strategy trend_union
```

`14_rerank_recommendations.py` 使用推荐 method：

```sh
uv run python src/14_rerank_recommendations.py --method recent_popularity
uv run python src/14_rerank_recommendations.py --method pop_similarity_trend
```

`15_eval_recommendations.py` 只评价单个 method 的推荐结果：

```sh
uv run python src/15_eval_recommendations.py --method pop_similarity_trend
```

`16_run_recommendation_experiment.py` 负责实验编排：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main
```

该入口运行 valid 权重搜索、固定权重 test 推荐、单方法评价和消融汇总。它不替代 `15_eval_recommendations.py`。原计划中的报告导出阶段顺延到后续编号；实现本设计时需要同步 README 和 implementation plan 的脚本编号。

## Artifact 契约

推荐中间产物：

```text
data/processed/recommend/time_windows.parquet
data/processed/recommend/target_users.parquet
data/processed/recommend/evaluation_labels.parquet
data/processed/recommend/user_profile.parquet
data/processed/recommend/candidates/<strategy>/candidate_items.parquet
```

`time_windows.parquet` 是推荐窗口事实表：

```text
split
cutoff_week
label_week
```

`target_users.parquet` 定义离线推荐要服务的 eligible 用户集合：

```text
split
cutoff_week
label_week
customer_id
history_purchase_count
label_purchase_count
```

`evaluation_labels.parquet` 只供评价阶段读取，不能作为候选或 ranking feature 输入：

```text
split
cutoff_week
label_week
customer_id
article_id
```

`evaluation_labels.parquet` 的唯一键是：

```text
split
cutoff_week
label_week
customer_id
article_id
```

H&M 同一用户在同一周可能多次购买同一商品。构建 `evaluation_labels.parquet` 时必须先按上述唯一键去重；MAP、Recall、NDCG 的 relevant set 也按去重后的 article 集合计算。reader 应拒绝已经写出的 label 表中存在重复唯一键，避免重复购买扭曲 Recall 分母和排名指标。

`user_profile.parquet` 使用长表：

```text
split
cutoff_week
label_week
customer_id
attr_id
attr_type
attr_value
preference_score
purchase_count
last_purchase_week
```

`candidate_items.parquet` 使用 strategy-scoped 路径，并在表内保留 strategy 字段：

```text
split
cutoff_week
label_week
strategy
customer_id
article_id
candidate_sources
primary_source
best_source_rank
```

`candidate_sources` 可以用稳定分隔文本表示，例如 `popularity|similarity|trend`。`primary_source` 是该候选的最高优先来源；`best_source_rank` 是候选在所有命中来源中的最小 rank，用于稳定审计和 tie-break。不要使用含义不明的单列 `source_rank`。如果后续需要每个来源的完整 rank，再扩展为单独明细表，不在第一版引入嵌套 parquet。

每个 method 的输出目录：

```text
outputs/recommendation/<method>/
  recommendations.csv
  recommendation_items.csv
  params.json
  metadata.json
  metrics.json
```

`recommendations.csv` 是最终短表：

```text
customer_id
split
cutoff_week
label_week
method
prediction
```

`prediction` 是空格分隔的 Top-12 `article_id` 字符串。每个 `customer_id + split + cutoff_week + label_week + method` 最多一行，推荐列表最多 12 个商品，且同一行内商品不能重复。

`recommendation_items.csv` 是解释和审计长表：

```text
customer_id
split
cutoff_week
label_week
method
article_id
rank
score
pop_score
sim_score
trend_score
recent_score
candidate_sources
```

`params.json` 记录：

- method 名称。
- candidate strategy。
- Top-K。
- score feature 列表。
- 权重。
- 是否过滤已购商品。
- 窗口配置。
- trend score 配置。

`metadata.json` 记录：

- 输入 artifact 路径。
- stable trend model 来源和预测文件路径。
- 生成时间。
- method、strategy、split、窗口摘要。
- 输入行数、候选行数、推荐用户数、推荐行数。
- 每个 split 的推荐用户数和平均候选数。

单 method 评价输出：

```text
outputs/recommendation/<method>/metrics.json
```

实验汇总输出：

```text
outputs/recommendation/experiments/<experiment_id>/experiment.json
```

实验 run 输出和 stable method 输出必须分离。`16_run_recommendation_experiment.py` 在 valid 权重搜索时不能反复覆盖：

```text
outputs/recommendation/<method>/
```

权重搜索候选可以只在内存中评价；如果需要落盘调试，必须写到 experiment-scoped run 目录：

```text
outputs/recommendation/experiments/<experiment_id>/
  experiment.json
  runs/
    <run_id>/
      recommendations.csv
      recommendation_items.csv
      params.json
      metadata.json
      metrics.json
```

`experiment_id` 必须是安全路径片段，不能为空，不能是 `.` 或 `..`，不能包含 `/` 或 `\`。`run_id` 由实验 runner 内部生成，格式采用 `YYYYMMDD-HHMMSS-<8hex>`；即使不接受用户输入，也必须经过同样的安全路径片段校验后才能创建目录。

method-scoped 目录只保存最终发布结果：baseline 的正式结果，以及 `pop_similarity_trend` 使用 valid 最佳权重后生成的结果。

## Method Registry

推荐方法通过 `recommendation/registry.py` 统一注册：

```text
global_popularity
recent_popularity
attribute_similarity
pop_similarity
pop_similarity_trend
```

`methods/base.py` 定义协议：

```text
RecommendationMethod
  name
  method_type
  default_candidate_strategy
  default_weights
  required_features
  build_recommendations(context) -> RecommendationResult
```

`methods/*` 只声明组合策略和调用底层能力，不重复实现具体打分逻辑。具体计算放在：

```text
retrieval/
ranking/
evaluation/
experiments/
```

未知 method 必须失败并列出可用 method。method 名称必须通过安全路径片段校验，不能逃逸 `outputs/recommendation/`。

## Baseline 方法

`global_popularity`：

- 使用 `week_id <= cutoff_week` 的累计销量排序。
- 同一窗口内的基础全局热门排序相同。
- 若 `exclude_seen=true`，在用户级过滤历史已购商品后，最终推荐列表可以因用户而不同；该配置必须写入 `params.json`。
- 不依赖 `user_profile.parquet` 和 `candidate_items.parquet`。

`recent_popularity`：

- 使用 cutoff 前最近 1 周和 4 周销量生成近期热门分。
- 不依赖用户画像。
- 可以不读取 `candidate_items.parquet`，直接由 method runner 生成 Top-12。

`attribute_similarity`：

- 读取 `user_profile.parquet` 或在 context 中按相同逻辑构建用户属性偏好。
- 用用户历史偏好属性与商品属性计算相似度。
- 第一版采用加权 Jaccard 或等价的归一化匹配分。
- 冷启动或画像为空用户回退到 `recent_popularity`。

`pop_similarity`：

- 无趋势主对照。
- 使用 `pop_score + sim_score + recent_score`。
- 作为 `pop_similarity_trend` 的消融对照。

## Trend-aware 主方法

`pop_similarity_trend` 是最终主方法。它使用：

```text
pop_score
sim_score
trend_score
recent_score
```

默认初始权重：

```json
{
  "pop_score": 0.35,
  "sim_score": 0.35,
  "trend_score": 0.25,
  "recent_score": 0.05
}
```

实验阶段会在 valid 窗口上搜索权重，test 使用 valid 选出的固定权重。权重必须非负，和必须为 1。无效权重直接失败，不能自动归一化掩盖配置错误。

## Trend Score 契约

推荐阶段不直接把 `pred_share_t1` 当推荐强度。`pred_share_t1` 是由预测增长率派生并归一化的下一周份额，不等价于商品推荐强度。

`trend_score` 从 stable LightGBM 的 `pred_target_growth` 生成：

```text
source: outputs/models/lightgbm/predictions.csv
prediction rows: predictions.week_id == cutoff_week
grouping: split + cutoff_week + attr_type
```

处理步骤：

1. 读取标准趋势预测表并执行严格列契约校验。
2. 按 `split + week_id + attr_type` 对 `pred_target_growth` 做组内标准化到 `[0, 1]`。
3. 将属性趋势分通过商品-属性边映射到 article。
4. 只使用 `recommendation.contracts.RECOMMENDATION_CORE_ATTR_TYPES` 中的核心属性类型聚合，避免非核心层级属性稀释趋势信号。
5. 商品多个属性命中时按属性类型权重聚合。
6. 商品没有趋势属性命中时 `trend_score = 0`。

默认属性类型权重也固定在 `recommendation.contracts`：

```json
{
  "product_type_name": 0.35,
  "colour_group_name": 0.25,
  "garment_group_name": 0.20,
  "product_group_name": 0.10,
  "graphical_appearance_name": 0.10
}
```

如果某个商品缺少部分属性类型，只在命中的属性类型上重新归一化权重。若全部缺失，则 `trend_score = 0`。

## Retrieval Strategy

候选 strategy 与最终 method 分离。

首批 strategy：

```text
popularity
similarity
trend_union
default
```

`popularity` 生成近期热门候选。`similarity` 生成用户属性相似候选。`trend_union` 生成趋势属性命中的活跃商品候选。`default` 是三者并集，用于 `pop_similarity` 和 `pop_similarity_trend`。

候选生成只负责召回，不负责最终 method 排序。`candidate_items.parquet` 必须记录 `strategy` 和 `candidate_sources`，避免多 strategy 输出互相覆盖或读错。

## Ranking Features

`ranking/features.py` 负责生成标准打分列：

```text
pop_score
sim_score
trend_score
recent_score
```

所有 score 必须是有限数值，并位于 `[0, 1]`。每个 score 的归一化边界需要按窗口计算，不能跨 split 或跨 label week 混合。缺失特征使用显式规则填充，例如缺少 trend 命中时 `trend_score = 0`。

归一化采用显式 min-max 规则：

```text
normalized = (value - min_value) / (max_value - min_value)
```

当归一化组内 `max_value == min_value` 时，该 score 在该组内全部填 `0.0`，表示该特征对当前组没有排序区分能力。不能填 NaN，也不能静默跳过该 feature。

各 score 的归一化作用域：

- `pop_score`：按 `split + cutoff_week + label_week` 在候选 article 集合内归一化。
- `recent_score`：按 `split + cutoff_week + label_week` 在候选 article 集合内归一化。
- `sim_score`：按 `split + cutoff_week + label_week + customer_id` 在该用户候选集内归一化。
- `trend_score`：先按 `split + cutoff_week + attr_type` 归一化属性趋势分，再映射聚合到 article；若后续对 article 级 trend score 再归一化，作用域必须是 `split + cutoff_week + label_week`。

`ranking/scoring.py` 只负责线性加权：

```text
score = sum(weight[feature] * feature_value)
```

排序稳定规则：

```text
score desc
article_id asc
```

Tie-breaker 必须稳定，保证重复运行可复现。

## 评价指标

`15_eval_recommendations.py --method <method>` 读取：

```text
outputs/recommendation/<method>/recommendations.csv
```

写入：

```text
outputs/recommendation/<method>/metrics.json
```

评价指标：

```text
MAP@12
Recall@12
HitRate@12
NDCG@12
Coverage
```

评价用户：

- `label_week` 有真实购买记录。
- `cutoff_week` 前有历史记录。
- 出现在 `target_users.parquet` 中。

`target_users.parquet` 是评价用户集合的唯一来源。推荐结果缺少某个 eligible 用户窗口时，默认按空推荐计 0 分并在 metrics metadata 中记录缺失数量；strict 模式可改为直接失败。评价逻辑不能通过筛选“已有推荐结果的用户”来缩小分母。

`Coverage` 按窗口先计算，再在 split 内做算术平均。单个窗口的分子是该窗口被推荐的唯一 `article_id` 数，分母是该窗口可推荐商品池规模。可推荐商品池第一版定义为 `week_id <= cutoff_week` 曾出现过的商品集合。`metrics.json` 需要记录 `coverage_by_window` 和 split-level average，避免多个 valid/test 窗口混成一个不清楚的全局分母。

指标 payload 必须是严格 JSON object，数值必须有限。没有有效评价用户时直接失败，不能输出空成功指标。

## 实验编排

`16_run_recommendation_experiment.py --experiment main` 负责论文实验版主流程。

流程：

1. 生成或读取 valid/test 推荐窗口。
2. 生成或读取 `time_windows.parquet`、`target_users.parquet` 和 `evaluation_labels.parquet`。
3. 确保 user profile 和 default candidates 存在。
4. 运行 baseline methods。
5. 在 valid 窗口上对 `pop_similarity_trend` 做权重网格搜索。
6. 使用 valid 最佳权重生成 test 推荐，并发布到 method-scoped 目录。
7. 评价所有 method。
8. 输出消融汇总。

权重搜索候选集第一版沿用实施方案：

```text
pop_score: 0.2, 0.3, 0.4
sim_score: 0.2, 0.3, 0.4
trend_score: 0.1, 0.2, 0.3
recent_score: 0.0, 0.05, 0.1
sum(weights) == 1
```

选择指标默认使用 valid `MAP@12`，并在实验 JSON 中同时记录 Recall@12、HitRate@12、NDCG@12 和 Coverage。test 只使用 valid 选出的权重，不能用 test 重新搜索。

grid search 的候选权重评估默认在内存中完成。若为了审计需要保存每个候选权重的推荐和指标，必须写入 `outputs/recommendation/experiments/<experiment_id>/runs/<run_id>/`，不能写入 stable method 目录。只有最终选中的权重可以写入 `outputs/recommendation/pop_similarity_trend/params.json`。

最小消融：

```text
global_popularity
recent_popularity
attribute_similarity
pop_similarity
pop_similarity_trend
```

可选增强消融：

```text
trend_only
pop_similarity_trend_without_recent
```

实现验收不要求趋势方法一定提升 baseline。代码验收关注产物正确、无泄漏、可复现；论文结论根据真实 metrics 判断提升、持平或主要提供解释性。

## 错误处理

必须显式失败的情况：

- 缺少周级交易、商品属性边或必要推荐产物。
- trend-aware method 或 `trend_union` strategy 缺少 stable trend predictions。
- `cutoff_week >= label_week`。
- 读取到 `predictions.week_id == label_week` 作为趋势分输入。
- method、strategy 或 experiment 名称未知。
- method、strategy、experiment_id 或实验 run_id 不是安全路径片段。
- 候选表列缺失、列重排、strategy 字段与路径不一致。
- 评价时缺少 `target_users.parquet` 或 `evaluation_labels.parquet`。
- `evaluation_labels.parquet` 中存在重复的 `split + cutoff_week + label_week + customer_id + article_id`。
- 非评价阶段读取 `evaluation_labels.parquet` 的 `article_id` 作为特征或候选来源。
- stable method 目录被 grid search 中间 run 覆盖。
- 归一化常量组产生 NaN 或非有限值。
- 推荐结果中同一用户窗口超过 12 个商品或出现重复商品。
- score 非有限或超出 `[0, 1]`。
- 权重为负、缺少 required feature 或权重和不等于 1。
- 评价用户为空。
- metrics payload 不是严格 JSON object 或含非有限数值。

不允许通过静默 fallback、空输出或 mock 成功掩盖真实问题。fallback 只用于明确策略，例如 attribute similarity 画像为空时回退 popularity，并必须记录在 metadata 或解释字段中。

## 测试设计

新增测试按能力边界拆分：

```text
tests/test_recommendation_time_windows.py
tests/test_recommendation_retrieval.py
tests/test_recommendation_ranking.py
tests/test_recommendation_methods.py
tests/test_recommendation_evaluation.py
tests/test_recommendation_experiments.py
```

单元测试覆盖：

- `time_windows.py` 生成 `label_week = cutoff_week + 1`。
- 非法窗口、重复窗口、缺失 split 失败。
- `target_users.parquet` 与 `evaluation_labels.parquet` 的 eligible user 口径一致。
- `evaluation_labels.parquet` 按 `customer_id + article_id + window` 去重，重复购买不会扩大 relevant set。
- 推荐结果缺少 eligible 用户时，默认按空推荐计 0 分；strict 模式失败。
- 读取趋势分时必须使用 `prediction.week_id == cutoff_week`。
- retrieval 只使用 `week_id <= cutoff_week`。
- `13 --strategy default` 输出 strategy-scoped 候选路径。
- `14 --method recent_popularity` 不要求 `user_profile.parquet`。
- `pop_similarity_trend` 缺少趋势预测时失败。
- ranking score 有限且位于 `[0, 1]`。
- min-max 常量组统一填 `0.0`，不产生 NaN。
- 核心属性类型来自 `recommendation.contracts.RECOMMENDATION_CORE_ATTR_TYPES`。
- 权重校验拒绝负数、缺失 feature 和权重和不等于 1。
- registry unknown method 报错并列出可用 method。
- method、strategy、experiment_id 和内部生成的 run_id 都必须通过安全路径片段校验。
- MAP@12、Recall@12、HitRate@12、NDCG@12、Coverage 用小样本精确断言；Coverage 覆盖多窗口先按窗口计算再聚合。
- `recommendations.csv` 和 `recommendation_items.csv` 列契约严格。
- `params.json`、`metadata.json`、`metrics.json` payload 必需字段完整。
- grid search 中间 run 不覆盖 `outputs/recommendation/<method>/`。

集成测试覆盖：

- 小型 fixture 跑通 `12 -> 13 -> 14 -> 15`，且 `12` 同时写出 windows、target users、evaluation labels 和 user profile。
- `global_popularity` 和 `recent_popularity` 不依赖 user profile 或 candidates。
- `pop_similarity_trend` 使用 default candidates、stable trend predictions 和配置权重生成推荐。
- 每个用户窗口最多 12 个推荐商品，且无重复 article。
- `metadata.json` 中的输入路径和行数与真实输出一致。
- `tests/test_architecture_boundaries.py` 继续通过，确保 recommendation 只读公开上游接口。

真实产物验收命令：

```sh
uv run pytest tests/test_recommendation_*.py tests/test_architecture_boundaries.py
uv run python -m compileall -q src
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy default
uv run python src/14_rerank_recommendations.py --method pop_similarity_trend
uv run python src/15_eval_recommendations.py --method pop_similarity_trend
uv run python src/16_run_recommendation_experiment.py --experiment main
```

实现过程中若修改 README、implementation plan、架构边界测试或 artifact 路径，需要同步运行对应聚焦测试和 `git diff --check`。
