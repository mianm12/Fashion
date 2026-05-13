# 推荐增强设计

## 背景

当前 `Fashion` 已经完成属性周级趋势预测、LightGBM 趋势主模型、轻量 Top-N 推荐实验、推荐消融、论文素材导出和本地答辩展示应用。推荐模块的定位仍是趋势预测的应用验证层，不是复现 H&M Kaggle 高分推荐方案。

现有推荐结果的主要问题不是代码闭环，而是推荐候选覆盖偏弱。当前 `pop_similarity_trend` 能证明趋势预测结果可以进入推荐解释链路，但 MAP@12、NDCG@12 和 candidate recall 仍偏低；`recent_popularity` 仍是强 test baseline。因此本设计借鉴 `recomd` 的有效推荐经验，在不破坏论文主线的前提下增强 `Fashion` 的推荐指标。

`recomd` 的高推荐指标主要来自 H&M 推荐赛风格的召回与排序：复购、同 `product_code` 变体、短期热门、年龄段热门、用户偏好属性热门和 LightGBM Ranker。`Fashion` 不能直接搬运该实现，因为现有项目已经有明确的 `strategy`、`method`、`experiment` 三层边界、周级时间窗口、防泄漏规则、freshness metadata 和论文表述约束。

## 已确认决策

- 采用两阶段设计：第一阶段做可控的周级候选与线性排序增强；第二阶段仅预留 `lightgbm_ranker` 推荐方法。
- 第一阶段以论文稳妥为优先目标：candidate recall、MAP@12、NDCG@12 有可解释提升即可；不要求全面超过 `recent_popularity`。
- 第一阶段只把 `customers.csv` 中的 `age` 派生为 `age_bucket` 并作为用户分群推荐信号；`club_member_status` 和 `fashion_news_frequency` 如保留，只作为受控输入 artifact 的审计字段或后续预留字段，不能在第一阶段 retrieval 或 ranking 中参与召回、排序。
- 第一阶段坚持使用周级交易输入，不引入日级 `t_dat` 推荐 artifact。
- 第一阶段采用“候选增强 + 线性排序特征增强”，不训练 Ranker。
- 增强实验不覆盖现有 `main` 实验和 `pop_similarity_trend` stable 输出。

## 目标

- 提升现有趋势感知推荐的 candidate recall、MAP@12 和 NDCG@12。
- 借鉴 `recomd` 中被验证有效的复购、同款变体、用户基础画像和偏好属性热门召回。
- 保持 `Fashion` 论文主线为属性级趋势预测，推荐增强只作为应用验证增强实验。
- 保持现有 valid/test 时间窗口、防泄漏规则、target users 和 evaluation labels 口径不变。
- 生成可审计、可复现、可进入论文对比表的增强实验 artifact。
- 显式评估 H&M 任务中的复购语义，避免新增 `reorder` 来源后又被现有 seen-item 过滤完全抵消。

## 非目标

- 不引入日级交易 artifact 或直接使用 `t_dat` 做短期日级窗口。
- 不在第一阶段训练 LightGBM Ranker、双塔、LightGCN 或深度推荐模型。
- 不直接迁移 `recomd` 的单文件实现。
- 不缩小评价用户、不改变 label 口径、不只统计命中用户来制造指标提升。
- 不把推荐增强写成论文主贡献。
- 不默认让增强结果覆盖 reports、defense app 或现有 stable 推荐方法。
- 不改变现有 `main` 实验的 seen-item 过滤默认语义。

## 与 recomd 的借鉴边界

可借鉴：

- 复购召回：用户历史购买商品或历史强偏好商品。
- 同款变体召回：基于 `product_code` 找同款不同商品。
- 用户分群热门：第一阶段只使用 `age_bucket`，避免细分过稀。
- 用户偏好属性热门：基于用户历史偏好的品类、颜色、服装组等核心属性召回近期热门商品。
- 候选来源 flag、source rank、source count 等可解释排序特征。

不可直接借鉴：

- `recomd` 的日级 `1/2/3/7/14/30` 天窗口；第一阶段只做周级近窗口。
- `recomd` 的趋势分公式；`Fashion` 必须继续使用趋势预测模型输出。
- `recomd` 的最后 7 天单窗口评价；`Fashion` 必须保留 valid/test 多窗口评价。
- 单文件式 LGBMRanker 实现；第二阶段如做 Ranker，必须拆入现有 recommendation 分层。

## 架构设计

继续沿用现有三层边界：

```text
strategy   -> 候选召回策略
method     -> 排序和 Top-N 输出方法
experiment -> 编排、调参、消融和报告 payload
```

第一阶段新增：

```text
enhanced_default candidate strategy
enhanced_pop_similarity_trend method
recommendation_enhanced experiment
```

建议模块变化：

```text
src/fashion_trend/recommendation/
  inputs.py
  contracts.py
  readers.py
  paths.py

  retrieval/
    reorder.py
    product_variants.py
    customer_segments.py
    preference_popularity.py
    candidates.py

  ranking/
    features.py
    scoring.py
    weights.py

  methods/
    trend_aware/
      enhanced_pop_similarity_trend.py

  experiments/
    enhanced_runner.py 或在 runner.py 中保持清晰分区
```

如果修改 `experiments/runner.py` 会继续放大文件，应优先拆出增强实验 runner，主入口只做薄编排。新增增强召回和增强排序特征也应优先拆到独立 source/enhanced feature 模块，避免把复购、变体、分群热门和特征公式继续堆进现有通用文件。

## Artifact 契约

新增输入 artifact：

```text
data/processed/recommend/customer_profile.parquet
data/processed/recommend/article_product_map.parquet
```

`customer_profile.parquet` 必需列：

```text
customer_id
age
age_bucket
club_member_status
fashion_news_frequency
```

契约：

- `customer_id` 必须是字符串并唯一。
- `age` 保留原始可解析年龄值；无法解析时允许为空，但不得被填成 0 岁或平均年龄。
- `age_bucket` 必须非空；合法枚举固定为 `unknown`、`0-19`、`20-29`、`30-39`、`40-49`、`50-59`、`60+`。
- 无法解析年龄或年龄缺失时，`age_bucket` 使用 `unknown`。
- `club_member_status` 和 `fashion_news_frequency` 缺失填 `unknown`，第一阶段不得作为推荐特征使用。
- 该 artifact 由推荐输入编排层以 `customers.csv` 路径或 DataFrame 作为显式输入构建，并写入 recommendation input metadata；recommendation 核心包只消费生成后的 artifact 和 path metadata，不直接依赖 datasets 路径模块。
- metadata 必须记录 customer profile 的 schema version、分桶算法版本、customers 源路径和 fingerprint。

`article_product_map.parquet` 必需列：

```text
article_id
product_code
```

契约：

- 该 artifact 由推荐输入阶段基于 `data/interim/articles_clean.csv` 或等价公开 catalog reader 构建。
- `article_id` 必须是字符串并唯一。
- `product_code` 保持字符串语义；基于当前 `articles_clean.csv` 构建时缺失率必须为 0。
- recommendation input metadata 必须记录其来源路径、schema version 和 catalog fingerprint。

新增候选 strategy：

```text
data/processed/recommend/candidates/enhanced_default/candidate_items.parquet
data/processed/recommend/candidates/enhanced_default/metadata.json
```

`metadata.json` 必须记录 weekly transactions、catalog/product attributes、article product map、customer profile、user profile、trend predictions、time windows 和 target users 的 fingerprint。

新增 method 输出：

```text
outputs/recommendation/enhanced_pop_similarity_trend/recommendation_items.parquet
outputs/recommendation/enhanced_pop_similarity_trend/recommendations.csv
outputs/recommendation/enhanced_pop_similarity_trend/params.json
outputs/recommendation/enhanced_pop_similarity_trend/metadata.json
outputs/recommendation/enhanced_pop_similarity_trend/metrics.json
```

新增 experiment 输出：

```text
outputs/recommendation/experiments/recommendation_enhanced/experiment.json
```

增强实验不得覆盖 `outputs/recommendation/experiments/main/experiment.json`。

## 候选增强设计

### reorder

按每个 `customer_id + split + cutoff_week + label_week` 的 cutoff 前历史交易召回用户历史商品。排序依据：

1. `last_purchase_week` 越近越靠前。
2. `purchase_count` 越高越靠前。
3. `article_id` 稳定升序 tie-break。

第一阶段因为没有日级数据，不使用购买日期间隔。H&M 标签允许用户重复购买历史商品，因此增强实验必须显式建模 seen-item 策略：

- `enhanced_pop_similarity_trend` 默认使用 `include_seen_for_reorder=true`，只允许 `reorder` 来源的历史商品保留为候选。
- 非 `reorder` 来源继续沿用现有 seen-item 过滤，避免热门、属性和趋势来源把大量已购买商品重新塞回候选池。
- 候选合并后必须仍能判断某个 `customer_id + article_id + window` 是否由 `reorder` 命中。设计上可通过 `has_reorder_source` / `allow_seen` 字段，或等价的 source-level 明细 sidecar 表表达，但不能只依赖去重后的 `primary_source`。
- 多来源命中同一历史商品时，只要 `candidate_sources` 中包含 `reorder`，该商品可被 `allow_seen=true` 保留；不含 `reorder` 的历史商品必须被过滤。
- `candidate_seen_flags` 在增强策略下必须能表达 `is_seen` 与 `allow_seen`，method 层按 `is_seen=true and allow_seen=false` 过滤。
- 输出 metadata 必须记录该策略；实验 payload 必须包含一个 `enhanced_seen_filtered` 对照行，它使用与 Full Model 相同候选来源和权重，但将所有历史已购商品都过滤，用于说明复购语义对指标的影响。
- 不允许静默退化为 method-level `exclude_seen=false`。如果后续实施发现 source-level seen 过滤不可承受，必须重新修订设计或单独降级实验口径。

### product variants

基于用户近期或高频历史商品的 `product_code`，召回同 `product_code` 下其他商品。候选商品排序使用 cutoff 前该商品周级购买量、变体 rank 和稳定 tie-break。

要求：

- 推荐输入必须通过 `article_product_map.parquet` 或等价公开 reader 拿到 `article_id -> product_code`，并进入候选 metadata fingerprint。
- 当前商品本身不能作为自己的变体候选。
- 缺失 `product_code` 的商品跳过，不使用 `unknown` 合并成大组。

### customer segment popularity

基于 `customer_profile.age_bucket` 构建年龄段热门候选。第一阶段不组合 `club_member_status` 和 `fashion_news_frequency`，避免过度稀疏。

窗口：

- 使用 `week_id <= cutoff_week` 且近 `4` 周的历史。
- 第一阶段不在 `age_popularity` source 内部做全局热门 backfill；年龄段候选不足时允许该 source 少于 Top-12，由 existing `popularity` source 承担全局热门覆盖，避免用户分群贡献归因混淆。

### preference popularity

借鉴 `recomd` 的 user_type/user_section 思路。对每个用户窗口：

1. 从现有 `user_profile.parquet` 取用户 Top 属性偏好。
2. 限定核心属性类型，例如 `product_type_name`、`colour_group_name`、`garment_group_name`、`product_group_name`、`graphical_appearance_name`。
3. 在 cutoff 前近 `4` 周中，召回这些属性下的热门商品。

该来源与现有 attribute similarity 不同：attribute similarity 更像“用户偏好与商品属性匹配”，preference popularity 是“用户偏好属性下的近期热门商品召回”。

### 第一阶段候选预算

候选预算固定如下，并必须写入 candidate metadata：

- existing `popularity`、`similarity`、`trend` 沿用现有每 source 每用户窗口 Top-12 预算。
- `reorder` 每用户窗口 Top-12。
- `product_variant` 仅使用该用户窗口内 `reorder` 排序后的 Top-6 历史商品作为 seed；每个 seed product_code 最多召回 Top-3 变体，最终每用户窗口最多保留 Top-12 `product_variant` 候选。
- `age_popularity` 每个 `age_bucket + window` 预先取 Top-50 热门池，每用户窗口最多保留 Top-12。
- `preference_popularity` 每用户窗口最多使用 Top-3 偏好属性，每个属性最多召回 Top-4 近期热门商品，最终每用户窗口最多保留 Top-12。
- 所有 source 的 Top-N 都先在各自 source 内排序后截断，再进入 `enhanced_default` 合并；不得通过扩大 source cap 来调参。

### enhanced_default 合并

`enhanced_default` 合并以下来源：

- existing `popularity`
- existing `similarity`
- existing `trend`
- `reorder`
- `product_variant`
- `age_popularity`
- `preference_popularity`

`SOURCE_ORDER` 必须显式包含全部 source。候选合并仍按 `split + cutoff_week + label_week + customer_id + article_id` 去重，保留：

- `candidate_sources`
- `primary_source`
- `best_source_rank`
- `has_reorder_source`
- `allow_seen`

source-specific rank 字段如后续需要，应进入 feature cache 或 source-level 明细 sidecar，而不是把每个 source 的排序细节直接塞进候选基础列契约。

## 排序特征与方法设计

在现有 `pop_score`、`recent_score`、`sim_score`、`trend_score` 基础上新增：

- `reorder_score`
- `variant_score`
- `age_pop_score`
- `preference_pop_score`
- `source_rank_score`
- `source_count_score`

所有新增分数必须在 `customer_id + split + cutoff_week + label_week` 组内归一化到 `[0, 1]`，并拒绝非有限值。

新增分数的原始信号契约固定如下。`rank_norm(rank, cap) = (cap - rank + 1) / cap`，rank 从 `1` 开始，缺失来源或缺失 join 时默认原始值为 `0`；组内 min-max 归一化时如果最大值等于最小值，则输出 `0`。

| score column | 原始信号 | cache feature | cache join key | 缺失默认 |
| --- | --- | --- | --- | --- |
| `reorder_score` | `0.7 * rank_norm(last_purchase_rank, 12) + 0.3 * rank_norm(purchase_count_rank, 12)`，仅对 `reorder` 命中候选生效 | `reorder_scores` | `split, cutoff_week, label_week, strategy, customer_id, article_id` | `0` |
| `variant_score` | `0.7 * rank_norm(variant_pop_rank, 12) + 0.3 * rank_norm(seed_reorder_rank, 6)`，仅对 `product_variant` 命中候选生效 | `variant_scores` | `split, cutoff_week, label_week, strategy, customer_id, article_id` | `0` |
| `age_pop_score` | `rank_norm(age_bucket_pop_rank, 12)`，仅对 `age_popularity` 命中候选生效 | `age_popularity_scores` | `split, cutoff_week, label_week, strategy, customer_id, article_id` | `0` |
| `preference_pop_score` | 对命中的用户 Top 属性取 `max(preference_score * rank_norm(attribute_pop_rank, 4))`，仅对 `preference_popularity` 命中候选生效 | `preference_popularity_scores` | `split, cutoff_week, label_week, strategy, customer_id, article_id` | `0` |
| `source_rank_score` | 对当前候选保留的全部 source-specific rank 取最大 `rank_norm(source_rank, source_cap)`；source-level ablation 后按过滤后 source 重算 | `source_rank_scores` | `split, cutoff_week, label_week, strategy, customer_id, article_id` | `0` |
| `source_count_score` | `min(source_count, source_count_cap) / source_count_cap`，其中 `source_count_cap` 是当前策略或消融候选池保留的 source 数 | `source_count_scores` | `split, cutoff_week, label_week, strategy, customer_id, article_id` | `0` |

新增分数必须进入现有 feature cache/freshness 体系。第一阶段固定 feature name 与 score column 的对应关系：

```text
reorder_scores -> reorder_score
variant_scores -> variant_score
age_popularity_scores -> age_pop_score
preference_popularity_scores -> preference_pop_score
source_rank_scores -> source_rank_score
source_count_scores -> source_count_score
```

缓存路径沿用现有分区形态：

```text
data/processed/recommend/features/<feature_name>/strategy=enhanced_default/split=<split>/cutoff_week=<week>/part.parquet
```

feature metadata 必须记录对应输入 artifact、schema version、算法版本和 strategy。`enhanced_pop_similarity_trend` 的 method metadata 必须引用实际使用的 feature partition 与 metadata，不能复用 `default` strategy 的旧分区。

`enhanced_pop_similarity_trend` 使用线性加权：

```text
score =
  w_pop * pop_score
  + w_recent * recent_score
  + w_sim * sim_score
  + w_trend * trend_score
  + w_reorder * reorder_score
  + w_variant * variant_score
  + w_age * age_pop_score
  + w_pref_pop * preference_pop_score
  + w_source_rank * source_rank_score
  + w_source_count * source_count_score
```

`enhanced_pop_similarity_trend` 禁用 method-level backfill，不通过排序后追加近期热门来补齐 Top-12；当某些用户窗口不足 Top-12 时，只记录 underfilled diagnostics。若后续需要 backfill，必须新增显式候选 source，并写入 source cap、metadata、fingerprint、recall 和 contribution 口径。

权重由 valid split 有界搜索选择，test split 只做最终评价。第一阶段搜索空间必须是预先枚举的权重组，不超过 `32` 组；所有权重非负、和为 1，并在 experiment payload 中记录完整候选权重、valid 选择指标和最终入选权重。选择指标以 valid MAP@12 为主，NDCG@12 作为并列时的 tie-break；candidate recall 只作诊断，不参与权重选择。不允许基于 test split 继续调参。`recommendation_enhanced` payload 必须记录 `selection_metric=map_at_12`、`tie_break=ndcg_at_12`；该选择只适用于增强实验，不改变现有 `main` 实验语义。

## 实验设计

新增实验：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment recommendation_enhanced
```

对比组：

- `recent_popularity`
- `pop_similarity`
- `pop_similarity_trend`
- `enhanced_pop_similarity_trend`
- `enhanced_w/o Trend Score`
- `enhanced_w/o Trend Source+Score`
- `enhanced_w/o Reorder/Variant`
- `enhanced_w/o Customer Segment`
- `enhanced_seen_filtered`

`enhanced_w/o Trend Score` 从增强 Full Model 权重中删除 `trend_score` 后归一化剩余权重，但仍允许候选池包含 existing `trend` source，因此只能用于说明趋势打分贡献。`enhanced_w/o Trend Source+Score` 同时排除 existing `trend` source 与 `trend_score`，是支撑“趋势信号带来推荐增益”表述的严格消融。`enhanced_w/o Reorder/Variant` 与 `enhanced_w/o Customer Segment` 使用 source-level 消融，避免只把权重置零但仍保留候选池带来的混淆。

消融口径固定如下：

- `enhanced_w/o Trend Source+Score` 是 source-level ablation：排除 existing `trend` 来源，并移除 `trend_score` 权重；该消融同样适用 source-derived 字段和特征重算规则。
- `enhanced_w/o Reorder/Variant` 是 source-level ablation：重建或过滤候选池，排除 `reorder` 和 `product_variant` 来源，并移除对应特征权重。
- `enhanced_w/o Customer Segment` 是 source-level ablation：排除 `age_popularity` 来源，并移除 `age_pop_score` 权重。
- `enhanced_seen_filtered` 使用与 Full Model 相同候选来源和权重，但对所有 source 启用现有 seen-item 过滤，用来衡量复购召回贡献。
- 所有 source-level ablation 都必须使用相同 target users、time windows 和 evaluation labels。
- source-level ablation 不得覆盖 `enhanced_default` 候选 artifact。设计上可使用独立 strategy ID 或 experiment run 内派生候选池；无论是否物化，experiment payload 都必须记录 source filter、候选行数、avg candidates per user、candidate recall、使用的 fingerprints，以及是否写出独立候选 artifact。
- source-level ablation 必须基于过滤后的候选池重新计算 `candidate_sources`、`primary_source`、`best_source_rank`、`has_reorder_source`、`allow_seen`、`source_rank_score` 和 `source_count_score`；不得复用 Full Model 中包含被移除 source 的 source-derived 字段或分数。对应 lineage 必须写入 experiment payload。

实验 payload 必须包含：

- valid/test 推荐指标。
- candidate recall 诊断，至少包含 `candidate_recall_pre_seen` 和 `candidate_recall_post_seen`。
- avg candidates per user。
- source coverage。
- source hit contribution，至少包含 `source_hit_contribution_pre_seen` 和 `source_hit_contribution_post_seen`。
- 权重配置。
- 消融结果。

candidate recall 口径固定为：在每个 `split + cutoff_week + label_week` 内，以完整 target users 对应的 evaluation labels 为分母，统计 label 商品是否出现在候选池中；没有候选或没有推荐命中的用户也必须计入分母。`pre_seen` 使用 seen 策略前候选池，`post_seen` 使用最终进入 ranking 的候选池。

source contribution 口径固定为：命中 label 的候选按 `candidate_sources` 记录 all-source 命中，同时按命中 source 数做 fractional contribution，使各 source 贡献可加总；`primary_source` 只作为辅助诊断，不作为论文主贡献归因。实验 payload 同时输出 `source_hit_contribution_pre_seen` 和 `source_hit_contribution_post_seen`；论文主表默认使用 post-seen 口径。跨窗口汇总使用 label item micro-average，window-level macro-average 只作为诊断字段。

## Go/No-Go 标准

第一阶段成功条件：

- `enhanced_pop_similarity_trend` 相对 `pop_similarity_trend` 在 MAP@12 或 NDCG@12 上有提升。
- candidate recall 相对现有 `default` 候选池有明确提升。
- valid/test 都有完整指标。
- 指标提升能通过 source contribution 或消融结果解释。

不要求：

- 不要求全面超过 `recent_popularity`。
- 不要求趋势分消融一定显著下降。

如果增强方法没有超过 `recent_popularity`，论文仍可写作“增强趋势感知推荐相对原融合方法有效，但 H&M 数据上近期热门仍是强 baseline”。如果严格 `enhanced_w/o Trend Source+Score` 没有明显下降，趋势信号必须如实写成解释性辅助信号，不能写成稳定推荐增益来源。

## 错误处理

必须显式失败的情况：

- `customer_profile.parquet` 缺少必需列、`customer_id` 重复或 ID 被数值化。
- `age_bucket` 为空或不属于固定合法枚举。
- `article_product_map.parquet` 缺失、`article_id` 重复或未进入 recommendation input metadata。
- 基于当前 `articles_clean.csv` 构建 `article_product_map.parquet` 时，`product_code` 缺失率不为 0，或同款变体逻辑把缺失值聚合成大组。
- 新 source 未注册进 `SOURCE_ORDER`。
- `enhanced_default` 缺少 `has_reorder_source` / `allow_seen` 等价信息，导致 source-level seen 语义不可复查。
- `reorder` source 的 seen-item 策略未写入 metadata。
- `enhanced_default` metadata 缺少任一关键输入 fingerprint。
- 增强 feature cache 试图复用 `default` strategy 的旧分区。
- `enhanced_pop_similarity_trend` 缺少 required feature。
- 任一新增 score 含非有限值或越界。
- 任一新增 score 缺少原始信号、join key 或缺失默认值定义。
- `enhanced_pop_similarity_trend` 启用了 method-level backfill，或追加了未注册为显式 source 的补齐候选。
- 实验 payload 缺少 valid/test 任一 split。
- candidate recall 诊断只统计有推荐用户，而不是完整 target users。
- candidate metadata 缺少 source-level 候选预算，或实际 source cap 超出第一阶段固定预算。
- source-level ablation 覆盖或污染 `enhanced_default` 候选 artifact。
- source-level ablation 没有重算 source-derived 字段或特征。

## 测试与验证

输入层测试：

- customer profile 构建、缺失值填充、固定年龄分桶枚举和重复用户拒绝。
- customers 输入路径进入 metadata/fingerprint。
- article product map 构建、`article_id` 唯一性和 catalog fingerprint。

retrieval 测试：

- reorder 只使用 cutoff 前历史。
- 各 source 候选数不超过第一阶段固定预算。
- product variants 只召回同 `product_code` 的其他商品。
- age popularity 只使用 cutoff 前近 4 周历史。
- age popularity 不做 source 内部全局热门 backfill。
- preference popularity 只使用用户历史偏好属性和 cutoff 前热门。
- source-level seen 过滤只保留 `reorder` 来源的历史商品，其他来源仍过滤 seen items。

候选合并测试：

- `enhanced_default` 合并多 source 后按 key 去重。
- `candidate_sources` 顺序稳定。
- `has_reorder_source` / `allow_seen` 在多来源去重后仍可复查。
- unknown source 拒绝。

ranking feature 测试：

- 新增 score 范围在 `[0, 1]`。
- 组内常量分数归一化为 0。
- 缺失必要输入时按契约失败。
- 新增 score 的 feature cache 分区和 metadata 不能复用 `default` strategy。

method 测试：

- `enhanced_pop_similarity_trend` required features 完整。
- `enhanced_pop_similarity_trend` 不启用 method-level backfill，并记录 underfilled diagnostics。
- 权重非负、有限、和为 1。
- 最多 Top-12 输出 rank 连续、无重复商品。

experiment 测试：

- valid search 和 test final 分离。
- 增强消融权重从 Full Model 动态派生。
- source-level ablation 的候选来源排除规则被记录并可复查。
- source-level ablation 后的 source-derived 字段和特征已重算。
- candidate recall pre/post seen 和 source contribution pre/post seen 指标完整。
- `recommendation_enhanced` payload 记录 `selection_metric=map_at_12` 和 `tie_break=ndcg_at_12`。
- freshness guard 覆盖 candidates、cache、method output 和 experiment payload。

架构测试：

- recommendation 仍只消费 public contracts/readers。
- retrieval/ranking 层不能直接读取原始 CSV。
- 不引入 `recomd` 作为依赖。

## 论文表述边界

可以写：

> 在保持属性趋势预测主线和严格时间窗口评价不变的前提下，本文进一步借鉴推荐系统中的复购、同款变体和用户基础画像召回策略，构建增强趋势感知推荐实验。结果表明，增强候选召回可以改善推荐应用验证指标，但近期热门仍是 H&M 数据上的强基线。

不能写：

- 趋势分显著提升推荐性能，除非严格消融支持。
- 推荐系统达到 Kaggle 高分水平。
- 用户画像或知识图谱是监督排序模型核心输入。
- 推荐增强是本文主贡献。
- 增强实验证明趋势预测模型本身带来稳定推荐增益，除非严格 `enhanced_w/o Trend Source+Score` 显著下降。

## 第二阶段预留：lightgbm_ranker

第二阶段可新增 `lightgbm_ranker` method，但必须满足：

- 仍使用 `Fashion` 的 valid/test 多窗口和 target users/evaluation labels。
- 候选来源复用 `enhanced_default`。
- 训练样本只使用 cutoff 前特征和 label_week 标签。
- Ranker 输出作为独立 method，不覆盖 `enhanced_pop_similarity_trend`。
- 论文表述为“推荐增强扩展实验”，不改变主贡献。

第二阶段不在第一阶段实现计划中展开。只有当第一阶段增强不足以支撑论文推荐应用验证，或用户明确要求冲推荐指标时，再单独写 Ranker 设计或实施计划。
