# 论文撰写数据汇总

更新时间：2026-05-14

## 1. 使用边界

本文档汇总论文撰写需要反复引用的项目事实、模型算法、阶段产物和实验指标。数据以当前仓库稳定 artifact 为准，主要来源包括：

- `data/processed/basic/data_profile.csv`
- `data/processed/basic/date_range.json`
- `data/processed/features/trend_model_samples_split_metadata.json`
- `data/processed/recommend/metadata.json`
- `data/processed/recommend/customer_profile.parquet`
- `data/processed/recommend/article_product_map.parquet`
- `data/processed/recommend/candidates/<strategy>/candidate_items.parquet`
- `outputs/metrics/<model>/trend_metrics.json`
- `outputs/models/lightgbm/params.json`
- `outputs/models/lightgbm/feature_importance.csv`
- `outputs/recommendation/<method>/metrics.json`
- `outputs/recommendation/experiments/main/experiment.json`
- `outputs/recommendation/experiments/recommendation_enhanced/experiment.json`
- `outputs/reports/tables/*.md`
- `outputs/reports/manifest.json`

论文题目保持为《融合用户行为与知识图谱的服装流行趋势预测与推荐》。本文档中的“属性级趋势预测”和“轻量 Top-N 推荐”不是对题目的重命名或收缩，而是当前仓库对该题目的可复现实现路径：以 H&M 用户交易序列作为用户行为信号，以商品属性层次图作为知识图谱表示，在属性周级热度上建模服装流行趋势，并将趋势预测结果融合到离线推荐排序中。论文表述应围绕“用户行为 + 知识图谱 + 趋势预测 + 推荐”展开，同时如实说明当前推荐模块是离线实验层，不是在线服务、生产推荐平台、深度推荐模型或 Kaggle 高分方案。

## 2. 研究主线与阶段产物

```text
H&M transactions_train.csv（用户购买行为）
    -> 周级交易表
    -> 商品周销量
    -> 属性周热度
    -> 趋势标签与趋势样本
    -> 趋势预测模型
    -> 趋势感知 Top-N 推荐实验

H&M articles.csv（商品属性知识）
    -> 商品清洗
    -> 商品属性层次图 / 知识图谱
    -> 趋势样本图结构特征
    -> 推荐候选与解释字段
```

题目中的核心概念在当前实现中的落点如下：

| 题目关键词 | 当前实现落点 | 论文表述建议 |
| --- | --- | --- |
| 用户行为 | `transactions_train.csv`、周级交易、商品周销量、用户历史购买画像、推荐评价标签 | 写成用户购买行为序列和历史偏好信号 |
| 知识图谱 | 商品节点、属性节点、商品-属性边、属性层级边 | 写成基于商品结构化属性构建的服装商品属性知识图谱 |
| 流行趋势预测与推荐 | 属性周热度、趋势标签、LightGBM 趋势预测、趋势感知 Top-N 推荐 | 写成先预测服装属性流行趋势，再将趋势信号融合到推荐排序 |

| 阶段 | 当前状态 | 关键产物 |
| --- | --- | --- |
| 原始数据 profile | 已实现 | `data/processed/basic/data_profile.csv`、`date_range.json` |
| 周级交易 | 已实现 | `data/interim/transactions_train_weekly.parquet` |
| 商品清洗 | 已实现 | `data/interim/articles_clean.csv` |
| 属性图 | 已实现 | `data/processed/graph/*.csv` |
| 商品周销量 | 已实现 | `data/processed/trend/article_week_sales.csv` |
| 属性周热度 | 已实现 | `data/processed/trend/attribute_week_heat.csv` |
| 趋势标签 | 已实现 | `data/processed/trend/attribute_week_target.csv` |
| 趋势样本与切分 | 已实现 | `data/processed/features/trend_model_samples*.parquet` |
| 趋势 baseline | 已实现 | `last_week`、`previous_growth`、`moving_average` |
| 趋势主模型 | 已实现 | `lightgbm` |
| 趋势评价 | 已实现 | `outputs/metrics/<model>/trend_metrics.json` |
| 推荐输入 | 已实现 | `time_windows`、`target_users`、`evaluation_labels`、`user_profile`、`customer_profile`、`article_product_map` |
| 推荐候选 | 已实现 | `popularity`、`similarity`、`trend_union`、`default`、`enhanced_default` |
| 推荐方法 | 已实现 | 5 个默认稳定 Top-N 方法；1 个可选增强方法通过实验入口评价 |
| 推荐评价与主实验 | 已实现 | `outputs/recommendation/<method>/metrics.json`、`experiments/main/experiment.json` |
| 可选推荐增强实验 | 已实现 | `experiments/recommendation_enhanced/experiment.json` |
| 论文素材导出 | 已实现 | `outputs/reports/figures/`、`tables/`、`case_studies/`、`manifest.json` |
| 本地答辩展示应用 | 已实现 | `outputs/defense_app/fashion_demo.sqlite`、`apps/defense_app/` |

## 3. 原始数据概况

数据集：Kaggle H&M Personalized Fashion Recommendations。

| 原始表 | 行数 | 列数 | 缺失单元格数 | 关键唯一值 |
| --- | ---: | ---: | ---: | --- |
| `transactions_train.csv` | 31,788,324 | 5 | 0 | `customer_id=1,362,281`；`article_id=104,547` |
| `articles.csv` | 105,542 | 25 | 416 | `article_id=105,542`；`product_code=47,224` |
| `customers.csv` | 1,371,980 | 7 | 1,840,560 | `customer_id=1,371,980`；`postal_code=352,899` |

| 字段 | 最早日期 | 最晚日期 | 覆盖天数 |
| --- | --- | --- | ---: |
| `transactions.t_dat` | 2018-09-20 | 2020-09-22 | 734 |

可写入论文的数据质量要点：

- 交易表核心字段无缺失。
- 商品表主要结构化属性无缺失，`detail_desc` 缺失 416 条，缺失率约 0.3942%。
- 用户表 `FN`、`Active` 缺失较多；本项目推荐实验主要依赖交易历史和商品属性，不依赖这两个字段作为核心建模输入。
- `article_id` 与 `customer_id` 在处理链路中保持字符串语义，避免丢失前导 0 或长 ID 精度。

## 4. 处理后数据规模

| 领域 | artifact | 行数 | 列数 | 论文用途 |
| --- | --- | ---: | ---: | --- |
| transactions | `data/interim/transactions_train_weekly.parquet` | 31,788,324 | 6 | 原始交易到周级交易的转换规模 |
| catalog | `data/interim/articles_clean.csv` | 105,542 | 13 | 商品属性清洗规模 |
| attribute_graph | `nodes_article.csv` | 105,542 | 4 | 属性图商品节点规模 |
| attribute_graph | `nodes_attribute.csv` | 592 | 7 | 属性图属性节点规模 |
| attribute_graph | `edges_article_attribute.csv` | 1,055,420 | 7 | 商品-属性边规模 |
| attribute_graph | `edges_attribute_hierarchy.csv` | 658 | 6 | 属性层级边规模 |
| trend | `article_week_sales.csv` | 2,203,988 | 5 | 商品周销量聚合规模 |
| trend | `attribute_week_heat.csv` | 62,160 | 9 | 属性周热度规模 |
| trend | `attribute_week_target.csv` | 61,568 | 12 | 趋势标签规模 |
| trend | `trend_model_samples.parquet` | 59,200 | 37 | 趋势模型样本规模 |
| recommendation | `time_windows.parquet` | 16 | 3 | 推荐评价窗口 |
| recommendation | `target_users.parquet` | 1,203,649 | 6 | 推荐目标用户集合 |
| recommendation | `evaluation_labels.parquet` | 4,195,886 | 5 | 推荐真实标签 |
| recommendation | `user_profile.parquet` | 3,610,947 | 10 | 用户属性画像 |
| recommendation | `customer_profile.parquet` | 1,371,980 | 5 | 用户基础画像和年龄段增强特征 |
| recommendation | `article_product_map.parquet` | 105,542 | 2 | 商品款式族映射和变体增强候选 |

属性图覆盖 10 类商品属性字段：`colour_group_name`、`department_name`、`garment_group_name`、`graphical_appearance_name`、`index_group_name`、`index_name`、`perceived_colour_master_name`、`product_group_name`、`product_type_name`、`section_name`。每个商品连接 10 个属性字段，因此商品-属性边为 `105,542 * 10 = 1,055,420`。

## 5. 商品属性知识图谱设计

商品属性知识图谱在当前实现中表现为静态异构图，用可审查的 CSV 节点表和边表保存，不引入 Neo4j 或外部图数据库。论文中可形式化定义为：

```text
G = (V, E)
V = V_article union V_attribute
E = E_article_attribute union E_attribute_attribute
```

| 图元素 | 含义 | 当前产物 |
| --- | --- | --- |
| `V_article` | 商品节点集合，一件商品对应一个节点 | `nodes_article.csv` |
| `V_attribute` | 属性节点集合，不同属性类型和取值组合对应一个节点 | `nodes_attribute.csv` |
| `E_article_attribute` | 商品到属性的隶属边 | `edges_article_attribute.csv` |
| `E_attribute_attribute` | 属性之间的父子层级边 | `edges_attribute_hierarchy.csv` |

### 5.1 节点设计

商品节点来自 `articles_clean.csv`，保留商品 ID、款式族和名称：

| 字段 | 设计 |
| --- | --- |
| `article_id` | 原始商品 ID，按字符串保存并保留前导 0 |
| `article_node_id` | 图内商品节点 ID，格式为 `article_<article_id>` |
| `product_code` | 款式族编码，用于同款不同色等扩展分析 |
| `prod_name` | 商品名称，用于展示和案例解释 |

属性节点由 10 个结构化商品属性列聚合得到。属性节点 ID 使用稳定可读规则：

```text
attr_id = attr_type + "::" + attr_value
```

例如 `colour_group_name::Black`、`product_type_name::Trousers`。属性节点字段设计：

| 字段 | 设计 |
| --- | --- |
| `attr_id` | 属性节点唯一 ID，格式为 `attr_type::attr_value` |
| `attr_type` | 属性类型，即来源字段名 |
| `attr_value` | 属性取值 |
| `attr_node_id` | 图中属性节点 ID，当前与 `attr_id` 一致 |
| `article_count` | 关联到该属性的商品数量 |
| `is_core_attr` | 核心商品属性为 1，层级增强属性为 0 |
| `level` | 属性在层级图中的角色：`parent`、`child`、`parent_child` 或 `flat` |

核心属性包括 `product_group_name`、`product_type_name`、`garment_group_name`、`colour_group_name`、`graphical_appearance_name`。层级增强属性包括 `perceived_colour_master_name`、`index_group_name`、`index_name`、`section_name`、`department_name`。

### 5.2 边设计

商品-属性边把每个商品展开到 10 个属性节点，是后续把商品周销量映射为属性周热度的核心桥梁：

| 字段 | 设计 |
| --- | --- |
| `article_id` | 原始商品 ID |
| `article_node_id` | 商品节点 ID |
| `attr_id` | 属性节点 ID |
| `attr_type` | 属性类型 |
| `attr_value` | 属性取值 |
| `edge_type` | 商品到属性的边类型，例如 `has_colour_group`、`has_product_type` |
| `edge_weight` | 当前固定为 1.0，表示商品具备该属性 |

属性层级边来自同一商品表中的父子属性共现关系，`edge_weight` 表示该父子属性组合在商品表中共现的商品数量：

| 父属性类型 | 子属性类型 | 关系类型 |
| --- | --- | --- |
| `product_group_name` | `product_type_name` | `product_group_contains_type` |
| `perceived_colour_master_name` | `colour_group_name` | `colour_master_contains_colour` |
| `index_group_name` | `index_name` | `index_group_contains_index` |
| `index_name` | `section_name` | `index_contains_section` |
| `section_name` | `department_name` | `section_contains_department` |

### 5.3 构建与校验逻辑

属性图由 `src/04_build_attribute_graph.py` 调用 `fashion_trend.catalog.graph` 构建。构建顺序为：

1. 校验 `articles_clean.csv` 中 `article_id` 唯一且必要属性列无缺失。
2. 构建商品节点表 `nodes_article.csv`。
3. 按属性类型聚合属性取值，构建 `nodes_attribute.csv`。
4. 将商品按 10 个属性列展开，构建 `edges_article_attribute.csv`。
5. 按固定父子属性关系聚合共现次数，构建 `edges_attribute_hierarchy.csv`。
6. 校验所有商品-属性边和属性层级边引用的节点都已经存在。

这一设计的论文价值在于：属性图把商品交易行为连接到结构化服装属性，使商品周销量可以聚合为属性周热度；同时，属性节点的 `article_count`、层级角色和层级边可转化为趋势模型的图结构特征，并为推荐解释提供“商品为什么被推荐”的属性依据。

## 6. 时间切分

趋势预测使用严格时间切分：

| domain | split | week_start | week_end | week_count | row_count | attribute_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| trend | train | 4 | 87 | 84 | 49,728 | 592 |
| trend | valid | 88 | 95 | 8 | 4,736 | 592 |
| trend | test | 96 | 103 | 8 | 4,736 | 592 |

推荐评价使用 16 个 user-window：

| domain | split | week_start | week_end | week_count | row_count | user_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| recommendation | valid | 88 | 96 | 9 | 8 | 385,258 |
| recommendation | test | 96 | 104 | 9 | 8 | 336,499 |

论文表述中应强调：

- 趋势预测按周时间顺序切分，valid/test 位于训练窗口之后。
- 推荐候选、用户画像、近期热门和趋势分只使用 `cutoff_week` 及之前信息。
- `label_week` 只用于离线评价，避免时间泄漏。

## 7. 趋势预测任务与特征

趋势预测的建模对象是属性级周趋势。核心标签为 `target_growth`，表示属性下一周趋势增长；核心评价同时关注数值误差和排序质量。

`trend_model_samples.parquet` 的主要特征组：

| 特征组 | 代表字段 | 说明 |
| --- | --- | --- |
| level | `heat_t`、`share_t`、`log_heat_t`、`rank_in_type_t` | 当前周热度、份额和同类型排名 |
| lag | `heat_lag_1` 到 `heat_lag_4`、`share_lag_1` 到 `share_lag_4` | 历史 1 到 4 周滞后特征 |
| growth | `growth_lag_1`、`growth_lag_2`、`acc_lag_1` | 历史增长率和增长加速度 |
| rolling | `heat_ma_4`、`share_ma_4`、`share_std_4`、`share_max_4`、`share_min_4` | 4 周滚动统计 |
| graph | `article_count`、`parent_count`、`child_count`、`degree`、`is_core_attr` | 属性图结构与属性覆盖规模 |
| history | `history_total_heat_t`、`history_active_weeks_t`、`is_trend_eligible_t` | 历史活跃度和趋势展示资格 |
| time | `week_index`、`week_mod_52` | 时间序号和年度周期位置 |
| categorical | `attr_type` | LightGBM 原生类别特征 |

## 8. 趋势模型与参数

当前趋势预测已实现 4 个模型：

| 模型 | 类型 | 算法思想 | 输出目录 |
| --- | --- | --- | --- |
| `last_week` | baseline | 用当前周同类型归一化热度作为下一周预测 | `outputs/models/last_week/` |
| `previous_growth` | baseline | 使用上一期增长率预测下一期增长 | `outputs/models/previous_growth/` |
| `moving_average` | baseline | 使用最近两期增长率均值做平滑预测 | `outputs/models/moving_average/` |
| `lightgbm` | 主模型 | 融合历史热度、增长率、图结构和时间特征预测 `target_growth` | `outputs/models/lightgbm/` |

LightGBM stable 参数：

| 参数 | 当前值 |
| --- | --- |
| objective | `regression_l1` |
| n_estimators | 300 |
| learning_rate | 0.05 |
| num_leaves | 31 |
| max_depth | 6 |
| min_child_samples | 40 |
| subsample | 0.8 |
| subsample_freq | 1 |
| colsample_bytree | 0.55 |
| reg_alpha | 0.0 |
| reg_lambda | 0.0 |
| min_split_gain | 0.0 |
| random_state | 42 |
| early stopping | `stopping_rounds=30` |
| best_iteration | 163 |
| target_column | `target_growth` |
| categorical_features | `attr_type` |

LightGBM normalized gain 重要性 Top-10：

| 排名 | 特征 | split importance | normalized gain |
| ---: | --- | ---: | ---: |
| 1 | `week_mod_52` | 679 | 0.197364 |
| 2 | `week_index` | 639 | 0.183256 |
| 3 | `history_total_heat_t` | 269 | 0.074568 |
| 4 | `rank_in_type_t` | 396 | 0.073904 |
| 5 | `article_count` | 370 | 0.058008 |
| 6 | `history_active_weeks_t` | 290 | 0.056126 |
| 7 | `growth_lag_1` | 212 | 0.041010 |
| 8 | `share_std_4` | 203 | 0.033379 |
| 9 | `growth_lag_2` | 239 | 0.032544 |
| 10 | `heat_ma_4` | 104 | 0.024821 |

## 9. 趋势预测指标

| model_name | split | MAE | RMSE | Spearman | NDCG@10 | Precision@10 | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `last_week` | valid | 0.191413 | 0.331036 | -0.040761 | 0.675373 | 0.420000 | 0.420000 |
| `last_week` | test | 0.227444 | 0.389658 | -0.038707 | 0.697259 | 0.436250 | 0.436250 |
| `previous_growth` | valid | 0.297339 | 0.523231 | -0.111956 | 0.625441 | 0.442500 | 0.442500 |
| `previous_growth` | test | 0.330640 | 0.599574 | 0.017277 | 0.653270 | 0.458750 | 0.458750 |
| `moving_average` | valid | 0.263523 | 0.439867 | -0.227924 | 0.600542 | 0.403750 | 0.403750 |
| `moving_average` | test | 0.286279 | 0.491744 | -0.010412 | 0.655005 | 0.443750 | 0.443750 |
| `lightgbm` | valid | 0.183269 | 0.310875 | 0.295685 | 0.818344 | 0.562500 | 0.562500 |
| `lightgbm` | test | 0.216160 | 0.359267 | 0.279048 | 0.810126 | 0.567500 | 0.567500 |

可写结论：

- LightGBM 在 valid/test 的 MAE、RMSE 和 NDCG@10 上均优于三个 baseline。
- LightGBM 在 valid/test 的 Spearman 均为正，三个 baseline 多数接近 0 或为负。
- 这支持“结构化属性趋势特征对趋势预测有效”的论文结论。

表述边界：

- 不要把 LightGBM 写成深度学习模型。
- 不要把趋势预测结果描述成生产级预测能力。
- `previous_growth` 和 `moving_average` 只是确定性 baseline，不是复杂时间序列模型。

## 10. 推荐输入、候选与方法

推荐模块对应题目中的“推荐”部分，用于在离线 Top-N 场景中融合用户行为、商品属性知识图谱和趋势预测信号。它依赖已发布的 LightGBM stable 预测、周级交易、用户历史画像、用户基础画像、商品款式族映射和商品属性边；实现目标是构造可解释、可复现的趋势感知推荐实验，而不是扩展为在线推荐服务。

推荐候选策略：

| strategy | 候选行数 | 每用户候选数 | 说明 |
| --- | ---: | --- | --- |
| `popularity` | 14,443,788 | 12 | 近期热门候选 |
| `similarity` | 14,443,788 | 12 | 用户属性相似候选 |
| `trend_union` | 14,443,788 | 12 | 趋势属性商品候选 |
| `default` | 43,331,363 | 35-36 | 热门、相似、趋势候选合并 |
| `enhanced_default` | 76,790,040 | 63-64 | 可选增强候选，加入复购、款式变体、年龄段和偏好热门等多源候选 |

推荐方法：

| 方法 | 类型 | 使用分数 | 是否使用趋势分 | 默认或实验产物 |
| --- | --- | --- | --- | --- |
| `global_popularity` | baseline | 全局热门 | 否 | `outputs/recommendation/global_popularity/` |
| `recent_popularity` | 强 baseline | 近期热门 | 否 | `outputs/recommendation/recent_popularity/` |
| `attribute_similarity` | baseline | 用户属性偏好相似度 | 否 | `outputs/recommendation/attribute_similarity/` |
| `pop_similarity` | 融合 baseline | `pop_score`、`sim_score` | 否 | `outputs/recommendation/pop_similarity/` |
| `pop_similarity_trend` | 趋势感知主方法 | `pop_score`、`sim_score`、`trend_score`、`recent_score` | 是 | `outputs/recommendation/pop_similarity_trend/` |
| `enhanced_pop_similarity_trend` | 可选增强实验方法 | 热门、近期、相似、趋势、复购、变体、年龄段、偏好热门和 source 质量分 | 是 | `outputs/recommendation/experiments/recommendation_enhanced/experiment.json` |

前 5 个 stable method 当前输出 `recommendations.csv` 1,203,649 行、`recommendation_items.parquet` 14,443,788 行。默认内部长表是 `recommendation_items.parquet`；历史存在的 `recommendation_items.csv` 不应作为默认 reader 来源。

`enhanced_pop_similarity_trend` 当前只作为 `recommendation_enhanced` 实验中的可选增强方法记录指标，仓库当前没有 `outputs/recommendation/enhanced_pop_similarity_trend/` stable 单方法输出目录。reports 和 defense app 默认仍消费 `outputs/recommendation/pop_similarity_trend/`，不受可选增强实验影响。

`pop_similarity_trend` 当前权重：

| 分数项 | 权重 |
| --- | ---: |
| `pop_score` | 0.2 |
| `sim_score` | 0.2 |
| `trend_score` | 0.1 |
| `recent_score` | 0.5 |

`recommendation_enhanced` 当前 best weights：

| 分数项 | 权重 |
| --- | ---: |
| `pop_score` | 0.16 |
| `recent_score` | 0.28 |
| `sim_score` | 0.10 |
| `trend_score` | 0.08 |
| `reorder_score` | 0.16 |
| `variant_score` | 0.08 |
| `age_pop_score` | 0.04 |
| `preference_pop_score` | 0.04 |
| `source_rank_score` | 0.03 |
| `source_count_score` | 0.03 |

## 11. 默认稳定推荐评价指标

下表只包含当前默认 stable method 的 `metrics.json` 指标；可选增强实验指标单独放在 12.1 节，避免把增强实验误写成默认主结果。

| method | split | MAP@12 | Recall@12 | HitRate@12 | NDCG@12 | Coverage | user_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attribute_similarity` | valid | 0.000058 | 0.000266 | 0.000926 | 0.000149 | 0.005460 | 660,710 |
| `attribute_similarity` | test | 0.000077 | 0.000343 | 0.001103 | 0.000188 | 0.004958 | 542,939 |
| `global_popularity` | valid | 0.002102 | 0.006871 | 0.021751 | 0.004402 | 0.000242 | 660,710 |
| `global_popularity` | test | 0.002364 | 0.006843 | 0.020430 | 0.004629 | 0.000224 | 542,939 |
| `pop_similarity` | valid | 0.002120 | 0.005688 | 0.020756 | 0.004269 | 0.004992 | 660,710 |
| `pop_similarity` | test | 0.002942 | 0.009438 | 0.027705 | 0.006002 | 0.004565 | 542,939 |
| `pop_similarity_trend` | valid | 0.002745 | 0.008566 | 0.030549 | 0.005922 | 0.000712 | 660,710 |
| `pop_similarity_trend` | test | 0.003703 | 0.013685 | 0.039141 | 0.007987 | 0.000665 | 542,939 |
| `recent_popularity` | valid | 0.002512 | 0.008691 | 0.031059 | 0.005715 | 0.000206 | 660,710 |
| `recent_popularity` | test | 0.003781 | 0.013939 | 0.039935 | 0.008087 | 0.000191 | 542,939 |

可写结论：

- `pop_similarity_trend` 在 valid split 上取得最高 NDCG@12：0.005922。
- `recent_popularity` 在 test split 上略高于 `pop_similarity_trend`：0.008087 vs 0.007987。
- `pop_similarity_trend` 相比 `pop_similarity` 有明显提升：test NDCG@12 从 0.006002 提升到 0.007987。
- `attribute_similarity` 覆盖率较高，但命中类指标很低，说明单纯属性相似不足以支撑推荐。

表述边界：

- 不要写趋势推荐全面超过近期热门 baseline。
- 不要把低绝对值 MAP@12/NDCG@12 包装成高业务性能。
- 推荐实验应写成题目中推荐任务的轻量离线实现，同时说明趋势预测在推荐排序中主要提供解释性补充信号，不把它包装成完整生产推荐系统或高性能竞赛方案。

## 12. 推荐主实验与消融

`main` 实验包含 25 组权重搜索、方法级对比、严格命名消融和 trend weight bucket 代表组合。

严格命名消融摘要：

| 版本 | split | MAP@12 | Recall@12 | HitRate@12 | NDCG@12 | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full Model | valid | 0.002745 | 0.008566 | 0.030549 | 0.005922 | 0.000712 |
| Full Model | test | 0.003703 | 0.013685 | 0.039141 | 0.007987 | 0.000665 |
| w/o Trend in Rec | valid | 0.002744 | 0.008572 | 0.030567 | 0.005923 | 0.000707 |
| w/o Trend in Rec | test | 0.003695 | 0.013685 | 0.039137 | 0.007977 | 0.000656 |
| w/o Similarity | valid | 0.002590 | 0.008585 | 0.030611 | 0.005757 | 0.000333 |
| w/o Similarity | test | 0.003635 | 0.013694 | 0.039178 | 0.007915 | 0.000265 |
| w/o Recent | valid | 0.001891 | 0.005037 | 0.018274 | 0.003787 | 0.005147 |
| w/o Recent | test | 0.002534 | 0.008045 | 0.024082 | 0.005157 | 0.004766 |
| Recent Only | valid | 0.002512 | 0.008691 | 0.031059 | 0.005715 | 0.000206 |
| Recent Only | test | 0.003781 | 0.013939 | 0.039935 | 0.008087 | 0.000191 |
| Pop + Similarity baseline | valid | 0.002120 | 0.005688 | 0.020756 | 0.004269 | 0.004992 |
| Pop + Similarity baseline | test | 0.002942 | 0.009438 | 0.027705 | 0.006002 | 0.004565 |

trend weight bucket 代表组合：

| trend_score | pop_score | sim_score | recent_score | valid NDCG@12 | test NDCG@12 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.4 | 0.1 | 0.5 | 0.005813 | 0.007879 |
| 0.1 | 0.2 | 0.2 | 0.5 | 0.005922 | 0.007987 |
| 0.2 | 0.2 | 0.1 | 0.5 | 0.005869 | 0.007982 |
| 0.3 | 0.2 | 0.0 | 0.5 | 0.005775 | 0.007916 |
| 0.4 | 0.3 | 0.0 | 0.3 | 0.005712 | 0.007854 |

消融结论应谨慎写：

- 严格 `w/o Trend in Rec` 与 Full Model 基本持平，趋势分在当前推荐实验中的独立边际贡献较弱。
- 去掉 `recent_score` 后指标下降明显，说明近期热门是 H&M 短期推荐中的强信号。
- 趋势分更适合作为可解释的轻量补充信号，而不是单独决定推荐排序的主因。

### 12.1 可选增强推荐实验

`recommendation_enhanced` 是第一阶段推荐增强的独立可选实验，只写入 `outputs/recommendation/experiments/recommendation_enhanced/experiment.json`。它使用 `enhanced_default` 候选和 `enhanced_pop_similarity_trend` 方法，不覆盖 `experiments/main/experiment.json`、`outputs/recommendation/pop_similarity_trend/`、reports 或 defense app 默认输入。

增强实验主结果：

| method | split | MAP@12 | Recall@12 | HitRate@12 | NDCG@12 | Coverage | user_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `enhanced_pop_similarity_trend` | valid | 0.006618 | 0.025382 | 0.065545 | 0.014079 | 0.146255 | 660,710 |
| `enhanced_pop_similarity_trend` | test | 0.007032 | 0.027712 | 0.066308 | 0.014800 | 0.088785 | 542,939 |

增强实验 valid 消融摘要：

| 版本 | MAP@12 | Recall@12 | HitRate@12 | NDCG@12 | Coverage | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full Model | 0.006618 | 0.025382 | 0.065545 | 0.014079 | 0.146255 | 完整增强候选和增强打分 |
| enhanced_w/o Trend Score | 0.006609 | 0.025333 | 0.065399 | 0.014056 | 0.145570 | 只去掉趋势打分，指标基本持平 |
| enhanced_w/o Trend Source+Score | 0.006747 | 0.025753 | 0.066108 | 0.014293 | 0.150395 | 同时去掉趋势候选源和趋势打分，valid 指标略高 |
| enhanced_w/o Customer Segment | 0.007584 | 0.028748 | 0.071470 | 0.015818 | 0.195626 | 去掉年龄段热门源后 valid 指标更高 |

增强实验结论应作为扩展结果谨慎表述：

- 相比默认 stable `pop_similarity_trend`，增强实验的 MAP@12、NDCG@12、Recall@12 和 Coverage 都明显更高，但它同时扩大了候选集并增加多源重排特征，因此应写成“可选增强实验整体有效”，不要写成单一趋势信号带来的提升。
- 去掉趋势打分几乎不影响指标，同时去掉趋势候选源和趋势打分后 valid 指标略高，说明增强版趋势来源的独立正贡献还没有被该实验严格证明。
- 去掉年龄段热门源后 valid 指标更高，说明当前用户年龄段增强特征还不适合作为论文中的强贡献点，可放入后续优化或误差分析。

## 13. 已导出的论文素材

`outputs/reports/manifest.json` 当前记录：

| 项目 | 数量或取值 |
| --- | --- |
| schema_version | `paper_assets_manifest/v1` |
| generated_at | `2026-05-13T13:14:09.167113+00:00` |
| figure_count | 16 |
| table_count | 16 |
| case_count | 3 |
| figure_formats | `svg`、`png` |
| selected_font | `Songti SC` |
| top_k | 10 |
| trend_week | 103 |

8 类核心图表均已导出 SVG 和 PNG：

| 图表 | 文件前缀 | 论文用途 |
| --- | --- | --- |
| 数据处理流程图 | `data_pipeline` | 介绍整体 pipeline |
| 属性层次图示意图 | `attribute_graph_schema` | 说明商品-属性图建模 |
| 典型趋势属性曲线 | `trend_curve_examples` | 展示趋势预测案例 |
| LightGBM 特征重要性 | `lightgbm_feature_importance` | 解释主模型关键特征 |
| 趋势模型指标对比 | `trend_model_metrics` | 对比 baseline 和 LightGBM |
| 推荐方法指标对比 | `recommendation_method_metrics` | 对比 Top-N 方法 |
| Top-K 趋势属性榜 | `topk_trend_attributes` | 展示预测趋势属性 |
| 推荐权重分析 | `recommendation_weight_analysis` | 展示权重搜索和趋势权重影响 |

8 类表格均已导出 CSV 和 Markdown：

| 表格 | 文件前缀 | 论文用途 |
| --- | --- | --- |
| 数据 artifact 汇总 | `data_artifact_summary` | 数据规模表 |
| 时间切分汇总 | `time_split_summary` | 训练/验证/测试切分表 |
| 属性图汇总 | `attribute_graph_summary` | 节点与边统计 |
| 趋势特征汇总 | `trend_feature_summary` | 特征设计说明 |
| 趋势模型指标 | `trend_model_metrics` | 趋势实验主表 |
| 按属性类型趋势指标 | `trend_metrics_by_attr_type` | 分类型误差和排序分析 |
| 推荐方法指标 | `recommendation_method_metrics` | 推荐实验主表 |
| 推荐实验汇总 | `recommendation_experiment_summary` | 权重搜索、消融和趋势权重分析 |

推荐解释案例：

| 案例 | JSON | Markdown |
| --- | --- | --- |
| case 01 | `outputs/reports/case_studies/case_01.json` | `outputs/reports/case_studies/case_01.md` |
| case 02 | `outputs/reports/case_studies/case_02.json` | `outputs/reports/case_studies/case_02.md` |
| case 03 | `outputs/reports/case_studies/case_03.json` | `outputs/reports/case_studies/case_03.md` |

## 14. 本地答辩展示应用数据

答辩展示应用是只读展示层，依赖已构建的 SQLite 展示库，不直接读取原始 H&M 数据、上游 Parquet/CSV、训练 run 或历史推荐 CSV。

| 组成 | 路径 | 说明 |
| --- | --- | --- |
| SQLite 展示库 | `outputs/defense_app/fashion_demo.sqlite` | 生成产物，不提交 |
| 静态 reports 资产 | `outputs/defense_app/static/reports/` | 复制论文图表 SVG/PNG |
| 后端 | `apps/defense_app/backend/` | FastAPI 只读查询层 |
| 前端 | `apps/defense_app/frontend/` | Vue/Vite 桌面展示界面 |

展示页面覆盖趋势看板、属性详情、属性图展示、推荐展示和推荐理由。论文或答辩文案中应称为“本地只读答辩展示应用”，不要描述成在线服务或生产系统。

## 15. 可直接写入论文的核心观点

1. 本项目围绕《融合用户行为与知识图谱的服装流行趋势预测与推荐》展开，用户交易记录提供行为信号，商品属性层次图提供知识图谱语义。
2. 本文不是复现 H&M Kaggle 高分方案，而是在该数据集上构建可解释、可复现的服装流行趋势预测与推荐实验闭环。
3. 商品属性层次图将商品和结构化属性连接起来，为趋势特征、属性热度聚合和推荐解释提供统一语义层。
4. 周级属性热度、趋势标签和时间切分构成了可复现的服装流行趋势预测实验设置。
5. LightGBM 融合用户行为聚合后的历史热度、增长、图结构和时间特征，在趋势预测上显著优于简单 baseline。
6. 默认稳定推荐实验中，`pop_similarity_trend` 在验证集上取得最高 NDCG@12，并在测试集上接近强近期热门 baseline。
7. 可选增强推荐实验在 MAP@12、NDCG@12、Recall@12 和 Coverage 上明显高于默认 stable 推荐方法，但它不是 reports 和 defense app 的默认输入，应作为扩展实验单独表述。
8. 严格推荐层消融显示趋势分的独立边际增益较弱，但它能作为解释性补充信号服务于推荐展示。
9. 当前 reports 和 defense app 已覆盖论文图表、实验表格和可复现推荐解释案例。

## 16. 当前缺口与风险

| 项目 | 当前状态 | 论文处理建议 |
| --- | --- | --- |
| 趋势模型特征消融 | `w/o Graph`、`w/o Growth`、`w/o Rank` 尚未训练独立 LightGBM 变体 | 如果论文必须写趋势模型特征消融，需要额外设计并重训；否则不要声称已完成 |
| 推荐指标绝对值 | MAP@12 和 NDCG@12 绝对值较低 | 写成应用验证和解释性实验，不写成高性能推荐系统 |
| 近期热门 baseline | `recent_popularity` 在 test 上略优于趋势主方法 | 如实报告，强调短期热门信号强 |
| 趋势分边际贡献 | `w/o Trend in Rec` 与 Full Model 基本持平 | 不夸大趋势分的单因素效果 |
| 可选增强实验边界 | `recommendation_enhanced` 独立写入 experiment，不替换 stable 主推荐输出 | 作为扩展实验写，默认论文图表和展示应用仍以 stable method 为准 |
| 增强趋势来源贡献 | `enhanced_w/o Trend Source+Score` 的 valid 指标略高于 Full Model | 不声称增强趋势候选源已证明单独有效，只写整体增强候选和多源重排效果 |
| 用户年龄段增强 | `enhanced_w/o Customer Segment` 的 valid 指标高于 Full Model | 不把年龄段热门源写成已验证强特征，可列入后续优化 |
| 增强单方法 stable 输出 | 当前不存在 `outputs/recommendation/enhanced_pop_similarity_trend/` 目录 | 指标以 `experiments/recommendation_enhanced/experiment.json` 为准 |
| 历史 CSV 长表 | 本地存在历史 `recommendation_items.csv` | 论文和代码契约以 `recommendation_items.parquet` 为默认内部长表 |
| 图表字体 | reports 依赖可用 CJK 字体 | 重新导出前确保本机有可用中文字体；当前 manifest 记录 `Songti SC` |

## 17. 复现实验与导出命令

完整流水线入口按编号脚本组织。论文收尾阶段通常不需要重跑全部原始数据处理；如需刷新稳定产物，可按依赖顺序运行。

趋势训练与评价：

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model lightgbm
```

默认稳定推荐实验：

```sh
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy default
uv run python src/14_rerank_recommendations.py --method pop_similarity_trend
uv run python src/15_eval_recommendations.py --method pop_similarity_trend
uv run python src/16_run_recommendation_experiment.py --experiment main
```

可选增强推荐实验：

```sh
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy enhanced_default
uv run python src/14_rerank_recommendations.py --method enhanced_pop_similarity_trend
uv run python src/15_eval_recommendations.py --method enhanced_pop_similarity_trend
uv run python src/16_run_recommendation_experiment.py --experiment recommendation_enhanced
```

论文素材与展示库：

```sh
uv run python src/17_export_paper_assets.py
uv run python src/18_build_defense_app_db.py
```

相关聚焦验证：

```sh
uv run pytest tests/test_reports_*.py tests/test_architecture_boundaries.py
uv run pytest tests/test_recommendation_*.py tests/test_architecture_boundaries.py
uv run pytest tests/test_presentation_*.py tests/test_architecture_boundaries.py
uv run --group app pytest apps/defense_app/backend/tests
npm --prefix apps/defense_app/frontend run typecheck
npm --prefix apps/defense_app/frontend run build
```
