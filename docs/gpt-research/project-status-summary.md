# 项目现状与论文素材汇总

更新时间：2026-05-13

## 1. 总体结论

当前项目已经完成从 H&M 原始交易数据到属性周级趋势预测，再到轻量 Top-N 推荐实验、论文素材导出和本地答辩展示应用的主要闭环。项目主线与 `docs/gpt-research/implementation-plan.md` 保持一致：

```text
H&M articles.csv
    -> 商品属性层次图
H&M transactions_train.csv
    -> 周级商品销量
    -> 属性周热度
    -> 属性趋势预测
    -> 趋势感知 Top-N 推荐
```

论文应把核心贡献表述为：将 H&M 个性化推荐数据集重新组织为服装属性周级趋势预测任务，并用轻量推荐实验验证趋势预测结果的应用价值。推荐模块是应用验证层，不是完整生产推荐系统，也不是深度召回或在线服务。

当前状态可以开始论文撰写和答辩材料整理。方法设计、数据处理、趋势预测实验、推荐应用验证、静态论文素材和本地展示应用都已有可引用产物。后续更适合做论文排版和文字整理，而不是继续扩大模型范围。

## 2. 论文需要覆盖的内容

论文建议围绕以下章节组织：

| 章节 | 可写内容 | 当前材料状态 |
| --- | --- | --- |
| 任务定义 | 属性级周趋势预测任务；趋势感知 Top-N 推荐任务 | 已具备 |
| 数据预处理 | H&M 三张核心表、ID 保留、周切分、时间泄漏规避 | 已具备 |
| 商品属性层次图 | 商品节点、属性节点、商品-属性边、属性层级边 | 已具备 |
| 属性周热度 | `heat_cnt`、`heat_share`、`log_heat`、同类型归一化 | 已具备 |
| 趋势标签与样本 | `target_growth`、lag 特征、rolling 特征、图结构特征、时间切分 | 已具备 |
| 趋势模型 | 三类 baseline 与 LightGBM 主模型 | 已具备 |
| 趋势评价 | MAE、RMSE、Spearman、Precision@K、Recall@K、NDCG@K | 已具备 |
| 推荐模块 | 候选召回、用户属性偏好、趋势分、线性重排序 | 已具备 |
| 推荐评价 | MAP@12、Recall@12、HitRate@12、NDCG@12、Coverage | 已具备 |
| 结果分析 | LightGBM 优于趋势 baseline；趋势推荐与强热门 baseline 的对照 | 已具备 |
| 案例展示 | Top-K 趋势属性、用户 Top-12 推荐、推荐解释 | 已补文本案例、静态图表和本地展示应用 |

## 3. 关键数据与产物

### 3.1 数据处理产物

| 产物 | 路径 | 行数 | 论文用途 |
| --- | --- | ---: | --- |
| 周级交易表 | `data/interim/transactions_train_weekly.parquet` | 31,788,324 | 说明日交易到周交易的转换结果 |
| 商品周销量 | `data/processed/trend/article_week_sales.csv` | 2,203,988 | 商品销量到属性热度的上游输入 |
| 属性周热度 | `data/processed/trend/attribute_week_heat.csv` | 62,160 | 属性动态热度序列核心表 |
| 趋势标签 | `data/processed/trend/attribute_week_target.csv` | 61,568 | 趋势预测标签表 |
| 趋势样本 | `data/processed/features/trend_model_samples.parquet` | 59,200 | 趋势模型训练与评价样本 |
| train split | `data/processed/features/trend_model_samples_train.parquet` | 49,728 | 模型训练 |
| valid split | `data/processed/features/trend_model_samples_valid.parquet` | 4,736 | 调参和模型选择 |
| test split | `data/processed/features/trend_model_samples_test.parquet` | 4,736 | 最终测试评价 |
| 推荐时间窗口 | `data/processed/recommend/time_windows.parquet` | 16 | 推荐 valid/test 周窗口 |
| 推荐目标用户 | `data/processed/recommend/target_users.parquet` | 1,203,649 | 推荐评价用户集合 |
| 推荐评价标签 | `data/processed/recommend/evaluation_labels.parquet` | 4,195,886 | 用户未来购买标签 |
| 用户属性画像 | `data/processed/recommend/user_profile.parquet` | 3,610,947 | 属性相似推荐与解释 |

### 3.2 原始数据 profile

当前原始数据基础统计来自 `data/processed/basic/data_profile.csv` 和
`data/processed/basic/date_range.json`。

| 原始表 | 行数 | 列数 | 缺失单元格数 | 关键唯一值 |
| --- | ---: | ---: | ---: | --- |
| `transactions_train.csv` | 31,788,324 | 5 | 0 | `customer_id=1,362,281`；`article_id=104,547` |
| `articles.csv` | 105,542 | 25 | 416 | `article_id=105,542`；`product_code=47,224` |
| `customers.csv` | 1,371,980 | 7 | 1,840,560 | `customer_id=1,371,980`；`postal_code=352,899` |

原始交易日期范围：

| 字段 | 最早日期 | 最晚日期 | 覆盖天数 |
| --- | --- | --- | ---: |
| `t_dat` | 2018-09-20 | 2020-09-22 | 734 |

交易表字段缺失率：

| 字段 | 缺失数 | 缺失率 |
| --- | ---: | ---: |
| `t_dat` | 0 | 0.000000 |
| `customer_id` | 0 | 0.000000 |
| `article_id` | 0 | 0.000000 |
| `price` | 0 | 0.000000 |
| `sales_channel_id` | 0 | 0.000000 |

商品表字段缺失率：

| 字段 | 缺失数 | 缺失率 |
| --- | ---: | ---: |
| `article_id` | 0 | 0.000000 |
| `product_code` | 0 | 0.000000 |
| `prod_name` | 0 | 0.000000 |
| `product_type_no` | 0 | 0.000000 |
| `product_type_name` | 0 | 0.000000 |
| `product_group_name` | 0 | 0.000000 |
| `graphical_appearance_no` | 0 | 0.000000 |
| `graphical_appearance_name` | 0 | 0.000000 |
| `colour_group_code` | 0 | 0.000000 |
| `colour_group_name` | 0 | 0.000000 |
| `perceived_colour_value_id` | 0 | 0.000000 |
| `perceived_colour_value_name` | 0 | 0.000000 |
| `perceived_colour_master_id` | 0 | 0.000000 |
| `perceived_colour_master_name` | 0 | 0.000000 |
| `department_no` | 0 | 0.000000 |
| `department_name` | 0 | 0.000000 |
| `index_code` | 0 | 0.000000 |
| `index_name` | 0 | 0.000000 |
| `index_group_no` | 0 | 0.000000 |
| `index_group_name` | 0 | 0.000000 |
| `section_no` | 0 | 0.000000 |
| `section_name` | 0 | 0.000000 |
| `garment_group_no` | 0 | 0.000000 |
| `garment_group_name` | 0 | 0.000000 |
| `detail_desc` | 416 | 0.003942 |

用户表字段缺失率：

| 字段 | 缺失数 | 缺失率 |
| --- | ---: | ---: |
| `customer_id` | 0 | 0.000000 |
| `FN` | 895,050 | 0.652378 |
| `Active` | 907,576 | 0.661508 |
| `club_member_status` | 6,062 | 0.004418 |
| `fashion_news_frequency` | 16,011 | 0.011670 |
| `age` | 15,861 | 0.011561 |
| `postal_code` | 0 | 0.000000 |

连接覆盖情况：

| 检查项 | 数值 | 结论 |
| --- | ---: | --- |
| 交易中出现的唯一 `article_id` | 104,547 | 全部可在 `articles.csv` 找到 |
| 商品表唯一 `article_id` | 105,542 | 其中 995 个商品未出现在交易中 |
| 交易唯一商品连接成功率 | 100.000000% | 交易商品维表连接完整 |
| 商品被交易覆盖率 | 99.057247% | 绝大多数商品有交易记录 |
| 交易中出现的唯一 `customer_id` | 1,362,281 | 全部可在 `customers.csv` 找到 |
| 用户表唯一 `customer_id` | 1,371,980 | 其中 9,699 个用户未出现在交易中 |
| 交易唯一用户连接成功率 | 100.000000% | 交易用户维表连接完整 |
| 用户被交易覆盖率 | 99.293065% | 绝大多数用户有交易记录 |

### 3.3 时间切分

趋势预测采用严格时间切分：

| split | 周范围 | 周数 | 行数 | 属性数 |
| --- | --- | ---: | ---: | ---: |
| train | 4-87 | 84 | 49,728 | 592 |
| valid | 88-95 | 8 | 4,736 | 592 |
| test | 96-103 | 8 | 4,736 | 592 |

推荐评价使用 16 个窗口：

| split | cutoff_week | label_week | 含义 |
| --- | --- | --- | --- |
| valid | 88-95 | 89-96 | 用于选择推荐权重 |
| test | 96-103 | 97-104 | 用于最终推荐评价 |

所有推荐窗口均满足 `cutoff_week < label_week`。候选、用户画像和热门分只能使用 `cutoff_week` 及之前的信息，`label_week` 只用于评价。

## 4. 属性图与属性热度现状

属性图产物来自 `data/processed/graph/`，当前精确规模如下：

| 图元素 | 路径 | 数量 |
| --- | --- | ---: |
| 商品节点 | `data/processed/graph/nodes_article.csv` | 105,542 |
| 属性节点 | `data/processed/graph/nodes_attribute.csv` | 592 |
| 商品-属性边 | `data/processed/graph/edges_article_attribute.csv` | 1,055,420 |
| 属性层级边 | `data/processed/graph/edges_attribute_hierarchy.csv` | 658 |

各属性类型节点数量：

| 属性类型 | 节点数 |
| --- | ---: |
| `colour_group_name` | 50 |
| `department_name` | 250 |
| `garment_group_name` | 21 |
| `graphical_appearance_name` | 30 |
| `index_group_name` | 5 |
| `index_name` | 10 |
| `perceived_colour_master_name` | 20 |
| `product_group_name` | 19 |
| `product_type_name` | 131 |
| `section_name` | 56 |

商品-属性边按属性类型分布。当前每个商品连接 10 个属性字段，因此每类边都是 105,542 条：

| 属性类型 | 商品-属性边数量 |
| --- | ---: |
| `colour_group_name` | 105,542 |
| `department_name` | 105,542 |
| `garment_group_name` | 105,542 |
| `graphical_appearance_name` | 105,542 |
| `index_group_name` | 105,542 |
| `index_name` | 105,542 |
| `perceived_colour_master_name` | 105,542 |
| `product_group_name` | 105,542 |
| `product_type_name` | 105,542 |
| `section_name` | 105,542 |

属性层级边按关系类型分布：

| 父属性类型 | 子属性类型 | 关系类型 | 边数量 |
| --- | --- | --- | ---: |
| `index_group_name` | `index_name` | `index_group_contains_index` | 10 |
| `index_name` | `section_name` | `index_contains_section` | 64 |
| `perceived_colour_master_name` | `colour_group_name` | `colour_master_contains_colour` | 153 |
| `product_group_name` | `product_type_name` | `product_group_contains_type` | 132 |
| `section_name` | `department_name` | `section_contains_department` | 299 |

属性周热度表覆盖同样 10 类属性：

| 属性类型 |
| --- |
| `colour_group_name` |
| `department_name` |
| `garment_group_name` |
| `graphical_appearance_name` |
| `index_group_name` |
| `index_name` |
| `perceived_colour_master_name` |
| `product_group_name` |
| `product_type_name` |
| `section_name` |

关键数据检查结论：

| 检查项 | 结果 |
| --- | --- |
| `article_id` 字符串宽度 | 10 位，前导 0 保留 |
| `customer_id` 字符串宽度 | 64 位，字符串语义保留 |
| `article_week_sales` 重复键 | 0 |
| `attribute_week_heat` 重复键 | 0 |
| `attribute_week_target` 重复键 | 0 |
| `trend_model_samples` 重复键 | 0 |
| `heat_share` 范围 | 0.000000 到 0.718808 |
| `heat_share` 同周同类型求和 | 约等于 1.0 |
| `target_growth` 范围 | -4.796881 到 6.399516 |

这些检查说明当前产物可以支撑论文中的数据可靠性描述。

## 5. 趋势预测实现

### 5.1 模型与输出

趋势预测统一使用：

```sh
uv run python src/10_train_trend_model.py --model <model>
uv run python src/11_eval_trend_model.py --model <model>
```

当前已实现模型：

| 模型 | 类型 | 主要思想 | 输出路径 |
| --- | --- | --- | --- |
| `last_week` | baseline | 用当前周同类型归一化热度作为下一周预测 | `outputs/models/last_week/` |
| `previous_growth` | baseline | 使用上一期增长率预测下一期增长 | `outputs/models/previous_growth/` |
| `moving_average` | baseline | 使用最近两期增长率均值作为平滑预测 | `outputs/models/moving_average/` |
| `lightgbm` | 主模型 | 使用历史热度、增长率、图结构和时间特征预测 `target_growth` | `outputs/models/lightgbm/` |

每个模型的标准预测输出均为：

```text
outputs/models/<model>/predictions.csv
outputs/models/<model>/params.json
outputs/models/<model>/metadata.json
```

LightGBM 额外输出：

```text
outputs/models/lightgbm/feature_importance.csv
outputs/models/lightgbm/model.txt
```

当前四个模型的 `predictions.csv` 均为 59,200 行，并包含 train、valid、test 三个 split。趋势评价默认只报告 valid 和 test。

### 5.2 LightGBM 最终参数

当前 stable LightGBM 参数来自 `outputs/models/lightgbm/params.json`。默认训练会优先复用该 stable 参数文件。

| 参数 | 当前值 |
| --- | ---: |
| `objective` | `regression_l1` |
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `max_depth` | 6 |
| `min_child_samples` | 40 |
| `subsample` | 0.8 |
| `subsample_freq` | 1 |
| `colsample_bytree` | 0.55 |
| `reg_alpha` | 0.0 |
| `reg_lambda` | 0.0 |
| `min_split_gain` | 0.0 |
| `random_state` | 42 |
| `verbosity` | -1 |
| early stopping | `stopping_rounds=30` |
| `best_iteration` | 163 |

训练目标为 `target_growth`。从预测增长率反推 `pred_share_t1` 时使用 `epsilon=1e-6`，并在 `split + week_id + attr_type` 内归一化。

### 5.3 完整特征列清单

LightGBM 最终使用 30 个数值特征和 1 个类别特征。类别特征只包含 `attr_type`，使用 LightGBM 原生 categorical feature；训练时由 train split 固定类别集合，valid/test 复用同一类别集合以避免编码漂移。

当前热度特征：

| 特征 | 含义 |
| --- | --- |
| `heat_t` | 当前周属性热度计数 |
| `share_t` | 当前周同类型归一化热度份额 |
| `log_heat_t` | 当前周 log 热度 |
| `rank_in_type_t` | 当前周同类型属性热度排名 |

lag 特征：

| 特征 | 含义 |
| --- | --- |
| `heat_lag_1` | 前 1 周热度 |
| `heat_lag_2` | 前 2 周热度 |
| `heat_lag_3` | 前 3 周热度 |
| `heat_lag_4` | 前 4 周热度 |
| `share_lag_1` | 前 1 周同类型份额 |
| `share_lag_2` | 前 2 周同类型份额 |
| `share_lag_3` | 前 3 周同类型份额 |
| `share_lag_4` | 前 4 周同类型份额 |

增长和趋势变化特征：

| 特征 | 含义 |
| --- | --- |
| `growth_lag_1` | 前一期增长率 |
| `growth_lag_2` | 前两期增长率 |
| `acc_lag_1` | 增长变化加速度 |

rolling 特征：

| 特征 | 含义 |
| --- | --- |
| `heat_ma_4` | 最近 4 周热度均值 |
| `share_ma_4` | 最近 4 周份额均值 |
| `share_std_4` | 最近 4 周份额标准差 |
| `share_max_4` | 最近 4 周份额最大值 |
| `share_min_4` | 最近 4 周份额最小值 |

图结构与历史活跃度特征：

| 特征 | 含义 |
| --- | --- |
| `article_count` | 关联该属性的商品数 |
| `is_core_attr` | 是否属于核心属性类型 |
| `parent_count` | 属性层级图中的父节点数量 |
| `child_count` | 属性层级图中的子节点数量 |
| `degree` | 属性层级图度数 |
| `history_total_heat_t` | 截止当前周的历史累计热度 |
| `history_active_weeks_t` | 截止当前周的历史活跃周数 |
| `is_trend_eligible_t` | 是否满足趋势展示/评价资格 |

时间特征：

| 特征 | 含义 |
| --- | --- |
| `week_index` | 周序号 |
| `week_mod_52` | 52 周周期位置 |

类别特征：

| 特征 | 编码方式 |
| --- | --- |
| `attr_type` | LightGBM 原生 categorical feature |

不进入模型特征矩阵的列：

| 列 | 用途 |
| --- | --- |
| `attr_id` | 标识列，仅用于对齐和输出 |
| `attr_value` | 展示列，仅用于输出解释 |
| `week_id` | 时间标识列，不直接输入模型；由 `week_index` 和 `week_mod_52` 表达时间 |
| `split` | 切分标识，仅用于训练/评价边界 |
| `target_growth` | 标签列 |
| `target_log_heat_t1` | 辅助标签列 |
| `target_rank_in_type_t1` | 真实下一周排名列，用于评价和输出 |

LightGBM gain 重要性最高的前 10 个特征：

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

### 5.4 趋势预测指标

| 模型 | split | MAE | RMSE | Spearman | NDCG@10 | Precision@10 | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `last_week` | valid | 0.191413 | 0.331036 | -0.040761 | 0.675373 | 0.420000 | 0.420000 |
| `last_week` | test | 0.227444 | 0.389658 | -0.038707 | 0.697259 | 0.436250 | 0.436250 |
| `previous_growth` | valid | 0.297339 | 0.523231 | -0.111956 | 0.625441 | 0.442500 | 0.442500 |
| `previous_growth` | test | 0.330640 | 0.599574 | 0.017277 | 0.653270 | 0.458750 | 0.458750 |
| `moving_average` | valid | 0.263523 | 0.439867 | -0.227924 | 0.600542 | 0.403750 | 0.403750 |
| `moving_average` | test | 0.286279 | 0.491744 | -0.010412 | 0.655005 | 0.443750 | 0.443750 |
| `lightgbm` | valid | 0.183269 | 0.310875 | 0.295685 | 0.818344 | 0.562500 | 0.562500 |
| `lightgbm` | test | 0.216160 | 0.359267 | 0.279048 | 0.810126 | 0.567500 | 0.567500 |

### 5.5 趋势预测分析

LightGBM 是当前趋势预测主模型，主要证据如下：

- 在 valid split 上，LightGBM 的 NDCG@10 为 0.818344，高于最强 baseline `last_week` 的 0.675373。
- 在 test split 上，LightGBM 的 NDCG@10 为 0.810126，高于 `last_week` 的 0.697259。
- LightGBM 在 valid 和 test 上的 MAE 都低于三个 baseline。
- 三个 baseline 的 Spearman 多数为负或接近 0，说明仅靠历史热度或增长率难以稳定捕捉趋势排序；LightGBM 在 valid/test 上均为正相关。

论文中可以写：

> 相比仅使用上一周热度或上一期增长率的简单基线，LightGBM 通过融合历史热度、增长率、图结构和时间特征，在数值误差和排序质量上均取得更稳定表现，说明构造属性级趋势样本具有实际预测价值。

需要避免的表述：

- 不要说趋势预测已经达到生产级预测能力。
- 不要把 `previous_growth` 或 `moving_average` 解释为复杂时间序列模型。
- 不要把 LightGBM 写成深度学习模型。

### 5.6 Top-K 趋势属性案例

以下榜单取 test split 的 `week_id=103`，即基于第 103 周特征预测第 104 周趋势。为了避免低频属性造成极端增长率，榜单过滤条件为：

```text
is_trend_eligible_t = 1
heat_t >= 20
history_total_heat_t >= 100
history_active_weeks_t >= 8
```

上升颜色 Top-10：

| 排名 | 属性值 | heat_t | share_t | pred_growth | true_growth | pred_share_t1 | true_rank_t1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Light Green | 621 | 0.002581 | 0.002278 | -0.064504 | 0.002627 | 35 |
| 2 | Light Yellow | 587 | 0.002439 | 0.002180 | -0.076354 | 0.002483 | 36 |
| 3 | Green | 2,795 | 0.011615 | 0.000582 | 0.110256 | 0.011805 | 16 |
| 4 | Beige | 14,913 | 0.061971 | -0.000746 | 0.016324 | 0.062902 | 3 |
| 5 | Dark Red | 2,614 | 0.010863 | -0.006373 | 0.026070 | 0.010964 | 19 |
| 6 | Greenish Khaki | 4,002 | 0.016630 | -0.009220 | -0.078861 | 0.016738 | 14 |
| 7 | Dark Yellow | 442 | 0.001837 | -0.009628 | 0.507678 | 0.001848 | 33 |
| 8 | Pink | 4,408 | 0.018318 | -0.009888 | -0.150777 | 0.018423 | 13 |
| 9 | Turquoise | 543 | 0.002256 | -0.009983 | -0.019427 | 0.002269 | 37 |
| 10 | Dark Grey | 7,865 | 0.032683 | -0.010643 | 0.122977 | 0.032847 | 7 |

上升品类 Top-10：

| 排名 | 属性值 | heat_t | share_t | pred_growth | true_growth | pred_share_t1 | true_rank_t1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Sunglasses | 248 | 0.001031 | 0.046328 | -0.099328 | 0.001095 | 50 |
| 2 | Umbrella | 38 | 0.000158 | 0.032033 | -0.219499 | 0.000165 | 75 |
| 3 | Tie | 22 | 0.000091 | 0.032009 | -0.025625 | 0.000096 | 85 |
| 4 | Sneakers | 446 | 0.001853 | 0.012386 | 0.014329 | 0.001904 | 40 |
| 5 | Jumpsuit/Playsuit | 539 | 0.002240 | 0.005189 | -0.089046 | 0.002284 | 37 |
| 6 | Sandals | 143 | 0.000594 | 0.002544 | -0.132382 | 0.000604 | 59 |
| 7 | Hat/brim | 105 | 0.000436 | 0.001850 | 0.647426 | 0.000443 | 54 |
| 8 | Bag | 1,362 | 0.005660 | 0.000278 | 0.148188 | 0.005743 | 22 |
| 9 | Swimsuit | 554 | 0.002302 | 0.000251 | -0.190026 | 0.002336 | 39 |
| 10 | Other accessories | 642 | 0.002668 | -0.001704 | 0.085795 | 0.002702 | 34 |

上升图案 Top-10：

| 排名 | 属性值 | heat_t | share_t | pred_growth | true_growth | pred_share_t1 | true_rank_t1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Neps | 38 | 0.000158 | 0.000581 | -0.378480 | 0.000160 | 27 |
| 2 | Lace | 3,460 | 0.014378 | -0.003349 | -0.080998 | 0.014537 | 9 |
| 3 | Jacquard | 1,816 | 0.007546 | -0.007220 | -0.182500 | 0.007600 | 12 |
| 4 | Application/3D | 551 | 0.002290 | -0.007985 | 0.236973 | 0.002304 | 16 |
| 5 | Solid | 152,503 | 0.633731 | -0.010195 | -0.017066 | 0.636362 | 1 |
| 6 | Slub | 50 | 0.000208 | -0.012040 | -0.053395 | 0.000208 | 26 |
| 7 | Chambray | 50 | 0.000208 | -0.012122 | 0.015261 | 0.000208 | 25 |
| 8 | Embroidery | 2,459 | 0.010218 | -0.012578 | 0.192080 | 0.010236 | 10 |
| 9 | Metallic | 271 | 0.001126 | -0.013277 | -0.056423 | 0.001127 | 21 |
| 10 | Dot | 1,297 | 0.005390 | -0.014048 | -0.065167 | 0.005391 | 14 |

`Light Green` 最近 8 周真实热度与预测趋势：

| week_id | heat_t | share_t | true_growth | pred_growth | pred_share_t1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 2,130 | 0.007074 | 0.161343 | -0.018386 | 0.006953 |
| 97 | 2,699 | 0.008313 | -0.518188 | 0.012749 | 0.008456 |
| 98 | 1,520 | 0.004951 | -0.070223 | -0.029305 | 0.004838 |
| 99 | 1,068 | 0.004615 | -0.225943 | 0.007662 | 0.004688 |
| 100 | 977 | 0.003681 | -0.108884 | -0.028355 | 0.003607 |
| 101 | 891 | 0.003302 | -0.309022 | 0.001981 | 0.003372 |
| 102 | 663 | 0.002424 | 0.062741 | -0.031618 | 0.002397 |
| 103 | 621 | 0.002581 | -0.064504 | 0.002278 | 0.002627 |

## 6. 推荐实现

### 6.1 推荐模块定位

推荐模块是趋势预测结果的应用验证层。它不追求 Kaggle 排名，也不构建复杂推荐系统，而是验证趋势分能否作为轻量 Top-N 推荐的重排序因子。

当前推荐入口：

```sh
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy <strategy>
uv run python src/14_rerank_recommendations.py --method <method>
uv run python src/15_eval_recommendations.py --method <method>
uv run python src/16_run_recommendation_experiment.py --experiment main
```

### 6.2 推荐候选集

| strategy | 行数 | 每用户候选数 | 说明 |
| --- | ---: | --- | --- |
| `popularity` | 14,443,788 | 12-12 | 近期热门候选 |
| `similarity` | 14,443,788 | 12-12 | 用户属性相似候选 |
| `trend_union` | 14,443,788 | 12-12 | 趋势属性商品候选 |
| `default` | 43,331,363 | 35-36 | 热门、相似、趋势候选合并 |

候选池按 strategy 隔离，推荐结果按 method 隔离。`recommendation_items.parquet` 是默认内部长表，`recommendation_items.csv` 只应作为显式导出或历史本地产物，不作为默认 reader 来源。

### 6.3 推荐方法与输出

| 方法 | 类型 | 是否使用趋势分 | 输出路径 |
| --- | --- | --- | --- |
| `global_popularity` | baseline | 否 | `outputs/recommendation/global_popularity/` |
| `recent_popularity` | 强 baseline | 否 | `outputs/recommendation/recent_popularity/` |
| `attribute_similarity` | baseline | 否 | `outputs/recommendation/attribute_similarity/` |
| `pop_similarity` | 融合 baseline | 否 | `outputs/recommendation/pop_similarity/` |
| `pop_similarity_trend` | 趋势感知主方法 | 是 | `outputs/recommendation/pop_similarity_trend/` |

每个方法当前输出：

| 方法 | `recommendations.csv` 行数 | `recommendation_items.parquet` 行数 |
| --- | ---: | ---: |
| `global_popularity` | 1,203,649 | 14,443,788 |
| `recent_popularity` | 1,203,649 | 14,443,788 |
| `attribute_similarity` | 1,203,649 | 14,443,788 |
| `pop_similarity` | 1,203,649 | 14,443,788 |
| `pop_similarity_trend` | 1,203,649 | 14,443,788 |

推荐结果检查结论：

| 检查项 | 结果 |
| --- | --- |
| Top-K 长度 | 每用户 12 个商品 |
| rank 范围 | 1 到 12 |
| 缺失推荐用户 | 0 |
| 重复推荐商品 | 0 |
| `article_id` 前导 0 | 保留 |
| `customer_id` 字符串语义 | 保留 |

### 6.4 推荐主实验

当前 `main` 实验使用 valid split 的 NDCG@12 选择 `pop_similarity_trend` 权重。有限网格搜索共 25 组，当前最佳权重：

| 分数项 | 权重 |
| --- | ---: |
| `pop_score` | 0.2 |
| `sim_score` | 0.2 |
| `trend_score` | 0.1 |
| `recent_score` | 0.5 |

实验摘要：

| 项目 | 值 |
| --- | --- |
| experiment_id | `main` |
| grid search 数量 | 25 |
| best weights | `pop=0.2, sim=0.2, trend=0.1, recent=0.5` |
| ablation 行数 | 10 |
| 主选择指标 | valid NDCG@12 |

### 6.5 消融实验与权重搜索

当前 `outputs/recommendation/experiments/main/experiment.json` 已包含 5 个推荐方法在 valid/test 上的对比，可作为推荐消融的基础表：

| 版本 | 对应方法 | split | MAP@12 | Recall@12 | HitRate@12 | NDCG@12 | Coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full Model | `pop_similarity_trend` | valid | 0.002745 | 0.008566 | 0.030549 | 0.005922 | 0.000712 |
| Full Model | `pop_similarity_trend` | test | 0.003703 | 0.013685 | 0.039141 | 0.007987 | 0.000665 |
| w/o Trend | `pop_similarity` | valid | 0.002120 | 0.005688 | 0.020756 | 0.004269 | 0.004992 |
| w/o Trend | `pop_similarity` | test | 0.002942 | 0.009438 | 0.027705 | 0.006002 | 0.004565 |
| Recent Only | `recent_popularity` | valid | 0.002512 | 0.008691 | 0.031059 | 0.005715 | 0.000206 |
| Recent Only | `recent_popularity` | test | 0.003781 | 0.013939 | 0.039935 | 0.008087 | 0.000191 |
| Global Popularity | `global_popularity` | valid | 0.002102 | 0.006871 | 0.021751 | 0.004402 | 0.000242 |
| Global Popularity | `global_popularity` | test | 0.002364 | 0.006843 | 0.020430 | 0.004629 | 0.000224 |
| Attribute Similarity Only | `attribute_similarity` | valid | 0.000058 | 0.000266 | 0.000926 | 0.000149 | 0.005460 |
| Attribute Similarity Only | `attribute_similarity` | test | 0.000077 | 0.000343 | 0.001103 | 0.000188 | 0.004958 |

不同 `trend_score` 权重在 valid split 上的最好结果如下。当前 grid search 只记录 valid 指标，不记录每组权重的 test 指标：

| trend_score | grid_index | pop_score | sim_score | recent_score | MAP@12 | Recall@12 | HitRate@12 | NDCG@12 | Coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 4 | 0.4 | 0.1 | 0.5 | 0.002644 | 0.008587 | 0.030613 | 0.005813 | 0.000519 |
| 0.1 | 10 | 0.2 | 0.2 | 0.5 | 0.002745 | 0.008566 | 0.030549 | 0.005922 | 0.000712 |
| 0.2 | 7 | 0.2 | 0.1 | 0.5 | 0.002689 | 0.008585 | 0.030608 | 0.005869 | 0.000557 |
| 0.3 | 8 | 0.2 | 0.0 | 0.5 | 0.002606 | 0.008586 | 0.030614 | 0.005775 | 0.000293 |
| 0.4 | 24 | 0.3 | 0.0 | 0.3 | 0.002603 | 0.008435 | 0.029916 | 0.005712 | 0.000277 |

valid split 上排名前 10 的权重组合：

| grid_index | pop_score | sim_score | trend_score | recent_score | MAP@12 | Recall@12 | HitRate@12 | NDCG@12 | Coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.2 | 0.2 | 0.1 | 0.5 | 0.002745 | 0.008566 | 0.030549 | 0.005922 | 0.000712 |
| 7 | 0.2 | 0.1 | 0.2 | 0.5 | 0.002689 | 0.008585 | 0.030608 | 0.005869 | 0.000557 |
| 4 | 0.4 | 0.1 | 0.0 | 0.5 | 0.002644 | 0.008587 | 0.030613 | 0.005813 | 0.000519 |
| 5 | 0.3 | 0.1 | 0.2 | 0.4 | 0.002660 | 0.008524 | 0.030305 | 0.005811 | 0.000533 |
| 22 | 0.2 | 0.3 | 0.0 | 0.5 | 0.002719 | 0.008301 | 0.029608 | 0.005808 | 0.000985 |
| 9 | 0.4 | 0.0 | 0.1 | 0.5 | 0.002626 | 0.008586 | 0.030609 | 0.005794 | 0.000358 |
| 8 | 0.2 | 0.0 | 0.3 | 0.5 | 0.002606 | 0.008586 | 0.030614 | 0.005775 | 0.000293 |
| 6 | 0.3 | 0.0 | 0.2 | 0.5 | 0.002607 | 0.008586 | 0.030611 | 0.005772 | 0.000321 |
| 13 | 0.4 | 0.1 | 0.1 | 0.4 | 0.002621 | 0.008416 | 0.029912 | 0.005728 | 0.000529 |
| 11 | 0.3 | 0.2 | 0.1 | 0.4 | 0.002670 | 0.008201 | 0.029291 | 0.005722 | 0.000659 |

当前缺失：严格意义上的 `w/o Similarity`、`w/o Recent`、以及每组 trend 权重的 test 指标尚未作为独立 artifact 保存。当前 grid 中有 `sim_score=0` 的组合，但没有 `recent_score=0` 的组合；如果论文要单独写完整消融小节，建议补跑并保存这些实验。

### 6.6 推荐评价指标

| 方法 | split | MAP@12 | Recall@12 | HitRate@12 | NDCG@12 | Coverage | 用户数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_popularity` | valid | 0.002102 | 0.006871 | 0.021751 | 0.004402 | 0.000242 | 660,710 |
| `global_popularity` | test | 0.002364 | 0.006843 | 0.020430 | 0.004629 | 0.000224 | 542,939 |
| `recent_popularity` | valid | 0.002512 | 0.008691 | 0.031059 | 0.005715 | 0.000206 | 660,710 |
| `recent_popularity` | test | 0.003781 | 0.013939 | 0.039935 | 0.008087 | 0.000191 | 542,939 |
| `attribute_similarity` | valid | 0.000058 | 0.000266 | 0.000926 | 0.000149 | 0.005460 | 660,710 |
| `attribute_similarity` | test | 0.000077 | 0.000343 | 0.001103 | 0.000188 | 0.004958 | 542,939 |
| `pop_similarity` | valid | 0.002120 | 0.005688 | 0.020756 | 0.004269 | 0.004992 | 660,710 |
| `pop_similarity` | test | 0.002942 | 0.009438 | 0.027705 | 0.006002 | 0.004565 | 542,939 |
| `pop_similarity_trend` | valid | 0.002745 | 0.008566 | 0.030549 | 0.005922 | 0.000712 | 660,710 |
| `pop_similarity_trend` | test | 0.003703 | 0.013685 | 0.039141 | 0.007987 | 0.000665 | 542,939 |

### 6.7 推荐结果分析

推荐实验的核心结论应谨慎表述：

- `pop_similarity_trend` 在 valid split 上取得最高 NDCG@12：0.005922，高于 `recent_popularity` 的 0.005715。
- `pop_similarity_trend` 在 test split 上 NDCG@12 为 0.007987，略低于 `recent_popularity` 的 0.008087。
- `pop_similarity_trend` 相比不含趋势分的 `pop_similarity` 有明显提升：valid NDCG@12 从 0.004269 提升到 0.005922，test NDCG@12 从 0.006002 提升到 0.007987。
- `recent_popularity` 是非常强的短期热门 baseline，说明 H&M 交易数据中近期热度对短期购买预测非常有效。
- `attribute_similarity` 覆盖率较高，但命中类指标很低，说明单纯属性相似不足以支撑推荐，需要结合热门、近期活跃和趋势分。

论文中建议写：

> 趋势感知融合模型在验证集上取得最优 NDCG@12，并在测试集上接近强近期热门基线，同时显著优于不含趋势分的 Pop + Similarity 模型。这说明趋势预测分能够作为推荐重排序的有效补充特征，但短期热门仍是 H&M 购买预测中的强信号。

需要避免的表述：

- 不要写 `pop_similarity_trend` 在所有推荐指标上全面超过 `recent_popularity`。
- 不要把推荐模块称为完整个性化推荐系统。
- 不要把低绝对值的 MAP@12/NDCG@12 包装成高业务性能。

### 6.8 用户推荐解释案例

下面案例来自 `pop_similarity_trend` 在 test split、`cutoff_week=103`、`label_week=104` 的推荐结果。该用户 Top-12 推荐中命中 2 个真实购买商品。

用户：

```text
7009db8064307737b96229806d6b8ab09b34510dc93c20fe09c58ffc1ed5d126
```

用户历史偏好属性：

| attr_type | attr_value | preference_score | purchase_count | last_purchase_week |
| --- | --- | ---: | ---: | ---: |
| `graphical_appearance_name` | Solid | 0.114286 | 24 | 95 |
| `product_group_name` | Garment Upper body | 0.080952 | 17 | 95 |
| `colour_group_name` | Black | 0.066667 | 14 | 95 |

该窗口代表性趋势属性：

| attr_type | 示例趋势属性 | 说明 |
| --- | --- | --- |
| `colour_group_name` | Light Green、Light Yellow、Green | 颜色趋势榜前列 |
| `product_type_name` | Sunglasses、Umbrella、Tie、Sneakers | 品类趋势榜前列 |
| `graphical_appearance_name` | Neps、Lace、Jacquard、Application/3D | 图案趋势榜前列 |

Top-12 推荐及解释字段：

| rank | article_id | 命中 | 商品名 | 类型 | 颜色 | 图案 | candidate_sources | score | pop | sim | trend | recent |
| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `0706016001` |  | Jade HW Skinny Denim TRS | Trousers | Black | Solid | popularity | 0.801732 | 1.000000 | 0.585366 | 0.504057 | 0.868507 |
| 2 | `0751471001` |  | Pluto RW slacks (1) | Trousers | Black | Solid | popularity | 0.738694 | 0.356075 | 0.585366 | 0.504057 | 1.000000 |
| 3 | `0915529003` |  | Liliana | Sweater | Black | Solid | popularity | 0.662980 | 0.045209 | 1.000000 | 0.501963 | 0.807483 |
| 4 | `0915526001` | 是 | Nika vest | Sweater | Off White | Solid | popularity | 0.624491 | 0.053127 | 0.658537 | 0.499026 | 0.864511 |
| 5 | `0896152002` |  | Amelie | T-shirt | Black | Solid | popularity | 0.606239 | 0.050968 | 1.000000 | 0.502413 | 0.691609 |
| 6 | `0863595006` |  | Baraboom throw-on | Cardigan | Black | Solid | popularity | 0.581176 | 0.048928 | 1.000000 | 0.501048 | 0.642572 |
| 7 | `0915526002` |  | Nika vest | Sweater | Black | Solid | popularity | 0.540751 | 0.037951 | 1.000000 | 0.501963 | 0.565928 |
| 8 | `0448509014` |  | Perrie Slim Mom Denim TRS | Trousers | Blue | Solid | popularity | 0.503277 | 0.375610 | 0.243902 | 0.497337 | 0.659281 |
| 9 | `0916468003` | 是 | Bailey | Cardigan | Blue | Solid | popularity | 0.496750 | 0.053347 | 0.658537 | 0.494328 | 0.609880 |
| 10 | `0918292001` |  | STRONG HW seamless tights | Leggings/Tights | Black | Melange | popularity | 0.484772 | 0.065924 | 0.000000 | 0.493206 | 0.844533 |
| 11 | `0751471043` |  | Pluto RW slacks (1) | Trousers | Black | Check | popularity | 0.480334 | 0.051928 | 0.000000 | 0.502242 | 0.839448 |
| 12 | `0898694001` |  | Jasba jersey shacket | Blazer | Light Beige | Melange | popularity | 0.408649 | 0.037051 | 0.073171 | 0.498814 | 0.673447 |

案例解读：

- 排名靠前商品多数同时具有近期热门信号和较高 recent 分。
- 多个商品颜色或图案与用户历史偏好 `Black`、`Solid` 对齐，因此 sim 分较高。
- trend 分在该窗口中提供稳定的补充信号，但不是单独决定排序的主要因素。
- 命中商品 `0915526001` 和 `0916468003` 都是 `Solid` 图案，并且与用户历史偏好有重叠。

当前 reports 导出阶段已按可复现规则生成 3 个推荐解释案例，文件位于 `outputs/reports/case_studies/`。

## 7. 最终论文素材清单

### 7.1 必备表格

| 表格 | 数据来源 | 状态 |
| --- | --- | --- |
| 原始数据与处理后数据规模表 | README、数据 profile、产物行数 | 可写 |
| 属性字段与属性类型表 | `articles_clean.csv`、`nodes_attribute.csv` | 可写 |
| 属性图节点与边统计表 | `nodes_article.csv`、`nodes_attribute.csv`、边表 | 可写 |
| 趋势样本特征表 | `trend_model_samples.parquet` 列契约 | 可写 |
| 时间切分表 | split metadata | 可写 |
| 趋势模型对比表 | `outputs/metrics/<model>/trend_metrics.json` | 可写 |
| 推荐方法对比表 | `outputs/recommendation/<method>/metrics.json` | 可写 |
| 推荐主实验权重表 | `outputs/recommendation/experiments/main/experiment.json` | 可写 |
| 消融实验表 | `outputs/recommendation/experiments/main/experiment.json` 的 `ablation` | 可写 |

### 7.2 已导出的核心图表与案例

`src/17_export_paper_assets.py` 当前会导出 8 张核心图，每张同时生成 SVG 和 PNG。默认输出目录为 `outputs/reports/figures/`：

| 图表 | 内容 | SVG | PNG |
| --- | --- | --- | --- |
| 数据处理流程图 | 原始数据、属性图、趋势预测、推荐实验到论文素材的整体链路 | `outputs/reports/figures/data_pipeline.svg` | `outputs/reports/figures/data_pipeline.png` |
| 属性层次图示意图 | 商品节点、属性节点、属性层级边和商品-属性边 | `outputs/reports/figures/attribute_graph_schema.svg` | `outputs/reports/figures/attribute_graph_schema.png` |
| 典型趋势属性曲线 | 代表性趋势属性最近 8 周的热度、预测份额和预测增长 | `outputs/reports/figures/trend_curve_examples.svg` | `outputs/reports/figures/trend_curve_examples.png` |
| LightGBM 特征重要性图 | `feature_importance.csv` 中 normalized gain Top-N | `outputs/reports/figures/lightgbm_feature_importance.svg` | `outputs/reports/figures/lightgbm_feature_importance.png` |
| 趋势模型指标柱状图 | `last_week`、`previous_growth`、`moving_average`、`lightgbm` 的 NDCG@10 对比 | `outputs/reports/figures/trend_model_metrics.svg` | `outputs/reports/figures/trend_model_metrics.png` |
| 推荐方法指标柱状图 | 五种推荐方法的 NDCG@12 对比 | `outputs/reports/figures/recommendation_method_metrics.svg` | `outputs/reports/figures/recommendation_method_metrics.png` |
| Top-K 趋势属性榜 | test week 103 下颜色、品类、图案等趋势属性 Top-K | `outputs/reports/figures/topk_trend_attributes.svg` | `outputs/reports/figures/topk_trend_attributes.png` |
| 推荐权重分析图 | trend 权重与 valid NDCG@12，以及主实验权重构成 | `outputs/reports/figures/recommendation_weight_analysis.svg` | `outputs/reports/figures/recommendation_weight_analysis.png` |

案例导出位于 `outputs/reports/case_studies/`，当前包含 `case_01`、`case_02`、`case_03` 的 JSON 和 Markdown。每个案例包含用户历史偏好、代表性趋势属性、Top-12 推荐商品、命中标记、商品属性和分数分解。

### 7.3 本地答辩展示应用

本地答辩展示应用已实现为只读展示层：

| 组成 | 路径 | 说明 |
| --- | --- | --- |
| 展示库构建入口 | `src/18_build_defense_app_db.py` | 从稳定 artifact 构建 SQLite，不训练模型、不重跑推荐 |
| 展示库业务包 | `src/fashion_trend/presentation/` | schema、路径、上游读取、表构建、SQLite writer 和 runner |
| SQLite 产物 | `outputs/defense_app/fashion_demo.sqlite` | 生成产物，不提交 |
| 后端 | `apps/defense_app/backend/` | FastAPI 只读查询 SQLite |
| 前端 | `apps/defense_app/frontend/` | Vue/Vite，只调用 FastAPI `/api` |

运行命令：

```sh
uv run python src/18_build_defense_app_db.py
uv run --group app uvicorn app.main:app --reload --app-dir apps/defense_app/backend
cd apps/defense_app/frontend
npm install
npm run dev
```

展示页面覆盖趋势看板、属性详情、商品属性图、推荐案例列表和推荐解释页。文案边界应保持“本地答辩展示应用、离线趋势预测、轻量 Top-N 推荐实验、推荐解释”，不要把它描述成在线服务、生产推荐平台、实时个性化系统或深度推荐模型。

### 7.4 可直接引用的核心观点

1. 项目不是复现 H&M Kaggle 高分方案，而是将推荐数据重构为属性趋势预测任务。
2. 属性周热度和趋势标签构成了可解释的服装趋势建模对象。
3. LightGBM 在趋势预测上显著优于简单历史 baseline，说明结构化属性趋势特征有效。
4. 趋势分用于推荐时可以提升相对不含趋势分的融合模型，但不应夸大为全面击败近期热门。
5. 轻量推荐模块的价值在于展示趋势预测的应用路径和解释性，而不是追求复杂推荐系统性能。
6. 本地展示应用只用于答辩演示和结果审查，不改变论文实验口径。

## 8. 当前仍需补充的论文工作

当前不建议继续扩大模型范围。更优先的收尾工作是：

报告导出阶段已实现为 `src/17_export_paper_assets.py`。当前真实 artifact 验证已生成 `outputs/reports/` 下 16 个 figure 文件、16 个 table 文件、3 个案例和 manifest；该阶段只读取稳定 artifact，不训练模型，也不重跑推荐实验。

| 优先级 | 任务 | 说明 |
| --- | --- | --- |
| 高 | 按需重新运行报告导出和展示库构建命令 | 已生成论文 figures、tables、3 个推荐解释案例、manifest 和本地展示库；论文提交或答辩前可重新运行确保产物最新 |
| 中 | 写实验局限性 | 推荐指标绝对值较低、近期热门强、未做深度推荐 |
| 中 | 补严格推荐消融 | 当前缺少独立 `w/o Similarity`、`w/o Recent` 和每组权重 test 指标 |
| 低 | 更多模型扩展 | 不建议作为当前论文主线继续扩 |

当前明确缺失或尚未固化为 artifact 的内容：

| 缺失项 | 当前状态 | 建议处理 |
| --- | --- | --- |
| `w/o Similarity` 独立消融 | grid 中存在 `sim_score=0` 组合，但没有命名为独立消融产物 | 如论文要写消融小节，单独保存该版本 valid/test 指标 |
| `w/o Recent` 独立消融 | 当前 25 组 grid 没有 `recent_score=0` 组合 | 需要新增权重组合并评价 |
| 每组 trend 权重的 test 指标 | 当前 grid search 只保存 valid 指标 | 若要严谨比较不同 trend 权重，应补 test 评价 |
| 趋势属性可视化图 | 已导出 `trend_curve_examples` 和 `topk_trend_attributes` | 论文排版时选择 SVG 或 PNG |
| 推荐解释案例扩展 | 已导出 3 个案例 | 论文正文可选 1 个详细展示，附录可放其余案例 |

## 9. 风险与论文表述边界

| 风险 | 处理方式 |
| --- | --- |
| 推荐指标绝对值较低 | 强调推荐是趋势预测应用验证，不追求 Kaggle 排名 |
| `recent_popularity` 在 test 上略优于趋势主方法 | 如实写成强 baseline，并强调趋势分相对 `pop_similarity` 的增益 |
| 属性相似方法命中差 | 作为负结果分析，说明用户历史属性偏好需要热门和趋势信号补充 |
| 本地存在历史 `recommendation_items.csv` | 论文和代码契约以 `recommendation_items.parquet` 为默认内部长表 |
| 报告图表需本机 CJK 字体 | reports 阶段会 fail-fast，避免生成缺字中文图 |

## 10. 验证记录

最近一次项目核查包含：

```sh
uv run pytest tests/test_presentation_*.py tests/test_architecture_boundaries.py
uv run --group app pytest apps/defense_app/backend/tests
uv run python -m compileall -q src
uv run --group app python -m compileall -q apps/defense_app/backend
uv run python src/18_build_defense_app_db.py
cd apps/defense_app/frontend
npm run typecheck
npm run build
git diff --check
```

结果：

| 验证 | 结果 |
| --- | --- |
| presentation + architecture pytest | 62 passed |
| defense app backend pytest | 38 passed |
| compileall src | 通过 |
| compileall backend | 通过 |
| defense app SQLite 构建 | 通过，`demo_users=3`，`recommendation_items=36`，`report_assets=16` |
| frontend typecheck | 通过 |
| frontend build | 通过；存在 Vite chunk size warning，不影响构建产物 |
| diff check | 通过 |

此外，真实 artifact 审计确认：

- 关键数据产物存在。
- 趋势预测输出列契约和分布校验通过。
- 推荐输出 Top-12、rank、重复项、缺失用户检查通过。
- `article_id` 和 `customer_id` 的字符串语义保持正确。
- 展示库表结构存在，每个 demo case 均有 12 条推荐项，四类核心趋势属性均已写入。
- 浏览器 smoke 确认趋势看板、属性详情、商品属性图、推荐案例、推荐解释页和 `/docs` 均可加载。
