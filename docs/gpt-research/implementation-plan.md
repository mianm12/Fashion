# 《基于 H&M 交易数据与商品属性层次图的服装属性周级趋势预测与轻量 Top-N 推荐》实施方案

## 0. 先给最终执行结论

你的毕业设计应该按下面这条主线做：

```text
H&M articles.csv
    ↓
构建静态商品属性层次图
    ↓
H&M transactions_train.csv
    ↓
按周聚合商品销量
    ↓
把商品销量映射到属性节点
    ↓
得到属性周热度序列
    ↓
构造属性级趋势标签
    ↓
训练趋势预测 baseline 与 LightGBM Regressor
    ↓
预测下一周上升属性
    ↓
将属性趋势分映射回商品
    ↓
结合近期热门、用户历史偏好、趋势分
    ↓
生成轻量 Top-12 商品推荐
```

这篇毕设的核心不是“做一个很强的推荐系统”，而是：

> **把 H&M 推荐数据集改造成服装属性周级趋势预测问题，并用轻量推荐展示趋势预测的应用价值。**

H&M 数据集本身提供交易记录、商品元数据和用户元数据，其中 `transactions_train.csv` 连接 `customer_id` 与 `article_id`，`articles.csv` 提供商品的品类、颜色、部门、图案等丰富属性，适合构造“商品—属性”关系和属性热度序列。([Hugging Face][1]) 原 Kaggle 任务是预测训练集结束后未来 7 天用户可能购买的商品，常见评价口径是 MAP@12；但你的课题不需要追求 Kaggle 高分，只需要借用它的 Top-N 推荐评价形式。([Ashwin Mathur][2])

---

# 1. 整体技术路线

## 1.1 总体路线图

| 阶段          | 输入                                                      | 核心处理                                 | 输出                            |
| ----------- | ------------------------------------------------------- | ------------------------------------ | ----------------------------- |
| 1. 数据读取与检查  | `transactions_train.csv`、`articles.csv`、`customers.csv` | 类型转换、缺失值检查、日期处理                      | 数据概况表                         |
| 2. 周切分      | `transactions_train.csv`                                | 按 7 天生成 `week_id`                    | 周级交易表                         |
| 3. 静态属性图构建  | `articles.csv`                                          | 构造商品节点、属性节点、商品—属性边、属性层级边             | 节点表、边表                        |
| 4. 属性周热度计算  | 周级交易表 + 商品—属性边                                          | 将商品购买次数聚合到属性节点                       | `attribute_week_heat.csv`     |
| 5. 趋势标签定义   | 属性周热度表                                                  | 计算下一周增长率、热度变化                        | `attribute_week_target.csv`   |
| 6. 趋势样本构造   | 属性热度 + 图特征 + 时间特征                                       | 构造 lag、移动平均、增长率、父级热度等特征              | `trend_model_samples.parquet` |
| 7. 趋势模型训练   | 趋势样本                                                    | Last Week Heat、Previous Growth、Moving Average + LightGBM Regressor | 趋势预测模型与预测结果                   |
| 8. 趋势评价     | 真实趋势 + 预测趋势                                             | MAE、RMSE、Spearman、NDCG@K、Precision@K | 趋势实验结果                        |
| 9. 推荐模块     | 用户历史 + 商品属性 + 趋势预测                                      | 候选召回 + 线性加权重排序                       | Top-12 推荐列表                   |
| 10. 推荐评价与展示 | 推荐列表 + 测试周真实购买                                          | MAP@12、Recall@12、HitRate@12、Coverage | 推荐实验结果与展示页面                   |

---

# 2. 每个阶段具体做什么

## 阶段 1：数据读取与基础检查

### 必须做

| 任务               | 说明                                                      |
| ---------------- | ------------------------------------------------------- |
| 定位三张原始表          | `transactions_train.csv`、`articles.csv`、`customers.csv` |
| 检查原始文件           | 确认 `transactions_train.csv`、`articles.csv`、`customers.csv` 存在且可读 |
| 打印数据规模           | 输出三张原始 CSV 的行数日志                                               |

H&M 数据集中商品元数据、用户元数据和交易表的关系非常清楚：交易表通过 `article_id` 连接商品表，通过 `customer_id` 连接用户表。([Hugging Face][1])

### 当前实现输出

```text
控制台行数日志；不写入稳定文件产物。
```

---

## 阶段 2：按周切分交易数据

### 必须做

你需要把每日交易转成周级交易。

定义：

$$
\mathrm{week\_id} = \left\lfloor \frac{t_{\mathrm{dat}} - t_{\min}}{7} \right\rfloor
$$

处理后得到：

| t_dat      | week_id | customer_id | article_id | price | sales_channel_id |
| ---------- | ------: | ----------- | ---------- | ----: | ---------------: |
| 2018-09-20 |       0 | xxx         | 0706016001 |  0.02 |                2 |
| 2018-09-23 |       0 | xxx         | 0507909001 |  0.03 |                1 |
| 2018-09-28 |       1 | xxx         | 0706016001 |  0.02 |                2 |

### 推荐规则

| 项目       | 建议                                |
| -------- | --------------------------------- |
| 周起点      | 使用数据集中最早日期作为第 0 周                 |
| 时间粒度     | 固定 7 天                            |
| 是否按自然周   | 不强制，7 天窗口即可                       |
| 是否考虑价格   | MVP 不考虑；增强版可做销售额热度                |
| 是否区分线上线下 | MVP 不区分；增强版可分析 `sales_channel_id` |

### 输出文件

```text
data/interim/transactions_train_weekly.parquet
```

---

# 3. 数据如何预处理

## 3.1 `transactions_train.csv`

| 字段                 | 处理方式             | 用途        |
| ------------------ | ---------------- | --------- |
| `t_dat`            | 转日期，生成 `week_id` | 时间划分、周级聚合 |
| `customer_id`      | 保留字符串            | 推荐任务用户标识  |
| `article_id`       | 转字符串，保留前导 0      | 连接商品表     |
| `price`            | 保留，可选使用          | 可选销售额热度   |
| `sales_channel_id` | 保留，可选统计          | 可选渠道分析    |

必须生成：

```text
week_id
```

可选生成：

```text
date
month
season
```

---

## 3.2 `articles.csv`

建议先使用这些字段：

| 字段                             | MVP 是否使用 | 稳妥版是否使用 | 说明         |
| ------------------------------ | -------: | ------: | ---------- |
| `article_id`                   |       必须 |      必须 | 商品节点       |
| `product_group_name`           |       必须 |      必须 | 商品大类       |
| `product_type_name`            |       必须 |      必须 | 商品小类       |
| `garment_group_name`           |       必须 |      必须 | 服装组别       |
| `colour_group_name`            |       必须 |      必须 | 具体颜色       |
| `graphical_appearance_name`    |       必须 |      必须 | 图案外观       |
| `perceived_colour_master_name` |       可选 |      建议 | 主色系        |
| `perceived_colour_value_name`  |       可选 |      建议 | 明暗属性       |
| `index_group_name`             |       可选 |      建议 | 商品业务大类     |
| `index_name`                   |       可选 |      建议 | 商品业务线      |
| `section_name`                 |       可选 |      建议 | 商品区域       |
| `department_name`              |       可选 |      建议 | 部门         |
| `detail_desc`                  |      不建议 |    可选增强 | 文本处理会加重工作量 |

### MVP 属性字段

建议 MVP 先用 5 个字段：

```text
product_group_name
product_type_name
garment_group_name
colour_group_name
graphical_appearance_name
```

这 5 个字段足够支撑：

* 品类趋势；
* 颜色趋势；
* 图案趋势；
* 服装组趋势；
* 商品属性图构建；
* 推荐解释。

---

## 3.3 `customers.csv`

主任务不需要使用客户画像字段。

| 字段                       | MVP 是否进入主模型 | 建议                |
| ------------------------ | ----------: | ----------------- |
| `age`                    |           否 | 可用于推荐增强或用户分组分析    |
| `club_member_status`     |           否 | 可选统计              |
| `fashion_news_frequency` |           否 | 可选统计              |
| `FN`                     |           否 | 不建议主模型使用          |
| `Active`                 |           否 | 不建议主模型使用          |
| `postal_code`            |           否 | 不建议使用，解释性弱且有隐私敏感性 |

主任务是属性趋势预测，不是用户画像建模，所以不要把客户字段混进趋势主模型。

---

# 4. 静态商品属性图如何构建

## 4.1 图的概念定义

论文中可以这样定义：

$$
G = (V, E)
$$

其中：

$$
V = V_{\mathrm{article}} \cup V_{\mathrm{attribute}}
$$

$$
E = E_{\mathrm{article-attribute}} \cup E_{\mathrm{attribute-attribute}}
$$

解释：

| 符号                        | 含义       |
| ------------------------- | -------- |
| $V_{article}$             | 商品节点集合   |
| $V_{attribute}$           | 属性节点集合   |
| $E_{article-attribute}$   | 商品—属性隶属边 |
| $E_{attribute-attribute}$ | 属性—属性层级边 |

概念上，你可以说它包含：

```text
商品节点
颜色属性节点
品类属性节点
图案属性节点
服装组属性节点
部门属性节点
section 属性节点
```

实现上，统一存成：

```text
attribute 节点表
```

然后用：

```text
attr_type
```

区分属性类型。

---

## 4.2 不使用 Neo4j 的存储方案

不用 Neo4j。
直接使用：

```text
节点表 + 边表 + 周热度表 + 趋势样本表
```

推荐存储格式：

| 数据类型    | 推荐格式            |
| ------- | --------------- |
| 小型节点表   | CSV             |
| 小型边表    | CSV             |
| 大型交易聚合表 | Parquet         |
| 趋势训练样本  | Parquet         |
| 模型文件    | pickle / joblib |
| 实验结果    | CSV / JSON      |

---

# 5. 节点表、边表、周热度表、趋势样本表设计

## 5.1 商品节点表：`nodes_article.csv`

路径：

```text
data/processed/graph/nodes_article.csv
```

字段设计：

| 字段                | 类型  | 说明        |
| ----------------- | --- | --------- |
| `article_id`      | str | 原始商品 ID   |
| `article_node_id` | str | 图中商品节点 ID |
| `product_code`    | str | 可选，商品系列编码 |
| `prod_name`       | str | 可选，商品名称   |

示例：

| article_id | article_node_id    | product_code | prod_name  |
| ---------- | ------------------ | ------------ | ---------- |
| 0706016001 | article_0706016001 | 706016       | Trousers   |
| 0507909001 | article_0507909001 | 507909       | Jersey Top |

MVP 只需要：

```text
article_id
article_node_id
```

---

## 5.2 属性节点表：`nodes_attribute.csv`

路径：

```text
data/processed/graph/nodes_attribute.csv
```

字段设计：

| 字段              | 类型  | 说明            |
| --------------- | --- | ------------- |
| `attr_id`       | str | 属性节点唯一 ID     |
| `attr_type`     | str | 属性类型          |
| `attr_value`    | str | 属性取值          |
| `attr_node_id`  | str | 图中节点 ID       |
| `article_count` | int | 关联商品数量        |
| `is_core_attr`  | int | 是否为核心属性       |
| `level`         | str | 可选，父级/子级/普通属性 |

示例：

| attr_id     | attr_type                 | attr_value         | article_count | is_core_attr |
| ----------- | ------------------------- | ------------------ | ------------: | -----------: |
| attr_000001 | product_group_name        | Garment Upper body |         32120 |            1 |
| attr_000002 | product_type_name         | T-shirt            |         15432 |            1 |
| attr_000003 | colour_group_name         | Black              |         22670 |            1 |
| attr_000004 | graphical_appearance_name | Solid              |         49780 |            1 |

### `attr_id` 生成建议

建议用稳定规则：

```text
attr_id = attr_type + "::" + attr_value
```

例如：

```text
colour_group_name::Black
product_type_name::Dress
garment_group_name::Jersey Basic
```

这样比 `attr_000001` 更容易调试。

---

## 5.3 商品—属性边表：`edges_article_attribute.csv`

路径：

```text
data/processed/graph/edges_article_attribute.csv
```

字段设计：

| 字段                | 类型    | 说明      |
| ----------------- | ----- | ------- |
| `article_id`      | str   | 商品 ID   |
| `article_node_id` | str   | 商品节点 ID |
| `attr_id`         | str   | 属性 ID   |
| `attr_type`       | str   | 属性类型    |
| `attr_value`      | str   | 属性取值    |
| `edge_type`       | str   | 边类型     |
| `edge_weight`     | float | 默认 1.0  |

示例：

| article_id | attr_id                          | attr_type          | attr_value   | edge_type         | edge_weight |
| ---------- | -------------------------------- | ------------------ | ------------ | ----------------- | ----------: |
| 0706016001 | product_type_name::Trousers      | product_type_name  | Trousers     | has_product_type  |         1.0 |
| 0706016001 | colour_group_name::Black         | colour_group_name  | Black        | has_colour_group  |         1.0 |
| 0706016001 | garment_group_name::Jersey Basic | garment_group_name | Jersey Basic | has_garment_group |         1.0 |

这个表是后续计算属性热度的核心桥梁。

---

## 5.4 属性层级边表：`edges_attribute_hierarchy.csv`

路径：

```text
data/processed/graph/edges_attribute_hierarchy.csv
```

字段设计：

| 字段                 | 类型  | 说明           |
| ------------------ | --- | ------------ |
| `parent_attr_id`   | str | 父属性 ID       |
| `child_attr_id`    | str | 子属性 ID       |
| `parent_attr_type` | str | 父属性类型        |
| `child_attr_type`  | str | 子属性类型        |
| `relation_type`    | str | 层级关系         |
| `edge_weight`      | int | 该关系在商品表中共现次数 |

推荐构建这些层级边：

| 父属性                            | 子属性                 | 关系                            |
| ------------------------------ | ------------------- | ----------------------------- |
| `product_group_name`           | `product_type_name` | product_group_contains_type   |
| `perceived_colour_master_name` | `colour_group_name` | colour_master_contains_colour |
| `index_group_name`             | `index_name`        | index_group_contains_index    |
| `index_name`                   | `section_name`      | index_contains_section        |
| `section_name`                 | `department_name`   | section_contains_department   |

示例：

| parent_attr_id                         | child_attr_id                 | relation_type                 | edge_weight |
| -------------------------------------- | ----------------------------- | ----------------------------- | ----------: |
| product_group_name::Garment Upper body | product_type_name::T-shirt    | product_group_contains_type   |        5021 |
| perceived_colour_master_name::Blue     | colour_group_name::Light Blue | colour_master_contains_colour |         842 |
| index_group_name::Ladieswear           | index_name::Ladieswear        | index_group_contains_index    |       12031 |

### 注意

H&M 商品字段不一定是严格树结构。某些子属性可能在数据中对应多个父属性。

处理方式：

| 情况         | 建议                      |
| ---------- | ----------------------- |
| 子属性只有一个父属性 | 直接建立父子边                 |
| 子属性对应多个父属性 | 保留多条边，并记录 `edge_weight` |
| 想简化图结构     | 只保留 `edge_weight` 最大的父边 |
| 论文表述       | 称为“基于商品共现关系构造的经验层级边”    |

---

## 5.5 商品周销量表：`article_week_sales.csv`

路径：

```text
data/processed/trend/article_week_sales.csv
```

字段设计：

| 字段               | 类型    | 说明           |
| ---------------- | ----- | ------------ |
| `week_id`        | int   | 周编号          |
| `article_id`     | str   | 商品 ID        |
| `sales_cnt`      | int   | 该商品本周购买次数    |
| `sales_user_cnt` | int   | 购买该商品的用户数，可选 |
| `sales_amount`   | float | 销售额，可选       |

示例：

| week_id | article_id | sales_cnt | sales_user_cnt |
| ------: | ---------- | --------: | -------------: |
|      80 | 0706016001 |       125 |            118 |
|      80 | 0507909001 |        89 |             84 |
|      81 | 0706016001 |       132 |            127 |

---

## 5.6 属性周热度表：`attribute_week_heat.csv`

路径：

```text
data/processed/trend/attribute_week_heat.csv
```

字段设计：

| 字段                | 类型    | 说明                |
| ----------------- | ----- | ----------------- |
| `week_id`         | int   | 周编号               |
| `attr_id`         | str   | 属性 ID             |
| `attr_type`       | str   | 属性类型              |
| `attr_value`      | str   | 属性取值              |
| `heat_cnt`        | int   | 属性原始热度            |
| `type_total_heat` | int   | 同类型属性本周总热度        |
| `heat_share`      | float | 属性在同类型中的热度占比      |
| `log_heat`        | float | `log1p(heat_cnt)` |
| `rank_in_type`    | int   | 同类型属性本周热度排名       |

示例：

| week_id | attr_type         | attr_value | heat_cnt | heat_share | rank_in_type |
| ------: | ----------------- | ---------- | -------: | ---------: | -----------: |
|      80 | colour_group_name | Black      |    12650 |      0.231 |            1 |
|      80 | colour_group_name | Blue       |     8300 |      0.152 |            2 |
|      80 | product_type_name | Dress      |     8420 |      0.156 |            3 |

这是趋势预测最核心的数据表。

---

## 5.7 属性趋势标签表：`attribute_week_target.csv`

路径：

```text
data/processed/trend/attribute_week_target.csv
```

字段设计：

| 字段                       | 类型    | 说明         |
| ------------------------ | ----- | ---------- |
| `week_id`                | int   | 当前周 $t$    |
| `attr_id`                | str   | 属性 ID      |
| `attr_type`              | str   | 属性类型       |
| `heat_t`                 | float | 当前周热度      |
| `heat_t1`                | float | 下一周热度      |
| `share_t`                | float | 当前周占比      |
| `share_t1`               | float | 下一周占比      |
| `target_log_heat_t1`     | float | 下一周 log 热度 |
| `target_growth`          | float | 下一周增长率标签   |
| `target_rank_in_type_t1` | int   | 下一周同类型排名   |

示例：

| week_id | attr_type         | attr_value | share_t | share_t1 | target_growth |
| ------: | ----------------- | ---------- | ------: | -------: | ------------: |
|      80 | colour_group_name | Black      |   0.231 |    0.228 |        -0.013 |
|      80 | colour_group_name | Blue       |   0.152 |    0.160 |         0.051 |
|      80 | product_type_name | Dress      |   0.156 |    0.169 |         0.080 |

---

## 5.8 趋势训练样本表：`trend_model_samples.parquet`

路径：

```text
data/processed/features/trend_model_samples.parquet
```

字段设计：

| 字段类型  | 字段示例                                                                   |
| ----- | ---------------------------------------------------------------------- |
| 标识字段  | `week_id`, `attr_id`, `attr_type`, `attr_value`                        |
| 当前热度  | `heat_t`, `share_t`, `log_heat_t`, `rank_in_type_t`                    |
| 滞后热度  | `heat_lag_1`, `heat_lag_2`, `heat_lag_3`, `heat_lag_4`                 |
| 滞后占比  | `share_lag_1`, `share_lag_2`, `share_lag_3`, `share_lag_4`             |
| 增长特征  | `growth_lag_1`, `growth_lag_2`, `acc_lag_1`                            |
| 移动统计  | `heat_ma_4`, `share_ma_4`, `share_std_4`, `share_max_4`, `share_min_4` |
| 图结构特征 | `article_count`, `degree`, `parent_share_t`, `sibling_share_mean_t`    |
| 时间特征  | `month`, `week_of_year`, `season_id`                                   |
| 目标字段  | `target_growth`, `target_log_heat_t1`                                  |

---

# 6. 属性周热度如何计算

## 6.1 原始热度

对属性 $a$，第 $t$ 周热度定义为：

$$
h_{a,t} = \sum_{r \in R_t} \mathbf{1}(a \in A(i_r))
$$

其中：

| 符号        | 含义                 |
| --------- | ------------------ |
| $R_t$     | 第 $t$ 周的交易记录集合     |
| $i_r$     | 交易记录 $r$ 对应的商品     |
| $A(i_r)$  | 商品 $i_r$ 关联的属性集合   |
| $a$       | 某个属性节点             |
| $h_{a,t}$ | 属性 $a$ 在第 $t$ 周的热度 |

简单理解：

> 某个属性关联的商品在这一周被买了多少次，这个属性的热度就是多少。

例如：

| 商品        | 商品周销量 | 属性             |
| --------- | ----: | -------------- |
| article_1 |   100 | Black, T-shirt |
| article_2 |    50 | Black, Dress   |
| article_3 |    30 | Blue, T-shirt  |

则：

```text
Black 热度 = 100 + 50 = 150
T-shirt 热度 = 100 + 30 = 130
Dress 热度 = 50
Blue 热度 = 30
```

---

## 6.2 同类型占比热度

只看原始热度会让大类属性长期占优，所以主模型建议使用同类型占比：

$$
s_{a,t} = \frac{h_{a,t}}{\sum_{a' \in C(a)}h_{a',t}}
$$

其中 $C(a)$ 是与属性 $a$ 同类型的属性集合。

例如：

```text
colour_group_name 内部：
Black、Blue、White、Pink 之间比较

product_type_name 内部：
Dress、T-shirt、Trousers 之间比较
```

这样可以避免把颜色属性和品类属性直接混在一起比较。

---

## 6.3 建议同时保存三种热度

| 热度字段         | 公式                | 用途       |
| ------------ | ----------------- | -------- |
| `heat_cnt`   | $h_{a,t}$         | 原始销量解释   |
| `heat_share` | $s_{a,t}$         | 趋势建模主输入  |
| `log_heat`   | $\log(1+h_{a,t})$ | 缓解头部属性过大 |

---

# 7. 趋势标签如何定义

## 7.1 主标签：属性占比增长率

主标签建议使用：

$$
y_{a,t} = \log \frac{s_{a,t+1}+\epsilon}{s_{a,t}+\epsilon}
$$

其中：

| 符号          | 含义                 |
| ----------- | ------------------ |
| $s_{a,t}$   | 属性 $a$ 当前周的同类型热度占比 |
| $s_{a,t+1}$ | 属性 $a$ 下一周的同类型热度占比 |
| $\epsilon$  | 平滑项，建议取 $10^{-6}$  |

这个标签表示：

> 属性 $a$ 从当前周到下一周是否变得更流行。

---

## 7.2 可选辅助标签：下一周热度

辅助标签：

$$
z_{a,t} = \log(1+h_{a,t+1})
$$

用途：

| 标签                   | 适合回答的问题    |
| -------------------- | ---------- |
| `target_growth`      | 哪些属性下周上升最快 |
| `target_log_heat_t1` | 哪些属性下周最热门  |

你的主任务应该使用：

```text
target_growth
```

辅助实验可以加入：

```text
target_log_heat_t1
```

---

## 7.3 低频属性过滤

增长率对低频属性很敏感。比如某属性从 1 次购买变成 3 次购买，增长率很高，但意义不大。

建议过滤规则：

| 过滤条件   | MVP 建议                       |
| ------ | ---------------------------- |
| 属性总热度  | `total_heat >= 100`          |
| 活跃周数   | `active_weeks >= 8`          |
| 单周最低热度 | 计算标签时允许为 0，但展示 Top-K 时过滤低频属性 |

答辩时可以说：

> 为降低低频属性导致的趋势噪声，本文对总热度和活跃周数过低的属性进行过滤。

---

# 8. LightGBM 的训练样本如何构造

LightGBM 不应该输入“所有属性组成的一个大向量”，而应该采用：

> **每个属性在每一周形成一条训练样本。**

LightGBM 官方的 `LGBMRegressor` 是回归模型，适合你用历史热度、增长率、图结构和时间特征预测连续型趋势分。([LightGBM][3])

---

## 8.1 一条训练样本长什么样

| week_id | attr_id                  | attr_type         | share_t | share_lag_1 | share_ma_4 | growth_lag_1 | parent_share_t | target_growth |
| ------: | ------------------------ | ----------------- | ------: | ----------: | ---------: | -----------: | -------------: | ------------: |
|      80 | colour_group_name::Black | colour_group_name |   0.231 |       0.225 |      0.229 |        0.026 |          0.231 |        -0.013 |
|      80 | product_type_name::Dress | product_type_name |   0.156 |       0.149 |      0.145 |        0.046 |          0.402 |         0.080 |

也就是说：

```text
输入：属性 a 在第 t 周及之前若干周的特征
输出：属性 a 从第 t 周到第 t+1 周的趋势增长
```

---

## 8.2 特征设计

### 必须特征

| 特征类别   | 字段                                                         |
| ------ | ---------------------------------------------------------- |
| 当前热度   | `heat_t`, `share_t`, `log_heat_t`                          |
| 历史热度   | `share_lag_1`, `share_lag_2`, `share_lag_3`, `share_lag_4` |
| 历史增长   | `growth_lag_1`, `growth_lag_2`                             |
| 移动平均   | `share_ma_4`, `heat_ma_4`                                  |
| 排名     | `rank_in_type_t`                                           |
| 属性类型   | `attr_type`                                                |
| 静态属性规模 | `article_count`                                            |

### 稳妥版增强特征

| 特征类别  | 字段                                            |
| ----- | --------------------------------------------- |
| 图结构   | `degree`, `parent_heat_t`, `parent_share_t`   |
| 同级属性  | `sibling_share_mean_t`, `sibling_share_std_t` |
| 趋势加速度 | `acc_lag_1 = growth_lag_1 - growth_lag_2`     |
| 时间特征  | `month`, `week_of_year`, `season_id`          |

### 不建议 MVP 使用

| 特征                 | 原因          |
| ------------------ | ----------- |
| `detail_desc` 文本向量 | 会引入 NLP 复杂度 |
| 商品图片特征             | 会引入深度学习     |
| 用户画像字段             | 主任务不是用户趋势   |
| GNN embedding      | 过重，不利于答辩稳定  |

---

## 8.3 时间划分

设最大周编号为：

$$
W
$$

推荐划分：

| 用途    | 当前周 $t$     | 目标周           |
| ----- | ----------- | ------------- |
| 趋势训练集 | $t \le W-3$ | $t+1 \le W-2$ |
| 趋势验证集 | $t = W-2$   | $W-1$         |
| 趋势测试集 | $t = W-1$   | $W$           |

这样保证：

* 训练时不使用测试周信息；
* 验证集用于调参；
* 测试集只用于最终报告。

如果总周数充足，也可以使用更稳妥的多周验证：

| 用途 | 周范围      |
| -- | -------- |
| 训练 | 前 80% 周  |
| 验证 | 中间 10% 周 |
| 测试 | 最后 10% 周 |

但本科实现中，使用最后 1 周测试、倒数第 2 周验证已经足够清楚。

---

# 9. 需要训练哪些 baseline 和主模型

## 9.1 趋势预测 baseline

| 模型              | 当前实现名 | 公式 / 方法                                                                 | 是否必须 | 作用       |
| ----------------- | ---------- | --------------------------------------------------------------------------- | ---: | ---------- |
| Last Week Heat    | `last_week` | $\hat{s}_{a,t+1}=\operatorname{normalize}(s_{a,t})$                         |   必须 | 当前 share 热度持平/归一化基线 |
| Previous Growth   | `previous_growth` | $\hat{y}_{a,t}=\mathrm{growth\_lag\_1}$                                      |   必须 | 上一期增长率基线 |
| Moving Average    | `moving_average` | $\hat{y}_{a,t}=\operatorname{mean}(\mathrm{growth\_lag\_1},\mathrm{growth\_lag\_2})$ |   必须 | 平滑增长基线 |

---

## 9.2 主模型

主模型：

```text
LightGBM Regressor
```

预测目标：

```text
target_growth
```

输入：

```text
属性-周级特征
```

输出：

```text
下一周属性增长趋势分
```

建议参数先用：

| 参数                  |          建议值 |
| ------------------- | -----------: |
| `objective`         | `regression_l1` |
| `n_estimators`      |          300 |
| `learning_rate`     |         0.05 |
| `num_leaves`        |           31 |
| `max_depth`         |       -1 或 6 |
| `min_child_samples` |           30 |
| `subsample`         |          0.8 |
| `colsample_bytree`  |          0.6 |

当前默认训练参数优先来自 `outputs/models/lightgbm/params.json` 中已经发布的 stable 参数；stable 参数文件不存在时才使用源码中的 built-in 默认参数。显式 `--params` 或 `--param` 会进入自定义实验参数模式，不读取 stable 参数。

不要把参数调优作为论文重点。
本科毕设更重要的是：

* 问题定义清楚；
* 时间划分正确；
* baseline 完整；
* 趋势结果可解释；
* 推荐展示有闭环。

---

# 10. 趋势预测如何评价

趋势预测既是回归任务，也是排序任务。

所以要同时评估：

```text
数值误差 + 排序质量
```

---

## 10.1 回归误差指标

| 指标   | 公式含义   | 用途     |
| ---- | ------ | ------ |
| MAE  | 平均绝对误差 | 直观稳定   |
| RMSE | 均方根误差  | 对大误差敏感 |

建议报告：

```text
MAE
RMSE
```

---

## 10.2 排序指标

你的最终应用是找出“下周上升属性”，所以排序指标更重要。

| 指标          | 用途                    | 是否推荐 |
| ----------- | --------------------- | ---: |
| Spearman    | 预测排序与真实排序相关性          | 强烈推荐 |
| Precision@K | 预测 Top-K 中有多少真实 Top-K | 强烈推荐 |
| Recall@K    | 真实 Top-K 中有多少被找回      |   建议 |
| NDCG@K      | 考虑排名位置的 Top-K 质量      | 强烈推荐 |
| HitRate@K   | 是否命中至少一个真实趋势属性        |   可选 |

建议 K 取：

```text
K = 5, 10, 20
```

并且按 `attr_type` 分开评价：

| attr_type                 | MAE | Spearman | NDCG@10 |
| ------------------------- | --: | -------: | ------: |
| colour_group_name         | ... |      ... |     ... |
| product_type_name         | ... |      ... |     ... |
| garment_group_name        | ... |      ... |     ... |
| graphical_appearance_name | ... |      ... |     ... |

这样论文解释性更强。

---

# 11. 推荐模块如何构造候选集、如何打分、如何评价

## 11.1 推荐模块定位

推荐模块是辅助任务。

论文中应写：

> 本文推荐模块不追求复杂推荐系统性能，而是作为趋势预测结果的应用验证。推荐阶段采用候选召回与线性加权重排序，将趋势预测分与用户历史偏好、商品近期热度结合，生成轻量级 Top-N 推荐列表。

---

## 11.2 推荐目标

推荐对象：

```text
用户 u
```

推荐结果：

```text
Top-12 article_id
```

推荐长度建议沿用：

```text
12
```

因为 H&M 原任务常用 MAP@12 评价 Top-12 商品列表。([Ashwin Mathur][2])

---

## 11.3 推荐时间划分

设最大周为 $W$。

| 用途 | 用户历史     | 预测趋势     | 真实购买      |
| -- | -------- | -------- | --------- |
| 验证 | 截止 $W-2$ | 预测 $W-1$ | 第 $W-1$ 周 |
| 测试 | 截止 $W-1$ | 预测 $W$   | 第 $W$ 周   |

评价时：

```text
只评估测试周有购买记录的用户
```

原因：

* 没有未来购买的用户无法判断推荐是否命中；
* 这也更接近 H&M 竞赛的评价逻辑。

---

## 11.4 候选集召回

不要对所有商品直接排序。
应该先构造候选集。

候选集：

$$
C(u,t) = C_{pop}(t) \cup C_{sim}(u,t) \cup C_{trend}(t)
$$

### 候选来源 1：近期热门商品

$$
C_{pop}(t)
$$

取最近 1 周或 4 周销量最高的商品。

| 参数         |              建议 |
| ---------- | --------------: |
| 最近热门 Top-M |             200 |
| 时间窗口       | 最近 1 周 + 最近 4 周 |

---

### 候选来源 2：用户历史属性相似商品

$$
C_{sim}(u,t)
$$

步骤：

1. 获取用户历史购买商品；
2. 提取这些商品的属性；
3. 形成用户属性偏好；
4. 找到属性相似的商品。

相似度可以用 Jaccard：

$$
Sim(u,i)=\frac{|A(u)\cap A(i)|}{|A(u)\cup A(i)|}
$$

其中：

| 符号     | 含义           |
| ------ | ------------ |
| $A(u)$ | 用户历史偏好属性集合   |
| $A(i)$ | 商品 $i$ 的属性集合 |

---

### 候选来源 3：趋势属性商品

$$
C_{trend}(t)
$$

步骤：

1. 获取趋势预测 Top-K 属性；
2. 找到包含这些属性的商品；
3. 从这些商品中取近期活跃商品。

例如：

```text
预测下周上升属性：
colour_group_name::Light Blue
product_type_name::Dress
garment_group_name::Dresses Ladies

召回包含这些属性且最近 4 周有销量的商品
```

---

## 11.5 商品趋势分

商品 $i$ 的趋势分由它关联的属性趋势分聚合得到：

$$
Trend(i,t)=\frac{1}{|A(i)|}\sum_{a\in A(i)}\hat{y}_{a,t}
$$

更稳妥的加权版本：

$$
\begin{aligned}
Trend(i,t)=&
0.35\hat{y}_{\mathrm{product\_type}}
+0.25\hat{y}_{\mathrm{colour}}
+0.20\hat{y}_{\mathrm{garment}}\\
&+0.10\hat{y}_{\mathrm{product\_group}}
+0.10\hat{y}_{\mathrm{appearance}}
\end{aligned}
$$

MVP 可以用平均值。

答辩稳妥版可以用加权值。

---

## 11.6 线性重排序公式

最终推荐分数：

$$
\begin{aligned}
Score(u,i,t)=&
\alpha Pop(i,t)
+\beta Sim(u,i)\\
&+\gamma Trend(i,t)
+\delta Recent(i,t)
\end{aligned}
$$

其中：

| 分数            | 含义              |
| ------------- | --------------- |
| $Pop(i,t)$    | 商品近期热门程度        |
| $Sim(u,i)$    | 商品与用户历史偏好的属性相似度 |
| $Trend(i,t)$  | 商品关联属性的预测趋势分    |
| $Recent(i,t)$ | 商品最近是否活跃        |

建议初始权重：

| 权重       |    值 |
| -------- | ---: |
| $\alpha$ | 0.35 |
| $\beta$  | 0.35 |
| $\gamma$ | 0.25 |
| $\delta$ | 0.05 |

更稳妥做法：

在验证集上网格搜索：

| 参数       | 候选值            |
| -------- | -------------- |
| $\alpha$ | 0.2, 0.3, 0.4  |
| $\beta$  | 0.2, 0.3, 0.4  |
| $\gamma$ | 0.1, 0.2, 0.3  |
| $\delta$ | 0.0, 0.05, 0.1 |

限制：

$$
\alpha+\beta+\gamma+\delta=1
$$

---

## 11.7 推荐 baseline

| 推荐方法                     | 是否必须 | 说明           |
| ------------------------ | ---: | ------------ |
| Global Popularity        |   必须 | 全局热门商品       |
| Recent Popularity        |   必须 | 最近 1/4 周热门商品 |
| Attribute Similarity     |   建议 | 用户历史属性相似商品   |
| Pop + Similarity         |   必须 | 无趋势版本        |
| Pop + Similarity + Trend |   必须 | 最终趋势感知版本     |

你最终要证明：

```text
加入趋势分之后，推荐结果在 NDCG@12 / Recall@12 / MAP@12 / Coverage 上有一定提升，
或者至少推荐列表更具有趋势解释性，并能和 recent_popularity 强 baseline 做实证对照。
```

---

## 11.8 推荐评价指标

| 指标                 | 是否使用 | 说明            |
| ------------------ | ---: | ------------- |
| MAP@12             |   必须 | 与 H&M 原任务风格一致 |
| Recall@12          |   必须 | 推荐命中能力        |
| HitRate@12         |   必须 | 用户级是否命中       |
| NDCG@12            |   必须 | 排名质量；推荐主模型调参的主选择指标 |
| Coverage           |   建议 | 推荐商品覆盖度       |
| Long-tail Coverage |   可选 | 是否推荐长尾商品      |

---

# 12. 最小可行版本 MVP

MVP 的目标是：

> 能完整跑通“属性图 → 周热度 → 趋势预测 → 推荐展示”的闭环。

## 12.1 MVP 必须完成

| 模块       | MVP 要求                                   |
| -------- | ---------------------------------------- |
| 数据读取     | 三张表能读取，完成日期和 ID 处理                       |
| 周切分      | 生成 `week_id`                             |
| 属性字段     | 使用 5 个核心属性字段                             |
| 静态图      | 生成属性节点表、商品—属性边表                          |
| 层级边      | 至少构建 `product_group → product_type`      |
| 属性热度     | 计算 `heat_cnt`, `heat_share`, `log_heat`  |
| 趋势标签     | 使用 `target_growth`                       |
| baseline | Last Week Heat、Previous Growth、Moving Average |
| 主模型      | LightGBM Regressor                       |
| 趋势评价     | MAE、Spearman、NDCG@10                     |
| 推荐       | Recent Popularity + Similarity + Trend   |
| 推荐评价     | Recall@12、MAP@12、HitRate@12              |
| 展示       | Top-K 趋势属性 + 某用户 Top-12 推荐               |

## 12.2 MVP 不做

| 内容                 | 原因           |
| ------------------ | ------------ |
| Neo4j              | 没必要          |
| GNN                | 过重           |
| LightGCN           | 推荐主任务会变复杂    |
| 双塔模型               | 训练和负采样复杂     |
| Transformer / LSTM | 周级属性样本较少，不稳定 |
| 图谱推理               | 不符合当前任务      |
| 图片特征               | 增加深度学习负担     |
| 文本 embedding       | 增加 NLP 负担    |

---

# 13. 答辩稳妥版本

在 MVP 基础上，答辩稳妥版建议增加：

| 增强内容           | 作用          |
| -------------- | ----------- |
| 完整属性层级边        | 体现属性图设计     |
| 父属性热度特征        | 体现图结构参与建模   |
| 同级属性均值特征       | 体现层级上下文     |
| LightGBM 特征重要性 | 增强可解释性      |
| 按属性类型分组评价      | 展示哪些属性更易预测  |
| 推荐消融实验         | 证明趋势分有作用    |
| 趋势曲线可视化        | 展示真实热度与预测热度 |
| 用户推荐解释         | 展示“为什么推荐”   |

答辩稳妥版本的展示页面建议包含：

| 页面    | 内容                          |
| ----- | --------------------------- |
| 趋势看板  | 下周上升颜色、品类、图案、服装组            |
| 属性详情  | 某属性最近 8 周热度曲线和预测值           |
| 属性图展示 | 某个商品连接到哪些属性                 |
| 推荐展示  | 输入 customer_id，输出 Top-12 商品 |
| 推荐理由  | 用户偏好属性 + 趋势属性 + 近期热门        |

---

# 14. 有余力时可以增强哪些内容

## 14.1 趋势预测增强

| 增强          | 是否推荐 | 说明               |
| ----------- | ---: | ---------------- |
| 按属性类型分别训练模型 |   推荐 | 颜色、品类、图案的规律不同    |
| 加入销售额热度     |   可选 | 用 `price` 加权     |
| 加入季节特征      |   推荐 | 服装有季节性           |
| 加入父级/同级图特征  |   推荐 | 体现属性图价值          |
| 加入趋势加速度     |   推荐 | 增强趋势变化刻画         |
| 多步预测        |   可选 | 预测未来 2/3 周，但风险增加 |

---

## 14.2 推荐增强

| 增强      | 是否推荐 | 说明             |
| ------- | ---: | -------------- |
| 权重网格搜索  |   推荐 | 比手工权重更有说服力     |
| 用户年龄段热门 |   可选 | 使用 `age` 做轻量增强 |
| 推荐覆盖率分析 |   推荐 | 避免只推热门         |
| 长尾推荐分析  |   可选 | 展示趋势分对长尾商品的帮助  |
| 商品图片展示  |   推荐 | 只用于网页展示，不进入模型  |

---

## 14.3 不建议增强

| 内容          | 原因        |
| ----------- | --------- |
| GNN         | 会转移主线     |
| 知识图谱推理      | 与当前属性图不匹配 |
| Neo4j 系统    | 工程负担大     |
| 双塔模型        | 推荐系统复杂度过高 |
| Transformer | 周级属性样本量有限 |
| 复杂 NLP      | 非主线       |

---

# 15. 推荐项目目录结构

```text
hm-fashion-trend-rec/
│
├── data/
│   ├── raw/
│   │   └── hm/
│   │       ├── transactions_train.csv
│   │       ├── articles.csv
│   │       ├── customers.csv
│   │       └── sample_submission.csv
│   │
│   ├── interim/
│   │   └── transactions_train_weekly.parquet
│   │
│   ├── processed/
│   │   ├── graph/
│   │   │   ├── nodes_article.csv
│   │   │   ├── nodes_attribute.csv
│   │   │   ├── edges_article_attribute.csv
│   │   │   └── edges_attribute_hierarchy.csv
│   │   │
│   │   ├── trend/
│   │   │   ├── article_week_sales.csv
│   │   │   ├── attribute_week_heat.csv
│   │   │   └── attribute_week_target.csv
│   │   │
│   │   ├── features/
│   │   │   ├── trend_model_samples.parquet
│   │   │   ├── trend_model_samples_train.parquet
│   │   │   ├── trend_model_samples_valid.parquet
│   │   │   └── trend_model_samples_test.parquet
│   │   │
│   │   └── recommend/
│   │       ├── time_windows.parquet
│   │       ├── target_users.parquet
│   │       ├── evaluation_labels.parquet
│   │       ├── user_profile.parquet
│   │       ├── metadata.json
│   │       ├── features/
│   │       │   └── <feature_name>/
│   │       │       └── strategy=<strategy>/
│   │       │           └── split=<split>/
│   │       │               └── cutoff_week=<week>/
│   │       │                   └── part.parquet
│   │       └── candidates/
│   │           └── <strategy>/
│   │               └── candidate_items.parquet
│
├── outputs/
│   ├── models/
│   │   └── <model>/
│   │       ├── predictions.csv
│   │       ├── metadata.json
│   │       └── params.json
│   │
│   └── metrics/
│       └── <model>/
│           └── trend_metrics.json
│
│   └── recommendation/
│       ├── <method>/
│       │   ├── recommendations.csv
│       │   ├── recommendation_items.parquet
│       │   ├── params.json
│       │   ├── metadata.json
│       │   └── metrics.json
│       └── experiments/
│           └── <experiment_id>/
│               └── experiment.json
│
│   └── reports/
│       ├── figures/
│       │   ├── data_pipeline.svg
│       │   ├── data_pipeline.png
│       │   ├── attribute_graph_schema.svg
│       │   ├── trend_curve_examples.svg
│       │   └── ...
│       ├── tables/
│       │   ├── trend_model_metrics.csv
│       │   ├── trend_model_metrics.md
│       │   ├── recommendation_method_metrics.csv
│       │   └── ...
│       ├── case_studies/
│       │   ├── case_01.json
│       │   ├── case_01.md
│       │   └── ...
│       └── manifest.json
│
├── src/
│   ├── fashion_trend/
│   │   ├── foundation/
│   │   │   ├── paths.py
│   │   │   ├── logging.py
│   │   │   ├── io.py
│   │   │   ├── dataframe.py
│   │   │   └── artifacts.py
│   │   ├── datasets/
│   │   ├── transactions/
│   │   ├── catalog/
│   │   ├── trend/
│   │   │   ├── heat/
│   │   │   ├── labels/
│   │   │   ├── features/
│   │   │   ├── splits/
│   │   │   ├── training/
│   │   │   ├── evaluation/
│   │   │   └── models/
│   │   │       ├── baselines/
│   │   │       └── supervised/
│   │   ├── recommendation/
│   │   └── reports/
│   │
│   ├── 01_data_check.py
│   ├── 02_build_weekly_transactions.py
│   ├── 03_clean_articles.py
│   ├── 04_build_attribute_graph.py
│   ├── 05_compute_article_week_sales.py
│   ├── 06_compute_attribute_week_heat.py
│   ├── 07_build_trend_targets.py
│   ├── 08_build_trend_model_samples.py
│   ├── 09_split_trend_model_samples.py
│   ├── 10_train_trend_model.py
│   ├── 11_eval_trend_model.py
│   ├── 12_build_recommendation_inputs.py
│   ├── 13_build_recommend_candidates.py
│   ├── 14_rerank_recommendations.py
│   ├── 15_eval_recommendations.py
│   ├── 16_run_recommendation_experiment.py
│   ├── 17_export_paper_assets.py
│   └── ...
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_trend_analysis.ipynb
│   └── 03_recommendation_demo.ipynb
│
├── requirements.txt
└── README.md
```

当前实现中，已落地的用户可运行脚本是 `src/00_*.py` 到 `src/17_*.py`。报告导出入口为 `src/17_export_paper_assets.py`，只读取已发布稳定 artifact，输出论文 figures、tables、case studies 和 manifest，不训练模型、不重跑推荐方法，也不是在线 dashboard。内部计算事实按业务域进入 `src/fashion_trend/`。默认路径根常量位于 `foundation.paths`；数据集、交易、catalog、trend、recommendation 和 reports 的业务路径由各自领域的 `paths.py` 持有。商品清洗和属性图在 `src/fashion_trend/catalog/`，趋势训练 runner 在 `src/fashion_trend/trend/training/`，趋势评价在 `src/fashion_trend/trend/evaluation/`，趋势模型实现和注册表在 `src/fashion_trend/trend/models/`；推荐时间窗口、输入、候选、排序、方法、评价和实验编排位于 `src/fashion_trend/recommendation/`；论文素材读取、表格、图表、案例和 manifest 编排位于 `src/fashion_trend/reports/`。

---

# 16. 每个阶段的预期产出文件

| 阶段          | 脚本                                  | 产出文件                                                                                                       |
| ----------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 数据检查        | `01_data_check.py`                  | 控制台行数日志，不写稳定文件产物                                                                      |
| 周切分         | `02_build_weekly_transactions.py`   | `data/interim/transactions_train_weekly.parquet`                                                           |
| articles 清洗 | `03_clean_articles.py`              | `articles_clean_mvp.csv`, `articles_clean.csv`                                                             |
| 属性图构建       | `04_build_attribute_graph.py`       | `nodes_article.csv`, `nodes_attribute.csv`, `edges_article_attribute.csv`, `edges_attribute_hierarchy.csv` |
| 商品周销量       | `05_compute_article_week_sales.py`  | `article_week_sales.csv`                                                                                   |
| 属性周热度       | `06_compute_attribute_week_heat.py` | `attribute_week_heat.csv`                                                                                  |
| 趋势标签        | `07_build_trend_targets.py`         | `attribute_week_target.csv`                                                                                |
| 趋势样本        | `08_build_trend_model_samples.py`   | `trend_model_samples.parquet`                                                                              |
| baseline 训练 | `10_train_trend_model.py --model <model>` | `outputs/models/<model>/predictions.csv`, `params.json`, `metadata.json`                                   |
| LightGBM 训练 | 已注册实现：复用 `10_train_trend_model.py --model lightgbm` | `outputs/models/lightgbm/predictions.csv`, `params.json`, `metadata.json`, `feature_importance.csv`, `model.txt` |
| LightGBM 调参 run | `10_train_trend_model.py --model lightgbm --run-id <run_id> --no-promote` | `outputs/models/lightgbm/runs/<run_id>/...` |
| LightGBM run 评价 | `11_eval_trend_model.py --model lightgbm --run-id <run_id>` | `outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json` |
| 已评估 run 发布 | `10_train_trend_model.py --model lightgbm --promote-run <run_id>` | 发布到 `outputs/models/lightgbm/` 和 `outputs/metrics/lightgbm/trend_metrics.json` |
| 趋势评价        | `11_eval_trend_model.py`            | `outputs/metrics/<model>/trend_metrics.json`                                                               |
| 推荐输入        | `src/12_build_recommendation_inputs.py` | `time_windows.parquet`, `target_users.parquet`, `evaluation_labels.parquet`, `user_profile.parquet`, `data/processed/recommend/metadata.json` |
| 候选召回        | `src/13_build_recommend_candidates.py --strategy <strategy>` | `data/processed/recommend/candidates/<strategy>/candidate_items.parquet` |
| 推荐特征缓存     | `src/16_run_recommendation_experiment.py --experiment main --force-cache` | `data/processed/recommend/features/<feature_name>/strategy=<strategy>/split=<split>/cutoff_week=<week>/part.parquet` |
| 推荐重排序       | `src/14_rerank_recommendations.py --method <method>` | `outputs/recommendation/<method>/recommendations.csv`, `recommendation_items.parquet`, `params.json`, `metadata.json` |
| 推荐评价        | `src/15_eval_recommendations.py --method <method>` | `outputs/recommendation/<method>/metrics.json`                                                             |
| 推荐实验        | `src/16_run_recommendation_experiment.py --experiment main` | `outputs/recommendation/experiments/<experiment_id>/experiment.json`                                       |
| 报告导出        | `src/17_export_paper_assets.py` | `outputs/reports/figures/*.{svg,png}`, `outputs/reports/tables/*.{csv,md}`, `outputs/reports/case_studies/*.{json,md}`, `outputs/reports/manifest.json` |

当前 `main` 推荐实验使用 valid split 的 `NDCG@12` 选择
`pop_similarity_trend` 权重。2026-05-11 的有限网格调参搜索 25 组权重，最佳权重为
`pop_score=0.2`、`sim_score=0.2`、`trend_score=0.1`、`recent_score=0.5`。
该主模型 valid `NDCG@12=0.005922`，高于 `recent_popularity` 的
valid `NDCG@12=0.005715`；test `NDCG@12=0.007987`，明显高于旧主模型，
但略低于 `recent_popularity` 的 test `NDCG@12=0.008087`。报告中应把
`recent_popularity` 作为强 baseline，而不是把趋势感知模型描述成所有指标均最优。

---

# 17. 从数据处理到实验完成的开发顺序

## 第 1 步：检查原始数据文件

目标：

```text
确认三张原始 CSV 存在、可读，并打印行数摘要。
```

先写：

```text
src/01_data_check.py
```

输出：

```text
控制台行数日志；不写入稳定文件产物。
```

---

## 第 2 步：生成周级交易表

目标：

```text
给每条交易加 week_id。
```

写：

```text
src/02_build_weekly_transactions.py
```

输出：

```text
transactions_train_weekly.parquet
```

---

## 第 3 步：清洗 articles 商品表

目标：

```text
从 articles.csv 生成后续属性图使用的清洗商品表。
```

写：

```text
src/03_clean_articles.py
```

输出：

```text
articles_clean_mvp.csv
articles_clean.csv
```

---

## 第 4 步：构建静态商品属性图

目标：

```text
从 articles.csv 生成节点表和边表。
```

写：

```text
src/04_build_attribute_graph.py
```

输出：

```text
nodes_article.csv
nodes_attribute.csv
edges_article_attribute.csv
edges_attribute_hierarchy.csv
```

---

## 第 5 步：计算商品周销量

目标：

```text
按 week_id + article_id 聚合销量。
```

写：

```text
src/05_compute_article_week_sales.py
```

输出：

```text
article_week_sales.csv
```

---

## 第 6 步：计算属性周热度

目标：

```text
将商品销量通过 article-attribute 边映射到属性节点。
```

写：

```text
src/06_compute_attribute_week_heat.py
```

输出：

```text
attribute_week_heat.csv
```

---

## 第 7 步：构造趋势标签

目标：

```text
计算 target_growth。
```

写：

```text
src/07_build_trend_targets.py
```

输出：

```text
attribute_week_target.csv
```

---

## 第 8 步：构造趋势训练样本

目标：

```text
生成 lag、移动平均、增长率、图结构特征。
```

写：

```text
src/08_build_trend_model_samples.py
```

输出：

```text
trend_model_samples.parquet
```

---

## 第 9 步：先跑 baseline

目标：

```text
确认趋势预测任务能跑通。
```

写：

```text
src/10_train_trend_model.py --model last_week
src/10_train_trend_model.py --model previous_growth
src/10_train_trend_model.py --model moving_average
```

当前实现中，`last_week` 是当前 share 热度持平/归一化基线：
`pred_share_t1 = group_normalize(share_t)`，再由预测份额和当前份额派生
`pred_target_growth`；`previous_growth` 使用 `growth_lag_1` 预测 `target_growth`；
`moving_average` 使用最近两段增长的均值作为平滑增长 baseline。`previous_growth`
和 `moving_average` 派生 `pred_share_t1` 时，先按 `target_growth` 的逆运算得到原始
预测占比，再在同一 `split/week_id/attr_type` 内对非负原始值归一化，保证输出是合法占比分布。

必须实现：

```text
Last Week Heat
Previous Growth
Moving Average
```

---

## 第 10 步：训练 LightGBM 主模型

目标：

```text
训练主线趋势预测模型。
```

`lightgbm` 已注册到统一趋势模型训练和评价入口：

```sh
src/10_train_trend_model.py --model lightgbm
src/11_eval_trend_model.py --model lightgbm
```

标准模型产物位于：

```text
outputs/models/lightgbm/predictions.csv
outputs/models/lightgbm/metadata.json
outputs/models/lightgbm/params.json
outputs/models/lightgbm/feature_importance.csv
outputs/models/lightgbm/model.txt
```

趋势评价产物位于：

```text
outputs/metrics/lightgbm/trend_metrics.json
```

---

## 第 11 步：趋势预测评价

目标：

```text
当前比较三类 baseline 和 LightGBM 主模型，均复用同一趋势评价入口。
```

当前实现入口：

```text
src/11_eval_trend_model.py
```

当前标准产物：

```text
outputs/metrics/<model>/trend_metrics.json
```

至少包含：

```text
MAE
RMSE
Spearman
Precision@10
NDCG@10
```

---

## 已实现第 12 步：构建推荐输入

目标：

```text
生成推荐时间窗口、eligible 用户、评价标签和用户历史属性画像。
```

写：

```text
src/12_build_recommendation_inputs.py
```

输出：

```text
data/processed/recommend/time_windows.parquet
data/processed/recommend/target_users.parquet
data/processed/recommend/evaluation_labels.parquet
data/processed/recommend/user_profile.parquet
```

---

## 已实现第 13 步：构造推荐候选集

目标：

```text
按候选 strategy 生成用户候选商品，候选池与推荐 method 解耦。
```

写：

```text
src/13_build_recommend_candidates.py --strategy <strategy>
```

候选来源：

```text
近期热门
用户属性相似
趋势属性商品
```

输出：

```text
data/processed/recommend/candidates/<strategy>/candidate_items.parquet
```

---

## 已实现第 14 步：线性重排序

目标：

```text
按推荐 method 读取默认候选 strategy，用 Pop + Sim + Trend + Recent 给候选商品打分。
```

写：

```text
src/14_rerank_recommendations.py --method <method>
```

输出：

```text
outputs/recommendation/<method>/recommendations.csv
outputs/recommendation/<method>/recommendation_items.parquet
outputs/recommendation/<method>/params.json
outputs/recommendation/<method>/metadata.json
```

`recommendation_items.parquet` 是默认内部长表产物；`recommendation_items.csv`
仅在显式导出时生成。

格式：

| customer_id | prediction                |
| ----------- | ------------------------- |
| xxx         | 0706016001 0507909001 ... |

---

## 已实现第 15 步：推荐评价

目标：

```text
验证趋势分是否提升推荐效果。
```

写：

```text
src/15_eval_recommendations.py --method <method>
```

输出：

```text
outputs/recommendation/<method>/metrics.json
```

指标：

```text
MAP@12
Recall@12
HitRate@12
NDCG@12
Coverage
```

---

## 已实现第 16 步：运行推荐实验

目标：

```text
编排轻量离线推荐实验，运行主方法、评价和消融汇总。
```

写：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
uv run python src/16_run_recommendation_experiment.py --experiment main --force-method pop_similarity
uv run python src/16_run_recommendation_experiment.py --experiment main --force-cache
uv run python src/16_run_recommendation_experiment.py --experiment main --force-candidates
uv run python src/16_run_recommendation_experiment.py --experiment main --force-rebuild-all
```

输出：

```text
outputs/recommendation/experiments/<experiment_id>/experiment.json
```

---

# 18. 实验设计建议

## 18.1 趋势预测实验

| 实验                                      | 目的               |
| --------------------------------------- | ---------------- |
| Last Week Heat vs Previous Growth vs Moving Average vs LightGBM | 证明主模型优于简单基线      |
| 使用 `heat_cnt` vs `heat_share`           | 证明标准化热度更适合趋势     |
| 无图特征 vs 有图特征                            | 证明属性图有价值         |
| 按属性类型评价                                 | 分析颜色、品类、图案哪个更易预测 |
| Top-K 趋势属性可视化                           | 展示模型解释性          |

---

## 18.2 推荐实验

| 实验                       | 目的          |
| ------------------------ | ----------- |
| Global Popularity        | 最简单推荐基线     |
| Recent Popularity        | 时间感知热门基线    |
| Pop + Similarity         | 个性化但无趋势     |
| Pop + Similarity + Trend | 最终趋势感知推荐    |
| 不同 $\gamma$ 权重           | 分析趋势分对推荐的影响 |

---

## 18.3 消融实验

当前实现把推荐层消融与趋势模型特征消融分开处理。已实现的推荐层消融包括
`Full Model`、严格 `w/o Trend in Rec`、严格 `w/o Similarity`、严格
`w/o Recent` 和稳定 baseline 对照。`w/o Graph`、`w/o Growth`、`w/o Rank`
属于趋势模型特征消融，需要重训 LightGBM 变体，不属于本轮推荐消融实现。

| 模型版本             | 去掉的部分         | 当前状态       |
| ---------------- | ------------- | ---------- |
| Full Model       | 全部推荐融合特征      | 已在推荐实验输出   |
| w/o Trend in Rec | 推荐阶段去掉趋势分     | 已在推荐实验输出   |
| w/o Similarity   | 推荐阶段去掉相似度分    | 已在推荐实验输出   |
| w/o Recent       | 推荐阶段去掉近期热度分   | 已在推荐实验输出   |
| w/o Graph        | 去掉父级、同级、图结构特征 | 后续趋势模型特征消融 |
| w/o Growth       | 去掉历史增长率       | 后续趋势模型特征消融 |
| w/o Rank         | 去掉同类型排名       | 后续趋势模型特征消融 |

---

# 19. 论文方法章节可以这样安排

```text
第 3 章 方法设计

3.1 任务定义
    3.1.1 属性级周趋势预测任务
    3.1.2 趋势感知 Top-N 推荐任务

3.2 数据预处理与时间切分
    3.2.1 H&M 数据表说明
    3.2.2 周级时间窗口构建
    3.2.3 时间泄漏规避

3.3 商品属性层次图构建
    3.3.1 节点类型定义
    3.3.2 商品—属性边构建
    3.3.3 属性层级边构建
    3.3.4 静态图与动态热度权重

3.4 属性周热度建模
    3.4.1 原始热度定义
    3.4.2 同类型归一化热度
    3.4.3 趋势标签定义

3.5 属性趋势预测模型
    3.5.1 趋势预测样本构造
    3.5.2 baseline 方法
    3.5.3 LightGBM 趋势预测模型

3.6 趋势感知 Top-N 推荐
    3.6.1 候选集召回
    3.6.2 用户属性偏好建模
    3.6.3 商品趋势分计算
    3.6.4 线性加权重排序

3.7 评价指标
    3.7.1 趋势预测评价指标
    3.7.2 推荐评价指标
```

---

# 20. 关键实现注意事项

## 20.1 必须避免时间泄漏

| 易泄漏点           | 正确做法                 |
| -------------- | -------------------- |
| 计算趋势标签时用了未来周   | 标签可以用未来周，但特征不能用未来周   |
| 推荐候选集用了测试周商品销量 | 候选集只能用测试周之前的数据       |
| 用户画像用了测试周购买    | 用户画像只能用历史购买          |
| 热门商品用了测试周销量    | 只能用历史窗口销量            |
| 标准化时用了全量时间统计   | 尽量在训练/验证/测试各自历史窗口内计算 |

---

## 20.2 属性热度表要补 0

如果某属性某周没有交易，需要补 0。

否则会导致：

* lag 特征错位；
* 增长率计算错误；
* 模型只看到活跃周，产生偏差。

做法：

```text
所有 week_id × 所有 attr_id 笛卡尔积
缺失 heat_cnt 填 0
```

---

## 20.3 展示 Top-K 趋势属性时要过滤低频属性

否则容易出现：

```text
某属性从 1 次涨到 5 次，增长率很高
```

但这种趋势没有业务意义。

展示前建议过滤：

```text
heat_t >= 20
total_heat >= 100
active_weeks >= 8
```

---

## 20.4 推荐时是否过滤用户已购买商品

有两种做法：

| 做法     | 说明                     |
| ------ | ---------------------- |
| 不过滤已购买 | 更接近 H&M 原任务，因为用户可能重复购买 |
| 过滤已购买  | 更符合应用展示中的“推荐新品”直觉      |

建议：

```text
实验评价：不强制过滤
网页展示：可以提供过滤开关
```

---

# 21. 最终你应该达到的成果

## 21.1 必须有的成果

| 成果           | 说明                            |
| ------------ | ----------------------------- |
| 静态商品属性图      | 节点表、边表、图示                     |
| 属性周热度序列      | `attribute_week_heat.csv`     |
| 趋势预测数据集      | `trend_model_samples.parquet` |
| baseline 实验  | Last Week Heat、Previous Growth、Moving Average |
| LightGBM 主模型 | 趋势预测结果                        |
| 趋势评价表        | MAE、Spearman、NDCG@K           |
| 趋势属性榜单       | 下周上升颜色、品类、图案                  |
| 轻量推荐模块       | Top-12 商品推荐                   |
| 推荐评价表        | MAP@12、Recall@12、HitRate@12   |
| 推荐解释案例       | 用户历史偏好 + 趋势属性 + 推荐商品          |

---

## 21.2 论文中最重要的创新点表述

可以写成：

> 本文没有直接复现复杂个性化推荐模型，而是将 H&M 交易数据重新组织为服装属性周级趋势预测任务。通过构建商品属性层次图，将商品交易行为按周聚合到属性节点上，形成属性级动态热度序列；在此基础上，结合历史热度、增长率、属性层级和图结构特征，使用 LightGBM 预测下一周属性趋势，并将预测趋势分作为轻量 Top-N 推荐的重排序因子，实现趋势预测与推荐应用的结合。

---

# 22. 历史开发顺序与当前下一步

本节保留基础脚本的历史开发顺序，用来说明数据流如何从原始数据推进到趋势训练样本。
当前实现状态已经推进到 `00` 到 `16` 号入口：基础数据流、三类 baseline、
LightGBM 主模型、趋势评价和轻量离线推荐实验均已落地。

推荐模块已包含输入构建、strategy-scoped 候选、method-scoped 推荐输出、单方法评价和 `main` 实验编排。基础脚本的历史顺序如下：

## 第 1 个脚本：`01_data_check.py`

功能：

* 定位三张原始 CSV；
* 检查原始 CSV 文件存在且可读；
* 输出行数日志摘要。

---

## 第 2 个脚本：`02_build_weekly_transactions.py`

功能：

* `t_dat` 转日期；
* 生成 `week_id`；
* 保存周级交易基础表。

---

## 第 3 个脚本：`03_clean_articles.py`

功能：

* 选择核心属性字段；
* 输出后续属性图使用的清洗商品表。

---

## 第 4 个脚本：`04_build_attribute_graph.py`

功能：

* 构造 `nodes_article.csv`；
* 构造 `nodes_attribute.csv`；
* 构造 `edges_article_attribute.csv`；
* 构造 `edges_attribute_hierarchy.csv`。

---

## 第 5 个脚本：`05_compute_article_week_sales.py`

功能：

* 按 `week_id + article_id` 聚合；
* 得到商品每周销量。

---

## 第 6 个脚本：`06_compute_attribute_week_heat.py`

功能：

* 连接 `article_week_sales.csv` 和 `edges_article_attribute.csv`；
* 按 `week_id + attr_id` 聚合；
* 计算 `heat_cnt`、`heat_share`、`log_heat`、`rank_in_type`。

---

## 第 7 个脚本：`07_build_trend_targets.py`

功能：

* 对每个属性按周排序；
* 计算下一周 `share_t1`；
* 计算：

$$
\mathrm{target\_growth} = \log \frac{\mathrm{share}_{t+1}+\epsilon}{\mathrm{share}_{t}+\epsilon}
$$

---

## 第 8 个脚本：`08_build_trend_model_samples.py`

功能：

* 构造 lag 特征；
* 构造 moving average；
* 构造历史增长率；
* 加入属性静态特征；
* 加入父属性热度特征。

完成趋势样本构建并执行时间切分后，主任务数据集就已经成型，可以开始训练三类 baseline 和已注册的 LightGBM 主模型。

[1]: https://huggingface.co/datasets/davanstrien/datasets_with_metadata_and_summaries/viewer "davanstrien/datasets_with_metadata_and_summaries · Datasets at Hugging Face"
[2]: https://awinml.github.io/h-m-personalized-product-recommendations/ "H&M Personalized Product Recommendations"
[3]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html "lightgbm.LGBMRegressor — LightGBM 4.6.0.99 documentation"
