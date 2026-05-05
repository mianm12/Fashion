# Fashion

时尚趋势与轻量推荐实验项目，当前围绕 Kaggle H&M 个性化时尚推荐数据集构建周级交易表、商品属性层次图、商品周销量、属性周热度和基础属性趋势预测 baseline，为后续趋势感知 Top-N 推荐做准备。

## 研究主线

本项目参考 `docs/gpt-research/implementation-plan.md`，把原始 H&M 推荐任务收缩为更容易复现和解释的两段式课题：

```text
H&M articles.csv
    -> 商品属性层次图
H&M transactions_train.csv
    -> 周级商品销量
    -> 属性周热度
    -> 属性趋势预测
    -> 趋势感知 Top-N 推荐
```

现阶段已经完成到趋势 `last_week` baseline 闭环：

| 阶段 | 状态 | 主要产物 |
| :--- | :--- | :--- |
| 数据下载 | 已实现 | `data/raw/h-and-m-personalized-fashion-recommendations/` |
| 周级交易表 | 已实现 | `data/interim/transactions_train_weekly.parquet` |
| articles 清洗 | 已实现 | `articles_clean_mvp.csv`、`articles_clean.csv` |
| 商品属性层次图 | 已实现 | `nodes_article.csv`、`nodes_attribute.csv`、`edges_article_attribute.csv`、`edges_attribute_hierarchy.csv` |
| 商品周销量 | 已实现 | `article_week_sales.csv` |
| 属性周热度 | 已实现 | `attribute_week_heat.csv` |
| 趋势标签 | 已实现 | `attribute_week_target.csv` |
| 趋势样本 | 已实现 | `trend_model_samples.parquet` |
| 趋势样本时间切分 | 已实现 | `trend_model_samples_train.parquet`、`trend_model_samples_valid.parquet`、`trend_model_samples_test.parquet` |
| Last Week baseline | 已实现 | `outputs/models/last_week/predictions.csv`、`params.json`、`metadata.json` |
| 推荐评价 | 尚未实现 | 后续推荐结果 |

## 数据集

项目使用 Kaggle 比赛数据集：`h-and-m-personalized-fashion-recommendations`。

该数据集来自 H&M 个性化时尚推荐比赛，主要用于根据用户历史交易、商品信息和用户信息预测用户未来可能购买的商品。下载后通常会包含以下类型的数据文件：

- `transactions_train.csv`：用户历史交易记录。
- `articles.csv`：商品信息，包括商品编号、类别、颜色、描述等字段。
- `customers.csv`：用户信息。
- `sample_submission.csv`：Kaggle 提交格式示例。
- 商品图片文件：部分数据包中会包含按商品编号组织的图片资源。

数据文件体积较大，`data/` 目录默认不会包含到版本库。

### 下载数据集

下载入口位于：

```sh
src/00_download_data.py
```

默认配置集中在 `src/fashion_trend/config.py`：

- `DEFAULT_COMPETITION`：`h-and-m-personalized-fashion-recommendations`
- `RAW_DIR`：项目根目录下的 `data/raw`

默认会下载 `h-and-m-personalized-fashion-recommendations`，并保存到项目根目录下的：

```sh
data/raw/h-and-m-personalized-fashion-recommendations/
```

脚本会自动创建目标目录；如果目标目录已经存在且非空，默认会跳过重新下载。需要重新下载时使用 `--force`。

#### 1. 准备环境

项目使用 Python 3.10 至 3.12，并通过 `uv` 管理依赖。首次运行前先安装依赖：

```sh
uv sync
```

首次下载 Kaggle 数据前，需要确保当前环境已经具备 Kaggle 访问权限，并且已在 Kaggle 页面接受该比赛的数据使用规则。

如果需要配置 Kaggle API 凭据，请只放在本机安全位置或环境变量中，不要将 `kaggle.json`、Token、密码等敏感信息提交到仓库。

#### 2. 执行下载

```sh
uv run python src/00_download_data.py
```

脚本会自动创建目标目录，并默认解压下载得到的 zip 文件。

#### 3. 常用参数

指定其他 Kaggle competition slug：

```sh
uv run python src/00_download_data.py --competition h-and-m-personalized-fashion-recommendations
```

指定数据保存根目录：

```sh
uv run python src/00_download_data.py --data-dir data/raw
```

仅下载，不自动解压：

```sh
uv run python src/00_download_data.py --no-unzip
```

即使目标目录已有文件，也重新下载：

```sh
uv run python src/00_download_data.py --force
```

查看完整命令帮助：

```sh
uv run python src/00_download_data.py --help
```

#### 4. 下载行为说明

- 默认目标目录：项目根目录下的 `data/raw/<competition>/`。
- 默认行为：目标目录不存在或为空时下载；目标目录已有内容时跳过下载。
- `--force`：忽略已有内容，重新调用 Kaggle 下载。
- `--no-unzip`：保留下载得到的 zip 文件，不执行解压。
- 默认解压：脚本会解压目标目录下的 zip 文件，并拒绝解压会逃逸出目标目录的异常路径。

### 下载目录结构

下载完成后，目录大致如下：

```text
data/
+-- raw/
    +-- h-and-m-personalized-fashion-recommendations/
        +-- articles.csv
        +-- customers.csv
        +-- sample_submission.csv
        +-- transactions_train.csv
        +-- ...
```

## 数据预处理

当前已实现流水线按下面顺序运行：

```sh
uv run python src/00_download_data.py
uv run python src/02_build_weekly_transactions.py
uv run python src/03_clean_articles.py
uv run python src/04_build_attribute_graph.py
uv run python src/05_compute_article_week_sales.py
uv run python src/06_compute_attribute_week_heat.py
uv run python src/07_build_trend_targets.py
uv run python src/08_build_trend_model_samples.py
uv run python src/09_split_trend_model_samples.py
uv run python src/10_train_trend_baseline.py --model last_week
```

### 1. transactions_train.csv

| 字段               | 处理方式               | 用途               |
| :----------------- | :--------------------- | :----------------- |
| `t_dat`            | 转日期，生成 `week_id` | 时间划分、周级聚合 |
| `customer_id`      | 保留字符串             | 推荐任务用户标识   |
| `article_id`       | 转字符串，保留前导 0   | 连接商品表         |
| `price`            | 保留，可选使用         | 可选销售额热度     |
| `sales_channel_id` | 保留，可选统计         | 可选渠道分析       |

利用`t_dat`生成`week_id`:

$$
\mathrm{week\_id} = \left\lfloor \frac{t_{\mathrm{dat}} - t_{\min}}{7} \right\rfloor
$$

输出文件:

```sh
data/interim/transactions_train_weekly.parquet
```

运行命令:

```sh
uv run python src/02_build_weekly_transactions.py
```

### 2. articles.csv

| 字段                           | 当前 MVP 是否保留 | 当前稳妥版是否保留 | 说明             | 推荐用途                                           |
| ------------------------------ | ----------------: | -----------------: | ---------------- | -------------------------------------------------- |
| `article_id`                   |              必须 |               必须 | 商品唯一编号     | 连接 `transactions_train.csv`，构建商品节点        |
| `product_code`                 |              保留 |               保留 | 商品款式族编号   | 可分析同款不同色；当前用于审查和后续扩展           |
| `prod_name`                    |            展示用 |             展示用 | 商品名称         | 推荐结果展示，不建议作为模型特征                   |
| `product_type_no`              |                否 |                 否 | 商品类型编号     | 与 `product_type_name` 对应，主要作映射            |
| `product_type_name`            |                是 |                 是 | 商品具体类型     | 核心属性字段，适合做品类趋势                       |
| `product_group_name`           |                是 |                 是 | 商品大类         | 可与 `product_type_name` 构成品类层级              |
| `graphical_appearance_no`      |                否 |                 否 | 图案外观编号     | 与 `graphical_appearance_name` 对应，主要作映射    |
| `graphical_appearance_name`    |                是 |                 是 | 图案 / 外观      | 适合分析 Solid、Stripe、Print 等风格趋势           |
| `colour_group_code`            |                否 |                 否 | 颜色编号         | 与 `colour_group_name` 对应，主要作映射            |
| `colour_group_name`            |                是 |                 是 | 具体颜色         | 核心属性字段，适合做颜色趋势                       |
| `perceived_colour_value_id`    |                否 |                 否 | 感知颜色明暗编号 | 与 `perceived_colour_value_name` 对应，主要作映射  |
| `perceived_colour_value_name`  |                否 |                 否 | 感知颜色明暗     | 可作为后续增强字段，当前清洗表未保留               |
| `perceived_colour_master_id`   |                否 |                 否 | 主色系编号       | 与 `perceived_colour_master_name` 对应，主要作映射 |
| `perceived_colour_master_name` |                否 |                 是 | 主色系           | 可与 `colour_group_name` 构成颜色层级              |
| `department_no`                |                否 |                 否 | 部门编号         | 与 `department_name` 对应，主要作映射              |
| `department_name`              |                否 |                 是 | 商品部门         | 粒度较细，容易稀疏；当前用于属性层级边             |
| `index_code`                   |                否 |                 否 | 业务线编号       | 与 `index_name` 对应，主要作映射                   |
| `index_name`                   |                否 |                 是 | 业务线           | 可用于构建组织层级，解释性较好                     |
| `index_group_no`               |                否 |                 否 | 业务大类编号     | 与 `index_group_name` 对应，主要作映射             |
| `index_group_name`             |                否 |                 是 | 业务大类         | 可区分 Ladieswear、Menswear、Baby/Children 等大类  |
| `section_no`                   |                否 |                 否 | 商品区域编号     | 与 `section_name` 对应，主要作映射                 |
| `section_name`                 |                否 |                 是 | 商品区域         | 适合构建组织层级，解释性较强                       |
| `garment_group_no`             |                否 |                 否 | 服装组别编号     | 与 `garment_group_name` 对应，主要作映射           |
| `garment_group_name`           |                是 |                 是 | 服装组别         | 核心属性字段，适合分析服装风格 / 材质趋势          |
| `detail_desc`                  |                否 |                 否 | 商品文本描述     | 可用于后续 NLP 增强；当前清洗表未保留              |

本轮会先对 `articles.csv` 做字段过滤和基础校验，在 `data/interim/` 下生成两份中间表，再基于中间表构建属性图。

MVP 版中间表只保留商品标识、展示字段和 5 个核心属性字段：

| 字段 | 处理方式 | 用途 |
| :--- | :--- | :--- |
| `article_id` | 转字符串，保留前导 0 | 连接交易表、构建商品节点 |
| `product_code` | 转字符串 | 审查同款族，暂不作为核心属性 |
| `prod_name` | 保留字符串 | 推荐结果展示 |
| `product_group_name` | 保留字符串 | 核心属性，商品大类 |
| `product_type_name` | 保留字符串 | 核心属性，商品小类 |
| `garment_group_name` | 保留字符串 | 核心属性，服装组别 |
| `colour_group_name` | 保留字符串 | 核心属性，具体颜色 |
| `graphical_appearance_name` | 保留字符串 | 核心属性，图案 / 外观 |

输出文件:

```sh
data/interim/articles_clean_mvp.csv
```

稳妥版中间表在 MVP 版基础上增加 5 个层级属性字段，用于后续构建属性层级边：

| 字段 | 处理方式 | 用途 |
| :--- | :--- | :--- |
| `perceived_colour_master_name` | 保留字符串 | 与 `colour_group_name` 构成颜色层级 |
| `index_group_name` | 保留字符串 | 与 `index_name` 构成业务层级 |
| `index_name` | 保留字符串 | 与 `section_name` 构成业务线到区域层级 |
| `section_name` | 保留字符串 | 与 `department_name` 构成区域到部门层级 |
| `department_name` | 保留字符串 | 部门层级叶子属性 |

输出文件:

```sh
data/interim/articles_clean.csv
```

运行命令:

```sh
uv run python src/03_clean_articles.py
```

清洗规则:

- 只输出本轮所需字段，不携带 `detail_desc`、编号映射字段或图片字段。
- `article_id` 必须按字符串读取，避免丢失前导 0。
- 输出字段存在缺失值时直接失败，不静默填充。
- `articles_clean_mvp.csv` 和 `articles_clean.csv` 的行数、`article_id` 集合必须与原始 `articles.csv` 保持一致。
- 中间表和属性图 CSV 全字段使用双引号引用，避免 VS Code / DuckDB 等工具按前几万行采样时把后续含逗号的属性值误解析成额外列。

### 3. 商品属性层次图

基于 `data/interim/articles_clean.csv` 构建静态商品属性层次图。这里不引入 Neo4j，而是使用可审查的节点表和边表 CSV。

运行命令:

```sh
uv run python src/04_build_attribute_graph.py
```

输出文件:

```sh
data/processed/graph/nodes_article.csv
data/processed/graph/nodes_attribute.csv
data/processed/graph/edges_article_attribute.csv
data/processed/graph/edges_attribute_hierarchy.csv
```

#### nodes_article.csv

| 字段 | 说明 |
| :--- | :--- |
| `article_id` | 原始商品 ID，保留前导 0 |
| `article_node_id` | 图中商品节点 ID，格式为 `article_<article_id>` |
| `product_code` | 商品款式族编号 |
| `prod_name` | 商品名称，用于展示 |

#### nodes_attribute.csv

| 字段 | 说明 |
| :--- | :--- |
| `attr_id` | 属性节点唯一 ID，格式为 `attr_type::attr_value` |
| `attr_type` | 属性类型，即来源字段名 |
| `attr_value` | 属性取值 |
| `attr_node_id` | 图中属性节点 ID，当前与 `attr_id` 一致 |
| `article_count` | 关联到该属性的商品数量 |
| `is_core_attr` | 5 个核心属性为 `1`，层级增强属性为 `0` |
| `level` | 属性在层级图中的角色：`parent`、`child`、`parent_child` 或 `flat` |

#### edges_article_attribute.csv

每个商品会连接到 `articles_clean.csv` 中 10 个属性字段，因此该表是后续把商品销量映射到属性热度的核心桥梁。

| 字段 | 说明 |
| :--- | :--- |
| `article_id` | 原始商品 ID |
| `article_node_id` | 商品节点 ID |
| `attr_id` | 属性节点 ID |
| `attr_type` | 属性类型 |
| `attr_value` | 属性取值 |
| `edge_type` | 商品到属性的边类型，如 `has_colour_group` |
| `edge_weight` | 当前固定为 `1.0` |

#### edges_attribute_hierarchy.csv

属性层级边基于同一商品中的父子属性共现关系生成，`edge_weight` 表示父子组合关联的商品数量。

| 父字段 | 子字段 | 关系 |
| :--- | :--- | :--- |
| `product_group_name` | `product_type_name` | `product_group_contains_type` |
| `perceived_colour_master_name` | `colour_group_name` | `colour_master_contains_colour` |
| `index_group_name` | `index_name` | `index_group_contains_index` |
| `index_name` | `section_name` | `index_contains_section` |
| `section_name` | `department_name` | `section_contains_department` |

图构建会校验商品-属性边和属性层级边都能引用到已生成的节点；如果引用不完整会直接失败。

### 4. article_week_sales.csv

基于 `data/interim/transactions_train_weekly.parquet`，按 `week_id + article_id` 聚合每个商品每周购买次数、购买用户数和销售额。

输出文件:

```sh
data/processed/trend/article_week_sales.csv
```

运行命令:

```sh
uv run python src/05_compute_article_week_sales.py
```

输出字段:

| 字段 | 说明 |
| :--- | :--- |
| `week_id` | 周编号 |
| `article_id` | 商品唯一编号，保留前导 0 |
| `sales_cnt` | 商品每周购买次数 |
| `sales_user_cnt` | 商品每周购买用户数 |
| `sales_amount` | 商品每周销售额 |

### 5. attribute_week_heat.csv

基于 `article_week_sales.csv` 和 `data/processed/graph/edges_article_attribute.csv`，使用商品周销量中的 `sales_cnt` 作为购买次数热度，将商品热度映射到商品关联属性节点。

`attribute_week_heat.csv` 是完整属性-周面板：每个 `week_id` 都覆盖 `nodes_attribute.csv` 中的全部 `attr_id`。无观测购买的属性周不会被丢弃，而是保留 `heat_cnt = 0`、`heat_share = 0`、`log_heat = 0`。

输出文件:

```sh
data/processed/trend/attribute_week_heat.csv
```

运行命令:

```sh
uv run python src/06_compute_attribute_week_heat.py
```

输出字段:

| 字段 | 说明 |
| :--- | :--- |
| `week_id` | 周编号 |
| `attr_id` | 属性节点编号 |
| `attr_type` | 属性类型 |
| `attr_value` | 属性值 |
| `heat_cnt` | 关联商品 `sales_cnt` 求和 |
| `type_total_heat` | 同类型属性本周 `heat_cnt` 总和 |
| `heat_share` | `heat_cnt / type_total_heat` |
| `log_heat` | `log1p(heat_cnt)` |
| `rank_in_type` | 属性在同类型内的周热度排名 |

计算规则:

- `heat_cnt`：某属性关联商品在该周的 `sales_cnt` 求和。
- `type_total_heat`：同一 `week_id + attr_type` 下所有属性的 `heat_cnt` 总和。
- `heat_share`：属性在同类型属性内的热度占比。
- `rank_in_type`：同一 `week_id + attr_type` 下按 `heat_cnt` 降序排名，热度相同则按 `attr_id` 稳定排序。

属性周热度默认覆盖当前属性图中的全部 10 个属性字段。后续如果只分析 MVP 核心属性，可通过 `nodes_attribute.csv` 的 `is_core_attr = 1` 过滤。

### 6. attribute_week_target.csv

基于完整 `attribute_week_heat.csv` 构造下一周趋势标签。输出文件：

```sh
data/processed/trend/attribute_week_target.csv
```

每行表示一个属性在当前周 `t` 的目标，包含下一周热度、下一周热度占比、占比增长和下一周排名等字段。其中占比增长使用平滑后的对数增长：

```text
target_growth = log((share_t1 + 1e-6) / (share_t + 1e-6))
```

### 7. trend_model_samples.parquet

基于完整 `attribute_week_heat.csv` 和 `attribute_week_target.csv` 构造趋势训练样本。输出文件：

```sh
data/processed/features/trend_model_samples.parquet
```

样本表只使用当前周及历史周特征，包括 lag、移动平均、增长率、图结构和时间特征；`t+1` 信息只保留在目标字段中，避免训练特征泄漏未来信息。

### 8. trend_model_samples_train/valid/test.parquet

基于 `trend_model_samples.parquet` 按时间顺序切分训练、验证和测试样本。默认最后 8 个样本周为 test，之前 8 个样本周为 valid，更早样本周为 train。切分配置集中在 `src/fashion_trend/config.py`：

```text
TREND_SPLIT_VALID_WEEKS = 8
TREND_SPLIT_TEST_WEEKS = 8
```

输出文件：

```sh
data/processed/features/trend_model_samples_train.parquet
data/processed/features/trend_model_samples_valid.parquet
data/processed/features/trend_model_samples_test.parquet
data/processed/features/trend_model_samples_split_metadata.json
```

运行命令：

```sh
uv run python src/09_split_trend_model_samples.py
```

### 9. last_week baseline

`last_week` baseline 使用上一样本周的已观测属性占比增长 `growth_lag_1` 预测下一段增长：

```text
pred_target_growth = growth_lag_1
```

预测结果、参数和元数据统一写入：

```sh
outputs/models/last_week/predictions.csv
outputs/models/last_week/params.json
outputs/models/last_week/metadata.json
```

运行命令：

```sh
uv run python src/10_train_trend_baseline.py --model last_week
```

## 后续阶段

趋势 baseline 已经落地到 `last_week`，README 继续按计划记录后续边界：

| 阶段 | 计划产物 | 说明 |
| :--- | :--- | :--- |
| 趋势模型扩展 | 更多模型文件和趋势预测结果 | 后续再做 Moving Average、EWMA baseline，再考虑 LightGBM |
| 推荐模块 | Top-12 推荐列表和评价结果 | 将趋势分映射回商品，结合近期热门、用户历史属性偏好和 Item-CF 候选做轻量重排序 |

后续实现时需要继续遵守时间切分原则：任一周 `T` 的特征只能使用 `T` 及之前的数据，不能把 `T+1` 的热度、候选或用户行为泄漏进训练特征。

## 验证

当前测试使用标准库 `unittest`，不依赖真实 H&M 数据：

```sh
uv run python -m unittest discover -s tests -v
```

已覆盖的核心逻辑包括：

- articles 清洗字段、缺失值、重复 `article_id` 和文件写出回滚。
- 属性节点、商品-属性边、属性层级边的结构和引用完整性。
- 商品周销量、属性周热度的聚合、读取、写出和派生字段校验。
- 趋势标签的下一周目标计算、公式一致性和异常输入校验。
- 趋势样本的 lag、移动窗口、图特征合入、目标合入和标签表与当前热度表一致性校验。
- 趋势样本 train/valid/test 时间切分、切分读取、元数据写出和 split 合法性校验。
- `last_week` baseline 预测公式、预测表校验、训练入口 metadata 和写出顺序校验。
