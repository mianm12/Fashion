# Fashion

时尚趋势与轻量推荐实验项目，当前围绕 Kaggle H&M 个性化时尚推荐数据集构建周级交易表、商品属性层次图、商品周销量、属性周热度、趋势 baseline、LightGBM 主模型、趋势评价、轻量离线 Top-N 推荐实验、论文素材导出和本地答辩展示应用。

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

现阶段已经完成到三类趋势 baseline、LightGBM 主模型、趋势评价、轻量离线推荐闭环、论文素材导出和本地答辩展示应用：

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
| Last Week baseline | 已实现（运行命令后生成） | `outputs/models/last_week/predictions.csv`、`params.json`、`metadata.json` |
| Previous Growth baseline | 已实现（运行命令后生成） | `outputs/models/previous_growth/predictions.csv`、`params.json`、`metadata.json` |
| Moving Average baseline | 已实现（运行命令后生成） | `outputs/models/moving_average/predictions.csv`、`params.json`、`metadata.json` |
| LightGBM 主模型 | 已实现（运行命令后生成） | `outputs/models/lightgbm/predictions.csv`、`params.json`、`metadata.json`、`feature_importance.csv`、`model.txt` |
| 趋势评价 | 已实现（运行命令后生成） | `outputs/metrics/last_week/trend_metrics.json`、`outputs/metrics/previous_growth/trend_metrics.json`、`outputs/metrics/moving_average/trend_metrics.json`、`outputs/metrics/lightgbm/trend_metrics.json` |
| 推荐模块与推荐评价 | 已实现（运行命令后生成） | `outputs/recommendation/<method>/recommendations.csv`、`recommendation_items.parquet`、`params.json`、`metadata.json`、`metrics.json` |
| 论文素材导出 | 已实现（运行命令后生成） | `outputs/reports/figures/`、`outputs/reports/tables/`、`outputs/reports/case_studies/`、`outputs/reports/manifest.json` |
| 本地答辩展示应用 | 已实现（运行命令后生成） | `outputs/defense_app/fashion_demo.sqlite`、`apps/defense_app/backend/`、`apps/defense_app/frontend/` |

上表中 baseline、LightGBM、趋势评价、推荐实验、论文素材和展示应用的产物是对应训练、评价、推荐、导出或构建命令运行后的标准输出路径；功能已实现，但文件是否已存在取决于当前工作区是否运行过相应命令。

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

默认下载配置位于 `src/fashion_trend/datasets/paths.py`，路径根目录位于
`src/fashion_trend/foundation/paths.py`：

- `datasets.paths.DEFAULT_COMPETITION`：`h-and-m-personalized-fashion-recommendations`
- `foundation.paths.RAW_DIR`：项目根目录下的 `data/raw`

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

当前已实现流水线可按下面顺序从原始数据推进到本地展示库。`12` 号之后的推荐、reports 和展示库构建依赖上游 stable artifact，耗时较长时可按阶段单独运行：

```sh
uv run python src/00_download_data.py
uv run python src/01_data_check.py
uv run python src/02_build_weekly_transactions.py
uv run python src/03_clean_articles.py
uv run python src/04_build_attribute_graph.py
uv run python src/05_compute_article_week_sales.py
uv run python src/06_compute_attribute_week_heat.py
uv run python src/07_build_trend_targets.py
uv run python src/08_build_trend_model_samples.py
uv run python src/09_split_trend_model_samples.py
uv run python src/10_train_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model lightgbm
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy default
uv run python src/14_rerank_recommendations.py --method pop_similarity_trend
uv run python src/15_eval_recommendations.py --method pop_similarity_trend
uv run python src/16_run_recommendation_experiment.py --experiment main
uv run python src/17_export_paper_assets.py
uv run python src/18_build_defense_app_db.py
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

基于 `trend_model_samples.parquet` 按时间顺序切分训练、验证和测试样本。默认最后 8 个样本周为 test，之前 8 个样本周为 valid，更早样本周为 train。切分配置位于 `src/fashion_trend/trend/paths.py`：

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

`last_week` baseline 通过通用趋势模型训练入口运行，模型细节位于
`src/fashion_trend/trend/models/baselines/last_week.py`。当前模型是 Last Week Heat 基线，
使用当前 `share_t` 作为下一期热度分布预测，并在同一 `split/week_id/attr_type` 内归一化：

```text
pred_share_t1 = group_normalize(share_t)
pred_target_growth = log((pred_share_t1 + epsilon) / (share_t + epsilon))
```

预测结果、参数和元数据统一写入：

```sh
outputs/models/last_week/predictions.csv
outputs/models/last_week/params.json
outputs/models/last_week/metadata.json
```

运行命令：

```sh
uv run python src/10_train_trend_model.py --model last_week
```

### 10. previous_growth baseline

`previous_growth` baseline 复用通用趋势模型训练入口，模型细节位于
`src/fashion_trend/trend/models/baselines/previous_growth.py`。当前模型直接沿用上一期增长率预测下一段增长：

```text
pred_target_growth = growth_lag_1
```

派生 `pred_share_t1` 时，先按目标公式的逆运算得到原始预测占比，再在同一
`split/week_id/attr_type` 内对非负原始值归一化，确保输出是合法占比分布。

预测结果、参数和元数据统一写入：

```sh
outputs/models/previous_growth/predictions.csv
outputs/models/previous_growth/params.json
outputs/models/previous_growth/metadata.json
```

运行命令：

```sh
uv run python src/10_train_trend_model.py --model previous_growth
```

### 11. moving_average baseline

`moving_average` baseline 复用通用趋势模型训练入口，模型细节位于
`src/fashion_trend/trend/models/baselines/moving_average.py`。当前模型使用最近两段已观测属性占比增长的简单平均预测下一段增长：

```text
pred_target_growth = mean(growth_lag_1, growth_lag_2)
```

派生 `pred_share_t1` 的归一化口径与 `previous_growth` 一致，预测表契约与
`last_week`、`previous_growth` 保持一致。

预测结果、参数和元数据统一写入：

```sh
outputs/models/moving_average/predictions.csv
outputs/models/moving_average/params.json
outputs/models/moving_average/metadata.json
```

运行命令：

```sh
uv run python src/10_train_trend_model.py --model moving_average
```

### 12. LightGBM 主模型

`lightgbm` 主模型复用通用趋势模型训练入口，模型细节位于
`src/fashion_trend/trend/models/supervised/lightgbm.py`。第一版使用现有
`trend_model_samples` 中的数值特征和 `attr_type` 分类特征，预测目标为：

```text
target_growth
```

模型使用 train split 拟合，valid split 做 early stopping，test split 只进入统一趋势评价。
标准预测产物和可解释产物写入：

```text
outputs/models/lightgbm/predictions.csv
outputs/models/lightgbm/params.json
outputs/models/lightgbm/metadata.json
outputs/models/lightgbm/feature_importance.csv
outputs/models/lightgbm/model.txt
```

运行命令：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm
```

LightGBM 也支持按 `run_id` 保存调参实验，实验 run 默认不覆盖 stable 主结果：

```sh
uv run python src/10_train_trend_model.py --model lightgbm --run-id smoke-lightgbm --no-promote
uv run python src/11_eval_trend_model.py --model lightgbm --run-id smoke-lightgbm
uv run python src/10_train_trend_model.py --model lightgbm --promote-run smoke-lightgbm
```

run 级模型和评价产物写入：

```text
outputs/models/lightgbm/runs/<run_id>/predictions.csv
outputs/models/lightgbm/runs/<run_id>/params.json
outputs/models/lightgbm/runs/<run_id>/metadata.json
outputs/models/lightgbm/runs/<run_id>/feature_importance.csv
outputs/models/lightgbm/runs/<run_id>/model.txt
outputs/models/lightgbm/runs/index.jsonl
outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json
outputs/metrics/lightgbm/runs/evaluations.jsonl
```

调参选择只能读取 `evaluations.jsonl` 的 `selection_metrics`；test 指标只用于最终选中 run 的一次性报告。

默认运行 `uv run python src/10_train_trend_model.py --model lightgbm` 时，训练参数优先读取 `outputs/models/lightgbm/params.json` 中已经发布的 stable 参数；如果该文件不存在，才使用源码中的 built-in 默认参数。这条默认训练路径仍会自动发布 stable，因此会用本次训练结果覆盖 `outputs/models/lightgbm/`。

显式 `--params` 或 `--param` 会进入自定义实验参数模式，不读取 stable 参数，并继续默认不覆盖 stable 主结果。若只想使用默认参数生成 run 但不覆盖 stable，可显式传入 `--no-promote`。

### 13. 趋势评价

趋势评价通过独立入口运行，读取已经生成的趋势模型预测表：

```sh
outputs/models/last_week/predictions.csv
outputs/models/previous_growth/predictions.csv
outputs/models/moving_average/predictions.csv
outputs/models/lightgbm/predictions.csv
```

评价结果按模型写入：

```sh
outputs/metrics/last_week/trend_metrics.json
outputs/metrics/previous_growth/trend_metrics.json
outputs/metrics/moving_average/trend_metrics.json
outputs/metrics/lightgbm/trend_metrics.json
```

运行命令：

```sh
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model lightgbm
```

第一版趋势评价只评价 `valid` 和 `test` split，不把 `train` 作为正式指标。排序目标与训练目标保持一致：

```text
target_growth vs pred_target_growth
```

指标包含：

```text
MAE
RMSE
Spearman
Precision@5/10/20
Recall@5/10/20
NDCG@5/10/20
```

排序指标按 `split + week_id + attr_type` 逐组计算，再汇总到 overall 和 by_attr_type，便于观察不同属性类型的趋势预测质量。

### 14. 轻量离线推荐实验

推荐模块已实现轻量离线实验版，用于把趋势预测结果映射回商品 Top-12 推荐，并通过离线指标验证趋势分的应用价值。入口为：

```sh
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy <strategy>
uv run python src/14_rerank_recommendations.py --method <method>
uv run python src/15_eval_recommendations.py --method <method>
uv run python src/16_run_recommendation_experiment.py --experiment main
```

推荐输入产物写入：

```text
data/processed/recommend/time_windows.parquet
data/processed/recommend/target_users.parquet
data/processed/recommend/evaluation_labels.parquet
data/processed/recommend/user_profile.parquet
data/processed/recommend/metadata.json
data/processed/recommend/candidates/<strategy>/candidate_items.parquet
data/processed/recommend/features/<feature_name>/strategy=<strategy>/split=<split>/cutoff_week=<week>/part.parquet
```

候选池按 strategy 隔离，推荐输出按 method 隔离。标准推荐产物写入：

```text
outputs/recommendation/<method>/recommendations.csv
outputs/recommendation/<method>/recommendation_items.parquet
outputs/recommendation/<method>/params.json
outputs/recommendation/<method>/metadata.json
outputs/recommendation/<method>/metrics.json
outputs/recommendation/experiments/<experiment_id>/experiment.json
```

`recommendation_items.parquet` 是默认内部长表产物；`recommendation_items.csv`
仅作为显式导出用途，不是默认推荐输出。

当前 `main` 实验使用 valid split 的 `NDCG@12` 选择
`pop_similarity_trend` 权重。最近一次有限网格调参搜索 25 组权重，最佳权重为
`pop_score=0.2`、`sim_score=0.2`、`trend_score=0.1`、`recent_score=0.5`。
该主模型在 valid split 上达到 `NDCG@12=0.005922`、`MAP@12=0.002745`、
`Recall@12=0.008566`，高于 `recent_popularity` 的 valid `NDCG@12=0.005715`。
在 test split 上，调参后主模型达到 `NDCG@12=0.007987`、`MAP@12=0.003703`、
`Recall@12=0.013685`，明显高于旧主模型结果，但略低于 `recent_popularity`
的 test `NDCG@12=0.008087`。论文叙述应把它定位为“趋势感知融合主模型”，并保留
`recent_popularity` 作为强 baseline 对照。

`16_run_recommendation_experiment.py` 默认复用新鲜的输入、候选、feature
cache 和 method 产物。需要重建时使用明确 force 参数：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
uv run python src/16_run_recommendation_experiment.py --experiment main --force-method pop_similarity
uv run python src/16_run_recommendation_experiment.py --experiment main --force-cache
uv run python src/16_run_recommendation_experiment.py --experiment main --force-candidates
uv run python src/16_run_recommendation_experiment.py --experiment main --force-rebuild-all
```

### 15. 论文图表与案例导出

reports 阶段只读取已经发布的稳定数据、模型和推荐 artifact，不重新训练模型，也不重跑推荐方法。入口为：

```sh
uv run python src/17_export_paper_assets.py
```

默认输出：

```text
outputs/reports/figures/*.svg
outputs/reports/figures/*.png
outputs/reports/tables/*.csv
outputs/reports/tables/*.md
outputs/reports/case_studies/*.json
outputs/reports/case_studies/*.md
outputs/reports/manifest.json
```

本阶段使用 `matplotlib` 导出静态 SVG/PNG 图表；Markdown 表格由项目内 writer 生成，不依赖 `tabulate`。如果本机缺少可用 CJK 字体，导出会 fail-fast，避免生成缺字的中文论文图。

### 16. 本地答辩展示应用

答辩展示应用是 downstream presentation artifact，用于本地展示离线趋势预测、属性图和轻量 Top-N 推荐解释。它不训练模型、不重跑推荐、不作为在线推荐服务或生产平台。先构建只读 SQLite 展示库：

```sh
uv run python src/18_build_defense_app_db.py
```

默认生成：

```text
outputs/defense_app/fashion_demo.sqlite
outputs/defense_app/static/reports/
```

SQLite 文件和静态图表副本都是生成产物，不应提交。构建展示库时会从 `outputs/reports/` 复制展示所需图表到 `outputs/defense_app/static/reports/`；构建失败时会回滚旧数据库和旧静态资源。后端只读该 SQLite，前端只调用 FastAPI：

```sh
uv run --group app uvicorn app.main:app --reload --app-dir apps/defense_app/backend
cd apps/defense_app/frontend
npm install
npm run dev
```

前端页面覆盖趋势看板、属性详情、商品属性图、推荐案例和推荐解释。详细说明见 `apps/defense_app/README.md`。

## 后续阶段

趋势模型训练与评价框架已经落地到 `last_week`、`previous_growth`、`moving_average`
三类必须 baseline、`lightgbm` 主模型、推荐离线实验、论文素材导出和本地答辩展示应用。README 继续记录可选增强边界：

| 阶段 | 计划产物 | 说明 |
| :--- | :--- | :--- |
| 趋势模型扩展 | 更多模型文件和趋势预测结果 | LightGBM 已实现并完成首轮 stable 调参；后续可考虑更多监督模型或 EWMA 等增强 baseline |
| 推荐实验增强 | 推荐召回、消融和案例证据 | 当前已支持轻量离线 Top-12、单方法评价、`main` 实验、reports 导出和展示应用消费；后续优先补充更强候选召回和更严格的命名消融 |
| 答辩展示完善 | 演示脚本和少量交互优化 | 当前已具备 SQLite 展示库、FastAPI 后端和 Vue/Vite 前端；后续只做不改变研究结论的展示增强 |

后续实现时需要继续遵守时间切分原则：任一周 `T` 的特征只能使用 `T` 及之前的数据，不能把 `T+1` 的热度、候选或用户行为泄漏进训练特征。

## 实现位置

项目内部代码按业务域组织在 `src/fashion_trend/` 下：

默认路径根常量位于 `foundation.paths`；数据集、交易、catalog、trend、recommendation、reports 和 presentation 的业务路径由各自领域的 `paths.py` 持有。

- `foundation/`：路径、日志、原子写入、通用校验和 artifact 安全。
- `datasets/`：原始数据下载、解压和基础检查。
- `transactions/`：周级交易表和交易窗口。
- `catalog/`：商品表清洗和静态属性图。
- `trend/`：属性热度、标签、样本、时间切分、趋势模型训练和趋势评价。
- `recommendation/`：推荐契约、路径、reader、时间窗口、输入构建、候选召回、排序特征、方法注册、推荐评价和实验编排。
- `reports/`：论文图表、表格、案例和 manifest 导出；只读消费稳定 artifact，不作为在线 dashboard。
- `presentation/`：本地答辩展示库 schema、上游 artifact 抽取、SQLite 写入和构建 runner；只读消费稳定 artifact。

当前已实现的 `src/00_*.py` 到 `src/18_*.py` 是用户运行入口；后续新增的编号脚本继续沿用同一约定作为流程索引，计算事实位于业务包。保持现有用户命令不变。

本地答辩展示应用位于 `apps/defense_app/`，其中 `backend/` 是只读 FastAPI API，`frontend/` 是 Vue/Vite 前端。该应用依赖 `outputs/defense_app/fashion_demo.sqlite`，不直接读取原始 H&M 数据、训练 run 目录或历史推荐 CSV。

趋势共享实现位于 `src/fashion_trend/trend/` 子包。`heat/`、`labels/`、`features/`、`splits/`、`training/`、`evaluation/` 和 `predictions.py` 分别对应当前趋势流水线阶段与训练/评价共享契约；`trend/models/baselines/` 存放当前 baseline，`trend/models/supervised/` 存放 LightGBM 等监督模型；`trend/__init__.py` 只是包标记，不重新导出旧入口。内部代码必须直接导入具体模块。

## 验证

当前测试使用 `pytest`，不依赖真实 H&M 数据：

```sh
uv run pytest
```

答辩展示应用相关验证需要显式安装和使用应用依赖：

```sh
uv run --group app pytest apps/defense_app/backend/tests
uv run --group app python -m compileall -q apps/defense_app/backend
cd apps/defense_app/frontend
npm run typecheck
npm run build
```

已覆盖的核心逻辑包括：

- articles 清洗字段、缺失值、重复 `article_id` 和文件写出回滚。
- 属性节点、商品-属性边、属性层级边的结构和引用完整性。
- 商品周销量、属性周热度的聚合、读取、写出和派生字段校验。
- 趋势标签的下一周目标计算、公式一致性和异常输入校验。
- 趋势样本的 lag、移动窗口、图特征合入、目标合入和标签表与当前热度表一致性校验。
- 趋势样本 train/valid/test 时间切分、切分读取、元数据写出和 split 合法性校验。
- `last_week`、`previous_growth` 与 `moving_average` baseline 预测公式、预测表校验、通用训练 runner metadata、artifact 和写出顺序校验。
- `lightgbm` 主模型的特征准备、延迟原生包导入、metadata 诊断、特征重要性、artifact 和训练/评价 runner 接入校验。
- 趋势评价的预测读取、输入校验、分组指标、JSON payload、写出边界和 CLI 行为校验。
- 推荐输入、候选召回、feature cache、排序打分、方法注册、Top-N 评价、主实验编排和 freshness contract。
- reports loaders、图表、表格、Markdown、推荐解释案例、manifest 和只读架构边界。
- presentation 展示库 schema、表构建、SQLite 写入、架构边界和 `src/18_build_defense_app_db.py` 入口。
- defense app 后端只读 API、错误响应、推荐解释、metrics summary 和 SQLite 连接边界。
