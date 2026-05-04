# Fashion

时尚推荐相关实验项目，当前主要提供 Kaggle H&M 个性化时尚推荐数据集的下载入口和基础目录约定。

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

### 数据目录结构

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

### 2. articles.csv

| 字段                           | MVP 是否使用 |    稳妥版是否使用 | 说明             | 推荐用途                                           |
| ------------------------------ | -----------: | ----------------: | ---------------- | -------------------------------------------------- |
| `article_id`                   |         必须 |              必须 | 商品唯一编号     | 连接 `transactions_train.csv`，构建商品节点        |
| `product_code`                 |           否 |              可选 | 商品款式族编号   | 可分析同款不同色；MVP 不需要                       |
| `prod_name`                    |       展示用 |            展示用 | 商品名称         | 推荐结果展示，不建议作为模型特征                   |
| `product_type_no`              |           否 |            可保留 | 商品类型编号     | 与 `product_type_name` 对应，主要作映射            |
| `product_type_name`            |           是 |                是 | 商品具体类型     | 核心属性字段，适合做品类趋势                       |
| `product_group_name`           |           是 |                是 | 商品大类         | 可与 `product_type_name` 构成品类层级              |
| `graphical_appearance_no`      |           否 |            可保留 | 图案外观编号     | 与 `graphical_appearance_name` 对应，主要作映射    |
| `graphical_appearance_name`    |           是 |                是 | 图案 / 外观      | 适合分析 Solid、Stripe、Print 等风格趋势           |
| `colour_group_code`            |           否 |            可保留 | 颜色编号         | 与 `colour_group_name` 对应，主要作映射            |
| `colour_group_name`            |           是 |                是 | 具体颜色         | 核心属性字段，适合做颜色趋势                       |
| `perceived_colour_value_id`    |           否 |            可保留 | 感知颜色明暗编号 | 与 `perceived_colour_value_name` 对应，主要作映射  |
| `perceived_colour_value_name`  |           否 |                是 | 感知颜色明暗     | 可增强颜色趋势，如 Dark、Light、Dusty              |
| `perceived_colour_master_id`   |           否 |            可保留 | 主色系编号       | 与 `perceived_colour_master_name` 对应，主要作映射 |
| `perceived_colour_master_name` |           否 |                是 | 主色系           | 可与 `colour_group_name` 构成颜色层级              |
| `department_no`                |           否 |            可保留 | 部门编号         | 与 `department_name` 对应，主要作映射              |
| `department_name`              |           否 |              可选 | 商品部门         | 粒度较细，容易稀疏；可作为增强字段                 |
| `index_code`                   |           否 |            可保留 | 业务线编号       | 与 `index_name` 对应，主要作映射                   |
| `index_name`                   |           否 |                是 | 业务线           | 可用于构建组织层级，解释性较好                     |
| `index_group_no`               |           否 |            可保留 | 业务大类编号     | 与 `index_group_name` 对应，主要作映射             |
| `index_group_name`             |           否 |                是 | 业务大类         | 可区分 Ladieswear、Menswear、Baby/Children 等大类  |
| `section_no`                   |           否 |            可保留 | 商品区域编号     | 与 `section_name` 对应，主要作映射                 |
| `section_name`                 |           否 |                是 | 商品区域         | 适合构建组织层级，解释性较强                       |
| `garment_group_no`             |           否 |            可保留 | 服装组别编号     | 与 `garment_group_name` 对应，主要作映射           |
| `garment_group_name`           |           是 |                是 | 服装组别         | 核心属性字段，适合分析服装风格 / 材质趋势          |
| `detail_desc`                  |           否 | 展示用 / 可选增强 | 商品文本描述     | 可用于推荐展示；若进模型需要 NLP，MVP 不建议       |

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

清洗规则:

- 只输出本轮所需字段，不携带 `detail_desc`、编号映射字段或图片字段。
- `article_id` 必须按字符串读取，避免丢失前导 0。
- 输出字段存在缺失值时直接失败，不静默填充。
- `articles_clean_mvp.csv` 和 `articles_clean.csv` 的行数、`article_id` 集合必须与原始 `articles.csv` 保持一致。
- 中间表和属性图 CSV 全字段使用双引号引用，避免 VS Code / DuckDB 等工具按前几万行采样时把后续含逗号的属性值误解析成额外列。

### 3. article_week_sales.csv

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

### 4. attribute_week_heat.csv

基于 `article_week_sales.csv` 和 `data/processed/graph/edges_article_attribute.csv`，使用商品周销量中的 `sales_cnt` 作为购买次数热度，将商品热度映射到商品关联属性节点。

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
