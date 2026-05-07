# 商品周销量与属性周热度设计

## 状态

本设计是较早期的历史设计，早于后续领域驱动模块迁移。当前实现不再使用单文件 `src/fashion_trend/trend.py` 或 `fashion_trend.trend` 聚合入口；内部代码必须直接导入具体模块。

当前 ownership：

- 周级交易稳定产物读取：`fashion_trend.transactions.weekly.read_weekly_transactions`
- 商品目录图读取：`fashion_trend.catalog.graph.read_article_attribute_edges`、`fashion_trend.catalog.graph.read_attribute_nodes`
- 商品周销量阶段：`fashion_trend.trend.article_sales`
- 属性周热度阶段：`fashion_trend.trend.attribute_heat`
- CSV/Parquet/JSON 原子写出：`fashion_trend.foundation.io`

## 范围

本轮按 `docs/gpt-research/implementation-plan.md` 中的开发顺序推进两个连续产物：

- 先从周级交易表生成商品周销量表 `data/processed/trend/article_week_sales.csv`。
- 再从商品周销量表和商品-属性边表生成属性周热度表 `data/processed/trend/attribute_week_heat.csv`。

本轮只覆盖趋势主线的数据聚合层，不构造趋势标签、趋势特征、模型训练、趋势评价或推荐结果。

新增顶层脚本：

- `src/05_compute_article_week_sales.py`
- `src/06_compute_attribute_week_heat.py`

新增通用模块（历史设计；当前已拆分为具体领域模块）：

- `src/fashion_trend/transactions/weekly.py`
- `src/fashion_trend/catalog/graph.py`
- `src/fashion_trend/trend/article_sales.py`
- `src/fashion_trend/trend/attribute_heat.py`

顶层脚本需要表达清楚的工作逻辑顺序，而不是只作为 CLI 包装层。可复用的数据读取、校验、聚合和写出逻辑放在当前 ownership 对应的具体模块中。

## 架构

### `src/05_compute_article_week_sales.py`

脚本负责按顺序编排商品周销量计算：

1. 从 `fashion_trend.config.PATH` 读取输入输出路径。
2. 校验 `transactions_train_weekly.parquet` 是否存在。
3. 调用 `fashion_trend.transactions.weekly.read_weekly_transactions` 读取必要列。
4. 调用 `fashion_trend.trend.article_sales.build_article_week_sales_frame` 聚合商品周销量。
5. 调用 `fashion_trend.trend.article_sales.validate_article_week_sales` 校验输出表。
6. 调用 `fashion_trend.foundation.io.write_csv_atomic` 写出 CSV。
7. 输出周数、商品数、行数和目标文件路径日志。

脚本只捕获可定位的输入、校验和写出错误，并返回非零退出码。错误信息需要保留具体文件路径、缺失字段或异常数据范围。

### `src/06_compute_attribute_week_heat.py`

脚本负责按顺序编排属性周热度计算：

1. 从 `fashion_trend.config.PATH` 读取输入输出路径。
2. 校验 `article_week_sales.csv` 和 `edges_article_attribute.csv` 是否存在。
3. 调用 `fashion_trend.trend.article_sales.read_article_week_sales` 读取商品周销量。
4. 调用 `fashion_trend.catalog.graph.read_article_attribute_edges` 和 `read_attribute_nodes` 读取商品-属性边与属性节点。
5. 调用 `fashion_trend.trend.article_sales.validate_article_week_sales` 和 `fashion_trend.trend.attribute_heat` 中的 heat 专属 validator 校验输入。
6. 调用 `fashion_trend.trend.attribute_heat.build_attribute_week_heat_frame` 连接、聚合并计算热度指标。
7. 调用 `fashion_trend.trend.attribute_heat.validate_attribute_week_heat` 校验输出表。
8. 调用 `fashion_trend.foundation.io.write_csv_atomic` 写出 CSV。
9. 输出周数、属性类型数、属性节点数、行数和目标文件路径日志。

如果商品销量表中存在无法映射到商品-属性边的 `article_id`，脚本应失败，不静默丢弃这些销量。

### 当前具体模块

历史设计中的单文件趋势模块已经拆分。当前可测试业务逻辑按 ownership 分散在：

- `src/fashion_trend/transactions/weekly.py`：周级交易读取。
- `src/fashion_trend/catalog/graph.py`：商品-属性边和属性节点读取。
- `src/fashion_trend/trend/article_sales.py`：商品周销量聚合和校验。
- `src/fashion_trend/trend/attribute_heat.py`：属性周热度聚合和校验。
- `src/fashion_trend/foundation/io.py`：原子写出。

这些模块不依赖命令行参数，不读取全局状态。脚本传入路径，模块函数返回 DataFrame、行数统计或抛出可定位异常。

## 数据表

### `article_week_sales.csv`

路径：

```text
data/processed/trend/article_week_sales.csv
```

字段顺序固定：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `week_id` | int | 周编号 |
| `article_id` | str | 商品 ID，保留前导 0 |
| `sales_cnt` | int | 该商品本周购买次数 |
| `sales_user_cnt` | int | 该商品本周购买用户数 |
| `sales_amount` | float | 该商品本周销售额，按 `price` 求和 |

`sales_cnt` 是后续属性热度计算的主输入。`sales_user_cnt` 和 `sales_amount` 保留为低成本扩展字段，后续模型可以不用。

### `attribute_week_heat.csv`

路径：

```text
data/processed/trend/attribute_week_heat.csv
```

字段顺序固定：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `week_id` | int | 周编号 |
| `attr_id` | str | 属性 ID |
| `attr_type` | str | 属性类型 |
| `attr_value` | str | 属性取值 |
| `heat_cnt` | int | 属性原始热度，来自关联商品的 `sales_cnt` 求和 |
| `type_total_heat` | int | 同一 `attr_type` 在本周的总热度 |
| `heat_share` | float | `heat_cnt / type_total_heat` |
| `log_heat` | float | `log1p(heat_cnt)` |
| `rank_in_type` | int | 同一 `week_id + attr_type` 内按热度降序排名 |

属性周热度默认覆盖当前属性图中的全部 10 个属性字段，不只计算 5 个核心属性。后续趋势样本或展示如果只需要核心属性，应通过 `nodes_attribute.csv.is_core_attr` 过滤。

## 数据流

### 商品周销量

输入：

```text
data/interim/transactions_train_weekly.parquet
```

读取列：

- `week_id`
- `article_id`
- `customer_id`
- `price`

处理流程：

1. 校验必要列存在。
2. 校验 `week_id` 和 `article_id` 不缺失。
3. 将 `article_id` 转为 pandas string 类型。
4. 按 `week_id + article_id` 分组。
5. 计算 `sales_cnt = size()`。
6. 计算 `sales_user_cnt = customer_id.nunique()`。
7. 计算 `sales_amount = price.sum()`。
8. 按 `week_id`, `article_id` 排序。
9. 写出 CSV。

### 属性周热度

输入：

```text
data/processed/trend/article_week_sales.csv
data/processed/graph/edges_article_attribute.csv
```

商品周销量读取列：

- `week_id`
- `article_id`
- `sales_cnt`

商品-属性边读取列：

- `article_id`
- `attr_id`
- `attr_type`
- `attr_value`

处理流程：

1. 校验商品周销量表的 `week_id + article_id` 唯一。
2. 校验商品-属性边表的 `article_id + attr_id` 唯一。
3. 校验所有有销量的 `article_id` 都存在于商品-属性边表。
4. 用 `article_id` inner join 商品周销量和商品-属性边。
5. 按 `week_id + attr_id + attr_type + attr_value` 聚合 `sales_cnt`，得到 `heat_cnt`。
6. 在 `week_id + attr_type` 内计算 `type_total_heat`。
7. 计算 `heat_share = heat_cnt / type_total_heat`。
8. 计算 `log_heat = log1p(heat_cnt)`。
9. 在 `week_id + attr_type` 内按 `heat_cnt` 降序、`attr_id` 升序稳定排名，得到 `rank_in_type`。
10. 按 `week_id`, `attr_type`, `rank_in_type`, `attr_id` 排序。
11. 写出 CSV。

## 校验规则

### 商品周销量输入校验

遇到以下情况直接失败：

- `transactions_train_weekly.parquet` 不存在。
- 输入缺少 `week_id`, `article_id`, `customer_id`, `price` 中任一字段。
- `week_id`, `article_id` 或 `customer_id` 存在缺失值。
- `price` 存在缺失值或负值。

### 商品周销量输出校验

写出前校验：

- 输出列顺序与设计一致。
- `week_id + article_id` 唯一。
- `sales_cnt >= 1`。
- `sales_user_cnt >= 1`。
- `sales_amount >= 0`。

### 商品-属性边输入校验

遇到以下情况直接失败：

- `edges_article_attribute.csv` 不存在。
- 输入缺少 `article_id`, `attr_id`, `attr_type`, `attr_value` 中任一字段。
- 任一必要字段存在缺失值。
- `article_id + attr_id` 不唯一。

### 属性周热度输出校验

写出前校验：

- 输出列顺序与设计一致。
- `week_id + attr_id` 唯一。
- `heat_cnt >= 1`。
- `type_total_heat >= heat_cnt`。
- `0 < heat_share <= 1`。
- 每个 `week_id + attr_type` 的 `heat_share` 总和接近 1，容忍浮点误差 `1e-9`。
- 每个 `week_id + attr_type` 的 `rank_in_type` 从 1 开始且不重复。

## 写出策略

两个 CSV 都写入 `data/processed/trend/`，全字段使用双引号引用，延续当前 articles 清洗和属性图输出习惯。

写出应使用临时文件再替换目标文件：

1. 写入 `*.tmp`。
2. 成功后替换目标文件。
3. 失败时删除临时文件。

本轮不需要实现多文件回滚。两个阶段是独立脚本，`attribute_week_heat.csv` 依赖已存在且通过校验的 `article_week_sales.csv`。

## 配置

修改 `src/fashion_trend/config.py`：

- 新增 `TREND_DIR = PROCESSED_DIR / "trend"`。
- 新增 `PATH["trend_article_week_sales"] = TREND_DIR / "article_week_sales.csv"`。
- 新增 `PATH["trend_attribute_week_heat"] = TREND_DIR / "attribute_week_heat.csv"`。

保留现有 raw、interim、graph 路径命名，不重命名已有文件。

## 测试

新增 `tests/test_trend.py`，使用小型 DataFrame 和临时目录，不依赖真实 H&M 数据。

测试覆盖：

- `build_article_week_sales_frame` 正确聚合 `sales_cnt`, `sales_user_cnt`, `sales_amount`。
- `build_article_week_sales_frame` 保留字符串形式 `article_id`。
- `validate_article_week_sales` 拒绝重复 `week_id + article_id`。
- `build_attribute_week_heat_frame` 正确计算 `heat_cnt`, `type_total_heat`, `heat_share`, `log_heat`, `rank_in_type`。
- `build_attribute_week_heat_frame` 在销量商品缺少属性边时失败。
- `validate_article_attribute_edges_for_heat` 在 `article_id + attr_id` 重复时失败。
- 文件级函数写出目标 CSV，列顺序和全字段引用符合预期。

最小验证命令：

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

真实数据验证命令：

```sh
PYTHONPATH=src .venv/bin/python src/05_compute_article_week_sales.py
PYTHONPATH=src .venv/bin/python src/06_compute_attribute_week_heat.py
```

真实数据验证完成后，应确认：

- `data/processed/trend/article_week_sales.csv` 存在。
- `data/processed/trend/attribute_week_heat.csv` 存在。
- 日志中的周数、商品数、属性类型数和行数非零且与输入规模相符。

## 非目标

本轮不做以下事项：

- 不计算 `attribute_week_target.csv`。
- 不构造 `trend_model_samples.parquet`。
- 不训练 baseline、LightGBM 或推荐模型。
- 不修改已有属性图生成逻辑。
- 不引入新依赖。
- 不新增 CLI 参数系统，脚本优先使用项目配置路径。
