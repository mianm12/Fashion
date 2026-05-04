# 趋势标签与趋势训练样本设计

## 范围

本轮按推荐方案分两步推进：

1. 先调整上一阶段 `attribute_week_heat.csv` 的生成逻辑，使其从稀疏热度表变成完整属性-周面板。
2. 再新增阶段 5 的趋势标签表和趋势训练样本表。

本轮产物包括：

- 更新：`data/processed/trend/attribute_week_heat.csv`
- 新增：`data/processed/trend/attribute_week_target.csv`
- 新增：`data/processed/features/trend_model_samples.parquet`

本轮不训练模型、不做趋势预测评价、不做推荐模块。阶段 5 只负责把属性周热度转成可训练的监督学习数据集，并保证时间特征没有未来泄漏。

## 设计原则

- `attribute_week_heat.csv` 是阶段 5 的事实输入，必须显式覆盖所有 `week_id x attr_id` 组合。
- 缺失属性-周不代表数据错误，而是该属性在该周热度为 0。
- 标签和特征只能使用当前周 `t` 及之前的信息；`t+1` 只能出现在目标字段中。
- 低频属性不在样本生成阶段直接删除，而是生成可过滤字段，供后续训练、评价和展示选择。
- 继续复用 `src/fashion_trend/trend.py` 承载可测试的数据逻辑，脚本层只负责路径、日志、编排和错误返回。

## 上一阶段调整：完整属性-周面板

### 输入

`src/06_compute_attribute_week_heat.py` 将读取三张输入表：

- `data/processed/trend/article_week_sales.csv`
- `data/processed/graph/edges_article_attribute.csv`
- `data/processed/graph/nodes_attribute.csv`

`nodes_attribute.csv` 是完整属性集合的来源。周集合来自 `article_week_sales.csv` 中出现的 `week_id`。

### 输出字段

`attribute_week_heat.csv` 保持原字段顺序不变：

| 字段 | 说明 |
| --- | --- |
| `week_id` | 周编号 |
| `attr_id` | 属性 ID |
| `attr_type` | 属性类型 |
| `attr_value` | 属性取值 |
| `heat_cnt` | 属性原始热度，缺失属性-周补 0 |
| `type_total_heat` | 同一 `week_id + attr_type` 的总热度 |
| `heat_share` | `heat_cnt / type_total_heat`，总热度为 0 时为 0 |
| `log_heat` | `log1p(heat_cnt)` |
| `rank_in_type` | 同一 `week_id + attr_type` 内稳定排名 |

### 面板构造规则

1. 先按现有逻辑从商品周销量和商品-属性边聚合观测热度。
2. 用所有周和所有属性节点构造笛卡尔面板。
3. 将观测热度左连接到完整面板。
4. 对缺失 `heat_cnt` 补 0。
5. 按 `week_id + attr_type` 计算 `type_total_heat`。
6. 计算 `heat_share` 和 `log_heat`。
7. 在 `week_id + attr_type` 内按 `heat_cnt` 降序、`attr_id` 升序生成 `rank_in_type`。

零热度属性参与排名，排在同类型属性的后部。这样后续标签能够覆盖从 0 到正数、正数到 0、持续为 0 和持续活跃四类变化。

### 校验规则

写出前校验：

- 输出列顺序固定。
- `week_id + attr_id` 唯一。
- 行数等于 `week_count * attribute_count`。
- 输出属性集合等于 `nodes_attribute.attr_id`。
- 输出周集合等于 `article_week_sales.week_id`。
- `heat_cnt >= 0`。
- `type_total_heat >= heat_cnt`。
- `0 <= heat_share <= 1`。
- `log_heat = log1p(heat_cnt)`。
- 每个 `week_id + attr_type` 的 `type_total_heat` 与 `heat_cnt` 分组求和一致。
- `type_total_heat > 0` 的分组内 `heat_share` 之和接近 1；若某类属性该周总热度为 0，则该分组 `heat_share` 之和应为 0。
- 每个 `week_id + attr_type` 的 `rank_in_type` 从 1 开始、连续、不重复，并符合排序规则。

## 阶段 5 新增一：趋势标签表

### 输出路径

```text
data/processed/trend/attribute_week_target.csv
```

### 输出字段

字段顺序固定：

| 字段 | 说明 |
| --- | --- |
| `week_id` | 当前周 `t` |
| `attr_id` | 属性 ID |
| `attr_type` | 属性类型 |
| `attr_value` | 属性取值 |
| `heat_t` | 当前周热度 |
| `heat_t1` | 下一周热度 |
| `share_t` | 当前周同类型热度占比 |
| `share_t1` | 下一周同类型热度占比 |
| `rank_in_type_t` | 当前周同类型排名 |
| `target_log_heat_t1` | 下一周 `log1p(heat_cnt)` |
| `target_growth` | 下一周占比对数增长 |
| `target_rank_in_type_t1` | 下一周同类型排名 |

### 标签公式

主标签：

```text
target_growth = log((share_t1 + 1e-6) / (share_t + 1e-6))
```

辅助标签：

```text
target_log_heat_t1 = log1p(heat_t1)
target_rank_in_type_t1 = rank_in_type at t + 1
```

标签表排除最后一周，因为最后一周没有 `t+1`。完整面板保证每个非最后周的每个属性都有一行目标记录。

### 校验规则

- 输入 `attribute_week_heat.csv` 必须先通过完整面板校验。
- 标签输出行数等于 `(week_count - 1) * attribute_count`。
- `week_id + attr_id` 唯一。
- 不允许缺失值。
- `heat_t >= 0`，`heat_t1 >= 0`。
- `0 <= share_t <= 1`，`0 <= share_t1 <= 1`。
- `target_growth`、`target_log_heat_t1` 必须是有限数值。
- 对每行重新计算公式，验证目标字段一致。

## 阶段 5 新增二：趋势训练样本表

### 输出路径

```text
data/processed/features/trend_model_samples.parquet
```

`features` 目录新增到 `src/fashion_trend/config.py`，与 `trend` 和 `graph` 目录分离。

### 样本粒度

每行是一条属性-周样本：

```text
(week_id=t, attr_id=a)
```

输入特征来自 `t` 及以前的属性热度、属性静态信息和图结构信息。目标来自 `attribute_week_target.csv` 中的 `t -> t+1` 标签。

默认使用 `min_lag_weeks = 4`，因此样本从第 4 周开始，最后一周同样不进入样本。

### 标识与当前周字段

- `week_id`
- `attr_id`
- `attr_type`
- `attr_value`
- `heat_t`
- `share_t`
- `log_heat_t`
- `rank_in_type_t`

### 历史热度特征

- `heat_lag_1`
- `heat_lag_2`
- `heat_lag_3`
- `heat_lag_4`
- `share_lag_1`
- `share_lag_2`
- `share_lag_3`
- `share_lag_4`

这些字段通过同一 `attr_id` 内按 `week_id` 排序后 `shift` 生成。

### 历史增长特征

- `growth_lag_1`
- `growth_lag_2`
- `acc_lag_1`

计算方式：

```text
growth_lag_1 = log((share_t + 1e-6) / (share_lag_1 + 1e-6))
growth_lag_2 = log((share_lag_1 + 1e-6) / (share_lag_2 + 1e-6))
acc_lag_1 = growth_lag_1 - growth_lag_2
```

### 移动统计特征

移动窗口只使用当前周和历史周：

- `heat_ma_4`
- `share_ma_4`
- `share_std_4`
- `share_max_4`
- `share_min_4`

窗口为 `t-3` 到 `t`。因为样本从第 4 周开始，窗口总是完整。

### 静态和图结构特征

从 `nodes_attribute.csv` 读取：

- `article_count`
- `is_core_attr`

从 `edges_attribute_hierarchy.csv` 派生：

- `degree`
- `parent_count`
- `child_count`

其中 `degree = parent_count + child_count`。本轮不引入父级热度、同级均值等需要更复杂图遍历的增强特征，避免阶段 5 范围过大。

### 低频过滤辅助字段

样本表保留以下历史口径字段：

- `history_total_heat_t`
- `history_active_weeks_t`
- `is_trend_eligible_t`

计算只使用截至当前周 `t` 的信息：

```text
history_total_heat_t = sum(heat_cnt from week 0 to t)
history_active_weeks_t = count(weeks where heat_cnt > 0 from week 0 to t)
is_trend_eligible_t = history_total_heat_t >= 100 and history_active_weeks_t >= 8
```

这使后续训练或趋势 Top-K 展示可以过滤低频属性，同时不在数据构造阶段丢失样本。

### 时间特征

本轮不依赖日历日期表，先使用周编号可直接派生的稳定特征：

- `week_index`
- `week_mod_52`

其中：

```text
week_index = week_id
week_mod_52 = week_id % 52
```

后续如果需要真实月份、自然周或季节，可基于 `date_range.json` 增强，但不放入本轮 MVP。

### 目标字段

训练样本表连接趋势标签表，保留：

- `target_growth`
- `target_log_heat_t1`
- `target_rank_in_type_t1`

## 新增脚本

新增脚本：

```text
src/07_build_trend_targets.py
src/08_build_trend_model_samples.py
```

`07_build_trend_targets.py`：

1. 读取 `PATH["trend_attribute_week_heat"]`。
2. 校验完整属性-周面板。
3. 构建 `attribute_week_target.csv`。
4. 校验标签表。
5. 写出 CSV，并输出行数、周数、属性数和目标路径。

`08_build_trend_model_samples.py`：

1. 读取 `attribute_week_heat.csv`、`attribute_week_target.csv`、`nodes_attribute.csv`、`edges_attribute_hierarchy.csv`。
2. 校验输入字段和关键唯一性。
3. 构造 lag、增长、移动统计、静态、图结构、低频过滤和时间特征。
4. 连接目标字段。
5. 校验样本表。
6. 写出 Parquet，并输出行数、周数、属性数和目标路径。

## 配置

更新 `src/fashion_trend/config.py`：

- 新增 `FEATURES_DIR = PROCESSED_DIR / "features"`。
- 新增 `PATH["trend_attribute_week_target"] = TREND_DIR / "attribute_week_target.csv"`。
- 新增 `PATH["features_trend_model_samples"] = FEATURES_DIR / "trend_model_samples.parquet"`。

保留已有路径命名，不重命名上一阶段产物。

## 错误处理

- 输入文件不存在时抛出 `FileNotFoundError`，脚本层记录路径并返回非零退出码。
- 输入缺少必要字段、存在重复键或派生字段不一致时抛出 `ValueError`。
- 读取 CSV / Parquet 失败时保留具体文件路径。
- 写 CSV 延续现有临时文件替换策略。
- 写 Parquet 时先写临时文件，成功后替换目标文件；失败时删除临时文件。

## 测试

扩展 `tests/test_trend.py`，继续使用小型 DataFrame 和临时目录，不依赖真实 H&M 数据。

测试覆盖：

- 完整属性-周面板会补齐零热度属性。
- 面板行数等于 `week_count * attribute_count`。
- 零热度属性的 `heat_cnt`、`heat_share`、`log_heat` 正确为 0。
- `rank_in_type` 覆盖零热度属性并保持稳定排序。
- 面板校验拒绝缺行、重复 `week_id + attr_id`、错误属性集合、错误热度派生字段。
- 趋势标签表排除最后一周。
- `target_growth`、`target_log_heat_t1`、`target_rank_in_type_t1` 公式正确。
- 趋势标签校验拒绝非有限增长值和重复键。
- 趋势训练样本从第 4 周开始，且不包含最后一周。
- lag、增长、移动统计、低频过滤字段只使用当前周及历史周。
- 图结构字段能从层级边正确派生。
- Parquet 写出会创建目录、替换旧文件，并在失败时清理临时文件。

最小验证命令：

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_trend -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python src/06_compute_attribute_week_heat.py
PYTHONPATH=src .venv/bin/python src/07_build_trend_targets.py
PYTHONPATH=src .venv/bin/python src/08_build_trend_model_samples.py
```

运行真实脚本后需要抽查：

- `attribute_week_heat.csv` 行数等于 `week_count * attribute_count`。
- `attribute_week_target.csv` 行数等于 `(week_count - 1) * attribute_count`。
- `trend_model_samples.parquet` 不包含最后一周，且最小样本周不早于第 4 周。
- 三个产物没有缺失值，关键派生字段和公式一致。
