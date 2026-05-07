# 趋势评价模块设计

## 状态

本设计是较早期的历史设计，早于后续领域驱动模块迁移。当前实现不再使用单文件 `src/fashion_trend/trend.py` 承载通用趋势工具；评价模块位于 `fashion_trend.trend.evaluation`，并直接复用 `fashion_trend.trend.schema`、`fashion_trend.trend.predictions` 和 `fashion_trend.foundation.io` 等具体模块。

## 范围

本轮实现趋势预测评价闭环，用于评价已经训练完成的趋势模型预测结果。

第一版只消费现有通用训练框架产物：

```text
outputs/models/<model_name>/predictions.csv
```

并输出趋势评价标准产物：

```text
outputs/metrics/<model_name>/trend_metrics.json
```

当前首个目标模型是：

```text
last_week
```

本轮不实现新模型，不进入推荐模块，不实现推荐评价，不把评价自动接入训练命令，也不输出 CSV 指标表。评价模块先以 JSON 作为稳定产物，后续模型只要继续写出同一预测表契约，就能复用同一评价入口。

## 设计结论

采用独立趋势评价 runner 加薄 CLI 的方案。

新增核心模块：

```text
src/fashion_trend/trend/evaluation.py
```

新增顶层脚本：

```text
src/11_eval_trend_model.py
```

CLI 只负责解析 `--model`、调用 runner、打印摘要和返回退出码。评价逻辑、路径推导、预测读取、指标计算、JSON payload 构造和写出都放在 `fashion_trend.trend.evaluation` 中。

这样可以保持当前训练层已经建立的边界：

- 训练产物继续放在 `outputs/models/<model_name>/`。
- 趋势评价产物放在 `outputs/metrics/<model_name>/`。
- 后续推荐评价使用独立产物命名，不和趋势评价混在一起。
- 后续 Moving Average、EWMA、LightGBM 只要遵守 `predictions.csv` 契约，就可以直接复用评价入口。

## 文件组织

新增或调整以下文件：

```text
src/11_eval_trend_model.py
src/fashion_trend/trend/evaluation.py
tests/test_trend.py
README.md
docs/gpt-research/implementation-plan.md
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `src/11_eval_trend_model.py` | 趋势模型评价 CLI，解析 `--model` 并调用评价 runner |
| `src/fashion_trend/trend/evaluation.py` | 评价路径推导、预测读取、指标计算、JSON payload 和写出 |
| `tests/test_trend.py` | 增加评价模块的单元测试 |
| `README.md` | 同步当前趋势评价命令、产物和验证说明 |
| `docs/gpt-research/implementation-plan.md` | 补充当前实现脚本名与产物命名，避免和旧计划脚本名漂移 |

通用数据校验、预测契约和写出工具已拆到具体模块。评价模块可以复用 `fashion_trend.trend.schema.TREND_MODEL_PREDICTION_COLUMNS`、`fashion_trend.trend.schema.TREND_MODEL_SPLIT_VALUES` 和 `fashion_trend.foundation.io.write_json_atomic()`，但不把评价业务逻辑放进趋势数据模块。

## CLI

推荐命令：

```sh
uv run python src/11_eval_trend_model.py --model last_week
```

第一版只需要一个必需参数：

```text
--model
```

默认输入路径由模型名推导：

```text
outputs/models/<model_name>/predictions.csv
```

默认输出路径由模型名推导：

```text
outputs/metrics/<model_name>/trend_metrics.json
```

CLI 成功时打印：

- 模型名称。
- 评价 split。
- 指标组数。
- `test` 的核心指标摘要。
- 输出文件路径。

CLI 对以下可预期错误返回非零状态码：

- 输入预测文件不存在。
- 预测表 schema 不符合契约。
- 预测表缺少 `valid` 或 `test` split。
- 预测表 `model_name` 与 CLI 请求不一致。
- 指标 payload 无法序列化为合法 JSON。
- 文件读取或写出失败。

## 数据流

评价入口运行流程：

```text
outputs/models/<model_name>/predictions.csv
    -> read_trend_model_predictions()
    -> validate_trend_model_predictions_for_evaluation()
    -> compute_trend_metrics()
    -> build_trend_metrics_payload()
    -> write_trend_metrics()
    -> outputs/metrics/<model_name>/trend_metrics.json
```

评价只读取预测表，不重新读取训练样本，不重新计算 split，不修改模型训练产物。

正式评价 split 为：

```text
valid
test
```

`train` 可以存在于预测表中，但不参与正式趋势评价指标。

## 输入契约

评价模块读取训练 runner 写出的预测表。列必须与当前趋势模型预测契约完全一致：

```text
week_id
attr_id
attr_type
attr_value
model_name
split
share_t
pred_share_t1
target_growth
pred_target_growth
target_rank_in_type_t1
```

评价指标只使用：

```text
target_growth
pred_target_growth
```

`pred_share_t1` 暂不用于趋势评价。它作为预测审查和后续推荐映射字段保留。

读取后需要校验：

- 列顺序与 `TREND_MODEL_PREDICTION_COLUMNS` 完全一致。
- 必需列不存在缺失值。
- `split` 只允许 `train`、`valid`、`test`。
- 预测表必须包含 `valid` 和 `test`。
- `model_name` 列只能有一个值，并且等于 CLI 请求的模型名。
- 数值列必须都是有限数值，不能有 `NaN`、`Infinity` 或 `-Infinity`。
- `week_id` 必须可安全解释为整数。

## 指标定义

趋势预测既是回归任务，也是排序任务。第一版输出以下指标：

```text
MAE
RMSE
Spearman
Precision@5
Precision@10
Precision@20
Recall@5
Recall@10
Recall@20
NDCG@5
NDCG@10
NDCG@20
```

排序目标统一使用增长率：

- 真实排序分数：`target_growth`
- 预测排序分数：`pred_target_growth`

评价分组粒度为：

```text
split + week_id + attr_type
```

也就是先评价每个 split 内每一周、每一种属性类型的 Top-K 趋势属性，再汇总到 split 和属性类型层面。这比把整段时间合并排序更贴近实际应用场景：每一周都需要判断下一周哪些属性更可能上升。

### 回归指标

组内 MAE：

```text
mean(abs(target_growth - pred_target_growth))
```

组内 RMSE：

```text
sqrt(mean((target_growth - pred_target_growth)^2))
```

汇总时对组指标做算术平均，而不是直接对所有行整体加权平均。这样可以避免大属性类型因为行数更多而压过小属性类型。

### Spearman

组内 Spearman 使用 `target_growth` 与 `pred_target_growth` 的 rank correlation。

如果组内真实值或预测值恒定，相关系数不可定义，该组 `spearman` 写为 `null`。汇总均值跳过 `null`，如果某个汇总范围内所有组都不可定义，则汇总 `spearman` 也为 `null`。

### Precision@K 和 Recall@K

对每个分组：

- 预测 Top-K：按 `pred_target_growth` 降序排序。
- 真实 Top-K：按 `target_growth` 降序排序。
- 并列时使用 `attr_id` 升序打破，保证结果可复现。

有效 K 为：

```text
effective_k = min(K, group_size)
```

Precision@K：

```text
hits / effective_k
```

Recall@K：

```text
hits / effective_k
```

在本轮定义下，真实集合和预测集合大小都使用同一个 `effective_k`，所以单个组内 Precision@K 与 Recall@K 数值可能相同。仍保留两个字段，是为了让产物和研究计划中的趋势评价指标保持一致，并给未来改成真实趋势阈值集合留下空间。

### NDCG@K

NDCG@K 使用预测排序位置上的真实 relevance 计算 DCG，再除以理想 DCG。

由于 `target_growth` 可能为负，组内 relevance 使用非负平移：

```text
relevance = target_growth - min(target_growth)
```

如果平移后所有 relevance 都为 0，说明该组真实增长没有可区分排序信号，组内 NDCG@K 写为 `null`，汇总时跳过 `null`。

## JSON 产物契约

输出文件固定为：

```text
outputs/metrics/<model_name>/trend_metrics.json
```

第一版 JSON 结构：

```json
{
  "model_name": "last_week",
  "prediction_path": "outputs/models/last_week/predictions.csv",
  "output_path": "outputs/metrics/last_week/trend_metrics.json",
  "evaluated_splits": ["valid", "test"],
  "ranking": {
    "target_column": "target_growth",
    "prediction_column": "pred_target_growth",
    "group_by": ["split", "week_id", "attr_type"],
    "k_values": [5, 10, 20]
  },
  "overall": {
    "valid": {
      "mae": 0.0,
      "rmse": 0.0,
      "spearman": 0.0,
      "precision_at_k": {
        "5": 0.0,
        "10": 0.0,
        "20": 0.0
      },
      "recall_at_k": {
        "5": 0.0,
        "10": 0.0,
        "20": 0.0
      },
      "ndcg_at_k": {
        "5": 0.0,
        "10": 0.0,
        "20": 0.0
      }
    },
    "test": {}
  },
  "by_attr_type": {
    "valid": {
      "colour_group_name": {}
    },
    "test": {}
  },
  "groups": {
    "valid": {
      "rows": 4736,
      "weeks": 8,
      "attr_types": 10,
      "ranking_groups": 80
    },
    "test": {}
  }
}
```

实际输出必须满足：

- 所有指标值是 JSON number 或 `null`。
- 不允许出现 `NaN`、`Infinity` 或 `-Infinity`。
- K 值在 JSON object key 中使用字符串 `"5"`、`"10"`、`"20"`，避免不同 JSON 解析器对数字 key 的处理差异。
- `overall` 和 `by_attr_type` 都只包含 `valid` 和 `test`。
- `groups` 记录每个 split 的行数、周数、属性类型数和实际评价组数。

## 写出策略

评价写出复用项目已有 JSON 写出方式：

```text
write_json(payload, output_path)
```

写出前先完整构建 payload 并校验 JSON 可序列化，避免留下部分产物。

评价模块只写：

```text
outputs/metrics/<model_name>/trend_metrics.json
```

不写入、不删除、不覆盖：

```text
outputs/models/<model_name>/
```

## 测试设计

测试继续放在现有 `tests/test_trend.py`，使用标准库 `unittest`，不依赖真实 H&M 数据。

需要覆盖：

- 路径推导：`outputs/metrics/<model>/trend_metrics.json`。
- 预测读取：列顺序、必需列、非法 split、非有限数值。
- 模型匹配：`model_name` 不唯一或和 CLI 请求不一致时报错。
- split 边界：缺少 `valid` 或 `test` 时报错；`train` 存在但不参与正式指标。
- 指标函数：MAE、RMSE、Spearman、Precision@K、Recall@K、NDCG@K 的小样本可手算结果。
- 聚合逻辑：先按 `split + week_id + attr_type` 计算组指标，再汇总到 `overall` 和 `by_attr_type`。
- 不可定义指标：Spearman 或 NDCG 不可定义时写 `null`，汇总时跳过。
- JSON 安全：payload 不包含 `NaN`、`Infinity` 或 `-Infinity`。
- 写出边界：只写 `outputs/metrics/<model>/trend_metrics.json`，不影响 `outputs/models/<model>/`。
- CLI 行为：参数错误保留 argparse 错误码，缺输入或非法预测返回非零，成功时打印摘要。

## 文档更新

README 需要同步：

- 阶段表新增趋势评价产物。
- 数据预处理流水线追加评价命令：

```sh
uv run python src/11_eval_trend_model.py --model last_week
```

- `last_week` baseline 后增加趋势评价小节，说明输入、输出、指标和运行命令。
- 后续阶段表继续保留趋势模型扩展和推荐模块，但不再把趋势评价描述为未实现。
- 验证章节补充评价模块测试覆盖。

`docs/gpt-research/implementation-plan.md` 只做轻量同步：

- 标注当前实现入口使用 `src/11_eval_trend_model.py`。
- 标注趋势评价产物使用 `outputs/metrics/<model>/trend_metrics.json`。
- 保留研究计划中的指标说明，不大范围重写原方案。

## 验收标准

实现完成后应满足：

- `uv run python src/11_eval_trend_model.py --model last_week` 能读取当前 `outputs/models/last_week/predictions.csv`。
- 命令生成 `outputs/metrics/last_week/trend_metrics.json`。
- JSON 中包含 `valid` 和 `test` 的 `overall`、`by_attr_type`、`groups`。
- 指标包含 MAE、RMSE、Spearman、Precision@5/10/20、Recall@5/10/20、NDCG@5/10/20。
- 排序指标按 `split + week_id + attr_type` 逐组计算，再汇总。
- 正式趋势评价不包含 `train`。
- 评价模块不修改 `outputs/models/<model_name>/` 下已有训练产物。
- README 与 implementation plan 中的命令、脚本名和产物路径与实现一致。
- 单元测试通过。
- 真实 `last_week` 评价命令通过，并检查实际 JSON 结构。

## 非目标

本轮明确不做：

- Moving Average、EWMA、LightGBM 或其他新趋势模型。
- 推荐模块。
- 推荐评价。
- 图表生成。
- 指标 CSV 表。
- 自动在训练命令后运行评价。
- 用 `pred_share_t1` 做趋势排序评价。
- 把评价结果用于模型选择或推荐重排序。
