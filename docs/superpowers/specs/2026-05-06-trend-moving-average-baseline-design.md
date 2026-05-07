# Moving Average 趋势 baseline 设计

## 范围

本轮在已有通用趋势模型训练框架上新增第二个 baseline：

```text
moving_average
```

目标是完成一个可训练、可评价、可和 `last_week` 对比的基础趋势预测模型。实现后仍然使用现有入口：

```sh
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

本轮不修改 `last_week`，不新增模型参数 CLI，不实现 EWMA、LightGBM 或推荐模块，也不改变趋势评价指标定义。`moving_average` 必须遵守现有训练 runner、预测表、metadata 和评价 JSON 契约。

## 设计结论

采用独立 trainer 文件加 registry 注册的方案。

新增模型文件：

```text
src/fashion_trend/trend/models/moving_average.py
```

`moving_average` trainer 只负责模型公式和预测表构造，不直接读取全局路径、不处理命令行、不写文件。训练 runner 继续负责 split 读取、结果校验、metadata 构造和产物写出。评价 runner 继续读取标准预测表并写出标准趋势评价结果。

这样可以沿用已经稳定的模型扩展方式：

- 每个模型一个独立文件。
- 每个模型实现自己的 `TrendModelTrainer`。
- `registry.py` 是唯一 model name 到 trainer 的映射点。
- 顶层 CLI 不出现具体模型分支判断。

## 文件组织

计划新增或调整以下文件：

```text
src/fashion_trend/trend/models/moving_average.py
src/fashion_trend/trend/models/registry.py
tests/test_trend.py
README.md
docs/gpt-research/implementation-plan.md
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `src/fashion_trend/trend/models/moving_average.py` | `moving_average` trainer、参数常量和预测公式 |
| `src/fashion_trend/trend/models/registry.py` | 注册 `moving_average`，让通用训练入口可以发现模型 |
| `tests/test_trend.py` | 增加公式、registry、runner、评价复用相关测试 |
| `README.md` | 同步 Moving Average baseline 的运行命令和产物 |
| `docs/gpt-research/implementation-plan.md` | 对齐当前统一训练入口和标准评价产物 |

不新增单独的训练脚本。`src/10_train_trend_model.py` 继续作为唯一趋势模型训练 CLI，`src/11_eval_trend_model.py` 继续作为趋势模型评价 CLI。

## 模型语义

`moving_average` 直接预测趋势评价目标 `target_growth`。

输入样本已经包含最近两段可用增长：

```text
growth_lag_1
growth_lag_2
```

预测公式固定为：

```text
pred_target_growth = mean(growth_lag_1, growth_lag_2)
pred_share_t1 = exp(pred_target_growth) * (share_t + epsilon) - epsilon
```

其中：

```text
epsilon = 1e-6
```

这个 baseline 的含义是用最近两段已观测增长的简单平均，预测下一段属性占比增长。它比 `last_week` 的单段增长更平滑，但仍然是无拟合参数的确定性 baseline。

当前 `trend_model_samples.parquet` 只有 `growth_lag_1` 和 `growth_lag_2` 两个增长滞后字段。本轮不修改上游样本构造以增加更多增长 lag，也不引入可配置窗口。

## 输入契约

`moving_average` trainer 读取训练 runner 传入的 split 样本。必需列为：

```text
split
week_id
attr_id
attr_type
attr_value
share_t
growth_lag_1
growth_lag_2
target_growth
target_rank_in_type_t1
```

trainer 需要校验：

- 必需列存在。
- `split` 只包含 `train`、`valid`、`test`。
- 公式所需数值字段可以计算出有限结果。

如果缺少必需列或 split 非法，抛出带模型名的 `ValueError`，不要静默 fallback。

## 输出契约

训练产物写入：

```text
outputs/models/moving_average/predictions.csv
outputs/models/moving_average/params.json
outputs/models/moving_average/metadata.json
```

`predictions.csv` 列必须完全等于：

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

`model_name` 必须全表为：

```text
moving_average
```

`params.json` 至少记录：

```json
{
  "model_name": "moving_average",
  "formula": "pred_target_growth = mean(growth_lag_1, growth_lag_2)",
  "derived_formula": "pred_share_t1 = exp(pred_target_growth) * (share_t + epsilon) - epsilon",
  "epsilon": 0.000001,
  "growth_lags": ["growth_lag_1", "growth_lag_2"]
}
```

`metadata.json` 的核心字段继续由 runner 统一生成。`moving_average` 不声明额外 artifact，不保存模型权重。

## 评价设计

评价复用现有趋势评价入口：

```sh
uv run python src/11_eval_trend_model.py --model moving_average
```

评价输入：

```text
outputs/models/moving_average/predictions.csv
```

评价输出：

```text
outputs/metrics/moving_average/trend_metrics.json
```

评价 split 仍然只包含：

```text
valid
test
```

排序和回归指标继续使用：

```text
target_growth vs pred_target_growth
```

`pred_share_t1` 不参与当前趋势评价，只作为预测审查和后续推荐映射字段。

## 测试设计

测试重点覆盖新增模型和既有 runner 的扩展边界：

- `moving_average` 参数稳定，模型名为 `moving_average`。
- registry 能列出并返回 `MovingAverageTrainer`。
- 未知模型错误中的可用模型列表包含 `last_week` 和 `moving_average`。
- `predict_moving_average()` 输出列顺序等于 `TREND_MODEL_PREDICTION_COLUMNS`。
- `pred_target_growth` 等于 `growth_lag_1` 与 `growth_lag_2` 的平均。
- `pred_share_t1` 根据 `pred_target_growth`、`share_t` 和 `epsilon` 推导。
- 缺少 `growth_lag_2` 等必需列时失败。
- 非法 split 失败。
- `run_trend_model_training("moving_average")` 能写出标准三件套。
- `run_trend_model_evaluation("moving_average")` 能读取预测并写出 `trend_metrics.json`。
- CLI 测试确认 `src/10_train_trend_model.py --model moving_average` 走通用 runner，不新增模型专属分支。

实现完成后的最小验证命令：

```sh
uv run python -m py_compile src/fashion_trend/trend/models/moving_average.py src/fashion_trend/trend/models/registry.py
uv run python -m unittest discover -s tests -v
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

真实数据验证需要检查：

- `outputs/models/moving_average/predictions.csv` 存在。
- `outputs/models/moving_average/params.json` 存在。
- `outputs/models/moving_average/metadata.json` 存在。
- `outputs/metrics/moving_average/trend_metrics.json` 存在。
- 预测表无缺失值。
- `model_name` 只有 `moving_average`。
- 预测表与输入 split 样本合并后，`pred_target_growth` 与 `growth_lag_1/growth_lag_2` 平均值一致。
- 评价 JSON 的 `model_name`、`evaluated_splits`、`overall`、`by_attr_type` 和 `groups` 字段完整。

## 文档同步

README 需要同步：

- 阶段表加入或扩展 Moving Average baseline 产物。
- 数据预处理流水线追加 `moving_average` 的训练和评价命令。
- `last_week` baseline 后新增 `moving_average` baseline 小节，说明公式、产物和命令。
- 后续阶段表从“后续再做 Moving Average”调整为“后续再做 EWMA baseline，再考虑 LightGBM”。
- 验证章节补充 Moving Average baseline 的测试覆盖。

`docs/gpt-research/implementation-plan.md` 需要继续对齐当前实现：

- baseline 训练入口使用 `src/10_train_trend_model.py --model <model>`。
- 趋势评价入口使用 `src/11_eval_trend_model.py --model <model>`。
- 趋势评价产物使用 `outputs/metrics/<model>/trend_metrics.json`。

## 非目标

本轮明确不做：

- 不重命名或重定义已有 `last_week`。
- 不新增 `--window`、`--lags` 或其他模型参数 CLI。
- 不修改 `trend_model_samples.parquet` 的上游特征构造。
- 不实现 `moving_average_share`。
- 不实现 EWMA、Linear Regression、Ridge Regression 或 LightGBM。
- 不把训练命令和评价命令自动串联。
- 不实现模型对比汇总表或图表。
- 不进入推荐模块和推荐评价。
