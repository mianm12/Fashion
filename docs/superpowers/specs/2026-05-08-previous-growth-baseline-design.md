# Previous Growth 与 Last Week baseline 语义迁移设计

## 范围

本轮在已有通用趋势模型训练和评价框架上完成三个必须 baseline 的语义对齐：

```text
last_week
previous_growth
moving_average
```

当前 `last_week` 已实现的是 `pred_target_growth = growth_lag_1`，实际对应实施方案中的 Previous Growth 语义。本轮将这段增长率基线迁移到新模型名 `previous_growth`，并把 `last_week` 改回真正的 Last Week Heat baseline。

实现后仍然使用现有入口：

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
```

本轮不新增编号脚本，不修改趋势评价指标，不实现 LightGBM、推荐模块或更多 baseline。`moving_average` 保持现有语义不变。

## 设计结论

采用语义迁移方案：

- `previous_growth` 接管当前 `last_week` 的增长率基线实现。
- `last_week` 保留模型名，但公式改成上一周属性份额不变。
- `moving_average` 不改动，只继续作为增长率平滑 baseline。

这样三个必须 baseline 的含义分别是：

| 模型 | 语义 | 核心公式 |
| --- | --- | --- |
| `last_week` | 上一周热度不变 | `pred_share_t1 = share_t` |
| `previous_growth` | 上一段增长延续 | `pred_target_growth = growth_lag_1` |
| `moving_average` | 最近两段增长平滑 | `pred_target_growth = mean(growth_lag_1, growth_lag_2)` |

这是一次有意的语义迁移。迁移后，旧 `outputs/models/last_week/` 不再代表 Previous Growth；旧增长基线应由 `outputs/models/previous_growth/` 表示。

## 文件组织

计划新增或调整以下文件：

```text
src/fashion_trend/trend/models/baselines/previous_growth.py
src/fashion_trend/trend/models/baselines/last_week.py
src/fashion_trend/trend/models/registry.py
tests/test_trend_training.py
tests/test_trend_evaluation.py
README.md
docs/gpt-research/implementation-plan.md
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `previous_growth.py` | 承载增长率延续 baseline、参数常量和 trainer |
| `last_week.py` | 承载 Last Week Heat baseline、参数常量和 trainer |
| `registry.py` | 注册 `last_week`、`previous_growth`、`moving_average` |
| `tests/test_trend_training.py` | 覆盖三类 baseline 的公式、注册表、runner 和 CLI 行为 |
| `tests/test_trend_evaluation.py` | 确认新模型名复用标准评价 runner |
| `README.md` | 同步当前已实现 baseline、运行命令、产物和测试覆盖 |
| `implementation-plan.md` | 对齐研究计划中的 baseline 表和第 9 步说明 |

不新增模型专属 CLI 分支。顶层 `src/10_train_trend_model.py` 和 `src/11_eval_trend_model.py` 继续只接受 `--model`，具体模型由 registry 和 runner 处理。

## 模型语义

### previous_growth

`previous_growth` 直接预测趋势评价目标 `target_growth`：

```text
pred_target_growth = growth_lag_1
raw_pred_share_t1 = exp(pred_target_growth) * (share_t + epsilon) - epsilon
pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))
```

其中：

```text
epsilon = 1e-6
```

该模型是不拟合参数的确定性 baseline，含义是把上一段属性占比增长率延续到下一段。

### last_week

`last_week` 改为真正的 Last Week Heat baseline，核心预测是下一周属性份额等于当前周份额：

```text
pred_share_t1 = group_normalize(max(share_t, 0))
pred_target_growth = log((pred_share_t1 + epsilon) / (share_t + epsilon))
```

当输入 `share_t` 在 `split/week_id/attr_type` 内已经是合法分布时，`pred_share_t1` 等于 `share_t`。保留轻量归一化可以抵抗真实数据中的微小浮点误差，并继续满足标准预测契约。

`pred_target_growth` 由预测份额反推，目的是让趋势评价继续统一使用：

```text
target_growth vs pred_target_growth
```

### moving_average

`moving_average` 保持现有语义：

```text
pred_target_growth = mean(growth_lag_1, growth_lag_2)
```

本轮不修改它的 trainer、参数或产物契约，除非测试或文档需要把三模型列表补齐。

## 输入契约

三个 baseline 都读取训练 runner 传入的 split 样本，不直接读全局路径，不处理命令行，不写文件。

`previous_growth` 必需列：

```text
split
week_id
attr_id
attr_type
attr_value
share_t
growth_lag_1
target_growth
target_rank_in_type_t1
```

`last_week` 必需列：

```text
split
week_id
attr_id
attr_type
attr_value
share_t
target_growth
target_rank_in_type_t1
```

trainer 需要校验：

- 必需列存在。
- `split` 只包含 `train`、`valid`、`test`。
- 参与公式的数值字段可以转换为有限数值。
- 输出预测表满足 `TREND_MODEL_PREDICTION_COLUMNS` 和 `validate_trend_model_predictions()`。

如果输入不满足契约，抛出带模型名的 `ValueError`，不使用静默 fallback。

## 输出契约

标准训练产物为：

```text
outputs/models/last_week/predictions.csv
outputs/models/last_week/params.json
outputs/models/last_week/metadata.json

outputs/models/previous_growth/predictions.csv
outputs/models/previous_growth/params.json
outputs/models/previous_growth/metadata.json

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

`params.json` 需要让模型语义可审查：

- `last_week` 记录 `pred_share_t1 = normalized share_t` 和反推增长公式。
- `previous_growth` 记录 `pred_target_growth = growth_lag_1` 和从增长率派生份额的公式。
- `moving_average` 保持当前参数。

`metadata.json` 继续由 runner 统一生成。本轮不新增额外 artifact，不保存模型权重。

## 评价设计

趋势评价继续复用现有入口：

```sh
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
```

评价产物为：

```text
outputs/metrics/last_week/trend_metrics.json
outputs/metrics/previous_growth/trend_metrics.json
outputs/metrics/moving_average/trend_metrics.json
```

评价 split 仍然只包含：

```text
valid
test
```

指标继续使用：

```text
target_growth vs pred_target_growth
```

`pred_share_t1` 仍不参与当前趋势评价，但必须形成合法分布，为后续推荐映射保留稳定字段。

## 测试设计

训练层测试覆盖：

- `previous_growth` 参数稳定，模型名为 `previous_growth`。
- registry 能列出并返回 `PreviousGrowthTrainer`。
- `predict_previous_growth()` 输出列顺序等于 `TREND_MODEL_PREDICTION_COLUMNS`。
- `predict_previous_growth()` 的 `pred_target_growth` 等于 `growth_lag_1`。
- `last_week` 参数更新为 Last Week Heat 语义。
- `predict_last_week()` 的 `pred_share_t1` 等于按分组归一化后的 `share_t`。
- `predict_last_week()` 的 `pred_target_growth` 按 `pred_share_t1` 和 `share_t` 反推。
- 缺少必需列、非法 split、非有限数值时失败。
- `run_trend_model_training("previous_growth")` 能写出标准三件套。
- `run_trend_model_training("last_week")` 写出的 `last_week` 产物使用新语义。
- CLI 测试确认 `--model previous_growth` 走通用 runner。

评价层测试覆盖：

- 新模型名可以通过标准预测契约进入 `run_trend_model_evaluation()`。
- 评价 runner 不需要模型专属分支。
- `trend_metrics.json` 仍只聚合 valid/test。

真实验收命令：

```sh
uv run pytest
uv run python src/10_train_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
```

真实产物检查至少确认：

- 三个模型的 `predictions.csv`、`params.json`、`metadata.json` 都存在。
- 三个模型的 `trend_metrics.json` 都存在。
- `last_week` 的 `params.json` 不再写旧 `growth_lag_1` 公式。
- `previous_growth` 的 `params.json` 明确记录旧增长率基线公式。
- 三个预测表的 `pred_share_t1` 在 `split/week_id/attr_type` 内归一化。

## 风险和迁移说明

主要风险是语义迁移导致旧产物名被重新解释。迁移后：

```text
outputs/models/last_week/
```

表示 Last Week Heat，不再表示 Previous Growth。旧增长基线结果需要重新生成到：

```text
outputs/models/previous_growth/
```

README 和实施计划必须同步明确这一点，避免后续实验表述继续混用 `last_week` 与 Previous Growth。

另一个风险是 `share_t` 的浮点归一化误差。`last_week` 应在同一 `split/week_id/attr_type` 内对非负 `share_t` 做轻量归一化，再写入 `pred_share_t1`，同时继续让通用预测校验负责最终分布检查。
