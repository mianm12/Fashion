# 趋势 Last Week baseline 设计

## 范围

本轮只实现趋势预测 baseline 中最简单的一种：

```text
last_week
```

它用于打通第一条模型阶段闭环：

```text
trend_model_samples.parquet
    -> last_week baseline 预测
    -> outputs/models/last_week/
```

本轮不训练 LightGBM，不实现 Moving Average，不做完整趋势评价模块，也不进入推荐模块。baseline 的重点是建立统一的模型接口、输出格式和无未来泄漏的预测口径。

## 设计结论

`last_week` 按统一的 `target_growth` 目标口径实现。它使用上一段已经观测到的属性占比增长，预测下一段属性占比增长：

```text
pred_target_growth = growth_lag_1
```

其中：

```text
growth_lag_1 = log((share_t + 1e-6) / (share_lag_1 + 1e-6))
```

也就是假设“上一段增长趋势会延续到下一段”。这个口径直接对齐主标签：

```text
target_growth = log((share_t1 + 1e-6) / (share_t + 1e-6))
```

baseline 同时输出由预测增长反推得到的下一周预测占比：

```text
pred_share_t1 = exp(pred_target_growth) * (share_t + 1e-6) - 1e-6
```

`pred_share_t1` 是派生审查字段，不作为训练目标。第一版不做截断或归一化，避免把预测逻辑和评价展示策略混在一起。

命名上仍使用 `last_week`，表示使用上一段可观测变化作为下一段预测。第一版不再额外新增 `previous_growth` 模型名，避免两个 baseline 在 `target_growth` 口径下语义重复。

## 文件组织

新增顶层脚本：

```text
src/09_train_trend_baseline.py
```

新增模型包：

```text
src/fashion_trend/models/
    __init__.py
    baseline_last_week.py
```

`src/09_train_trend_baseline.py` 是 CLI 入口，负责解析参数、读取路径、调用模型、写出结果和输出日志。

`src/fashion_trend/models/baseline_last_week.py` 只负责 `last_week` baseline 的 DataFrame 级预测逻辑。它不读取全局路径，不处理命令行参数，不写文件。

## CLI

第一版命令：

```sh
uv run python src/09_train_trend_baseline.py --model last_week
```

第一版只支持：

```text
--model last_week
```

如果传入未知模型，脚本应失败并给出可定位错误，例如：

```text
不支持的 baseline 模型: moving_average
```

不使用 `--modle` 拼写。后续新增 baseline 时继续扩展 `--model` 的可选值。

## 参数放置

项目级路径放在 `src/fashion_trend/config.py`：

```text
OUTPUT_DIR
OUTPUT_MODELS_DIR
OUTPUT_METRICS_DIR
OUTPUT_FIGURES_DIR
OUTPUT_REPORTS_DIR
PATH["output_model_last_week_dir"]
PATH["output_model_last_week_predictions"]
PATH["output_model_last_week_params"]
PATH["output_model_last_week_metadata"]
```

路径分别指向：

```text
outputs/
outputs/models/
outputs/metrics/
outputs/figures/
outputs/reports/
outputs/models/last_week/
outputs/models/last_week/predictions.csv
outputs/models/last_week/params.json
outputs/models/last_week/metadata.json
```

`data/processed/` 继续保存可复用的数据流水线产物，例如趋势样本、标签和热度表。`outputs/` 统一保存实验产物，并按用途分为 `models/`、`metrics/`、`figures/` 和 `reports/`。本轮只写入 `outputs/models/last_week/`；评价指标、图表和报告目录先建立路径约定，不生成实际文件。

`last_week` 没有训练得到的权重，但仍保存参数文件：

```text
epsilon = 1e-6
model_name = "last_week"
```

`params.json` 至少包含：

```json
{
  "model_name": "last_week",
  "formula": "pred_target_growth = growth_lag_1",
  "derived_formula": "pred_share_t1 = exp(pred_target_growth) * (share_t + epsilon) - epsilon",
  "epsilon": 0.000001
}
```

`metadata.json` 至少包含输入路径、预测输出路径、行数、覆盖周数和覆盖属性数。第一版不记录运行时间戳，避免测试结果随时间变化。

CLI 第一版只暴露 `--model`。训练结束周、测试周、输出目录、归一化策略等参数暂不加入，避免第一版变成过早的实验配置系统。

## 输入数据

输入文件：

```text
data/processed/features/trend_model_samples.parquet
```

`last_week` 需要的输入列：

| 字段 | 用途 |
| --- | --- |
| `week_id` | 样本周 |
| `attr_id` | 属性 ID |
| `attr_type` | 属性类型 |
| `attr_value` | 属性值 |
| `share_t` | 当前周同类型热度占比 |
| `growth_lag_1` | 上一段已观测占比增长 |
| `target_growth` | 真实下一周占比增长，用于输出对照 |
| `target_rank_in_type_t1` | 真实下一周排名，用于后续排序评价 |

这些列都已经在 `trend_model_samples.parquet` 中生成。`last_week` 不读取 `share_t1`，不读取未来周热度，也不从 `attribute_week_target.csv` 重新拼接标签。

## 输出数据

输出文件：

```text
outputs/models/last_week/predictions.csv
```

字段顺序固定：

| 字段 | 说明 |
| --- | --- |
| `week_id` | 样本周 |
| `attr_id` | 属性 ID |
| `attr_type` | 属性类型 |
| `attr_value` | 属性值 |
| `model_name` | 固定为 `last_week` |
| `share_t` | 当前周真实同类型热度占比 |
| `pred_share_t1` | 预测下一周同类型热度占比 |
| `target_growth` | 真实下一周占比增长 |
| `pred_target_growth` | 预测下一周占比增长 |
| `target_rank_in_type_t1` | 真实下一周同类型排名 |

第一版不输出评价指标。评价指标会在后续独立阶段基于该预测表计算。

## 数据流

脚本执行流程：

1. 解析 `--model`。
2. 从 `PATH["features_trend_model_samples"]` 读取趋势样本。
3. 校验输入文件存在。
4. 校验 `last_week` 所需列存在。
5. 调用 `baseline_last_week.predict_last_week(samples)`。
6. 校验输出列顺序、唯一键和数值合法性。
7. 使用 CSV 写出函数写入 `PATH["output_model_last_week_predictions"]`。
8. 将参数写入 `PATH["output_model_last_week_params"]`。
9. 将运行元数据写入 `PATH["output_model_last_week_metadata"]`。
10. 输出预测行数、覆盖周数、覆盖属性数、预测输出路径和模型目录日志。

模型预测流程：

1. 从样本表复制标识列和目标列。
2. 设置 `model_name = "last_week"`。
3. 设置 `pred_target_growth = growth_lag_1`。
4. 按公式从 `pred_target_growth` 和 `share_t` 反推 `pred_share_t1`。
5. 按 `week_id`, `attr_type`, `attr_id` 稳定排序。

## 校验规则

输入校验：

- 必要列必须存在。
- `week_id + attr_id` 必须唯一。
- `share_t` 必须在 `[0, 1]` 范围内。
- `growth_lag_1` 必须是有限数值。
- `target_growth` 必须是有限数值。
- `target_rank_in_type_t1` 必须不缺失。

输出校验：

- 输出列顺序固定。
- `week_id + attr_id + model_name` 必须唯一。
- 不允许缺失值。
- `pred_target_growth` 必须是有限数值。
- 对 `last_week`，`pred_target_growth` 必须等于 `growth_lag_1`。
- 对每行重新按公式计算 `pred_share_t1`，结果必须一致。

`pred_share_t1` 第一版不强制在 `[0, 1]` 范围内，也不做 `week_id + attr_type` 内归一化。原因是它由增长率预测反推得到，可能小于 0、大于 1，或导致同一属性类型内预测占比和不等于 1。排序评价和推荐趋势分优先使用 `pred_target_growth`；如果后续要把预测占比用于展示，再在评价或展示阶段单独设计截断和归一化策略。

## 测试

新增或扩展 `tests/test_trend.py`，继续使用小型 DataFrame，不依赖真实 H&M 数据。

覆盖点：

- `last_week` 输出 `pred_target_growth = growth_lag_1`。
- `pred_share_t1` 按公式从 `pred_target_growth` 和 `share_t` 反推。
- 输入缺少必要列时失败。
- 输入存在重复 `week_id + attr_id` 时失败。
- 输入 `share_t` 超出 `[0, 1]` 时失败。
- 输入 `growth_lag_1` 存在非有限值时失败。
- 输出预测表列顺序固定，且没有缺失值和非有限数值。

最小验证命令：

```sh
uv run python -m unittest tests.test_trend -v
uv run python -m py_compile src/09_train_trend_baseline.py src/fashion_trend/models/baseline_last_week.py
```

如果本地已有完整真实数据，还应运行：

```sh
uv run python src/09_train_trend_baseline.py --model last_week
```

并直接检查：

- `outputs/models/last_week/predictions.csv` 存在。
- `outputs/models/last_week/params.json` 存在。
- `outputs/models/last_week/metadata.json` 存在。
- 输出行数等于 `trend_model_samples.parquet` 行数。
- `model_name` 只有 `last_week`。
- `pred_target_growth` 与 `growth_lag_1` 一致。
- `pred_target_growth` 全部为有限数值。
- `pred_share_t1` 与反推公式一致。
- `params.json` 中的 `epsilon` 与预测公式使用值一致。

## 非目标

- 不实现 `moving_average`。
- 不新增独立的 `previous_growth` 模型名。
- 不引入 scikit-learn、LightGBM 或其他新依赖。
- 不保存 pickle、joblib 或权重文件，因为 `last_week` 无需拟合参数。
- 不改动已有趋势样本构造逻辑。
- 不提交生成的数据文件，除非后续用户明确要求并完成产物审查。
