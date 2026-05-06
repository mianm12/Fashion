# 趋势 Last Week baseline 设计

## 范围

本轮只实现趋势预测 baseline 中最简单的一种：

```text
last_week
```

它用于打通第一条模型阶段闭环：

```text
trend_model_samples.parquet
    -> train/valid/test 时间切分
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
src/09_split_trend_model_samples.py
src/10_train_trend_baseline.py
```

新增模型包：

```text
src/fashion_trend/models/
    __init__.py
    baseline_last_week.py
```

`src/09_split_trend_model_samples.py` 是时间切分入口，负责读取完整趋势样本，按 `config.py` 中的切分配置生成 `train`、`valid`、`test` 三份样本文件。

`src/10_train_trend_baseline.py` 是 baseline CLI 入口，负责解析参数、读取已切分样本、调用模型、写出结果和输出日志。它不负责划分数据集。

`src/fashion_trend/models/baseline_last_week.py` 只负责 `last_week` baseline 的 DataFrame 级预测逻辑。它不读取全局路径，不处理命令行参数，不写文件。

## CLI

第一版命令：

```sh
uv run python src/09_split_trend_model_samples.py
uv run python src/10_train_trend_baseline.py --model last_week
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
TREND_SPLIT_VALID_WEEKS
TREND_SPLIT_TEST_WEEKS
OUTPUT_DIR
OUTPUT_MODELS_DIR
OUTPUT_METRICS_DIR
OUTPUT_FIGURES_DIR
OUTPUT_REPORTS_DIR
PATH["features_trend_model_samples_train"]
PATH["features_trend_model_samples_valid"]
PATH["features_trend_model_samples_test"]
PATH["features_trend_model_samples_split_metadata"]
PATH["output_model_last_week_dir"]
PATH["output_model_last_week_predictions"]
PATH["output_model_last_week_params"]
PATH["output_model_last_week_metadata"]
```

路径分别指向：

```text
data/processed/features/trend_model_samples_train.parquet
data/processed/features/trend_model_samples_valid.parquet
data/processed/features/trend_model_samples_test.parquet
data/processed/features/trend_model_samples_split_metadata.json
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

时间切分参数统一放在 `config.py`：

```text
TREND_SPLIT_VALID_WEEKS = 8
TREND_SPLIT_TEST_WEEKS = 8
```

baseline 训练脚本不接受切分参数，也不重新计算切分边界。

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

`metadata.json` 至少包含三份 split 输入路径、预测输出路径、行数、覆盖周数、覆盖属性数、每个 split 的周范围和行数。第一版不记录运行时间戳，避免测试结果随时间变化。

CLI 第一版只暴露 `--model`。训练结束周、测试周、输出目录、归一化策略等参数暂不加入，避免第一版变成过早的实验配置系统。

## 输入数据

时间切分脚本输入文件：

```text
data/processed/features/trend_model_samples.parquet
```

baseline 训练脚本输入文件：

```text
data/processed/features/trend_model_samples_train.parquet
data/processed/features/trend_model_samples_valid.parquet
data/processed/features/trend_model_samples_test.parquet
```

`last_week` 需要的输入列：

| 字段 | 用途 |
| --- | --- |
| `split` | 数据集切分标记，由时间切分脚本生成 |
| `week_id` | 样本周 |
| `attr_id` | 属性 ID |
| `attr_type` | 属性类型 |
| `attr_value` | 属性值 |
| `share_t` | 当前周同类型热度占比 |
| `growth_lag_1` | 上一段已观测占比增长 |
| `target_growth` | 真实下一周占比增长，用于输出对照 |
| `target_rank_in_type_t1` | 真实下一周排名，用于后续排序评价 |

除 `split` 外，这些列都已经在 `trend_model_samples.parquet` 中生成。`split` 由 `src/09_split_trend_model_samples.py` 加入到三份切分后的 parquet 中。`last_week` 不读取 `share_t1`，不读取未来周热度，也不从 `attribute_week_target.csv` 重新拼接标签。

## 时间切分

需要划分 `train`、`valid`、`test`，但必须由独立脚本完成，不能和 baseline 训练脚本混在一起。切分必须按时间进行，不能随机切分。

默认切分策略：

```text
test_weeks = TREND_SPLIT_TEST_WEEKS
valid_weeks = TREND_SPLIT_VALID_WEEKS
test:  最后的 8 个样本周
valid: test 之前的 8 个样本周
train: 更早的全部样本周
```

切分基于 `trend_model_samples.parquet` 中实际出现的样本 `week_id`。设样本最大周为 `max_sample_week`：

```text
test_start_week = max_sample_week - test_weeks + 1
valid_start_week = test_start_week - valid_weeks

train: week_id < valid_start_week
valid: valid_start_week <= week_id < test_start_week
test:  week_id >= test_start_week
```

`last_week` 没有拟合参数，所以不会只在 `train` 上训练；它会对所有样本周生成预测，并在输出中标记 `split`。后续评价阶段可只报告 `valid` 和 `test`，LightGBM 阶段也复用同一时间切分边界，保证 baseline 和主模型可比较。

如果样本周数不足以同时保留非空 `train`、`valid`、`test`，脚本应失败并说明当前样本周数、`valid_weeks` 和 `test_weeks`。

切分脚本输出四个文件：

```text
data/processed/features/trend_model_samples_train.parquet
data/processed/features/trend_model_samples_valid.parquet
data/processed/features/trend_model_samples_test.parquet
data/processed/features/trend_model_samples_split_metadata.json
```

三份 parquet 都保留完整样本字段，并新增固定值 `split` 字段。`trend_model_samples_split_metadata.json` 记录切分配置、输入路径、输出路径、每个 split 的周范围、行数和属性数。

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
| `split` | `train`、`valid` 或 `test` |
| `share_t` | 当前周真实同类型热度占比 |
| `pred_share_t1` | 预测下一周同类型热度占比 |
| `target_growth` | 真实下一周占比增长 |
| `pred_target_growth` | 预测下一周占比增长 |
| `target_rank_in_type_t1` | 真实下一周同类型排名 |

第一版不输出评价指标。评价指标会在后续独立阶段基于该预测表计算。

## 数据流

时间切分脚本流程：

1. 从 `PATH["features_trend_model_samples"]` 读取完整趋势样本。
2. 从 `TREND_SPLIT_VALID_WEEKS` 和 `TREND_SPLIT_TEST_WEEKS` 读取切分配置。
3. 基于样本 `week_id` 计算 `train`、`valid`、`test` 时间边界。
4. 分别生成三份带 `split` 字段的 DataFrame。
5. 校验三份 split 非空、周范围连续且互不重叠。
6. 写出 `PATH["features_trend_model_samples_train"]`。
7. 写出 `PATH["features_trend_model_samples_valid"]`。
8. 写出 `PATH["features_trend_model_samples_test"]`。
9. 写出 `PATH["features_trend_model_samples_split_metadata"]`。
10. 输出每个 split 的周范围、行数、属性数和目标文件路径日志。

baseline 脚本流程：

1. 解析 `--model`。
2. 读取 `PATH["features_trend_model_samples_train"]`。
3. 读取 `PATH["features_trend_model_samples_valid"]`。
4. 读取 `PATH["features_trend_model_samples_test"]`。
5. 校验三份 split 输入文件存在。
6. 校验 `last_week` 所需列存在。
7. 合并三份 split 样本，只保留输入已有的 `split` 字段，不重新计算切分边界。
8. 调用 `baseline_last_week.predict_last_week(samples)`。
9. 校验输出列顺序、唯一键、split 和数值合法性。
10. 使用 CSV 写出函数写入 `PATH["output_model_last_week_predictions"]`。
11. 将参数写入 `PATH["output_model_last_week_params"]`。
12. 将运行元数据写入 `PATH["output_model_last_week_metadata"]`。
13. 输出预测行数、覆盖周数、覆盖属性数、split 周范围、预测输出路径和模型目录日志。

模型预测流程：

1. 从样本表复制标识列和目标列。
2. 设置 `model_name = "last_week"`。
3. 设置 `pred_target_growth = growth_lag_1`。
4. 按公式从 `pred_target_growth` 和 `share_t` 反推 `pred_share_t1`。

baseline 脚本在模型输出后保留输入已有的 `split` 字段，并按 `week_id`, `attr_type`, `attr_id` 稳定排序。

## 校验规则

切分脚本输入校验：

- 必要列必须存在。
- `week_id + attr_id` 必须唯一。
- `week_id` 必须是整数周编号。
- 样本周数必须足够产生非空 `train`、`valid`、`test`。

切分输出校验：

- 三份 parquet 都必须写入 `data/processed/features/`。
- 三份 parquet 的列集合必须与原始样本一致，并额外包含 `split` 字段。
- 每份 parquet 的 `split` 字段必须分别固定为 `train`、`valid`、`test`。
- 三份 split 都必须非空。
- 三份 split 的 `week_id` 范围必须连续且互不重叠。
- 三份 split 合并后的 `week_id + attr_id` 集合必须等于原始样本。
- `trend_model_samples_split_metadata.json` 中的周范围和行数必须与三份 parquet 一致。

baseline 输入校验：

- 三份 split parquet 必须存在。
- 必要列必须存在。
- `week_id + attr_id` 必须唯一。
- `split` 只允许为 `train`、`valid`、`test`。
- `train`、`valid`、`test` 都必须非空。
- split 必须按 `week_id` 连续且互不重叠。
- `share_t` 必须在 `[0, 1]` 范围内。
- `growth_lag_1` 必须是有限数值。
- `target_growth` 必须是有限数值。
- `target_rank_in_type_t1` 必须不缺失。

输出校验：

- 输出列顺序固定。
- `week_id + attr_id + model_name` 必须唯一。
- `split` 只允许为 `train`、`valid`、`test`。
- `train`、`valid`、`test` 都必须非空。
- split 必须与输入 split 一致，不能在 baseline 脚本内被重新划分。
- 不允许缺失值。
- `pred_target_growth` 必须是有限数值。
- 对 `last_week`，`pred_target_growth` 必须等于 `growth_lag_1`。
- 对每行重新按公式计算 `pred_share_t1`，结果必须一致。

`pred_share_t1` 第一版不强制在 `[0, 1]` 范围内，也不做 `week_id + attr_type` 内归一化。原因是它由增长率预测反推得到，可能小于 0、大于 1，或导致同一属性类型内预测占比和不等于 1。排序评价和推荐趋势分优先使用 `pred_target_growth`；如果后续要把预测占比用于展示，再在评价或展示阶段单独设计截断和归一化策略。

## 测试

新增或扩展 `tests/test_trend.py`，继续使用小型 DataFrame，不依赖真实 H&M 数据。

覆盖点：

- 时间切分脚本按 `config.py` 中的 `TREND_SPLIT_VALID_WEEKS` 和 `TREND_SPLIT_TEST_WEEKS` 生成三份 parquet。
- 三份 split 的周范围连续且互不重叠。
- 三份 split 合并后覆盖原始样本全集。
- `trend_model_samples_split_metadata.json` 与三份 parquet 的周范围和行数一致。
- `last_week` 输出 `pred_target_growth = growth_lag_1`。
- `pred_share_t1` 按公式从 `pred_target_growth` 和 `share_t` 反推。
- 输入缺少必要列时失败。
- 输入存在重复 `week_id + attr_id` 时失败。
- 输入 `share_t` 超出 `[0, 1]` 时失败。
- 输入 `growth_lag_1` 存在非有限值时失败。
- 样本周数不足以产生非空 `train`、`valid`、`test` 时失败。
- baseline 脚本读取已切分数据，保留输入 `split`，不重新计算切分。
- 输出预测表列顺序固定，且没有缺失值和非有限数值。

最小验证命令：

```sh
uv run python -m unittest tests.test_trend -v
uv run python -m py_compile src/09_split_trend_model_samples.py src/10_train_trend_baseline.py src/fashion_trend/models/baseline_last_week.py
```

如果本地已有完整真实数据，还应运行：

```sh
uv run python src/09_split_trend_model_samples.py
uv run python src/10_train_trend_baseline.py --model last_week
```

并直接检查：

- `data/processed/features/trend_model_samples_train.parquet` 存在。
- `data/processed/features/trend_model_samples_valid.parquet` 存在。
- `data/processed/features/trend_model_samples_test.parquet` 存在。
- `data/processed/features/trend_model_samples_split_metadata.json` 存在。
- `outputs/models/last_week/predictions.csv` 存在。
- `outputs/models/last_week/params.json` 存在。
- `outputs/models/last_week/metadata.json` 存在。
- 三份 split parquet 行数之和等于 `trend_model_samples.parquet` 行数。
- `predictions.csv` 行数等于三份 split parquet 行数之和。
- `model_name` 只有 `last_week`。
- `pred_target_growth` 与 `growth_lag_1` 一致。
- `pred_target_growth` 全部为有限数值。
- `pred_share_t1` 与反推公式一致。
- `split` 包含 `train`、`valid`、`test`，且与三份 split parquet 一致。
- `params.json` 中的 `epsilon` 与预测公式使用值一致。
- `metadata.json` 中的 split 周范围和行数与 `predictions.csv` 一致。

## 非目标

- 不实现 `moving_average`。
- 不新增独立的 `previous_growth` 模型名。
- 不引入 scikit-learn、LightGBM 或其他新依赖。
- 不保存 pickle、joblib 或权重文件，因为 `last_week` 无需拟合参数。
- 不改动已有趋势样本构造逻辑。
- 不提交生成的数据文件，除非后续用户明确要求并完成产物审查。
