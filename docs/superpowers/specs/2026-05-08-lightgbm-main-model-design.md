# LightGBM 趋势主模型设计

## 范围

本轮在已经完成三类必须 baseline 的趋势模型训练与评价框架上，实现主模型：

```text
lightgbm
```

目标是完成 LightGBM Regressor 的训练、标准预测输出、统一趋势评价和基础可解释产物。实现后仍然使用现有入口：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm
```

本轮不新增编号脚本，不扩展上游趋势样本特征，不实现推荐模块，不生成持久化 baseline 对比报告，也不做复杂超参数搜索。LightGBM 必须遵守现有 `TrendTrainResult`、`predictions.csv`、`metadata.json`、`params.json` 和趋势评价 JSON 契约。

实现验收时必须读取四个模型的趋势评价结果：

```text
outputs/metrics/last_week/trend_metrics.json
outputs/metrics/previous_growth/trend_metrics.json
outputs/metrics/moving_average/trend_metrics.json
outputs/metrics/lightgbm/trend_metrics.json
```

验收摘要至少说明 LightGBM 在 valid/test 上的 MAE、RMSE、Spearman、NDCG@10 是否超过三类 baseline 中的最强结果。首版不要求 LightGBM 全面胜出；如果没有胜出，仍可视为训练闭环完成，但最终结果必须显式说明差距和可能原因，不能只报告“命令成功”。

## 设计结论

采用独立监督模型 trainer 加 registry 注册的方案。

新增模型文件：

```text
src/fashion_trend/trend/models/supervised/lightgbm.py
```

`lightgbm` trainer 只负责模型训练、预测表构造和模型专属 artifact 载荷；通用训练 runner 继续负责 split 读取、训练结果校验、metadata 构造和原子写盘。评价 runner 继续读取标准预测表并写出 `trend_metrics.json`。

这样可以保留已经稳定的扩展边界：

- `src/10_train_trend_model.py` 仍然只是 `--model` 入口。
- `src/11_eval_trend_model.py` 仍然按模型名读取标准预测表。
- `src/fashion_trend/trend/models/registry.py` 是唯一模型名到 trainer 的映射点。
- baseline 位于 `trend/models/baselines/`，监督模型位于 `trend/models/supervised/`。
- LightGBM 的模型细节不进入通用 runner。
- LightGBM native 包不能在 registry 导入路径上加载，避免 baseline 命令被 LightGBM 运行时依赖影响。

## 文件组织

计划新增或调整以下文件：

```text
src/fashion_trend/trend/models/supervised/lightgbm.py
src/fashion_trend/trend/models/registry.py
tests/test_trend_lightgbm.py
tests/test_trend_training.py
tests/test_trend_evaluation.py
README.md
docs/gpt-research/implementation-plan.md
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `lightgbm.py` | LightGBM trainer、特征清单、参数常量、延迟导入 LightGBM native 包、训练和预测逻辑 |
| `registry.py` | 注册 `lightgbm`，让统一训练入口可以发现主模型 |
| `tests/test_trend_lightgbm.py` | 覆盖监督模型专属行为：特征选择、分类编码、训练参数、artifact 和错误路径 |
| `tests/test_trend_training.py` | 只补通用训练 runner、registry 和 CLI 对 `lightgbm` 的接入点 |
| `tests/test_trend_evaluation.py` | 确认 `lightgbm` 预测表复用标准评价 runner |
| `README.md` | 同步 LightGBM 运行命令、产物路径和当前阶段状态 |
| `implementation-plan.md` | 将“后续计划”更新为已实现主模型入口和产物约定 |

不新增 `12_train_lightgbm_trend_model.py`。顶层编号脚本继续作为业务流程索引，业务包中的 trainer 是计算事实来源。

## 依赖导入边界

`src/fashion_trend/trend/models/registry.py` 会导入并注册所有 trainer，因此 LightGBM 的 native runtime 不能出现在 registry 的常规导入路径上。

硬约束：

- `src/fashion_trend/trend/models/supervised/lightgbm.py` 顶层不得执行 `import lightgbm` 或 `from lightgbm ... import ...`。
- 顶层只能导入标准库、`numpy`、`pandas`、项目内契约和纯 Python helper。
- LightGBM native 包只允许在 `LightGBMTrendTrainer.train()` 调用链内部延迟导入，例如私有函数 `_fit_lightgbm_model()`。
- 缺少 `lightgbm`、`libomp.dylib` 或其他 native runtime 时，只能让 `--model lightgbm` 失败；`--model last_week`、`--model previous_growth`、`--model moving_average` 不应受影响。
- 延迟导入捕获 `ImportError` 和 `OSError`，包装成带 `lightgbm`、依赖名称和安装/运行时线索的 `ValueError`。

测试需要覆盖 registry 和 baseline trainer 的导入不依赖 LightGBM native runtime。实现时不能用顶层导入换取代码简短。

## 模型语义

`lightgbm` 直接预测趋势评价目标：

```text
target_growth
```

训练输入是属性-周级样本。每一行代表某个属性在某个 week_id 的历史特征，预测该属性从当前周到下一周的增长趋势。

LightGBM 输出的原始预测作为：

```text
pred_target_growth
```

`pred_share_t1` 不由 LightGBM 单独学习，而是复用现有共享函数从预测增长率派生：

```text
raw_pred_share_t1 = exp(pred_target_growth) * (share_t + epsilon) - epsilon
pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))
```

其中：

```text
epsilon = 1e-6
```

这样可以保证 `pred_share_t1` 在同一 `split/week_id/attr_type` 内形成合法分布，避免把未归一化或越界份额传给后续推荐阶段。

## 特征设计

第一版只使用现有 `TREND_MODEL_SAMPLE_COLUMNS`，不修改 `trend_model_samples.parquet` 的上游构造。

数值特征：

```text
heat_t
share_t
log_heat_t
rank_in_type_t
heat_lag_1
heat_lag_2
heat_lag_3
heat_lag_4
share_lag_1
share_lag_2
share_lag_3
share_lag_4
growth_lag_1
growth_lag_2
acc_lag_1
heat_ma_4
share_ma_4
share_std_4
share_max_4
share_min_4
article_count
is_core_attr
parent_count
child_count
degree
history_total_heat_t
history_active_weeks_t
is_trend_eligible_t
week_index
week_mod_52
```

分类特征：

```text
attr_type
```

明确排除：

```text
attr_id
attr_value
target_growth
target_log_heat_t1
target_rank_in_type_t1
split
```

`attr_id` 和 `attr_value` 第一版不作为分类特征，避免高基数属性标识让小样本测试和后续泛化解释变得不稳定。`attr_type` 必须使用稳定的跨 split 分类编码，不能让 train、valid、test 各自独立推断 category levels。

实现中新增模型内部辅助函数：

```text
prepare_lightgbm_feature_frame(samples, attr_type_categories=None)
```

该函数负责：

- 校验并转换数值特征。
- 将 `attr_type` 转换为 pandas `category`。
- train split 调用时从 train 样本固化 `attr_type_categories`。
- valid/test 调用时复用 train 的 `attr_type_categories`。
- valid/test 如果出现 train 中不存在的 `attr_type`，直接抛出 `ValueError`。

这样可以避免 LightGBM 在不同 split 上看到不一致的分类编码，也让缺失分类、未知分类和空 split 都能被单独测试。

## 训练策略

训练方式固定为：

- train split 用于拟合模型。
- valid split 用于 early stopping 和训练摘要记录。
- test split 不参与拟合和 early stopping，只通过统一评价入口进入最终指标。

第一版参数采用实施计划中的稳妥默认值，并固定随机种子。默认 objective 仍为平方误差回归：

```json
{
  "objective": "regression",
  "n_estimators": 300,
  "learning_rate": 0.05,
  "num_leaves": 31,
  "max_depth": 6,
  "min_child_samples": 20,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "verbosity": -1
}
```

真实 `target_growth` 存在重尾和大量 `share_t=0` 样本，默认 `regression` 不是唯一语义。实现中需要把 objective 作为受控参数常量记录在 `params.json`，并保留后续切换到 `regression_l1` 等 robust objective 的清晰位置；本轮不新增 CLI 参数做 objective 切换。

训练产物必须记录目标分布和残差诊断，结构固定为 `{split: {metric: value}}`：

- `target_distribution`：按 train/valid/test 记录 count、min、max、mean、std、p01、p05、p50、p95、p99、`abs_gt_2`。
- `zero_share_rows`：按 split 记录 `share_t == 0` 的行数。
- `residual_distribution`：按 valid/test 记录 `target_growth - pred_target_growth` 的 count、min、max、mean、std、p01、p05、p50、p95、p99、mae、rmse。

这些诊断写入 LightGBM trainer 追加的 metadata 字段，用于解释平方误差回归在重尾目标上的表现；不能只保存 `best_iteration`。

early stopping 使用 valid split，第一版固定：

```json
{
  "stopping_rounds": 30
}
```

如果训练样本过小导致 early stopping 不能有效触发，仍然保留 `best_iteration` 记录；但不能把 valid 或 test 合并进 train 来掩盖样本不足。

## 输入契约

`lightgbm` trainer 读取 `TrendTrainContext` 中的 `split_frames`，不直接读取全局路径，不处理命令行，不写文件。

训练前必须校验：

- `train`、`valid`、`test` split 都存在且非空。
- 所有数值特征、分类特征和 `target_growth` 列存在。
- 数值特征和目标列可以转换为有限数值。
- `attr_type` 不缺失。
- split 样本只包含合法 split 名称。

输入不满足契约时抛出带 `lightgbm` 模型名和缺失字段信息的 `ValueError`，不使用静默 fallback，不 mock 成功路径。

## 输出契约

标准训练产物写入：

```text
outputs/models/lightgbm/predictions.csv
outputs/models/lightgbm/params.json
outputs/models/lightgbm/metadata.json
```

可解释和模型 artifact 写入：

```text
outputs/models/lightgbm/feature_importance.csv
outputs/models/lightgbm/model.txt
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
lightgbm
```

`params.json` 至少记录：

- `model_name`
- `model_type`
- `target_column`
- `numeric_features`
- `categorical_features`
- `excluded_columns`
- `epsilon`
- `lightgbm_params`
- `early_stopping`
- `best_iteration`
- `objective`
- `allowed_objectives`

`metadata.json` 的 runner 核心字段仍由通用训练 runner 生成。LightGBM trainer 只能追加非核心摘要字段，例如：

- `target_column`
- `numeric_features`
- `categorical_features`
- `attr_type_categories`
- `best_iteration`
- `best_score`
- `target_distribution`
- `zero_share_rows`
- `residual_distribution`

`feature_importance.csv` 至少包含：

```text
feature
split_importance
gain_importance
normalized_gain_importance
```

`split_importance` 使用 LightGBM split 次数重要性，`gain_importance` 使用 gain 重要性。`normalized_gain_importance = gain_importance / total_gain`；当 `total_gain == 0` 时，所有 `normalized_gain_importance` 写为 `0.0`，禁止写出 NaN、inf 或触发除零错误。

`model.txt` 使用 LightGBM booster 的文本格式，作为本轮可审查模型文件。artifact 路径必须是输出目录下的安全相对路径，继续由现有 artifact safety 逻辑校验。

## 评价设计

趋势评价继续复用现有入口：

```sh
uv run python src/11_eval_trend_model.py --model lightgbm
```

评价输入：

```text
outputs/models/lightgbm/predictions.csv
```

评价输出：

```text
outputs/metrics/lightgbm/trend_metrics.json
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

`pred_share_t1` 不参与当前趋势评价指标，但必须通过预测契约校验，因为后续推荐阶段会消费该字段。

## 验收对比

LightGBM 训练和评价完成后，验收步骤必须读取以下四个模型的 `trend_metrics.json`：

```text
last_week
previous_growth
moving_average
lightgbm
```

如果任一 baseline 的 `trend_metrics.json` 不存在，验收前必须先补跑对应模型的训练和评价命令：

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
```

如果某个 baseline 预测文件存在但 metrics 缺失，可以只补跑对应评价命令；如果预测文件也缺失，必须先补跑训练。验收摘要不能依赖当前机器上偶然存在的 `outputs/`。

摘要至少覆盖：

- valid/test 的 MAE。
- valid/test 的 RMSE。
- valid/test 的 Spearman。
- valid/test 的 NDCG@10。
- 每个 split 上三类 baseline 的最强结果。
- LightGBM 是否超过对应 split 和指标上的最强 baseline。

指标方向固定为：MAE 和 RMSE 越低越好，Spearman 和 NDCG@10 越高越好。

这一步不生成正式报告文件，也不把 baseline 对比表纳入标准产物；它是本轮实现验收的一部分，目的是判断主模型是否真的带来增益。如果 LightGBM 未超过最强 baseline，最终说明必须明确指出失败指标和可能原因，例如目标重尾、特征不足、参数保守或样本窗口过短。

## 错误处理

LightGBM trainer 的错误处理遵循 debug-first 原则：

- 缺少必需列时报告缺失列名。
- 数值列无法转换或包含非有限值时报告模型输入字段问题。
- 分类列缺失时报告 `attr_type` 输入问题。
- split 缺失或为空时报告具体 split。
- LightGBM native 依赖缺失时，只让 `lightgbm` 模型训练失败，并报告延迟导入失败原因。
- LightGBM 训练或预测失败时向上抛出可定位错误，不写部分成功产物。

trainer 不吞掉异常，不返回假成功，不用默认常数预测掩盖训练失败。标准产物写出仍由通用 runner 的暂存目录和回滚逻辑负责。

## 测试设计

测试重点覆盖主模型接入点和监督模型专属行为：

新增 `tests/test_trend_lightgbm.py`，专门覆盖监督模型内部行为：

- `lightgbm` 的模型名、模型类型、参数、特征清单稳定。
- `supervised/lightgbm.py` 顶层不导入 LightGBM native 包，registry 和 baseline trainer 导入不依赖 native runtime。
- `prepare_lightgbm_feature_frame()` 使用 train categories 固化 `attr_type` 编码。
- valid/test 出现未知 `attr_type` 时失败。
- 缺失特征、非有限数值、缺失 `attr_type`、空 split 都失败。
- trainer 在小样本上返回标准 `TrendTrainResult`。
- `predictions.csv` 满足 `TREND_MODEL_PREDICTION_COLUMNS`。
- `pred_target_growth` 是有限数值。
- `pred_share_t1` 在 `split/week_id/attr_type` 内归一化。
- metadata 包含 target 分布、零 share 行数和 valid/test 残差分布。
- distribution metadata 结构固定为 `{split: {metric: value}}`。
- `feature_importance.csv` 包含 `feature`、`split_importance`、`gain_importance`、`normalized_gain_importance`。
- gain 总和为 0 时 `normalized_gain_importance` 全部为 `0.0`。
- `model.txt` 作为 binary artifact 写入输出目录。

现有测试文件只补通用接入点：

- `tests/test_trend_training.py` 覆盖 registry 能列出并返回 `LightGBMTrendTrainer`，`run_trend_model_training("lightgbm")` 写出标准三件套和两个 artifact，CLI 接受 `--model lightgbm` 并走通用 runner。
- `tests/test_trend_evaluation.py` 覆盖 `run_trend_model_evaluation("lightgbm")` 能读取预测并写出 `trend_metrics.json`。

实现完成后的验证命令：

```sh
uv run pytest tests/test_trend_lightgbm.py tests/test_trend_training.py tests/test_trend_evaluation.py
uv run pytest
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm
uv run black --check src tests
uv run isort --check-only src tests
```

如果 `uv run` 因本机 cache 权限失败，可使用 `.venv/bin/python -m pytest` 做测试验证，并在最终结果中说明环境原因。真实训练和评价仍优先使用项目 README 中的 `uv run python ...` 命令。

## 非目标

本轮明确不做：

- 不新增 LightGBM 专属编号脚本。
- 不新增上游趋势样本字段。
- 不把 `attr_id` 或 `attr_value` 作为第一版分类特征。
- 不做网格搜索、贝叶斯调参或多实验管理。
- 不生成持久化 baseline 对比汇总表或正式报告。
- 不实现推荐重排序或推荐评价。
- 不提交 `outputs/` 下生成的数据和模型产物。
