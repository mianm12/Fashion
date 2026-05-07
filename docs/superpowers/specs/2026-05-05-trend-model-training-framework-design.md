# 趋势模型训练框架设计

## 状态

本设计是较早期的历史设计，早于后续领域驱动模块迁移。当前实现不再使用单文件 `src/fashion_trend/trend.py` 承载趋势逻辑；趋势数据、样本、切分和预测契约分别位于 `fashion_trend.trend.*` 具体模块，稳定交易 reader 位于 `fashion_trend.transactions.weekly`，目录图 reader 位于 `fashion_trend.catalog.graph`。

## 范围

本轮将当前 `last_week` baseline 训练入口升级为面向所有趋势预测模型的训练框架，但只迁移和实现一个模型：

```text
last_week
```

框架需要同时适配两类后续模型：

- 无拟合参数的 baseline，例如 `last_week`、Moving Average、EWMA。
- 需要拟合和保存模型文件的监督模型，例如 LightGBM Regressor。

本轮不新增 Moving Average、EWMA、LightGBM，不实现评价指标，不进入推荐模块。重点是把训练入口、模型接口、输出目录、metadata 和产物写出契约稳定下来，避免后续新增模型时继续把模型细节写进顶层 CLI。

这是个人学生项目，不需要保留旧 baseline 训练入口。实现时删除 `src/10_train_trend_baseline.py`，新增通用训练入口 `src/10_train_trend_model.py`，README 和计划文档同步使用新命令。

## 设计结论

采用统一 trainer 接口加 registry 的方案。

每个趋势模型都实现一个 `TrendModelTrainer`。顶层 CLI 不直接 import 具体模型函数，也不判断 `if model_name == "last_week"`。CLI 只做以下事情：

1. 解析 `--model`。
2. 从 registry 查找对应 trainer。
3. 读取已切分的 `train`、`valid`、`test` 样本。
4. 构造 `TrendTrainContext`。
5. 调用 `trainer.train(context)`。
6. 校验 `TrendTrainResult`。
7. 统一写出 `predictions.csv`、`params.json`、`metadata.json` 和可选额外 artifact。

模型实现只负责训练或预测逻辑，返回结构化结果，不直接读全局路径、不处理命令行、不写文件。

## 文件组织

新增或调整以下文件：

```text
src/10_train_trend_model.py
src/fashion_trend/models/
    __init__.py
    base.py
    registry.py
    last_week.py
src/fashion_trend/training.py
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `src/10_train_trend_model.py` | 通用趋势模型训练 CLI，解析参数并调用 runner |
| `src/fashion_trend/models/base.py` | 定义 `TrendModelTrainer`、`TrendTrainContext`、`TrendTrainResult`、`TrendArtifact` 等通用接口 |
| `src/fashion_trend/models/registry.py` | 注册并查找可用趋势模型 |
| `src/fashion_trend/models/last_week.py` | `last_week` trainer 与预测公式实现 |
| `src/fashion_trend/training.py` | 通用 runner、metadata 构造、输出路径派生、artifact 写出 |

`src/10_train_trend_baseline.py` 不保留 wrapper。实现时删除旧文件，并同步 README 中的流水线命令。

趋势数据构造、读取、通用校验和 CSV/Parquet/JSON 写出工具已按领域拆到具体模块。模型注册、模型训练和模型专属逻辑不放入趋势数据模块。

## 通用接口

`TrendTrainContext` 表示一次训练运行的上下文：

```python
@dataclass(frozen=True)
class TrendTrainContext:
    model_name: str
    split_frames: Mapping[str, pd.DataFrame]
    input_paths: Mapping[str, Path]
    output_dir: Path
    split_order: tuple[str, ...] = ("train", "valid", "test")
```

`TrendTrainResult` 表示 trainer 返回的结果：

```python
@dataclass(frozen=True)
class TrendTrainResult:
    model_name: str
    model_type: str
    predictions: pd.DataFrame
    params: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)
    artifacts: tuple[TrendArtifact, ...] = ()
```

`TrendArtifact` 表示模型声明的额外产物：

```python
@dataclass(frozen=True)
class TrendArtifact:
    relative_path: str
    kind: str
    payload: pd.DataFrame | dict[str, object] | bytes
```

`TrendModelTrainer` 使用 `typing.Protocol` 表达：

```python
class TrendModelTrainer(Protocol):
    name: str
    model_type: str

    def train(self, context: TrendTrainContext) -> TrendTrainResult:
        raise NotImplementedError
```

第一版 `model_type` 使用稳定字符串：

```text
baseline
supervised
```

当前只注册：

```text
last_week -> LastWeekTrainer
```

## 数据流

训练入口运行流程：

```text
trend_model_samples_train.parquet
trend_model_samples_valid.parquet
trend_model_samples_test.parquet
    -> read_trend_model_split()
    -> TrendTrainContext
    -> registry.get_trend_model_trainer(model_name)
    -> trainer.train(context)
    -> validate_trend_train_result()
    -> write_trend_model_outputs()
    -> outputs/models/<model_name>/
```

`last_week` trainer 的内部流程：

1. 按 `context.split_order` 合并三份 split 样本。
2. 校验 `last_week` 所需输入列。
3. 生成预测表。
4. 调用通用 `validate_trend_model_predictions()` 校验预测表。
5. 返回 `TrendTrainResult`。

`last_week` 继续使用当前公式：

```text
pred_target_growth = growth_lag_1
pred_share_t1 = exp(pred_target_growth) * (share_t + 1e-6) - 1e-6
```

`last_week` 不拟合参数，不保存模型权重，不输出额外 artifact。

## CLI

新的推荐命令：

```sh
uv run python src/10_train_trend_model.py --model last_week
```

旧命令不保留。后续文档和日常使用统一改为 `src/10_train_trend_model.py`。

如果传入未知模型，CLI 返回非零状态码，并输出可用模型名，例如：

```text
不支持的趋势模型: moving_average。可用模型: last_week
```

不支持 `--modle` 拼写。

## 输出契约

所有模型默认写入：

```text
outputs/models/<model_name>/
    predictions.csv
    params.json
    metadata.json
```

`last_week` 当前写入：

```text
outputs/models/last_week/predictions.csv
outputs/models/last_week/params.json
outputs/models/last_week/metadata.json
```

后续监督模型可以声明额外 artifact，例如：

```text
outputs/models/lightgbm/model.pkl
outputs/models/lightgbm/feature_importance.csv
```

额外 artifact 必须由 trainer 通过 `TrendTrainResult.artifacts` 声明，由 runner 统一写出。trainer 不能绕过 runner 直接写文件。

## Metadata 契约

`metadata.json` 的核心字段由 runner 统一生成：

```json
{
  "model_name": "last_week",
  "model_type": "baseline",
  "input_paths": {
    "train": "data/processed/features/trend_model_samples_train.parquet",
    "valid": "data/processed/features/trend_model_samples_valid.parquet",
    "test": "data/processed/features/trend_model_samples_test.parquet"
  },
  "output_dir": "outputs/models/last_week",
  "prediction_path": "outputs/models/last_week/predictions.csv",
  "params_path": "outputs/models/last_week/params.json",
  "rows": 59200,
  "weeks": 100,
  "attributes": 592,
  "splits": {
    "train": {
      "rows": 49728,
      "weeks": 84,
      "attributes": 592,
      "week_min": 4,
      "week_max": 87
    },
    "valid": {
      "rows": 4736,
      "weeks": 8,
      "attributes": 592,
      "week_min": 88,
      "week_max": 95
    },
    "test": {
      "rows": 4736,
      "weeks": 8,
      "attributes": 592,
      "week_min": 96,
      "week_max": 103
    }
  },
  "extra_artifacts": []
}
```

模型可以通过 `TrendTrainResult.metadata` 追加模型专属字段，例如：

```json
{
  "target_column": "target_growth",
  "feature_columns": ["growth_lag_1", "share_ma_4"]
}
```

模型专属 metadata 不允许覆盖 runner 生成的核心字段。发生 key 冲突时 runner 必须失败，避免不同模型输出不兼容的 metadata 形状。

第一版不写运行时间戳，保持测试输出稳定。

## 参数契约

`params.json` 保存模型可复现实验所需的稳定参数。

`last_week` 参数为：

```json
{
  "model_name": "last_week",
  "formula": "pred_target_growth = growth_lag_1",
  "derived_formula": "pred_share_t1 = exp(pred_target_growth) * (share_t + epsilon) - epsilon",
  "epsilon": 0.000001
}
```

后续模型的参数仍由 trainer 返回，但写出由 runner 统一完成。

## 校验与错误处理

runner 在写任何产物前完成所有核心校验：

- registry 中必须存在 `--model` 指定的模型。
- split 文件必须存在且通过 `read_trend_model_split()` 校验。
- `TrendTrainResult.model_name` 必须等于 CLI 指定模型名。
- `TrendTrainResult.model_type` 必须是已知类型。
- `predictions` 必须通过统一预测表列契约校验。
- `params` 必须是可 JSON 序列化的字典。
- `metadata` 不能覆盖 runner 核心字段。
- artifact 路径必须是相对路径，不能包含绝对路径或 `..`。

如果任一步失败，CLI 返回非零状态码，并且不写 `predictions.csv`、`params.json`、`metadata.json`。写文件继续使用临时文件替换策略，避免半写入产物。

## 测试

继续使用 `unittest` 和小型 DataFrame，不依赖真实 H&M 数据。

新增或调整测试覆盖：

- registry 能列出 `last_week`。
- registry 查询未知模型失败，并在错误信息中列出可用模型。
- `LastWeekTrainer.train()` 返回 `TrendTrainResult`。
- `last_week` 预测公式保持 `pred_target_growth = growth_lag_1`。
- 通用 runner 从 split parquet 读取数据，构造 `TrendTrainContext`，调用 registry trainer。
- runner 输出目录由模型名派生为 `outputs/models/<model_name>/`。
- runner 在 metadata 和预测表校验通过前不写任何产物。
- trainer 返回错误 `model_name` 时失败。
- trainer metadata 覆盖核心字段时失败。
- artifact 使用绝对路径或 `..` 路径时失败。
- README 流水线命令使用 `src/10_train_trend_model.py --model last_week`，不再引用旧 baseline 入口。

最小验证命令：

```sh
uv run python -m unittest discover -s tests -v
uv run python -m py_compile src/10_train_trend_model.py src/fashion_trend/models/base.py src/fashion_trend/models/registry.py src/fashion_trend/models/last_week.py src/fashion_trend/training.py
```

如果本地已有真实训练样本，还应运行：

```sh
uv run python src/10_train_trend_model.py --model last_week
```

并直接检查：

- `outputs/models/last_week/predictions.csv` 存在。
- `outputs/models/last_week/params.json` 存在。
- `outputs/models/last_week/metadata.json` 存在。
- `predictions.csv` 行数等于三份 split parquet 行数之和。
- `metadata.json` 中行数、周范围和 split 摘要与预测表一致。
- README 中的训练命令能直接生成上述产物。

## 非目标

- 不新增 Moving Average、EWMA、LightGBM。
- 不实现评价指标、图表或报告。
- 不引入实验配置 YAML。
- 不引入 MLflow、Weights & Biases 或数据库。
- 不改变趋势样本构造逻辑。
- 不提交 `data/` 或 `outputs/` 下生成产物。
- 不把模型细节重新写进顶层 CLI。
- 不保留 `src/10_train_trend_baseline.py` 历史 CLI。
