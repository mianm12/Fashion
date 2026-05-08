# LightGBM 调参 Run 产物设计

## 范围

本轮目标是为 `lightgbm` 调参建立可保留、可回溯、可评价的单轮训练产物契约。当前 `lightgbm` 已通过统一入口运行：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm
```

但训练和评价路径只由 `model_name` 决定，因此每次训练都会覆盖：

```text
outputs/models/lightgbm/
outputs/metrics/lightgbm/
```

本设计解决覆盖问题，同时保持现有稳定命令可继续使用。实现后，每轮 LightGBM 训练会生成独立 `run_id` 目录，并同步发布一份当前主结果到稳定目录。

本轮不实现自动网格搜索、贝叶斯调参、MLflow、数据库、外部服务或新依赖；不改变 `predictions.csv` 列契约；不把 `run_id` 写入预测表的 `model_name`。系统调参后续在本设计之上继续扩展 batch runner。

## 设计结论

采用“稳定目录 + 历史 run 目录”的方案。

训练默认行为：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
```

会自动生成安全 `run_id`，写入：

```text
outputs/models/lightgbm/runs/<run_id>/
```

并把同一轮产物同步发布到：

```text
outputs/models/lightgbm/
```

显式 run 行为：

```sh
uv run python src/10_train_trend_model.py --model lightgbm --run-id depth6-lr005
uv run python src/11_eval_trend_model.py --model lightgbm --run-id depth6-lr005
```

会让训练和评价都绑定到同一个 run 目录。`run_id` 必须通过安全路径片段校验，不能为空，不能是 `.`、`..`，不能包含 `/`，也不能逃逸模型输出根目录。

如果用户手动指定的 `run_id` 已存在，训练默认失败，避免覆盖历史实验。稳定目录仍然允许被最新训练更新。

## 目录契约

LightGBM 模型产物目录：

```text
outputs/models/lightgbm/
  predictions.csv
  params.json
  metadata.json
  feature_importance.csv
  model.txt
  runs/
    <run_id>/
      predictions.csv
      params.json
      metadata.json
      feature_importance.csv
      model.txt
```

LightGBM 评价产物目录：

```text
outputs/metrics/lightgbm/
  trend_metrics.json
  runs/
    <run_id>/
      trend_metrics.json
```

稳定目录代表“当前主结果”。`runs/<run_id>/` 代表不可覆盖的历史实验实例。稳定目录和 run 目录的 `metadata.json` 都记录同一个 `run_id`，这样不带 `--run-id` 的下游命令也能回溯到具体实验。

baseline 暂不强制 run 化，继续使用当前稳定目录：

```text
outputs/models/<baseline>/
outputs/metrics/<baseline>/
```

## CLI 契约

训练入口新增可选参数：

```text
--run-id <safe_id>
--params <json_path>
--param <key=value>
```

默认训练：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
```

显式训练：

```sh
uv run python src/10_train_trend_model.py --model lightgbm --run-id depth6-lr005
```

参数文件训练：

```sh
uv run python src/10_train_trend_model.py \
  --model lightgbm \
  --params configs/trend/lightgbm/depth6_lr005.json
```

临时参数覆盖：

```sh
uv run python src/10_train_trend_model.py \
  --model lightgbm \
  --param learning_rate=0.03 \
  --param num_leaves=63
```

评价入口新增可选参数：

```text
--run-id <safe_id>
```

默认评价：

```sh
uv run python src/11_eval_trend_model.py --model lightgbm
```

读取：

```text
outputs/models/lightgbm/predictions.csv
```

写入：

```text
outputs/metrics/lightgbm/trend_metrics.json
```

显式 run 评价：

```sh
uv run python src/11_eval_trend_model.py --model lightgbm --run-id depth6-lr005
```

读取：

```text
outputs/models/lightgbm/runs/depth6-lr005/predictions.csv
```

写入：

```text
outputs/metrics/lightgbm/runs/depth6-lr005/trend_metrics.json
```

## 参数合并规则

LightGBM 参数不再只能通过修改源码中的常量调整。内置默认参数仍保留，保证不传任何配置时行为稳定。

合并顺序固定为：

```text
内置默认参数 < 参数文件 < CLI --param 覆盖
```

约束：

- `--params` 只接受 JSON object。
- `--param key=value` 按 JSON literal 优先解析，例如 `63` 是整数、`0.03` 是浮点、`true` 是布尔；解析失败时保留字符串。
- 参数 key 必须属于允许清单。
- `objective` 仍限制为 `regression` 或 `regression_l1`。
- early stopping 使用独立配置段 `early_stopping`，首批只允许 `stopping_rounds`。

首批允许的 LightGBM 参数：

```text
objective
n_estimators
learning_rate
num_leaves
max_depth
min_child_samples
subsample
colsample_bytree
random_state
verbosity
```

最终训练参数写入 `params.json`。参数来源写入 `metadata.json`，包括内置默认、参数文件路径和 CLI 覆盖键值。

## Metadata 与 Run Manifest

每个 run 的 `metadata.json` 是该 run 的事实记录，至少包含：

```json
{
  "model_name": "lightgbm",
  "run_id": "20260508-153012-a1b2c3",
  "run_dir": "outputs/models/lightgbm/runs/20260508-153012-a1b2c3",
  "stable_output_dir": "outputs/models/lightgbm",
  "promoted_to_stable": true,
  "prediction_path": "outputs/models/lightgbm/runs/20260508-153012-a1b2c3/predictions.csv",
  "params_path": "outputs/models/lightgbm/runs/20260508-153012-a1b2c3/params.json",
  "param_source": {
    "default": "builtin",
    "params_file": "configs/trend/lightgbm/depth6_lr005.json",
    "overrides": {
      "learning_rate": 0.03,
      "num_leaves": 63
    }
  }
}
```

稳定目录的 `metadata.json` 记录同一个 `run_id`，并把路径字段指向稳定目录中的文件。这样稳定目录始终是可被现有命令消费的当前主结果，而 run 目录保留不可覆盖的历史实验证据。

训练成功后追加轻量索引：

```text
outputs/models/lightgbm/runs/index.jsonl
```

每行包含 `run_id`、时间、参数摘要、产物路径和是否发布到稳定目录。后续系统调参可以直接读取该索引汇总结果；如果索引缺失，也可以扫描 `runs/*/metadata.json` 恢复。

## 评价关联

`trend_metrics.json` 继续使用当前评价指标结构，并增加 run 关联字段：

```json
{
  "model_name": "lightgbm",
  "run_id": "depth6-lr005",
  "prediction_path": "outputs/models/lightgbm/runs/depth6-lr005/predictions.csv",
  "output_path": "outputs/metrics/lightgbm/runs/depth6-lr005/trend_metrics.json"
}
```

评价 runner 从预测目录旁的 `metadata.json` 读取 `run_id`。如果 metadata 不存在，或 metadata 没有 `run_id`，评价不失败，`run_id` 记录为 `null`。这保证历史 baseline 和旧产物仍能被评价。

显式 `--run-id` 评价必须读取对应 run 目录，不能退回稳定目录；如果 run 目录不存在，应直接失败并给出包含 `run_id` 和预测路径的错误信息。

## 实现边界

本轮只为 `lightgbm` 建立 run 化训练和评价能力。baseline 仍保持当前稳定输出行为，不新增 run 目录。

不新增编号脚本。`src/10_train_trend_model.py` 继续是训练入口，`src/11_eval_trend_model.py` 继续是评价入口。模型细节仍留在 `src/fashion_trend/trend/models/supervised/lightgbm.py`；通用路径、metadata、写盘和评价寻址能力放在 `src/fashion_trend/trend/training/` 与 `src/fashion_trend/trend/evaluation/`。

不修改 `predictions.csv` 的列契约。`model_name` 仍然是 `lightgbm`，不是 `lightgbm_<run_id>`。run 信息只写入 metadata、params 来源记录和 metrics payload。

## 测试验收

需要覆盖以下行为：

- `derive_trend_model_output_paths("lightgbm", run_id="abc")` 派生到 `outputs/models/lightgbm/runs/abc/`。
- 非安全 `run_id` 被拒绝，例如空字符串、`.`、`../x`、`nested/x`。
- 自动 `run_id` 格式稳定，且不会覆盖已有 run 目录。
- LightGBM 不传参数时仍使用当前默认参数。
- `--params` JSON 和 `--param` 覆盖按“内置默认 < 文件 < CLI”合并。
- 未允许参数、非法 `objective`、非法 `early_stopping` 会失败并给出可定位错误。
- 训练成功后同时写 run 目录和稳定目录，两个目录的 `metadata.json` 都记录同一 `run_id`。
- 显式 run 评价读取 `runs/<run_id>/predictions.csv`，写 `outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json`。
- 默认评价仍读取稳定目录，现有 baseline 的评价测试不受影响。
- README 和实施计划同步说明 run 目录、参数配置和评价命令。

计划验证命令：

```sh
uv run pytest tests/test_trend_training.py tests/test_trend_lightgbm.py tests/test_trend_evaluation.py
uv run pytest
uv run python src/10_train_trend_model.py --model lightgbm --run-id smoke-lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm --run-id smoke-lightgbm
```

真实 smoke 如果已有同名 run，应改用新的安全 id，不能覆盖已有历史 run。

## 不做范围

本轮不做以下事项：

- 自动网格搜索或系统调参 runner。
- MLflow、数据库、远程 artifact store 或新依赖。
- baseline run 化。
- 推荐模块产物消费逻辑。
- 上游趋势样本新增特征。
- `predictions.csv` schema 变更。
- 将 `run_id` 编码进 `model_name` 或 registry。

这些事项需要在单轮 run 产物契约稳定后单独设计。
