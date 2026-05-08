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

本设计解决覆盖问题，同时保持现有稳定命令可继续使用。实现后，每轮 LightGBM 训练会生成独立 `run_id` 目录；稳定目录只在 promotion 规则允许时更新，避免调参候选把“最后一次尝试”误当成当前主结果。

本轮不实现自动网格搜索、贝叶斯调参、MLflow、数据库、外部服务或新依赖；不改变 `predictions.csv` 列契约；不把 `run_id` 写入预测表的 `model_name`。系统调参后续在本设计之上继续扩展 batch runner。

## 设计结论

采用“稳定目录 + 历史 run 目录”的方案。

无参数训练的默认行为：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
```

会自动生成安全 `run_id`，写入：

```text
outputs/models/lightgbm/runs/<run_id>/
```

并默认发布到稳定目录：

```text
outputs/models/lightgbm/
```

实验 run 行为：

```sh
uv run python src/10_train_trend_model.py --model lightgbm --run-id depth6-lr005
uv run python src/11_eval_trend_model.py --model lightgbm --run-id depth6-lr005
```

会让训练和评价都绑定到同一个 run 目录。带 `--run-id`、`--params` 或 `--param` 的训练默认不发布到稳定目录；只有显式 `--promote` 才会更新 `outputs/models/lightgbm/`。也可以对无参数训练传 `--no-promote`，只保留 run 目录。

`run_id` 必须通过安全路径片段校验，不能为空，不能是 `.`、`..`，不能包含 `/`，也不能逃逸模型输出根目录。自动生成格式固定为 `YYYYMMDD-HHMMSS-<8hex>`，时间使用本地时区；如果自动生成的目录已存在，则重新生成随机后缀，最多重试 10 次。

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
    index.jsonl
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
    evaluations.jsonl
    <run_id>/
      trend_metrics.json
```

稳定目录代表“当前主结果”。`runs/<run_id>/` 代表不可覆盖的历史实验实例。稳定目录只有在 promotion 成功后更新；稳定目录和 run 目录的 `metadata.json` 都记录同一个 `run_id`，这样不带 `--run-id` 的下游命令也能回溯到具体实验。

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
--promote
--no-promote
--promote-run <safe_id>
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

发布已评估 run：

```sh
uv run python src/10_train_trend_model.py --model lightgbm --promote-run depth6-lr005
```

`--promote-run` 是训练入口的互斥模式。它不重新训练，只校验 `outputs/models/lightgbm/runs/<run_id>/` 下的模型产物完整性，要求对应 `outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json` 已存在，然后把该 run 的模型产物发布到稳定目录、生成 stable metadata，并同步发布 stable metrics。

promotion 规则：

- 未传 `--run-id`、`--params`、`--param` 的无参数训练默认 `--promote`。
- 传入 `--run-id`、`--params` 或 `--param` 的实验训练默认 `--no-promote`。
- `--promote` 强制在 run 成功后发布稳定目录。
- `--no-promote` 强制只保留 run 目录。
- `--promote-run`、`--promote` 和 `--no-promote` 不能互相组合。
- `--promote-run` 不能与 `--run-id`、`--params` 或 `--param` 组合。
- `--run-id`、`--params`、`--param`、`--promote`、`--no-promote`、`--promote-run` 只允许 `--model lightgbm` 使用；baseline 传入这些参数必须直接失败，不能静默忽略。
- 训练命令内的 `--promote` 只发布模型产物，不自动运行评价，也不发布 metrics。
- `--promote-run` 发布已评估 run，必须同步发布模型产物和 stable metrics，避免 stable model 与 stable metrics 指向不同 run。
- 正式主模型的推荐流程是先 `train --no-promote`，再 `eval --run-id`，确认 valid/test 摘要后通过 `--promote-run` 发布 stable。

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

LightGBM 参数不再只能通过修改源码中的常量调整。内置默认参数仍作为默认配置层保留，保证不传任何配置时仍有稳定入口。

合并顺序固定为：

```text
内置默认参数 < 参数文件 < CLI --param 覆盖
```

`--params` 文件结构固定为：

```json
{
  "lightgbm_params": {
    "learning_rate": 0.03,
    "num_leaves": 63
  },
  "early_stopping": {
    "stopping_rounds": 50
  }
}
```

约束：

- `--params` 只接受 JSON object，并且只能包含 `lightgbm_params` 和 `early_stopping` 两个顶层 key。
- `lightgbm_params` 和 `early_stopping` 是可选顶层 key；如果出现，则必须是 JSON object。因此只覆盖 `lightgbm_params` 或只覆盖 `early_stopping` 的参数文件都合法。
- `--param key=value` 默认写入 `lightgbm_params.<key>`。
- `--param early_stopping.stopping_rounds=50` 写入 `early_stopping.stopping_rounds`。
- 除 `early_stopping.stopping_rounds` 外，其他 dotted key 一律拒绝；例如不支持 `lightgbm_params.learning_rate=0.03`。
- `--param key=value` 按 JSON literal 优先解析，例如 `63` 是整数、`0.03` 是浮点、`true` 是布尔；解析失败时保留字符串。
- 参数 key 必须属于允许清单。
- 非法参数类型或范围必须在调用 LightGBM 前失败，不能把错误交给 LightGBM native 层。

首批允许的 LightGBM 参数：

```text
objective
n_estimators
learning_rate
num_leaves
max_depth
min_child_samples
subsample
subsample_freq
colsample_bytree
reg_alpha
reg_lambda
min_split_gain
random_state
verbosity
```

参数 schema：

| 参数 | 类型与范围 |
| --- | --- |
| `objective` | 字符串，只允许 `regression` 或 `regression_l1` |
| `n_estimators` | 正整数 |
| `learning_rate` | 大于 0 的有限数值 |
| `num_leaves` | 正整数 |
| `max_depth` | `-1` 或正整数 |
| `min_child_samples` | 正整数 |
| `subsample` | `(0, 1]` 的有限数值 |
| `subsample_freq` | 非负整数；当 `subsample < 1` 时必须为正整数，默认使用 `1` 让行采样实际生效 |
| `colsample_bytree` | `(0, 1]` 的有限数值 |
| `reg_alpha` | 非负有限数值 |
| `reg_lambda` | 非负有限数值 |
| `min_split_gain` | 非负有限数值 |
| `random_state` | 整数 |
| `verbosity` | 整数 |
| `early_stopping.stopping_rounds` | 正整数 |

当前内置默认参数需要有意补入 `subsample_freq: 1`，否则 `subsample: 0.8` 在 LightGBM sklearn API 下不会启用逐轮行采样。这是本轮对 LightGBM 首版默认参数的修正，不属于“完全保持旧结果”。默认训练结果以本轮重新跑出的结果为准，验收必须重新运行 LightGBM 训练、评价和三类 baseline 对比，确认新的默认结果与文档摘要一致。

最终训练参数写入 `params.json`。参数来源写入 `metadata.json`，包括内置默认、参数文件路径和 CLI 覆盖键值。

## Metadata 与 Run Manifest

每个 run 的 `metadata.json` 是该 run 的事实记录，至少包含：

```json
{
  "model_name": "lightgbm",
  "run_id": "20260508-153012-a1b2c3",
  "run_dir": "outputs/models/lightgbm/runs/20260508-153012-a1b2c3",
  "stable_output_dir": "outputs/models/lightgbm",
  "promotion_requested": true,
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

必须明确生成两份 metadata：

- run metadata：写入 `outputs/models/lightgbm/runs/<run_id>/metadata.json`，路径字段指向 run 目录。
- stable metadata：只在 promotion 成功时写入 `outputs/models/lightgbm/metadata.json`，路径字段指向稳定目录。

两份 metadata 共享 `run_id`、模型参数、参数来源、诊断字段、`best_iteration`、`best_score`、特征清单和 artifact 摘要；路径字段必须分别按所在目录生成，不能简单复制。现有 `build_trend_train_metadata()` 的 `output_dir`、`prediction_path`、`params_path` 等核心字段仍由 runner 根据目标目录生成，训练器不能覆盖。run metadata 只记录 promotion 请求上下文，例如 `promotion_requested`，不承诺记录最终 promotion 状态。

稳定目录的 `metadata.json` 记录同一个 `run_id`。这样稳定目录始终是可被现有命令消费的当前主结果，而 run 目录保留不可覆盖的历史实验证据。

训练成功后更新轻量索引：

```text
outputs/models/lightgbm/runs/index.jsonl
```

`index.jsonl` 是可重建的 run summary 索引，不是事实来源。文件中每行代表一个 `run_id` 的当前摘要：

```json
{
  "run_id": "depth6-lr005",
  "created_at": "2026-05-08T15:30:12+08:00",
  "run_dir": "outputs/models/lightgbm/runs/depth6-lr005",
  "promotion_status": "not_requested",
  "params_path": "outputs/models/lightgbm/runs/depth6-lr005/params.json",
  "metadata_path": "outputs/models/lightgbm/runs/depth6-lr005/metadata.json"
}
```

`promotion_status` 只允许 `not_requested`、`succeeded`、`failed`。训练、训练内 promotion 和 `--promote-run` 都通过原子重写 index 来 upsert 对应 `run_id` 的摘要。如果索引缺失，可以扫描 `runs/*/metadata.json`、稳定目录 metadata 和 metrics 文件恢复主要状态。

## Promotion 事务边界

run 目录是事实来源，stable 目录是 promotion 结果。训练输出顺序固定为：

1. 在内存中完成训练、预测、参数、metadata 和 artifact 载荷构造。
2. 先写入并发布 `outputs/models/lightgbm/runs/<run_id>/`。
3. run 发布成功后，按 promotion 规则决定是否更新 `outputs/models/lightgbm/`。
4. promotion 成功时，stable metadata 指向稳定目录，并把 `index.jsonl` 中该 run 的 `promotion_status` 更新为 `succeeded`。
5. promotion 失败时，run 目录保留为事实来源，并尽力把 `index.jsonl` 中该 run 的 `promotion_status` 更新为 `failed` 且记录错误摘要；CLI 返回失败，不能把失败的 stable 更新伪装成成功训练。

run 写出失败时，不得更新 stable。stable 写出失败时，不得删除已经成功发布的 run。run metadata 发布后不再要求为了 promotion 结果回写；如果 index 更新也因磁盘或权限问题失败，CLI 仍返回失败，并在日志中输出 run 目录、stable 目录和 promotion 错误。本轮实现包含训练命令内 promotion 和 `--promote-run` 两种发布路径。

`--promote-run <run_id>` 复用同一套 promotion 发布逻辑，但输入来自已有 run 目录。它必须先校验标准模型产物、run metadata、`params.json`、对应 run metrics 和 run metrics payload 都存在且路径安全，再发布 stable。run metrics payload 必须满足 `model_name == "lightgbm"`、`run_id == <run_id>`，并且 `prediction_path` 指向 `outputs/models/lightgbm/runs/<run_id>/predictions.csv`；否则拒绝发布。发布失败时不得修改 run 目录；发布成功后稳定目录 metadata 指向 stable 路径，并记录来源 `run_id`。

`--promote-run` 还必须同步发布 stable metrics。发布方式是读取：

```text
outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json
```

然后生成 stable metrics payload，至少把 `prediction_path` 和 `output_path` 改为稳定路径：

```text
outputs/models/lightgbm/predictions.csv
outputs/metrics/lightgbm/trend_metrics.json
```

`model_name` 和 `run_id` 保持不变。写出目标为：

```text
outputs/metrics/lightgbm/trend_metrics.json
```

模型产物、stable metadata 和 stable metrics 必须作为一次 promotion 操作处理。由于它们跨 `outputs/models/lightgbm/` 和 `outputs/metrics/lightgbm/` 两个目录，promotion 发布 stable 时必须对 stable model 产物、stable metadata 和 stable metrics 做统一 staging/backup/rollback；任一 stable 文件替换失败，都要恢复已替换的 stable 文件，保留 run 目录不动，并返回失败。

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

评价成功后追加轻量评价索引：

```text
outputs/metrics/lightgbm/runs/evaluations.jsonl
```

每行包含 `run_id`、metrics 路径、评价时间，以及拆分后的指标摘要：

```json
{
  "run_id": "depth6-lr005",
  "metrics_path": "outputs/metrics/lightgbm/runs/depth6-lr005/trend_metrics.json",
  "selection_metrics": {
    "split": "valid",
    "ndcg_at_10": 0.42,
    "spearman": 0.31,
    "mae": 0.12,
    "rmse": 0.18
  },
  "report_metrics": {
    "valid": {},
    "test": {}
  }
}
```

`selection_metrics` 只允许来自 valid split，后续自动排序只能读取这个对象。`report_metrics` 可以包含 valid/test 摘要，只用于最终报告和人工回溯。该索引用于后续汇总，不替代单个 run 的 `trend_metrics.json`。

## 调参选择与 Test 防泄漏

系统调参必须只用 valid 指标排序，不得用 test 指标选择参数。后续 tuning runner 的选择规则固定为：

1. valid `NDCG@10` 最大。
2. valid `NDCG@10` 差距很小时，再比较 valid `Spearman`。
3. MAE 和 RMSE 作为 guardrail，不能比最强 baseline 明显退化。
4. test 只用于最终选中 run 的一次性报告，不参与候选筛选、top-N 排序或稳定性复跑选择。

本轮不实现 tuning runner，但文档和 metadata 必须把这个边界写清楚。`evaluations.jsonl` 可以在 `report_metrics` 保存 test 摘要用于最终报告回溯；任何自动排序字段都必须来自 `selection_metrics`。

## 实现边界

本轮只为 `lightgbm` 建立 run 化训练、评价和已评估 run 发布能力。baseline 仍保持当前稳定输出行为，不新增 run 目录。baseline 遇到 `--run-id`、`--params`、`--param`、`--promote`、`--no-promote` 或 `--promote-run` 必须直接失败。

不新增编号脚本。`src/10_train_trend_model.py` 继续是训练入口，`src/11_eval_trend_model.py` 继续是评价入口。模型细节仍留在 `src/fashion_trend/trend/models/supervised/lightgbm.py`；通用路径、metadata、写盘和评价寻址能力放在 `src/fashion_trend/trend/training/` 与 `src/fashion_trend/trend/evaluation/`。

不修改 `predictions.csv` 的列契约。`model_name` 仍然是 `lightgbm`，不是 `lightgbm_<run_id>`。run 信息只写入 metadata、params 来源记录和 metrics payload。

## 测试验收

需要覆盖以下行为：

- `derive_trend_model_output_paths("lightgbm", run_id="abc")` 派生到 `outputs/models/lightgbm/runs/abc/`。
- 非安全 `run_id` 被拒绝，例如空字符串、`.`、`../x`、`nested/x`。
- 自动 `run_id` 格式为 `YYYYMMDD-HHMMSS-<8hex>`，使用本地时区；自动生成冲突时最多重试 10 次。
- 手动指定的 `run_id` 已存在时直接失败。
- 无参数 LightGBM 训练默认 promote；带 `--run-id`、`--params` 或 `--param` 的实验训练默认不 promote。
- `--promote-run`、`--promote` 和 `--no-promote` 互斥。
- `--promote-run` 不能与 `--run-id`、`--params` 或 `--param` 组合。
- baseline 传入 run、参数或 promote 相关参数时直接失败。
- LightGBM 不传参数时使用内置默认配置；本轮会有意新增 `subsample_freq: 1`，默认结果需要重新验收。
- `--params` JSON 和 `--param` 覆盖按“内置默认 < 文件 < CLI”合并。
- `--params` 只接受固定 shape：`lightgbm_params` 和 `early_stopping`。
- `--param early_stopping.stopping_rounds=50` 能覆盖 early stopping。
- 除 `early_stopping.stopping_rounds` 外，其他 dotted key 一律失败。
- 未允许参数、非法 `objective`、非法 `early_stopping`、非法类型或非法范围会失败并给出可定位错误。
- 训练成功后先写 run 目录；promotion 成功时再写稳定目录。
- run metadata 和 stable metadata 分别按各自输出目录生成路径字段，并记录同一 `run_id`。
- `index.jsonl` 是一行一个 run 的 summary index，`promotion_status` 只能是 `not_requested`、`succeeded` 或 `failed`。
- promotion 失败时保留 run，通过 `runs/index.jsonl` 更新失败状态；如果 index 也无法写入，则日志必须输出失败摘要，训练命令返回失败。
- `--promote-run <run_id>` 不重新训练，要求 run 产物和 run metrics 都存在，发布成功后稳定目录 metadata 指向 stable 路径并记录来源 run。
- `--promote-run <run_id>` 同步发布 stable metrics，把 run metrics payload 的 `prediction_path` 和 `output_path` 改为稳定路径，且保留 `run_id`。
- 显式 run 评价读取 `runs/<run_id>/predictions.csv`，写 `outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json`。
- run 评价写入 `evaluations.jsonl`，其中 `selection_metrics` 只来自 valid split，`report_metrics` 可包含 test 摘要。
- `--promote` 不自动运行评价或发布 metrics。
- 默认评价仍读取稳定目录，现有 baseline 的评价测试不受影响。
- README 和实施计划同步说明 run 目录、参数配置、`--promote-run` 工作流、promotion 风险、评价命令和重新验收要求。

计划验证命令：

```sh
uv run pytest tests/test_trend_training.py tests/test_trend_lightgbm.py tests/test_trend_evaluation.py
uv run pytest
uv run python src/10_train_trend_model.py --model lightgbm --run-id smoke-lightgbm --no-promote
uv run python src/11_eval_trend_model.py --model lightgbm --run-id smoke-lightgbm
uv run python src/10_train_trend_model.py --model lightgbm --promote-run smoke-lightgbm
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
