# LightGBM Stable 参数默认来源设计

## 范围

本轮目标是调整 `lightgbm` 默认训练参数来源，让默认训练命令优先复用已经发布到 stable 目录的参数：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
```

当前 `lightgbm` 已支持 run 级调参、stable promotion、`--params` 参数文件和 `--param` CLI 覆盖。现有实现中，不传 `--params` / `--param` 时会使用源码中的 built-in 默认参数。本设计将默认参数来源改为两种互斥模式：

```text
显式参数模式：built-in 默认参数 < 显式 --params JSON 文件 < CLI --param 覆盖
默认参数模式：outputs/models/lightgbm/params.json；如果缺失才使用 built-in 默认参数
```

只要用户显式传入 `--params` 或 `--param`，就进入显式参数模式，并且不读取 stable 参数文件。没有传入 `--params` 或 `--param` 时，才进入默认参数模式。

这样用户通过 `--promote-run` 发布调参结果后，后续普通 `lightgbm` 训练会默认使用 stable 已发布参数，而不需要再手动复制参数到源码。

本轮不实现自动调参、参数搜索、best run 自动选择、metrics 驱动 promotion 或新依赖；不改变 `predictions.csv`、`metadata.json`、run 目录和 stable 目录的 artifact 契约；不改变 baseline 行为。

## 设计结论

采用 runner 层解析默认参数来源的方案。

`src/10_train_trend_model.py` 继续只在用户显式传入 `--params` 或 `--param` 时构造 `lightgbm_config` 并放入 `trainer_options`。这条路径仍代表自定义调参实验，默认不发布 stable。

`src/fashion_trend/trend/training/runner.py` 的 LightGBM 训练路径在没有显式 `lightgbm_config` 时，根据 `stable_paths["params"]` 解析默认配置：

- 如果 `outputs/models/lightgbm/params.json` 存在，读取其中的 `lightgbm_params` 和 `early_stopping`。
- 如果 stable 参数文件不存在，使用 built-in 默认参数。
- 如果 stable 参数文件存在但不可解析或参数非法，训练失败并给出可定位错误。

stable 参数作为“自动默认来源”，不算自定义实验。因此下面命令仍保持现有默认 promotion 行为：

```sh
uv run python src/10_train_trend_model.py --model lightgbm
```

它会用当前 stable 参数训练一个新的 auto run，并在训练成功后自动发布到 `outputs/models/lightgbm/`。这意味着 stable 目录下的 `predictions.csv`、`params.json`、`metadata.json`、`feature_importance.csv` 和 `model.txt` 会被本次训练结果覆盖。

如果用户希望用默认参数生成 run 但不覆盖 stable，应显式使用：

```sh
uv run python src/10_train_trend_model.py --model lightgbm --no-promote
```

## 参数模式与优先级

参数解析分为两种互斥模式，避免把显式实验参数和 stable 默认参数混在一起。

显式参数模式在用户传入 `--params` 或 `--param` 时启用，优先级固定为：

```text
built-in 默认参数 < 显式 --params JSON 文件 < CLI --param 覆盖
```

其中 `--params` 与 `--param` 保持当前语义：

- `--params` 读取用户指定 JSON 文件。
- `--param key=value` 覆盖显式参数文件或 built-in 默认来源中的单个参数。
- `--param early_stopping.stopping_rounds=<int>` 继续覆盖 early stopping。
- 显式参数路径只支持 `--model lightgbm`，baseline 使用这些参数必须失败。

当用户传入 `--params` 或 `--param` 时，不读取 stable 参数文件。显式参数的语义是“我正在做一次自定义实验”，因此继续默认不 promote，除非用户同时传入 `--promote`。

例如在已经发布 stable 参数后执行：

```sh
uv run python src/10_train_trend_model.py --model lightgbm --param learning_rate=0.03
```

这条命令应以 built-in 默认参数为基底，再应用 `learning_rate=0.03`，而不是读取 stable 参数后再覆盖 learning rate。

默认参数模式在用户没有传入 `--params` 或 `--param` 时启用，优先级固定为：

```text
outputs/models/lightgbm/params.json；如果该文件不存在，则使用 built-in 默认参数
```

默认参数模式不会把 stable 参数与 built-in 参数做宽松合并。stable 参数文件存在时，它必须提供完整核心参数字段；缺失核心字段应失败，而不是补回 built-in 后继续训练。

## Stable 参数文件解析

stable 参数文件复用当前 `params.json` artifact 结构。下一次训练只读取以下训练配置字段；示例中的 `lightgbm_params` 是必须完整保留的核心参数集合：

```json
{
  "lightgbm_params": {
    "objective": "regression_l1",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "min_split_gain": 0.0,
    "random_state": 42,
    "verbosity": -1
  },
  "early_stopping": {
    "stopping_rounds": 30
  }
}
```

其他字段，例如 `model_name`、`model_type`、`numeric_features`、`categorical_features`、`best_iteration`、`objective`、`allowed_objectives` 和 `epsilon`，只属于 artifact metadata，不参与下一次训练配置。

stable 参数文件不是一个新的配置格式。它应复用 `--params` 文件的内部参数校验逻辑，但允许 stable artifact 保留额外 metadata 顶层字段。校验应确保：

- 顶层必须是 JSON object。
- 必须包含 `lightgbm_params`，且必须是 JSON object。
- 必须包含 `early_stopping`，且必须是 JSON object。
- `lightgbm_params` 只能包含允许清单中的 LightGBM 参数。
- `early_stopping` 只能包含 `stopping_rounds`。
- `lightgbm_params` 必须包含所有 built-in LightGBM 参数键，不能只写部分覆盖。
- `early_stopping` 必须包含 `stopping_rounds`。
- 参数类型、取值范围和 `subsample` / `subsample_freq` 组合规则继续沿用现有校验。

## Promotion 语义

promotion 默认判断需要区分“显式自定义参数”和“自动默认参数来源”。

保持现有规则：

- 无 `--run-id`、无显式 `--params` / `--param`、无 `--no-promote` 的默认 `lightgbm` 训练会自动 promote。
- 传入 `--run-id` 的训练默认不 promote。
- 传入显式 `--params` 或 `--param` 的训练默认不 promote。
- `--promote` 强制发布 stable。
- `--no-promote` 强制不发布 stable。
- `--promote-run` 继续只发布已评估 run，不重新训练。

新增规则：

- 自动读取 stable 参数文件不算显式自定义参数。
- 自动读取 built-in 默认参数也不算显式自定义参数。
- 因此 stable 参数存在时，默认 `lightgbm` 训练仍自动 promote。

实现时不能用注入默认 config 后的 `bool(trainer_options)` 判断是否为自定义实验。runner 应显式区分“用户是否传入显式 config”，例如在注入 stable 或 built-in 默认 config 前计算 `promotion_requested`，或引入单独的 `explicit_user_config` 标志。自动默认 config 可以传给 trainer，但不能让默认 promotion 被误关。

这保证“默认训练参数来源变了”，但“默认命令代表当前主模型训练并发布 stable”的行为不变。

## Metadata 记录

训练结果的 metadata 应继续记录 `param_source`，让后续审查能看出参数来源。

built-in cold start 场景：

```json
{
  "param_source": {
    "default": "builtin",
    "params_file": null,
    "overrides": {}
  }
}
```

stable 参数默认场景：

```json
{
  "param_source": {
    "default": "stable",
    "params_file": "outputs/models/lightgbm/params.json",
    "overrides": {}
  }
}
```

显式参数场景继续记录用户参数文件和 CLI 覆盖：

```json
{
  "param_source": {
    "default": "builtin",
    "params_file": "configs/trend/lightgbm/example.json",
    "overrides": {
      "learning_rate": 0.03
    }
  }
}
```

如果显式 `--params` 的内部实现仍以 built-in 为基础再合并文件和 CLI 覆盖，`default` 可以继续保持 `builtin`，避免把显式参数与 stable 参数来源混淆。

## 错误处理

stable 参数文件不存在时，不报错，使用 built-in 默认参数。

stable 参数文件存在但有问题时，必须 fail fast，不能静默回退到 built-in。需要失败的情况包括：

- 文件不是合法 JSON。
- 顶层不是 JSON object。
- 缺少 `lightgbm_params` 或 `early_stopping`。
- `lightgbm_params` 或 `early_stopping` 类型错误。
- `lightgbm_params` 缺少任一核心参数键。
- `early_stopping` 缺少 `stopping_rounds`。
- 包含未知参数名。
- 参数类型或取值范围不符合现有校验。
- `early_stopping.stopping_rounds` 非正整数。

失败信息应包含 `LightGBM` 和 stable 参数文件路径，便于定位是哪一个 artifact 损坏。不要吞掉错误，也不要把损坏的 stable 参数当成“文件不存在”处理。

## 代码边界

首选改动位置：

```text
src/fashion_trend/trend/models/supervised/lightgbm_config.py
src/fashion_trend/trend/training/runner.py
src/10_train_trend_model.py
tests/test_trend_lightgbm.py
tests/test_trend_training.py
README.md
docs/gpt-research/implementation-plan.md
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `lightgbm_config.py` | 提供可复用的参数解析 helper，支持显式参数文件和 stable artifact 参数文件共用校验 |
| `training/runner.py` | 在无显式 config 时决定 stable params 或 built-in params，并保持 promotion 默认语义 |
| `10_train_trend_model.py` | 保持 CLI 显式参数优先级，不负责读取 stable 参数 |
| `tests/test_trend_lightgbm.py` | 覆盖参数解析、stable artifact 读取和错误路径 |
| `tests/test_trend_training.py` | 覆盖 runner promotion 行为和 config 传递 |
| `README.md` | 说明默认参数来源优先级和 `--no-promote` 用法 |
| `implementation-plan.md` | 同步 LightGBM 调参和默认参数语义 |

不要把 stable 参数读取放入 `LightGBMTrendTrainer`。trainer 应继续只消费 `TrendTrainContext.trainer_options` 中已经解析好的 config，避免模型训练器理解 output root、stable 目录或 promotion 规则。

不要让 `lightgbm_config.py` 直接硬编码仓库级 `OUTPUT_MODELS_DIR`。它可以接受一个 `Path` 参数读取参数文件，但默认来源选择应留给 runner。

## 测试计划

新增或更新聚焦测试：

- `resolve_lightgbm_config()` 保持显式 `--params` + `--param` 合并和 CLI 最高优先级。
- stable 参数文件存在时，新的 helper 读取 `lightgbm_params` 和 `early_stopping`。
- stable 参数文件不存在时，返回 built-in 默认参数。
- stable 参数文件存在但 JSON 非法、shape 非法、缺少 `lightgbm_params`、缺少 `early_stopping`、缺少任一核心参数键或参数非法时抛出 `ValueError`。
- 无显式参数且 stable params 存在时，`run_trend_model_training()` 传给 fake LightGBM fit 的 config 来自 stable。
- 无显式参数且 stable params 存在时，默认训练仍自动 promote，stable 目录产物存在。
- 显式 `--param` 仍默认不 promote，并且不读取 stable 参数。
- 显式 `--param` 在无 `--params` 时以 built-in 为基底，不叠加 stable 参数。
- baseline 继续拒绝 `--params` / `--param` / `--run-id` 等 LightGBM-only 参数。

建议验证命令：

```sh
uv run pytest tests/test_trend_lightgbm.py tests/test_trend_training.py
uv run python -m compileall -q src
```

如果实现只触及参数解析和 runner，可先运行上述聚焦测试；最终收尾时再根据 diff 决定是否补跑完整 `uv run pytest`。

## 文档更新

README 中当前“内置默认 LightGBM 参数来自已选中的调参 run”的表述需要替换为：

- 默认训练优先读取 `outputs/models/lightgbm/params.json`。
- 如果 stable 参数文件缺失，才使用源码内 built-in 默认参数。
- 显式 `--params` / `--param` 进入显式参数模式，不读取 stable 参数。
- 默认读取 stable 参数仍会自动 promote。
- 若只想复用默认参数生成 run，不覆盖 stable，应使用 `--no-promote`。

`docs/gpt-research/implementation-plan.md` 中如有 LightGBM 默认参数或调参发布语义，也做最小同步，避免文档继续暗示 built-in 参数就是当前最佳参数来源。

## 验收标准

实现完成后应满足：

- `uv run python src/10_train_trend_model.py --model lightgbm` 在 stable params 存在时使用 stable 参数。
- 同一命令在 stable params 缺失时仍能使用 built-in 默认参数完成训练。
- stable params 存在但损坏时命令失败，而不是静默 fallback。
- 默认 `lightgbm` 训练仍自动发布 stable。
- 显式 `--params` / `--param` 仍绕过 stable params，并继续默认不 promote。
- README 和 implementation plan 与实际行为一致。
