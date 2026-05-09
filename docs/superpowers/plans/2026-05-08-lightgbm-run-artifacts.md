# LightGBM Run Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `lightgbm` 增加可回溯 run 产物、参数覆盖、run 评价、已评估 run 发布到 stable 的闭环。

**Architecture:** 保留现有 stable 目录 `outputs/models/lightgbm/` 与 `outputs/metrics/lightgbm/`，新增 run 目录和轻量索引。通用 runner 仍负责 split 读取、校验、metadata 和写盘；LightGBM 专属配置、run id、promotion、evaluation index 放到小模块中，避免把 `outputs.py`、`runner.py` 继续膨胀。

**Tech Stack:** Python 3.10-3.12、pandas、LightGBM `LGBMRegressor`、pytest、现有 `fashion_trend.trend.training` / `evaluation` / `models.supervised.lightgbm` 契约。

---

## File Structure

- Modify: `src/fashion_trend/trend/models/base.py`
  - 给 `TrendTrainContext` 增加 `trainer_options`，让 runner 能把 LightGBM 参数配置传给具体 trainer，baseline 保持默认空配置。
- Create: `src/fashion_trend/trend/models/supervised/lightgbm_config.py`
  - 负责默认参数、允许参数清单、`--params` JSON shape、`--param` 解析、类型与范围校验、参数来源 metadata。
- Modify: `src/fashion_trend/trend/models/supervised/lightgbm.py`
  - 改为从 `lightgbm_config.py` 读取默认配置；训练时使用 `TrendTrainContext.trainer_options["lightgbm_config"]` 覆盖默认参数；默认补入 `subsample_freq: 1`。
- Modify: `src/fashion_trend/trend/training/outputs.py`
  - 扩展 `derive_trend_model_output_paths(..., run_id=None)`；扩展 `build_trend_train_metadata(..., run_id=None, run_dir=None, stable_output_dir=None, promotion_requested=None)`。
- Modify: `src/fashion_trend/foundation/io.py`
  - 增加公共 `write_text_atomic()`，供 training 和 evaluation 写 JSONL 索引，避免 evaluation 依赖 training 私有 helper。
- Create: `src/fashion_trend/trend/training/run_artifacts.py`
  - 负责 LightGBM run id 校验/生成、run index upsert、run 目录完整性校验、stable promotion、跨 models/metrics 的 staging/backup/rollback。
- Modify: `src/fashion_trend/trend/training/runner.py`
  - 为 LightGBM 训练增加 run 目录写入、默认 promotion 判定、训练内 promotion、索引更新；baseline 传入 run/参数/promotion 选项时失败。
- Modify: `src/fashion_trend/trend/training/__init__.py`
  - 导出新增 runner 类型和 run artifact helper 中需要被测试直接导入的函数。
- Modify: `src/fashion_trend/trend/evaluation/payloads.py`
  - 扩展 `derive_trend_metric_output_paths(..., run_id=None)`；`build_trend_metrics_payload(..., run_id=None)` 写入 `run_id`。
- Create: `src/fashion_trend/trend/evaluation/run_artifacts.py`
  - 负责从模型 metadata 读取 `run_id`、构造 `selection_metrics` / `report_metrics`、upsert `evaluations.jsonl`、校验 run metrics payload。
- Modify: `src/fashion_trend/trend/evaluation/runner.py`
  - 支持 `run_id`，显式 run 评价读写 run 目录并更新 evaluation index；默认评价保持 stable 行为。
- Modify: `src/fashion_trend/trend/evaluation/__init__.py`
  - 导出 run evaluation 相关 helper。
- Modify: `src/10_train_trend_model.py`
  - 增加 `--run-id`、`--params`、`--param`、`--promote`、`--no-promote`、`--promote-run`；实现 baseline 参数拒绝和 promote-run 模式。
- Modify: `src/11_eval_trend_model.py`
  - 增加 `--run-id`；baseline 传入时按 CLI 用法错误失败。
- Modify: `tests/test_trend_training.py`
  - 覆盖路径派生、run id、index、metadata、training runner、promotion 和 CLI 训练行为。
- Modify: `tests/test_trend_lightgbm.py`
  - 覆盖参数 schema、默认 `subsample_freq: 1`、配置合并、trainer 使用传入配置。
- Modify: `tests/test_trend_evaluation.py`
  - 覆盖 run 评价、`run_id` payload、`evaluations.jsonl`、promote-run 对 run metrics 的一致性校验。
- Modify: `README.md`
  - 增加 LightGBM run 目录、参数文件、`--promote-run` 推荐流程、test 防泄漏边界和重新验收说明。
- Modify: `docs/gpt-research/implementation-plan.md`
  - 同步 LightGBM run artifact 当前能力边界和正式调参前提。

Each task ends with a commit command. Execute commit commands only after the user explicitly authorizes commits for the implementation stage.

---

### Task 1: Run Path And Run Id Primitives

**Files:**
- Modify: `tests/test_trend_training.py`
- Modify: `src/fashion_trend/trend/training/outputs.py`
- Create: `src/fashion_trend/trend/training/run_artifacts.py`
- Modify: `src/fashion_trend/trend/training/__init__.py`

- [ ] **Step 1: Write failing path and run id tests**

Append these tests to `tests/test_trend_training.py` inside `TestTrendTraining`:

```python
    def test_derive_trend_model_output_paths_uses_lightgbm_run_id(self) -> None:
        paths = derive_trend_model_output_paths(
            "lightgbm",
            Path("outputs/models"),
            run_id="depth6-lr005",
        )

        assert paths["output_dir"] == Path("outputs/models/lightgbm/runs/depth6-lr005")
        assert paths["stable_output_dir"] == Path("outputs/models/lightgbm")
        assert paths["run_root"] == Path("outputs/models/lightgbm/runs")
        assert paths["index"] == Path("outputs/models/lightgbm/runs/index.jsonl")
        assert paths["predictions"] == Path(
            "outputs/models/lightgbm/runs/depth6-lr005/predictions.csv"
        )
        assert paths["params"] == Path(
            "outputs/models/lightgbm/runs/depth6-lr005/params.json"
        )
        assert paths["metadata"] == Path(
            "outputs/models/lightgbm/runs/depth6-lr005/metadata.json"
        )

    def test_derive_trend_model_output_paths_rejects_run_id_for_baseline(self) -> None:
        with pytest.raises(ValueError, match="lightgbm|run_id"):
            derive_trend_model_output_paths(
                "last_week",
                Path("outputs/models"),
                run_id="baseline-run",
            )

    @pytest.mark.parametrize("run_id", ["", ".", "..", "../x", "nested/x"])
    def test_validate_lightgbm_run_id_rejects_unsafe_values(self, run_id: str) -> None:
        from fashion_trend.trend.training.run_artifacts import validate_lightgbm_run_id

        with pytest.raises(ValueError, match="run_id"):
            validate_lightgbm_run_id(run_id)

    @pytest.mark.parametrize("run_id", ["index.jsonl", "evaluations.jsonl"])
    def test_validate_lightgbm_run_id_rejects_reserved_names(
        self,
        run_id: str,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import validate_lightgbm_run_id

        with pytest.raises(ValueError, match="保留|run_id"):
            validate_lightgbm_run_id(run_id)

    def test_generate_lightgbm_run_id_uses_local_timestamp_and_hex_suffix(self) -> None:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from fashion_trend.trend.training.run_artifacts import generate_lightgbm_run_id

        run_id = generate_lightgbm_run_id(
            run_root=Path("outputs/models/lightgbm/runs"),
            now_factory=lambda: datetime(2026, 5, 8, 15, 30, 12, tzinfo=ZoneInfo("Asia/Shanghai")),
            token_factory=lambda: "a1b2c3d4",
        )

        assert run_id == "20260508-153012-a1b2c3d4"

    def test_generate_lightgbm_run_id_retries_existing_directory(self, tmp_path: Path) -> None:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from fashion_trend.trend.training.run_artifacts import generate_lightgbm_run_id

        run_root = tmp_path / "outputs" / "models" / "lightgbm" / "runs"
        (run_root / "20260508-153012-aaaaaaaa").mkdir(parents=True)
        tokens = iter(["aaaaaaaa", "bbbbbbbb"])

        run_id = generate_lightgbm_run_id(
            run_root=run_root,
            now_factory=lambda: datetime(2026, 5, 8, 15, 30, 12, tzinfo=ZoneInfo("Asia/Shanghai")),
            token_factory=lambda: next(tokens),
        )

        assert run_id == "20260508-153012-bbbbbbbb"
```

- [ ] **Step 2: Run the failing tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_derive_trend_model_output_paths_uses_lightgbm_run_id tests/test_trend_training.py::TestTrendTraining::test_derive_trend_model_output_paths_rejects_run_id_for_baseline tests/test_trend_training.py::TestTrendTraining::test_validate_lightgbm_run_id_rejects_unsafe_values tests/test_trend_training.py::TestTrendTraining::test_validate_lightgbm_run_id_rejects_reserved_names tests/test_trend_training.py::TestTrendTraining::test_generate_lightgbm_run_id_uses_local_timestamp_and_hex_suffix tests/test_trend_training.py::TestTrendTraining::test_generate_lightgbm_run_id_retries_existing_directory -q
```

Expected: FAIL because `run_id` path support and `run_artifacts.py` do not exist yet.

- [ ] **Step 3: Create run id helpers**

Create `src/fashion_trend/trend/training/run_artifacts.py` with these public constants and functions:

```python
from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fashion_trend.foundation.artifacts import validate_safe_path_segment

LIGHTGBM_RUN_RESERVED_NAMES: frozenset[str] = frozenset(
    {"index.jsonl", "evaluations.jsonl"}
)
LIGHTGBM_RUN_ID_RETRY_LIMIT = 10
LIGHTGBM_RUN_ID_SUFFIX_LENGTH = 8


@dataclass(frozen=True)
class LightGBMRunSummary:
    run_id: str
    created_at: str
    run_dir: str
    promotion_status: str
    params_path: str
    metadata_path: str
    promotion_error: str | None = None


def validate_lightgbm_run_id(run_id: str) -> None:
    validate_safe_path_segment(run_id, "run_id")
    if run_id in LIGHTGBM_RUN_RESERVED_NAMES:
        raise ValueError(f"run_id 是保留名称，不能作为实验目录: {run_id}")


def generate_lightgbm_run_id(
    run_root: Path,
    *,
    now_factory: Callable[[], datetime] = datetime.now,
    token_factory: Callable[[], str] | None = None,
) -> str:
    now = now_factory()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    make_token = token_factory or (lambda: uuid.uuid4().hex[:LIGHTGBM_RUN_ID_SUFFIX_LENGTH])
    for _attempt in range(LIGHTGBM_RUN_ID_RETRY_LIMIT):
        suffix = make_token()
        run_id = f"{timestamp}-{suffix}"
        validate_lightgbm_run_id(run_id)
        if not (run_root / run_id).exists():
            return run_id
    raise FileExistsError(
        f"自动生成 lightgbm run_id 连续冲突 {LIGHTGBM_RUN_ID_RETRY_LIMIT} 次: {run_root}"
    )
```

- [ ] **Step 4: Extend model output path derivation**

Modify `derive_trend_model_output_paths()` in `src/fashion_trend/trend/training/outputs.py` to accept `run_id: str | None = None`. Import `validate_lightgbm_run_id` inside the function or at module top. The function must return the existing keys for stable output and these additional keys when `run_id` is present:

```python
def derive_trend_model_output_paths(
    model_name: str,
    output_root: Path = OUTPUT_MODELS_DIR,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    validate_safe_path_segment(model_name, "model_name")
    stable_output_dir = output_root / model_name
    if run_id is None:
        output_dir = stable_output_dir
        validate_output_parent_dirs(output_dir, output_root)
        return {
            "output_dir": output_dir,
            "predictions": output_dir / "predictions.csv",
            "params": output_dir / "params.json",
            "metadata": output_dir / "metadata.json",
        }

    from fashion_trend.trend.training.run_artifacts import validate_lightgbm_run_id

    if model_name != "lightgbm":
        raise ValueError("只有 lightgbm 支持 run_id。")
    validate_lightgbm_run_id(run_id)
    run_root = stable_output_dir / "runs"
    output_dir = run_root / run_id
    validate_output_parent_dirs(output_dir, output_root)
    return {
        "output_dir": output_dir,
        "stable_output_dir": stable_output_dir,
        "run_root": run_root,
        "index": run_root / "index.jsonl",
        "predictions": output_dir / "predictions.csv",
        "params": output_dir / "params.json",
        "metadata": output_dir / "metadata.json",
    }
```

- [ ] **Step 5: Export the new helpers**

Add these imports and `__all__` entries to `src/fashion_trend/trend/training/__init__.py`:

```python
from fashion_trend.trend.training.run_artifacts import (
    LIGHTGBM_RUN_RESERVED_NAMES,
    generate_lightgbm_run_id,
    validate_lightgbm_run_id,
)
```

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_derive_trend_model_output_paths_uses_model_name tests/test_trend_training.py::TestTrendTraining::test_derive_trend_model_output_paths_uses_lightgbm_run_id tests/test_trend_training.py::TestTrendTraining::test_derive_trend_model_output_paths_rejects_run_id_for_baseline tests/test_trend_training.py::TestTrendTraining::test_validate_lightgbm_run_id_rejects_unsafe_values tests/test_trend_training.py::TestTrendTraining::test_validate_lightgbm_run_id_rejects_reserved_names tests/test_trend_training.py::TestTrendTraining::test_generate_lightgbm_run_id_uses_local_timestamp_and_hex_suffix tests/test_trend_training.py::TestTrendTraining::test_generate_lightgbm_run_id_retries_existing_directory -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```sh
git add src/fashion_trend/trend/training/outputs.py src/fashion_trend/trend/training/run_artifacts.py src/fashion_trend/trend/training/__init__.py tests/test_trend_training.py
git commit -m "feat(trend): 添加 LightGBM run 路径基础"
```

---

### Task 2: LightGBM Parameter Schema

**Files:**
- Create: `src/fashion_trend/trend/models/supervised/lightgbm_config.py`
- Modify: `src/fashion_trend/trend/models/supervised/lightgbm.py`
- Modify: `tests/test_trend_lightgbm.py`

- [ ] **Step 1: Write failing parameter schema tests**

Add these imports to `tests/test_trend_lightgbm.py`:

```python
import json
```

Append these tests inside `TestLightGBMTrendModel`:

```python
    def test_lightgbm_default_params_enable_subsample_freq(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)

        assert lightgbm_model.LIGHTGBM_PARAMS["subsample"] == 0.8
        assert lightgbm_model.LIGHTGBM_PARAMS["subsample_freq"] == 1

    def test_resolve_lightgbm_config_merges_file_and_cli_overrides(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        params_path = tmp_path / "params.json"
        params_path.write_text(
            json.dumps(
                {
                    "lightgbm_params": {"learning_rate": 0.03, "num_leaves": 63},
                    "early_stopping": {"stopping_rounds": 50},
                }
            ),
            encoding="utf-8",
        )

        config = config_module.resolve_lightgbm_config(
            params_path=params_path,
            cli_params=["num_leaves=31", "early_stopping.stopping_rounds=80"],
        )

        assert config.lightgbm_params["learning_rate"] == 0.03
        assert config.lightgbm_params["num_leaves"] == 31
        assert config.early_stopping == {"stopping_rounds": 80}
        assert config.param_source["params_file"] == str(params_path)
        assert config.param_source["overrides"] == {
            "num_leaves": 31,
            "early_stopping.stopping_rounds": 80,
        }

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {"unknown": {}},
            {"lightgbm_params": []},
            {"early_stopping": []},
        ],
    )
    def test_resolve_lightgbm_config_rejects_invalid_params_file_shape(
        self,
        tmp_path: Path,
        payload: object,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        params_path = tmp_path / "params.json"
        params_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="params|lightgbm_params|early_stopping"):
            config_module.resolve_lightgbm_config(params_path=params_path)

    @pytest.mark.parametrize(
        "cli_param",
        [
            "unknown=1",
            "objective=binary",
            "n_estimators=0",
            "learning_rate=0",
            "max_depth=0",
            "subsample=1.2",
            "subsample_freq=0",
            "colsample_bytree=0",
            "reg_alpha=-1",
            "reg_lambda=-1",
            "min_split_gain=-0.1",
            "early_stopping.stopping_rounds=0",
            "lightgbm_params.learning_rate=0.03",
        ],
    )
    def test_resolve_lightgbm_config_rejects_invalid_cli_param(
        self,
        cli_param: str,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )

        with pytest.raises(ValueError, match="参数|objective|subsample|early_stopping"):
            config_module.resolve_lightgbm_config(cli_params=[cli_param])
```

`subsample_freq=0` is intentionally invalid with the default `subsample=0.8`, because schema requires positive `subsample_freq` when row sampling is enabled.

- [ ] **Step 2: Run failing parameter tests**

Run:

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_lightgbm_default_params_enable_subsample_freq tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_merges_file_and_cli_overrides tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_rejects_invalid_params_file_shape tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_rejects_invalid_cli_param -q
```

Expected: FAIL because `lightgbm_config.py` does not exist and `LIGHTGBM_PARAMS` lacks `subsample_freq`.

- [ ] **Step 3: Create LightGBM config module**

Create `src/fashion_trend/trend/models/supervised/lightgbm_config.py` with these public definitions:

```python
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIGHTGBM_ALLOWED_OBJECTIVES: tuple[str, ...] = ("regression", "regression_l1")
LIGHTGBM_DEFAULT_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "min_split_gain": 0.0,
    "random_state": 42,
    "verbosity": -1,
}
LIGHTGBM_DEFAULT_EARLY_STOPPING: dict[str, int] = {"stopping_rounds": 30}
LIGHTGBM_ALLOWED_PARAM_KEYS: frozenset[str] = frozenset(LIGHTGBM_DEFAULT_PARAMS)


@dataclass(frozen=True)
class LightGBMTrainingConfig:
    lightgbm_params: dict[str, object]
    early_stopping: dict[str, int]
    param_source: dict[str, object]


def resolve_lightgbm_config(
    *,
    params_path: Path | None = None,
    cli_params: list[str] | None = None,
) -> LightGBMTrainingConfig:
    lightgbm_params = dict(LIGHTGBM_DEFAULT_PARAMS)
    early_stopping = dict(LIGHTGBM_DEFAULT_EARLY_STOPPING)
    overrides: dict[str, object] = {}
    params_file_value: str | None = None

    if params_path is not None:
        file_payload = _read_params_file(params_path)
        params_file_value = str(params_path)
        lightgbm_params.update(file_payload.get("lightgbm_params", {}))
        early_stopping.update(file_payload.get("early_stopping", {}))

    for raw_param in cli_params or []:
        key, value = _parse_cli_param(raw_param)
        if key == "early_stopping.stopping_rounds":
            early_stopping["stopping_rounds"] = value
        else:
            lightgbm_params[key] = value
        overrides[key] = value

    _validate_lightgbm_params(lightgbm_params)
    _validate_early_stopping(early_stopping)
    return LightGBMTrainingConfig(
        lightgbm_params=lightgbm_params,
        early_stopping={"stopping_rounds": int(early_stopping["stopping_rounds"])},
        param_source={
            "default": "builtin",
            "params_file": params_file_value,
            "overrides": overrides,
        },
    )
```

Implement private helpers in the same file:

```python
def _read_params_file(params_path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LightGBM 参数文件不是合法 JSON: {params_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LightGBM --params 必须是 JSON object。")
    unknown_keys = set(payload) - {"lightgbm_params", "early_stopping"}
    if unknown_keys:
        raise ValueError(f"LightGBM --params 包含不支持的顶层 key: {sorted(unknown_keys)}")
    for key in ("lightgbm_params", "early_stopping"):
        if key in payload and not isinstance(payload[key], dict):
            raise ValueError(f"LightGBM --params 的 {key} 必须是 JSON object。")
    return {
        "lightgbm_params": dict(payload.get("lightgbm_params", {})),
        "early_stopping": dict(payload.get("early_stopping", {})),
    }


def _parse_cli_param(raw_param: str) -> tuple[str, object]:
    if "=" not in raw_param:
        raise ValueError(f"LightGBM --param 必须是 key=value: {raw_param}")
    key, raw_value = raw_param.split("=", maxsplit=1)
    if not key:
        raise ValueError("LightGBM --param key 不能为空。")
    if "." in key and key != "early_stopping.stopping_rounds":
        raise ValueError(f"LightGBM --param 不支持 dotted key: {key}")
    if key != "early_stopping.stopping_rounds" and key not in LIGHTGBM_ALLOWED_PARAM_KEYS:
        raise ValueError(f"LightGBM 参数不在允许清单中: {key}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value
```

Add validators for each schema rule:

```python
def _validate_lightgbm_params(params: dict[str, object]) -> None:
    unknown_keys = set(params) - LIGHTGBM_ALLOWED_PARAM_KEYS
    if unknown_keys:
        raise ValueError(f"LightGBM 参数不在允许清单中: {sorted(unknown_keys)}")
    if params["objective"] not in LIGHTGBM_ALLOWED_OBJECTIVES:
        raise ValueError(f"LightGBM objective 不支持: {params['objective']}")
    _require_positive_int(params["n_estimators"], "n_estimators")
    _require_positive_int(params["num_leaves"], "num_leaves")
    _require_positive_int(params["min_child_samples"], "min_child_samples")
    _require_int(params["random_state"], "random_state")
    _require_int(params["verbosity"], "verbosity")
    max_depth = _require_int(params["max_depth"], "max_depth")
    if max_depth != -1 and max_depth <= 0:
        raise ValueError("LightGBM max_depth 必须是 -1 或正整数。")
    _require_positive_number(params["learning_rate"], "learning_rate")
    subsample = _require_unit_interval(params["subsample"], "subsample")
    subsample_freq = _require_non_negative_int(params["subsample_freq"], "subsample_freq")
    if subsample < 1.0 and subsample_freq <= 0:
        raise ValueError("LightGBM subsample < 1 时 subsample_freq 必须是正整数。")
    _require_unit_interval(params["colsample_bytree"], "colsample_bytree")
    _require_non_negative_number(params["reg_alpha"], "reg_alpha")
    _require_non_negative_number(params["reg_lambda"], "reg_lambda")
    _require_non_negative_number(params["min_split_gain"], "min_split_gain")


def _validate_early_stopping(early_stopping: dict[str, object]) -> None:
    unknown_keys = set(early_stopping) - {"stopping_rounds"}
    if unknown_keys:
        raise ValueError(f"LightGBM early_stopping 不支持 key: {sorted(unknown_keys)}")
    _require_positive_int(early_stopping.get("stopping_rounds"), "early_stopping.stopping_rounds")
```

Use helper functions that reject `bool` as an integer:

```python
def _require_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"LightGBM {key} 必须是整数。")
    return int(value)


def _require_positive_int(value: object, key: str) -> int:
    number = _require_int(value, key)
    if number <= 0:
        raise ValueError(f"LightGBM {key} 必须是正整数。")
    return number


def _require_non_negative_int(value: object, key: str) -> int:
    number = _require_int(value, key)
    if number < 0:
        raise ValueError(f"LightGBM {key} 必须是非负整数。")
    return number


def _require_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LightGBM {key} 必须是数值。")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"LightGBM {key} 必须是有限数值。")
    return number


def _require_positive_number(value: object, key: str) -> float:
    number = _require_number(value, key)
    if number <= 0:
        raise ValueError(f"LightGBM {key} 必须大于 0。")
    return number


def _require_non_negative_number(value: object, key: str) -> float:
    number = _require_number(value, key)
    if number < 0:
        raise ValueError(f"LightGBM {key} 必须是非负数值。")
    return number


def _require_unit_interval(value: object, key: str) -> float:
    number = _require_number(value, key)
    if number <= 0 or number > 1:
        raise ValueError(f"LightGBM {key} 必须在 (0, 1]。")
    return number
```

- [ ] **Step 4: Wire defaults into `lightgbm.py`**

In `src/fashion_trend/trend/models/supervised/lightgbm.py`, replace local default constants with imports:

```python
from fashion_trend.trend.models.supervised.lightgbm_config import (
    LIGHTGBM_ALLOWED_OBJECTIVES,
    LIGHTGBM_DEFAULT_EARLY_STOPPING,
    LIGHTGBM_DEFAULT_PARAMS,
    LightGBMTrainingConfig,
    resolve_lightgbm_config,
)

LIGHTGBM_PARAMS: dict[str, object] = dict(LIGHTGBM_DEFAULT_PARAMS)
LIGHTGBM_EARLY_STOPPING: dict[str, int] = dict(LIGHTGBM_DEFAULT_EARLY_STOPPING)
```

Update the existing `test_lightgbm_constants_are_stable()` expected `LIGHTGBM_PARAMS` block to include the new tunable defaults:

```python
        assert lightgbm_model.LIGHTGBM_PARAMS == {
            "objective": "regression",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_split_gain": 0.0,
            "random_state": 42,
            "verbosity": -1,
        }
```

- [ ] **Step 5: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_lightgbm_default_params_enable_subsample_freq tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_merges_file_and_cli_overrides tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_rejects_invalid_params_file_shape tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_rejects_invalid_cli_param -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```sh
git add src/fashion_trend/trend/models/supervised/lightgbm_config.py src/fashion_trend/trend/models/supervised/lightgbm.py tests/test_trend_lightgbm.py
git commit -m "feat(trend): 增加 LightGBM 参数 schema"
```

---

### Task 3: Trainer Options And Metadata Separation

**Files:**
- Modify: `src/fashion_trend/trend/models/base.py`
- Modify: `src/fashion_trend/trend/models/supervised/lightgbm.py`
- Modify: `src/fashion_trend/trend/training/outputs.py`
- Modify: `tests/test_trend_lightgbm.py`
- Modify: `tests/test_trend_training.py`

- [ ] **Step 1: Write failing trainer option test**

Append this test to `tests/test_trend_lightgbm.py`:

```python
    def test_trainer_uses_context_lightgbm_config(self, monkeypatch) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        from fashion_trend.trend.models.base import TrendTrainContext
        from fashion_trend.trend.splits import build_trend_model_split_frames
        from tests.trend_samples import sample_trend_model_samples_for_split

        captured: dict[str, object] = {}

        def fake_fit(train_features, train_target, valid_features, valid_target, *, config):
            captured["params"] = dict(config.lightgbm_params)
            captured["early_stopping"] = dict(config.early_stopping)
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        config = config_module.resolve_lightgbm_config(
            cli_params=["learning_rate=0.03", "early_stopping.stopping_rounds=50"]
        )

        result = lightgbm_model.LightGBMTrendTrainer().train(
            TrendTrainContext(
                model_name="lightgbm",
                split_frames=split_frames,
                input_paths={
                    "train": Path("train.parquet"),
                    "valid": Path("valid.parquet"),
                    "test": Path("test.parquet"),
                },
                output_dir=Path("outputs/models/lightgbm/runs/custom"),
                trainer_options={"lightgbm_config": config},
            )
        )

        assert captured["params"]["learning_rate"] == 0.03
        assert captured["early_stopping"] == {"stopping_rounds": 50}
        assert result.params["lightgbm_params"]["learning_rate"] == 0.03
        assert result.params["early_stopping"] == {"stopping_rounds": 50}
        assert result.metadata["param_source"]["overrides"] == {
            "learning_rate": 0.03,
            "early_stopping.stopping_rounds": 50,
        }
```

- [ ] **Step 2: Write failing metadata path separation test**

Append this test to `tests/test_trend_training.py`:

```python
    def test_build_trend_train_metadata_records_run_context_paths(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        context = TrendTrainContext(
            model_name=LIGHTGBM_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/lightgbm/runs/depth6-lr005"),
        )
        result = TrendTrainResult(
            model_name=LIGHTGBM_MODEL_NAME,
            model_type=MODEL_TYPE_SUPERVISED,
            predictions=sample_trend_predictions_for_evaluation().assign(model_name="lightgbm"),
            params={"model_name": "lightgbm"},
            metadata={"param_source": {"default": "builtin", "params_file": None, "overrides": {}}},
        )
        paths = derive_trend_model_output_paths(
            LIGHTGBM_MODEL_NAME,
            Path("outputs/models"),
            run_id="depth6-lr005",
        )

        metadata = build_trend_train_metadata(
            result,
            context,
            paths,
            run_id="depth6-lr005",
            run_dir=paths["output_dir"],
            stable_output_dir=paths["stable_output_dir"],
            promotion_requested=False,
        )

        assert metadata["run_id"] == "depth6-lr005"
        assert isinstance(metadata["created_at"], str)
        assert metadata["created_at"]
        assert metadata["run_dir"] == "outputs/models/lightgbm/runs/depth6-lr005"
        assert metadata["stable_output_dir"] == "outputs/models/lightgbm"
        assert metadata["promotion_requested"] is False
        assert metadata["prediction_path"].endswith("runs/depth6-lr005/predictions.csv")
```

Add imports where needed:

```python
from tests.trend_samples import sample_trend_predictions_for_evaluation
```

- [ ] **Step 3: Run failing tests**

Run:

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_trainer_uses_context_lightgbm_config tests/test_trend_training.py::TestTrendTraining::test_build_trend_train_metadata_records_run_context_paths -q
```

Expected: FAIL because `trainer_options` and run metadata fields are not implemented.

- [ ] **Step 4: Add trainer options to context**

Modify `TrendTrainContext` in `src/fashion_trend/trend/models/base.py`:

```python
@dataclass(frozen=True)
class TrendTrainContext:
    """通用训练 runner 传给具体趋势模型训练器的输入上下文。"""

    model_name: str
    split_frames: Mapping[str, pd.DataFrame]
    input_paths: Mapping[str, Path]
    output_dir: Path
    split_order: tuple[str, ...] = ("train", "valid", "test")
    trainer_options: Mapping[str, object] = field(default_factory=dict)
```

- [ ] **Step 5: Make LightGBM trainer use resolved config**

In `LightGBMTrendTrainer.train()`, read config before feature preparation:

```python
config = _resolve_context_config(context)
```

Pass the config into fit:

```python
model = _fit_lightgbm_model(
    train_prepared.features,
    _read_target(split_frames["train"]),
    valid_prepared.features,
    _read_target(split_frames["valid"]),
    config=config,
)
```

Add this helper in `lightgbm.py`:

```python
def _resolve_context_config(context: TrendTrainContext) -> LightGBMTrainingConfig:
    config = context.trainer_options.get("lightgbm_config")
    if config is None:
        return resolve_lightgbm_config()
    if not isinstance(config, LightGBMTrainingConfig):
        raise ValueError("lightgbm_config 必须是 LightGBMTrainingConfig。")
    return config
```

Change `_fit_lightgbm_model()` signature to require the resolved config:

```python
def _fit_lightgbm_model(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    valid_features: pd.DataFrame,
    valid_target: pd.Series,
    *,
    config: LightGBMTrainingConfig,
):
```

Inside the function, keep the existing delayed LightGBM import, `LGBMRegressor.fit()`, `eval_set`, `eval_metric`, and callbacks structure. Only change the parameter source to `LGBMRegressor(**config.lightgbm_params)` and the early stopping round source to `int(config.early_stopping["stopping_rounds"])`.

Update all tests that call `_fit_lightgbm_model()` directly to pass `config=lightgbm_model.resolve_lightgbm_config()` or import the config module.

Update all tests that monkeypatch `_fit_lightgbm_model()` so the fake accepts the new keyword-only config. In the current test suite, these existing fakes must change:

```python
# tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_trainer_returns_standard_train_result
def fake_fit(train_features, train_target, valid_features, valid_target, *, config):
    assert config.lightgbm_params["subsample_freq"] == 1
    return _FakeLightGBMModel(train_features.columns.tolist())
```

```python
# tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_lightgbm_outputs
def fake_fit(train_features, train_target, valid_features, valid_target, *, config):
    assert config.lightgbm_params["subsample_freq"] == 1
    return FakeModel()
```

In `_build_lightgbm_params()`, accept `config`:

```python
def _build_lightgbm_params(model, config: LightGBMTrainingConfig) -> dict[str, object]:
    return {
        "model_name": LIGHTGBM_MODEL_NAME,
        "model_type": MODEL_TYPE_SUPERVISED,
        "target_column": LIGHTGBM_TARGET_COLUMN,
        "numeric_features": list(LIGHTGBM_NUMERIC_FEATURES),
        "categorical_features": list(LIGHTGBM_CATEGORICAL_FEATURES),
        "excluded_columns": list(LIGHTGBM_EXCLUDED_COLUMNS),
        "epsilon": LIGHTGBM_EPSILON,
        "lightgbm_params": dict(config.lightgbm_params),
        "early_stopping": dict(config.early_stopping),
        "best_iteration": _read_best_iteration(model),
        "objective": str(config.lightgbm_params["objective"]),
        "allowed_objectives": list(LIGHTGBM_ALLOWED_OBJECTIVES),
    }
```

Add `param_source` to trainer metadata:

```python
"param_source": dict(config.param_source),
```

- [ ] **Step 6: Extend metadata builder**

Modify `build_trend_train_metadata()` signature in `src/fashion_trend/trend/training/outputs.py`:

```python
def build_trend_train_metadata(
    result: TrendTrainResult,
    context: TrendTrainContext,
    output_paths: Mapping[str, Path],
    *,
    run_id: str | None = None,
    run_dir: Path | None = None,
    stable_output_dir: Path | None = None,
    promotion_requested: bool | None = None,
) -> dict[str, object]:
```

After `core_metadata` is built, add:

```python
    if run_id is not None:
        from datetime import datetime

        core_metadata["run_id"] = run_id
        core_metadata["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        core_metadata["run_dir"] = str(run_dir or output_paths["output_dir"])
        if stable_output_dir is not None:
            core_metadata["stable_output_dir"] = str(stable_output_dir)
        if promotion_requested is not None:
            core_metadata["promotion_requested"] = bool(promotion_requested)
```

- [ ] **Step 7: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_lightgbm.py tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_lightgbm_outputs tests/test_trend_training.py::TestTrendTraining::test_build_trend_train_metadata_records_run_context_paths -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```sh
git add src/fashion_trend/trend/models/base.py src/fashion_trend/trend/models/supervised/lightgbm.py src/fashion_trend/trend/training/outputs.py tests/test_trend_lightgbm.py tests/test_trend_training.py
git commit -m "feat(trend): 支持 LightGBM 训练配置传递"
```

---

### Task 4: Training Runner Run Writes And Index Preparation

**Files:**
- Modify: `tests/test_trend_training.py`
- Modify: `src/fashion_trend/foundation/io.py`
- Modify: `src/fashion_trend/trend/training/run_artifacts.py`
- Modify: `src/fashion_trend/trend/training/runner.py`

**Commit boundary:** Do not commit at the end of this task. Task 4 changes the LightGBM write path to run directories, while Task 5 completes the default stable promotion behavior required by the existing no-argument LightGBM tests. Commit Task 4 and Task 5 together after Task 5 verification.

- [ ] **Step 1: Write failing training run tests**

Append these tests to `tests/test_trend_training.py`:

```python
    def test_run_trend_model_training_writes_lightgbm_run_without_stable_promotion(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

        def fake_fit(train_features, train_target, valid_features, valid_target, *, config):
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        input_paths = {
            "train": tmp_path / "trend_model_samples_train.parquet",
            "valid": tmp_path / "trend_model_samples_valid.parquet",
            "test": tmp_path / "trend_model_samples_test.parquet",
        }
        for split_name, split_frame in split_frames.items():
            write_parquet_atomic(split_frame, input_paths[split_name])

        metadata = run_trend_model_training(
            LIGHTGBM_MODEL_NAME,
            input_paths=input_paths,
            output_root=tmp_path / "outputs" / "models",
            run_id="depth6-lr005",
            promote=False,
        )

        run_dir = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005"
        stable_dir = tmp_path / "outputs" / "models" / "lightgbm"
        assert (run_dir / "predictions.csv").exists()
        assert (run_dir / "params.json").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "feature_importance.csv").exists()
        assert (run_dir / "model.txt").exists()
        assert not (stable_dir / "predictions.csv").exists()
        assert metadata["run_id"] == "depth6-lr005"
        assert metadata["promotion_requested"] is False

        index_lines = (stable_dir / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(index_lines) == 1
        summary = json.loads(index_lines[0])
        assert summary["run_id"] == "depth6-lr005"
        assert summary["promotion_status"] == "not_requested"

    def test_run_trend_model_training_rejects_existing_manual_lightgbm_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

        monkeypatch.setattr(
            lightgbm_model,
            "_fit_lightgbm_model",
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(train_features.columns.tolist()),
        )
        run_dir = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005"
        run_dir.mkdir(parents=True)

        with pytest.raises(FileExistsError, match="depth6-lr005"):
            run_trend_model_training(
                LIGHTGBM_MODEL_NAME,
                input_paths=_write_sample_split_inputs(tmp_path),
                output_root=tmp_path / "outputs" / "models",
                run_id="depth6-lr005",
                promote=False,
            )

    def test_run_trend_model_training_manual_run_id_defaults_to_no_promote(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

        monkeypatch.setattr(
            lightgbm_model,
            "_fit_lightgbm_model",
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(train_features.columns.tolist()),
        )

        run_trend_model_training(
            LIGHTGBM_MODEL_NAME,
            input_paths=_write_sample_split_inputs(tmp_path),
            output_root=tmp_path / "outputs" / "models",
            run_id="depth6-lr005",
        )

        stable_dir = tmp_path / "outputs" / "models" / "lightgbm"
        run_dir = stable_dir / "runs" / "depth6-lr005"
        assert (run_dir / "predictions.csv").exists()
        assert not (stable_dir / "predictions.csv").exists()
        row = json.loads((stable_dir / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["promotion_status"] == "not_requested"

    def test_run_trend_model_training_custom_params_default_to_no_promote(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model
        from fashion_trend.trend.models.supervised.lightgbm_config import (
            resolve_lightgbm_config,
        )

        monkeypatch.setattr(
            lightgbm_model,
            "_fit_lightgbm_model",
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(train_features.columns.tolist()),
        )

        metadata = run_trend_model_training(
            LIGHTGBM_MODEL_NAME,
            input_paths=_write_sample_split_inputs(tmp_path),
            output_root=tmp_path / "outputs" / "models",
            trainer_options={
                "lightgbm_config": resolve_lightgbm_config(
                    cli_params=["learning_rate=0.03"]
                )
            },
        )

        stable_dir = tmp_path / "outputs" / "models" / "lightgbm"
        run_dir = stable_dir / "runs" / str(metadata["run_id"])
        assert (run_dir / "predictions.csv").exists()
        assert not (stable_dir / "predictions.csv").exists()
        row = json.loads((stable_dir / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["promotion_status"] == "not_requested"

    def test_run_trend_model_training_rejects_run_options_for_baseline(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="lightgbm"):
            run_trend_model_training(
                LAST_WEEK_MODEL_NAME,
                input_paths=_write_sample_split_inputs(tmp_path),
                output_root=tmp_path / "outputs" / "models",
                run_id="bad",
                promote=False,
            )
```

Add module-level helpers near the bottom of `tests/test_trend_training.py`:

```python
def _write_sample_split_inputs(tmp_path: Path) -> dict[str, Path]:
    split_frames = build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )
    input_paths = {
        "train": tmp_path / "trend_model_samples_train.parquet",
        "valid": tmp_path / "trend_model_samples_valid.parquet",
        "test": tmp_path / "trend_model_samples_test.parquet",
    }
    for split_name, split_frame in split_frames.items():
        write_parquet_atomic(split_frame, input_paths[split_name])
    return input_paths
```

Add these fake classes at module bottom in `tests/test_trend_training.py`:

```python
class _FakeBooster:
    def __init__(self, feature_names: list[str]) -> None:
        self._feature_names = feature_names

    def feature_name(self) -> list[str]:
        return list(self._feature_names)

    def feature_importance(self, importance_type: str):
        if importance_type == "split":
            return [1 for _ in self._feature_names]
        if importance_type == "gain":
            return [1.0 for _ in self._feature_names]
        raise AssertionError(f"unexpected importance_type={importance_type}")

    def model_to_string(self) -> str:
        return "fake lightgbm model"


class _FakeLightGBMModel:
    best_iteration_ = 7
    best_score_ = {"valid_0": {"l2": 0.12}}

    def __init__(self, feature_names: list[str]) -> None:
        self.booster_ = _FakeBooster(feature_names)

    def predict(self, features: pd.DataFrame, num_iteration: int | None = None):
        return features["growth_lag_1"].astype(float).to_numpy()
```

- [ ] **Step 2: Run failing tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_lightgbm_run_without_stable_promotion tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_rejects_existing_manual_lightgbm_run tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_manual_run_id_defaults_to_no_promote tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_custom_params_default_to_no_promote tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_rejects_run_options_for_baseline -q
```

Expected: FAIL because `run_trend_model_training()` has not accepted run options.

- [ ] **Step 3: Add public text atomic writer and index upsert helper**

Add `write_text_atomic()` to `src/fashion_trend/foundation/io.py`:

```python
def write_text_atomic(text: str, output_path: Path) -> None:
    """先创建父目录，再通过临时文本文件原子替换目标产物。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)
```

Add these functions to `src/fashion_trend/trend/training/run_artifacts.py`:

```python
from fashion_trend.foundation.io import write_text_atomic

LIGHTGBM_PROMOTION_STATUSES: frozenset[str] = frozenset(
    {"not_requested", "succeeded", "failed"}
)


def build_lightgbm_run_summary(
    *,
    run_id: str,
    metadata: dict[str, object],
    promotion_status: str,
    promotion_error: str | None = None,
) -> LightGBMRunSummary:
    if promotion_status not in LIGHTGBM_PROMOTION_STATUSES:
        raise ValueError(f"未知 LightGBM promotion_status: {promotion_status}")
    return LightGBMRunSummary(
        run_id=run_id,
        created_at=str(metadata.get("created_at", "")),
        run_dir=str(metadata["run_dir"]),
        promotion_status=promotion_status,
        params_path=str(metadata["params_path"]),
        metadata_path=str(Path(str(metadata["run_dir"])) / "metadata.json"),
        promotion_error=promotion_error,
    )


def upsert_lightgbm_run_index(index_path: Path, summary: LightGBMRunSummary) -> None:
    summaries: dict[str, dict[str, object]] = {}
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            summaries[str(payload["run_id"])] = payload
    row = {
        "run_id": summary.run_id,
        "created_at": summary.created_at,
        "run_dir": summary.run_dir,
        "promotion_status": summary.promotion_status,
        "params_path": summary.params_path,
        "metadata_path": summary.metadata_path,
    }
    if summary.promotion_error is not None:
        row["promotion_error"] = summary.promotion_error
    summaries[summary.run_id] = row
    lines = [
        json.dumps(summaries[key], ensure_ascii=False, sort_keys=True)
        for key in sorted(summaries)
    ]
    write_text_atomic("\n".join(lines) + "\n", index_path)
```

- [ ] **Step 4: Extend training runner signature and baseline guard**

Modify `run_trend_model_training()` in `src/fashion_trend/trend/training/runner.py`:

```python
def run_trend_model_training(
    model_name: str,
    input_paths: Mapping[str, Path] | None = None,
    output_root: Path = OUTPUT_MODELS_DIR,
    *,
    run_id: str | None = None,
    trainer_options: Mapping[str, object] | None = None,
    promote: bool | None = None,
) -> dict[str, object]:
```

Add a guard after trainer lookup:

```python
    is_lightgbm = model_name == "lightgbm"
    if not is_lightgbm and (run_id is not None or trainer_options or promote is not None):
        raise ValueError("run、参数和 promotion 选项只支持 --model lightgbm。")
```

- [ ] **Step 5: Implement LightGBM run write path**

In `runner.py`, import the module so tests can monkeypatch run artifact operations consistently:

```python
from fashion_trend.trend.training import run_artifacts
```

In `run_trend_model_training()`, branch LightGBM into a helper:

```python
    if is_lightgbm:
        return _run_lightgbm_training(
            trainer=trainer,
            model_name=model_name,
            input_paths=resolved_input_paths,
            output_root=output_root,
            run_id=run_id,
            trainer_options=trainer_options or {},
            promote=promote,
        )
```

Implement `_run_lightgbm_training()` in the same file:

```python
def _run_lightgbm_training(
    *,
    trainer,
    model_name: str,
    input_paths: Mapping[str, Path],
    output_root: Path,
    run_id: str | None,
    trainer_options: Mapping[str, object],
    promote: bool | None,
) -> dict[str, object]:
    split_frames = read_trend_model_split_frames(input_paths)
    stable_paths = derive_trend_model_output_paths(model_name, output_root)
    run_root = stable_paths["output_dir"] / "runs"
    explicit_run_id = run_id is not None
    resolved_run_id = run_id or generate_lightgbm_run_id(run_root)
    run_paths = derive_trend_model_output_paths(
        model_name,
        output_root,
        run_id=resolved_run_id,
    )
    if run_paths["output_dir"].exists():
        raise FileExistsError(f"LightGBM run_id 已存在: {resolved_run_id}")
    promotion_requested = _resolve_lightgbm_promotion_default(
        explicit_run_id=explicit_run_id,
        trainer_options=trainer_options,
        promote=promote,
    )
    context = TrendTrainContext(
        model_name=model_name,
        split_frames=split_frames,
        input_paths=input_paths,
        output_dir=run_paths["output_dir"],
        trainer_options=trainer_options,
    )
    result = trainer.train(context)
    validate_trend_train_result(result, context)
    metadata = build_trend_train_metadata(
        result,
        context,
        run_paths,
        run_id=resolved_run_id,
        run_dir=run_paths["output_dir"],
        stable_output_dir=run_paths["stable_output_dir"],
        promotion_requested=promotion_requested,
    )
    write_trend_model_outputs(result, metadata, run_paths)
    run_artifacts.upsert_lightgbm_run_index(
        run_paths["index"],
        run_artifacts.build_lightgbm_run_summary(
            run_id=resolved_run_id,
            metadata=metadata,
            promotion_status="not_requested",
        ),
    )
    return metadata
```

Add promotion default helper:

```python
def _resolve_lightgbm_promotion_default(
    *,
    explicit_run_id: bool,
    trainer_options: Mapping[str, object],
    promote: bool | None,
) -> bool:
    if promote is not None:
        return bool(promote)
    has_custom_config = bool(trainer_options)
    return not explicit_run_id and not has_custom_config
```

- [ ] **Step 6: Extract and keep baseline path unchanged**

Move the current non-LightGBM body into `_run_standard_trend_model_training()`:

```python
def _run_standard_trend_model_training(
    *,
    trainer,
    model_name: str,
    input_paths: Mapping[str, Path],
    output_root: Path,
) -> dict[str, object]:
    split_frames = read_trend_model_split_frames(input_paths)
    output_paths = derive_trend_model_output_paths(model_name, output_root)
    context = TrendTrainContext(
        model_name=model_name,
        split_frames=split_frames,
        input_paths=input_paths,
        output_dir=output_paths["output_dir"],
    )
    result = trainer.train(context)
    validate_trend_train_result(result, context)
    metadata = build_trend_train_metadata(result, context, output_paths)
    write_trend_model_outputs(result, metadata, output_paths)
    return metadata
```

Call this helper for every non-LightGBM model. It must call `derive_trend_model_output_paths(model_name, output_root)` without `run_id`.

- [ ] **Step 7: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_standard_outputs tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_lightgbm_run_without_stable_promotion tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_rejects_existing_manual_lightgbm_run tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_manual_run_id_defaults_to_no_promote tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_custom_params_default_to_no_promote tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_rejects_run_options_for_baseline -q
```

Expected: PASS.

- [ ] **Step 8: Continue directly to Task 5**

Do not commit yet. The existing `tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_lightgbm_outputs` still expects default LightGBM training to publish stable outputs, and that behavior is completed in Task 5.

---

### Task 5: Training-Time Promotion To Stable Model Outputs

**Files:**
- Modify: `tests/test_trend_training.py`
- Modify: `src/fashion_trend/trend/training/run_artifacts.py`
- Modify: `src/fashion_trend/trend/training/runner.py`

- [ ] **Step 1: Write failing promotion tests**

Append these tests to `tests/test_trend_training.py`:

```python
    def test_run_trend_model_training_promotes_default_lightgbm_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

        monkeypatch.setattr(
            lightgbm_model,
            "_fit_lightgbm_model",
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(train_features.columns.tolist()),
        )

        metadata = run_trend_model_training(
            LIGHTGBM_MODEL_NAME,
            input_paths=_write_sample_split_inputs(tmp_path),
            output_root=tmp_path / "outputs" / "models",
        )

        stable_dir = tmp_path / "outputs" / "models" / "lightgbm"
        run_dir = stable_dir / "runs" / str(metadata["run_id"])
        assert (run_dir / "predictions.csv").exists()
        assert (stable_dir / "predictions.csv").exists()
        assert (stable_dir / "params.json").exists()
        assert (stable_dir / "metadata.json").exists()
        stable_metadata = json.loads((stable_dir / "metadata.json").read_text(encoding="utf-8"))
        assert stable_metadata["run_id"] == metadata["run_id"]
        assert stable_metadata["output_dir"] == str(stable_dir)
        assert stable_metadata["prediction_path"] == str(stable_dir / "predictions.csv")
        assert stable_metadata["run_dir"] == str(run_dir)

        index_rows = [
            json.loads(line)
            for line in (stable_dir / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert index_rows[0]["promotion_status"] == "succeeded"

    def test_run_trend_model_training_promote_failure_keeps_run_and_returns_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model
        from fashion_trend.trend.training import run_artifacts

        monkeypatch.setattr(
            lightgbm_model,
            "_fit_lightgbm_model",
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(train_features.columns.tolist()),
        )

        def broken_promotion(*args, **kwargs):
            raise OSError("stable write failed")

        monkeypatch.setattr(run_artifacts, "publish_lightgbm_run_to_stable", broken_promotion)

        with pytest.raises(OSError, match="stable write failed"):
            run_trend_model_training(
                LIGHTGBM_MODEL_NAME,
                input_paths=_write_sample_split_inputs(tmp_path),
                output_root=tmp_path / "outputs" / "models",
                run_id="depth6-lr005",
                promote=True,
            )

        run_dir = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005"
        assert (run_dir / "predictions.csv").exists()
        index_path = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "index.jsonl"
        row = json.loads(index_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["promotion_status"] == "failed"
        assert "stable write failed" in row["promotion_error"]

    def test_run_trend_model_training_promote_failure_preserves_original_error_when_index_update_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model
        from fashion_trend.trend.training import run_artifacts

        monkeypatch.setattr(
            lightgbm_model,
            "_fit_lightgbm_model",
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(train_features.columns.tolist()),
        )
        monkeypatch.setattr(
            run_artifacts,
            "publish_lightgbm_run_to_stable",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("stable write failed")),
        )

        original_upsert = run_artifacts.upsert_lightgbm_run_index
        calls = {"count": 0}

        def flaky_upsert(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return original_upsert(*args, **kwargs)
            raise OSError("index write failed")

        monkeypatch.setattr(run_artifacts, "upsert_lightgbm_run_index", flaky_upsert)

        with pytest.raises(OSError, match="stable write failed"):
            run_trend_model_training(
                LIGHTGBM_MODEL_NAME,
                input_paths=_write_sample_split_inputs(tmp_path),
                output_root=tmp_path / "outputs" / "models",
                run_id="depth6-lr005",
                promote=True,
            )

        captured = capsys.readouterr()
        assert "stable write failed" in captured.err
        assert "index write failed" in captured.err
        assert "depth6-lr005" in captured.err

    def test_run_trend_model_training_promote_success_index_failure_does_not_mark_failed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model
        from fashion_trend.trend.training import run_artifacts

        monkeypatch.setattr(
            lightgbm_model,
            "_fit_lightgbm_model",
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(train_features.columns.tolist()),
        )

        def successful_publish(*, stable_paths: dict[str, Path], **kwargs) -> dict[str, object]:
            stable_paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
            stable_paths["predictions"].write_text("stable published\n", encoding="utf-8")
            return {"run_id": "depth6-lr005"}

        monkeypatch.setattr(
            run_artifacts,
            "publish_lightgbm_run_to_stable",
            successful_publish,
        )
        original_upsert = run_artifacts.upsert_lightgbm_run_index
        calls = {"count": 0}

        def flaky_upsert(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return original_upsert(*args, **kwargs)
            raise OSError("succeeded index write failed")

        monkeypatch.setattr(run_artifacts, "upsert_lightgbm_run_index", flaky_upsert)

        with pytest.raises(OSError, match="succeeded index write failed"):
            run_trend_model_training(
                LIGHTGBM_MODEL_NAME,
                input_paths=_write_sample_split_inputs(tmp_path),
                output_root=tmp_path / "outputs" / "models",
                run_id="depth6-lr005",
                promote=True,
            )

        stable_prediction_path = tmp_path / "outputs" / "models" / "lightgbm" / "predictions.csv"
        assert stable_prediction_path.read_text(encoding="utf-8") == "stable published\n"
        captured = capsys.readouterr()
        assert "promotion succeeded" in captured.err
        assert "succeeded index write failed" in captured.err
        index_path = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "index.jsonl"
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        assert rows == [
            {
                "created_at": rows[0]["created_at"],
                "metadata_path": str(tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005" / "metadata.json"),
                "params_path": str(tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005" / "params.json"),
                "promotion_status": "not_requested",
                "run_dir": str(tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005"),
                "run_id": "depth6-lr005",
            }
        ]

    def test_write_promotion_items_atomic_rolls_back_cross_directory_partial_publish(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import (
            PromotionItem,
            write_promotion_items_atomic,
        )

        stable_model_path = tmp_path / "outputs" / "models" / "lightgbm" / "predictions.csv"
        stable_model_path.parent.mkdir(parents=True)
        stable_model_path.write_text("old model\n", encoding="utf-8")
        broken_metrics_parent = tmp_path / "outputs" / "metrics" / "lightgbm"
        broken_metrics_parent.parent.mkdir(parents=True)
        broken_metrics_parent.write_text("not a directory\n", encoding="utf-8")

        with pytest.raises(OSError):
            write_promotion_items_atomic(
                [
                    PromotionItem(stable_model_path, b"new model\n"),
                    PromotionItem(broken_metrics_parent / "trend_metrics.json", {"new": True}),
                ],
                tmp_path / "outputs" / "models" / "lightgbm",
            )

        assert stable_model_path.read_text(encoding="utf-8") == "old model\n"
        assert broken_metrics_parent.read_text(encoding="utf-8") == "not a directory\n"
```

- [ ] **Step 2: Run failing tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promotes_default_lightgbm_run tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promote_failure_keeps_run_and_returns_error tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promote_failure_preserves_original_error_when_index_update_fails tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promote_success_index_failure_does_not_mark_failed tests/test_trend_training.py::TestTrendTraining::test_write_promotion_items_atomic_rolls_back_cross_directory_partial_publish -q
```

Expected: FAIL because training-time promotion is not implemented.

- [ ] **Step 3: Add cross-directory promotion item writer**

Add to `src/fashion_trend/trend/training/run_artifacts.py`:

```python
import shutil

from fashion_trend.foundation import logging as log
from fashion_trend.foundation.io import remove_file_if_exists, write_json_atomic


@dataclass(frozen=True)
class PromotionItem:
    final_path: Path
    payload: bytes | dict[str, object]


def write_promotion_items_atomic(items: list[PromotionItem], staging_root: Path) -> None:
    staging_dir = staging_root / f".tmp-lightgbm-promotion-{uuid.uuid4().hex}"
    published_paths: list[tuple[Path, Path | None]] = []
    try:
        staged_paths: list[tuple[Path, Path]] = []
        for index, item in enumerate(items):
            suffix = item.final_path.suffix
            staging_path = staging_dir / f"item-{index}{suffix}"
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(item.payload, dict):
                write_json_atomic(item.payload, staging_path)
            else:
                staging_path.write_bytes(item.payload)
            staged_paths.append((item.final_path, staging_path))

        for final_path, staging_path in staged_paths:
            backup_path = None
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                backup_path = final_path.with_name(
                    f".{final_path.name}.bak-{uuid.uuid4().hex}"
                )
                final_path.replace(backup_path)
            published_paths.append((final_path, backup_path))
            staging_path.replace(final_path)
    except Exception:
        _rollback_promoted_outputs(published_paths)
        raise
    else:
        _remove_promotion_backups(published_paths)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
```

Add rollback helpers:

```python
def _rollback_promoted_outputs(published_paths: list[tuple[Path, Path | None]]) -> None:
    for final_path, backup_path in reversed(published_paths):
        if backup_path is None:
            remove_file_if_exists(final_path)
            continue
        remove_file_if_exists(final_path)
        backup_path.replace(final_path)


def _remove_promotion_backups(published_paths: list[tuple[Path, Path | None]]) -> None:
    for _final_path, backup_path in published_paths:
        if backup_path is not None:
            remove_file_if_exists(backup_path)
```

- [ ] **Step 4: Add model-only stable promotion helper**

Add this function to `run_artifacts.py`:

```python
def publish_lightgbm_run_to_stable(
    *,
    result,
    run_metadata: dict[str, object],
    run_context,
    stable_paths: dict[str, Path],
    include_metrics: bool = False,
    metrics_item: PromotionItem | None = None,
) -> dict[str, object]:
    stable_metadata = build_trend_train_metadata(
        result,
        run_context,
        stable_paths,
        run_id=str(run_metadata["run_id"]),
        run_dir=Path(str(run_metadata["run_dir"])),
        stable_output_dir=stable_paths["output_dir"],
        promotion_requested=True,
    )
    items = [
        PromotionItem(stable_paths["predictions"], run_metadata_path(run_metadata, "predictions").read_bytes()),
        PromotionItem(stable_paths["params"], run_metadata_path(run_metadata, "params").read_bytes()),
    ]
    for artifact in result.artifacts:
        source_path = Path(str(run_metadata["run_dir"])) / artifact.relative_path
        items.append(PromotionItem(stable_paths["output_dir"] / artifact.relative_path, source_path.read_bytes()))
    items.append(PromotionItem(stable_paths["metadata"], stable_metadata))
    if include_metrics:
        if metrics_item is None:
            raise ValueError("发布 stable metrics 时必须提供 metrics_item。")
        items.append(metrics_item)
    write_promotion_items_atomic(items, stable_paths["output_dir"])
    return stable_metadata


def run_metadata_path(run_metadata: dict[str, object], artifact_name: str) -> Path:
    if artifact_name == "predictions":
        return Path(str(run_metadata["prediction_path"]))
    if artifact_name == "params":
        return Path(str(run_metadata["params_path"]))
    if artifact_name == "metadata":
        return Path(str(run_metadata["run_dir"])) / "metadata.json"
    raise ValueError(f"未知 LightGBM run artifact: {artifact_name}")
```

Import `build_trend_train_metadata` at module top from `fashion_trend.trend.training.outputs`; `outputs.py` only imports `run_artifacts` inside `derive_trend_model_output_paths()`, so this top-level import does not create a cycle.

- [ ] **Step 5: Add best-effort failure index update**

Add this helper to `run_artifacts.py` so promotion errors are not masked by a secondary index write failure:

```python
def record_lightgbm_promotion_failure(
    *,
    index_path: Path,
    summary: LightGBMRunSummary,
    run_dir: Path,
    stable_dir: Path,
    promotion_error: BaseException,
) -> None:
    try:
        upsert_lightgbm_run_index(index_path, summary)
    except Exception as index_error:
        log.error(
            "LightGBM promotion 失败，且 run index 更新失败: "
            f"run_dir={run_dir}, stable_dir={stable_dir}, "
            f"promotion_error={promotion_error}, index_error={index_error}",
            source="lightgbm-run-artifacts",
        )


def record_lightgbm_index_update_failure(
    *,
    run_dir: Path,
    stable_dir: Path,
    index_error: BaseException,
    attempted_status: str,
) -> None:
    log.error(
        "LightGBM promotion succeeded, but run index update failed: "
        f"run_dir={run_dir}, stable_dir={stable_dir}, "
        f"attempted_status={attempted_status}, index_error={index_error}",
        source="lightgbm-run-artifacts",
    )
```

- [ ] **Step 6: Call promotion from runner**

In `runner.py`, import the module:

```python
from fashion_trend.trend.training import run_artifacts
```

Then replace `_run_lightgbm_training()` helper tail after the initial `not_requested` index update. The previous Task 4 code returned immediately after writing the index; move `return metadata` to the end of this block so default LightGBM promotion cannot be skipped. Use module-qualified calls through `run_artifacts` for promotion helpers so tests can monkeypatch failure paths:

```python
    if promotion_requested:
        stable_paths = derive_trend_model_output_paths(model_name, output_root)
        try:
            run_artifacts.publish_lightgbm_run_to_stable(
                result=result,
                run_metadata=metadata,
                run_context=context,
                stable_paths=stable_paths,
            )
        except Exception as exc:
            run_artifacts.record_lightgbm_promotion_failure(
                index_path=run_paths["index"],
                summary=run_artifacts.build_lightgbm_run_summary(
                    run_id=resolved_run_id,
                    metadata=metadata,
                    promotion_status="failed",
                    promotion_error=str(exc),
                ),
                run_dir=run_paths["output_dir"],
                stable_dir=stable_paths["output_dir"],
                promotion_error=exc,
            )
            raise

        try:
            run_artifacts.upsert_lightgbm_run_index(
                run_paths["index"],
                run_artifacts.build_lightgbm_run_summary(
                    run_id=resolved_run_id,
                    metadata=metadata,
                    promotion_status="succeeded",
                ),
            )
        except Exception as exc:
            run_artifacts.record_lightgbm_index_update_failure(
                run_dir=run_paths["output_dir"],
                stable_dir=stable_paths["output_dir"],
                attempted_status="succeeded",
                index_error=exc,
            )
            raise
    return metadata
```

- [ ] **Step 7: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_lightgbm_outputs tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_writes_lightgbm_run_without_stable_promotion tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promotes_default_lightgbm_run tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promote_failure_keeps_run_and_returns_error tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promote_failure_preserves_original_error_when_index_update_fails tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promote_success_index_failure_does_not_mark_failed tests/test_trend_training.py::TestTrendTraining::test_write_promotion_items_atomic_rolls_back_cross_directory_partial_publish -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```sh
git add src/fashion_trend/foundation/io.py src/fashion_trend/trend/training/run_artifacts.py src/fashion_trend/trend/training/runner.py tests/test_trend_training.py
git commit -m "feat(trend): 写入并发布 LightGBM run"
```

---

### Task 6: Training CLI Options

**Files:**
- Modify: `src/10_train_trend_model.py`
- Modify: `tests/test_trend_training.py`

- [ ] **Step 1: Write failing CLI tests**

Append these tests to `tests/test_trend_training.py`:

```python
    def test_train_trend_model_main_passes_lightgbm_run_options(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        calls: list[dict[str, object]] = []
        original = train_model.run_trend_model_training

        def fake_run_trend_model_training(model_name: str, **kwargs) -> dict[str, object]:
            calls.append({"model_name": model_name, **kwargs})
            return {
                "model_name": "lightgbm",
                "model_type": MODEL_TYPE_SUPERVISED,
                "run_id": "depth6-lr005",
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": _sample_split_metadata(),
                "output_dir": "outputs/models/lightgbm/runs/depth6-lr005",
                "prediction_path": "outputs/models/lightgbm/runs/depth6-lr005/predictions.csv",
                "params_path": "outputs/models/lightgbm/runs/depth6-lr005/params.json",
            }

        try:
            train_model.run_trend_model_training = fake_run_trend_model_training
            exit_code = train_model.main(
                [
                    "--model",
                    "lightgbm",
                    "--run-id",
                    "depth6-lr005",
                    "--param",
                    "learning_rate=0.03",
                    "--no-promote",
                ]
            )
        finally:
            train_model.run_trend_model_training = original

        assert exit_code == 0
        assert calls[0]["model_name"] == "lightgbm"
        assert calls[0]["run_id"] == "depth6-lr005"
        assert calls[0]["promote"] is False
        assert "lightgbm_config" in calls[0]["trainer_options"]

    @pytest.mark.parametrize(
        "args",
        [
            ["--model", "last_week", "--run-id", "bad"],
            ["--model", "last_week", "--params", "configs/trend/lightgbm/depth6_lr005.json"],
            ["--model", "last_week", "--param", "learning_rate=0.03"],
            ["--model", "last_week", "--promote"],
            ["--model", "last_week", "--no-promote"],
            ["--model", "last_week", "--promote-run", "depth6-lr005"],
        ],
    )
    def test_train_trend_model_main_rejects_lightgbm_only_args_for_baseline(
        self,
        args: list[str],
    ) -> None:
        train_model = importlib.import_module("10_train_trend_model")

        assert train_model.main(args) == 2

    @pytest.mark.parametrize(
        "args",
        [
            ["--model", "lightgbm", "--promote", "--no-promote"],
            ["--model", "lightgbm", "--promote", "--promote-run", "depth6-lr005"],
            ["--model", "lightgbm", "--promote-run", "depth6-lr005", "--run-id", "other"],
            ["--model", "lightgbm", "--promote-run", "depth6-lr005", "--params", "configs/trend/lightgbm/depth6_lr005.json"],
            ["--model", "lightgbm", "--promote-run", "depth6-lr005", "--param", "learning_rate=0.03"],
        ],
    )
    def test_train_trend_model_main_rejects_invalid_promotion_combinations(
        self,
        args: list[str],
    ) -> None:
        train_model = importlib.import_module("10_train_trend_model")

        assert train_model.main(args) == 2

    def test_train_trend_model_main_promote_run_does_not_call_training_runner(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        original = train_model.run_trend_model_training
        calls: list[str] = []

        def fake_run_trend_model_training(model_name: str, **kwargs) -> dict[str, object]:
            calls.append(model_name)
            raise AssertionError("--promote-run must not train")

        try:
            train_model.run_trend_model_training = fake_run_trend_model_training
            exit_code = train_model.main(
                ["--model", "lightgbm", "--promote-run", "depth6-lr005"]
            )
        finally:
            train_model.run_trend_model_training = original

        assert exit_code == 1
        assert calls == []
```

Add helper near bottom:

```python
def _sample_split_metadata() -> dict[str, dict[str, int]]:
    return {
        "train": {"rows": 24, "weeks": 12, "attributes": 2, "week_min": 4, "week_max": 15},
        "valid": {"rows": 8, "weeks": 4, "attributes": 2, "week_min": 16, "week_max": 19},
        "test": {"rows": 8, "weeks": 4, "attributes": 2, "week_min": 20, "week_max": 23},
    }
```

- [ ] **Step 2: Run failing CLI tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_passes_lightgbm_run_options tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_rejects_lightgbm_only_args_for_baseline tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_rejects_invalid_promotion_combinations tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_promote_run_does_not_call_training_runner -q
```

Expected: FAIL because CLI args do not exist.

- [ ] **Step 3: Extend `parse_args()`**

Modify `src/10_train_trend_model.py`:

```python
from pathlib import Path

from fashion_trend.trend.models.supervised.lightgbm import LIGHTGBM_MODEL_NAME
from fashion_trend.trend.models.supervised.lightgbm_config import resolve_lightgbm_config
```

Add parser args:

```python
    parser.add_argument("--run-id", help="LightGBM run id。")
    parser.add_argument("--params", type=Path, help="LightGBM 参数 JSON 文件。")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="LightGBM 参数覆盖，格式为 key=value。",
    )
    promotion_group = parser.add_mutually_exclusive_group()
    promotion_group.add_argument("--promote", action="store_true", help="训练后发布到 stable。")
    promotion_group.add_argument("--no-promote", action="store_true", help="训练后不发布 stable。")
    promotion_group.add_argument("--promote-run", help="发布一个已评估 LightGBM run。")
```

After `args = parser.parse_args(argv)`, enforce usage errors:

```python
    lightgbm_only_used = any(
        [
            args.run_id,
            args.params,
            args.param,
            args.promote,
            args.no_promote,
            args.promote_run,
        ]
    )
    if args.model != LIGHTGBM_MODEL_NAME and lightgbm_only_used:
        parser.error("run、参数和 promotion 选项只支持 --model lightgbm。")
    if args.promote_run and (args.run_id or args.params or args.param):
        parser.error("--promote-run 不能与 --run-id、--params 或 --param 组合。")
```

- [ ] **Step 4: Branch promote-run before any training call**

In `main()`, handle `--promote-run` before resolving training config or calling `run_trend_model_training()`:

```python
        if args.promote_run:
            raise ValueError("--promote-run 会在已评估 run 发布任务中启用。")
```

For this intermediate commit, valid `--promote-run` input has passed parsing but the publish helper is added in Task 8. The important invariant is that this branch must return before any training runner call.

- [ ] **Step 5: Pass config and run options to runner**

After the `--promote-run` branch, build training config:

```python
        trainer_options = None
        if args.model == LIGHTGBM_MODEL_NAME and (args.params or args.param):
            trainer_options = {
                "lightgbm_config": resolve_lightgbm_config(
                    params_path=args.params,
                    cli_params=args.param,
                )
            }
        promote = None
        if args.promote:
            promote = True
        elif args.no_promote:
            promote = False
```

Call:

```python
        metadata = run_trend_model_training(
            args.model,
            run_id=args.run_id,
            trainer_options=trainer_options,
            promote=promote,
        )
```

Task 8 replaces this stub with the real function.

Update existing CLI test fakes in `tests/test_trend_training.py` so they accept the new keyword arguments:

```python
        def fake_run_trend_model_training(model_name: str, **kwargs) -> dict[str, object]:
            assert kwargs == {
                "run_id": None,
                "trainer_options": None,
                "promote": None,
            }
            calls.append(model_name)
            return {
                "model_name": LAST_WEEK_MODEL_NAME,
                "model_type": MODEL_TYPE_BASELINE,
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": _sample_split_metadata(),
                "output_dir": "outputs/models/last_week",
                "prediction_path": "outputs/models/last_week/predictions.csv",
                "params_path": "outputs/models/last_week/params.json",
            }
```

Apply the same fake signature pattern to the existing moving_average CLI test, with the existing moving_average return payload.

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_runs_training_and_logs_summary tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_passes_lightgbm_run_options tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_rejects_lightgbm_only_args_for_baseline tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_rejects_invalid_promotion_combinations tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_promote_run_does_not_call_training_runner -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```sh
git add src/10_train_trend_model.py tests/test_trend_training.py
git commit -m "feat(trend): 增加 LightGBM run 训练参数"
```

---

### Task 7: Run Evaluation And Evaluation Index

**Files:**
- Modify: `tests/test_trend_evaluation.py`
- Modify: `src/fashion_trend/trend/evaluation/payloads.py`
- Create: `src/fashion_trend/trend/evaluation/run_artifacts.py`
- Modify: `src/fashion_trend/trend/evaluation/runner.py`
- Modify: `src/fashion_trend/trend/evaluation/__init__.py`
- Modify: `src/11_eval_trend_model.py`

- [ ] **Step 1: Write failing run evaluation tests**

Append these tests to `tests/test_trend_evaluation.py`:

```python
    def test_derive_trend_metric_output_paths_uses_lightgbm_run_id(self) -> None:
        paths = derive_trend_metric_output_paths(
            "lightgbm",
            model_output_root=Path("outputs/models"),
            metrics_output_root=Path("outputs/metrics"),
            run_id="depth6-lr005",
        )

        assert paths["output_dir"] == Path("outputs/metrics/lightgbm/runs/depth6-lr005")
        assert paths["predictions"] == Path(
            "outputs/models/lightgbm/runs/depth6-lr005/predictions.csv"
        )
        assert paths["metrics"] == Path(
            "outputs/metrics/lightgbm/runs/depth6-lr005/trend_metrics.json"
        )
        assert paths["evaluations_index"] == Path(
            "outputs/metrics/lightgbm/runs/evaluations.jsonl"
        )

    def test_derive_trend_metric_output_paths_rejects_run_id_for_baseline(self) -> None:
        with pytest.raises(ValueError, match="lightgbm|run_id"):
            derive_trend_metric_output_paths(
                "last_week",
                model_output_root=Path("outputs/models"),
                metrics_output_root=Path("outputs/metrics"),
                run_id="baseline-run",
            )

    def test_run_trend_model_evaluation_writes_lightgbm_run_metrics_and_index(
        self,
        tmp_path: Path,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation().copy()
        predictions["model_name"] = "lightgbm"
        model_root = tmp_path / "outputs" / "models"
        metrics_root = tmp_path / "outputs" / "metrics"
        run_dir = model_root / "lightgbm" / "runs" / "depth6-lr005"
        write_csv_atomic(predictions, run_dir / "predictions.csv")
        write_json_atomic(
            {
                "model_name": "lightgbm",
                "run_id": "depth6-lr005",
                "prediction_path": str(run_dir / "predictions.csv"),
            },
            run_dir / "metadata.json",
        )

        payload = run_trend_model_evaluation(
            "lightgbm",
            model_output_root=model_root,
            metrics_output_root=metrics_root,
            run_id="depth6-lr005",
        )

        metrics_path = metrics_root / "lightgbm" / "runs" / "depth6-lr005" / "trend_metrics.json"
        index_path = metrics_root / "lightgbm" / "runs" / "evaluations.jsonl"
        assert metrics_path.exists()
        assert payload["run_id"] == "depth6-lr005"
        assert payload["prediction_path"] == str(run_dir / "predictions.csv")
        row = json.loads(index_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["run_id"] == "depth6-lr005"
        assert isinstance(row["evaluated_at"], str)
        assert row["evaluated_at"]
        assert row["selection_metrics"]["split"] == "valid"
        assert "ndcg_at_10" in row["selection_metrics"]
        assert set(row["report_metrics"]) == {"valid", "test"}
```

- [ ] **Step 2: Write failing evaluation CLI tests**

Append:

```python
    def test_eval_trend_model_main_passes_lightgbm_run_id(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")
        original = eval_model.run_trend_model_evaluation
        calls: list[dict[str, object]] = []

        def fake_run_trend_model_evaluation(model_name: str, **kwargs) -> dict[str, object]:
            calls.append({"model_name": model_name, **kwargs})
            return {
                "model_name": "lightgbm",
                "run_id": "depth6-lr005",
                "evaluated_splits": ["valid", "test"],
                "overall": {
                    "valid": {
                        "mae": 0.5,
                        "rmse": 0.7,
                        "spearman": 0.2,
                        "precision_at_k": {"10": 0.4},
                        "recall_at_k": {"10": 0.4},
                        "ndcg_at_k": {"10": 0.6},
                    },
                    "test": {
                        "mae": 0.6,
                        "rmse": 0.8,
                        "spearman": 0.3,
                        "precision_at_k": {"10": 0.5},
                        "recall_at_k": {"10": 0.5},
                        "ndcg_at_k": {"10": 0.7},
                    },
                },
                "groups": {"valid": {"ranking_groups": 4}, "test": {"ranking_groups": 4}},
                "output_path": "outputs/metrics/lightgbm/runs/depth6-lr005/trend_metrics.json",
            }

        try:
            eval_model.run_trend_model_evaluation = fake_run_trend_model_evaluation
            exit_code = eval_model.main(["--model", "lightgbm", "--run-id", "depth6-lr005"])
        finally:
            eval_model.run_trend_model_evaluation = original

        assert exit_code == 0
        assert calls == [{"model_name": "lightgbm", "run_id": "depth6-lr005"}]

    def test_eval_trend_model_main_rejects_run_id_for_baseline(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")

        assert eval_model.main(["--model", "last_week", "--run-id", "bad"]) == 2
```

- [ ] **Step 3: Run failing tests**

Run:

```sh
uv run pytest tests/test_trend_evaluation.py::TestTrendEvaluation::test_derive_trend_metric_output_paths_uses_lightgbm_run_id tests/test_trend_evaluation.py::TestTrendEvaluation::test_derive_trend_metric_output_paths_rejects_run_id_for_baseline tests/test_trend_evaluation.py::TestTrendEvaluation::test_run_trend_model_evaluation_writes_lightgbm_run_metrics_and_index tests/test_trend_evaluation.py::TestTrendEvaluation::test_eval_trend_model_main_passes_lightgbm_run_id tests/test_trend_evaluation.py::TestTrendEvaluation::test_eval_trend_model_main_rejects_run_id_for_baseline -q
```

Expected: FAIL because run evaluation is not implemented.

- [ ] **Step 4: Extend metric path derivation and payload**

Modify `derive_trend_metric_output_paths()` in `payloads.py`:

```python
def derive_trend_metric_output_paths(
    model_name: str,
    model_output_root: Path = OUTPUT_MODELS_DIR,
    metrics_output_root: Path = OUTPUT_METRICS_DIR,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    validate_safe_path_segment(model_name, "model_name")
    prediction_dir = model_output_root / model_name
    output_dir = metrics_output_root / model_name
    if run_id is None:
        validate_output_parent_dirs(prediction_dir, model_output_root)
        validate_output_parent_dirs(output_dir, metrics_output_root)
        return {
            "output_dir": output_dir,
            "predictions": prediction_dir / "predictions.csv",
            "metrics": output_dir / "trend_metrics.json",
        }

    from fashion_trend.trend.training.run_artifacts import validate_lightgbm_run_id

    if model_name != "lightgbm":
        raise ValueError("只有 lightgbm 支持 run_id。")
    validate_lightgbm_run_id(run_id)
    run_prediction_dir = prediction_dir / "runs" / run_id
    run_output_dir = output_dir / "runs" / run_id
    validate_output_parent_dirs(run_prediction_dir, model_output_root)
    validate_output_parent_dirs(run_output_dir, metrics_output_root)
    return {
        "output_dir": run_output_dir,
        "predictions": run_prediction_dir / "predictions.csv",
        "metrics": run_output_dir / "trend_metrics.json",
        "evaluations_index": output_dir / "runs" / "evaluations.jsonl",
    }
```

Modify `build_trend_metrics_payload()`:

```python
def build_trend_metrics_payload(
    predictions: pd.DataFrame,
    model_name: str,
    prediction_path: Path,
    output_path: Path,
    k_values: Sequence[int] = TREND_EVALUATION_K_VALUES,
    *,
    run_id: str | None = None,
) -> dict[str, object]:
```

Include:

```python
        "run_id": run_id,
```

- [ ] **Step 5: Add evaluation index helpers**

Create `src/fashion_trend/trend/evaluation/run_artifacts.py`:

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fashion_trend.foundation.io import write_text_atomic


def read_run_id_from_model_metadata(metadata_path: Path) -> str | None:
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    return str(run_id) if run_id is not None else None


def build_lightgbm_evaluation_summary(
    *,
    run_id: str,
    metrics_path: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    overall = payload["overall"]
    valid = overall["valid"]
    return {
        "run_id": run_id,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metrics_path": str(metrics_path),
        "selection_metrics": {
            "split": "valid",
            "ndcg_at_10": valid["ndcg_at_k"]["10"],
            "spearman": valid["spearman"],
            "mae": valid["mae"],
            "rmse": valid["rmse"],
        },
        "report_metrics": {
            "valid": overall["valid"],
            "test": overall["test"],
        },
    }


def upsert_lightgbm_evaluation_index(
    index_path: Path,
    summary: dict[str, object],
) -> None:
    summaries: dict[str, dict[str, object]] = {}
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                summaries[str(payload["run_id"])] = payload
    summaries[str(summary["run_id"])] = summary
    lines = [
        json.dumps(summaries[key], ensure_ascii=False, sort_keys=True)
        for key in sorted(summaries)
    ]
    write_text_atomic("\n".join(lines) + "\n", index_path)
```

- [ ] **Step 6: Extend evaluation runner**

Modify `run_trend_model_evaluation()` signature:

```python
def run_trend_model_evaluation(
    model_name: str,
    model_output_root: Path = OUTPUT_MODELS_DIR,
    metrics_output_root: Path = OUTPUT_METRICS_DIR,
    *,
    run_id: str | None = None,
) -> dict[str, object]:
```

Add baseline guard:

```python
    if run_id is not None and model_name != "lightgbm":
        raise ValueError("--run-id 只支持 --model lightgbm。")
```

Derive output paths with the run id before reading predictions:

```python
    output_paths = derive_trend_metric_output_paths(
        model_name,
        model_output_root,
        metrics_output_root,
        run_id=run_id,
    )
```

After deriving output paths:

```python
    metadata_run_id = read_run_id_from_model_metadata(
        output_paths["predictions"].parent / "metadata.json"
    )
    resolved_run_id = run_id if run_id is not None else metadata_run_id
```

Pass `run_id=resolved_run_id` into `build_trend_metrics_payload()`. After `write_trend_metrics()`, if `run_id is not None`, call:

```python
        upsert_lightgbm_evaluation_index(
            output_paths["evaluations_index"],
            build_lightgbm_evaluation_summary(
                run_id=run_id,
                metrics_path=output_paths["metrics"],
                payload=payload,
            ),
        )
```

- [ ] **Step 7: Extend evaluation CLI**

In `src/11_eval_trend_model.py`, add `--run-id` and baseline guard using `parser.error`:

```python
    parser.add_argument("--run-id", help="需要评价的 LightGBM run id。")
    args = parser.parse_args(argv)
    if args.run_id and args.model != "lightgbm":
        parser.error("--run-id 只支持 --model lightgbm。")
```

Call:

```python
        metrics = run_trend_model_evaluation(args.model, run_id=args.run_id)
```

Update the existing `test_eval_trend_model_main_runs_evaluation_and_logs_summary()` fake to accept the new keyword argument:

```python
        def fake_run_trend_model_evaluation(model_name: str, **kwargs) -> dict[str, object]:
            assert model_name == "last_week"
            assert kwargs == {"run_id": None}
            return {
                "model_name": "last_week",
                "evaluated_splits": ["valid", "test"],
                "overall": {
                    "valid": {
                        "mae": 0.5,
                        "rmse": 0.7,
                        "spearman": 0.2,
                        "precision_at_k": {"10": 0.4},
                        "recall_at_k": {"10": 0.4},
                        "ndcg_at_k": {"10": 0.6},
                    },
                    "test": {
                        "mae": 0.6,
                        "rmse": 0.8,
                        "spearman": 0.3,
                        "precision_at_k": {"10": 0.5},
                        "recall_at_k": {"10": 0.5},
                        "ndcg_at_k": {"10": 0.7},
                    },
                },
                "groups": {
                    "valid": {"ranking_groups": 4},
                    "test": {"ranking_groups": 4},
                },
                "output_path": "outputs/metrics/last_week/trend_metrics.json",
            }
```

- [ ] **Step 8: Export helper functions**

Update `src/fashion_trend/trend/evaluation/__init__.py` exports for:

```python
build_lightgbm_evaluation_summary
read_run_id_from_model_metadata
upsert_lightgbm_evaluation_index
```

- [ ] **Step 9: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_evaluation.py::TestTrendEvaluation::test_derive_trend_metric_output_paths_uses_model_name tests/test_trend_evaluation.py::TestTrendEvaluation::test_derive_trend_metric_output_paths_uses_lightgbm_run_id tests/test_trend_evaluation.py::TestTrendEvaluation::test_derive_trend_metric_output_paths_rejects_run_id_for_baseline tests/test_trend_evaluation.py::TestTrendEvaluation::test_run_trend_model_evaluation_writes_lightgbm_run_metrics_and_index tests/test_trend_evaluation.py::TestTrendEvaluation::test_eval_trend_model_main_passes_lightgbm_run_id tests/test_trend_evaluation.py::TestTrendEvaluation::test_eval_trend_model_main_rejects_run_id_for_baseline -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```sh
git add src/fashion_trend/trend/evaluation/payloads.py src/fashion_trend/trend/evaluation/run_artifacts.py src/fashion_trend/trend/evaluation/runner.py src/fashion_trend/trend/evaluation/__init__.py src/11_eval_trend_model.py tests/test_trend_evaluation.py
git commit -m "feat(trend): 支持 LightGBM run 评价"
```

---

### Task 8: Promote Existing Evaluated Run

**Files:**
- Modify: `tests/test_trend_training.py`
- Modify: `tests/test_trend_evaluation.py`
- Modify: `src/fashion_trend/trend/training/run_artifacts.py`
- Modify: `src/10_train_trend_model.py`

- [ ] **Step 1: Write failing promote-run tests**

Append this test to `tests/test_trend_training.py`:

```python
    def test_promote_existing_lightgbm_run_publishes_model_and_metrics(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import promote_existing_lightgbm_run

        model_root = tmp_path / "outputs" / "models"
        metrics_root = tmp_path / "outputs" / "metrics"
        run_dir = model_root / "lightgbm" / "runs" / "depth6-lr005"
        run_metrics_dir = metrics_root / "lightgbm" / "runs" / "depth6-lr005"
        predictions = sample_trend_predictions_for_evaluation().copy()
        predictions["model_name"] = "lightgbm"
        write_csv_atomic(predictions, run_dir / "predictions.csv")
        write_json_atomic({"lightgbm_params": {"learning_rate": 0.03}}, run_dir / "params.json")
        write_json_atomic(
            {
                "model_name": "lightgbm",
                "model_type": "supervised",
                "run_id": "depth6-lr005",
                "run_dir": str(run_dir),
                "output_dir": str(run_dir),
                "prediction_path": str(run_dir / "predictions.csv"),
                "params_path": str(run_dir / "params.json"),
                "rows": len(predictions),
                "weeks": 5,
                "attributes": 5,
                "splits": _sample_split_metadata(),
                "extra_artifacts": [
                    {"path": "feature_importance.csv", "kind": "csv"},
                    {"path": "model.txt", "kind": "binary"},
                ],
            },
            run_dir / "metadata.json",
        )
        write_csv_atomic(pd.DataFrame({"feature": ["growth_lag_1"]}), run_dir / "feature_importance.csv")
        (run_dir / "model.txt").write_text("fake model", encoding="utf-8")
        write_json_atomic(
            {
                "model_name": "lightgbm",
                "run_id": "depth6-lr005",
                "prediction_path": str(run_dir / "predictions.csv"),
                "output_path": str(run_metrics_dir / "trend_metrics.json"),
                "evaluated_splits": ["valid", "test"],
                "overall": {
                    "valid": {
                        "mae": 0.5,
                        "rmse": 0.7,
                        "spearman": 0.2,
                        "precision_at_k": {"10": 0.4},
                        "recall_at_k": {"10": 0.4},
                        "ndcg_at_k": {"10": 0.6},
                    },
                    "test": {
                        "mae": 0.6,
                        "rmse": 0.8,
                        "spearman": 0.3,
                        "precision_at_k": {"10": 0.5},
                        "recall_at_k": {"10": 0.5},
                        "ndcg_at_k": {"10": 0.7},
                    },
                },
                "groups": {"valid": {"ranking_groups": 4}, "test": {"ranking_groups": 4}},
            },
            run_metrics_dir / "trend_metrics.json",
        )

        stable_metadata = promote_existing_lightgbm_run(
            "depth6-lr005",
            model_output_root=model_root,
            metrics_output_root=metrics_root,
        )

        stable_model_dir = model_root / "lightgbm"
        stable_metrics_path = metrics_root / "lightgbm" / "trend_metrics.json"
        assert (stable_model_dir / "predictions.csv").exists()
        assert (stable_model_dir / "params.json").exists()
        assert (stable_model_dir / "metadata.json").exists()
        assert (stable_model_dir / "feature_importance.csv").exists()
        assert (stable_model_dir / "model.txt").exists()
        assert stable_metrics_path.exists()
        assert stable_metadata["run_id"] == "depth6-lr005"
        assert stable_metadata["promotion_requested"] is True
        assert stable_metadata["promotion_mode"] == "promote_run"
        stable_metrics = json.loads(stable_metrics_path.read_text(encoding="utf-8"))
        assert stable_metrics["run_id"] == "depth6-lr005"
        assert stable_metrics["prediction_path"] == str(stable_model_dir / "predictions.csv")
        assert stable_metrics["output_path"] == str(stable_metrics_path)

    def test_promote_existing_lightgbm_run_success_index_failure_does_not_mark_failed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from fashion_trend.trend.training import run_artifacts
        from fashion_trend.trend.training.run_artifacts import promote_existing_lightgbm_run

        model_root = tmp_path / "outputs" / "models"
        metrics_root = tmp_path / "outputs" / "metrics"
        run_dir = model_root / "lightgbm" / "runs" / "depth6-lr005"
        run_metrics_dir = metrics_root / "lightgbm" / "runs" / "depth6-lr005"
        run_dir.mkdir(parents=True)
        run_metrics_dir.mkdir(parents=True)
        (run_dir / "predictions.csv").write_text("predictions\n", encoding="utf-8")
        (run_dir / "params.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "feature_importance.csv").write_text("feature\n", encoding="utf-8")
        (run_dir / "model.txt").write_text("fake model\n", encoding="utf-8")
        write_json_atomic(
            {
                "model_name": "lightgbm",
                "model_type": "supervised",
                "run_id": "depth6-lr005",
                "run_dir": str(run_dir),
                "output_dir": str(run_dir),
                "prediction_path": str(run_dir / "predictions.csv"),
                "params_path": str(run_dir / "params.json"),
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": _sample_split_metadata(),
                "extra_artifacts": [
                    {"path": "feature_importance.csv", "kind": "csv"},
                    {"path": "model.txt", "kind": "binary"},
                ],
                "promotion_requested": False,
            },
            run_dir / "metadata.json",
        )
        write_json_atomic(
            {
                "model_name": "lightgbm",
                "run_id": "depth6-lr005",
                "prediction_path": str(run_dir / "predictions.csv"),
                "output_path": str(run_metrics_dir / "trend_metrics.json"),
                "evaluated_splits": ["valid", "test"],
                "overall": {"valid": {}, "test": {}},
            },
            run_metrics_dir / "trend_metrics.json",
        )

        def successful_publish(items, staging_root):
            item = items[0]
            item.final_path.parent.mkdir(parents=True, exist_ok=True)
            item.final_path.write_bytes(item.payload)

        monkeypatch.setattr(run_artifacts, "write_promotion_items_atomic", successful_publish)
        monkeypatch.setattr(
            run_artifacts,
            "upsert_lightgbm_run_index",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("succeeded index write failed")),
        )

        with pytest.raises(OSError, match="succeeded index write failed"):
            promote_existing_lightgbm_run(
                "depth6-lr005",
                model_output_root=model_root,
                metrics_output_root=metrics_root,
            )

        assert (model_root / "lightgbm" / "predictions.csv").exists()
        assert not (model_root / "lightgbm" / "runs" / "index.jsonl").exists()
        captured = capsys.readouterr()
        assert "promotion succeeded" in captured.err
        assert "succeeded index write failed" in captured.err
```

Append this consistency test to `tests/test_trend_evaluation.py`:

```python
    def test_validate_lightgbm_run_metrics_payload_rejects_mismatched_run(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.evaluation.run_artifacts import validate_lightgbm_run_metrics_payload

        run_dir = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005"
        payload = {
            "model_name": "lightgbm",
            "run_id": "other",
            "prediction_path": str(run_dir / "predictions.csv"),
        }

        with pytest.raises(ValueError, match="run_id"):
            validate_lightgbm_run_metrics_payload(
                payload,
                run_id="depth6-lr005",
                prediction_path=run_dir / "predictions.csv",
            )

    def test_validate_lightgbm_run_metadata_payload_rejects_mismatched_paths(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import (
            validate_lightgbm_run_metadata_payload,
        )
        from fashion_trend.trend.training import derive_trend_model_output_paths

        run_paths = derive_trend_model_output_paths(
            "lightgbm",
            tmp_path / "outputs" / "models",
            run_id="depth6-lr005",
        )
        payload = {
            "model_name": "lightgbm",
            "run_id": "depth6-lr005",
            "output_dir": str(run_paths["output_dir"]),
            "run_dir": str(run_paths["output_dir"]),
            "prediction_path": str(tmp_path / "wrong" / "predictions.csv"),
            "params_path": str(run_paths["params"]),
        }

        with pytest.raises(ValueError, match="prediction_path"):
            validate_lightgbm_run_metadata_payload(
                payload,
                run_id="depth6-lr005",
                run_paths=run_paths,
            )
```

- [ ] **Step 2: Run failing tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_promote_existing_lightgbm_run_publishes_model_and_metrics tests/test_trend_training.py::TestTrendTraining::test_promote_existing_lightgbm_run_success_index_failure_does_not_mark_failed tests/test_trend_evaluation.py::TestTrendEvaluation::test_validate_lightgbm_run_metrics_payload_rejects_mismatched_run tests/test_trend_evaluation.py::TestTrendEvaluation::test_validate_lightgbm_run_metadata_payload_rejects_mismatched_paths -q
```

Expected: FAIL because promote-run helpers do not exist.

- [ ] **Step 3: Add run metrics validation and stable metrics conversion**

In `src/fashion_trend/trend/evaluation/run_artifacts.py`, add:

```python
def validate_lightgbm_run_metrics_payload(
    payload: dict[str, object],
    *,
    run_id: str,
    prediction_path: Path,
) -> None:
    if payload.get("model_name") != "lightgbm":
        raise ValueError("LightGBM run metrics 的 model_name 必须是 lightgbm。")
    if payload.get("run_id") != run_id:
        raise ValueError(f"LightGBM run metrics 的 run_id 不匹配: {payload.get('run_id')}")
    if payload.get("prediction_path") != str(prediction_path):
        raise ValueError("LightGBM run metrics 的 prediction_path 不指向当前 run。")


def build_stable_metrics_payload(
    payload: dict[str, object],
    *,
    stable_prediction_path: Path,
    stable_metrics_path: Path,
) -> dict[str, object]:
    stable_payload = dict(payload)
    stable_payload["prediction_path"] = str(stable_prediction_path)
    stable_payload["output_path"] = str(stable_metrics_path)
    return stable_payload
```

- [ ] **Step 4: Add promote existing run helper**

In `src/fashion_trend/trend/training/run_artifacts.py`, first add a metadata validator:

```python
def validate_lightgbm_run_metadata_payload(
    payload: dict[str, object],
    *,
    run_id: str,
    run_paths: dict[str, Path],
) -> None:
    expected_values = {
        "model_name": "lightgbm",
        "run_id": run_id,
        "output_dir": str(run_paths["output_dir"]),
        "run_dir": str(run_paths["output_dir"]),
        "prediction_path": str(run_paths["predictions"]),
        "params_path": str(run_paths["params"]),
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"LightGBM run metadata 的 {key} 不匹配: "
                f"expected={expected}, actual={payload.get(key)}"
            )
```

Then add:

```python
def promote_existing_lightgbm_run(
    run_id: str,
    *,
    model_output_root: Path,
    metrics_output_root: Path,
) -> dict[str, object]:
    validate_lightgbm_run_id(run_id)
    run_paths = derive_trend_model_output_paths(
        "lightgbm",
        model_output_root,
        run_id=run_id,
    )
    stable_paths = derive_trend_model_output_paths("lightgbm", model_output_root)
    run_metrics_path = (
        metrics_output_root / "lightgbm" / "runs" / run_id / "trend_metrics.json"
    )
    stable_metrics_path = metrics_output_root / "lightgbm" / "trend_metrics.json"
    _validate_existing_lightgbm_run_files(run_paths, run_metrics_path)
    run_metadata = _read_json(run_paths["metadata"])
    run_metrics = _read_json(run_metrics_path)
    validate_lightgbm_run_metadata_payload(
        run_metadata,
        run_id=run_id,
        run_paths=run_paths,
    )

    from fashion_trend.trend.evaluation.run_artifacts import (
        build_stable_metrics_payload,
        validate_lightgbm_run_metrics_payload,
    )

    validate_lightgbm_run_metrics_payload(
        run_metrics,
        run_id=run_id,
        prediction_path=run_paths["predictions"],
    )
    stable_metadata = dict(run_metadata)
    stable_metadata["output_dir"] = str(stable_paths["output_dir"])
    stable_metadata["prediction_path"] = str(stable_paths["predictions"])
    stable_metadata["params_path"] = str(stable_paths["params"])
    stable_metadata["stable_output_dir"] = str(stable_paths["output_dir"])
    stable_metadata["run_dir"] = str(run_paths["output_dir"])
    stable_metadata["promotion_requested"] = True
    stable_metadata["promotion_mode"] = "promote_run"
    stable_metrics = build_stable_metrics_payload(
        run_metrics,
        stable_prediction_path=stable_paths["predictions"],
        stable_metrics_path=stable_metrics_path,
    )
    items = [
        PromotionItem(stable_paths["predictions"], run_paths["predictions"].read_bytes()),
        PromotionItem(stable_paths["params"], run_paths["params"].read_bytes()),
        PromotionItem(stable_paths["metadata"], stable_metadata),
        PromotionItem(stable_paths["output_dir"] / "feature_importance.csv", (run_paths["output_dir"] / "feature_importance.csv").read_bytes()),
        PromotionItem(stable_paths["output_dir"] / "model.txt", (run_paths["output_dir"] / "model.txt").read_bytes()),
        PromotionItem(stable_metrics_path, stable_metrics),
    ]
    try:
        write_promotion_items_atomic(items, stable_paths["output_dir"])
    except Exception as exc:
        record_lightgbm_promotion_failure(
            index_path=run_paths["index"],
            summary=LightGBMRunSummary(
                run_id=run_id,
                created_at=str(run_metadata.get("created_at", "")),
                run_dir=str(run_paths["output_dir"]),
                promotion_status="failed",
                params_path=str(run_paths["params"]),
                metadata_path=str(run_paths["metadata"]),
                promotion_error=str(exc),
            ),
            run_dir=run_paths["output_dir"],
            stable_dir=stable_paths["output_dir"],
            promotion_error=exc,
        )
        raise

    try:
        upsert_lightgbm_run_index(
            run_paths["index"],
            LightGBMRunSummary(
                run_id=run_id,
                created_at=str(run_metadata.get("created_at", "")),
                run_dir=str(run_paths["output_dir"]),
                promotion_status="succeeded",
                params_path=str(run_paths["params"]),
                metadata_path=str(run_paths["metadata"]),
            ),
        )
    except Exception as exc:
        record_lightgbm_index_update_failure(
            run_dir=run_paths["output_dir"],
            stable_dir=stable_paths["output_dir"],
            attempted_status="succeeded",
            index_error=exc,
        )
        raise
    return stable_metadata
```

Add private helpers:

```python
def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_existing_lightgbm_run_files(
    run_paths: dict[str, Path],
    run_metrics_path: Path,
) -> None:
    required_paths = [
        run_paths["predictions"],
        run_paths["params"],
        run_paths["metadata"],
        run_paths["output_dir"] / "feature_importance.csv",
        run_paths["output_dir"] / "model.txt",
        run_metrics_path,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"LightGBM promote-run 缺少产物: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"LightGBM promote-run 产物是目录: {path}")
```

- [ ] **Step 5: Wire `--promote-run` CLI**

In `src/10_train_trend_model.py`, replace the temporary `--promote-run` error with:

```python
        if args.promote_run:
            from fashion_trend.trend.training.run_artifacts import promote_existing_lightgbm_run

            metadata = promote_existing_lightgbm_run(
                args.promote_run,
                model_output_root=OUTPUT_MODELS_DIR,
                metrics_output_root=OUTPUT_METRICS_DIR,
            )
        else:
            trainer_options = None
            if args.params or args.param:
                trainer_options = {
                    "lightgbm_config": resolve_lightgbm_config(
                        params_path=args.params,
                        cli_params=args.param,
                    )
                }
            promote = None
            if args.promote:
                promote = True
            elif args.no_promote:
                promote = False
            metadata = run_trend_model_training(
                args.model,
                run_id=args.run_id,
                trainer_options=trainer_options,
                promote=promote,
            )
```

Add `OUTPUT_METRICS_DIR` import from `fashion_trend.trend.paths`.

Update `test_train_trend_model_main_promote_run_does_not_call_training_runner()` from Task 6 so the final behavior expects the promote helper to run and the training runner to remain unused:

```python
    def test_train_trend_model_main_promote_run_does_not_call_training_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        from fashion_trend.trend.training import run_artifacts

        training_calls: list[str] = []
        promote_calls: list[str] = []

        def fake_run_trend_model_training(model_name: str, **kwargs) -> dict[str, object]:
            training_calls.append(model_name)
            raise AssertionError("--promote-run must not train")

        def fake_promote_existing_lightgbm_run(run_id: str, **kwargs) -> dict[str, object]:
            promote_calls.append(run_id)
            return {
                "model_name": "lightgbm",
                "model_type": MODEL_TYPE_SUPERVISED,
                "run_id": run_id,
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": _sample_split_metadata(),
                "output_dir": "outputs/models/lightgbm",
                "prediction_path": "outputs/models/lightgbm/predictions.csv",
                "params_path": "outputs/models/lightgbm/params.json",
            }

        monkeypatch.setattr(train_model, "run_trend_model_training", fake_run_trend_model_training)
        monkeypatch.setattr(run_artifacts, "promote_existing_lightgbm_run", fake_promote_existing_lightgbm_run)

        assert train_model.main(["--model", "lightgbm", "--promote-run", "depth6-lr005"]) == 0
        assert training_calls == []
        assert promote_calls == ["depth6-lr005"]
```

- [ ] **Step 6: Run focused tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_promote_existing_lightgbm_run_publishes_model_and_metrics tests/test_trend_training.py::TestTrendTraining::test_promote_existing_lightgbm_run_success_index_failure_does_not_mark_failed tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_promote_run_does_not_call_training_runner tests/test_trend_evaluation.py::TestTrendEvaluation::test_validate_lightgbm_run_metrics_payload_rejects_mismatched_run tests/test_trend_evaluation.py::TestTrendEvaluation::test_validate_lightgbm_run_metadata_payload_rejects_mismatched_paths -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```sh
git add src/fashion_trend/trend/training/run_artifacts.py src/fashion_trend/trend/evaluation/run_artifacts.py src/10_train_trend_model.py tests/test_trend_training.py tests/test_trend_evaluation.py
git commit -m "feat(trend): 支持发布已评估 LightGBM run"
```

---

### Task 9: Docs And End-To-End Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`
- Run: focused tests, full tests, LightGBM smoke commands

- [ ] **Step 1: Update README**

Update the LightGBM section in `README.md` to include these exact command examples:

```sh
uv run python src/10_train_trend_model.py --model lightgbm --run-id smoke-lightgbm --no-promote
uv run python src/11_eval_trend_model.py --model lightgbm --run-id smoke-lightgbm
uv run python src/10_train_trend_model.py --model lightgbm --promote-run smoke-lightgbm
```

Document these paths in the same section:

```text
outputs/models/lightgbm/runs/<run_id>/predictions.csv
outputs/models/lightgbm/runs/<run_id>/params.json
outputs/models/lightgbm/runs/<run_id>/metadata.json
outputs/models/lightgbm/runs/<run_id>/feature_importance.csv
outputs/models/lightgbm/runs/<run_id>/model.txt
outputs/models/lightgbm/runs/index.jsonl
outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json
outputs/metrics/lightgbm/runs/evaluations.jsonl
```

Also state:

```text
调参选择只能读取 evaluations.jsonl 的 selection_metrics；test 指标只用于最终选中 run 的一次性报告。
本轮默认 LightGBM 参数有意新增 subsample_freq=1，因此默认训练结果需要重新跑训练、评价和 baseline 对比后再更新摘要。
```

- [ ] **Step 2: Update implementation plan document**

In `docs/gpt-research/implementation-plan.md`, update the training/evaluation contract table so LightGBM rows mention:

```text
LightGBM 调参 run: outputs/models/lightgbm/runs/<run_id>/...
LightGBM run 评价: outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json
已评估 run 发布: 10_train_trend_model.py --model lightgbm --promote-run <run_id>
```

- [ ] **Step 3: Run focused unit tests**

Run:

```sh
uv run pytest tests/test_trend_training.py tests/test_trend_lightgbm.py tests/test_trend_evaluation.py
```

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run:

```sh
uv run pytest
```

Expected: PASS.

- [ ] **Step 5: Re-run stable baselines and default LightGBM**

Run the three baseline train/eval pairs and default LightGBM train/eval after `subsample_freq=1` is implemented:

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model moving_average
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model lightgbm
```

Expected:

```text
outputs/models/last_week/predictions.csv exists
outputs/metrics/last_week/trend_metrics.json exists
outputs/models/previous_growth/predictions.csv exists
outputs/metrics/previous_growth/trend_metrics.json exists
outputs/models/moving_average/predictions.csv exists
outputs/metrics/moving_average/trend_metrics.json exists
outputs/models/lightgbm/predictions.csv exists
outputs/metrics/lightgbm/trend_metrics.json exists
outputs/models/lightgbm/metadata.json has a non-empty run_id
outputs/models/lightgbm/params.json lightgbm_params.subsample_freq is 1
```

- [ ] **Step 6: Print baseline comparison summary**

Run:

```sh
uv run python -c 'import json; from pathlib import Path; models=["last_week","previous_growth","moving_average","lightgbm"]; print("model,run_id,valid_ndcg@10,valid_spearman,valid_mae,valid_rmse,test_ndcg@10,test_spearman,test_mae,test_rmse"); [print(",".join([model, str((payload:=json.loads((Path("outputs/metrics")/model/"trend_metrics.json").read_text(encoding="utf-8"))).get("run_id")), str(payload["overall"]["valid"]["ndcg_at_k"]["10"]), str(payload["overall"]["valid"]["spearman"]), str(payload["overall"]["valid"]["mae"]), str(payload["overall"]["valid"]["rmse"]), str(payload["overall"]["test"]["ndcg_at_k"]["10"]), str(payload["overall"]["test"]["spearman"]), str(payload["overall"]["test"]["mae"]), str(payload["overall"]["test"]["rmse"])])) for model in models]'
```

Expected: one CSV-like table with four rows: `last_week`, `previous_growth`, `moving_average`, `lightgbm`. Use valid metrics for selection discussion; test columns are report-only and must not be used to select tuning candidates.

- [ ] **Step 7: Run LightGBM smoke without promotion**

Choose a new run id if `outputs/models/lightgbm/runs/smoke-lightgbm` already exists.

Run:

```sh
uv run python src/10_train_trend_model.py --model lightgbm --run-id smoke-lightgbm --no-promote
```

Expected:

```text
outputs/models/lightgbm/runs/smoke-lightgbm/predictions.csv exists
outputs/models/lightgbm/runs/smoke-lightgbm/params.json exists
outputs/models/lightgbm/runs/smoke-lightgbm/metadata.json exists
outputs/models/lightgbm/runs/smoke-lightgbm/feature_importance.csv exists
outputs/models/lightgbm/runs/smoke-lightgbm/model.txt exists
outputs/models/lightgbm/runs/index.jsonl has promotion_status not_requested for smoke-lightgbm
```

- [ ] **Step 8: Run run evaluation smoke**

Run:

```sh
uv run python src/11_eval_trend_model.py --model lightgbm --run-id smoke-lightgbm
```

Expected:

```text
outputs/metrics/lightgbm/runs/smoke-lightgbm/trend_metrics.json exists
outputs/metrics/lightgbm/runs/evaluations.jsonl has selection_metrics from valid split
```

- [ ] **Step 9: Promote evaluated run smoke**

Run:

```sh
uv run python src/10_train_trend_model.py --model lightgbm --promote-run smoke-lightgbm
```

Expected:

```text
outputs/models/lightgbm/metadata.json has run_id smoke-lightgbm
outputs/models/lightgbm/predictions.csv exists
outputs/metrics/lightgbm/trend_metrics.json has run_id smoke-lightgbm
outputs/metrics/lightgbm/trend_metrics.json prediction_path is outputs/models/lightgbm/predictions.csv
outputs/models/lightgbm/runs/index.jsonl has promotion_status succeeded for smoke-lightgbm
```

- [ ] **Step 10: Inspect changed files**

Run:

```sh
git diff --stat
git diff -- src/fashion_trend/foundation/io.py src/fashion_trend/trend/models/base.py src/fashion_trend/trend/models/supervised/lightgbm.py src/fashion_trend/trend/models/supervised/lightgbm_config.py src/fashion_trend/trend/training/outputs.py src/fashion_trend/trend/training/run_artifacts.py src/fashion_trend/trend/training/runner.py src/fashion_trend/trend/evaluation/payloads.py src/fashion_trend/trend/evaluation/run_artifacts.py src/fashion_trend/trend/evaluation/runner.py src/10_train_trend_model.py src/11_eval_trend_model.py
```

Expected: diff only covers LightGBM run artifacts, parameter schema, evaluation run support, CLI, tests, and docs.

- [ ] **Step 11: Commit**

```sh
git add README.md docs/gpt-research/implementation-plan.md
git commit -m "docs: 说明 LightGBM run 调参流程"
```

---

## Self-Review Notes

- Spec coverage:
  - Run directory, stable directory, reserved run ids, auto run id, no overwrite, run/stable metadata separation: Tasks 1, 3, 4, 5.
  - Parameter schema, `--params`, `--param`, default `subsample_freq: 1`, and CLI `--params` rejection/互斥: Tasks 2, 3, 6.
  - Default promote vs experiment no-promote, including manual `run_id` and custom config default no-promote: Tasks 4, 5, 6, 9.
  - Run evaluation, `run_id` metrics payload, `selection_metrics` vs `report_metrics`: Task 7.
  - Evaluation JSONL writing uses public `foundation.io.write_text_atomic()` rather than a training-private helper: Tasks 4 and 7.
  - `evaluations.jsonl` evaluation timestamp: Task 7.
  - `--promote-run` with stable metrics, stable metadata `promotion_requested=True`, `promotion_mode="promote_run"`, plus run metadata and run metrics consistency checks: Task 8.
  - `--promote-run` never retrains: Tasks 6 and 8.
  - Rollback across stable model and metrics, including partial publish failure: Tasks 5 and 8 use one `write_promotion_items_atomic()` for all final files.
  - Promotion failure index update is best effort and cannot mask the original promotion error; success-index failure after stable publish is reported without writing `promotion_status=failed`: Task 5 and Task 8.
  - Baseline rejects run/parameter/promotion args: Tasks 4, 6, 7.
  - README and implementation plan sync, smoke validation, default LightGBM rerun, and three-baseline comparison: Task 9.
- Placeholder scan:
  - The plan contains concrete paths, function names, command lines, expected outcomes, and code snippets for every implementation step.
  - No placeholder markers or unnamed edge-case steps are left for the implementer.
- Type consistency:
  - `run_id` is always `str | None`.
  - `trainer_options` is `Mapping[str, object]`.
  - `LightGBMTrainingConfig` is passed through `trainer_options["lightgbm_config"]`.
  - Training and evaluation path helpers both expose `run_id` keyword-only arguments.
