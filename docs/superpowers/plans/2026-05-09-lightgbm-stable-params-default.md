# LightGBM Stable Params Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让默认 `lightgbm` 训练优先读取 `outputs/models/lightgbm/params.json` 中的 stable 参数，缺失时才使用 built-in 默认参数，同时保持显式参数实验和默认 promotion 语义不变。

**Architecture:** 参数解析仍集中在 `lightgbm_config.py`，新增 stable artifact 参数解析入口，显式 `--params/--param` 与 stable 默认参数采用互斥模式。`training/runner.py` 在 LightGBM 路径中注入自动默认 config，但 promotion 判断必须基于“用户是否显式传参”，不能基于注入后的 `trainer_options`。

**Tech Stack:** Python 3.10-3.12、pytest、pandas、现有 `fashion_trend.trend.models.supervised.lightgbm_config`、`fashion_trend.trend.training.runner` 和 LightGBM run artifact 契约。

---

## File Structure

- Modify: `src/fashion_trend/trend/models/supervised/lightgbm_config.py`
  - 新增 `resolve_lightgbm_config_from_stable_or_default()`，stable 文件缺失时返回 built-in config；stable 文件存在时严格读取完整 `lightgbm_params` 与 `early_stopping`。
  - 保留现有 `resolve_lightgbm_config()` 的显式参数模式：built-in 默认参数 < `--params` < `--param`。
- Modify: `src/fashion_trend/trend/training/runner.py`
  - 无显式 `trainer_options` 时，基于 `stable_paths["params"]` 解析并注入 LightGBM config。
  - promotion 默认判断改为使用 `explicit_user_config`，避免自动注入 config 后误判为自定义实验。
- Modify: `tests/test_trend_lightgbm.py`
  - 覆盖 stable artifact 读取、缺失 fallback、缺字段 fail-fast、显式参数不叠加 stable。
- Modify: `tests/test_trend_training.py`
  - 覆盖 runner 使用 stable config、stable 损坏 fail-fast、默认 auto promote、显式参数默认不 promote 且不读 stable。
- Modify: `README.md`
  - 同步默认参数来源和 `--no-promote` 使用说明。
- Modify: `docs/gpt-research/implementation-plan.md`
  - 同步 LightGBM 默认参数来源，移除“代码内置默认参数就是调参结果来源”的旧表述。

Each task ends with a commit command. Execute commit commands only after the user explicitly authorizes commits for the implementation stage.

---

### Task 1: Stable 参数解析入口

**Files:**
- Modify: `tests/test_trend_lightgbm.py`
- Modify: `src/fashion_trend/trend/models/supervised/lightgbm_config.py`

- [ ] **Step 1: Write failing tests for stable config parsing**

Append these tests inside `class TestLightGBMTrendModel` in `tests/test_trend_lightgbm.py`, after `test_resolve_lightgbm_config_merges_file_and_cli_overrides`:

```python
    def test_resolve_lightgbm_config_from_stable_reads_complete_artifact(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "outputs" / "models" / "lightgbm" / "params.json"
        stable_params_path.parent.mkdir(parents=True)
        lightgbm_params = dict(config_module.LIGHTGBM_DEFAULT_PARAMS)
        lightgbm_params.update({"learning_rate": 0.03, "num_leaves": 63})
        stable_params_path.write_text(
            json.dumps(
                {
                    "model_name": "lightgbm",
                    "model_type": "supervised",
                    "best_iteration": 21,
                    "lightgbm_params": lightgbm_params,
                    "early_stopping": {"stopping_rounds": 45},
                }
            ),
            encoding="utf-8",
        )

        config = config_module.resolve_lightgbm_config_from_stable_or_default(
            stable_params_path
        )

        assert config.lightgbm_params["learning_rate"] == 0.03
        assert config.lightgbm_params["num_leaves"] == 63
        assert config.lightgbm_params["subsample_freq"] == 1
        assert config.early_stopping == {"stopping_rounds": 45}
        assert config.param_source == {
            "default": "stable",
            "params_file": str(stable_params_path),
            "overrides": {},
        }

    def test_resolve_lightgbm_config_from_stable_missing_file_uses_builtin(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "outputs" / "models" / "lightgbm" / "params.json"

        config = config_module.resolve_lightgbm_config_from_stable_or_default(
            stable_params_path
        )

        assert config.lightgbm_params == config_module.LIGHTGBM_DEFAULT_PARAMS
        assert config.early_stopping == config_module.LIGHTGBM_DEFAULT_EARLY_STOPPING
        assert config.param_source == {
            "default": "builtin",
            "params_file": None,
            "overrides": {},
        }

    def test_resolve_lightgbm_config_from_stable_rejects_missing_core_sections(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "params.json"
        stable_params_path.write_text(json.dumps({}), encoding="utf-8")

        with pytest.raises(ValueError, match="stable|params.json|lightgbm_params"):
            config_module.resolve_lightgbm_config_from_stable_or_default(
                stable_params_path
            )

    def test_resolve_lightgbm_config_from_stable_rejects_partial_params(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "params.json"
        partial_params = dict(config_module.LIGHTGBM_DEFAULT_PARAMS)
        partial_params.pop("learning_rate")
        stable_params_path.write_text(
            json.dumps(
                {
                    "lightgbm_params": partial_params,
                    "early_stopping": {"stopping_rounds": 30},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="stable|learning_rate"):
            config_module.resolve_lightgbm_config_from_stable_or_default(
                stable_params_path
            )

    def test_resolve_lightgbm_config_from_stable_rejects_missing_early_stopping_key(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "params.json"
        stable_params_path.write_text(
            json.dumps(
                {
                    "lightgbm_params": dict(config_module.LIGHTGBM_DEFAULT_PARAMS),
                    "early_stopping": {},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="stable|stopping_rounds"):
            config_module.resolve_lightgbm_config_from_stable_or_default(
                stable_params_path
            )

    def test_resolve_lightgbm_config_from_stable_rejects_invalid_json(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "params.json"
        stable_params_path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(ValueError, match="stable|JSON|params.json"):
            config_module.resolve_lightgbm_config_from_stable_or_default(
                stable_params_path
            )

    def test_resolve_lightgbm_config_from_stable_rejects_invalid_payloads(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        valid_params = dict(config_module.LIGHTGBM_DEFAULT_PARAMS)
        cases = [
            ("top_level_array", [], "object"),
            (
                "lightgbm_params_type",
                {"lightgbm_params": [], "early_stopping": {"stopping_rounds": 30}},
                "lightgbm_params",
            ),
            (
                "early_stopping_type",
                {"lightgbm_params": valid_params, "early_stopping": []},
                "early_stopping",
            ),
            (
                "unknown_param",
                {
                    "lightgbm_params": {**valid_params, "unknown": 1},
                    "early_stopping": {"stopping_rounds": 30},
                },
                "unknown|允许清单",
            ),
            (
                "invalid_param_value",
                {
                    "lightgbm_params": {**valid_params, "learning_rate": 0},
                    "early_stopping": {"stopping_rounds": 30},
                },
                "learning_rate|大于 0",
            ),
        ]
        for case_name, payload, error_match in cases:
            stable_params_path = tmp_path / f"{case_name}.json"
            stable_params_path.write_text(json.dumps(payload), encoding="utf-8")

            with pytest.raises(ValueError, match=error_match):
                config_module.resolve_lightgbm_config_from_stable_or_default(
                    stable_params_path
                )

    def test_resolve_lightgbm_config_explicit_param_does_not_use_stable(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "outputs" / "models" / "lightgbm" / "params.json"
        stable_params_path.parent.mkdir(parents=True)
        lightgbm_params = dict(config_module.LIGHTGBM_DEFAULT_PARAMS)
        lightgbm_params["num_leaves"] = 63
        stable_params_path.write_text(
            json.dumps(
                {
                    "lightgbm_params": lightgbm_params,
                    "early_stopping": {"stopping_rounds": 45},
                }
            ),
            encoding="utf-8",
        )

        config = config_module.resolve_lightgbm_config(
            cli_params=["learning_rate=0.03"]
        )

        assert config.lightgbm_params["learning_rate"] == 0.03
        assert config.lightgbm_params["num_leaves"] == 31
        assert config.early_stopping == {"stopping_rounds": 30}
        assert config.param_source == {
            "default": "builtin",
            "params_file": None,
            "overrides": {"learning_rate": 0.03},
        }
```

- [ ] **Step 2: Run the failing config tests**

Run:

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_reads_complete_artifact tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_missing_file_uses_builtin tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_missing_core_sections tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_partial_params tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_missing_early_stopping_key tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_invalid_json tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_invalid_payloads tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_explicit_param_does_not_use_stable -q
```

Expected: FAIL with `AttributeError` for `resolve_lightgbm_config_from_stable_or_default`.

- [ ] **Step 3: Implement stable config resolver**

Modify `src/fashion_trend/trend/models/supervised/lightgbm_config.py` as follows.

Replace the body of `resolve_lightgbm_config()` with:

```python
    lightgbm_params = dict(LIGHTGBM_DEFAULT_PARAMS)
    early_stopping: dict[str, object] = dict(LIGHTGBM_DEFAULT_EARLY_STOPPING)
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

    return _build_lightgbm_training_config(
        lightgbm_params=lightgbm_params,
        early_stopping=early_stopping,
        default_source="builtin",
        params_file_value=params_file_value,
        overrides=overrides,
    )
```

Add these functions below `resolve_lightgbm_config()`:

```python
def resolve_lightgbm_config_from_stable_or_default(
    stable_params_path: Path,
) -> LightGBMTrainingConfig:
    """读取 stable 参数 artifact；缺失时返回 built-in 默认训练配置。"""

    if not stable_params_path.exists():
        return _build_lightgbm_training_config(
            lightgbm_params=dict(LIGHTGBM_DEFAULT_PARAMS),
            early_stopping=dict(LIGHTGBM_DEFAULT_EARLY_STOPPING),
            default_source="builtin",
            params_file_value=None,
            overrides={},
        )

    file_payload = _read_stable_params_file(stable_params_path)
    return _build_lightgbm_training_config(
        lightgbm_params=file_payload["lightgbm_params"],
        early_stopping=file_payload["early_stopping"],
        default_source="stable",
        params_file_value=str(stable_params_path),
        overrides={},
    )


def _build_lightgbm_training_config(
    *,
    lightgbm_params: dict[str, object],
    early_stopping: dict[str, object],
    default_source: str,
    params_file_value: str | None,
    overrides: dict[str, object],
) -> LightGBMTrainingConfig:
    _validate_lightgbm_params(lightgbm_params)
    _validate_early_stopping(early_stopping)
    return LightGBMTrainingConfig(
        lightgbm_params=dict(lightgbm_params),
        early_stopping={"stopping_rounds": int(early_stopping["stopping_rounds"])},
        param_source={
            "default": default_source,
            "params_file": params_file_value,
            "overrides": dict(overrides),
        },
    )


def _read_stable_params_file(params_path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LightGBM stable 参数文件不是合法 JSON: {params_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"LightGBM stable 参数文件必须是 JSON object: {params_path}")
    for key in ("lightgbm_params", "early_stopping"):
        if key not in payload:
            raise ValueError(f"LightGBM stable 参数文件缺少 {key}: {params_path}")
        if not isinstance(payload[key], dict):
            raise ValueError(
                f"LightGBM stable 参数文件的 {key} 必须是 JSON object: {params_path}"
            )
    lightgbm_params = dict(payload["lightgbm_params"])
    early_stopping = dict(payload["early_stopping"])
    missing_param_keys = sorted(LIGHTGBM_ALLOWED_PARAM_KEYS - set(lightgbm_params))
    if missing_param_keys:
        raise ValueError(
            "LightGBM stable 参数文件 lightgbm_params 缺少核心参数: "
            f"{missing_param_keys} ({params_path})"
        )
    if "stopping_rounds" not in early_stopping:
        raise ValueError(
            "LightGBM stable 参数文件 early_stopping 缺少 stopping_rounds: "
            f"{params_path}"
        )
    return {
        "lightgbm_params": lightgbm_params,
        "early_stopping": early_stopping,
    }
```

- [ ] **Step 4: Run config tests**

Run:

```sh
uv run pytest tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_reads_complete_artifact tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_missing_file_uses_builtin tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_missing_core_sections tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_partial_params tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_missing_early_stopping_key tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_invalid_json tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_from_stable_rejects_invalid_payloads tests/test_trend_lightgbm.py::TestLightGBMTrendModel::test_resolve_lightgbm_config_explicit_param_does_not_use_stable -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```sh
git add src/fashion_trend/trend/models/supervised/lightgbm_config.py tests/test_trend_lightgbm.py
git commit -m "feat(trend): 解析 LightGBM stable 参数默认值"
```

---

### Task 2: Runner 默认参数注入与 promotion 语义

**Files:**
- Modify: `tests/test_trend_training.py`
- Modify: `src/fashion_trend/trend/training/runner.py`

- [ ] **Step 1: Write failing runner tests**

Append these tests inside `class TestTrendTraining` in `tests/test_trend_training.py`, after `test_run_trend_model_training_promotes_default_lightgbm_run`:

```python
    def test_run_trend_model_training_uses_stable_lightgbm_params_and_promotes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model
        from fashion_trend.trend.models.supervised.lightgbm_config import (
            LIGHTGBM_DEFAULT_PARAMS,
        )

        captured: dict[str, object] = {}

        def fake_fit(
            train_features,
            train_target,
            valid_features,
            valid_target,
            *,
            config,
        ):
            captured["params"] = dict(config.lightgbm_params)
            captured["source"] = dict(config.param_source)
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)
        stable_dir = tmp_path / "outputs" / "models" / "lightgbm"
        stable_params = dict(LIGHTGBM_DEFAULT_PARAMS)
        stable_params["learning_rate"] = 0.03
        write_json_atomic(
            {
                "model_name": "lightgbm",
                "lightgbm_params": stable_params,
                "early_stopping": {"stopping_rounds": 45},
            },
            stable_dir / "params.json",
        )

        metadata = run_trend_model_training(
            LIGHTGBM_MODEL_NAME,
            input_paths=_write_sample_split_inputs(tmp_path),
            output_root=tmp_path / "outputs" / "models",
        )

        run_dir = stable_dir / "runs" / str(metadata["run_id"])
        assert captured["params"]["learning_rate"] == 0.03
        assert captured["source"] == {
            "default": "stable",
            "params_file": str(stable_dir / "params.json"),
            "overrides": {},
        }
        assert (run_dir / "predictions.csv").exists()
        assert (stable_dir / "predictions.csv").exists()
        stable_metadata = json.loads(
            (stable_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert stable_metadata["param_source"]["default"] == "stable"
        row = json.loads(
            (stable_dir / "runs" / "index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assert row["promotion_status"] == "succeeded"

    def test_run_trend_model_training_missing_stable_params_uses_builtin_and_promotes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

        captured: dict[str, object] = {}

        def fake_fit(
            train_features,
            train_target,
            valid_features,
            valid_target,
            *,
            config,
        ):
            captured["source"] = dict(config.param_source)
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)

        metadata = run_trend_model_training(
            LIGHTGBM_MODEL_NAME,
            input_paths=_write_sample_split_inputs(tmp_path),
            output_root=tmp_path / "outputs" / "models",
        )

        stable_dir = tmp_path / "outputs" / "models" / "lightgbm"
        assert captured["source"] == {
            "default": "builtin",
            "params_file": None,
            "overrides": {},
        }
        assert (stable_dir / "runs" / str(metadata["run_id"]) / "predictions.csv").exists()
        assert (stable_dir / "predictions.csv").exists()

    def test_run_trend_model_training_rejects_broken_stable_params_before_run_write(
        self,
        tmp_path: Path,
    ) -> None:
        stable_dir = tmp_path / "outputs" / "models" / "lightgbm"
        write_json_atomic({}, stable_dir / "params.json")

        with pytest.raises(ValueError, match="stable|lightgbm_params"):
            run_trend_model_training(
                LIGHTGBM_MODEL_NAME,
                input_paths=_write_sample_split_inputs(tmp_path),
                output_root=tmp_path / "outputs" / "models",
            )

        assert not (stable_dir / "runs").exists()

    def test_run_trend_model_training_explicit_config_ignores_broken_stable_params(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model
        from fashion_trend.trend.models.supervised.lightgbm_config import (
            resolve_lightgbm_config,
        )

        captured: dict[str, object] = {}

        def fake_fit(
            train_features,
            train_target,
            valid_features,
            valid_target,
            *,
            config,
        ):
            captured["params"] = dict(config.lightgbm_params)
            captured["source"] = dict(config.param_source)
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)
        stable_dir = tmp_path / "outputs" / "models" / "lightgbm"
        write_json_atomic({}, stable_dir / "params.json")

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

        run_dir = stable_dir / "runs" / str(metadata["run_id"])
        assert captured["params"]["learning_rate"] == 0.03
        assert captured["params"]["num_leaves"] == 31
        assert captured["source"] == {
            "default": "builtin",
            "params_file": None,
            "overrides": {"learning_rate": 0.03},
        }
        assert (run_dir / "predictions.csv").exists()
        assert not (stable_dir / "predictions.csv").exists()
```

- [ ] **Step 2: Run the failing runner tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_uses_stable_lightgbm_params_and_promotes tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_missing_stable_params_uses_builtin_and_promotes tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_rejects_broken_stable_params_before_run_write tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_explicit_config_ignores_broken_stable_params -q
```

Expected: FAIL because runner does not read stable params yet.

- [ ] **Step 3: Inject default LightGBM config in runner**

Modify imports in `src/fashion_trend/trend/training/runner.py` by adding:

```python
from fashion_trend.trend.models.supervised.lightgbm_config import (
    resolve_lightgbm_config_from_stable_or_default,
)
```

Replace the start of `_run_lightgbm_training()` through `promotion_requested = ...` with:

```python
    split_frames = read_trend_model_split_frames(input_paths)
    stable_paths = derive_trend_model_output_paths(model_name, output_root)
    explicit_user_config = bool(trainer_options)
    resolved_trainer_options = _resolve_lightgbm_trainer_options(
        trainer_options,
        stable_params_path=stable_paths["params"],
    )
    run_root = stable_paths["output_dir"] / "runs"
    explicit_run_id = run_id is not None
    resolved_run_id = run_id or run_artifacts.generate_lightgbm_run_id(run_root)
    run_paths = derive_trend_model_output_paths(
        model_name,
        output_root,
        run_id=resolved_run_id,
    )
    if run_paths["output_dir"].exists():
        raise FileExistsError(f"LightGBM run_id 已存在: {resolved_run_id}")
    promotion_requested = _resolve_lightgbm_promotion_default(
        explicit_run_id=explicit_run_id,
        explicit_user_config=explicit_user_config,
        promote=promote,
    )
```

In the `TrendTrainContext(...)` construction in the same function, replace:

```python
        trainer_options=trainer_options,
```

with:

```python
        trainer_options=resolved_trainer_options,
```

Add this helper above `_resolve_lightgbm_promotion_default()`:

```python
def _resolve_lightgbm_trainer_options(
    trainer_options: Mapping[str, object],
    *,
    stable_params_path: Path,
) -> dict[str, object]:
    if trainer_options:
        return dict(trainer_options)
    return {
        "lightgbm_config": resolve_lightgbm_config_from_stable_or_default(
            stable_params_path
        )
    }
```

Replace `_resolve_lightgbm_promotion_default()` with:

```python
def _resolve_lightgbm_promotion_default(
    *,
    explicit_run_id: bool,
    explicit_user_config: bool,
    promote: bool | None,
) -> bool:
    if promote is not None:
        return bool(promote)
    return not explicit_run_id and not explicit_user_config
```

- [ ] **Step 4: Run runner tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_uses_stable_lightgbm_params_and_promotes tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_missing_stable_params_uses_builtin_and_promotes tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_rejects_broken_stable_params_before_run_write tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_explicit_config_ignores_broken_stable_params -q
```

Expected: PASS.

- [ ] **Step 5: Run existing LightGBM training behavior tests**

Run:

```sh
uv run pytest tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_custom_params_default_to_no_promote tests/test_trend_training.py::TestTrendTraining::test_run_trend_model_training_promotes_default_lightgbm_run tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_logs_stable_paths_for_promoted_lightgbm tests/test_trend_training.py::TestTrendTraining::test_train_trend_model_main_logs_deferred_output_for_auto_run_id -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```sh
git add src/fashion_trend/trend/training/runner.py tests/test_trend_training.py
git commit -m "feat(trend): 默认复用 LightGBM stable 参数"
```

---

### Task 3: Documentation sync

**Files:**
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`

- [ ] **Step 1: Update README LightGBM parameter text**

In `README.md`, replace the current paragraph:

```markdown
调参选择只能读取 `evaluations.jsonl` 的 `selection_metrics`；test 指标只用于最终选中 run 的一次性报告。
当前内置默认 LightGBM 参数来自已选中的调参 run `tune-20260509-r25-l1-col06-min30`，因此不传 `--params` / `--param` 时会使用 `objective=regression_l1`、`colsample_bytree=0.6` 和 `min_child_samples=30`。
```

with:

```markdown
调参选择只能读取 `evaluations.jsonl` 的 `selection_metrics`；test 指标只用于最终选中 run 的一次性报告。

默认运行 `uv run python src/10_train_trend_model.py --model lightgbm` 时，训练参数优先读取 `outputs/models/lightgbm/params.json` 中已经发布的 stable 参数；如果该文件不存在，才使用源码中的 built-in 默认参数。这条默认训练路径仍会自动发布 stable，因此会用本次训练结果覆盖 `outputs/models/lightgbm/`。

显式 `--params` 或 `--param` 会进入自定义实验参数模式，不读取 stable 参数，并继续默认不覆盖 stable 主结果。若只想使用默认参数生成 run 但不覆盖 stable，可显式传入 `--no-promote`。
```

- [ ] **Step 2: Update implementation plan parameter note**

In `docs/gpt-research/implementation-plan.md`, replace the current sentence:

```markdown
当前代码内置默认参数采用调参 run `tune-20260509-r25-l1-col06-min30` 的选择：`objective=regression_l1`、`colsample_bytree=0.6`、`min_child_samples=30`，其余参数保持上表设置。
```

with:

```markdown
当前默认训练参数优先来自 `outputs/models/lightgbm/params.json` 中已经发布的 stable 参数；stable 参数缺失时才使用源码中的 built-in 默认参数。显式 `--params` 或 `--param` 会进入自定义实验参数模式，不读取 stable 参数。
```

- [ ] **Step 3: Verify docs mention the new behavior**

Run:

```sh
rg -n "默认训练参数|stable 参数|自定义实验参数模式|内置默认 LightGBM 参数来自" README.md docs/gpt-research/implementation-plan.md
```

Expected: output includes the new default/stable parameter wording, and does not include `当前内置默认 LightGBM 参数来自`.

- [ ] **Step 4: Commit Task 3**

```sh
git add README.md docs/gpt-research/implementation-plan.md
git commit -m "docs: 说明 LightGBM stable 参数默认来源"
```

---

### Task 4: Final validation

**Files:**
- Verify: `src/fashion_trend/trend/models/supervised/lightgbm_config.py`
- Verify: `src/fashion_trend/trend/training/runner.py`
- Verify: `tests/test_trend_lightgbm.py`
- Verify: `tests/test_trend_training.py`
- Verify: `README.md`
- Verify: `docs/gpt-research/implementation-plan.md`

- [ ] **Step 1: Run focused test suite**

Run:

```sh
uv run pytest tests/test_trend_lightgbm.py tests/test_trend_training.py
```

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run:

```sh
uv run python -m compileall -q src
```

Expected: PASS with no output.

- [ ] **Step 3: Run formatting check**

Run:

```sh
uv run black --check src tests
```

Expected: PASS.

- [ ] **Step 4: Run import ordering check**

Run:

```sh
uv run isort --check-only src tests
```

Expected: PASS.

- [ ] **Step 5: Run diff hygiene check**

Run:

```sh
git diff --check
```

Expected: PASS with no output.

- [ ] **Step 6: Inspect final diff scope**

Run:

```sh
git status --short
git diff --stat
```

Expected: only the files listed in this plan are modified, unless a previous task commit already cleared them.

- [ ] **Step 7: Commit validation-only fixes if needed**

If Step 1, Step 2, Step 3, Step 4, or Step 5 required a small fix, commit that fix with:

```sh
git add src/fashion_trend/trend/models/supervised/lightgbm_config.py src/fashion_trend/trend/training/runner.py tests/test_trend_lightgbm.py tests/test_trend_training.py README.md docs/gpt-research/implementation-plan.md
git commit -m "fix(trend): 收紧 LightGBM stable 参数默认来源"
```

If no files changed after validation, do not create an empty commit.
