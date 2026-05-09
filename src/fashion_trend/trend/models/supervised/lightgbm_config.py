from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

LIGHTGBM_ALLOWED_OBJECTIVES: tuple[str, ...] = ("regression", "regression_l1")
LIGHTGBM_DEFAULT_PARAMS: dict[str, object] = {
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
    """解析内置默认、JSON 参数文件和 CLI 覆盖，返回已校验训练配置。"""

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
        params_file=params_file_value,
        overrides=overrides,
    )


def resolve_lightgbm_config_from_stable_or_default(
    stable_params_path: Path,
) -> LightGBMTrainingConfig:
    """从 stable 参数 artifact 解析完整配置；缺失时回退到内置默认。"""

    if not stable_params_path.exists():
        return _build_lightgbm_training_config(
            lightgbm_params=dict(LIGHTGBM_DEFAULT_PARAMS),
            early_stopping=dict(LIGHTGBM_DEFAULT_EARLY_STOPPING),
            default_source="builtin",
            params_file=None,
            overrides={},
        )

    stable_payload = _read_stable_params_file(stable_params_path)
    return _build_lightgbm_training_config(
        lightgbm_params=stable_payload["lightgbm_params"],
        early_stopping=stable_payload["early_stopping"],
        default_source="stable",
        params_file=str(stable_params_path),
        overrides={},
    )


def _build_lightgbm_training_config(
    *,
    lightgbm_params: dict[str, object],
    early_stopping: dict[str, object],
    default_source: str,
    params_file: str | None,
    overrides: dict[str, object],
) -> LightGBMTrainingConfig:
    _validate_lightgbm_params(lightgbm_params)
    _validate_early_stopping(early_stopping)
    return LightGBMTrainingConfig(
        lightgbm_params=dict(lightgbm_params),
        early_stopping={"stopping_rounds": int(early_stopping["stopping_rounds"])},
        param_source={
            "default": default_source,
            "params_file": params_file,
            "overrides": dict(overrides),
        },
    )


def _read_params_file(params_path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LightGBM 参数文件不是合法 JSON: {params_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LightGBM --params 必须是 JSON object。")
    unknown_keys = set(payload) - {"lightgbm_params", "early_stopping"}
    if unknown_keys:
        raise ValueError(
            f"LightGBM --params 包含不支持的顶层 key: {sorted(unknown_keys)}"
        )
    for key in ("lightgbm_params", "early_stopping"):
        if key in payload and not isinstance(payload[key], dict):
            raise ValueError(f"LightGBM --params 的 {key} 必须是 JSON object。")
    return {
        "lightgbm_params": dict(payload.get("lightgbm_params", {})),
        "early_stopping": dict(payload.get("early_stopping", {})),
    }


def _read_stable_params_file(params_path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LightGBM stable params.json 不是合法 JSON: {params_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"LightGBM stable params.json 必须是 JSON object: {params_path}"
        )
    for key in ("lightgbm_params", "early_stopping"):
        if key not in payload:
            raise ValueError(f"LightGBM stable params.json 缺少 {key}: {params_path}")
        if not isinstance(payload[key], dict):
            raise ValueError(
                f"LightGBM stable params.json 的 {key} 必须是 JSON object: {params_path}"
            )

    lightgbm_params = dict(payload["lightgbm_params"])
    missing_param_keys = LIGHTGBM_ALLOWED_PARAM_KEYS - set(lightgbm_params)
    if missing_param_keys:
        raise ValueError(
            "LightGBM stable params.json lightgbm_params 缺少 key: "
            f"{sorted(missing_param_keys)}"
        )

    early_stopping = dict(payload["early_stopping"])
    if "stopping_rounds" not in early_stopping:
        raise ValueError(
            "LightGBM stable params.json early_stopping 缺少 stopping_rounds: "
            f"{params_path}"
        )
    return {
        "lightgbm_params": lightgbm_params,
        "early_stopping": early_stopping,
    }


def _parse_cli_param(raw_param: str) -> tuple[str, object]:
    if "=" not in raw_param:
        raise ValueError(f"LightGBM --param 必须是 key=value: {raw_param}")
    key, raw_value = raw_param.split("=", maxsplit=1)
    if not key:
        raise ValueError("LightGBM --param key 不能为空。")
    if "." in key and key != "early_stopping.stopping_rounds":
        raise ValueError(f"LightGBM 参数不支持 dotted key: {key}")
    if (
        key != "early_stopping.stopping_rounds"
        and key not in LIGHTGBM_ALLOWED_PARAM_KEYS
    ):
        raise ValueError(f"LightGBM 参数不在允许清单中: {key}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


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
        raise ValueError("LightGBM 参数 max_depth 必须是 -1 或正整数。")
    _require_positive_number(params["learning_rate"], "learning_rate")
    subsample = _require_unit_interval(params["subsample"], "subsample")
    subsample_freq = _require_non_negative_int(
        params["subsample_freq"],
        "subsample_freq",
    )
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
    _require_positive_int(
        early_stopping.get("stopping_rounds"),
        "early_stopping.stopping_rounds",
    )


def _require_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"LightGBM 参数 {key} 必须是整数。")
    return int(value)


def _require_positive_int(value: object, key: str) -> int:
    number = _require_int(value, key)
    if number <= 0:
        raise ValueError(f"LightGBM 参数 {key} 必须是正整数。")
    return number


def _require_non_negative_int(value: object, key: str) -> int:
    number = _require_int(value, key)
    if number < 0:
        raise ValueError(f"LightGBM 参数 {key} 必须是非负整数。")
    return number


def _require_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LightGBM 参数 {key} 必须是数值。")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"LightGBM 参数 {key} 必须是有限数值。")
    return number


def _require_positive_number(value: object, key: str) -> float:
    number = _require_number(value, key)
    if number <= 0:
        raise ValueError(f"LightGBM 参数 {key} 必须大于 0。")
    return number


def _require_non_negative_number(value: object, key: str) -> float:
    number = _require_number(value, key)
    if number < 0:
        raise ValueError(f"LightGBM 参数 {key} 必须是非负数值。")
    return number


def _require_unit_interval(value: object, key: str) -> float:
    number = _require_number(value, key)
    if number <= 0 or number > 1:
        raise ValueError(f"LightGBM 参数 {key} 必须在 (0, 1]。")
    return number
