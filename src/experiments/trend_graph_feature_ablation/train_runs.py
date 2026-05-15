from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from experiments.trend_graph_feature_ablation.artifact_io import (
    assert_experiment_write_path,
)
from experiments.trend_graph_feature_ablation.contracts import (
    PREDICTION_COLUMNS,
    SCHEMA_VERSION,
)
from experiments.trend_graph_feature_ablation.feature_groups import (
    build_feature_mask_digest,
    build_variant_feature_masks,
)
from experiments.trend_graph_feature_ablation.paths import EXPERIMENT_ROOT
from fashion_trend.foundation.dataframe import validate_required_columns
from fashion_trend.foundation.io import (
    write_binary_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from fashion_trend.foundation.paths import OUTPUT_DIR
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_CATEGORICAL_FEATURES,
    LIGHTGBM_MODEL_NAME,
    LIGHTGBM_TARGET_COLUMN,
    _build_lightgbm_predictions,
    _dump_model_text,
    _fit_lightgbm_model,
    _predict_with_model,
    _read_best_iteration,
    build_feature_importance_frame,
)
from fashion_trend.trend.models.supervised.lightgbm_config import (
    resolve_lightgbm_config_from_stable_or_default,
)
from fashion_trend.trend.predictions import validate_trend_model_predictions
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_VALUES
from fashion_trend.trend.splits import validate_trend_model_split_frame

STABLE_LIGHTGBM_PARAMS_PATH = OUTPUT_DIR / "models" / "lightgbm" / "params.json"
RUN_ARTIFACTS: tuple[str, ...] = (
    "predictions.csv",
    "feature_importance.csv",
    "metadata.json",
    "params.json",
    "model.txt",
)


@dataclass(frozen=True)
class VariantFeatureFrame:
    features: pd.DataFrame
    attr_type_categories: tuple[str, ...]


def run_single_variant(
    variant: str,
    split_frames: Mapping[str, pd.DataFrame],
    *,
    output_dir: Path,
    input_hashes: Mapping[str, object],
    experiment_root: Path | None = None,
) -> dict[str, object]:
    """训练单个图特征消融 variant，并只写入 run-scoped artifact。"""

    output_root = _resolve_output_dir(output_dir, experiment_root=experiment_root)
    artifact_paths = _build_artifact_paths(
        output_root,
        experiment_root=experiment_root,
    )
    split_frames = _copy_split_frames(split_frames)
    mask = _resolve_feature_mask(variant)
    numeric_features = list(mask["numeric_features"])
    categorical_features = list(mask["categorical_features"])
    feature_mask_digest = build_feature_mask_digest(
        variant,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    config = resolve_lightgbm_config_from_stable_or_default(STABLE_LIGHTGBM_PARAMS_PATH)
    started_at = time.perf_counter()
    train_prepared = _prepare_variant_feature_frame(
        split_frames["train"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    valid_prepared = _prepare_variant_feature_frame(
        split_frames["valid"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        attr_type_categories=train_prepared.attr_type_categories,
    )
    test_prepared = _prepare_variant_feature_frame(
        split_frames["test"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        attr_type_categories=train_prepared.attr_type_categories,
    )
    model = _fit_lightgbm_model(
        train_prepared.features,
        _read_target(split_frames["train"]),
        valid_prepared.features,
        _read_target(split_frames["valid"]),
        config=config,
    )
    training_elapsed_seconds = time.perf_counter() - started_at

    prepared_frames = {
        "train": train_prepared,
        "valid": valid_prepared,
        "test": test_prepared,
    }
    prediction_frames = {
        split_name: _build_lightgbm_predictions(
            split_frames[split_name],
            _predict_with_model(model, prepared.features),
        )
        for split_name, prepared in prepared_frames.items()
    }
    predictions = pd.concat(
        [prediction_frames[split_name] for split_name in TREND_MODEL_SPLIT_VALUES],
        ignore_index=True,
    ).sort_values(["week_id", "attr_type", "attr_id"], ignore_index=True)
    predictions = predictions.loc[:, list(PREDICTION_COLUMNS)]
    split_samples = pd.concat(
        [split_frames[split_name] for split_name in TREND_MODEL_SPLIT_VALUES],
        ignore_index=True,
    )
    validate_trend_model_predictions(predictions, split_samples)

    booster = model.booster_
    best_iteration = _read_best_iteration(model)
    feature_importance = build_feature_importance_frame(booster)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_name": LIGHTGBM_MODEL_NAME,
        "variant": variant,
        "feature_mask": _feature_mask_payload(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        ),
        "feature_mask_digest": feature_mask_digest,
        "input_hashes": dict(input_hashes),
        "best_iteration": best_iteration,
        "training_elapsed_seconds": float(training_elapsed_seconds),
        "output_dir": str(output_root),
        "attr_type_categories": list(train_prepared.attr_type_categories),
        "target_column": LIGHTGBM_TARGET_COLUMN,
    }
    params = {
        "schema_version": SCHEMA_VERSION,
        "model_name": LIGHTGBM_MODEL_NAME,
        "variant": variant,
        "target_column": LIGHTGBM_TARGET_COLUMN,
        "feature_mask": _feature_mask_payload(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        ),
        "feature_mask_digest": feature_mask_digest,
        "lightgbm_params": dict(config.lightgbm_params),
        "early_stopping": dict(config.early_stopping),
        "param_source": dict(config.param_source),
        "best_iteration": best_iteration,
    }

    _write_csv_guarded(predictions, artifact_paths["predictions.csv"], root=output_root)
    _write_csv_guarded(
        feature_importance,
        artifact_paths["feature_importance.csv"],
        root=output_root,
    )
    _write_json_guarded(metadata, artifact_paths["metadata.json"], root=output_root)
    _write_json_guarded(params, artifact_paths["params.json"], root=output_root)
    _write_binary_guarded(
        _dump_model_text(booster),
        artifact_paths["model.txt"],
        root=output_root,
    )
    return metadata


def _resolve_output_dir(
    output_dir: Path,
    *,
    experiment_root: Path | None,
) -> Path:
    root = experiment_root or EXPERIMENT_ROOT
    return assert_experiment_write_path(Path(output_dir), root=root)


def _build_artifact_paths(
    output_dir: Path,
    *,
    experiment_root: Path | None,
) -> dict[str, Path]:
    root = experiment_root or EXPERIMENT_ROOT
    return {
        filename: assert_experiment_write_path(output_dir / filename, root=root)
        for filename in RUN_ARTIFACTS
    }


def _write_csv_guarded(dataframe: pd.DataFrame, path: Path, *, root: Path) -> None:
    _assert_atomic_write_paths(path, root=root)
    write_csv_atomic(dataframe, path)


def _write_json_guarded(payload: dict[str, object], path: Path, *, root: Path) -> None:
    _assert_atomic_write_paths(path, root=root)
    write_json_atomic(payload, path)


def _write_binary_guarded(payload: bytes, path: Path, *, root: Path) -> None:
    _assert_atomic_write_paths(path, root=root)
    write_binary_atomic(payload, path)


def _assert_atomic_write_paths(path: Path, *, root: Path) -> None:
    assert_experiment_write_path(path, root=root)
    assert_experiment_write_path(path.with_suffix(path.suffix + ".tmp"), root=root)


def _copy_split_frames(
    split_frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    missing_splits = [
        split for split in TREND_MODEL_SPLIT_VALUES if split not in split_frames
    ]
    if missing_splits:
        raise ValueError(f"trend graph ablation 缺少 split: {missing_splits}")
    copied = {
        split_name: split_frames[split_name].copy()
        for split_name in TREND_MODEL_SPLIT_VALUES
    }
    for split_name, frame in copied.items():
        validate_trend_model_split_frame(frame, expected_split=split_name)
    return copied


def _resolve_feature_mask(variant: str) -> dict[str, list[str]]:
    masks = build_variant_feature_masks()
    try:
        return masks[variant]
    except KeyError as exc:
        raise ValueError(f"未知趋势图特征消融 variant: {variant}") from exc


def _prepare_variant_feature_frame(
    samples: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    attr_type_categories: tuple[str, ...] | None = None,
) -> VariantFeatureFrame:
    validate_required_columns(
        samples,
        (*numeric_features, *categorical_features),
        source_name="trend graph ablation LightGBM 样本",
    )
    numeric = _read_numeric_features(samples, numeric_features)
    categorical = _read_categorical_features(
        samples,
        categorical_features,
        attr_type_categories=attr_type_categories,
    )
    features = pd.concat([numeric, categorical], axis=1)
    return VariantFeatureFrame(
        features=features.loc[:, [*numeric_features, *categorical_features]],
        attr_type_categories=tuple(categorical["attr_type"].cat.categories.astype(str)),
    )


def _read_numeric_features(
    samples: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    features = pd.DataFrame(index=samples.index)
    for column in numeric_features:
        try:
            features[column] = pd.to_numeric(samples[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"trend graph ablation 数值特征无法解析: {column}"
            ) from exc
        if not np.isfinite(features[column].to_numpy(dtype=float)).all():
            raise ValueError(f"trend graph ablation 数值特征存在非有限值: {column}")
    return features


def _read_categorical_features(
    samples: pd.DataFrame,
    categorical_features: list[str],
    *,
    attr_type_categories: tuple[str, ...] | None,
) -> pd.DataFrame:
    unsupported = sorted(set(categorical_features) - set(LIGHTGBM_CATEGORICAL_FEATURES))
    if unsupported:
        raise ValueError(
            f"trend graph ablation 不支持 categorical feature: {unsupported}"
        )
    if categorical_features != list(LIGHTGBM_CATEGORICAL_FEATURES):
        raise ValueError(
            "trend graph ablation categorical feature mask must match stable LightGBM"
        )
    return pd.DataFrame(
        {
            "attr_type": _read_attr_type(
                samples,
                attr_type_categories=attr_type_categories,
            )
        },
        index=samples.index,
    )


def _read_attr_type(
    samples: pd.DataFrame,
    *,
    attr_type_categories: tuple[str, ...] | None,
) -> pd.Series:
    if samples["attr_type"].isna().any():
        raise ValueError("trend graph ablation attr_type 不能为空")
    attr_values = samples["attr_type"].astype(str)
    if attr_type_categories is None:
        categories = tuple(sorted(attr_values.unique()))
    else:
        categories = tuple(attr_type_categories)
        unknown_values = sorted(set(attr_values.unique()) - set(categories))
        if unknown_values:
            examples = ", ".join(unknown_values[:5])
            raise ValueError(
                f"unknown attr_type in trend graph ablation split: {examples}"
            )
    attr_type = attr_values.astype(CategoricalDtype(categories=list(categories)))
    attr_type.name = "attr_type"
    return attr_type


def _read_target(samples: pd.DataFrame) -> pd.Series:
    try:
        target = pd.to_numeric(samples[LIGHTGBM_TARGET_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("trend graph ablation target_growth 必须为数值") from exc
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("trend graph ablation target_growth 存在非有限值")
    return target


def _feature_mask_payload(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, list[str]]:
    return {
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
    }
