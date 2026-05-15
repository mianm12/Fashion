from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.trend_graph_feature_ablation.train_runs as train_runs
from fashion_trend.foundation.paths import OUTPUT_DIR
from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
from fashion_trend.trend.splits import build_trend_model_split_frames
from tests.trend_samples import sample_trend_model_samples_for_split


def test_run_single_variant_writes_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    monkeypatch.setattr(
        train_runs,
        "STABLE_LIGHTGBM_PARAMS_PATH",
        tmp_path / "missing-stable-params.json",
    )
    output_dir = tmp_path / "runs" / "no_graph"

    metadata = train_runs.run_single_variant(
        "no_graph",
        _split_frames(),
        output_dir=output_dir,
        input_hashes={"enhanced_samples_train": {"hash": "abc"}},
        experiment_root=tmp_path,
    )

    assert set(_artifact_names(output_dir)) == {
        "predictions.csv",
        "feature_importance.csv",
        "metadata.json",
        "params.json",
        "model.txt",
    }
    assert metadata["variant"] == "no_graph"
    assert metadata["feature_mask_digest"]
    assert metadata["best_iteration"] == 7
    assert metadata["training_elapsed_seconds"] >= 0
    assert metadata["attr_type_categories"] == ["colour_group_name"]
    assert metadata["output_dir"] == str(output_dir.resolve(strict=False))

    predictions = pd.read_csv(
        output_dir / "predictions.csv",
        dtype={"attr_id": str},
    )
    assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)

    params = json.loads((output_dir / "params.json").read_text(encoding="utf-8"))
    assert params["lightgbm_params"]["objective"] == "regression_l1"
    assert params["early_stopping"] == {"stopping_rounds": 30}
    assert params["param_source"]["default"] == "builtin"
    assert params["best_iteration"] == 7
    assert params["feature_mask"] == metadata["feature_mask"]

    importance = pd.read_csv(output_dir / "feature_importance.csv")
    assert {
        "feature",
        "split_importance",
        "gain_importance",
        "normalized_gain_importance",
    }.issubset(importance.columns)
    assert (output_dir / "model.txt").read_text(encoding="utf-8") == (
        "fake lightgbm model"
    )


def test_run_single_variant_rejects_unknown_valid_attr_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    split_frames = _split_frames()
    split_frames["valid"] = split_frames["valid"].copy()
    split_frames["valid"].loc[
        split_frames["valid"].index[0],
        "attr_type",
    ] = "unknown_type"

    with pytest.raises(ValueError, match="unknown attr_type"):
        train_runs.run_single_variant(
            "no_graph",
            split_frames,
            output_dir=tmp_path / "runs" / "no_graph",
            input_hashes={},
            experiment_root=tmp_path,
        )


def test_run_single_variant_rejects_unknown_test_attr_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    split_frames = _split_frames()
    split_frames["test"] = split_frames["test"].copy()
    split_frames["test"].loc[
        split_frames["test"].index[0],
        "attr_type",
    ] = "unknown_type"

    with pytest.raises(ValueError, match="unknown attr_type"):
        train_runs.run_single_variant(
            "current_coarse_graph",
            split_frames,
            output_dir=tmp_path / "runs" / "current_coarse_graph",
            input_hashes={},
            experiment_root=tmp_path,
        )


def test_run_single_variant_rejects_forbidden_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    forbidden_dir = OUTPUT_DIR / "models" / "lightgbm" / "runs" / "x"

    with pytest.raises(ValueError, match="禁止写入稳定产物路径"):
        train_runs.run_single_variant(
            "no_graph",
            _split_frames(),
            output_dir=forbidden_dir,
            input_hashes={},
            experiment_root=OUTPUT_DIR / "models" / "lightgbm",
        )

    assert not (forbidden_dir / "metadata.json").exists()


def test_run_single_variant_rejects_non_finite_numeric_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightgbm_fit(monkeypatch)
    split_frames = _split_frames()
    split_frames["train"] = split_frames["train"].copy()
    split_frames["train"].loc[
        split_frames["train"].index[0],
        "growth_lag_1",
    ] = np.inf

    with pytest.raises(ValueError, match="非有限值: growth_lag_1"):
        train_runs.run_single_variant(
            "no_graph",
            split_frames,
            output_dir=tmp_path / "runs" / "no_graph",
            input_hashes={},
            experiment_root=tmp_path,
        )


def _split_frames() -> dict[str, pd.DataFrame]:
    return build_trend_model_split_frames(
        sample_trend_model_samples_for_split(),
        valid_weeks=4,
        test_weeks=4,
    )


def _artifact_names(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.iterdir() if path.is_file())


def _patch_lightgbm_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fit(
        train_features,
        train_target,
        valid_features,
        valid_target,
        *,
        config,
    ):
        assert len(train_features) == len(train_target)
        assert len(valid_features) == len(valid_target)
        assert config.lightgbm_params["objective"] == "regression_l1"
        assert str(train_features["attr_type"].dtype) == "category"
        assert str(valid_features["attr_type"].dtype) == "category"
        return _FakeLightGBMModel(train_features.columns.tolist())

    monkeypatch.setattr(train_runs, "_fit_lightgbm_model", fake_fit)


class _FakeBooster:
    def __init__(self, feature_names: list[str]) -> None:
        self._feature_names = list(feature_names)

    def feature_name(self) -> list[str]:
        return list(self._feature_names)

    def feature_importance(self, importance_type: str) -> list[float] | list[int]:
        if importance_type == "split":
            return [1 for _ in self._feature_names]
        if importance_type == "gain":
            return [float(index + 1) for index, _ in enumerate(self._feature_names)]
        raise AssertionError(f"unexpected importance_type={importance_type}")

    def model_to_string(self) -> str:
        return "fake lightgbm model"


class _FakeLightGBMModel:
    best_iteration_ = 7

    def __init__(self, feature_names: list[str]) -> None:
        self.booster_ = _FakeBooster(feature_names)

    def predict(self, features: pd.DataFrame, num_iteration: int | None = None):
        assert num_iteration == 7
        return features["growth_lag_1"].astype(float).to_numpy()
