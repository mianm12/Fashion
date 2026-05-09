from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion_trend.foundation.io import write_parquet_atomic
from fashion_trend.trend.models.base import (
    MODEL_TYPE_BASELINE,
    MODEL_TYPE_SUPERVISED,
    TrendArtifact,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.trend.models.baselines.last_week import (
    LAST_WEEK_MODEL_NAME,
    LAST_WEEK_PARAMS,
    LastWeekTrainer,
    predict_last_week,
)
from fashion_trend.trend.models.baselines.moving_average import (
    MOVING_AVERAGE_GROWTH_LAGS,
    MOVING_AVERAGE_MODEL_NAME,
    MOVING_AVERAGE_PARAMS,
    MovingAverageTrainer,
    predict_moving_average,
)
from fashion_trend.trend.models.baselines.previous_growth import (
    PREVIOUS_GROWTH_MODEL_NAME,
    PREVIOUS_GROWTH_PARAMS,
    PreviousGrowthTrainer,
    predict_previous_growth,
)
from fashion_trend.trend.models.registry import (
    UnknownTrendModelError,
    get_trend_model_trainer,
    list_trend_model_names,
)
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_MODEL_NAME,
    LightGBMTrendTrainer,
)
from fashion_trend.trend.paths import OUTPUT_MODELS_DIR
from fashion_trend.trend.predictions import validate_trend_model_predictions
from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
from fashion_trend.trend.splits import build_trend_model_split_frames
from fashion_trend.trend.training import (
    build_trend_train_metadata,
    derive_trend_model_output_paths,
    run_trend_model_training,
    validate_trend_train_result,
    write_trend_model_outputs,
)
from tests.trend_samples import sample_trend_model_samples_for_split


def _expected_normalized_pred_share(
    predictions: pd.DataFrame,
    epsilon: float,
) -> pd.Series:
    """按 baseline 预测公式计算分组归一化后的预期 share。"""
    raw_share = (
        predictions["pred_target_growth"].map(math.exp)
        * (predictions["share_t"] + epsilon)
        - epsilon
    )
    non_negative_share = raw_share.clip(lower=0.0)
    group_total = non_negative_share.groupby(
        [
            predictions["split"],
            predictions["week_id"],
            predictions["attr_type"],
        ]
    ).transform("sum")
    return non_negative_share / group_total


def _expected_current_share_distribution(predictions: pd.DataFrame) -> pd.Series:
    """按 last_week 的当前 share 语义计算分组归一化预测 share。"""
    current_share = predictions["share_t"].clip(lower=0.0)
    group_total = current_share.groupby(
        [
            predictions["split"],
            predictions["week_id"],
            predictions["attr_type"],
        ]
    ).transform("sum")
    return current_share / group_total


def _assert_pred_share_t1_distribution(predictions: pd.DataFrame) -> None:
    """断言预测 share 在每个 split/week/attr_type 分组内形成分布。"""
    assert predictions["pred_share_t1"].between(0.0, 1.0).all()
    share_totals = predictions.groupby(["split", "week_id", "attr_type"])[
        "pred_share_t1"
    ].sum()
    assert np.isclose(share_totals, 1.0, rtol=0, atol=1e-12).all()


class TestTrendTraining:
    def test_last_week_params_are_stable(self) -> None:
        assert LAST_WEEK_PARAMS == {
            "model_name": "last_week",
            "formula": "pred_share_t1 = group_normalize(share_t)",
            "derived_formula": (
                "pred_target_growth = log((pred_share_t1 + epsilon) / "
                "(share_t + epsilon))"
            ),
            "epsilon": 1e-6,
        }

    def test_registry_lists_registered_models(self) -> None:
        assert list_trend_model_names() == (
            LAST_WEEK_MODEL_NAME,
            LIGHTGBM_MODEL_NAME,
            MOVING_AVERAGE_MODEL_NAME,
            PREVIOUS_GROWTH_MODEL_NAME,
        )

    def test_registry_returns_last_week_trainer(self) -> None:
        trainer = get_trend_model_trainer(LAST_WEEK_MODEL_NAME)

        assert isinstance(trainer, LastWeekTrainer)
        assert trainer.name == LAST_WEEK_MODEL_NAME
        assert trainer.model_type == MODEL_TYPE_BASELINE

    def test_moving_average_params_are_stable(self) -> None:
        assert MOVING_AVERAGE_PARAMS == {
            "model_name": "moving_average",
            "formula": "pred_target_growth = mean(growth_lag_1, growth_lag_2)",
            "derived_formula": (
                "raw_pred_share_t1 = exp(pred_target_growth) * "
                "(share_t + epsilon) - epsilon; "
                "pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))"
            ),
            "epsilon": 1e-6,
            "growth_lags": ["growth_lag_1", "growth_lag_2"],
        }
        assert MOVING_AVERAGE_GROWTH_LAGS == ("growth_lag_1", "growth_lag_2")

    def test_previous_growth_params_are_stable(self) -> None:
        assert PREVIOUS_GROWTH_PARAMS == {
            "model_name": "previous_growth",
            "formula": "pred_target_growth = growth_lag_1",
            "derived_formula": (
                "raw_pred_share_t1 = exp(pred_target_growth) * "
                "(share_t + epsilon) - epsilon; "
                "pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))"
            ),
            "epsilon": 1e-6,
        }

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_predict_moving_average_rejects_non_finite_growth_lag(
        self,
        bad_value: float,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        samples.loc[samples.index[0], "growth_lag_2"] = bad_value

        with pytest.raises(ValueError, match="非有限|增长 lag"):
            predict_moving_average(samples)

    def test_moving_average_trainer_copies_mutable_params(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        context = TrendTrainContext(
            model_name=MOVING_AVERAGE_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/moving_average"),
        )

        result = MovingAverageTrainer().train(context)

        assert result.params == MOVING_AVERAGE_PARAMS
        assert result.params["growth_lags"] is not MOVING_AVERAGE_PARAMS["growth_lags"]

    def test_registry_returns_moving_average_trainer(self) -> None:
        trainer = get_trend_model_trainer(MOVING_AVERAGE_MODEL_NAME)

        assert isinstance(trainer, MovingAverageTrainer)
        assert trainer.name == MOVING_AVERAGE_MODEL_NAME
        assert trainer.model_type == MODEL_TYPE_BASELINE

    def test_registry_returns_lightgbm_trainer(self) -> None:
        trainer = get_trend_model_trainer(LIGHTGBM_MODEL_NAME)

        assert isinstance(trainer, LightGBMTrendTrainer)
        assert trainer.name == LIGHTGBM_MODEL_NAME
        assert trainer.model_type == MODEL_TYPE_SUPERVISED

    def test_registry_returns_previous_growth_trainer(self) -> None:
        trainer = get_trend_model_trainer(PREVIOUS_GROWTH_MODEL_NAME)

        assert isinstance(trainer, PreviousGrowthTrainer)
        assert trainer.name == PREVIOUS_GROWTH_MODEL_NAME
        assert trainer.model_type == MODEL_TYPE_BASELINE

    def test_registry_rejects_unknown_model(self) -> None:
        with pytest.raises(UnknownTrendModelError, match="unknown_model"):
            get_trend_model_trainer("unknown_model")

    def test_derive_trend_model_output_paths_uses_model_name(self) -> None:
        paths = derive_trend_model_output_paths("last_week", Path("outputs/models"))

        assert paths["output_dir"] == Path("outputs/models/last_week")
        assert paths["predictions"] == Path("outputs/models/last_week/predictions.csv")
        assert paths["params"] == Path("outputs/models/last_week/params.json")
        assert paths["metadata"] == Path("outputs/models/last_week/metadata.json")
        assert (
            derive_trend_model_output_paths("last_week")["output_dir"]
            == OUTPUT_MODELS_DIR / "last_week"
        )

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

    def test_derive_trend_model_output_paths_rejects_run_id_for_baseline(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="lightgbm|run_id"):
            derive_trend_model_output_paths(
                "last_week",
                Path("outputs/models"),
                run_id="baseline-run",
            )

    @pytest.mark.parametrize("run_id", ["", ".", "..", "../x", "nested/x"])
    def test_validate_lightgbm_run_id_rejects_unsafe_values(
        self,
        run_id: str,
    ) -> None:
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

    def test_generate_lightgbm_run_id_uses_local_timestamp_and_hex_suffix(
        self,
    ) -> None:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from fashion_trend.trend.training.run_artifacts import generate_lightgbm_run_id

        run_id = generate_lightgbm_run_id(
            run_root=Path("outputs/models/lightgbm/runs"),
            now_factory=lambda: datetime(
                2026,
                5,
                8,
                15,
                30,
                12,
                tzinfo=ZoneInfo("Asia/Shanghai"),
            ),
            token_factory=lambda: "a1b2c3d4",
        )

        assert run_id == "20260508-153012-a1b2c3d4"

    def test_generate_lightgbm_run_id_retries_existing_directory(
        self,
        tmp_path: Path,
    ) -> None:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from fashion_trend.trend.training.run_artifacts import generate_lightgbm_run_id

        run_root = tmp_path / "outputs" / "models" / "lightgbm" / "runs"
        (run_root / "20260508-153012-aaaaaaaa").mkdir(parents=True)
        tokens = iter(["aaaaaaaa", "bbbbbbbb"])

        run_id = generate_lightgbm_run_id(
            run_root=run_root,
            now_factory=lambda: datetime(
                2026,
                5,
                8,
                15,
                30,
                12,
                tzinfo=ZoneInfo("Asia/Shanghai"),
            ),
            token_factory=lambda: next(tokens),
        )

        assert run_id == "20260508-153012-bbbbbbbb"

    @pytest.mark.parametrize(
        "model_name",
        [
            "",
            ".",
            "../escape",
            "/tmp/escape",
            "nested/model",
            "model/..",
        ],
    )
    def test_derive_trend_model_output_paths_rejects_unsafe_model_name(
        self,
        model_name: str,
    ) -> None:
        with pytest.raises(ValueError, match="model_name"):
            derive_trend_model_output_paths(model_name, Path("outputs/models"))

    def test_validate_trend_train_result_rejects_wrong_model_name(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        result = TrendTrainResult(
            model_name="wrong",
            model_type=MODEL_TYPE_BASELINE,
            predictions=predict_last_week(samples),
            params=dict(LAST_WEEK_PARAMS),
        )
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/last_week"),
        )

        with pytest.raises(ValueError, match="model_name"):
            validate_trend_train_result(result, context)

    def test_validate_trend_train_result_rejects_prediction_model_name_mismatch(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        predictions = predict_last_week(samples)
        predictions["model_name"] = "moving_average"
        result = TrendTrainResult(
            model_name=LAST_WEEK_MODEL_NAME,
            model_type=MODEL_TYPE_BASELINE,
            predictions=predictions,
            params=dict(LAST_WEEK_PARAMS),
        )
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/last_week"),
        )

        with pytest.raises(ValueError, match="model_name"):
            validate_trend_train_result(result, context)

    def test_validate_trend_train_result_rejects_non_integral_week_id(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        split_frames = {
            split_name: split_frame.copy()
            for split_name, split_frame in split_frames.items()
        }
        split_frames["train"]["week_id"] = split_frames["train"]["week_id"].astype(
            "float64"
        )
        split_frames["train"].loc[split_frames["train"].index[0], "week_id"] = 4.5
        context = TrendTrainContext(
            model_name=PREVIOUS_GROWTH_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/previous_growth"),
        )
        result = PreviousGrowthTrainer().train(context)

        with pytest.raises(ValueError, match="week_id"):
            validate_trend_train_result(result, context)

    def test_build_trend_train_metadata_rejects_core_key_override(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/last_week"),
        )
        result = LastWeekTrainer().train(context)
        result = TrendTrainResult(
            model_name=result.model_name,
            model_type=result.model_type,
            predictions=result.predictions,
            params=result.params,
            metadata={"rows": 999},
        )
        paths = derive_trend_model_output_paths("last_week", Path("outputs/models"))

        with pytest.raises(ValueError, match="metadata"):
            build_trend_train_metadata(result, context, paths)

    def test_build_trend_train_metadata_rejects_non_integral_week_id(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        split_frames = {
            split_name: split_frame.copy()
            for split_name, split_frame in split_frames.items()
        }
        split_frames["train"]["week_id"] = split_frames["train"]["week_id"].astype(
            "float64"
        )
        split_frames["train"].loc[split_frames["train"].index[0], "week_id"] = 4.5
        context = TrendTrainContext(
            model_name=PREVIOUS_GROWTH_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/previous_growth"),
        )
        result = PreviousGrowthTrainer().train(context)
        paths = derive_trend_model_output_paths(
            "previous_growth",
            Path("outputs/models"),
        )

        with pytest.raises(ValueError, match="week_id"):
            build_trend_train_metadata(result, context, paths)

    @pytest.mark.parametrize(
        "unsafe_path",
        [
            "",
            "/tmp/leak.txt",
            "../leak.txt",
            "./leak.txt",
            "nested/./leak.txt",
            ".",
        ],
    )
    def test_write_trend_model_outputs_rejects_unsafe_artifact_path_before_writing(
        self,
        tmp_path: Path,
        unsafe_path: str,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        output_root = tmp_path / "models"
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=output_root / "last_week",
        )
        result = LastWeekTrainer().train(context)
        paths = derive_trend_model_output_paths("last_week", output_root)
        bad_result = TrendTrainResult(
            model_name=result.model_name,
            model_type=result.model_type,
            predictions=result.predictions,
            params=result.params,
            artifacts=(TrendArtifact(unsafe_path, "binary", b"bad"),),
        )
        metadata = build_trend_train_metadata(bad_result, context, paths)

        with pytest.raises(ValueError, match="artifact"):
            write_trend_model_outputs(bad_result, metadata, paths)

        assert not paths["predictions"].exists()

    def test_write_trend_model_outputs_rejects_bad_json_before_writing(
        self,
        tmp_path: Path,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        output_root = tmp_path / "models"
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=output_root / "last_week",
        )
        result = LastWeekTrainer().train(context)
        bad_result = TrendTrainResult(
            model_name=result.model_name,
            model_type=result.model_type,
            predictions=result.predictions,
            params={"bad": object()},
        )
        paths = derive_trend_model_output_paths("last_week", output_root)
        metadata = build_trend_train_metadata(bad_result, context, paths)

        with pytest.raises(ValueError, match="JSON"):
            write_trend_model_outputs(bad_result, metadata, paths)

        assert not paths["predictions"].exists()

    @pytest.mark.parametrize(
        "bad_artifact, match",
        [
            (TrendArtifact("bad.json", "json", {"bad": object()}), "JSON|artifact"),
            (TrendArtifact("bad.bin", "binary", object()), "JSON|artifact"),
        ],
        ids=["json-payload", "binary-payload"],
    )
    def test_write_trend_model_outputs_rejects_bad_artifact_payload_before_writing(
        self,
        tmp_path: Path,
        bad_artifact: TrendArtifact,
        match: str,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        output_root = tmp_path / "models"
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=output_root / "last_week",
        )
        result = LastWeekTrainer().train(context)
        paths = derive_trend_model_output_paths("last_week", output_root)
        bad_result = TrendTrainResult(
            model_name=result.model_name,
            model_type=result.model_type,
            predictions=result.predictions,
            params=result.params,
            artifacts=(bad_artifact,),
        )
        metadata = build_trend_train_metadata(bad_result, context, paths)

        with pytest.raises(ValueError, match=match):
            write_trend_model_outputs(bad_result, metadata, paths)

        assert not paths["predictions"].exists()

    def test_write_trend_model_outputs_does_not_publish_partial_files(
        self,
        tmp_path: Path,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        output_root = tmp_path / "models"
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=output_root / "last_week",
        )
        result = LastWeekTrainer().train(context)
        artifact = TrendArtifact(
            "feature_importance.csv",
            "csv",
            pd.DataFrame({"feature": ["growth_lag_1"], "importance": [1.0]}),
        )
        result = TrendTrainResult(
            model_name=result.model_name,
            model_type=result.model_type,
            predictions=result.predictions,
            params=result.params,
            artifacts=(artifact,),
        )
        paths = derive_trend_model_output_paths("last_week", output_root)
        metadata = build_trend_train_metadata(result, context, paths)
        paths["metadata"].mkdir(parents=True)

        with pytest.raises(OSError):
            write_trend_model_outputs(result, metadata, paths)

        assert not paths["predictions"].exists()
        assert not paths["params"].exists()
        assert not (paths["output_dir"] / artifact.relative_path).exists()

    def test_run_trend_model_training_rejects_missing_input_split(self) -> None:
        with pytest.raises(ValueError, match="split"):
            run_trend_model_training(
                LAST_WEEK_MODEL_NAME,
                input_paths={},
                output_root=Path("outputs/models"),
            )

    def test_run_trend_model_training_writes_standard_outputs(
        self,
        tmp_path: Path,
    ) -> None:
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
            LAST_WEEK_MODEL_NAME,
            input_paths=input_paths,
            output_root=tmp_path / "outputs" / "models",
        )

        output_dir = tmp_path / "outputs" / "models" / "last_week"
        assert (output_dir / "predictions.csv").exists()
        assert (output_dir / "params.json").exists()
        assert (output_dir / "metadata.json").exists()
        assert metadata["model_name"] == LAST_WEEK_MODEL_NAME
        assert metadata["model_type"] == MODEL_TYPE_BASELINE
        assert metadata["rows"] == 40
        assert metadata["extra_artifacts"] == []

    def test_run_trend_model_training_writes_moving_average_outputs(
        self,
        tmp_path: Path,
    ) -> None:
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
            MOVING_AVERAGE_MODEL_NAME,
            input_paths=input_paths,
            output_root=tmp_path / "outputs" / "models",
        )

        output_dir = tmp_path / "outputs" / "models" / "moving_average"
        assert (output_dir / "predictions.csv").exists()
        assert (output_dir / "params.json").exists()
        assert (output_dir / "metadata.json").exists()
        assert metadata["model_name"] == MOVING_AVERAGE_MODEL_NAME
        assert metadata["model_type"] == MODEL_TYPE_BASELINE
        assert metadata["rows"] == 40
        assert metadata["extra_artifacts"] == []
        params = json.loads((output_dir / "params.json").read_text(encoding="utf-8"))
        assert params["growth_lags"] == ["growth_lag_1", "growth_lag_2"]

    def test_run_trend_model_training_writes_lightgbm_outputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

        class FakeBooster:
            def feature_name(self) -> list[str]:
                return [*lightgbm_model.LIGHTGBM_NUMERIC_FEATURES, "attr_type"]

            def feature_importance(self, importance_type: str):
                feature_count = len(self.feature_name())
                if importance_type == "split":
                    return [1 for _ in range(feature_count)]
                if importance_type == "gain":
                    return [1.0 for _ in range(feature_count)]
                raise AssertionError(f"unexpected importance_type={importance_type}")

            def model_to_string(self) -> str:
                return "fake lightgbm model"

        class FakeModel:
            best_iteration_ = 7
            best_score_ = {"valid_0": {"l2": 0.1}}
            booster_ = FakeBooster()

            def predict(self, features: pd.DataFrame, num_iteration: int | None = None):
                return features["growth_lag_1"].astype(float).to_numpy()

        def fake_fit(train_features, train_target, valid_features, valid_target):
            return FakeModel()

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
        )

        output_dir = tmp_path / "outputs" / "models" / "lightgbm"
        assert (output_dir / "predictions.csv").exists()
        assert (output_dir / "params.json").exists()
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "feature_importance.csv").exists()
        assert (output_dir / "model.txt").exists()
        assert metadata["model_name"] == LIGHTGBM_MODEL_NAME
        assert metadata["model_type"] == MODEL_TYPE_SUPERVISED
        assert metadata["extra_artifacts"] == [
            {"path": "feature_importance.csv", "kind": "csv"},
            {"path": "model.txt", "kind": "binary"},
        ]

    def test_run_trend_model_training_writes_previous_growth_outputs(
        self,
        tmp_path: Path,
    ) -> None:
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
            PREVIOUS_GROWTH_MODEL_NAME,
            input_paths=input_paths,
            output_root=tmp_path / "outputs" / "models",
        )

        output_dir = tmp_path / "outputs" / "models" / "previous_growth"
        predictions_path = output_dir / "predictions.csv"
        assert predictions_path.exists()
        assert (output_dir / "params.json").exists()
        assert (output_dir / "metadata.json").exists()
        assert metadata["model_name"] == PREVIOUS_GROWTH_MODEL_NAME
        assert metadata["model_type"] == MODEL_TYPE_BASELINE
        assert metadata["rows"] == 40
        assert metadata["extra_artifacts"] == []
        params = json.loads((output_dir / "params.json").read_text(encoding="utf-8"))
        assert params == PREVIOUS_GROWTH_PARAMS
        predictions = pd.read_csv(predictions_path)
        assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
        assert set(predictions["model_name"]) == {PREVIOUS_GROWTH_MODEL_NAME}

    def test_last_week_trainer_returns_train_result(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        context = TrendTrainContext(
            model_name=LAST_WEEK_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/last_week"),
        )

        result = LastWeekTrainer().train(context)

        assert isinstance(result, TrendTrainResult)
        assert result.model_name == LAST_WEEK_MODEL_NAME
        assert result.model_type == MODEL_TYPE_BASELINE
        assert result.params == LAST_WEEK_PARAMS
        assert result.artifacts == ()
        assert result.metadata == {}
        assert len(result.predictions) == 40

    def test_moving_average_trainer_returns_train_result(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        context = TrendTrainContext(
            model_name=MOVING_AVERAGE_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/moving_average"),
        )

        result = MovingAverageTrainer().train(context)

        assert isinstance(result, TrendTrainResult)
        assert result.model_name == MOVING_AVERAGE_MODEL_NAME
        assert result.model_type == MODEL_TYPE_BASELINE
        assert result.params == MOVING_AVERAGE_PARAMS
        assert result.artifacts == ()
        assert result.metadata == {}
        assert len(result.predictions) == 40

    def test_previous_growth_trainer_returns_train_result(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        context = TrendTrainContext(
            model_name=PREVIOUS_GROWTH_MODEL_NAME,
            split_frames=split_frames,
            input_paths={
                "train": Path("train.parquet"),
                "valid": Path("valid.parquet"),
                "test": Path("test.parquet"),
            },
            output_dir=Path("outputs/models/previous_growth"),
        )

        result = PreviousGrowthTrainer().train(context)

        assert isinstance(result, TrendTrainResult)
        assert result.model_name == PREVIOUS_GROWTH_MODEL_NAME
        assert result.model_type == MODEL_TYPE_BASELINE
        assert result.params == PREVIOUS_GROWTH_PARAMS
        assert result.artifacts == ()
        assert result.metadata == {}
        assert len(result.predictions) == 40

    def test_train_trend_model_main_preserves_argparse_usage_error_code(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")

        assert train_model.main(["--unknown"]) == 2

    def test_train_trend_model_main_rejects_unknown_model(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")

        assert train_model.main(["--model", "unknown_model"]) == 1

    def test_train_trend_model_main_runs_training_and_logs_summary(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        calls: list[str] = []
        original_run_trend_model_training = train_model.run_trend_model_training

        def fake_run_trend_model_training(model_name: str) -> dict[str, object]:
            calls.append(model_name)
            return {
                "model_name": LAST_WEEK_MODEL_NAME,
                "model_type": MODEL_TYPE_BASELINE,
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": {
                    "train": {
                        "rows": 24,
                        "weeks": 12,
                        "attributes": 2,
                        "week_min": 4,
                        "week_max": 15,
                    },
                    "valid": {
                        "rows": 8,
                        "weeks": 4,
                        "attributes": 2,
                        "week_min": 16,
                        "week_max": 19,
                    },
                    "test": {
                        "rows": 8,
                        "weeks": 4,
                        "attributes": 2,
                        "week_min": 20,
                        "week_max": 23,
                    },
                },
                "output_dir": "outputs/models/last_week",
                "prediction_path": "outputs/models/last_week/predictions.csv",
                "params_path": "outputs/models/last_week/params.json",
            }

        # 手动替换训练 runner，避免 CLI 测试写入真实模型产物。
        try:
            train_model.run_trend_model_training = fake_run_trend_model_training

            assert train_model.main(["--model", LAST_WEEK_MODEL_NAME]) == 0
        finally:
            train_model.run_trend_model_training = original_run_trend_model_training

        assert calls == [LAST_WEEK_MODEL_NAME]

    def test_train_trend_model_main_accepts_moving_average(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        calls: list[str] = []
        original_run_trend_model_training = train_model.run_trend_model_training

        def fake_run_trend_model_training(model_name: str) -> dict[str, object]:
            calls.append(model_name)
            return {
                "model_name": MOVING_AVERAGE_MODEL_NAME,
                "model_type": MODEL_TYPE_BASELINE,
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": {
                    "train": {
                        "rows": 24,
                        "weeks": 12,
                        "attributes": 2,
                        "week_min": 4,
                        "week_max": 15,
                    },
                    "valid": {
                        "rows": 8,
                        "weeks": 4,
                        "attributes": 2,
                        "week_min": 16,
                        "week_max": 19,
                    },
                    "test": {
                        "rows": 8,
                        "weeks": 4,
                        "attributes": 2,
                        "week_min": 20,
                        "week_max": 23,
                    },
                },
                "output_dir": "outputs/models/moving_average",
                "prediction_path": "outputs/models/moving_average/predictions.csv",
                "params_path": "outputs/models/moving_average/params.json",
            }

        # 手动替换训练 runner，避免 CLI 测试写入真实模型产物。
        try:
            train_model.run_trend_model_training = fake_run_trend_model_training

            assert train_model.main(["--model", MOVING_AVERAGE_MODEL_NAME]) == 0
        finally:
            train_model.run_trend_model_training = original_run_trend_model_training

        assert calls == [MOVING_AVERAGE_MODEL_NAME]

    def test_train_trend_model_main_accepts_lightgbm(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        calls: list[str] = []
        original_run_trend_model_training = train_model.run_trend_model_training

        def fake_run_trend_model_training(model_name: str) -> dict[str, object]:
            calls.append(model_name)
            return {
                "model_name": LIGHTGBM_MODEL_NAME,
                "model_type": MODEL_TYPE_SUPERVISED,
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": {
                    "train": {
                        "rows": 24,
                        "weeks": 12,
                        "attributes": 2,
                        "week_min": 4,
                        "week_max": 15,
                    },
                    "valid": {
                        "rows": 8,
                        "weeks": 4,
                        "attributes": 2,
                        "week_min": 16,
                        "week_max": 19,
                    },
                    "test": {
                        "rows": 8,
                        "weeks": 4,
                        "attributes": 2,
                        "week_min": 20,
                        "week_max": 23,
                    },
                },
                "output_dir": "outputs/models/lightgbm",
                "prediction_path": "outputs/models/lightgbm/predictions.csv",
                "params_path": "outputs/models/lightgbm/params.json",
            }

        # 手动替换训练 runner，避免 CLI 测试写入真实模型产物。
        try:
            train_model.run_trend_model_training = fake_run_trend_model_training

            assert train_model.main(["--model", LIGHTGBM_MODEL_NAME]) == 0
        finally:
            train_model.run_trend_model_training = original_run_trend_model_training

        assert calls == [LIGHTGBM_MODEL_NAME]

    def test_train_trend_model_main_accepts_previous_growth(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        calls: list[str] = []
        original_run_trend_model_training = train_model.run_trend_model_training

        def fake_run_trend_model_training(model_name: str) -> dict[str, object]:
            calls.append(model_name)
            return {
                "model_name": PREVIOUS_GROWTH_MODEL_NAME,
                "model_type": MODEL_TYPE_BASELINE,
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": {
                    "train": {
                        "rows": 24,
                        "weeks": 12,
                        "attributes": 2,
                        "week_min": 4,
                        "week_max": 15,
                    },
                    "valid": {
                        "rows": 8,
                        "weeks": 4,
                        "attributes": 2,
                        "week_min": 16,
                        "week_max": 19,
                    },
                    "test": {
                        "rows": 8,
                        "weeks": 4,
                        "attributes": 2,
                        "week_min": 20,
                        "week_max": 23,
                    },
                },
                "output_dir": "outputs/models/previous_growth",
                "prediction_path": "outputs/models/previous_growth/predictions.csv",
                "params_path": "outputs/models/previous_growth/params.json",
            }

        # 手动替换训练 runner，避免 CLI 测试写入真实模型产物。
        try:
            train_model.run_trend_model_training = fake_run_trend_model_training

            assert train_model.main(["--model", PREVIOUS_GROWTH_MODEL_NAME]) == 0
        finally:
            train_model.run_trend_model_training = original_run_trend_model_training

        assert calls == [PREVIOUS_GROWTH_MODEL_NAME]

    def test_predict_last_week_uses_current_share(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)

        predictions = predict_last_week(samples)

        assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
        assert set(predictions["model_name"]) == {LAST_WEEK_MODEL_NAME}
        expected_share = _expected_current_share_distribution(predictions)
        expected_growth = np.log(
            (expected_share + float(LAST_WEEK_PARAMS["epsilon"]))
            / (predictions["share_t"] + float(LAST_WEEK_PARAMS["epsilon"]))
        )
        pd.testing.assert_series_equal(
            predictions["pred_share_t1"],
            expected_share,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            predictions["pred_target_growth"],
            expected_growth,
            check_names=False,
        )
        _assert_pred_share_t1_distribution(predictions)

    def test_predict_moving_average_uses_two_growth_lags(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)

        predictions = predict_moving_average(samples)
        ordered_samples = samples.sort_values(
            ["week_id", "attr_type", "attr_id"],
            ignore_index=True,
        )
        expected_growth = ordered_samples.loc[:, ["growth_lag_1", "growth_lag_2"]].mean(
            axis=1
        )

        assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
        assert set(predictions["model_name"]) == {MOVING_AVERAGE_MODEL_NAME}
        pd.testing.assert_series_equal(
            predictions["pred_target_growth"],
            expected_growth,
            check_names=False,
        )
        expected_share = _expected_normalized_pred_share(
            predictions,
            float(MOVING_AVERAGE_PARAMS["epsilon"]),
        )
        pd.testing.assert_series_equal(
            predictions["pred_share_t1"],
            expected_share,
            check_names=False,
        )
        _assert_pred_share_t1_distribution(predictions)

    def test_predict_previous_growth_uses_growth_lag_1(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)

        predictions = predict_previous_growth(samples)
        ordered_samples = samples.sort_values(
            ["week_id", "attr_type", "attr_id"],
            ignore_index=True,
        )

        assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
        assert set(predictions["model_name"]) == {PREVIOUS_GROWTH_MODEL_NAME}
        pd.testing.assert_series_equal(
            predictions["pred_target_growth"],
            ordered_samples["growth_lag_1"],
            check_names=False,
        )
        expected_share = _expected_normalized_pred_share(
            predictions,
            float(PREVIOUS_GROWTH_PARAMS["epsilon"]),
        )
        pd.testing.assert_series_equal(
            predictions["pred_share_t1"],
            expected_share,
            check_names=False,
        )
        _assert_pred_share_t1_distribution(predictions)

    def test_predict_moving_average_rejects_missing_growth_lag(self) -> None:
        samples = sample_trend_model_samples_for_split().assign(split="train")
        samples = samples.drop(columns=["growth_lag_2"])

        with pytest.raises(ValueError, match="growth_lag_2"):
            predict_moving_average(samples)

    def test_predict_previous_growth_rejects_missing_growth_lag(self) -> None:
        samples = sample_trend_model_samples_for_split().assign(split="train")
        samples = samples.drop(columns=["growth_lag_1"])

        with pytest.raises(ValueError, match="growth_lag_1"):
            predict_previous_growth(samples)

    def test_predict_last_week_does_not_require_growth_lag_1(self) -> None:
        samples = sample_trend_model_samples_for_split().assign(split="train")
        samples = samples.drop(columns=["growth_lag_1"])

        predictions = predict_last_week(samples)

        assert set(predictions["model_name"]) == {LAST_WEEK_MODEL_NAME}
        _assert_pred_share_t1_distribution(predictions)

    @pytest.mark.parametrize(
        "case",
        [
            "negative",
            "above_one",
            "bad_total",
            "all_zero",
            "inf",
            "nan",
            "negative_inf",
            "non_numeric",
        ],
    )
    def test_predict_last_week_rejects_invalid_share_t(self, case: str) -> None:
        samples = sample_trend_model_samples_for_split().assign(split="train")
        group_mask = (samples["week_id"] == 4) & (
            samples["attr_type"] == "colour_group_name"
        )
        black_mask = group_mask & (samples["attr_value"] == "Black")

        if case == "negative":
            samples.loc[black_mask, "share_t"] = -0.01
        elif case == "above_one":
            samples.loc[black_mask, "share_t"] = 1.01
        elif case == "bad_total":
            samples.loc[black_mask, "share_t"] = 0.80
        elif case == "all_zero":
            samples.loc[group_mask, "share_t"] = 0.0
        elif case == "inf":
            samples.loc[black_mask, "share_t"] = float("inf")
        elif case == "nan":
            samples.loc[black_mask, "share_t"] = float("nan")
        elif case == "negative_inf":
            samples.loc[black_mask, "share_t"] = float("-inf")
        elif case == "non_numeric":
            samples["share_t"] = samples["share_t"].astype(object)
            samples.loc[black_mask, "share_t"] = "bad-share"
        else:
            raise AssertionError(f"未知测试场景: {case}")

        with pytest.raises(ValueError, match="share_t"):
            predict_last_week(samples)

    def test_predict_moving_average_rejects_illegal_split(self) -> None:
        samples = sample_trend_model_samples_for_split().assign(split="holdout")

        with pytest.raises(ValueError, match="非法 split"):
            predict_moving_average(samples)

    def test_predict_last_week_rejects_missing_split(self) -> None:
        samples = sample_trend_model_samples_for_split()

        with pytest.raises(ValueError, match="缺少必需列"):
            predict_last_week(samples)

    def test_validate_trend_model_predictions_rejects_changed_split(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        predictions = predict_last_week(samples)
        predictions.loc[0, "split"] = "test"

        with pytest.raises(ValueError, match="趋势模型预测 split 与输入不一致"):
            validate_trend_model_predictions(predictions, samples)

    def test_validate_trend_model_predictions_rejects_changed_target_growth(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        predictions = predict_last_week(samples)
        predictions.loc[0, "target_growth"] = 999.0

        with pytest.raises(ValueError, match="趋势模型预测字段与输入不一致"):
            validate_trend_model_predictions(predictions, samples)

    def test_validate_trend_model_predictions_rejects_non_finite_numeric_value(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        predictions = predict_last_week(samples)
        predictions.loc[0, "pred_target_growth"] = float("inf")

        with pytest.raises(ValueError, match="非有限"):
            validate_trend_model_predictions(predictions, samples)

    def test_validate_trend_model_predictions_rejects_invalid_pred_share(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        predictions = predict_last_week(samples)
        predictions.loc[0, "pred_share_t1"] = 1.2

        with pytest.raises(ValueError, match="pred_share_t1"):
            validate_trend_model_predictions(predictions, samples)

    def test_validate_trend_model_predictions_rejects_unnormalized_pred_share(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        predictions = predict_last_week(samples)
        group_mask = (
            (predictions["split"] == predictions.loc[0, "split"])
            & (predictions["week_id"] == predictions.loc[0, "week_id"])
            & (predictions["attr_type"] == predictions.loc[0, "attr_type"])
        )
        predictions.loc[group_mask, "pred_share_t1"] *= 0.5

        with pytest.raises(ValueError, match="pred_share_t1"):
            validate_trend_model_predictions(predictions, samples)

    def test_validate_trend_model_predictions_rejects_extra_column(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)
        predictions = predict_last_week(samples)
        predictions["debug_score"] = 1.0

        with pytest.raises(ValueError, match="列"):
            validate_trend_model_predictions(predictions, samples)
