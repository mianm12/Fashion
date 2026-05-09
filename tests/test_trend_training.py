from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion_trend.foundation.io import (
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
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
from tests.trend_samples import (
    sample_trend_model_samples_for_split,
    sample_trend_predictions_for_evaluation,
)


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
            predictions=sample_trend_predictions_for_evaluation().assign(
                model_name="lightgbm"
            ),
            params={"model_name": "lightgbm"},
            metadata={
                "param_source": {
                    "default": "builtin",
                    "params_file": None,
                    "overrides": {},
                }
            },
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

        def fake_fit(
            train_features, train_target, valid_features, valid_target, *, config
        ):
            assert config.lightgbm_params["subsample_freq"] == 1
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

    def test_run_trend_model_training_writes_lightgbm_run_without_stable_promotion(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

        def fake_fit(
            train_features, train_target, valid_features, valid_target, *, config
        ):
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)

        metadata = run_trend_model_training(
            LIGHTGBM_MODEL_NAME,
            input_paths=_write_sample_split_inputs(tmp_path),
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

        index_lines = (
            (stable_dir / "runs" / "index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
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
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(
                train_features.columns.tolist()
            ),
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
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(
                train_features.columns.tolist()
            ),
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
        row = json.loads(
            (stable_dir / "runs" / "index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
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
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(
                train_features.columns.tolist()
            ),
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
        row = json.loads(
            (stable_dir / "runs" / "index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
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

    def test_run_trend_model_training_promotes_default_lightgbm_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fashion_trend.trend.models.supervised import lightgbm as lightgbm_model

        monkeypatch.setattr(
            lightgbm_model,
            "_fit_lightgbm_model",
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(
                train_features.columns.tolist()
            ),
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
        stable_metadata = json.loads(
            (stable_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert stable_metadata["run_id"] == metadata["run_id"]
        assert stable_metadata["output_dir"] == str(stable_dir)
        assert stable_metadata["prediction_path"] == str(stable_dir / "predictions.csv")
        assert stable_metadata["run_dir"] == str(run_dir)

        index_rows = [
            json.loads(line)
            for line in (stable_dir / "runs" / "index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
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
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(
                train_features.columns.tolist()
            ),
        )

        def broken_promotion(*args, **kwargs):
            raise OSError("stable write failed")

        monkeypatch.setattr(
            run_artifacts,
            "publish_lightgbm_run_to_stable",
            broken_promotion,
        )

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
        index_path = (
            tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "index.jsonl"
        )
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
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(
                train_features.columns.tolist()
            ),
        )
        monkeypatch.setattr(
            run_artifacts,
            "publish_lightgbm_run_to_stable",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("stable write failed")
            ),
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
            lambda train_features, train_target, valid_features, valid_target, *, config: _FakeLightGBMModel(
                train_features.columns.tolist()
            ),
        )

        def successful_publish(
            *,
            stable_paths: dict[str, Path],
            **kwargs,
        ) -> dict[str, object]:
            stable_paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
            stable_paths["predictions"].write_text(
                "stable published\n",
                encoding="utf-8",
            )
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

        stable_prediction_path = (
            tmp_path / "outputs" / "models" / "lightgbm" / "predictions.csv"
        )
        assert stable_prediction_path.read_text(encoding="utf-8") == (
            "stable published\n"
        )
        captured = capsys.readouterr()
        assert "promotion succeeded" in captured.err
        assert "succeeded index write failed" in captured.err
        index_path = (
            tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "index.jsonl"
        )
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
        ]
        assert rows == [
            {
                "created_at": rows[0]["created_at"],
                "metadata_path": str(
                    tmp_path
                    / "outputs"
                    / "models"
                    / "lightgbm"
                    / "runs"
                    / "depth6-lr005"
                    / "metadata.json"
                ),
                "params_path": str(
                    tmp_path
                    / "outputs"
                    / "models"
                    / "lightgbm"
                    / "runs"
                    / "depth6-lr005"
                    / "params.json"
                ),
                "promotion_status": "not_requested",
                "run_dir": str(
                    tmp_path
                    / "outputs"
                    / "models"
                    / "lightgbm"
                    / "runs"
                    / "depth6-lr005"
                ),
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

        stable_model_path = (
            tmp_path / "outputs" / "models" / "lightgbm" / "predictions.csv"
        )
        stable_model_path.parent.mkdir(parents=True)
        stable_model_path.write_text("old model\n", encoding="utf-8")
        broken_metrics_parent = tmp_path / "outputs" / "metrics" / "lightgbm"
        broken_metrics_parent.parent.mkdir(parents=True)
        broken_metrics_parent.write_text("not a directory\n", encoding="utf-8")

        with pytest.raises(OSError):
            write_promotion_items_atomic(
                [
                    PromotionItem(stable_model_path, b"new model\n"),
                    PromotionItem(
                        broken_metrics_parent / "trend_metrics.json",
                        {"new": True},
                    ),
                ],
                tmp_path / "outputs" / "models" / "lightgbm",
            )

        assert stable_model_path.read_text(encoding="utf-8") == "old model\n"
        assert broken_metrics_parent.read_text(encoding="utf-8") == "not a directory\n"

    def test_promote_existing_lightgbm_run_publishes_model_and_metrics(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import (
            promote_existing_lightgbm_run,
        )

        model_root = tmp_path / "outputs" / "models"
        metrics_root = tmp_path / "outputs" / "metrics"
        run_dir = model_root / "lightgbm" / "runs" / "depth6-lr005"
        run_metrics_dir = metrics_root / "lightgbm" / "runs" / "depth6-lr005"
        predictions = sample_trend_predictions_for_evaluation().copy()
        predictions["model_name"] = "lightgbm"
        write_csv_atomic(predictions, run_dir / "predictions.csv")
        write_json_atomic(
            {"lightgbm_params": {"learning_rate": 0.03}},
            run_dir / "params.json",
        )
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
        write_csv_atomic(
            pd.DataFrame({"feature": ["growth_lag_1"]}),
            run_dir / "feature_importance.csv",
        )
        (run_dir / "model.txt").write_text("fake model", encoding="utf-8")
        write_json_atomic(
            _sample_lightgbm_metrics_payload(
                run_dir,
                run_metrics_dir / "trend_metrics.json",
            ),
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
        assert stable_metrics["prediction_path"] == str(
            stable_model_dir / "predictions.csv"
        )
        assert stable_metrics["output_path"] == str(stable_metrics_path)

    def test_promote_existing_lightgbm_run_success_index_failure_does_not_mark_failed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from fashion_trend.trend.training import run_artifacts
        from fashion_trend.trend.training.run_artifacts import (
            promote_existing_lightgbm_run,
        )

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
            _sample_lightgbm_metrics_payload(
                run_dir,
                run_metrics_dir / "trend_metrics.json",
            ),
            run_metrics_dir / "trend_metrics.json",
        )

        def successful_publish(items, staging_root):
            item = items[0]
            item.final_path.parent.mkdir(parents=True, exist_ok=True)
            item.final_path.write_bytes(item.payload)

        monkeypatch.setattr(
            run_artifacts,
            "write_promotion_items_atomic",
            successful_publish,
        )
        monkeypatch.setattr(
            run_artifacts,
            "upsert_lightgbm_run_index",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("succeeded index write failed")
            ),
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

    def test_promote_existing_lightgbm_run_rejects_non_object_params_before_publishing(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import (
            promote_existing_lightgbm_run,
        )

        paths = _write_promotable_lightgbm_run(tmp_path)
        paths["params"].write_text("[]\n", encoding="utf-8")
        stable_params_path = paths["model_root"] / "lightgbm" / "params.json"
        write_json_atomic({"stable": "old"}, stable_params_path)

        with pytest.raises(ValueError, match="params.json.*object"):
            promote_existing_lightgbm_run(
                "depth6-lr005",
                model_output_root=paths["model_root"],
                metrics_output_root=paths["metrics_root"],
            )

        assert json.loads(stable_params_path.read_text(encoding="utf-8")) == {
            "stable": "old"
        }

    def test_promote_existing_lightgbm_run_rejects_incomplete_metrics_before_publishing(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import (
            promote_existing_lightgbm_run,
        )

        paths = _write_promotable_lightgbm_run(tmp_path)
        write_json_atomic(
            {
                "model_name": "lightgbm",
                "run_id": "depth6-lr005",
                "prediction_path": str(paths["run_dir"] / "predictions.csv"),
            },
            paths["metrics"],
        )
        stable_metrics_path = paths["metrics_root"] / "lightgbm" / "trend_metrics.json"
        write_json_atomic({"stable": "old"}, stable_metrics_path)

        with pytest.raises(ValueError, match="ranking|overall|evaluated_splits"):
            promote_existing_lightgbm_run(
                "depth6-lr005",
                model_output_root=paths["model_root"],
                metrics_output_root=paths["metrics_root"],
            )

        assert json.loads(stable_metrics_path.read_text(encoding="utf-8")) == {
            "stable": "old"
        }

    def test_promote_existing_lightgbm_run_rejects_invalid_metric_values(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import (
            promote_existing_lightgbm_run,
        )

        paths = _write_promotable_lightgbm_run(tmp_path)
        payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        payload["overall"]["valid"]["mae"] = float("nan")
        paths["metrics"].write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )
        stable_metrics_path = paths["metrics_root"] / "lightgbm" / "trend_metrics.json"
        write_json_atomic({"stable": "old"}, stable_metrics_path)

        with pytest.raises(ValueError, match="strict JSON|有限数值|mae"):
            promote_existing_lightgbm_run(
                "depth6-lr005",
                model_output_root=paths["model_root"],
                metrics_output_root=paths["metrics_root"],
            )

        assert json.loads(stable_metrics_path.read_text(encoding="utf-8")) == {
            "stable": "old"
        }

    def test_promote_existing_lightgbm_run_rejects_malformed_params_before_publishing(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import (
            promote_existing_lightgbm_run,
        )

        paths = _write_promotable_lightgbm_run(tmp_path)
        paths["params"].write_text("{\n", encoding="utf-8")
        stable_params_path = paths["model_root"] / "lightgbm" / "params.json"
        write_json_atomic({"stable": "old"}, stable_params_path)

        with pytest.raises(ValueError, match="params.json.*JSON"):
            promote_existing_lightgbm_run(
                "depth6-lr005",
                model_output_root=paths["model_root"],
                metrics_output_root=paths["metrics_root"],
            )

        assert json.loads(stable_params_path.read_text(encoding="utf-8")) == {
            "stable": "old"
        }

    @pytest.mark.parametrize(
        "payload_key, file_name",
        [
            ("metadata", "metadata.json"),
            ("metrics", "trend_metrics.json"),
        ],
    )
    def test_promote_existing_lightgbm_run_rejects_non_object_json_payloads(
        self,
        tmp_path: Path,
        payload_key: str,
        file_name: str,
    ) -> None:
        from fashion_trend.trend.training.run_artifacts import (
            promote_existing_lightgbm_run,
        )

        paths = _write_promotable_lightgbm_run(tmp_path)
        paths[payload_key].write_text("[]\n", encoding="utf-8")

        with pytest.raises(ValueError, match=f"{file_name}.*object"):
            promote_existing_lightgbm_run(
                "depth6-lr005",
                model_output_root=paths["model_root"],
                metrics_output_root=paths["metrics_root"],
            )

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

        def fake_run_trend_model_training(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
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

        def fake_run_trend_model_training(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
            assert kwargs == {
                "run_id": None,
                "trainer_options": None,
                "promote": None,
            }
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

        def fake_run_trend_model_training(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
            assert kwargs == {
                "run_id": None,
                "trainer_options": None,
                "promote": None,
            }
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

        def fake_run_trend_model_training(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
            assert kwargs == {
                "run_id": None,
                "trainer_options": None,
                "promote": None,
            }
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

    def test_train_trend_model_main_passes_lightgbm_run_options(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        calls: list[dict[str, object]] = []
        original = train_model.run_trend_model_training

        def fake_run_trend_model_training(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
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
                "prediction_path": (
                    "outputs/models/lightgbm/runs/depth6-lr005/predictions.csv"
                ),
                "params_path": (
                    "outputs/models/lightgbm/runs/depth6-lr005/params.json"
                ),
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
        stdout = capsys.readouterr().out
        assert (
            "业务阶段: split 样本 -> "
            "outputs/models/lightgbm/runs/depth6-lr005/predictions.csv"
        ) in stdout

    def test_train_trend_model_main_logs_deferred_output_for_auto_run_id(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_model = importlib.import_module("10_train_trend_model")

        def fake_run_trend_model_training(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
            return {
                "model_name": model_name,
                "model_type": MODEL_TYPE_SUPERVISED,
                "run_id": "auto-run",
                "rows": 40,
                "weeks": 20,
                "attributes": 2,
                "splits": _sample_split_metadata(),
                "output_dir": "outputs/models/lightgbm/runs/auto-run",
                "prediction_path": "outputs/models/lightgbm/runs/auto-run/predictions.csv",
                "params_path": "outputs/models/lightgbm/runs/auto-run/params.json",
            }

        monkeypatch.setattr(
            train_model,
            "run_trend_model_training",
            fake_run_trend_model_training,
        )

        assert (
            train_model.main(
                [
                    "--model",
                    "lightgbm",
                    "--param",
                    "learning_rate=0.03",
                ]
            )
            == 0
        )

        stdout = capsys.readouterr().out
        assert "run_id 生成后确定，最终路径以 metadata 为准" in stdout
        assert (
            "业务阶段: split 样本 -> outputs/models/lightgbm/predictions.csv"
            not in stdout
        )

    @pytest.mark.parametrize(
        "args",
        [
            ["--model", "last_week", "--run-id", "bad"],
            [
                "--model",
                "last_week",
                "--params",
                "configs/trend/lightgbm/depth6_lr005.json",
            ],
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
            [
                "--model",
                "lightgbm",
                "--promote-run",
                "depth6-lr005",
                "--run-id",
                "other",
            ],
            [
                "--model",
                "lightgbm",
                "--promote-run",
                "depth6-lr005",
                "--params",
                "configs/trend/lightgbm/depth6_lr005.json",
            ],
            [
                "--model",
                "lightgbm",
                "--promote-run",
                "depth6-lr005",
                "--param",
                "learning_rate=0.03",
            ],
        ],
    )
    def test_train_trend_model_main_rejects_invalid_promotion_combinations(
        self,
        args: list[str],
    ) -> None:
        train_model = importlib.import_module("10_train_trend_model")

        assert train_model.main(args) == 2

    def test_train_trend_model_main_promote_run_does_not_call_training_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_model = importlib.import_module("10_train_trend_model")
        from fashion_trend.trend.training import run_artifacts

        training_calls: list[str] = []
        promote_calls: list[str] = []

        def fake_run_trend_model_training(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
            training_calls.append(model_name)
            raise AssertionError("--promote-run must not train")

        def fake_promote_existing_lightgbm_run(
            run_id: str,
            **kwargs,
        ) -> dict[str, object]:
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

        monkeypatch.setattr(
            train_model,
            "run_trend_model_training",
            fake_run_trend_model_training,
        )
        monkeypatch.setattr(
            run_artifacts,
            "promote_existing_lightgbm_run",
            fake_promote_existing_lightgbm_run,
        )

        assert (
            train_model.main(["--model", "lightgbm", "--promote-run", "depth6-lr005"])
            == 0
        )
        assert training_calls == []
        assert promote_calls == ["depth6-lr005"]

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


def _sample_split_metadata() -> dict[str, dict[str, int]]:
    return {
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
    }


def _write_promotable_lightgbm_run(tmp_path: Path) -> dict[str, Path]:
    model_root = tmp_path / "outputs" / "models"
    metrics_root = tmp_path / "outputs" / "metrics"
    run_dir = model_root / "lightgbm" / "runs" / "depth6-lr005"
    run_metrics_dir = metrics_root / "lightgbm" / "runs" / "depth6-lr005"
    predictions = sample_trend_predictions_for_evaluation().copy()
    predictions["model_name"] = "lightgbm"
    write_csv_atomic(predictions, run_dir / "predictions.csv")
    write_json_atomic(
        {"lightgbm_params": {"learning_rate": 0.03}},
        run_dir / "params.json",
    )
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
    write_csv_atomic(
        pd.DataFrame({"feature": ["growth_lag_1"]}),
        run_dir / "feature_importance.csv",
    )
    (run_dir / "model.txt").write_text("fake model", encoding="utf-8")
    write_json_atomic(
        _sample_lightgbm_metrics_payload(
            run_dir,
            run_metrics_dir / "trend_metrics.json",
        ),
        run_metrics_dir / "trend_metrics.json",
    )
    return {
        "model_root": model_root,
        "metrics_root": metrics_root,
        "run_dir": run_dir,
        "run_metrics_dir": run_metrics_dir,
        "params": run_dir / "params.json",
        "metadata": run_dir / "metadata.json",
        "metrics": run_metrics_dir / "trend_metrics.json",
    }


def _sample_lightgbm_metrics_payload(
    run_dir: Path,
    metrics_path: Path,
) -> dict[str, object]:
    split_metrics = {
        "mae": 0.5,
        "rmse": 0.7,
        "spearman": 0.2,
        "precision_at_k": {"10": 0.4},
        "recall_at_k": {"10": 0.4},
        "ndcg_at_k": {"10": 0.6},
    }
    return {
        "model_name": "lightgbm",
        "run_id": "depth6-lr005",
        "prediction_path": str(run_dir / "predictions.csv"),
        "output_path": str(metrics_path),
        "evaluated_splits": ["valid", "test"],
        "ranking": {
            "target_column": "target_growth",
            "prediction_column": "pred_target_growth",
            "group_by": ["split", "week_id", "attr_type"],
            "k_values": [10],
        },
        "overall": {
            "valid": dict(split_metrics),
            "test": dict(split_metrics),
        },
        "by_attr_type": {
            "valid": {"colour_group_name": dict(split_metrics)},
            "test": {"colour_group_name": dict(split_metrics)},
        },
        "groups": {
            "valid": {"ranking_groups": 4},
            "test": {"ranking_groups": 4},
        },
    }


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
