from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fashion_trend.foundation.io import write_parquet_atomic
from fashion_trend.foundation.paths import OUTPUT_MODELS_DIR
from fashion_trend.models.base import (
    MODEL_TYPE_BASELINE,
    TrendArtifact,
    TrendTrainContext,
    TrendTrainResult,
)
from fashion_trend.models.last_week import (
    LAST_WEEK_MODEL_NAME,
    LAST_WEEK_PARAMS,
    LastWeekTrainer,
    predict_last_week,
)
from fashion_trend.models.moving_average import (
    MOVING_AVERAGE_GROWTH_LAGS,
    MOVING_AVERAGE_MODEL_NAME,
    MOVING_AVERAGE_PARAMS,
    MovingAverageTrainer,
    predict_moving_average,
)
from fashion_trend.models.registry import (
    UnknownTrendModelError,
    get_trend_model_trainer,
    list_trend_model_names,
)
from fashion_trend.training import (
    build_trend_train_metadata,
    derive_trend_model_output_paths,
    run_trend_model_training,
    validate_trend_train_result,
    write_trend_model_outputs,
)
from fashion_trend.trend.predictions import validate_trend_model_predictions
from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
from fashion_trend.trend.splits import build_trend_model_split_frames
from tests.trend_samples import sample_trend_model_samples_for_split


def _expected_normalized_pred_share(
    predictions: pd.DataFrame,
    epsilon: float,
) -> pd.Series:
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


def _assert_pred_share_t1_distribution(predictions: pd.DataFrame) -> None:
    assert predictions["pred_share_t1"].between(0.0, 1.0).all()
    share_totals = predictions.groupby(["split", "week_id", "attr_type"])[
        "pred_share_t1"
    ].sum()
    assert np.isclose(share_totals, 1.0, rtol=0, atol=1e-12).all()


class TestTrendTraining:
    def test_last_week_params_are_stable(self) -> None:
        assert LAST_WEEK_PARAMS == {
            "model_name": "last_week",
            "formula": "pred_target_growth = growth_lag_1",
            "derived_formula": (
                "raw_pred_share_t1 = exp(pred_target_growth) * "
                "(share_t + epsilon) - epsilon; "
                "pred_share_t1 = group_normalize(max(raw_pred_share_t1, 0))"
            ),
            "epsilon": 1e-6,
        }

    def test_registry_lists_registered_models(self) -> None:
        assert list_trend_model_names() == (
            LAST_WEEK_MODEL_NAME,
            MOVING_AVERAGE_MODEL_NAME,
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
        paths = derive_trend_model_output_paths("last_week", Path("outputs/models"))

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

        try:
            train_model.run_trend_model_training = fake_run_trend_model_training

            assert train_model.main(["--model", MOVING_AVERAGE_MODEL_NAME]) == 0
        finally:
            train_model.run_trend_model_training = original_run_trend_model_training

        assert calls == [MOVING_AVERAGE_MODEL_NAME]

    def test_predict_last_week_uses_growth_lag_1(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)

        predictions = predict_last_week(samples)

        assert predictions.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
        assert set(predictions["model_name"]) == {LAST_WEEK_MODEL_NAME}
        pd.testing.assert_series_equal(
            predictions["pred_target_growth"],
            samples.sort_values(["week_id", "attr_type", "attr_id"], ignore_index=True)[
                "growth_lag_1"
            ],
            check_names=False,
        )
        expected_share = (
            _expected_normalized_pred_share(
                predictions,
                float(LAST_WEEK_PARAMS["epsilon"]),
            )
        )
        pd.testing.assert_series_equal(
            predictions["pred_share_t1"],
            expected_share,
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
        expected_share = (
            _expected_normalized_pred_share(
                predictions,
                float(MOVING_AVERAGE_PARAMS["epsilon"]),
            )
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
