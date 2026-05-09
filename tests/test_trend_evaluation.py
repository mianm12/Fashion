from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import pytest

from fashion_trend.foundation.io import (
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from fashion_trend.trend.evaluation import (
    build_trend_metrics_payload,
    compute_trend_group_metrics,
    compute_trend_metrics,
    derive_trend_metric_output_paths,
    read_trend_model_predictions,
    run_trend_model_evaluation,
    validate_trend_model_predictions_for_evaluation,
    write_trend_metrics,
)
from fashion_trend.trend.models.baselines.moving_average import (
    MOVING_AVERAGE_MODEL_NAME,
)
from fashion_trend.trend.models.baselines.previous_growth import (
    PREVIOUS_GROWTH_MODEL_NAME,
)
from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
from fashion_trend.trend.splits import build_trend_model_split_frames
from fashion_trend.trend.training import run_trend_model_training
from tests.trend_samples import (
    sample_trend_model_samples_for_split,
    sample_trend_predictions_for_evaluation,
)


class TestTrendEvaluation:
    def test_derive_trend_metric_output_paths_uses_model_name(self) -> None:
        paths = derive_trend_metric_output_paths(
            "last_week",
            model_output_root=Path("outputs/models"),
            metrics_output_root=Path("outputs/metrics"),
        )

        assert paths["output_dir"] == Path("outputs/metrics/last_week")
        assert paths["predictions"] == Path("outputs/models/last_week/predictions.csv")
        assert paths["metrics"] == Path("outputs/metrics/last_week/trend_metrics.json")

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

    def test_derive_trend_metric_output_paths_rejects_run_id_for_baseline(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="lightgbm|run_id"):
            derive_trend_metric_output_paths(
                "last_week",
                model_output_root=Path("outputs/models"),
                metrics_output_root=Path("outputs/metrics"),
                run_id="baseline-run",
            )

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
    def test_derive_trend_metric_output_paths_rejects_unsafe_model_name(
        self,
        model_name: str,
    ) -> None:
        with pytest.raises(ValueError, match="model_name|模型名"):
            derive_trend_metric_output_paths(
                model_name,
                model_output_root=Path("outputs/models"),
                metrics_output_root=Path("outputs/metrics"),
            )

    def test_read_trend_model_predictions_preserves_contract_columns(
        self,
        tmp_path: Path,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        prediction_path = tmp_path / "predictions.csv"
        write_csv_atomic(predictions, prediction_path)

        loaded = read_trend_model_predictions(prediction_path)

        assert loaded.columns.tolist() == list(TREND_MODEL_PREDICTION_COLUMNS)
        assert len(loaded) == len(predictions)

    def test_read_trend_model_predictions_rejects_extra_column(
        self,
        tmp_path: Path,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions["debug_score"] = 1.0
        prediction_path = tmp_path / "predictions.csv"
        write_csv_atomic(predictions, prediction_path)

        with pytest.raises(ValueError, match="列"):
            read_trend_model_predictions(prediction_path)

    def test_read_trend_model_predictions_rejects_reordered_columns(
        self,
        tmp_path: Path,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions = predictions.loc[:, list(reversed(TREND_MODEL_PREDICTION_COLUMNS))]
        prediction_path = tmp_path / "predictions.csv"
        write_csv_atomic(predictions, prediction_path)

        with pytest.raises(ValueError, match="列"):
            read_trend_model_predictions(prediction_path)

    def test_validate_trend_model_predictions_for_evaluation_accepts_valid_table(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()

        validate_trend_model_predictions_for_evaluation(predictions, "last_week")

    def test_validate_trend_model_predictions_for_evaluation_rejects_missing_test(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions = predictions[predictions["split"] != "test"].copy()

        with pytest.raises(ValueError, match="缺少评价 split"):
            validate_trend_model_predictions_for_evaluation(predictions, "last_week")

    def test_validate_trend_model_predictions_for_evaluation_rejects_wrong_model(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()

        with pytest.raises(ValueError, match="model_name"):
            validate_trend_model_predictions_for_evaluation(
                predictions,
                "moving_average",
            )

    def test_validate_trend_model_predictions_for_evaluation_rejects_non_finite(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions.loc[predictions.index[0], "pred_target_growth"] = float("nan")

        with pytest.raises(ValueError, match="非有限数值"):
            validate_trend_model_predictions_for_evaluation(predictions, "last_week")

    def test_validate_trend_model_predictions_for_evaluation_rejects_bad_pred_share(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions.loc[predictions.index[0], "pred_share_t1"] = 1.2

        with pytest.raises(ValueError, match="pred_share_t1"):
            validate_trend_model_predictions_for_evaluation(predictions, "last_week")

    def test_compute_trend_group_metrics_reports_regression_and_ranking(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        group = predictions[
            (predictions["split"] == "valid")
            & (predictions["week_id"] == 10)
            & (predictions["attr_type"] == "colour_group_name")
        ].copy()

        metrics = compute_trend_group_metrics(group, k_values=(2, 3))

        assert math.isclose(metrics["mae"], 0.5666666667, rel_tol=1e-9)
        assert math.isclose(metrics["rmse"], math.sqrt(0.43), rel_tol=1e-9)
        assert math.isclose(metrics["spearman"], 0.5, rel_tol=1e-9)
        assert metrics["precision_at_k"]["2"] == 0.5
        assert metrics["recall_at_k"]["2"] == 0.5
        assert metrics["precision_at_k"]["3"] == 1.0
        assert metrics["recall_at_k"]["3"] == 1.0
        expected_ndcg_2 = 2.0 / (2.0 + (1.0 / math.log2(3.0)))
        assert math.isclose(
            metrics["ndcg_at_k"]["2"],
            expected_ndcg_2,
            rel_tol=1e-9,
        )
        expected_ndcg_3 = (2.0 + (1.0 / math.log2(4.0))) / (
            2.0 + (1.0 / math.log2(3.0))
        )
        assert math.isclose(
            metrics["ndcg_at_k"]["3"],
            expected_ndcg_3,
            rel_tol=1e-9,
        )
        assert metrics["ndcg_at_k"]["3"] < 1.0

    def test_compute_trend_group_metrics_uses_effective_k_for_small_group(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        group = predictions[
            (predictions["split"] == "valid")
            & (predictions["week_id"] == 10)
            & (predictions["attr_type"] == "product_type_name")
        ].copy()

        metrics = compute_trend_group_metrics(group, k_values=(5,))

        assert metrics["precision_at_k"]["5"] == 1.0
        assert metrics["recall_at_k"]["5"] == 1.0
        assert metrics["ndcg_at_k"]["5"] == 1.0

    def test_compute_trend_group_metrics_returns_null_for_constant_ranking(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        group = predictions[
            (predictions["split"] == "valid")
            & (predictions["week_id"] == 10)
            & (predictions["attr_type"] == "colour_group_name")
        ].copy()
        group["target_growth"] = 1.0
        group["pred_target_growth"] = 1.0

        metrics = compute_trend_group_metrics(group, k_values=(2,))

        assert metrics["spearman"] is None
        assert metrics["ndcg_at_k"]["2"] is None

    def test_compute_trend_metrics_summarizes_valid_and_test_only(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()

        metrics = compute_trend_metrics(predictions, k_values=(2, 3))

        assert set(metrics["overall"]) == {"valid", "test"}
        assert set(metrics["by_attr_type"]) == {"valid", "test"}
        assert metrics["groups"]["valid"]["rows"] == 10
        assert metrics["groups"]["valid"]["weeks"] == 2
        assert metrics["groups"]["valid"]["attr_types"] == 2
        assert metrics["groups"]["valid"]["ranking_groups"] == 4
        assert "train" not in metrics["overall"]
        assert "colour_group_name" in metrics["by_attr_type"]["test"]
        assert "product_type_name" in metrics["by_attr_type"]["test"]

    def test_build_trend_metrics_payload_records_contract(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        paths = derive_trend_metric_output_paths(
            "last_week",
            model_output_root=Path("outputs/models"),
            metrics_output_root=Path("outputs/metrics"),
        )

        payload = build_trend_metrics_payload(
            predictions,
            model_name="last_week",
            prediction_path=paths["predictions"],
            output_path=paths["metrics"],
            k_values=(2, 3),
        )

        assert payload["model_name"] == "last_week"
        assert payload["prediction_path"] == (
            "outputs/models/last_week/predictions.csv"
        )
        assert payload["output_path"] == (
            "outputs/metrics/last_week/trend_metrics.json"
        )
        assert payload["evaluated_splits"] == ["valid", "test"]
        assert payload["ranking"]["k_values"] == [2, 3]
        assert payload["ranking"]["group_by"] == ["split", "week_id", "attr_type"]
        json.dumps(payload, allow_nan=False)

    def test_write_trend_metrics_writes_json_without_touching_model_outputs(
        self,
        tmp_path: Path,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        prediction_path = (
            tmp_path / "outputs" / "models" / "last_week" / "predictions.csv"
        )
        metrics_path = (
            tmp_path / "outputs" / "metrics" / "last_week" / "trend_metrics.json"
        )
        model_metadata_path = prediction_path.parent / "metadata.json"
        write_csv_atomic(predictions, prediction_path)
        write_json_atomic({"model_name": "last_week"}, model_metadata_path)
        payload = build_trend_metrics_payload(
            predictions,
            model_name="last_week",
            prediction_path=prediction_path,
            output_path=metrics_path,
            k_values=(2,),
        )

        write_trend_metrics(payload, metrics_path)

        assert metrics_path.exists()
        assert prediction_path.exists()
        assert model_metadata_path.exists()
        written = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert written["model_name"] == "last_week"
        assert set(written["overall"]) == {"valid", "test"}

    def test_write_trend_metrics_rejects_non_strict_json_before_writing(
        self,
        tmp_path: Path,
    ) -> None:
        metrics_path = tmp_path / "outputs" / "metrics" / "trend_metrics.json"

        with pytest.raises(ValueError):
            write_trend_metrics({"bad": float("nan")}, metrics_path)

        assert not metrics_path.exists()

    def test_write_trend_metrics_preserves_existing_file_for_non_strict_json(
        self,
        tmp_path: Path,
    ) -> None:
        metrics_path = tmp_path / "outputs" / "metrics" / "trend_metrics.json"
        metrics_path.parent.mkdir(parents=True)
        metrics_path.write_text('{"status":"old"}\n', encoding="utf-8")

        with pytest.raises(ValueError):
            write_trend_metrics({"bad": float("nan")}, metrics_path)

        assert metrics_path.read_text(encoding="utf-8") == '{"status":"old"}\n'

    def test_run_trend_model_evaluation_reads_predictions_and_writes_metrics(
        self,
        tmp_path: Path,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        model_root = tmp_path / "outputs" / "models"
        metrics_root = tmp_path / "outputs" / "metrics"
        prediction_path = model_root / "last_week" / "predictions.csv"
        write_csv_atomic(predictions, prediction_path)

        payload = run_trend_model_evaluation(
            "last_week",
            model_output_root=model_root,
            metrics_output_root=metrics_root,
        )

        metrics_path = metrics_root / "last_week" / "trend_metrics.json"
        assert metrics_path.exists()
        assert payload["model_name"] == "last_week"
        assert payload["groups"]["test"]["ranking_groups"] == 4
        written = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert written["ranking"]["k_values"] == [5, 10, 20]

    def test_run_trend_model_evaluation_reads_moving_average_predictions(
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

        model_root = tmp_path / "outputs" / "models"
        metrics_root = tmp_path / "outputs" / "metrics"
        run_trend_model_training(
            MOVING_AVERAGE_MODEL_NAME,
            input_paths=input_paths,
            output_root=model_root,
        )

        payload = run_trend_model_evaluation(
            MOVING_AVERAGE_MODEL_NAME,
            model_output_root=model_root,
            metrics_output_root=metrics_root,
        )

        metrics_path = metrics_root / "moving_average" / "trend_metrics.json"
        assert metrics_path.exists()
        assert payload["model_name"] == MOVING_AVERAGE_MODEL_NAME
        assert payload["evaluated_splits"] == ["valid", "test"]
        assert "valid" in payload["overall"]
        assert "test" in payload["overall"]

    def test_run_trend_model_evaluation_writes_lightgbm_metrics(
        self,
        tmp_path: Path,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation().copy()
        predictions["model_name"] = "lightgbm"
        model_output_dir = tmp_path / "outputs" / "models" / "lightgbm"
        metrics_output_root = tmp_path / "outputs" / "metrics"
        write_csv_atomic(predictions, model_output_dir / "predictions.csv")

        payload = run_trend_model_evaluation(
            "lightgbm",
            model_output_root=tmp_path / "outputs" / "models",
            metrics_output_root=metrics_output_root,
        )

        assert payload["model_name"] == "lightgbm"
        assert (metrics_output_root / "lightgbm" / "trend_metrics.json").exists()

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

        metrics_path = (
            metrics_root / "lightgbm" / "runs" / "depth6-lr005" / "trend_metrics.json"
        )
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

    def test_run_trend_model_evaluation_reads_previous_growth_predictions(
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

        model_root = tmp_path / "outputs" / "models"
        metrics_root = tmp_path / "outputs" / "metrics"
        run_trend_model_training(
            PREVIOUS_GROWTH_MODEL_NAME,
            input_paths=input_paths,
            output_root=model_root,
        )

        payload = run_trend_model_evaluation(
            PREVIOUS_GROWTH_MODEL_NAME,
            model_output_root=model_root,
            metrics_output_root=metrics_root,
        )

        metrics_path = metrics_root / "previous_growth" / "trend_metrics.json"
        assert metrics_path.exists()
        assert payload["model_name"] == PREVIOUS_GROWTH_MODEL_NAME
        assert payload["evaluated_splits"] == ["valid", "test"]
        assert "valid" in payload["overall"]
        assert "test" in payload["overall"]

    def test_validate_lightgbm_run_metrics_payload_rejects_mismatched_run(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.evaluation.run_artifacts import (
            validate_lightgbm_run_metrics_payload,
        )

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

    def test_validate_lightgbm_run_metrics_payload_rejects_missing_contract_fields(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.evaluation.run_artifacts import (
            validate_lightgbm_run_metrics_payload,
        )

        run_dir = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005"

        with pytest.raises(ValueError, match="ranking|overall|evaluated_splits"):
            validate_lightgbm_run_metrics_payload(
                {
                    "model_name": "lightgbm",
                    "run_id": "depth6-lr005",
                    "prediction_path": str(run_dir / "predictions.csv"),
                },
                run_id="depth6-lr005",
                prediction_path=run_dir / "predictions.csv",
            )

    def test_validate_lightgbm_run_metrics_payload_rejects_non_object_payload(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.evaluation.run_artifacts import (
            validate_lightgbm_run_metrics_payload,
        )

        run_dir = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005"

        with pytest.raises(ValueError, match="metrics.*object"):
            validate_lightgbm_run_metrics_payload(
                [],
                run_id="depth6-lr005",
                prediction_path=run_dir / "predictions.csv",
            )

    @pytest.mark.parametrize(
        "field_path, bad_value, message",
        [
            (("overall", "valid", "mae"), float("nan"), "strict JSON|有限数值|mae"),
            (
                ("overall", "test", "precision_at_k", "10"),
                "bad",
                "有限数值|precision_at_k",
            ),
            (
                ("by_attr_type", "valid", "colour_group_name", "rmse"),
                "bad",
                "有限数值|by_attr_type",
            ),
        ],
    )
    def test_validate_lightgbm_run_metrics_payload_rejects_invalid_metric_values(
        self,
        tmp_path: Path,
        field_path: tuple[str, ...],
        bad_value: object,
        message: str,
    ) -> None:
        from fashion_trend.trend.evaluation.run_artifacts import (
            validate_lightgbm_run_metrics_payload,
        )

        run_dir = tmp_path / "outputs" / "models" / "lightgbm" / "runs" / "depth6-lr005"
        payload = _sample_lightgbm_run_metrics_payload(run_dir)
        target = payload
        for key in field_path[:-1]:
            target = target[key]
        target[field_path[-1]] = bad_value

        with pytest.raises(ValueError, match=message):
            validate_lightgbm_run_metrics_payload(
                payload,
                run_id="depth6-lr005",
                prediction_path=run_dir / "predictions.csv",
            )

    def test_validate_lightgbm_run_metadata_payload_rejects_mismatched_paths(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training import derive_trend_model_output_paths
        from fashion_trend.trend.training.run_artifacts import (
            validate_lightgbm_run_metadata_payload,
        )

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

    def test_validate_lightgbm_run_metadata_payload_rejects_non_object_payload(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.training import derive_trend_model_output_paths
        from fashion_trend.trend.training.run_artifacts import (
            validate_lightgbm_run_metadata_payload,
        )

        run_paths = derive_trend_model_output_paths(
            "lightgbm",
            tmp_path / "outputs" / "models",
            run_id="depth6-lr005",
        )

        with pytest.raises(ValueError, match="metadata.*object"):
            validate_lightgbm_run_metadata_payload(
                [],
                run_id="depth6-lr005",
                run_paths=run_paths,
            )

    def test_read_run_id_from_model_metadata_rejects_non_object_payload(
        self,
        tmp_path: Path,
    ) -> None:
        from fashion_trend.trend.evaluation.run_artifacts import (
            read_run_id_from_model_metadata,
        )

        metadata_path = tmp_path / "outputs" / "models" / "lightgbm" / "metadata.json"
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_text("[]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="metadata.json.*object"):
            read_run_id_from_model_metadata(metadata_path)

    def test_run_trend_model_evaluation_rejects_missing_predictions(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="预测文件不存在"):
            run_trend_model_evaluation(
                "last_week",
                model_output_root=tmp_path / "outputs" / "models",
                metrics_output_root=tmp_path / "outputs" / "metrics",
            )

    def test_build_trend_metrics_payload_rejects_missing_test_split(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions = predictions[predictions["split"] != "test"].copy()
        paths = derive_trend_metric_output_paths(
            "last_week",
            model_output_root=Path("outputs/models"),
            metrics_output_root=Path("outputs/metrics"),
        )

        with pytest.raises(ValueError, match="缺少评价 split"):
            build_trend_metrics_payload(
                predictions,
                model_name="last_week",
                prediction_path=paths["predictions"],
                output_path=paths["metrics"],
                k_values=(2, 3),
            )

    def test_build_trend_metrics_payload_rejects_wrong_model(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        paths = derive_trend_metric_output_paths(
            "moving_average",
            model_output_root=Path("outputs/models"),
            metrics_output_root=Path("outputs/metrics"),
        )

        with pytest.raises(ValueError, match="model_name"):
            build_trend_metrics_payload(
                predictions,
                model_name="moving_average",
                prediction_path=paths["predictions"],
                output_path=paths["metrics"],
                k_values=(2, 3),
            )

    def test_eval_trend_model_main_preserves_argparse_usage_error_code(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")

        exit_code = eval_model.main([])

        assert exit_code == 2

    def test_eval_trend_model_main_returns_error_for_missing_predictions(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")

        exit_code = eval_model.main(["--model", "missing_model"])

        assert exit_code == 1

    def test_eval_trend_model_main_runs_evaluation_and_logs_summary(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")
        original_run_trend_model_evaluation = eval_model.run_trend_model_evaluation

        def fake_run_trend_model_evaluation(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
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

        # 手动替换评价 runner，避免 CLI 测试读取或写入真实评价产物。
        try:
            eval_model.run_trend_model_evaluation = fake_run_trend_model_evaluation
            exit_code = eval_model.main(["--model", "last_week"])
        finally:
            eval_model.run_trend_model_evaluation = original_run_trend_model_evaluation

        assert exit_code == 0

    def test_eval_trend_model_main_passes_lightgbm_run_id(self, capsys) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")
        original = eval_model.run_trend_model_evaluation
        calls: list[dict[str, object]] = []

        def fake_run_trend_model_evaluation(
            model_name: str,
            **kwargs,
        ) -> dict[str, object]:
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
                "groups": {
                    "valid": {"ranking_groups": 4},
                    "test": {"ranking_groups": 4},
                },
                "output_path": (
                    "outputs/metrics/lightgbm/runs/depth6-lr005/trend_metrics.json"
                ),
            }

        try:
            eval_model.run_trend_model_evaluation = fake_run_trend_model_evaluation
            exit_code = eval_model.main(
                ["--model", "lightgbm", "--run-id", "depth6-lr005"]
            )
        finally:
            eval_model.run_trend_model_evaluation = original

        assert exit_code == 0
        assert calls == [{"model_name": "lightgbm", "run_id": "depth6-lr005"}]
        stdout = capsys.readouterr().out
        assert (
            "输入预测文件: " "outputs/models/lightgbm/runs/depth6-lr005/predictions.csv"
        ) in stdout
        assert (
            "业务阶段: "
            "outputs/models/lightgbm/runs/depth6-lr005/predictions.csv -> "
            "outputs/metrics/lightgbm/runs/depth6-lr005/trend_metrics.json"
        ) in stdout
        assert (
            "输出指标文件: "
            "outputs/metrics/lightgbm/runs/depth6-lr005/trend_metrics.json"
        ) in stdout

    def test_eval_trend_model_main_rejects_run_id_for_baseline(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")

        assert eval_model.main(["--model", "last_week", "--run-id", "bad"]) == 2


def _sample_lightgbm_run_metrics_payload(run_dir: Path) -> dict[str, object]:
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
        "output_path": str(run_dir / "trend_metrics.json"),
        "evaluated_splits": ["valid", "test"],
        "ranking": {
            "target_column": "target_growth",
            "prediction_column": "pred_target_growth",
            "group_by": ["week_id", "attr_type"],
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
