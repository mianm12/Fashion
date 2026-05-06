from __future__ import annotations

import importlib
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fashion_trend.evaluation import (
    build_trend_metrics_payload,
    compute_trend_group_metrics,
    compute_trend_metrics,
    derive_trend_metric_output_paths,
    read_trend_model_predictions,
    run_trend_model_evaluation,
    validate_trend_model_predictions_for_evaluation,
    write_trend_metrics,
)
from fashion_trend.models.moving_average import MOVING_AVERAGE_MODEL_NAME
from fashion_trend.training import run_trend_model_training
from fashion_trend.trend import (
    TREND_MODEL_PREDICTION_COLUMNS,
    build_trend_model_split_frames,
    write_json,
    write_trend_csv,
    write_trend_parquet,
)

from tests.trend_samples import (
    sample_trend_model_samples_for_split,
    sample_trend_predictions_for_evaluation,
)


class TrendEvaluationTests(unittest.TestCase):
    def test_derive_trend_metric_output_paths_uses_model_name(self) -> None:
        paths = derive_trend_metric_output_paths(
            "last_week",
            model_output_root=Path("outputs/models"),
            metrics_output_root=Path("outputs/metrics"),
        )

        self.assertEqual(paths["output_dir"], Path("outputs/metrics/last_week"))
        self.assertEqual(
            paths["predictions"], Path("outputs/models/last_week/predictions.csv")
        )
        self.assertEqual(
            paths["metrics"], Path("outputs/metrics/last_week/trend_metrics.json")
        )

    def test_derive_trend_metric_output_paths_rejects_unsafe_model_name(self) -> None:
        unsafe_model_names = [
            "",
            ".",
            "../escape",
            "/tmp/escape",
            "nested/model",
            "model/..",
        ]

        for model_name in unsafe_model_names:
            with self.subTest(model_name=model_name):
                with self.assertRaisesRegex(ValueError, "model_name|模型名"):
                    derive_trend_metric_output_paths(
                        model_name,
                        model_output_root=Path("outputs/models"),
                        metrics_output_root=Path("outputs/metrics"),
                    )

    def test_read_trend_model_predictions_preserves_contract_columns(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        with TemporaryDirectory() as tmp_dir:
            prediction_path = Path(tmp_dir) / "predictions.csv"
            write_trend_csv(predictions, prediction_path)

            loaded = read_trend_model_predictions(prediction_path)

        self.assertEqual(loaded.columns.tolist(), list(TREND_MODEL_PREDICTION_COLUMNS))
        self.assertEqual(len(loaded), len(predictions))

    def test_read_trend_model_predictions_rejects_extra_column(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions["debug_score"] = 1.0
        with TemporaryDirectory() as tmp_dir:
            prediction_path = Path(tmp_dir) / "predictions.csv"
            write_trend_csv(predictions, prediction_path)

            with self.assertRaisesRegex(ValueError, "列"):
                read_trend_model_predictions(prediction_path)

    def test_read_trend_model_predictions_rejects_reordered_columns(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions = predictions.loc[:, list(reversed(TREND_MODEL_PREDICTION_COLUMNS))]
        with TemporaryDirectory() as tmp_dir:
            prediction_path = Path(tmp_dir) / "predictions.csv"
            write_trend_csv(predictions, prediction_path)

            with self.assertRaisesRegex(ValueError, "列"):
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

        with self.assertRaisesRegex(ValueError, "缺少评价 split"):
            validate_trend_model_predictions_for_evaluation(predictions, "last_week")

    def test_validate_trend_model_predictions_for_evaluation_rejects_wrong_model(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()

        with self.assertRaisesRegex(ValueError, "model_name"):
            validate_trend_model_predictions_for_evaluation(
                predictions,
                "moving_average",
            )

    def test_validate_trend_model_predictions_for_evaluation_rejects_non_finite(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        predictions.loc[predictions.index[0], "pred_target_growth"] = float("nan")

        with self.assertRaisesRegex(ValueError, "非有限数值"):
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

        self.assertTrue(math.isclose(metrics["mae"], 0.5666666667, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["rmse"], math.sqrt(0.43), rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["spearman"], 0.5, rel_tol=1e-9))
        self.assertEqual(metrics["precision_at_k"]["2"], 0.5)
        self.assertEqual(metrics["recall_at_k"]["2"], 0.5)
        self.assertEqual(metrics["precision_at_k"]["3"], 1.0)
        self.assertEqual(metrics["recall_at_k"]["3"], 1.0)
        expected_ndcg_2 = 2.0 / (2.0 + (1.0 / math.log2(3.0)))
        self.assertTrue(
            math.isclose(metrics["ndcg_at_k"]["2"], expected_ndcg_2, rel_tol=1e-9)
        )
        expected_ndcg_3 = (2.0 + (1.0 / math.log2(4.0))) / (
            2.0 + (1.0 / math.log2(3.0))
        )
        self.assertTrue(
            math.isclose(metrics["ndcg_at_k"]["3"], expected_ndcg_3, rel_tol=1e-9)
        )
        self.assertLess(metrics["ndcg_at_k"]["3"], 1.0)

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

        self.assertEqual(metrics["precision_at_k"]["5"], 1.0)
        self.assertEqual(metrics["recall_at_k"]["5"], 1.0)
        self.assertEqual(metrics["ndcg_at_k"]["5"], 1.0)

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

        self.assertIsNone(metrics["spearman"])
        self.assertIsNone(metrics["ndcg_at_k"]["2"])

    def test_compute_trend_metrics_summarizes_valid_and_test_only(self) -> None:
        predictions = sample_trend_predictions_for_evaluation()

        metrics = compute_trend_metrics(predictions, k_values=(2, 3))

        self.assertEqual(set(metrics["overall"]), {"valid", "test"})
        self.assertEqual(set(metrics["by_attr_type"]), {"valid", "test"})
        self.assertEqual(metrics["groups"]["valid"]["rows"], 10)
        self.assertEqual(metrics["groups"]["valid"]["weeks"], 2)
        self.assertEqual(metrics["groups"]["valid"]["attr_types"], 2)
        self.assertEqual(metrics["groups"]["valid"]["ranking_groups"], 4)
        self.assertNotIn("train", metrics["overall"])
        self.assertIn("colour_group_name", metrics["by_attr_type"]["test"])
        self.assertIn("product_type_name", metrics["by_attr_type"]["test"])

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

        self.assertEqual(payload["model_name"], "last_week")
        self.assertEqual(
            payload["prediction_path"], "outputs/models/last_week/predictions.csv"
        )
        self.assertEqual(
            payload["output_path"], "outputs/metrics/last_week/trend_metrics.json"
        )
        self.assertEqual(payload["evaluated_splits"], ["valid", "test"])
        self.assertEqual(payload["ranking"]["k_values"], [2, 3])
        self.assertEqual(
            payload["ranking"]["group_by"],
            ["split", "week_id", "attr_type"],
        )
        json.dumps(payload, allow_nan=False)

    def test_write_trend_metrics_writes_json_without_touching_model_outputs(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prediction_path = (
                tmp_path / "outputs" / "models" / "last_week" / "predictions.csv"
            )
            metrics_path = (
                tmp_path / "outputs" / "metrics" / "last_week" / "trend_metrics.json"
            )
            model_metadata_path = prediction_path.parent / "metadata.json"
            write_trend_csv(predictions, prediction_path)
            write_json({"model_name": "last_week"}, model_metadata_path)
            payload = build_trend_metrics_payload(
                predictions,
                model_name="last_week",
                prediction_path=prediction_path,
                output_path=metrics_path,
                k_values=(2,),
            )

            write_trend_metrics(payload, metrics_path)

            self.assertTrue(metrics_path.exists())
            self.assertTrue(prediction_path.exists())
            self.assertTrue(model_metadata_path.exists())
            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(written["model_name"], "last_week")
            self.assertEqual(set(written["overall"]), {"valid", "test"})

    def test_write_trend_metrics_rejects_non_strict_json_before_writing(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "outputs" / "metrics" / "trend_metrics.json"

            with self.assertRaises(ValueError):
                write_trend_metrics({"bad": float("nan")}, metrics_path)

            self.assertFalse(metrics_path.exists())

    def test_write_trend_metrics_preserves_existing_file_for_non_strict_json(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "outputs" / "metrics" / "trend_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text('{"status":"old"}\n', encoding="utf-8")

            with self.assertRaises(ValueError):
                write_trend_metrics({"bad": float("nan")}, metrics_path)

            self.assertEqual(
                metrics_path.read_text(encoding="utf-8"),
                '{"status":"old"}\n',
            )

    def test_run_trend_model_evaluation_reads_predictions_and_writes_metrics(
        self,
    ) -> None:
        predictions = sample_trend_predictions_for_evaluation()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_root = tmp_path / "outputs" / "models"
            metrics_root = tmp_path / "outputs" / "metrics"
            prediction_path = model_root / "last_week" / "predictions.csv"
            write_trend_csv(predictions, prediction_path)

            payload = run_trend_model_evaluation(
                "last_week",
                model_output_root=model_root,
                metrics_output_root=metrics_root,
            )

            metrics_path = metrics_root / "last_week" / "trend_metrics.json"
            self.assertTrue(metrics_path.exists())
            self.assertEqual(payload["model_name"], "last_week")
            self.assertEqual(payload["groups"]["test"]["ranking_groups"], 4)
            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(written["ranking"]["k_values"], [5, 10, 20])

    def test_run_trend_model_evaluation_reads_moving_average_predictions(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_paths = {
                "train": tmp_path / "trend_model_samples_train.parquet",
                "valid": tmp_path / "trend_model_samples_valid.parquet",
                "test": tmp_path / "trend_model_samples_test.parquet",
            }
            for split_name, split_frame in split_frames.items():
                write_trend_parquet(split_frame, input_paths[split_name])

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
            self.assertTrue(metrics_path.exists())
            self.assertEqual(payload["model_name"], MOVING_AVERAGE_MODEL_NAME)
            self.assertEqual(payload["evaluated_splits"], ["valid", "test"])
            self.assertIn("valid", payload["overall"])
            self.assertIn("test", payload["overall"])

    def test_run_trend_model_evaluation_rejects_missing_predictions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with self.assertRaisesRegex(FileNotFoundError, "预测文件不存在"):
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

        with self.assertRaisesRegex(ValueError, "缺少评价 split"):
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

        with self.assertRaisesRegex(ValueError, "model_name"):
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

        self.assertEqual(exit_code, 2)

    def test_eval_trend_model_main_returns_error_for_missing_predictions(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")

        exit_code = eval_model.main(["--model", "missing_model"])

        self.assertEqual(exit_code, 1)

    def test_eval_trend_model_main_runs_evaluation_and_logs_summary(self) -> None:
        eval_model = importlib.import_module("11_eval_trend_model")
        original_run_trend_model_evaluation = eval_model.run_trend_model_evaluation

        def fake_run_trend_model_evaluation(model_name: str) -> dict[str, object]:
            self.assertEqual(model_name, "last_week")
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

        try:
            eval_model.run_trend_model_evaluation = fake_run_trend_model_evaluation
            exit_code = eval_model.main(["--model", "last_week"])
        finally:
            eval_model.run_trend_model_evaluation = original_run_trend_model_evaluation

        self.assertEqual(exit_code, 0)

if __name__ == "__main__":
    unittest.main()
