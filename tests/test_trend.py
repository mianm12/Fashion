from __future__ import annotations

import importlib
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fashion_trend.config import OUTPUT_MODELS_DIR
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
from fashion_trend.trend import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
    TREND_MODEL_SPLIT_COLUMNS,
    build_article_week_sales_frame,
    build_attribute_graph_features_frame,
    build_attribute_week_heat_frame,
    build_attribute_week_target_frame,
    build_trend_model_samples_frame,
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    read_article_attribute_edges,
    read_attribute_hierarchy_edges,
    read_attribute_nodes,
    read_article_week_sales,
    read_attribute_week_target,
    read_trend_model_split,
    read_weekly_transactions,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_article_week_sales,
    validate_attribute_week_heat,
    validate_attribute_week_target,
    validate_trend_model_predictions,
    validate_trend_model_samples,
    validate_trend_model_split_frames,
    write_json,
    write_trend_csv,
    write_trend_parquet,
)


def sample_weekly_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": [0, 0, 0, 1],
            "article_id": ["0108775015", "0108775015", "0110065001", "0108775015"],
            "customer_id": ["customer_1", "customer_2", "customer_1", "customer_1"],
            "price": [0.10, 0.20, 0.30, 0.40],
        }
    )


def sample_article_week_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": [0],
            "article_id": ["0108775015"],
            "sales_cnt": [1],
            "sales_user_cnt": [1],
            "sales_amount": [0.10],
        }
    )


def sample_attribute_article_week_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": [0, 0, 1],
            "article_id": ["0108775015", "0110065001", "0108775015"],
            "sales_cnt": [2, 1, 1],
            "sales_user_cnt": [2, 1, 1],
            "sales_amount": [0.30, 0.30, 0.40],
        }
    )


def sample_article_attribute_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [
                "0108775015",
                "0108775015",
                "0110065001",
                "0110065001",
            ],
            "article_node_id": [
                "article_0108775015",
                "article_0108775015",
                "article_0110065001",
                "article_0110065001",
            ],
            "attr_id": [
                "colour_group_name::Black",
                "product_type_name::Vest top",
                "colour_group_name::White",
                "product_type_name::Bra",
            ],
            "attr_type": [
                "colour_group_name",
                "product_type_name",
                "colour_group_name",
                "product_type_name",
            ],
            "attr_value": ["Black", "Vest top", "White", "Bra"],
            "edge_type": [
                "has_colour_group",
                "has_product_type",
                "has_colour_group",
                "has_product_type",
            ],
            "edge_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )


def sample_attribute_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "attr_id": [
                "colour_group_name::Black",
                "colour_group_name::White",
                "colour_group_name::Blue",
                "product_type_name::Vest top",
                "product_type_name::Bra",
                "product_type_name::Dress",
            ],
            "attr_type": [
                "colour_group_name",
                "colour_group_name",
                "colour_group_name",
                "product_type_name",
                "product_type_name",
                "product_type_name",
            ],
            "attr_value": ["Black", "White", "Blue", "Vest top", "Bra", "Dress"],
            "attr_node_id": [
                "colour_group_name::Black",
                "colour_group_name::White",
                "colour_group_name::Blue",
                "product_type_name::Vest top",
                "product_type_name::Bra",
                "product_type_name::Dress",
            ],
            "article_count": [2, 1, 0, 2, 1, 0],
            "is_core_attr": [1, 1, 1, 1, 1, 1],
            "level": ["child", "child", "child", "child", "child", "child"],
        }
    )


def sample_attribute_week_heat() -> pd.DataFrame:
    return build_attribute_week_heat_frame(
        sample_attribute_article_week_sales(),
        sample_article_attribute_edges(),
        sample_attribute_nodes(),
    )


def sample_long_attribute_week_heat() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for week_id, black_heat, white_heat, blue_heat in [
        (0, 2, 1, 0),
        (1, 1, 0, 0),
        (2, 3, 1, 0),
        (3, 4, 1, 0),
        (4, 8, 2, 0),
        (5, 4, 4, 2),
    ]:
        colour_total = black_heat + white_heat + blue_heat
        for attr_id, attr_value, heat_cnt in [
            ("colour_group_name::Black", "Black", black_heat),
            ("colour_group_name::White", "White", white_heat),
            ("colour_group_name::Blue", "Blue", blue_heat),
        ]:
            records.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": "colour_group_name",
                    "attr_value": attr_value,
                    "heat_cnt": heat_cnt,
                    "type_total_heat": colour_total,
                    "heat_share": heat_cnt / colour_total if colour_total else 0.0,
                    "log_heat": math.log1p(heat_cnt),
                    "rank_in_type": 1,
                }
            )
        product_total = 1
        for attr_id, attr_value, heat_cnt in [
            ("product_type_name::Vest top", "Vest top", 1),
            ("product_type_name::Bra", "Bra", 0),
            ("product_type_name::Dress", "Dress", 0),
        ]:
            records.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": "product_type_name",
                    "attr_value": attr_value,
                    "heat_cnt": heat_cnt,
                    "type_total_heat": product_total,
                    "heat_share": heat_cnt / product_total,
                    "log_heat": math.log1p(heat_cnt),
                    "rank_in_type": 1,
                }
            )
    heat = pd.DataFrame(records).sort_values(
        ["week_id", "attr_type", "heat_cnt", "attr_id"],
        ascending=[True, True, False, True],
        ignore_index=True,
    )
    heat["rank_in_type"] = (
        heat.groupby(["week_id", "attr_type"]).cumcount().add(1).astype("int64")
    )
    return heat.loc[:, list(ATTRIBUTE_WEEK_HEAT_COLUMNS)]


def sample_attribute_hierarchy_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_attr_id": [
                "colour_group_name::Black",
                "product_type_name::Vest top",
            ],
            "child_attr_id": [
                "colour_group_name::White",
                "product_type_name::Bra",
            ],
            "parent_attr_type": ["colour_group_name", "product_type_name"],
            "child_attr_type": ["colour_group_name", "product_type_name"],
            "relation_type": ["test_contains_colour", "test_contains_type"],
            "edge_weight": [2, 1],
        }
    )


def sample_trend_model_samples_for_split() -> pd.DataFrame:
    rows = []
    for week_id in range(4, 24):
        for attr_id, attr_type, attr_value in [
            ("colour_group_name::Black", "colour_group_name", "Black"),
            ("colour_group_name::White", "colour_group_name", "White"),
        ]:
            share_t = 0.60 if attr_value == "Black" else 0.40
            rows.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": attr_type,
                    "attr_value": attr_value,
                    "heat_t": 10 + week_id,
                    "share_t": share_t,
                    "log_heat_t": math.log1p(10 + week_id),
                    "rank_in_type_t": 1 if attr_value == "Black" else 2,
                    "heat_lag_1": 9 + week_id,
                    "heat_lag_2": 8 + week_id,
                    "heat_lag_3": 7 + week_id,
                    "heat_lag_4": 6 + week_id,
                    "share_lag_1": share_t - 0.01,
                    "share_lag_2": share_t - 0.02,
                    "share_lag_3": share_t - 0.03,
                    "share_lag_4": share_t - 0.04,
                    "growth_lag_1": 0.10 if attr_value == "Black" else -0.05,
                    "growth_lag_2": 0.05 if attr_value == "Black" else -0.02,
                    "acc_lag_1": 0.05 if attr_value == "Black" else -0.03,
                    "heat_ma_4": 8.5 + week_id,
                    "share_ma_4": share_t - 0.015,
                    "share_std_4": 0.01,
                    "share_max_4": share_t,
                    "share_min_4": share_t - 0.04,
                    "article_count": 10,
                    "is_core_attr": 1,
                    "parent_count": 1,
                    "child_count": 0,
                    "degree": 1,
                    "history_total_heat_t": 100 + week_id,
                    "history_active_weeks_t": week_id,
                    "is_trend_eligible_t": True,
                    "week_index": week_id,
                    "week_mod_52": week_id % 52,
                    "target_growth": 0.12 if attr_value == "Black" else -0.03,
                    "target_log_heat_t1": math.log1p(11 + week_id),
                    "target_rank_in_type_t1": 1 if attr_value == "Black" else 2,
                }
            )
    return pd.DataFrame(rows).loc[:, list(TREND_MODEL_SAMPLE_COLUMNS)]


def sample_trend_predictions_for_evaluation() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("train", 8),
        ("valid", 10),
        ("valid", 11),
        ("test", 12),
        ("test", 13),
    ]
    for split_name, week_id in specs:
        rows.extend(
            [
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.40,
                    "pred_share_t1": 0.52,
                    "target_growth": 3.0,
                    "pred_target_growth": 2.8,
                    "target_rank_in_type_t1": 1,
                },
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::White",
                    "attr_type": "colour_group_name",
                    "attr_value": "White",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.30,
                    "pred_share_t1": 0.20,
                    "target_growth": 2.0,
                    "pred_target_growth": 1.0,
                    "target_rank_in_type_t1": 2,
                },
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::Blue",
                    "attr_type": "colour_group_name",
                    "attr_value": "Blue",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.20,
                    "pred_share_t1": 0.25,
                    "target_growth": 1.0,
                    "pred_target_growth": 1.5,
                    "target_rank_in_type_t1": 3,
                },
                {
                    "week_id": week_id,
                    "attr_id": "product_type_name::Dress",
                    "attr_type": "product_type_name",
                    "attr_value": "Dress",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.55,
                    "pred_share_t1": 0.62,
                    "target_growth": 1.5,
                    "pred_target_growth": 1.0,
                    "target_rank_in_type_t1": 1,
                },
                {
                    "week_id": week_id,
                    "attr_id": "product_type_name::Vest top",
                    "attr_type": "product_type_name",
                    "attr_value": "Vest top",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.45,
                    "pred_share_t1": 0.40,
                    "target_growth": 0.5,
                    "pred_target_growth": 0.7,
                    "target_rank_in_type_t1": 2,
                },
            ]
        )
    return pd.DataFrame(rows).loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]


class ArticleWeekSalesFrameTests(unittest.TestCase):
    def test_read_weekly_transactions_rejects_missing_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "weekly_transactions.parquet"

            with self.assertRaisesRegex(FileNotFoundError, "周级交易表不存在"):
                read_weekly_transactions(input_path)

    def test_read_weekly_transactions_reports_missing_required_field(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "weekly_transactions.parquet"
            sample_weekly_transactions().drop(columns=["price"]).to_parquet(input_path)

            with self.assertRaisesRegex(ValueError, "price"):
                read_weekly_transactions(input_path)

    def test_read_weekly_transactions_reports_unreadable_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "weekly_transactions.parquet"
            input_path.write_text("not a parquet file", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "无法读取周级交易表"):
                read_weekly_transactions(input_path)

    def test_build_article_week_sales_frame_aggregates_sales_by_week_and_article(
        self,
    ) -> None:
        sales = build_article_week_sales_frame(sample_weekly_transactions())

        self.assertEqual(sales.columns.tolist(), list(ARTICLE_WEEK_SALES_COLUMNS))
        self.assertEqual(
            sales[["week_id", "article_id", "sales_cnt", "sales_user_cnt"]].to_dict(
                "records"
            ),
            [
                {
                    "week_id": 0,
                    "article_id": "0108775015",
                    "sales_cnt": 2,
                    "sales_user_cnt": 2,
                },
                {
                    "week_id": 0,
                    "article_id": "0110065001",
                    "sales_cnt": 1,
                    "sales_user_cnt": 1,
                },
                {
                    "week_id": 1,
                    "article_id": "0108775015",
                    "sales_cnt": 1,
                    "sales_user_cnt": 1,
                },
            ],
        )
        self.assertTrue(math.isclose(float(sales.loc[0, "sales_amount"]), 0.30))
        self.assertTrue(math.isclose(float(sales.loc[1, "sales_amount"]), 0.30))
        self.assertTrue(math.isclose(float(sales.loc[2, "sales_amount"]), 0.40))

    def test_build_article_week_sales_frame_preserves_article_id_as_string(self) -> None:
        sales = build_article_week_sales_frame(sample_weekly_transactions())

        self.assertEqual(sales["article_id"].dtype.name, "string")
        self.assertEqual(sales.loc[0, "article_id"], "0108775015")

    def test_build_article_week_sales_frame_rejects_missing_required_values(self) -> None:
        transactions = sample_weekly_transactions()
        transactions.loc[0, "customer_id"] = pd.NA

        with self.assertRaisesRegex(ValueError, "customer_id"):
            build_article_week_sales_frame(transactions)

    def test_build_article_week_sales_frame_rejects_negative_price(self) -> None:
        transactions = sample_weekly_transactions()
        transactions.loc[0, "price"] = -0.01

        with self.assertRaisesRegex(ValueError, "price"):
            build_article_week_sales_frame(transactions)

    def test_validate_article_week_sales_rejects_duplicate_week_article(self) -> None:
        sales = pd.DataFrame(
            {
                "week_id": [0, 0],
                "article_id": ["0108775015", "0108775015"],
                "sales_cnt": [1, 1],
                "sales_user_cnt": [1, 1],
                "sales_amount": [0.10, 0.20],
            }
        )

        with self.assertRaisesRegex(ValueError, "week_id, article_id"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_non_positive_sales_count(
        self,
    ) -> None:
        sales = sample_article_week_sales()
        sales.loc[0, "sales_cnt"] = 0

        with self.assertRaisesRegex(ValueError, "sales_cnt"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_non_positive_sales_user_count(
        self,
    ) -> None:
        sales = sample_article_week_sales()
        sales.loc[0, "sales_user_cnt"] = 0

        with self.assertRaisesRegex(ValueError, "sales_user_cnt"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_negative_sales_amount(self) -> None:
        sales = sample_article_week_sales()
        sales.loc[0, "sales_amount"] = -0.01

        with self.assertRaisesRegex(ValueError, "sales_amount"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_missing_required_output_column(
        self,
    ) -> None:
        sales = sample_article_week_sales().drop(columns=["sales_amount"])

        with self.assertRaisesRegex(ValueError, "sales_amount"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_missing_required_output_value(
        self,
    ) -> None:
        sales = sample_article_week_sales()
        sales.loc[0, "article_id"] = pd.NA

        with self.assertRaisesRegex(ValueError, "article_id"):
            validate_article_week_sales(sales)

    def test_read_article_week_sales_preserves_article_id_as_string(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "article_week_sales.csv"
            sample_article_week_sales().to_csv(input_path, index=False)

            sales = read_article_week_sales(input_path)

            self.assertEqual(sales["article_id"].dtype.name, "string")
            self.assertEqual(sales.loc[0, "article_id"], "0108775015")

    def test_read_article_week_sales_rejects_missing_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "article_week_sales.csv"

            with self.assertRaisesRegex(FileNotFoundError, "商品周销量表不存在"):
                read_article_week_sales(input_path)

    def test_read_article_week_sales_reports_missing_required_column(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "article_week_sales.csv"
            sample_article_week_sales().drop(columns=["sales_cnt"]).to_csv(
                input_path,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "sales_cnt"):
                read_article_week_sales(input_path)

    def test_read_article_week_sales_reports_invalid_numeric_value(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "article_week_sales.csv"
            input_path.write_text(
                "week_id,article_id,sales_cnt,sales_user_cnt,sales_amount\n"
                "0,0108775015,not_int,1,0.10\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "无法读取商品周销量表"):
                read_article_week_sales(input_path)


class TrendCsvWriteTests(unittest.TestCase):
    def test_write_trend_csv_creates_parent_directory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "attribute_week_heat.csv"

            write_trend_csv(sample_article_week_sales(), output_path)

            self.assertTrue(output_path.exists())

    def test_write_trend_csv_replaces_existing_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "attribute_week_heat.csv"
            output_path.write_text("stale", encoding="utf-8")

            write_trend_csv(sample_article_week_sales(), output_path)

            self.assertNotEqual(output_path.read_text(encoding="utf-8"), "stale")

    def test_write_trend_csv_removes_tmp_file_when_replace_fails(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "attribute_week_heat.csv"
            output_path.mkdir()

            with self.assertRaises(OSError):
                write_trend_csv(sample_article_week_sales(), output_path)

            self.assertFalse(output_path.with_suffix(".csv.tmp").exists())

    def test_write_trend_csv_quotes_all_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "attribute_week_heat.csv"
            dataframe = pd.DataFrame(
                {
                    "week_id": [0],
                    "attr_id": ["garment_group_name::Under-, Nightwear"],
                    "attr_type": ["garment_group_name"],
                    "attr_value": ["Under-, Nightwear"],
                    "heat_cnt": [2],
                    "type_total_heat": [2],
                    "heat_share": [1.0],
                    "log_heat": [1.0986122886681098],
                    "rank_in_type": [1],
                }
            )

            write_trend_csv(dataframe, output_path)

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                lines[0],
                '"week_id","attr_id","attr_type","attr_value","heat_cnt","type_total_heat","heat_share","log_heat","rank_in_type"',
            )
            self.assertIn('"garment_group_name::Under-, Nightwear"', lines[1])
            self.assertIn('"Under-, Nightwear"', lines[1])
            self.assertFalse(output_path.with_suffix(".csv.tmp").exists())


class AttributeWeekHeatFrameTests(unittest.TestCase):
    def test_read_article_attribute_edges_preserves_string_dtypes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "article_attribute_edges.csv"
            sample_article_attribute_edges().to_csv(input_path, index=False)

            edges = read_article_attribute_edges(input_path)

            self.assertEqual(edges["article_id"].dtype.name, "string")
            self.assertEqual(edges["attr_id"].dtype.name, "string")
            self.assertEqual(edges["attr_type"].dtype.name, "string")
            self.assertEqual(edges["attr_value"].dtype.name, "string")
            self.assertEqual(edges.loc[0, "article_id"], "0108775015")

    def test_read_article_attribute_edges_rejects_missing_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "article_attribute_edges.csv"

            with self.assertRaisesRegex(FileNotFoundError, "商品-属性边表不存在"):
                read_article_attribute_edges(input_path)

    def test_read_article_attribute_edges_reports_missing_required_column(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "article_attribute_edges.csv"
            sample_article_attribute_edges().drop(columns=["attr_value"]).to_csv(
                input_path,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "attr_value"):
                read_article_attribute_edges(input_path)

    def test_build_attribute_week_heat_frame_builds_complete_attribute_week_panel(
        self,
    ) -> None:
        heat = build_attribute_week_heat_frame(
            sample_attribute_article_week_sales(),
            sample_article_attribute_edges(),
            sample_attribute_nodes(),
        )

        self.assertEqual(len(heat), 12)
        self.assertEqual(heat.columns.tolist(), list(ATTRIBUTE_WEEK_HEAT_COLUMNS))
        self.assertEqual(set(heat["week_id"]), {0, 1})
        self.assertEqual(set(heat["attr_id"]), set(sample_attribute_nodes()["attr_id"]))

        zero_row = heat[
            (heat["week_id"] == 0)
            & (heat["attr_id"] == "colour_group_name::Blue")
        ].iloc[0]
        self.assertEqual(int(zero_row["heat_cnt"]), 0)
        self.assertEqual(float(zero_row["heat_share"]), 0.0)
        self.assertEqual(float(zero_row["log_heat"]), 0.0)

    def test_build_attribute_week_heat_frame_calculates_heat_metrics(self) -> None:
        heat = build_attribute_week_heat_frame(
            sample_attribute_article_week_sales(),
            sample_article_attribute_edges(),
            sample_attribute_nodes(),
        )

        self.assertEqual(heat.columns.tolist(), list(ATTRIBUTE_WEEK_HEAT_COLUMNS))

        week0_colour = heat[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ].sort_values("rank_in_type")
        self.assertEqual(
            week0_colour[
                ["attr_id", "heat_cnt", "type_total_heat", "rank_in_type"]
            ].to_dict("records"),
            [
                {
                    "attr_id": "colour_group_name::Black",
                    "heat_cnt": 2,
                    "type_total_heat": 3,
                    "rank_in_type": 1,
                },
                {
                    "attr_id": "colour_group_name::White",
                    "heat_cnt": 1,
                    "type_total_heat": 3,
                    "rank_in_type": 2,
                },
                {
                    "attr_id": "colour_group_name::Blue",
                    "heat_cnt": 0,
                    "type_total_heat": 3,
                    "rank_in_type": 3,
                },
            ],
        )
        self.assertTrue(math.isclose(float(week0_colour.iloc[0]["heat_share"]), 2 / 3))
        self.assertTrue(math.isclose(float(week0_colour.iloc[1]["heat_share"]), 1 / 3))
        self.assertTrue(
            math.isclose(float(week0_colour.iloc[0]["log_heat"]), math.log1p(2))
        )

        week1_product = heat[
            (heat["week_id"] == 1) & (heat["attr_type"] == "product_type_name")
        ].sort_values("rank_in_type")
        self.assertEqual(
            week1_product[
                ["attr_id", "heat_cnt", "type_total_heat", "rank_in_type"]
            ].to_dict("records"),
            [
                {
                    "attr_id": "product_type_name::Vest top",
                    "heat_cnt": 1,
                    "type_total_heat": 1,
                    "rank_in_type": 1,
                },
                {
                    "attr_id": "product_type_name::Bra",
                    "heat_cnt": 0,
                    "type_total_heat": 1,
                    "rank_in_type": 2,
                },
                {
                    "attr_id": "product_type_name::Dress",
                    "heat_cnt": 0,
                    "type_total_heat": 1,
                    "rank_in_type": 3,
                },
            ],
        )
        self.assertTrue(math.isclose(float(week1_product.iloc[0]["heat_share"]), 1.0))

    def test_build_attribute_week_heat_frame_rejects_unmapped_sales_articles(
        self,
    ) -> None:
        sales = sample_attribute_article_week_sales()
        sales.loc[len(sales)] = [0, "0999999999", 1, 1, 0.10]

        with self.assertRaisesRegex(ValueError, "无法映射到属性边"):
            build_attribute_week_heat_frame(
                sales,
                sample_article_attribute_edges(),
                sample_attribute_nodes(),
            )

    def test_build_attribute_week_heat_frame_rejects_attribute_node_metadata_mismatch(
        self,
    ) -> None:
        for column, value in [
            ("attr_type", "product_type_name"),
            ("attr_value", "Noir"),
        ]:
            with self.subTest(column=column):
                nodes = sample_attribute_nodes()
                nodes.loc[
                    nodes["attr_id"] == "colour_group_name::Black",
                    column,
                ] = value

                with self.assertRaisesRegex(
                    ValueError,
                    f"colour_group_name::Black.*{column}",
                ):
                    build_attribute_week_heat_frame(
                        sample_attribute_article_week_sales(),
                        sample_article_attribute_edges(),
                        nodes,
                    )

    def test_validate_article_attribute_edges_for_heat_rejects_duplicate_edges(
        self,
    ) -> None:
        edges = sample_article_attribute_edges()
        edges.loc[len(edges)] = edges.loc[0]

        with self.assertRaisesRegex(ValueError, "article_id, attr_id"):
            validate_article_attribute_edges_for_heat(edges)

    def test_validate_article_attribute_edges_for_heat_rejects_inconsistent_attr_id(
        self,
    ) -> None:
        edges = sample_article_attribute_edges()
        edges.loc[2, "attr_id"] = "colour_group_name::Black"

        with self.assertRaisesRegex(
            ValueError,
            "colour_group_name::Black.*colour_group_name=Black.*colour_group_name=White",
        ):
            validate_article_attribute_edges_for_heat(edges)

    def test_validate_attribute_week_heat_rejects_duplicate_week_attr(self) -> None:
        heat = sample_attribute_week_heat()
        heat.loc[len(heat)] = heat.loc[0]

        with self.assertRaisesRegex(ValueError, "week_id, attr_id"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_incomplete_expected_panel(
        self,
    ) -> None:
        complete_heat = sample_attribute_week_heat()
        heat = complete_heat.drop(index=0).reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "完整 week_id x attr_id 面板"):
            validate_attribute_week_heat(
                heat,
                expected_week_ids=sorted(complete_heat["week_id"].unique()),
                expected_attribute_nodes=sample_attribute_nodes(),
            )

    def test_validate_attribute_week_heat_rejects_invalid_share_total(self) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_mask = (
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        )
        heat.loc[week0_colour_mask.idxmax(), "heat_share"] = 0.5

        with self.assertRaisesRegex(ValueError, "占比和不等于 1"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_inconsistent_type_total_heat(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_mask = (
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        )
        heat.loc[week0_colour_mask, "type_total_heat"] = 999

        with self.assertRaisesRegex(ValueError, "type_total_heat"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_inconsistent_heat_share(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices, "heat_share"] = [0.5, 0.5, 0.0]

        with self.assertRaisesRegex(ValueError, "heat_share"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_inconsistent_log_heat(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        heat.loc[0, "log_heat"] = 999.0

        with self.assertRaisesRegex(ValueError, "log_heat"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_uses_strict_absolute_share_tolerance(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices[0], "heat_share"] = (
            0.999995 - heat.loc[week0_colour_indices[1], "heat_share"]
        )

        with self.assertRaisesRegex(ValueError, "占比和不等于 1"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_duplicate_rank_in_type(self) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices[1], "rank_in_type"] = 1

        with self.assertRaisesRegex(ValueError, "重复 rank_in_type"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_rank_not_starting_at_one(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices, "rank_in_type"] = [2, 3, 4]

        with self.assertRaisesRegex(ValueError, "未从 1 开始"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_non_consecutive_rank_in_type(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices, "rank_in_type"] = [1, 3, 4]

        with self.assertRaisesRegex(ValueError, "rank_in_type 不连续"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_rank_in_type_sort_mismatch(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices, "rank_in_type"] = [2, 1, 3]

        with self.assertRaisesRegex(ValueError, "rank_in_type 排序"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_negative_heat_values(
        self,
    ) -> None:
        for column in ["heat_cnt", "type_total_heat", "heat_share", "log_heat"]:
            with self.subTest(column=column):
                heat = sample_attribute_week_heat()
                heat.loc[0, column] = -1

                with self.assertRaisesRegex(ValueError, column):
                    validate_attribute_week_heat(heat)


class AttributeWeekTargetFrameTests(unittest.TestCase):
    def test_build_attribute_week_target_frame_calculates_next_week_targets(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())

        self.assertEqual(target.columns.tolist(), list(ATTRIBUTE_WEEK_TARGET_COLUMNS))
        self.assertEqual(len(target), 6)
        self.assertEqual(set(target["week_id"]), {0})

        black = target[target["attr_id"] == "colour_group_name::Black"].iloc[0]
        self.assertEqual(int(black["heat_t"]), 2)
        self.assertEqual(int(black["heat_t1"]), 1)
        self.assertTrue(math.isclose(float(black["share_t"]), 2 / 3))
        self.assertTrue(math.isclose(float(black["share_t1"]), 1.0))
        self.assertTrue(
            math.isclose(
                float(black["target_growth"]),
                math.log((1.0 + 1e-6) / ((2 / 3) + 1e-6)),
            )
        )
        self.assertTrue(
            math.isclose(float(black["target_log_heat_t1"]), math.log1p(1))
        )
        self.assertEqual(int(black["target_rank_in_type_t1"]), 1)

    def test_validate_attribute_week_target_rejects_inconsistent_growth(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[0, "target_growth"] = 999.0

        with self.assertRaisesRegex(ValueError, "target_growth"):
            validate_attribute_week_target(target)

    def test_validate_attribute_week_target_rejects_non_positive_epsilon(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())

        with self.assertRaisesRegex(ValueError, "epsilon"):
            validate_attribute_week_target(target, epsilon=0)

    def test_validate_attribute_week_target_rejects_non_finite_numeric_values(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target["heat_t"] = target["heat_t"].astype("float64")
        target.loc[0, "heat_t"] = float("inf")

        with self.assertRaisesRegex(ValueError, "非有限"):
            validate_attribute_week_target(target)

    def test_validate_attribute_week_target_rejects_inconsistent_log_heat_t1(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[0, "target_log_heat_t1"] = 999.0

        with self.assertRaisesRegex(ValueError, "target_log_heat_t1"):
            validate_attribute_week_target(target)

    def test_validate_attribute_week_target_rejects_share_greater_than_one(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[0, "share_t"] = 1.1

        with self.assertRaisesRegex(ValueError, "share 大于 1"):
            validate_attribute_week_target(target)

    def test_validate_attribute_week_target_rejects_duplicate_week_attr(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[len(target)] = target.loc[0]

        with self.assertRaisesRegex(ValueError, "week_id, attr_id"):
            validate_attribute_week_target(target)


class TrendModelSamplesFrameTests(unittest.TestCase):
    def test_build_trend_model_samples_frame_uses_lags_and_targets(self) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat)
        samples = build_trend_model_samples_frame(
            heat,
            target,
            sample_attribute_nodes(),
            sample_attribute_hierarchy_edges(),
        )

        self.assertEqual(samples.columns.tolist(), list(TREND_MODEL_SAMPLE_COLUMNS))
        self.assertEqual(set(samples["week_id"]), {4})

        black = samples[samples["attr_id"] == "colour_group_name::Black"].iloc[0]
        self.assertEqual(int(black["heat_t"]), 8)
        self.assertEqual(int(black["heat_lag_1"]), 4)
        self.assertEqual(int(black["heat_lag_4"]), 2)
        self.assertTrue(math.isclose(float(black["heat_ma_4"]), (1 + 3 + 4 + 8) / 4))
        self.assertTrue(
            math.isclose(
                float(black["growth_lag_1"]),
                math.log((black["share_t"] + 1e-6) / (black["share_lag_1"] + 1e-6)),
            )
        )
        self.assertEqual(int(black["child_count"]), 1)
        self.assertEqual(int(black["parent_count"]), 0)
        self.assertEqual(int(black["degree"]), 1)
        self.assertEqual(int(black["history_total_heat_t"]), 18)
        self.assertEqual(int(black["history_active_weeks_t"]), 5)
        self.assertFalse(bool(black["is_trend_eligible_t"]))
        self.assertIn("target_growth", samples.columns)

    def test_validate_trend_model_samples_rejects_missing_target(self) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat).drop(columns=["target_growth"])

        with self.assertRaisesRegex(ValueError, "target_growth"):
            build_trend_model_samples_frame(
                heat,
                target,
                sample_attribute_nodes(),
                sample_attribute_hierarchy_edges(),
            )

    def test_build_trend_model_samples_frame_rejects_missing_target_key(self) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat)
        target = target[
            ~(
                (target["week_id"] == 4)
                & (target["attr_id"] == "colour_group_name::Black")
            )
        ].copy()

        with self.assertRaisesRegex(ValueError, "趋势标签表.*缺失.*1"):
            build_trend_model_samples_frame(
                heat,
                target,
                sample_attribute_nodes(),
                sample_attribute_hierarchy_edges(),
            )

    def test_build_trend_model_samples_frame_rejects_stale_target_values(
        self,
    ) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat)
        black_week4_mask = (
            (target["week_id"] == 4)
            & (target["attr_id"] == "colour_group_name::Black")
        )
        target.loc[black_week4_mask, "heat_t1"] = 9
        target.loc[black_week4_mask, "share_t1"] = 0.9
        target.loc[black_week4_mask, "target_log_heat_t1"] = math.log1p(9)
        target.loc[black_week4_mask, "target_growth"] = math.log(
            (0.9 + 1e-6)
            / (float(target.loc[black_week4_mask, "share_t"].iloc[0]) + 1e-6)
        )

        with self.assertRaisesRegex(ValueError, "属性趋势标签表.*不一致"):
            build_trend_model_samples_frame(
                heat,
                target,
                sample_attribute_nodes(),
                sample_attribute_hierarchy_edges(),
            )

    def test_build_trend_model_samples_frame_rejects_missing_target_before_min_lag(
        self,
    ) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat)
        target = target[
            ~(
                (target["week_id"] == 0)
                & (target["attr_id"] == "colour_group_name::Black")
            )
        ].copy()

        with self.assertRaisesRegex(ValueError, "属性趋势标签表.*不一致.*缺失.*1"):
            build_trend_model_samples_frame(
                heat,
                target,
                sample_attribute_nodes(),
                sample_attribute_hierarchy_edges(),
                min_lag_weeks=4,
            )

    def test_build_attribute_graph_features_frame_rejects_unknown_edge_node(
        self,
    ) -> None:
        edges = sample_attribute_hierarchy_edges()
        edges.loc[0, "parent_attr_id"] = "colour_group_name::Missing"

        with self.assertRaisesRegex(ValueError, "属性层级边表.*无法映射.*Missing"):
            build_attribute_graph_features_frame(sample_attribute_nodes(), edges)

    def test_build_attribute_graph_features_frame_rejects_unknown_child_node(
        self,
    ) -> None:
        edges = sample_attribute_hierarchy_edges()
        edges.loc[0, "child_attr_id"] = "colour_group_name::Missing"

        with self.assertRaisesRegex(ValueError, "属性层级边表.*无法映射.*Missing"):
            build_attribute_graph_features_frame(sample_attribute_nodes(), edges)

    def test_build_attribute_graph_features_frame_rejects_duplicate_edges(
        self,
    ) -> None:
        edges = sample_attribute_hierarchy_edges()
        edges.loc[len(edges)] = edges.loc[0]

        with self.assertRaisesRegex(
            ValueError,
            "parent_attr_id, child_attr_id, relation_type",
        ):
            build_attribute_graph_features_frame(sample_attribute_nodes(), edges)

    def test_build_attribute_graph_features_frame_rejects_non_positive_edge_weight(
        self,
    ) -> None:
        for edge_weight in [0, -1]:
            with self.subTest(edge_weight=edge_weight):
                edges = sample_attribute_hierarchy_edges()
                edges.loc[0, "edge_weight"] = edge_weight

                with self.assertRaisesRegex(ValueError, "edge_weight"):
                    build_attribute_graph_features_frame(sample_attribute_nodes(), edges)

    def test_build_trend_model_samples_frame_keeps_feature_window_fixed_at_four(
        self,
    ) -> None:
        heat = sample_long_attribute_week_heat()
        week6 = heat[heat["week_id"] == 5].copy()
        week6["week_id"] = 6
        heat = (
            pd.concat([heat, week6], ignore_index=True)
            .sort_values(
                ["week_id", "attr_type", "heat_cnt", "attr_id"],
                ascending=[True, True, False, True],
                ignore_index=True,
            )
        )
        heat["rank_in_type"] = (
            heat.groupby(["week_id", "attr_type"]).cumcount().add(1).astype("int64")
        )
        heat = heat.loc[:, list(ATTRIBUTE_WEEK_HEAT_COLUMNS)]
        target = build_attribute_week_target_frame(heat)

        samples = build_trend_model_samples_frame(
            heat,
            target,
            sample_attribute_nodes(),
            sample_attribute_hierarchy_edges(),
            min_lag_weeks=5,
        )

        self.assertEqual(set(samples["week_id"]), {5})
        for lag in range(1, 5):
            self.assertIn(f"heat_lag_{lag}", samples.columns)
            self.assertIn(f"share_lag_{lag}", samples.columns)

        black = samples[samples["attr_id"] == "colour_group_name::Black"].iloc[0]
        self.assertTrue(math.isclose(float(black["heat_ma_4"]), (3 + 4 + 8 + 4) / 4))

    def test_build_trend_model_samples_frame_rejects_too_small_min_lag_weeks(
        self,
    ) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat)

        with self.assertRaisesRegex(ValueError, "min_lag_weeks 必须大于等于 4"):
            build_trend_model_samples_frame(
                heat,
                target,
                sample_attribute_nodes(),
                sample_attribute_hierarchy_edges(),
                min_lag_weeks=1,
            )


class TrendModelSplitFrameTests(unittest.TestCase):
    def test_build_trend_model_split_frames_uses_time_boundaries(self) -> None:
        samples = sample_trend_model_samples_for_split()

        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )

        self.assertEqual(set(split_frames), {"train", "valid", "test"})
        self.assertEqual(split_frames["train"]["week_id"].min(), 4)
        self.assertEqual(split_frames["train"]["week_id"].max(), 15)
        self.assertEqual(split_frames["valid"]["week_id"].min(), 16)
        self.assertEqual(split_frames["valid"]["week_id"].max(), 19)
        self.assertEqual(split_frames["test"]["week_id"].min(), 20)
        self.assertEqual(split_frames["test"]["week_id"].max(), 23)
        self.assertEqual(set(split_frames["train"]["split"]), {"train"})
        self.assertEqual(set(split_frames["valid"]["split"]), {"valid"})
        self.assertEqual(set(split_frames["test"]["split"]), {"test"})

    def test_build_trend_model_split_frames_rejects_too_few_weeks(self) -> None:
        samples = sample_trend_model_samples_for_split()
        samples = samples[samples["week_id"] < 10].copy()

        with self.assertRaisesRegex(ValueError, "样本周数不足"):
            build_trend_model_split_frames(samples, valid_weeks=4, test_weeks=4)

    def test_build_trend_model_split_metadata_reports_ranges(self) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )

        metadata = build_trend_model_split_metadata(
            split_frames,
            input_path=Path("data/processed/features/trend_model_samples.parquet"),
            output_paths={
                "train": Path(
                    "data/processed/features/trend_model_samples_train.parquet"
                ),
                "valid": Path(
                    "data/processed/features/trend_model_samples_valid.parquet"
                ),
                "test": Path(
                    "data/processed/features/trend_model_samples_test.parquet"
                ),
            },
            valid_weeks=4,
            test_weeks=4,
        )

        self.assertEqual(metadata["split_strategy"], "time")
        self.assertEqual(metadata["valid_weeks"], 4)
        self.assertEqual(metadata["test_weeks"], 4)
        self.assertEqual(metadata["splits"]["train"]["week_min"], 4)
        self.assertEqual(metadata["splits"]["train"]["week_max"], 15)
        self.assertEqual(metadata["splits"]["train"]["rows"], 24)
        self.assertEqual(metadata["splits"]["valid"]["week_min"], 16)
        self.assertEqual(metadata["splits"]["test"]["week_max"], 23)

    def test_read_trend_model_split_preserves_columns_for_legal_parquet(self) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "trend_model_samples_train.parquet"
            write_trend_parquet(split_frames["train"], input_path)

            split = read_trend_model_split(input_path)

        self.assertEqual(split.columns.tolist(), list(TREND_MODEL_SPLIT_COLUMNS))
        self.assertEqual(set(split["split"]), {"train"})

    def test_read_trend_model_split_rejects_invalid_split_value(self) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )
        invalid_split = split_frames["train"].copy()
        invalid_split.loc[invalid_split.index[0], "split"] = "holdout"

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "trend_model_samples_train.parquet"
            write_trend_parquet(invalid_split, input_path)

            with self.assertRaisesRegex(ValueError, "非法 split"):
                read_trend_model_split(input_path)

    def test_read_trend_model_split_rejects_duplicate_week_attr(self) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )
        duplicate_split = pd.concat(
            [split_frames["train"], split_frames["train"].iloc[[0]]],
            ignore_index=True,
        )

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "trend_model_samples_train.parquet"
            write_trend_parquet(duplicate_split, input_path)

            with self.assertRaisesRegex(ValueError, "week_id, attr_id"):
                read_trend_model_split(input_path)


class LastWeekBaselineTests(unittest.TestCase):
    def test_last_week_params_are_stable(self) -> None:
        self.assertEqual(
            LAST_WEEK_PARAMS,
            {
                "model_name": "last_week",
                "formula": "pred_target_growth = growth_lag_1",
                "derived_formula": (
                    "pred_share_t1 = exp(pred_target_growth) * "
                    "(share_t + epsilon) - epsilon"
                ),
                "epsilon": 1e-6,
            },
        )

    def test_registry_lists_registered_models(self) -> None:
        self.assertEqual(
            list_trend_model_names(),
            (LAST_WEEK_MODEL_NAME, MOVING_AVERAGE_MODEL_NAME),
        )

    def test_registry_returns_last_week_trainer(self) -> None:
        trainer = get_trend_model_trainer(LAST_WEEK_MODEL_NAME)

        self.assertIsInstance(trainer, LastWeekTrainer)
        self.assertEqual(trainer.name, LAST_WEEK_MODEL_NAME)
        self.assertEqual(trainer.model_type, MODEL_TYPE_BASELINE)

    def test_moving_average_params_are_stable(self) -> None:
        self.assertEqual(
            MOVING_AVERAGE_PARAMS,
            {
                "model_name": "moving_average",
                "formula": "pred_target_growth = mean(growth_lag_1, growth_lag_2)",
                "derived_formula": (
                    "pred_share_t1 = exp(pred_target_growth) * "
                    "(share_t + epsilon) - epsilon"
                ),
                "epsilon": 1e-6,
                "growth_lags": ["growth_lag_1", "growth_lag_2"],
            },
        )
        self.assertEqual(MOVING_AVERAGE_GROWTH_LAGS, ("growth_lag_1", "growth_lag_2"))

    def test_predict_moving_average_rejects_non_finite_growth_lag(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )

        for bad_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad_value=bad_value):
                samples = pd.concat(split_frames.values(), ignore_index=True)
                samples.loc[samples.index[0], "growth_lag_2"] = bad_value

                with self.assertRaisesRegex(ValueError, "非有限|增长 lag"):
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

        self.assertEqual(result.params, MOVING_AVERAGE_PARAMS)
        self.assertIsNot(
            result.params["growth_lags"],
            MOVING_AVERAGE_PARAMS["growth_lags"],
        )

    def test_registry_returns_moving_average_trainer(self) -> None:
        trainer = get_trend_model_trainer(MOVING_AVERAGE_MODEL_NAME)

        self.assertIsInstance(trainer, MovingAverageTrainer)
        self.assertEqual(trainer.name, MOVING_AVERAGE_MODEL_NAME)
        self.assertEqual(trainer.model_type, MODEL_TYPE_BASELINE)

    def test_registry_rejects_unknown_model(self) -> None:
        with self.assertRaisesRegex(UnknownTrendModelError, "unknown_model"):
            get_trend_model_trainer("unknown_model")

    def test_derive_trend_model_output_paths_uses_model_name(self) -> None:
        paths = derive_trend_model_output_paths("last_week", Path("outputs/models"))

        self.assertEqual(paths["output_dir"], Path("outputs/models/last_week"))
        self.assertEqual(
            paths["predictions"], Path("outputs/models/last_week/predictions.csv")
        )
        self.assertEqual(paths["params"], Path("outputs/models/last_week/params.json"))
        self.assertEqual(
            paths["metadata"], Path("outputs/models/last_week/metadata.json")
        )
        self.assertEqual(
            derive_trend_model_output_paths("last_week")["output_dir"],
            OUTPUT_MODELS_DIR / "last_week",
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

        with self.assertRaisesRegex(ValueError, "model_name"):
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

        with self.assertRaisesRegex(ValueError, "model_name"):
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

        with self.assertRaisesRegex(ValueError, "week_id"):
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

        with self.assertRaisesRegex(ValueError, "metadata"):
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

        with self.assertRaisesRegex(ValueError, "week_id"):
            build_trend_train_metadata(result, context, paths)

    def test_write_trend_model_outputs_rejects_unsafe_artifact_path_before_writing(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        with TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "models"
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

            for unsafe_path in (
                "",
                "/tmp/leak.txt",
                "../leak.txt",
                "./leak.txt",
                "nested/./leak.txt",
                ".",
            ):
                with self.subTest(unsafe_path=unsafe_path):
                    bad_result = TrendTrainResult(
                        model_name=result.model_name,
                        model_type=result.model_type,
                        predictions=result.predictions,
                        params=result.params,
                        artifacts=(TrendArtifact(unsafe_path, "binary", b"bad"),),
                    )
                    metadata = build_trend_train_metadata(bad_result, context, paths)

                    with self.assertRaisesRegex(ValueError, "artifact"):
                        write_trend_model_outputs(bad_result, metadata, paths)

                    self.assertFalse(paths["predictions"].exists())

    def test_write_trend_model_outputs_rejects_bad_json_before_writing(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        with TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "models"
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

            with self.assertRaisesRegex(ValueError, "JSON"):
                write_trend_model_outputs(bad_result, metadata, paths)

            self.assertFalse(paths["predictions"].exists())

    def test_write_trend_model_outputs_rejects_bad_artifact_payload_before_writing(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        with TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "models"
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
            bad_artifacts = (
                TrendArtifact("bad.json", "json", {"bad": object()}),
                TrendArtifact("bad.bin", "binary", object()),
            )
            paths = derive_trend_model_output_paths("last_week", output_root)

            for bad_artifact in bad_artifacts:
                with self.subTest(bad_artifact=bad_artifact.relative_path):
                    bad_result = TrendTrainResult(
                        model_name=result.model_name,
                        model_type=result.model_type,
                        predictions=result.predictions,
                        params=result.params,
                        artifacts=(bad_artifact,),
                    )
                    metadata = build_trend_train_metadata(bad_result, context, paths)

                    with self.assertRaisesRegex(ValueError, "JSON|artifact"):
                        write_trend_model_outputs(bad_result, metadata, paths)

                    self.assertFalse(paths["predictions"].exists())

    def test_write_trend_model_outputs_does_not_publish_partial_files(
        self,
    ) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        with TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "models"
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

            with self.assertRaises(OSError):
                write_trend_model_outputs(result, metadata, paths)

            self.assertFalse(paths["predictions"].exists())
            self.assertFalse(paths["params"].exists())
            self.assertFalse((paths["output_dir"] / artifact.relative_path).exists())

    def test_run_trend_model_training_rejects_missing_input_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "split"):
            run_trend_model_training(
                LAST_WEEK_MODEL_NAME,
                input_paths={},
                output_root=Path("outputs/models"),
            )

    def test_run_trend_model_training_writes_standard_outputs(self) -> None:
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

            metadata = run_trend_model_training(
                LAST_WEEK_MODEL_NAME,
                input_paths=input_paths,
                output_root=tmp_path / "outputs" / "models",
            )

            output_dir = tmp_path / "outputs" / "models" / "last_week"
            self.assertTrue((output_dir / "predictions.csv").exists())
            self.assertTrue((output_dir / "params.json").exists())
            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertEqual(metadata["model_name"], LAST_WEEK_MODEL_NAME)
            self.assertEqual(metadata["model_type"], MODEL_TYPE_BASELINE)
            self.assertEqual(metadata["rows"], 40)
            self.assertEqual(metadata["extra_artifacts"], [])

    def test_run_trend_model_training_writes_moving_average_outputs(self) -> None:
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

            metadata = run_trend_model_training(
                MOVING_AVERAGE_MODEL_NAME,
                input_paths=input_paths,
                output_root=tmp_path / "outputs" / "models",
            )

            output_dir = tmp_path / "outputs" / "models" / "moving_average"
            self.assertTrue((output_dir / "predictions.csv").exists())
            self.assertTrue((output_dir / "params.json").exists())
            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertEqual(metadata["model_name"], MOVING_AVERAGE_MODEL_NAME)
            self.assertEqual(metadata["model_type"], MODEL_TYPE_BASELINE)
            self.assertEqual(metadata["rows"], 40)
            self.assertEqual(metadata["extra_artifacts"], [])
            params = json.loads((output_dir / "params.json").read_text(encoding="utf-8"))
            self.assertEqual(params["growth_lags"], ["growth_lag_1", "growth_lag_2"])

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

        self.assertIsInstance(result, TrendTrainResult)
        self.assertEqual(result.model_name, LAST_WEEK_MODEL_NAME)
        self.assertEqual(result.model_type, MODEL_TYPE_BASELINE)
        self.assertEqual(result.params, LAST_WEEK_PARAMS)
        self.assertEqual(result.artifacts, ())
        self.assertEqual(result.metadata, {})
        self.assertEqual(len(result.predictions), 40)

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

        self.assertIsInstance(result, TrendTrainResult)
        self.assertEqual(result.model_name, MOVING_AVERAGE_MODEL_NAME)
        self.assertEqual(result.model_type, MODEL_TYPE_BASELINE)
        self.assertEqual(result.params, MOVING_AVERAGE_PARAMS)
        self.assertEqual(result.artifacts, ())
        self.assertEqual(result.metadata, {})
        self.assertEqual(len(result.predictions), 40)

    def test_train_trend_model_main_preserves_argparse_usage_error_code(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")

        self.assertEqual(train_model.main(["--unknown"]), 2)

    def test_train_trend_model_main_rejects_unknown_model(self) -> None:
        train_model = importlib.import_module("10_train_trend_model")

        self.assertEqual(train_model.main(["--model", "unknown_model"]), 1)

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

            self.assertEqual(train_model.main(["--model", LAST_WEEK_MODEL_NAME]), 0)
        finally:
            train_model.run_trend_model_training = original_run_trend_model_training

        self.assertEqual(calls, [LAST_WEEK_MODEL_NAME])

    def test_predict_last_week_uses_growth_lag_1(self) -> None:
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        samples = pd.concat(split_frames.values(), ignore_index=True)

        predictions = predict_last_week(samples)

        self.assertEqual(
            predictions.columns.tolist(), list(TREND_MODEL_PREDICTION_COLUMNS)
        )
        self.assertEqual(set(predictions["model_name"]), {LAST_WEEK_MODEL_NAME})
        pd.testing.assert_series_equal(
            predictions["pred_target_growth"],
            samples.sort_values(["week_id", "attr_type", "attr_id"], ignore_index=True)[
                "growth_lag_1"
            ],
            check_names=False,
        )
        expected_share = (
            predictions["pred_target_growth"].map(math.exp)
            * (predictions["share_t"] + LAST_WEEK_PARAMS["epsilon"])
            - LAST_WEEK_PARAMS["epsilon"]
        )
        pd.testing.assert_series_equal(
            predictions["pred_share_t1"],
            expected_share,
            check_names=False,
        )

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
        expected_growth = ordered_samples.loc[
            :, ["growth_lag_1", "growth_lag_2"]
        ].mean(axis=1)

        self.assertEqual(
            predictions.columns.tolist(), list(TREND_MODEL_PREDICTION_COLUMNS)
        )
        self.assertEqual(set(predictions["model_name"]), {MOVING_AVERAGE_MODEL_NAME})
        pd.testing.assert_series_equal(
            predictions["pred_target_growth"],
            expected_growth,
            check_names=False,
        )
        expected_share = (
            predictions["pred_target_growth"].map(math.exp)
            * (predictions["share_t"] + MOVING_AVERAGE_PARAMS["epsilon"])
            - MOVING_AVERAGE_PARAMS["epsilon"]
        )
        pd.testing.assert_series_equal(
            predictions["pred_share_t1"],
            expected_share,
            check_names=False,
        )

    def test_predict_moving_average_rejects_missing_growth_lag(self) -> None:
        samples = sample_trend_model_samples_for_split().assign(split="train")
        samples = samples.drop(columns=["growth_lag_2"])

        with self.assertRaisesRegex(ValueError, "growth_lag_2"):
            predict_moving_average(samples)

    def test_predict_moving_average_rejects_illegal_split(self) -> None:
        samples = sample_trend_model_samples_for_split().assign(split="holdout")

        with self.assertRaisesRegex(ValueError, "非法 split"):
            predict_moving_average(samples)

    def test_predict_last_week_rejects_missing_split(self) -> None:
        samples = sample_trend_model_samples_for_split()

        with self.assertRaisesRegex(ValueError, "缺少必需列"):
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

        with self.assertRaisesRegex(ValueError, "趋势模型预测 split 与输入不一致"):
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

        with self.assertRaisesRegex(ValueError, "趋势模型预测字段与输入不一致"):
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

        with self.assertRaisesRegex(ValueError, "非有限"):
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

        with self.assertRaisesRegex(ValueError, "列"):
            validate_trend_model_predictions(predictions, samples)


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


class TrendModelSplitWriteTests(unittest.TestCase):
    def test_write_json_creates_parent_and_writes_sorted_keys(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "metadata.json"

            write_json({"b": 2, "a": 1}, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{\n  "a": 1,\n  "b": 2\n}\n',
            )


if __name__ == "__main__":
    unittest.main()
