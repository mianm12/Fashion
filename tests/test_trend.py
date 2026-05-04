from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fashion_trend.trend import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    build_article_week_sales_frame,
    build_attribute_week_heat_frame,
    read_weekly_transactions,
    validate_article_attribute_edges_for_heat,
    validate_article_week_sales,
    validate_attribute_week_heat,
    write_trend_csv,
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


def sample_attribute_week_heat() -> pd.DataFrame:
    return build_attribute_week_heat_frame(
        sample_attribute_article_week_sales(),
        sample_article_attribute_edges(),
    )


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
    def test_build_attribute_week_heat_frame_calculates_heat_metrics(self) -> None:
        heat = build_attribute_week_heat_frame(
            sample_attribute_article_week_sales(),
            sample_article_attribute_edges(),
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
            ],
        )
        self.assertTrue(math.isclose(float(week0_colour.iloc[0]["heat_share"]), 2 / 3))
        self.assertTrue(math.isclose(float(week0_colour.iloc[1]["heat_share"]), 1 / 3))
        self.assertTrue(
            math.isclose(float(week0_colour.iloc[0]["log_heat"]), math.log1p(2))
        )

        week1_product = heat[
            (heat["week_id"] == 1) & (heat["attr_type"] == "product_type_name")
        ]
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
                }
            ],
        )
        self.assertTrue(math.isclose(float(week1_product.iloc[0]["heat_share"]), 1.0))

    def test_build_attribute_week_heat_frame_rejects_unmapped_sales_articles(
        self,
    ) -> None:
        sales = sample_attribute_article_week_sales()
        sales.loc[len(sales)] = [0, "0999999999", 1, 1, 0.10]

        with self.assertRaisesRegex(ValueError, "无法映射到属性边"):
            build_attribute_week_heat_frame(sales, sample_article_attribute_edges())

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

    def test_validate_attribute_week_heat_rejects_invalid_share_total(self) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_mask = (
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        )
        heat.loc[week0_colour_mask.idxmax(), "heat_share"] = 0.5

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
        heat.loc[week0_colour_indices, "rank_in_type"] = [2, 3]

        with self.assertRaisesRegex(ValueError, "未从 1 开始"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_non_positive_heat_values(
        self,
    ) -> None:
        for column in ["heat_cnt", "heat_share"]:
            with self.subTest(column=column):
                heat = sample_attribute_week_heat()
                heat.loc[0, column] = 0

                with self.assertRaisesRegex(ValueError, column):
                    validate_attribute_week_heat(heat)


if __name__ == "__main__":
    unittest.main()
