from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fashion_trend.trend import (
    ARTICLE_WEEK_SALES_COLUMNS,
    build_article_week_sales_frame,
    validate_article_week_sales,
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


class ArticleWeekSalesFrameTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
