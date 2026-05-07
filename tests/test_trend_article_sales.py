from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import fashion_trend.trend.article_sales as article_sales_module
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.transactions.weekly import read_weekly_transactions
from fashion_trend.trend.schema import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
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


class TestArticleWeekSalesFrame:
    def test_article_sales_module_does_not_reexport_weekly_transaction_reader(
        self,
    ) -> None:
        assert not hasattr(article_sales_module, "read_weekly_transactions")

    def test_read_weekly_transactions_rejects_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "weekly_transactions.parquet"

        with pytest.raises(FileNotFoundError, match="周级交易表不存在"):
            read_weekly_transactions(input_path)

    def test_read_weekly_transactions_reports_missing_required_field(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "weekly_transactions.parquet"
        sample_weekly_transactions().drop(columns=["price"]).to_parquet(input_path)

        with pytest.raises(ValueError, match="price"):
            read_weekly_transactions(input_path)

    def test_read_weekly_transactions_reports_unreadable_file(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "weekly_transactions.parquet"
        input_path.write_text("not a parquet file", encoding="utf-8")

        with pytest.raises(ValueError, match="无法读取周级交易表"):
            read_weekly_transactions(input_path)

    def test_build_article_week_sales_frame_aggregates_sales_by_week_and_article(
        self,
    ) -> None:
        sales = build_article_week_sales_frame(sample_weekly_transactions())

        assert sales.columns.tolist() == list(ARTICLE_WEEK_SALES_COLUMNS)
        assert sales[["week_id", "article_id", "sales_cnt", "sales_user_cnt"]].to_dict(
            "records"
        ) == [
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
        ]
        assert math.isclose(float(sales.loc[0, "sales_amount"]), 0.30)
        assert math.isclose(float(sales.loc[1, "sales_amount"]), 0.30)
        assert math.isclose(float(sales.loc[2, "sales_amount"]), 0.40)

    def test_build_article_week_sales_frame_preserves_article_id_as_string(
        self,
    ) -> None:
        sales = build_article_week_sales_frame(sample_weekly_transactions())

        assert sales["article_id"].dtype.name == "string"
        assert sales.loc[0, "article_id"] == "0108775015"

    def test_build_article_week_sales_frame_rejects_missing_required_values(
        self,
    ) -> None:
        transactions = sample_weekly_transactions()
        transactions.loc[0, "customer_id"] = pd.NA

        with pytest.raises(ValueError, match="customer_id"):
            build_article_week_sales_frame(transactions)

    def test_build_article_week_sales_frame_rejects_negative_price(self) -> None:
        transactions = sample_weekly_transactions()
        transactions.loc[0, "price"] = -0.01

        with pytest.raises(ValueError, match="price"):
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

        with pytest.raises(ValueError, match="week_id, article_id"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_non_positive_sales_count(
        self,
    ) -> None:
        sales = sample_article_week_sales()
        sales.loc[0, "sales_cnt"] = 0

        with pytest.raises(ValueError, match="sales_cnt"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_non_positive_sales_user_count(
        self,
    ) -> None:
        sales = sample_article_week_sales()
        sales.loc[0, "sales_user_cnt"] = 0

        with pytest.raises(ValueError, match="sales_user_cnt"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_negative_sales_amount(self) -> None:
        sales = sample_article_week_sales()
        sales.loc[0, "sales_amount"] = -0.01

        with pytest.raises(ValueError, match="sales_amount"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_missing_required_output_column(
        self,
    ) -> None:
        sales = sample_article_week_sales().drop(columns=["sales_amount"])

        with pytest.raises(ValueError, match="sales_amount"):
            validate_article_week_sales(sales)

    def test_validate_article_week_sales_rejects_missing_required_output_value(
        self,
    ) -> None:
        sales = sample_article_week_sales()
        sales.loc[0, "article_id"] = pd.NA

        with pytest.raises(ValueError, match="article_id"):
            validate_article_week_sales(sales)

    def test_read_article_week_sales_preserves_article_id_as_string(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "article_week_sales.csv"
        sample_article_week_sales().to_csv(input_path, index=False)

        sales = read_article_week_sales(input_path)

        assert sales["article_id"].dtype.name == "string"
        assert sales.loc[0, "article_id"] == "0108775015"

    def test_read_article_week_sales_rejects_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "article_week_sales.csv"

        with pytest.raises(FileNotFoundError, match="商品周销量表不存在"):
            read_article_week_sales(input_path)

    def test_read_article_week_sales_reports_missing_required_column(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "article_week_sales.csv"
        sample_article_week_sales().drop(columns=["sales_cnt"]).to_csv(
            input_path,
            index=False,
        )

        with pytest.raises(ValueError, match="sales_cnt"):
            read_article_week_sales(input_path)

    def test_read_article_week_sales_reports_invalid_numeric_value(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "article_week_sales.csv"
        input_path.write_text(
            "week_id,article_id,sales_cnt,sales_user_cnt,sales_amount\n"
            "0,0108775015,not_int,1,0.10\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="无法读取商品周销量表"):
            read_article_week_sales(input_path)


class TestTrendCsvWrite:
    def test_write_trend_csv_creates_parent_directory(self, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "attribute_week_heat.csv"

        write_csv_atomic(sample_article_week_sales(), output_path)

        assert output_path.exists()

    def test_write_trend_csv_replaces_existing_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "attribute_week_heat.csv"
        output_path.write_text("stale", encoding="utf-8")

        write_csv_atomic(sample_article_week_sales(), output_path)

        assert output_path.read_text(encoding="utf-8") != "stale"

    def test_write_trend_csv_removes_tmp_file_when_replace_fails(
        self,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "attribute_week_heat.csv"
        output_path.mkdir()

        with pytest.raises(OSError):
            write_csv_atomic(sample_article_week_sales(), output_path)

        assert not output_path.with_suffix(".csv.tmp").exists()

    def test_write_trend_csv_quotes_all_fields(self, tmp_path: Path) -> None:
        output_path = tmp_path / "attribute_week_heat.csv"
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

        write_csv_atomic(dataframe, output_path)

        lines = output_path.read_text(encoding="utf-8").splitlines()
        expected_header = ",".join(
            f'"{column}"' for column in ATTRIBUTE_WEEK_HEAT_COLUMNS
        )
        assert lines[0] == expected_header
        assert '"garment_group_name::Under-, Nightwear"' in lines[1]
        assert '"Under-, Nightwear"' in lines[1]
        assert not output_path.with_suffix(".csv.tmp").exists()
