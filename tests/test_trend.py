from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fashion_trend.trend import (
    ARTICLE_WEEK_SALES_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
    build_article_week_sales_frame,
    build_attribute_graph_features_frame,
    build_attribute_week_heat_frame,
    build_attribute_week_target_frame,
    build_trend_model_samples_frame,
    read_article_attribute_edges,
    read_attribute_hierarchy_edges,
    read_attribute_nodes,
    read_article_week_sales,
    read_attribute_week_target,
    read_weekly_transactions,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_article_week_sales,
    validate_attribute_week_heat,
    validate_attribute_week_target,
    validate_trend_model_samples,
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


if __name__ == "__main__":
    unittest.main()
