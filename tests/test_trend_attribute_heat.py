from __future__ import annotations

import math
from pathlib import Path

import pytest

from fashion_trend.trend import (
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    build_attribute_week_heat_frame,
    read_article_attribute_edges,
    validate_article_attribute_edges_for_heat,
    validate_attribute_week_heat,
)
from tests.trend_samples import (
    sample_article_attribute_edges,
    sample_attribute_article_week_sales,
    sample_attribute_nodes,
    sample_attribute_week_heat,
)


class TestAttributeWeekHeatFrame:
    def test_read_article_attribute_edges_preserves_string_dtypes(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "article_attribute_edges.csv"
        sample_article_attribute_edges().to_csv(input_path, index=False)

        edges = read_article_attribute_edges(input_path)

        assert edges["article_id"].dtype.name == "string"
        assert edges["attr_id"].dtype.name == "string"
        assert edges["attr_type"].dtype.name == "string"
        assert edges["attr_value"].dtype.name == "string"
        assert edges.loc[0, "article_id"] == "0108775015"

    def test_read_article_attribute_edges_rejects_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "article_attribute_edges.csv"

        with pytest.raises(FileNotFoundError, match="商品-属性边表不存在"):
            read_article_attribute_edges(input_path)

    def test_read_article_attribute_edges_reports_missing_required_column(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "article_attribute_edges.csv"
        sample_article_attribute_edges().drop(columns=["attr_value"]).to_csv(
            input_path,
            index=False,
        )

        with pytest.raises(ValueError, match="attr_value"):
            read_article_attribute_edges(input_path)

    def test_build_attribute_week_heat_frame_builds_complete_attribute_week_panel(
        self,
    ) -> None:
        heat = build_attribute_week_heat_frame(
            sample_attribute_article_week_sales(),
            sample_article_attribute_edges(),
            sample_attribute_nodes(),
        )

        assert len(heat) == 12
        assert heat.columns.tolist() == list(ATTRIBUTE_WEEK_HEAT_COLUMNS)
        assert set(heat["week_id"]) == {0, 1}
        assert set(heat["attr_id"]) == set(sample_attribute_nodes()["attr_id"])

        zero_row = heat[
            (heat["week_id"] == 0) & (heat["attr_id"] == "colour_group_name::Blue")
        ].iloc[0]
        assert int(zero_row["heat_cnt"]) == 0
        assert float(zero_row["heat_share"]) == 0.0
        assert float(zero_row["log_heat"]) == 0.0

    def test_build_attribute_week_heat_frame_calculates_heat_metrics(self) -> None:
        heat = build_attribute_week_heat_frame(
            sample_attribute_article_week_sales(),
            sample_article_attribute_edges(),
            sample_attribute_nodes(),
        )

        assert heat.columns.tolist() == list(ATTRIBUTE_WEEK_HEAT_COLUMNS)

        week0_colour = heat[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ].sort_values("rank_in_type")
        assert week0_colour[
            ["attr_id", "heat_cnt", "type_total_heat", "rank_in_type"]
        ].to_dict("records") == [
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
        ]
        assert math.isclose(float(week0_colour.iloc[0]["heat_share"]), 2 / 3)
        assert math.isclose(float(week0_colour.iloc[1]["heat_share"]), 1 / 3)
        assert math.isclose(float(week0_colour.iloc[0]["log_heat"]), math.log1p(2))

        week1_product = heat[
            (heat["week_id"] == 1) & (heat["attr_type"] == "product_type_name")
        ].sort_values("rank_in_type")
        assert week1_product[
            ["attr_id", "heat_cnt", "type_total_heat", "rank_in_type"]
        ].to_dict("records") == [
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
        ]
        assert math.isclose(float(week1_product.iloc[0]["heat_share"]), 1.0)

    def test_build_attribute_week_heat_frame_rejects_unmapped_sales_articles(
        self,
    ) -> None:
        sales = sample_attribute_article_week_sales()
        sales.loc[len(sales)] = [0, "0999999999", 1, 1, 0.10]

        with pytest.raises(ValueError, match="无法映射到属性边"):
            build_attribute_week_heat_frame(
                sales,
                sample_article_attribute_edges(),
                sample_attribute_nodes(),
            )

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("attr_type", "product_type_name"),
            ("attr_value", "Noir"),
        ],
    )
    def test_build_attribute_week_heat_frame_rejects_attribute_node_metadata_mismatch(
        self,
        column: str,
        value: str,
    ) -> None:
        nodes = sample_attribute_nodes()
        nodes.loc[
            nodes["attr_id"] == "colour_group_name::Black",
            column,
        ] = value

        with pytest.raises(
            ValueError,
            match=f"colour_group_name::Black.*{column}",
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

        with pytest.raises(ValueError, match="article_id, attr_id"):
            validate_article_attribute_edges_for_heat(edges)

    def test_validate_article_attribute_edges_for_heat_rejects_inconsistent_attr_id(
        self,
    ) -> None:
        edges = sample_article_attribute_edges()
        edges.loc[2, "attr_id"] = "colour_group_name::Black"

        with pytest.raises(
            ValueError,
            match="colour_group_name::Black.*colour_group_name=Black.*colour_group_name=White",
        ):
            validate_article_attribute_edges_for_heat(edges)

    def test_validate_attribute_week_heat_rejects_duplicate_week_attr(self) -> None:
        heat = sample_attribute_week_heat()
        heat.loc[len(heat)] = heat.loc[0]

        with pytest.raises(ValueError, match="week_id, attr_id"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_incomplete_expected_panel(
        self,
    ) -> None:
        complete_heat = sample_attribute_week_heat()
        heat = complete_heat.drop(index=0).reset_index(drop=True)

        with pytest.raises(ValueError, match="完整 week_id x attr_id 面板"):
            validate_attribute_week_heat(
                heat,
                expected_week_ids=sorted(complete_heat["week_id"].unique()),
                expected_attribute_nodes=sample_attribute_nodes(),
            )

    def test_validate_attribute_week_heat_rejects_invalid_share_total(self) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_mask = (heat["week_id"] == 0) & (
            heat["attr_type"] == "colour_group_name"
        )
        heat.loc[week0_colour_mask.idxmax(), "heat_share"] = 0.5

        with pytest.raises(ValueError, match="占比和不等于 1"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_inconsistent_type_total_heat(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_mask = (heat["week_id"] == 0) & (
            heat["attr_type"] == "colour_group_name"
        )
        heat.loc[week0_colour_mask, "type_total_heat"] = 999

        with pytest.raises(ValueError, match="type_total_heat"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_inconsistent_heat_share(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices, "heat_share"] = [0.5, 0.5, 0.0]

        with pytest.raises(ValueError, match="heat_share"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_inconsistent_log_heat(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        heat.loc[0, "log_heat"] = 999.0

        with pytest.raises(ValueError, match="log_heat"):
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

        with pytest.raises(ValueError, match="占比和不等于 1"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_duplicate_rank_in_type(self) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices[1], "rank_in_type"] = 1

        with pytest.raises(ValueError, match="重复 rank_in_type"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_rank_not_starting_at_one(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices, "rank_in_type"] = [2, 3, 4]

        with pytest.raises(ValueError, match="未从 1 开始"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_non_consecutive_rank_in_type(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices, "rank_in_type"] = [1, 3, 4]

        with pytest.raises(ValueError, match="rank_in_type 不连续"):
            validate_attribute_week_heat(heat)

    def test_validate_attribute_week_heat_rejects_rank_in_type_sort_mismatch(
        self,
    ) -> None:
        heat = sample_attribute_week_heat()
        week0_colour_indices = heat.index[
            (heat["week_id"] == 0) & (heat["attr_type"] == "colour_group_name")
        ]
        heat.loc[week0_colour_indices, "rank_in_type"] = [2, 1, 3]

        with pytest.raises(ValueError, match="rank_in_type 排序"):
            validate_attribute_week_heat(heat)

    @pytest.mark.parametrize(
        "column",
        ["heat_cnt", "type_total_heat", "heat_share", "log_heat"],
    )
    def test_validate_attribute_week_heat_rejects_negative_heat_values(
        self,
        column: str,
    ) -> None:
        heat = sample_attribute_week_heat()
        heat.loc[0, column] = -1

        with pytest.raises(ValueError, match=column):
            validate_attribute_week_heat(heat)
