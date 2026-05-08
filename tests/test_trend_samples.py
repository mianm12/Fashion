from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import fashion_trend.trend.samples as samples_module
from fashion_trend.catalog.readers import read_attribute_hierarchy_edges
from fashion_trend.trend.samples import (
    build_attribute_graph_features_frame,
    build_trend_model_samples_frame,
)
from fashion_trend.trend.schema import (
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
)
from fashion_trend.trend.targets import build_attribute_week_target_frame
from tests.trend_samples import (
    sample_attribute_hierarchy_edges,
    sample_attribute_nodes,
    sample_long_attribute_week_heat,
)


class TestTrendModelSamplesFrame:
    def test_samples_module_does_not_reexport_attribute_hierarchy_reader(
        self,
    ) -> None:
        assert not hasattr(samples_module, "read_attribute_hierarchy_edges")

    def test_read_attribute_hierarchy_edges_preserves_string_dtypes(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "attribute_hierarchy_edges.csv"
        sample_attribute_hierarchy_edges().to_csv(input_path, index=False)

        edges = read_attribute_hierarchy_edges(input_path)

        assert edges["parent_attr_id"].dtype.name == "string"
        assert edges["child_attr_id"].dtype.name == "string"
        assert edges["parent_attr_type"].dtype.name == "string"
        assert edges["child_attr_type"].dtype.name == "string"
        assert edges["relation_type"].dtype.name == "string"
        assert edges.loc[0, "parent_attr_id"] == "colour_group_name::Black"

    def test_read_attribute_hierarchy_edges_reports_missing_required_column(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "attribute_hierarchy_edges.csv"
        sample_attribute_hierarchy_edges().drop(columns=["relation_type"]).to_csv(
            input_path,
            index=False,
        )

        with pytest.raises(ValueError, match="relation_type"):
            read_attribute_hierarchy_edges(input_path)

    def test_build_trend_model_samples_frame_uses_lags_and_targets(self) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat)
        samples = build_trend_model_samples_frame(
            heat,
            target,
            sample_attribute_nodes(),
            sample_attribute_hierarchy_edges(),
        )

        assert samples.columns.tolist() == list(TREND_MODEL_SAMPLE_COLUMNS)
        assert set(samples["week_id"]) == {4}

        black = samples[samples["attr_id"] == "colour_group_name::Black"].iloc[0]
        assert int(black["heat_t"]) == 8
        assert int(black["heat_lag_1"]) == 4
        assert int(black["heat_lag_4"]) == 2
        assert math.isclose(float(black["heat_ma_4"]), (1 + 3 + 4 + 8) / 4)
        assert math.isclose(
            float(black["growth_lag_1"]),
            math.log((black["share_t"] + 1e-6) / (black["share_lag_1"] + 1e-6)),
        )
        assert int(black["child_count"]) == 1
        assert int(black["parent_count"]) == 0
        assert int(black["degree"]) == 1
        assert int(black["history_total_heat_t"]) == 18
        assert int(black["history_active_weeks_t"]) == 5
        assert not bool(black["is_trend_eligible_t"])
        assert "target_growth" in samples.columns

    def test_validate_trend_model_samples_rejects_missing_target(self) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat).drop(columns=["target_growth"])

        with pytest.raises(ValueError, match="target_growth"):
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

        with pytest.raises(ValueError, match="趋势标签表.*缺失.*1"):
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
        black_week4_mask = (target["week_id"] == 4) & (
            target["attr_id"] == "colour_group_name::Black"
        )
        target.loc[black_week4_mask, "heat_t1"] = 9
        target.loc[black_week4_mask, "share_t1"] = 0.9
        target.loc[black_week4_mask, "target_log_heat_t1"] = math.log1p(9)
        target.loc[black_week4_mask, "target_growth"] = math.log(
            (0.9 + 1e-6)
            / (float(target.loc[black_week4_mask, "share_t"].iloc[0]) + 1e-6)
        )

        with pytest.raises(ValueError, match="属性趋势标签表.*不一致"):
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

        with pytest.raises(ValueError, match="属性趋势标签表.*不一致.*缺失.*1"):
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

        with pytest.raises(ValueError, match="属性层级边表.*无法映射.*Missing"):
            build_attribute_graph_features_frame(sample_attribute_nodes(), edges)

    def test_build_attribute_graph_features_frame_rejects_unknown_child_node(
        self,
    ) -> None:
        edges = sample_attribute_hierarchy_edges()
        edges.loc[0, "child_attr_id"] = "colour_group_name::Missing"

        with pytest.raises(ValueError, match="属性层级边表.*无法映射.*Missing"):
            build_attribute_graph_features_frame(sample_attribute_nodes(), edges)

    def test_build_attribute_graph_features_frame_rejects_duplicate_edges(
        self,
    ) -> None:
        edges = sample_attribute_hierarchy_edges()
        edges.loc[len(edges)] = edges.loc[0]

        with pytest.raises(
            ValueError,
            match="parent_attr_id, child_attr_id, relation_type",
        ):
            build_attribute_graph_features_frame(sample_attribute_nodes(), edges)

    @pytest.mark.parametrize("edge_weight", [0, -1])
    def test_build_attribute_graph_features_frame_rejects_non_positive_edge_weight(
        self,
        edge_weight: int,
    ) -> None:
        edges = sample_attribute_hierarchy_edges()
        edges.loc[0, "edge_weight"] = edge_weight

        with pytest.raises(ValueError, match="edge_weight"):
            build_attribute_graph_features_frame(sample_attribute_nodes(), edges)

    def test_build_trend_model_samples_frame_keeps_feature_window_fixed_at_four(
        self,
    ) -> None:
        heat = sample_long_attribute_week_heat()
        week6 = heat[heat["week_id"] == 5].copy()
        week6["week_id"] = 6
        heat = pd.concat([heat, week6], ignore_index=True).sort_values(
            ["week_id", "attr_type", "heat_cnt", "attr_id"],
            ascending=[True, True, False, True],
            ignore_index=True,
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

        assert set(samples["week_id"]) == {5}
        for lag in range(1, 5):
            assert f"heat_lag_{lag}" in samples.columns
            assert f"share_lag_{lag}" in samples.columns

        black = samples[samples["attr_id"] == "colour_group_name::Black"].iloc[0]
        assert math.isclose(float(black["heat_ma_4"]), (3 + 4 + 8 + 4) / 4)

    def test_build_trend_model_samples_frame_rejects_too_small_min_lag_weeks(
        self,
    ) -> None:
        heat = sample_long_attribute_week_heat()
        target = build_attribute_week_target_frame(heat)

        with pytest.raises(ValueError, match="min_lag_weeks 必须大于等于 4"):
            build_trend_model_samples_frame(
                heat,
                target,
                sample_attribute_nodes(),
                sample_attribute_hierarchy_edges(),
                min_lag_weeks=1,
            )
