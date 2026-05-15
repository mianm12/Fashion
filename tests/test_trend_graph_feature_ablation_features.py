from __future__ import annotations

import math

import pandas as pd
import pytest

import experiments.trend_graph_feature_ablation.build_features as build_features_module
from experiments.trend_graph_feature_ablation.build_features import (
    build_graph_context_features,
)
from experiments.trend_graph_feature_ablation.feature_groups import (
    HIERARCHY_CONTEXT_FEATURES,
    LIGHT_STRUCTURE_FEATURES,
    SIBLING_COMPETITION_FEATURES,
)


def _sample_graph_samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": [4, 4, 4, 4, 4, 4],
            "attr_id": [
                "parent::A",
                "parent::B",
                "child::X",
                "child::Y",
                "child::Z",
                "orphan::N",
            ],
            "attr_type": ["parent", "parent", "child", "child", "child", "orphan"],
            "heat_t": [10.0, 30.0, 20.0, 5.0, 40.0, 7.0],
            "share_t": [0.25, 0.75, 0.80, 0.20, 0.50, 0.10],
            "growth_lag_1": [0.10, 0.30, 0.50, -0.10, 0.20, 0.00],
            "rank_in_type_t": [2, 1, 1, 3, 2, 1],
            "target_growth": [999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
            "target_log_heat_t1": [999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
            "target_rank_in_type_t1": [999, 999, 999, 999, 999, 999],
        }
    )


def _sample_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_attr_id": [
                "parent::A",
                "parent::B",
                "parent::A",
                "parent::A",
                "parent::B",
            ],
            "child_attr_id": [
                "child::X",
                "child::X",
                "child::Y",
                "child::Z",
                "child::Z",
            ],
            "parent_attr_type": ["parent", "parent", "parent", "parent", "parent"],
            "child_attr_type": ["child", "child", "child", "child", "child"],
            "relation_type": [
                "contains",
                "contains",
                "contains",
                "contains",
                "contains",
            ],
            "edge_weight": [2.0, 6.0, 2.0, 4.0, 1.0],
        }
    )


def _features_by_attr() -> pd.DataFrame:
    return build_graph_context_features(
        _sample_graph_samples(),
        _sample_edges(),
    ).set_index("attr_id")


def test_parent_context_uses_edge_weight_normalized_average() -> None:
    child_x = _features_by_attr().loc["child::X"]

    assert math.isclose(child_x["kg_parent_share_t_wavg"], 0.625)
    assert math.isclose(child_x["kg_parent_growth_lag_1_wavg"], 0.25)
    assert math.isclose(child_x["kg_parent_rank_pct_t_wavg"], 0.25)
    assert math.isclose(child_x["kg_self_parent_share_gap_t"], 0.175)
    assert math.isclose(child_x["kg_self_parent_growth_gap_lag_1"], 0.25)
    assert int(child_x["kg_has_parent"]) == 1
    assert int(child_x["kg_is_root_attr"]) == 0
    assert math.isclose(child_x["kg_parent_edge_weight_sum"], 8.0)
    assert math.isclose(child_x["kg_parent_edge_weight_log"], math.log1p(8.0))


def test_child_context_uses_edge_weight_normalized_average() -> None:
    parent_a = _features_by_attr().loc["parent::A"]

    assert math.isclose(parent_a["kg_child_share_t_wavg"], 0.50)
    assert math.isclose(parent_a["kg_child_growth_lag_1_wavg"], 0.20)
    assert math.isclose(parent_a["kg_child_rank_pct_t_wavg"], 0.50)
    assert math.isclose(parent_a["kg_self_child_share_gap_t"], -0.25)
    assert math.isclose(parent_a["kg_self_child_growth_gap_lag_1"], -0.10)
    assert int(parent_a["kg_has_child"]) == 1
    assert int(parent_a["kg_is_leaf_attr"]) == 0
    assert math.isclose(parent_a["kg_child_edge_weight_sum"], 8.0)
    assert math.isclose(parent_a["kg_child_edge_weight_log"], math.log1p(8.0))


def test_sibling_context_uses_fixed_shared_parent_weight_formula() -> None:
    child_x = _features_by_attr().loc["child::X"]

    # X shares A with Y (2*2=4), A with Z (2*4=8), and B with Z (6*1=6).
    # Sibling weights after summing shared parents: Y=4, Z=14.
    expected_share = (0.20 * 4.0 + 0.50 * 14.0) / 18.0
    expected_growth = (-0.10 * 4.0 + 0.20 * 14.0) / 18.0

    assert int(child_x["kg_has_sibling"]) == 1
    assert int(child_x["kg_sibling_count"]) == 2
    assert math.isclose(child_x["kg_sibling_share_t_wavg"], expected_share)
    assert math.isclose(child_x["kg_sibling_growth_lag_1_wavg"], expected_growth)
    assert math.isclose(
        child_x["kg_self_vs_sibling_share_gap_t"], 0.80 - expected_share
    )
    assert math.isclose(
        child_x["kg_self_vs_sibling_growth_gap_lag_1"],
        0.50 - expected_growth,
    )


def test_missing_neighbors_are_zero_filled_and_marked() -> None:
    features = _features_by_attr()
    parent_b = features.loc["parent::B"]
    orphan = features.loc["orphan::N"]

    assert int(parent_b["kg_has_parent"]) == 0
    assert int(parent_b["kg_is_root_attr"]) == 1
    assert int(parent_b["kg_has_sibling"]) == 0
    assert parent_b["kg_sibling_share_t_wavg"] == 0
    assert parent_b["kg_self_parent_share_gap_t"] == 0

    assert int(orphan["kg_has_parent"]) == 0
    assert int(orphan["kg_has_child"]) == 0
    assert int(orphan["kg_is_root_attr"]) == 1
    assert int(orphan["kg_is_leaf_attr"]) == 1
    assert int(orphan["kg_has_sibling"]) == 0
    zero_filled_columns = [
        column
        for column in orphan.index
        if column.startswith("kg_")
        and column not in {"kg_is_root_attr", "kg_is_leaf_attr"}
    ]
    assert orphan[zero_filled_columns].eq(0).all()


def test_output_keeps_sample_order_and_expected_columns_only() -> None:
    samples = _sample_graph_samples()
    features = build_graph_context_features(samples, _sample_edges())
    expected_columns = [
        "week_id",
        "attr_id",
        *HIERARCHY_CONTEXT_FEATURES,
        *SIBLING_COMPETITION_FEATURES,
        *LIGHT_STRUCTURE_FEATURES,
    ]

    assert list(features.columns) == expected_columns
    assert list(features["week_id"]) == list(samples["week_id"])
    assert list(features["attr_id"]) == list(samples["attr_id"])
    assert len(features) == len(samples)
    assert "rank_pct_t" not in features.columns


def test_target_week_columns_do_not_affect_graph_features() -> None:
    samples = _sample_graph_samples()
    baseline = build_graph_context_features(samples, _sample_edges())
    mutated = samples.copy()
    mutated["target_growth"] = -999.0
    mutated["target_log_heat_t1"] = -999.0
    mutated["target_rank_in_type_t1"] = -999

    changed = build_graph_context_features(mutated, _sample_edges())

    pd.testing.assert_frame_equal(changed, baseline)


def test_repeated_attr_ids_are_aggregated_by_week_without_row_amplification() -> None:
    samples = pd.concat(
        [
            _sample_graph_samples(),
            _sample_graph_samples().assign(
                week_id=5,
                share_t=[0.10, 0.90, 0.40, 0.30, 0.20, 0.05],
                growth_lag_1=[0.20, 0.40, 0.60, 0.10, -0.20, 0.00],
            ),
        ],
        ignore_index=True,
    )

    features = build_graph_context_features(samples, _sample_edges())
    child_x = features.set_index(["week_id", "attr_id"]).loc[(5, "child::X")]

    assert len(features) == len(samples)
    assert list(features["week_id"]) == list(samples["week_id"])
    assert list(features["attr_id"]) == list(samples["attr_id"])
    assert math.isclose(child_x["kg_parent_share_t_wavg"], 0.70)
    assert math.isclose(child_x["kg_self_parent_share_gap_t"], -0.30)


def test_duplicate_edges_accumulate_weight_in_one_hop_aggregation() -> None:
    edges = _sample_edges()
    duplicate = edges.iloc[[0]].copy()
    duplicate["relation_type"] = "also_contains"
    duplicate["edge_weight"] = 8.0
    edges = pd.concat([edges, duplicate], ignore_index=True)

    features = build_graph_context_features(_sample_graph_samples(), edges)
    child_x = features.set_index("attr_id").loc["child::X"]

    assert math.isclose(child_x["kg_parent_share_t_wavg"], 0.4375)
    assert math.isclose(child_x["kg_parent_edge_weight_sum"], 16.0)


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("heat_t", float("nan")),
        ("share_t", float("inf")),
        ("growth_lag_1", "-"),
        ("rank_in_type_t", "first"),
    ),
)
def test_source_dynamic_columns_must_be_numeric_and_finite(
    column: str,
    value: object,
) -> None:
    samples = _sample_graph_samples()
    samples[column] = samples[column].astype(object)
    samples.loc[0, column] = value

    with pytest.raises(ValueError, match=f"graph context source.*{column}"):
        build_graph_context_features(samples, _sample_edges())


def test_output_alignment_guard_rejects_row_amplification(monkeypatch) -> None:
    def duplicate_structure_rows(*args, **kwargs) -> pd.DataFrame:
        frame = build_features_module._zero_feature_frame(
            _sample_graph_samples(),
            list(LIGHT_STRUCTURE_FEATURES),
        )
        return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    monkeypatch.setattr(
        build_features_module,
        "_build_light_structure_features",
        duplicate_structure_rows,
    )

    with pytest.raises(ValueError, match="row alignment"):
        build_graph_context_features(_sample_graph_samples(), _sample_edges())


@pytest.mark.parametrize("edge_weight", [0, -1, float("nan"), "heavy"])
def test_edge_weight_must_be_positive_and_numeric(edge_weight: object) -> None:
    edges = _sample_edges()
    edges["edge_weight"] = edges["edge_weight"].astype(object)
    edges.loc[0, "edge_weight"] = edge_weight

    with pytest.raises(ValueError, match="edge_weight"):
        build_graph_context_features(_sample_graph_samples(), edges)
