from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.trend_graph_feature_ablation.feature_groups import (
    HIERARCHY_CONTEXT_FEATURES,
    LIGHT_STRUCTURE_FEATURES,
    SIBLING_COMPETITION_FEATURES,
)
from fashion_trend.foundation.dataframe import validate_required_columns

GRAPH_CONTEXT_SOURCE_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "heat_t",
    "share_t",
    "growth_lag_1",
    "rank_in_type_t",
)
EDGE_COLUMNS: tuple[str, ...] = (
    "parent_attr_id",
    "child_attr_id",
    "edge_weight",
)
_DYNAMIC_CONTEXT_COLUMNS: tuple[str, ...] = (
    "heat_t",
    "share_t",
    "growth_lag_1",
    "rank_pct_t",
)
_NUMERIC_SOURCE_COLUMNS: tuple[str, ...] = (
    "heat_t",
    "share_t",
    "growth_lag_1",
    "rank_in_type_t",
)


def build_graph_context_features(
    samples_all: pd.DataFrame,
    hierarchy_edges: pd.DataFrame,
) -> pd.DataFrame:
    """构建只依赖当前周及历史样本列的一跳图谱上下文特征。"""

    validate_required_columns(
        samples_all,
        GRAPH_CONTEXT_SOURCE_COLUMNS,
        source_name="graph context samples",
    )
    validate_required_columns(
        hierarchy_edges,
        EDGE_COLUMNS,
        source_name="attribute hierarchy edges",
    )
    edges = _prepare_hierarchy_edges(hierarchy_edges)
    base = _build_base_context_frame(samples_all)

    parent_features = _aggregate_neighbor_features(
        base,
        edges,
        target_column="child_attr_id",
        neighbor_column="parent_attr_id",
        prefix="kg_parent",
    )
    child_features = _aggregate_neighbor_features(
        base,
        edges,
        target_column="parent_attr_id",
        neighbor_column="child_attr_id",
        prefix="kg_child",
    )
    sibling_features = _aggregate_sibling_features(base, edges)
    structure_features = _build_light_structure_features(base, edges)

    output_columns = [
        "week_id",
        "attr_id",
        *HIERARCHY_CONTEXT_FEATURES,
        *SIBLING_COMPETITION_FEATURES,
        *LIGHT_STRUCTURE_FEATURES,
    ]
    features = base.loc[:, ["week_id", "attr_id", "share_t", "growth_lag_1"]].copy()
    for frame in (
        parent_features,
        child_features,
        sibling_features,
        structure_features,
    ):
        features = features.merge(frame, on=["week_id", "attr_id"], how="left")

    kg_columns = [
        *HIERARCHY_CONTEXT_FEATURES,
        *SIBLING_COMPETITION_FEATURES,
        *LIGHT_STRUCTURE_FEATURES,
    ]
    for column in kg_columns:
        if column not in features.columns:
            features[column] = 0.0
    features[kg_columns] = features[kg_columns].fillna(0.0)
    _add_hierarchy_gap_features(features)
    features = features.loc[:, output_columns]
    _validate_output_alignment(features, samples_all)
    _validate_kg_features(features, kg_columns)
    return features


def _prepare_hierarchy_edges(hierarchy_edges: pd.DataFrame) -> pd.DataFrame:
    edges = hierarchy_edges.loc[:, list(EDGE_COLUMNS)].copy()
    try:
        edges["edge_weight"] = pd.to_numeric(edges["edge_weight"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "attribute hierarchy edges contain non-numeric edge_weight"
        ) from exc
    if not np.isfinite(edges["edge_weight"].to_numpy(dtype=float)).all():
        raise ValueError("attribute hierarchy edges contain non-finite edge_weight")
    if bool(edges["edge_weight"].le(0).any()):
        raise ValueError("attribute hierarchy edges contain non-positive edge_weight")
    return edges


def _build_base_context_frame(samples_all: pd.DataFrame) -> pd.DataFrame:
    base = samples_all.loc[:, list(GRAPH_CONTEXT_SOURCE_COLUMNS)].copy()
    _coerce_numeric_source_columns(base)
    type_counts = base.groupby(["week_id", "attr_type"])["attr_id"].transform("count")
    denominator = (type_counts - 1).clip(lower=1)
    base["rank_pct_t"] = (
        base["rank_in_type_t"].astype(float) - 1.0
    ) / denominator.astype(float)
    return base


def _coerce_numeric_source_columns(base: pd.DataFrame) -> None:
    for column in _NUMERIC_SOURCE_COLUMNS:
        try:
            base[column] = pd.to_numeric(base[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"graph context source column {column} contains non-numeric values"
            ) from exc
        if not np.isfinite(base[column].to_numpy(dtype=float)).all():
            raise ValueError(
                f"graph context source column {column} contains non-finite values"
            )


def _aggregate_neighbor_features(
    base: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    target_column: str,
    neighbor_column: str,
    prefix: str,
) -> pd.DataFrame:
    links = edges.loc[:, [target_column, neighbor_column, "edge_weight"]].rename(
        columns={
            target_column: "target_attr_id",
            neighbor_column: "neighbor_attr_id",
        }
    )
    neighbor_values = base.loc[
        :, ["week_id", "attr_id", *_DYNAMIC_CONTEXT_COLUMNS]
    ].rename(columns={"attr_id": "neighbor_attr_id"})
    joined = links.merge(neighbor_values, on="neighbor_attr_id", how="inner")
    output_names = {
        "heat_t": f"{prefix}_heat_t_wavg",
        "share_t": f"{prefix}_share_t_wavg",
        "growth_lag_1": f"{prefix}_growth_lag_1_wavg",
        "rank_pct_t": f"{prefix}_rank_pct_t_wavg",
    }

    if joined.empty:
        return _zero_feature_frame(base, list(output_names.values()))

    weighted = joined.loc[:, ["week_id", "target_attr_id"]].copy()
    for source_column, output_column in output_names.items():
        weighted[output_column] = joined[source_column].astype(float) * joined[
            "edge_weight"
        ].astype(float)
    aggregated = weighted.groupby(
        ["week_id", "target_attr_id"],
        as_index=False,
        sort=False,
    ).sum()
    weight_sum = (
        joined.groupby(["week_id", "target_attr_id"], as_index=False, sort=False)[
            "edge_weight"
        ]
        .sum()
        .rename(columns={"edge_weight": "_edge_weight_sum"})
    )
    aggregated = aggregated.merge(
        weight_sum,
        on=["week_id", "target_attr_id"],
        how="inner",
    )
    for output_column in output_names.values():
        aggregated[output_column] = (
            aggregated[output_column] / aggregated["_edge_weight_sum"]
        )
    return aggregated.rename(columns={"target_attr_id": "attr_id"}).drop(
        columns=["_edge_weight_sum"]
    )


def _aggregate_sibling_features(
    base: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    sibling_links = _build_sibling_links(edges)
    if sibling_links.empty:
        return _zero_sibling_feature_frame(base)

    joined = _join_sibling_context(base, sibling_links)
    if joined.empty:
        return _zero_sibling_feature_frame(base)

    grouped = joined.groupby(["week_id", "current_attr_id"], sort=False)
    return pd.DataFrame(
        [
            _summarize_sibling_group(week_id, current_attr_id, group)
            for (week_id, current_attr_id), group in grouped
        ]
    )


def _build_sibling_links(edges: pd.DataFrame) -> pd.DataFrame:
    current_edges = edges.loc[
        :, ["parent_attr_id", "child_attr_id", "edge_weight"]
    ].rename(
        columns={
            "child_attr_id": "current_attr_id",
            "edge_weight": "current_parent_edge_weight",
        }
    )
    sibling_edges = edges.loc[
        :, ["parent_attr_id", "child_attr_id", "edge_weight"]
    ].rename(
        columns={
            "child_attr_id": "sibling_attr_id",
            "edge_weight": "sibling_parent_edge_weight",
        }
    )
    sibling_links = current_edges.merge(sibling_edges, on="parent_attr_id", how="inner")
    sibling_links = sibling_links.loc[
        sibling_links["current_attr_id"] != sibling_links["sibling_attr_id"]
    ].copy()
    if sibling_links.empty:
        return pd.DataFrame(
            columns=["current_attr_id", "sibling_attr_id", "sibling_weight"]
        )

    sibling_links["sibling_weight"] = sibling_links[
        "current_parent_edge_weight"
    ].astype(float) * sibling_links["sibling_parent_edge_weight"].astype(float)
    sibling_links = sibling_links.groupby(
        ["current_attr_id", "sibling_attr_id"],
        as_index=False,
        sort=False,
    )["sibling_weight"].sum()
    return sibling_links


def _join_sibling_context(
    base: pd.DataFrame,
    sibling_links: pd.DataFrame,
) -> pd.DataFrame:
    sibling_values = base.loc[
        :, ["week_id", "attr_id", *_DYNAMIC_CONTEXT_COLUMNS]
    ].rename(columns={"attr_id": "sibling_attr_id"})
    current_values = base.loc[
        :, ["week_id", "attr_id", "share_t", "growth_lag_1"]
    ].rename(
        columns={
            "attr_id": "current_attr_id",
            "share_t": "current_share_t",
            "growth_lag_1": "current_growth_lag_1",
        }
    )
    joined = sibling_links.merge(
        sibling_values,
        on="sibling_attr_id",
        how="inner",
    ).merge(
        current_values,
        on=["week_id", "current_attr_id"],
        how="inner",
    )
    return joined


def _summarize_sibling_group(
    week_id: object,
    current_attr_id: object,
    group: pd.DataFrame,
) -> dict[str, object]:
    weights = group["sibling_weight"].astype(float)
    weight_sum = float(weights.sum())
    sibling_share = _weighted_average(group["share_t"], weights, weight_sum)
    sibling_growth = _weighted_average(group["growth_lag_1"], weights, weight_sum)
    current_share = float(group["current_share_t"].iloc[0])
    current_growth = float(group["current_growth_lag_1"].iloc[0])
    return {
        "week_id": week_id,
        "attr_id": current_attr_id,
        "kg_sibling_count": int(group["sibling_attr_id"].nunique()),
        "kg_sibling_share_t_wavg": sibling_share,
        "kg_sibling_share_t_max": float(group["share_t"].max()),
        "kg_sibling_growth_lag_1_wavg": sibling_growth,
        "kg_sibling_rank_pct_t_wavg": _weighted_average(
            group["rank_pct_t"],
            weights,
            weight_sum,
        ),
        "kg_self_vs_sibling_share_gap_t": current_share - sibling_share,
        "kg_self_vs_sibling_growth_gap_lag_1": current_growth - sibling_growth,
        "kg_has_sibling": 1,
    }


def _build_light_structure_features(
    base: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    parent_weights = (
        edges.groupby("child_attr_id", sort=False)["edge_weight"]
        .sum()
        .rename("kg_parent_edge_weight_sum")
    )
    child_weights = (
        edges.groupby("parent_attr_id", sort=False)["edge_weight"]
        .sum()
        .rename("kg_child_edge_weight_sum")
    )
    structure = base.loc[:, ["week_id", "attr_id"]].copy()
    structure = structure.merge(
        parent_weights,
        left_on="attr_id",
        right_index=True,
        how="left",
    )
    structure = structure.merge(
        child_weights,
        left_on="attr_id",
        right_index=True,
        how="left",
    )
    structure[["kg_parent_edge_weight_sum", "kg_child_edge_weight_sum"]] = structure[
        ["kg_parent_edge_weight_sum", "kg_child_edge_weight_sum"]
    ].fillna(0.0)
    structure["kg_parent_edge_weight_log"] = np.log1p(
        structure["kg_parent_edge_weight_sum"]
    )
    structure["kg_child_edge_weight_log"] = np.log1p(
        structure["kg_child_edge_weight_sum"]
    )
    structure["kg_has_parent"] = (
        structure["kg_parent_edge_weight_sum"].gt(0).astype("int64")
    )
    structure["kg_has_child"] = (
        structure["kg_child_edge_weight_sum"].gt(0).astype("int64")
    )
    structure["kg_is_root_attr"] = 1 - structure["kg_has_parent"]
    structure["kg_is_leaf_attr"] = 1 - structure["kg_has_child"]
    return structure


def _add_hierarchy_gap_features(features: pd.DataFrame) -> None:
    features["kg_self_parent_share_gap_t"] = np.where(
        features["kg_has_parent"].eq(1),
        features["share_t"] - features["kg_parent_share_t_wavg"],
        0.0,
    )
    features["kg_self_parent_growth_gap_lag_1"] = np.where(
        features["kg_has_parent"].eq(1),
        features["growth_lag_1"] - features["kg_parent_growth_lag_1_wavg"],
        0.0,
    )
    features["kg_self_child_share_gap_t"] = np.where(
        features["kg_has_child"].eq(1),
        features["share_t"] - features["kg_child_share_t_wavg"],
        0.0,
    )
    features["kg_self_child_growth_gap_lag_1"] = np.where(
        features["kg_has_child"].eq(1),
        features["growth_lag_1"] - features["kg_child_growth_lag_1_wavg"],
        0.0,
    )


def _weighted_average(
    values: pd.Series,
    weights: pd.Series,
    weight_sum: float,
) -> float:
    return float((values.astype(float) * weights).sum() / weight_sum)


def _zero_sibling_feature_frame(base: pd.DataFrame) -> pd.DataFrame:
    return _zero_feature_frame(base, list(SIBLING_COMPETITION_FEATURES))


def _zero_feature_frame(base: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    frame = base.loc[:, ["week_id", "attr_id"]].copy()
    for column in feature_columns:
        frame[column] = 0.0
    return frame


def _validate_kg_features(features: pd.DataFrame, kg_columns: list[str]) -> None:
    values = features.loc[:, kg_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("graph context kg features contain non-finite values")


def _validate_output_alignment(
    features: pd.DataFrame,
    samples_all: pd.DataFrame,
) -> None:
    expected = samples_all.loc[:, ["week_id", "attr_id"]].reset_index(drop=True)
    actual = features.loc[:, ["week_id", "attr_id"]].reset_index(drop=True)
    if len(actual) != len(expected) or not actual.equals(expected):
        raise ValueError(
            "graph context row alignment mismatch: output rows must match samples_all"
        )
