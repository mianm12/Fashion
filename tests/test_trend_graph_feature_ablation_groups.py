from __future__ import annotations

import pytest

from experiments.trend_graph_feature_ablation import contracts
from experiments.trend_graph_feature_ablation.feature_groups import (
    build_feature_groups_payload,
    build_feature_mask_digest,
    build_feature_schema,
    build_variant_feature_masks,
    validate_variant_masks,
)
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_CATEGORICAL_FEATURES,
    LIGHTGBM_NUMERIC_FEATURES,
)


def _all_mask_features() -> set[str]:
    masks = build_variant_feature_masks()
    features: set[str] = set()
    for mask in masks.values():
        features.update(mask["numeric_features"])
        features.update(mask["categorical_features"])
    return features


def test_current_coarse_graph_matches_stable_lightgbm_features() -> None:
    masks = build_variant_feature_masks()
    current = masks["current_coarse_graph"]

    assert tuple(current["numeric_features"]) == LIGHTGBM_NUMERIC_FEATURES
    assert tuple(current["categorical_features"]) == LIGHTGBM_CATEGORICAL_FEATURES


def test_no_graph_excludes_coarse_graph_and_kg_features() -> None:
    numeric = build_variant_feature_masks()["no_graph"]["numeric_features"]

    assert "article_count" not in numeric
    assert "degree" not in numeric
    assert not any(feature.startswith("kg_") for feature in numeric)


def test_wo_hierarchy_context_removes_dynamic_context_but_keeps_light_structure() -> (
    None
):
    numeric = build_variant_feature_masks()["wo_hierarchy_context"]["numeric_features"]

    assert "kg_parent_share_t_wavg" not in numeric
    assert "kg_parent_growth_lag_1_wavg" not in numeric
    assert "kg_parent_edge_weight_sum" in numeric
    assert "kg_has_parent" in numeric


def test_rank_pct_helper_is_not_in_any_mask_or_schema() -> None:
    schema_features = {row["feature"] for row in build_feature_schema()}

    assert "rank_pct_t" not in _all_mask_features()
    assert "rank_pct_t" not in schema_features


def test_feature_schema_covers_kg_features_with_required_metadata() -> None:
    schema = {
        row["feature"]: row
        for row in build_feature_schema()
        if str(row["feature"]).startswith("kg_")
    }

    assert "kg_parent_share_t_wavg" in schema
    assert "kg_has_parent" in schema
    assert set(schema["kg_parent_share_t_wavg"]) == {
        "feature",
        "group",
        "dtype",
        "dynamic",
        "uses_target_information",
        "debug_only",
    }
    assert schema["kg_parent_share_t_wavg"]["group"] == "hierarchy_context"
    assert schema["kg_parent_share_t_wavg"]["dynamic"] is True
    assert schema["kg_parent_share_t_wavg"]["uses_target_information"] is False
    assert schema["kg_parent_share_t_wavg"]["debug_only"] is False
    assert schema["kg_has_parent"]["group"] == "light_structure"
    assert schema["kg_has_parent"]["dynamic"] is False


def test_validate_variant_masks_rejects_target_columns() -> None:
    masks = build_variant_feature_masks()
    masks["no_graph"]["numeric_features"].append("target_growth")

    with pytest.raises(ValueError, match="forbidden feature"):
        validate_variant_masks(masks)


def test_validate_variant_masks_rejects_identifiers_and_unknown_features() -> None:
    masks = build_variant_feature_masks()
    masks["no_graph"]["categorical_features"].append("attr_id")

    with pytest.raises(ValueError, match="forbidden feature"):
        validate_variant_masks(masks)

    masks = build_variant_feature_masks()
    masks["no_graph"]["numeric_features"].append("does_not_exist")

    with pytest.raises(ValueError, match="unknown feature"):
        validate_variant_masks(masks)


def test_feature_mask_digest_is_sensitive_to_feature_order() -> None:
    left = build_feature_mask_digest(
        "example",
        numeric_features=["a", "b"],
        categorical_features=["attr_type"],
    )
    right = build_feature_mask_digest(
        "example",
        numeric_features=["b", "a"],
        categorical_features=["attr_type"],
    )

    assert left != right


def test_variant_order_matches_ablation_variants() -> None:
    assert tuple(build_variant_feature_masks()) == contracts.ABLATION_VARIANTS


def test_full_enhanced_includes_hierarchy_gap_features_and_sibling_marker() -> None:
    numeric = set(build_variant_feature_masks()["full_enhanced"]["numeric_features"])

    assert {
        "kg_self_parent_share_gap_t",
        "kg_self_parent_growth_gap_lag_1",
        "kg_self_child_share_gap_t",
        "kg_self_child_growth_gap_lag_1",
        "kg_has_sibling",
    }.issubset(numeric)


def test_feature_groups_payload_contains_schema_variants_and_digests() -> None:
    payload = build_feature_groups_payload()

    assert payload["schema_version"] == contracts.SCHEMA_VERSION
    assert tuple(payload["feature_groups"]) == contracts.FEATURE_GROUP_NAMES
    assert tuple(payload["variants"]) == contracts.ABLATION_VARIANTS
    assert set(payload["feature_mask_digest"]) == set(contracts.ABLATION_VARIANTS)
    assert payload["feature_schema"]
