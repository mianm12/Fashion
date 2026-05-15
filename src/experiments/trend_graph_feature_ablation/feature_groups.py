from __future__ import annotations

from copy import deepcopy
from typing import Any

from experiments.trend_graph_feature_ablation.artifact_io import digest_json_payload
from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_VARIANTS,
    ALL_SAMPLE_KEY_COLUMNS,
    FEATURE_GROUP_NAMES,
    SCHEMA_VERSION,
    SPLIT_SAMPLE_KEY_COLUMNS,
    TARGET_COLUMNS,
)
from fashion_trend.trend.models.supervised.lightgbm import (
    LIGHTGBM_CATEGORICAL_FEATURES,
    LIGHTGBM_NUMERIC_FEATURES,
)

COARSE_GRAPH_FEATURES: tuple[str, ...] = (
    "article_count",
    "is_core_attr",
    "parent_count",
    "child_count",
    "degree",
)
BASE_NUMERIC_NON_GRAPH_FEATURES: tuple[str, ...] = tuple(
    feature
    for feature in LIGHTGBM_NUMERIC_FEATURES
    if feature not in COARSE_GRAPH_FEATURES
)
HIERARCHY_CONTEXT_FEATURES: tuple[str, ...] = (
    "kg_parent_heat_t_wavg",
    "kg_parent_share_t_wavg",
    "kg_parent_growth_lag_1_wavg",
    "kg_parent_rank_pct_t_wavg",
    "kg_child_heat_t_wavg",
    "kg_child_share_t_wavg",
    "kg_child_growth_lag_1_wavg",
    "kg_child_rank_pct_t_wavg",
    "kg_self_parent_share_gap_t",
    "kg_self_parent_growth_gap_lag_1",
    "kg_self_child_share_gap_t",
    "kg_self_child_growth_gap_lag_1",
)
SIBLING_COMPETITION_FEATURES: tuple[str, ...] = (
    "kg_sibling_count",
    "kg_sibling_share_t_wavg",
    "kg_sibling_share_t_max",
    "kg_sibling_growth_lag_1_wavg",
    "kg_sibling_rank_pct_t_wavg",
    "kg_self_vs_sibling_share_gap_t",
    "kg_self_vs_sibling_growth_gap_lag_1",
    "kg_has_sibling",
)
LIGHT_STRUCTURE_FEATURES: tuple[str, ...] = (
    "kg_parent_edge_weight_sum",
    "kg_child_edge_weight_sum",
    "kg_parent_edge_weight_log",
    "kg_child_edge_weight_log",
    "kg_has_parent",
    "kg_has_child",
    "kg_is_root_attr",
    "kg_is_leaf_attr",
)

IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    (*ALL_SAMPLE_KEY_COLUMNS, *SPLIT_SAMPLE_KEY_COLUMNS, "attr_value")
)
FORBIDDEN_FEATURES: frozenset[str] = frozenset((*TARGET_COLUMNS, *IDENTIFIER_COLUMNS))
_DYNAMIC_GROUPS = frozenset({"hierarchy_context", "sibling_competition"})
_INTEGER_FEATURES = frozenset(
    {
        "kg_sibling_count",
        "kg_has_sibling",
        "kg_has_parent",
        "kg_has_child",
        "kg_is_root_attr",
        "kg_is_leaf_attr",
        "is_core_attr",
        "parent_count",
        "child_count",
        "degree",
        "article_count",
        "history_active_weeks_t",
        "is_trend_eligible_t",
        "week_index",
        "week_mod_52",
        "rank_in_type_t",
    }
)


def build_feature_groups() -> dict[str, list[str]]:
    """返回图特征消融使用的稳定特征组定义。"""

    groups = {
        "base_numeric_non_graph": list(BASE_NUMERIC_NON_GRAPH_FEATURES),
        "categorical": list(LIGHTGBM_CATEGORICAL_FEATURES),
        "coarse_graph": list(COARSE_GRAPH_FEATURES),
        "hierarchy_context": list(HIERARCHY_CONTEXT_FEATURES),
        "sibling_competition": list(SIBLING_COMPETITION_FEATURES),
        "light_structure": list(LIGHT_STRUCTURE_FEATURES),
    }
    if tuple(groups) != FEATURE_GROUP_NAMES:
        raise ValueError("feature group order does not match contract")
    return groups


def build_variant_feature_masks() -> dict[str, dict[str, list[str]]]:
    """按 `ABLATION_VARIANTS` 顺序返回每个 variant 的 LightGBM 特征 mask。"""

    groups = build_feature_groups()
    full_numeric = [
        *LIGHTGBM_NUMERIC_FEATURES,
        *groups["hierarchy_context"],
        *groups["sibling_competition"],
        *groups["light_structure"],
    ]
    masks = {
        "no_graph": _build_mask(
            groups["base_numeric_non_graph"], groups["categorical"]
        ),
        "current_coarse_graph": _build_mask(
            LIGHTGBM_NUMERIC_FEATURES,
            LIGHTGBM_CATEGORICAL_FEATURES,
        ),
        "full_enhanced": _build_mask(full_numeric, groups["categorical"]),
        "wo_hierarchy_context": _build_mask(
            _without(full_numeric, groups["hierarchy_context"]),
            groups["categorical"],
        ),
        "wo_sibling_competition": _build_mask(
            _without(full_numeric, groups["sibling_competition"]),
            groups["categorical"],
        ),
    }
    validate_variant_masks(masks)
    return deepcopy(masks)


def validate_variant_masks(masks: dict[str, dict[str, list[str]]]) -> None:
    """校验 variant mask 不包含泄漏列、标识列或未知特征。"""

    if tuple(masks) != ABLATION_VARIANTS:
        raise ValueError("feature masks variant order does not match contract")

    known_features = _known_features()
    for variant, mask in masks.items():
        for feature in _mask_features(mask, variant):
            if feature in FORBIDDEN_FEATURES:
                raise ValueError(f"forbidden feature in {variant}: {feature}")
            if feature not in known_features:
                raise ValueError(f"unknown feature in {variant}: {feature}")

    current = masks["current_coarse_graph"]
    if tuple(current["numeric_features"]) != LIGHTGBM_NUMERIC_FEATURES:
        raise ValueError(
            "current_coarse_graph numeric features drifted from stable LightGBM"
        )
    if tuple(current["categorical_features"]) != LIGHTGBM_CATEGORICAL_FEATURES:
        raise ValueError(
            "current_coarse_graph categorical features drifted from stable LightGBM"
        )


def build_feature_mask_digest(
    variant: str,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> str:
    """返回对特征顺序敏感的 feature mask 摘要。"""

    return digest_json_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "variant": variant,
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
        }
    )


def build_feature_schema() -> list[dict[str, Any]]:
    """返回 feature_groups payload 中的特征 schema 行。"""

    rows: list[dict[str, Any]] = []
    for group, features in build_feature_groups().items():
        for feature in features:
            rows.append(_schema_row(feature, group))
    return rows


def build_feature_groups_payload() -> dict[str, object]:
    """构建可写入 feature_groups.json 的完整契约 payload。"""

    masks = build_variant_feature_masks()
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_groups": build_feature_groups(),
        "feature_schema": build_feature_schema(),
        "variants": masks,
        "feature_mask_digest": {
            variant: build_feature_mask_digest(
                variant,
                numeric_features=mask["numeric_features"],
                categorical_features=mask["categorical_features"],
            )
            for variant, mask in masks.items()
        },
    }


def _build_mask(
    numeric_features: tuple[str, ...] | list[str],
    categorical_features: tuple[str, ...] | list[str],
) -> dict[str, list[str]]:
    return {
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
    }


def _without(features: list[str], removed: list[str]) -> list[str]:
    removed_set = set(removed)
    return [feature for feature in features if feature not in removed_set]


def _known_features() -> set[str]:
    return {
        feature
        for group_features in build_feature_groups().values()
        for feature in group_features
    }


def _mask_features(mask: dict[str, list[str]], variant: str) -> list[str]:
    try:
        return [*mask["numeric_features"], *mask["categorical_features"]]
    except KeyError as exc:
        raise ValueError(
            f"feature mask missing key in {variant}: {exc.args[0]}"
        ) from exc


def _schema_row(feature: str, group: str) -> dict[str, Any]:
    return {
        "feature": feature,
        "group": group,
        "dtype": _feature_dtype(feature, group),
        "dynamic": group in _DYNAMIC_GROUPS,
        "uses_target_information": False,
        "debug_only": False,
    }


def _feature_dtype(feature: str, group: str) -> str:
    if group == "categorical":
        return "category"
    if feature in _INTEGER_FEATURES:
        return "int64"
    return "float64"
