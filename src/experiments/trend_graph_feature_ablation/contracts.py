from __future__ import annotations

from fashion_trend.trend.schema import (
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
)

SCHEMA_VERSION = "trend_graph_feature_ablation.v1"
ABLATION_EXPERIMENT_ID = "trend_graph_feature_ablation"

ABLATION_VARIANTS: tuple[str, ...] = (
    "no_graph",
    "current_coarse_graph",
    "full_enhanced",
    "wo_hierarchy_context",
    "wo_sibling_competition",
)

FEATURE_GROUP_NAMES: tuple[str, ...] = (
    "base_numeric_non_graph",
    "categorical",
    "coarse_graph",
    "hierarchy_context",
    "sibling_competition",
    "light_structure",
)

ALL_SAMPLE_KEY_COLUMNS: tuple[str, ...] = ("week_id", "attr_id")
SPLIT_SAMPLE_KEY_COLUMNS: tuple[str, ...] = ("split", "week_id", "attr_id")

TARGET_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in TREND_MODEL_SAMPLE_COLUMNS
    if column
    in {
        "target_growth",
        "target_log_heat_t1",
        "target_rank_in_type_t1",
    }
)

PREDICTION_COLUMNS: tuple[str, ...] = TREND_MODEL_PREDICTION_COLUMNS

SUMMARY_COLUMNS: tuple[str, ...] = (
    "variant",
    "feature_count",
    "best_iteration",
    "training_elapsed_seconds",
    "valid_ndcg_at_10",
    "valid_spearman",
    "valid_precision_at_10",
    "valid_recall_at_10",
    "test_ndcg_at_10",
    "test_spearman",
    "test_precision_at_10",
    "test_recall_at_10",
)

RUN_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "predictions.csv",
    "metrics.json",
    "feature_importance.csv",
    "metadata.json",
    "params.json",
    "model.txt",
)
