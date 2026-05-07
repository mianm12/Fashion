from __future__ import annotations

from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    read_weekly_transactions,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    read_article_attribute_edges,
    read_attribute_nodes,
    read_attribute_week_heat,
    validate_all_sales_articles_have_attribute_edges,
    validate_article_attribute_edges_for_heat,
    validate_attribute_edge_node_metadata_consistency,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.io import (
    remove_file_if_exists,
    write_json,
    write_trend_csv,
    write_trend_parquet,
)
from fashion_trend.trend.predictions import (
    derive_normalized_pred_share_t1,
    validate_pred_share_t1_distribution,
    validate_trend_model_predictions,
)
from fashion_trend.trend.samples import (
    build_attribute_graph_features_frame,
    build_trend_model_samples_frame,
    read_attribute_hierarchy_edges,
    validate_trend_model_samples,
)
from fashion_trend.trend.schema import (
    ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS,
    ARTICLE_ATTRIBUTE_EDGE_HEAT_DTYPES,
    ARTICLE_WEEK_SALES_COLUMNS,
    ARTICLE_WEEK_SALES_DTYPES,
    ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
    ATTRIBUTE_HIERARCHY_EDGE_DTYPES,
    ATTRIBUTE_NODE_HEAT_COLUMNS,
    ATTRIBUTE_NODE_HEAT_DTYPES,
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_DTYPES,
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_DTYPES,
    TREND_MODEL_PRED_SHARE_GROUP_COLUMNS,
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
    TREND_MODEL_SHARE_TOLERANCE,
    TREND_MODEL_SPLIT_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
    WEEKLY_TRANSACTION_COLUMNS,
)
from fashion_trend.trend.splits import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    read_trend_model_split,
    validate_trend_model_split_frame,
    validate_trend_model_split_frames,
)
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    read_attribute_week_target,
    validate_attribute_week_target,
    validate_attribute_week_target_matches_heat,
)
from fashion_trend.trend.validation import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
