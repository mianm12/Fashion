from __future__ import annotations

WEEKLY_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "customer_id",
    "price",
)

ARTICLE_WEEK_SALES_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "sales_cnt",
    "sales_user_cnt",
    "sales_amount",
)

ARTICLE_ATTRIBUTE_EDGE_HEAT_COLUMNS: tuple[str, ...] = (
    "article_id",
    "attr_id",
    "attr_type",
    "attr_value",
)

ATTRIBUTE_WEEK_HEAT_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "heat_cnt",
    "type_total_heat",
    "heat_share",
    "log_heat",
    "rank_in_type",
)

ATTRIBUTE_WEEK_TARGET_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "heat_t",
    "heat_t1",
    "share_t",
    "share_t1",
    "rank_in_type_t",
    "target_log_heat_t1",
    "target_growth",
    "target_rank_in_type_t1",
)

ATTRIBUTE_HIERARCHY_EDGE_COLUMNS: tuple[str, ...] = (
    "parent_attr_id",
    "child_attr_id",
    "parent_attr_type",
    "child_attr_type",
    "relation_type",
    "edge_weight",
)

TREND_MODEL_SAMPLE_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "heat_t",
    "share_t",
    "log_heat_t",
    "rank_in_type_t",
    "heat_lag_1",
    "heat_lag_2",
    "heat_lag_3",
    "heat_lag_4",
    "share_lag_1",
    "share_lag_2",
    "share_lag_3",
    "share_lag_4",
    "growth_lag_1",
    "growth_lag_2",
    "acc_lag_1",
    "heat_ma_4",
    "share_ma_4",
    "share_std_4",
    "share_max_4",
    "share_min_4",
    "article_count",
    "is_core_attr",
    "parent_count",
    "child_count",
    "degree",
    "history_total_heat_t",
    "history_active_weeks_t",
    "is_trend_eligible_t",
    "week_index",
    "week_mod_52",
    "target_growth",
    "target_log_heat_t1",
    "target_rank_in_type_t1",
)

TREND_MODEL_SPLIT_VALUES: tuple[str, ...] = ("train", "valid", "test")

TREND_MODEL_PREDICTION_COLUMNS: tuple[str, ...] = (
    "week_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "model_name",
    "split",
    "share_t",
    "pred_share_t1",
    "target_growth",
    "pred_target_growth",
    "target_rank_in_type_t1",
)

TREND_MODEL_PRED_SHARE_GROUP_COLUMNS: tuple[str, ...] = (
    "split",
    "week_id",
    "attr_type",
)
TREND_MODEL_SHARE_TOLERANCE = 1e-9

TREND_MODEL_SPLIT_COLUMNS: tuple[str, ...] = (
    "split",
    *TREND_MODEL_SAMPLE_COLUMNS,
)

ARTICLE_WEEK_SALES_DTYPES: dict[str, str] = {
    "week_id": "int64",
    "article_id": "string",
    "sales_cnt": "int64",
    "sales_user_cnt": "int64",
    "sales_amount": "float64",
}

ATTRIBUTE_WEEK_HEAT_DTYPES: dict[str, str] = {
    "week_id": "int64",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "heat_cnt": "int64",
    "type_total_heat": "int64",
    "heat_share": "float64",
    "log_heat": "float64",
    "rank_in_type": "int64",
}

ATTRIBUTE_WEEK_TARGET_DTYPES: dict[str, str] = {
    "week_id": "int64",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "heat_t": "int64",
    "heat_t1": "int64",
    "share_t": "float64",
    "share_t1": "float64",
    "rank_in_type_t": "int64",
    "target_log_heat_t1": "float64",
    "target_growth": "float64",
    "target_rank_in_type_t1": "int64",
}

ATTRIBUTE_HIERARCHY_EDGE_DTYPES: dict[str, str] = {
    "parent_attr_id": "string",
    "child_attr_id": "string",
    "parent_attr_type": "string",
    "child_attr_type": "string",
    "relation_type": "string",
    "edge_weight": "int64",
}

ARTICLE_ATTRIBUTE_EDGE_HEAT_DTYPES: dict[str, str] = {
    "article_id": "string",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
}

ATTRIBUTE_NODE_HEAT_COLUMNS: tuple[str, ...] = (
    "attr_id",
    "attr_type",
    "attr_value",
    "article_count",
    "is_core_attr",
)

ATTRIBUTE_NODE_HEAT_DTYPES: dict[str, str] = {
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "article_count": "int64",
    "is_core_attr": "int64",
}
