from __future__ import annotations

# 稳定 `article_week_sales.csv` 的完整输出列顺序。
ARTICLE_WEEK_SALES_COLUMNS: tuple[str, ...] = (
    "week_id",
    "article_id",
    "sales_cnt",
    "sales_user_cnt",
    "sales_amount",
)

# 稳定 `attribute_week_heat.csv` 的完整输出列顺序。
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

# 稳定 `attribute_week_target.csv` 的完整输出列顺序。
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

# 稳定 `trend_model_samples.parquet` 的训练样本列契约。
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

# 稳定时间切分和模型输出中允许出现的 split 名称。
TREND_MODEL_SPLIT_VALUES: tuple[str, ...] = ("train", "valid", "test")

# 稳定 `outputs/models/<model>/predictions.csv` 的完整输出列顺序。
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

# 稳定预测份额归一化使用的分组接口。
TREND_MODEL_PRED_SHARE_GROUP_COLUMNS: tuple[str, ...] = (
    "split",
    "week_id",
    "attr_type",
)
# 稳定预测份额分布校验的浮点容差。
TREND_MODEL_SHARE_TOLERANCE = 1e-9

# 稳定切分后 `trend_model_samples_<split>.parquet` 的完整列契约。
TREND_MODEL_SPLIT_COLUMNS: tuple[str, ...] = (
    "split",
    *TREND_MODEL_SAMPLE_COLUMNS,
)

# 稳定趋势评价 JSON payload 的下游消费必需键。
TREND_METRICS_PAYLOAD_REQUIRED_KEYS: tuple[str, ...] = (
    "model_name",
    "prediction_path",
    "output_path",
    "evaluated_splits",
    "ranking",
    "overall",
    "by_attr_type",
    "groups",
)

# 读取器使用这些类型保留商品标识符和销量数值类型。
ARTICLE_WEEK_SALES_DTYPES: dict[str, str] = {
    "week_id": "int64",
    "article_id": "string",
    "sales_cnt": "int64",
    "sales_user_cnt": "int64",
    "sales_amount": "float64",
}

# 读取器使用这些类型保留属性标识符、热度计数和派生数值类型。
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

# 读取器使用这些类型保留趋势标签中的标识符、位移热度和目标数值类型。
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
