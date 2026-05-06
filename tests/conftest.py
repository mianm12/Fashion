from __future__ import annotations

import pandas as pd

import math

from fashion_trend.trend import (
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    TREND_MODEL_PREDICTION_COLUMNS,
    TREND_MODEL_SAMPLE_COLUMNS,
    build_attribute_week_heat_frame,
)


def sample_attribute_article_week_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_id": [0, 0, 1],
            "article_id": ["0108775015", "0110065001", "0108775015"],
            "sales_cnt": [2, 1, 1],
            "sales_user_cnt": [2, 1, 1],
            "sales_amount": [0.30, 0.30, 0.40],
        }
    )


def sample_article_attribute_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [
                "0108775015",
                "0108775015",
                "0110065001",
                "0110065001",
            ],
            "article_node_id": [
                "article_0108775015",
                "article_0108775015",
                "article_0110065001",
                "article_0110065001",
            ],
            "attr_id": [
                "colour_group_name::Black",
                "product_type_name::Vest top",
                "colour_group_name::White",
                "product_type_name::Bra",
            ],
            "attr_type": [
                "colour_group_name",
                "product_type_name",
                "colour_group_name",
                "product_type_name",
            ],
            "attr_value": ["Black", "Vest top", "White", "Bra"],
            "edge_type": [
                "has_colour_group",
                "has_product_type",
                "has_colour_group",
                "has_product_type",
            ],
            "edge_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )


def sample_attribute_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "attr_id": [
                "colour_group_name::Black",
                "colour_group_name::White",
                "colour_group_name::Blue",
                "product_type_name::Vest top",
                "product_type_name::Bra",
                "product_type_name::Dress",
            ],
            "attr_type": [
                "colour_group_name",
                "colour_group_name",
                "colour_group_name",
                "product_type_name",
                "product_type_name",
                "product_type_name",
            ],
            "attr_value": ["Black", "White", "Blue", "Vest top", "Bra", "Dress"],
            "attr_node_id": [
                "colour_group_name::Black",
                "colour_group_name::White",
                "colour_group_name::Blue",
                "product_type_name::Vest top",
                "product_type_name::Bra",
                "product_type_name::Dress",
            ],
            "article_count": [2, 1, 0, 2, 1, 0],
            "is_core_attr": [1, 1, 1, 1, 1, 1],
            "level": ["child", "child", "child", "child", "child", "child"],
        }
    )


def sample_attribute_week_heat() -> pd.DataFrame:
    return build_attribute_week_heat_frame(
        sample_attribute_article_week_sales(),
        sample_article_attribute_edges(),
        sample_attribute_nodes(),
    )


def sample_long_attribute_week_heat() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for week_id, black_heat, white_heat, blue_heat in [
        (0, 2, 1, 0),
        (1, 1, 0, 0),
        (2, 3, 1, 0),
        (3, 4, 1, 0),
        (4, 8, 2, 0),
        (5, 4, 4, 2),
    ]:
        colour_total = black_heat + white_heat + blue_heat
        for attr_id, attr_value, heat_cnt in [
            ("colour_group_name::Black", "Black", black_heat),
            ("colour_group_name::White", "White", white_heat),
            ("colour_group_name::Blue", "Blue", blue_heat),
        ]:
            records.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": "colour_group_name",
                    "attr_value": attr_value,
                    "heat_cnt": heat_cnt,
                    "type_total_heat": colour_total,
                    "heat_share": heat_cnt / colour_total if colour_total else 0.0,
                    "log_heat": math.log1p(heat_cnt),
                    "rank_in_type": 1,
                }
            )
        product_total = 1
        for attr_id, attr_value, heat_cnt in [
            ("product_type_name::Vest top", "Vest top", 1),
            ("product_type_name::Bra", "Bra", 0),
            ("product_type_name::Dress", "Dress", 0),
        ]:
            records.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": "product_type_name",
                    "attr_value": attr_value,
                    "heat_cnt": heat_cnt,
                    "type_total_heat": product_total,
                    "heat_share": heat_cnt / product_total,
                    "log_heat": math.log1p(heat_cnt),
                    "rank_in_type": 1,
                }
            )
    heat = pd.DataFrame(records).sort_values(
        ["week_id", "attr_type", "heat_cnt", "attr_id"],
        ascending=[True, True, False, True],
        ignore_index=True,
    )
    heat["rank_in_type"] = (
        heat.groupby(["week_id", "attr_type"]).cumcount().add(1).astype("int64")
    )
    return heat.loc[:, list(ATTRIBUTE_WEEK_HEAT_COLUMNS)]


def sample_attribute_hierarchy_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_attr_id": [
                "colour_group_name::Black",
                "product_type_name::Vest top",
            ],
            "child_attr_id": [
                "colour_group_name::White",
                "product_type_name::Bra",
            ],
            "parent_attr_type": ["colour_group_name", "product_type_name"],
            "child_attr_type": ["colour_group_name", "product_type_name"],
            "relation_type": ["test_contains_colour", "test_contains_type"],
            "edge_weight": [2, 1],
        }
    )


def sample_trend_model_samples_for_split() -> pd.DataFrame:
    rows = []
    for week_id in range(4, 24):
        for attr_id, attr_type, attr_value in [
            ("colour_group_name::Black", "colour_group_name", "Black"),
            ("colour_group_name::White", "colour_group_name", "White"),
        ]:
            share_t = 0.60 if attr_value == "Black" else 0.40
            rows.append(
                {
                    "week_id": week_id,
                    "attr_id": attr_id,
                    "attr_type": attr_type,
                    "attr_value": attr_value,
                    "heat_t": 10 + week_id,
                    "share_t": share_t,
                    "log_heat_t": math.log1p(10 + week_id),
                    "rank_in_type_t": 1 if attr_value == "Black" else 2,
                    "heat_lag_1": 9 + week_id,
                    "heat_lag_2": 8 + week_id,
                    "heat_lag_3": 7 + week_id,
                    "heat_lag_4": 6 + week_id,
                    "share_lag_1": share_t - 0.01,
                    "share_lag_2": share_t - 0.02,
                    "share_lag_3": share_t - 0.03,
                    "share_lag_4": share_t - 0.04,
                    "growth_lag_1": 0.10 if attr_value == "Black" else -0.05,
                    "growth_lag_2": 0.05 if attr_value == "Black" else -0.02,
                    "acc_lag_1": 0.05 if attr_value == "Black" else -0.03,
                    "heat_ma_4": 8.5 + week_id,
                    "share_ma_4": share_t - 0.015,
                    "share_std_4": 0.01,
                    "share_max_4": share_t,
                    "share_min_4": share_t - 0.04,
                    "article_count": 10,
                    "is_core_attr": 1,
                    "parent_count": 1,
                    "child_count": 0,
                    "degree": 1,
                    "history_total_heat_t": 100 + week_id,
                    "history_active_weeks_t": week_id,
                    "is_trend_eligible_t": True,
                    "week_index": week_id,
                    "week_mod_52": week_id % 52,
                    "target_growth": 0.12 if attr_value == "Black" else -0.03,
                    "target_log_heat_t1": math.log1p(11 + week_id),
                    "target_rank_in_type_t1": 1 if attr_value == "Black" else 2,
                }
            )
    return pd.DataFrame(rows).loc[:, list(TREND_MODEL_SAMPLE_COLUMNS)]


def sample_trend_predictions_for_evaluation() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("train", 8),
        ("valid", 10),
        ("valid", 11),
        ("test", 12),
        ("test", 13),
    ]
    for split_name, week_id in specs:
        rows.extend(
            [
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.40,
                    "pred_share_t1": 0.52,
                    "target_growth": 3.0,
                    "pred_target_growth": 2.8,
                    "target_rank_in_type_t1": 1,
                },
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::White",
                    "attr_type": "colour_group_name",
                    "attr_value": "White",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.30,
                    "pred_share_t1": 0.20,
                    "target_growth": 2.0,
                    "pred_target_growth": 1.0,
                    "target_rank_in_type_t1": 2,
                },
                {
                    "week_id": week_id,
                    "attr_id": "colour_group_name::Blue",
                    "attr_type": "colour_group_name",
                    "attr_value": "Blue",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.20,
                    "pred_share_t1": 0.25,
                    "target_growth": 1.0,
                    "pred_target_growth": 1.5,
                    "target_rank_in_type_t1": 3,
                },
                {
                    "week_id": week_id,
                    "attr_id": "product_type_name::Dress",
                    "attr_type": "product_type_name",
                    "attr_value": "Dress",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.55,
                    "pred_share_t1": 0.62,
                    "target_growth": 1.5,
                    "pred_target_growth": 1.0,
                    "target_rank_in_type_t1": 1,
                },
                {
                    "week_id": week_id,
                    "attr_id": "product_type_name::Vest top",
                    "attr_type": "product_type_name",
                    "attr_value": "Vest top",
                    "model_name": "last_week",
                    "split": split_name,
                    "share_t": 0.45,
                    "pred_share_t1": 0.40,
                    "target_growth": 0.5,
                    "pred_target_growth": 0.7,
                    "target_rank_in_type_t1": 2,
                },
            ]
        )
    return pd.DataFrame(rows).loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]
