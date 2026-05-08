from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fashion_trend.foundation.dataframe import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
from fashion_trend.trend.heat.attribute_heat import validate_attribute_week_heat
from fashion_trend.trend.schema import (
    ATTRIBUTE_WEEK_TARGET_COLUMNS,
    ATTRIBUTE_WEEK_TARGET_DTYPES,
)


def read_attribute_week_target(attribute_week_target_path: Path) -> pd.DataFrame:
    """读取 `attribute_week_target.csv` 属性趋势标签表并保留契约列类型。"""
    if not attribute_week_target_path.exists():
        raise FileNotFoundError(f"属性趋势标签表不存在: {attribute_week_target_path}")

    try:
        header = pd.read_csv(attribute_week_target_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"无法读取属性趋势标签表: {attribute_week_target_path}"
        ) from exc

    missing_columns = sorted(set(ATTRIBUTE_WEEK_TARGET_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "属性趋势标签表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_week_target_path}"
        )

    try:
        return pd.read_csv(
            attribute_week_target_path,
            usecols=list(ATTRIBUTE_WEEK_TARGET_COLUMNS),
            dtype=ATTRIBUTE_WEEK_TARGET_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"无法读取属性趋势标签表: {attribute_week_target_path}"
        ) from exc


def build_attribute_week_target_frame(
    attribute_week_heat: pd.DataFrame,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """由属性周热度表构造 `t -> t+1` 趋势标签。

    Args:
        attribute_week_heat: 完整属性周热度面板。
        epsilon: 用于平滑占比增长率的正数，避免零占比分母。

    Returns:
        属性趋势标签表。每行保留第 `t` 周热度、占比和排名，并连接同一属性
        第 `t+1` 周的热度、占比、`log_heat` 与排名；`target_growth`
        使用 `log((share_t1 + epsilon) / (share_t + epsilon))`。没有
        `week_id + 1` 记录的周会被排除；在当前完整连续周面板下，这等价于
        最后一周被排除。

    Raises:
        ValueError: 当输入热度表不满足契约或 `epsilon` 不是正数时抛出。
    """
    validate_attribute_week_heat(attribute_week_heat)
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数。")

    current = attribute_week_heat.loc[
        :,
        [
            "week_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "heat_cnt",
            "heat_share",
            "rank_in_type",
        ],
    ].rename(
        columns={
            "heat_cnt": "heat_t",
            "heat_share": "share_t",
            "rank_in_type": "rank_in_type_t",
        }
    )
    next_week = attribute_week_heat.loc[
        :, ["week_id", "attr_id", "heat_cnt", "heat_share", "log_heat", "rank_in_type"]
    ].copy()
    # 将 t+1 的记录回写到 t 的 week_id，内连接只保留存在 week_id + 1 的周。
    next_week["week_id"] = next_week["week_id"] - 1
    next_week = next_week.rename(
        columns={
            "heat_cnt": "heat_t1",
            "heat_share": "share_t1",
            "log_heat": "target_log_heat_t1",
            "rank_in_type": "target_rank_in_type_t1",
        }
    )

    target = current.merge(next_week, on=["week_id", "attr_id"], how="inner")
    target["target_growth"] = np.log(
        (target["share_t1"] + epsilon) / (target["share_t"] + epsilon)
    )
    target = target.loc[:, list(ATTRIBUTE_WEEK_TARGET_COLUMNS)].sort_values(
        ["week_id", "attr_type", "attr_id"],
        ignore_index=True,
    )
    return target


def validate_attribute_week_target(
    attribute_week_target: pd.DataFrame,
    expected_week_count: int | None = None,
    expected_attribute_count: int | None = None,
    epsilon: float = 1e-6,
) -> None:
    """校验属性趋势标签表的列契约、键约束和目标公式。

    Args:
        attribute_week_target: 待校验的属性趋势标签表。
        expected_week_count: 可选的源热度表周数，用于按连续周面板校验目标行数。
        expected_attribute_count: 可选的属性数量，用于配合周数校验完整行数。
        epsilon: 与构造阶段一致的增长率平滑参数。

    Raises:
        ValueError: 当行数、数值有限性、占比范围、`target_growth`
            或 `target_log_heat_t1` 不满足契约时抛出。
    """
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数。")

    validate_required_columns(
        attribute_week_target,
        ATTRIBUTE_WEEK_TARGET_COLUMNS,
        source_name="属性趋势标签表",
    )
    validate_no_missing_values(
        attribute_week_target,
        ATTRIBUTE_WEEK_TARGET_COLUMNS,
        source_name="属性趋势标签表",
    )
    validate_unique_key(
        attribute_week_target,
        ["week_id", "attr_id"],
        source_name="属性趋势标签表",
    )
    numeric_columns = [
        column
        for column in ATTRIBUTE_WEEK_TARGET_COLUMNS
        if column not in {"attr_id", "attr_type", "attr_value"}
    ]
    try:
        numeric_values = attribute_week_target.loc[:, numeric_columns].to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("属性趋势标签表存在非有限数值字段。") from exc
    if not np.isfinite(numeric_values).all():
        raise ValueError("属性趋势标签表存在非有限数值字段。")

    validate_non_negative_values(
        attribute_week_target,
        ["heat_t", "heat_t1", "share_t", "share_t1", "target_log_heat_t1"],
        source_name="属性趋势标签表",
    )
    validate_positive_values(
        attribute_week_target,
        ["rank_in_type_t", "target_rank_in_type_t1"],
        source_name="属性趋势标签表",
    )
    if expected_week_count is not None and expected_attribute_count is not None:
        expected_rows = (expected_week_count - 1) * expected_attribute_count
        if len(attribute_week_target) != expected_rows:
            raise ValueError(
                f"属性趋势标签表行数应为 {expected_rows}，实际为 {len(attribute_week_target)}。"
            )
    if (attribute_week_target[["share_t", "share_t1"]] > 1).any().any():
        raise ValueError("属性趋势标签表存在 share 大于 1 的记录。")
    expected_growth = np.log(
        (attribute_week_target["share_t1"] + epsilon)
        / (attribute_week_target["share_t"] + epsilon)
    )
    if not np.allclose(
        attribute_week_target["target_growth"].to_numpy(dtype=float),
        expected_growth.to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("属性趋势标签表存在 target_growth 与公式不一致。")
    expected_log_heat_t1 = np.log1p(attribute_week_target["heat_t1"])
    if not np.allclose(
        attribute_week_target["target_log_heat_t1"].to_numpy(dtype=float),
        expected_log_heat_t1.to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("属性趋势标签表存在 target_log_heat_t1 与公式不一致。")


def validate_attribute_week_target_matches_heat(
    attribute_week_heat: pd.DataFrame,
    attribute_week_target: pd.DataFrame,
    epsilon: float = 1e-6,
) -> None:
    """校验趋势标签表与当前属性周热度表重新派生的结果完全一致。

    Args:
        attribute_week_heat: 作为事实来源的完整属性周热度表。
        attribute_week_target: 待校验的属性趋势标签表。
        epsilon: 与构造阶段一致的增长率平滑参数。

    Raises:
        ValueError: 当目标键、属性元数据或数值字段与热度表派生结果不一致时抛出。
    """
    expected_target = build_attribute_week_target_frame(
        attribute_week_heat,
        epsilon=epsilon,
    )
    validate_attribute_week_target(attribute_week_target, epsilon=epsilon)

    key_columns = ["week_id", "attr_id"]
    actual_keys = attribute_week_target.loc[:, key_columns]
    expected_keys = expected_target.loc[:, key_columns]
    key_diff = expected_keys.merge(
        actual_keys,
        on=key_columns,
        how="outer",
        indicator=True,
    )
    if (key_diff["_merge"] != "both").any():
        missing_count = int((key_diff["_merge"] == "left_only").sum())
        extra_count = int((key_diff["_merge"] == "right_only").sum())
        raise ValueError(
            "属性趋势标签表与当前属性周热度表派生结果不一致：趋势标签表"
            f"缺失 {missing_count} 个目标键，多余 {extra_count} 个目标键。"
        )

    compare_columns = [
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
    ]
    actual = attribute_week_target.loc[:, key_columns + compare_columns].sort_values(
        key_columns,
        ignore_index=True,
    )
    expected = expected_target.loc[:, key_columns + compare_columns].sort_values(
        key_columns,
        ignore_index=True,
    )

    for column in ["attr_type", "attr_value"]:
        if (
            not actual[column]
            .astype("string")
            .equals(expected[column].astype("string"))
        ):
            raise ValueError(
                "属性趋势标签表与当前属性周热度表派生结果不一致："
                f"{column} 字段不一致。"
            )

    numeric_columns = [
        column
        for column in compare_columns
        if column not in {"attr_type", "attr_value"}
    ]
    if not np.allclose(
        actual.loc[:, numeric_columns].to_numpy(dtype=float),
        expected.loc[:, numeric_columns].to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError(
            "属性趋势标签表与当前属性周热度表派生结果不一致：" "数值字段不一致。"
        )
