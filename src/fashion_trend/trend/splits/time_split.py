from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion_trend.foundation.dataframe import (
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_key,
)
from fashion_trend.trend.features.samples import validate_trend_model_samples
from fashion_trend.trend.schema import (
    TREND_MODEL_SPLIT_COLUMNS,
    TREND_MODEL_SPLIT_VALUES,
)


def build_trend_model_split_frames(
    trend_model_samples: pd.DataFrame,
    valid_weeks: int,
    test_weeks: int,
) -> dict[str, pd.DataFrame]:
    """按时间顺序切分趋势模型样本。

    Args:
        trend_model_samples: 完整趋势训练样本表。
        valid_weeks: 从测试区间之前留出的连续 `week_id` 验证区间宽度。
        test_weeks: 从样本尾部留出的连续 `week_id` 测试区间宽度。

    Returns:
        `train`、`valid`、`test` 三个样本表。训练集覆盖验证窗口之前的周，
        验证集覆盖测试窗口之前的连续 `week_id` 区间，测试集覆盖末尾连续
        `week_id` 区间。输入样本应来自完整连续周面板；如果 `week_id`
        有缺口，实际观测周数可能少于配置值。

    Raises:
        ValueError: 当样本契约、窗口大小或可切分周数不满足要求时抛出。
    """
    validate_trend_model_samples(trend_model_samples)
    if valid_weeks <= 0:
        raise ValueError("valid_weeks 必须为正整数。")
    if test_weeks <= 0:
        raise ValueError("test_weeks 必须为正整数。")

    week_ids = sorted(trend_model_samples["week_id"].unique().tolist())
    required_week_count = valid_weeks + test_weeks + 1
    if len(week_ids) < required_week_count:
        raise ValueError(
            "样本周数不足，无法生成非空 train/valid/test: "
            f"当前 {len(week_ids)} 周，valid_weeks={valid_weeks}, "
            f"test_weeks={test_weeks}。"
        )

    max_sample_week = max(week_ids)
    test_start_week = max_sample_week - test_weeks + 1
    valid_start_week = test_start_week - valid_weeks

    split_masks = {
        "train": trend_model_samples["week_id"] < valid_start_week,
        "valid": (trend_model_samples["week_id"] >= valid_start_week)
        & (trend_model_samples["week_id"] < test_start_week),
        "test": trend_model_samples["week_id"] >= test_start_week,
    }
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = trend_model_samples.loc[split_masks[split_name]].copy()
        split_frame.insert(0, "split", split_name)
        split_frame = split_frame.loc[:, list(TREND_MODEL_SPLIT_COLUMNS)].sort_values(
            ["week_id", "attr_type", "attr_id"],
            ignore_index=True,
        )
        split_frames[split_name] = split_frame

    validate_trend_model_split_frames(split_frames, trend_model_samples)
    return split_frames


def validate_trend_model_split_frames(
    split_frames: dict[str, pd.DataFrame],
    original_samples: pd.DataFrame | None = None,
) -> None:
    """校验训练、验证、测试切分完整且周范围按时间递增。

    Args:
        split_frames: 按 split 名称索引的切分样本表。
        original_samples: 可选的原始样本表，用于校验切分合并后覆盖全集。

    Raises:
        ValueError: 当 split 缺失、单个 split 无效、周范围重叠或覆盖不全时抛出。
    """
    missing_splits = set(TREND_MODEL_SPLIT_VALUES) - set(split_frames)
    if missing_splits:
        raise ValueError(f"趋势样本切分缺少 split: {sorted(missing_splits)}")

    combined_parts: list[pd.DataFrame] = []
    previous_max_week: int | None = None
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = split_frames[split_name]
        validate_trend_model_split_frame(split_frame, expected_split=split_name)
        min_week = int(split_frame["week_id"].min())
        max_week = int(split_frame["week_id"].max())
        if previous_max_week is not None and min_week <= previous_max_week:
            raise ValueError("趋势样本 split 周范围必须按时间递增且互不重叠。")
        previous_max_week = max_week
        combined_parts.append(split_frame.drop(columns=["split"]))

    if original_samples is not None:
        combined = pd.concat(combined_parts, ignore_index=True)
        combined_keys = combined.loc[:, ["week_id", "attr_id"]].sort_values(
            ["week_id", "attr_id"],
            ignore_index=True,
        )
        original_keys = original_samples.loc[:, ["week_id", "attr_id"]].sort_values(
            ["week_id", "attr_id"],
            ignore_index=True,
        )
        if not combined_keys.equals(original_keys):
            raise ValueError("趋势样本 split 合并后无法覆盖原始样本全集。")


def validate_trend_model_split_frame(
    split_frame: pd.DataFrame,
    expected_split: str | None = None,
) -> None:
    """校验单个趋势样本 split 的列契约、非空约束、split 值和唯一键。"""
    validate_required_columns(
        split_frame,
        TREND_MODEL_SPLIT_COLUMNS,
        source_name="趋势样本 split",
    )
    validate_no_missing_values(
        split_frame,
        TREND_MODEL_SPLIT_COLUMNS,
        source_name="趋势样本 split",
    )
    if split_frame.empty:
        raise ValueError("趋势样本 split 为空。")

    split_values = set(split_frame["split"])
    invalid_split_values = sorted(split_values - set(TREND_MODEL_SPLIT_VALUES))
    if invalid_split_values:
        raise ValueError(f"趋势样本 split 存在非法 split: {invalid_split_values}")
    if expected_split is not None and split_values != {expected_split}:
        raise ValueError(f"{expected_split} 趋势样本 split 字段不一致。")
    if expected_split is None and len(split_values) != 1:
        raise ValueError("趋势样本 split 字段必须固定为单一值。")

    validate_unique_key(
        split_frame,
        ["week_id", "attr_id"],
        source_name="趋势样本 split",
    )


def build_trend_model_split_metadata(
    split_frames: dict[str, pd.DataFrame],
    input_path: Path,
    output_paths: dict[str, Path],
    valid_weeks: int,
    test_weeks: int,
) -> dict[str, object]:
    """构造时间切分元数据 JSON payload。

    Args:
        split_frames: 已通过校验的训练、验证、测试切分表。
        input_path: 原始趋势训练样本表路径。
        output_paths: 各 split 对应的写出路径。
        valid_weeks: 本次切分使用的验证 `week_id` 区间宽度。
        test_weeks: 本次切分使用的测试 `week_id` 区间宽度。

    Returns:
        描述时间切分策略、输入路径、区间宽度和各 split 行数及周范围的字典。

    Raises:
        ValueError: 当切分表不满足时间切分契约时抛出。
    """
    validate_trend_model_split_frames(split_frames)
    split_metadata: dict[str, dict[str, object]] = {}
    for split_name in TREND_MODEL_SPLIT_VALUES:
        split_frame = split_frames[split_name]
        split_metadata[split_name] = {
            "path": str(output_paths[split_name]),
            "rows": int(len(split_frame)),
            "weeks": int(split_frame["week_id"].nunique()),
            "attributes": int(split_frame["attr_id"].nunique()),
            "week_min": int(split_frame["week_id"].min()),
            "week_max": int(split_frame["week_id"].max()),
        }
    return {
        "split_strategy": "time",
        "valid_weeks": int(valid_weeks),
        "test_weeks": int(test_weeks),
        "input_path": str(input_path),
        "splits": split_metadata,
    }


def read_trend_model_split(input_path: Path) -> pd.DataFrame:
    """读取单个 `trend_model_samples_<split>.parquet` 时间切分样本表。"""
    if not input_path.exists():
        raise FileNotFoundError(f"趋势样本 split 不存在: {input_path}")
    dataframe = pd.read_parquet(input_path)
    validate_required_columns(
        dataframe,
        TREND_MODEL_SPLIT_COLUMNS,
        source_name=f"趋势样本 split: {input_path}",
    )
    split_frame = dataframe.loc[:, list(TREND_MODEL_SPLIT_COLUMNS)].copy()
    validate_trend_model_split_frame(split_frame)
    return split_frame
