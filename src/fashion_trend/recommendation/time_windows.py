from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.contracts import (
    TIME_WINDOW_COLUMNS,
    TIME_WINDOW_KEY_COLUMNS,
    VALID_RECOMMENDATION_SPLITS,
)
from fashion_trend.recommendation.readers import reject_duplicate_key, validate_columns


def build_recommendation_windows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build recommendation windows from stable trend predictions."""
    required_columns = {"split", "week_id"}
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"趋势预测缺少列: {missing_columns}")

    windows = (
        predictions.loc[
            predictions["split"].isin(VALID_RECOMMENDATION_SPLITS),
            ["split", "week_id"],
        ]
        .drop_duplicates()
        .rename(columns={"week_id": "cutoff_week"})
    )
    windows["label_week"] = windows["cutoff_week"] + 1
    windows = _sort_windows(windows.loc[:, list(TIME_WINDOW_COLUMNS)])

    validate_recommendation_windows(windows)
    return windows


def validate_recommendation_windows(windows: pd.DataFrame) -> None:
    """Validate recommendation cutoff and label windows."""
    validate_columns(windows, TIME_WINDOW_COLUMNS, "time_windows")
    reject_duplicate_key(windows, TIME_WINDOW_KEY_COLUMNS, "time_windows")

    invalid_splits = sorted(set(windows["split"]) - set(VALID_RECOMMENDATION_SPLITS))
    if invalid_splits:
        raise ValueError(f"推荐窗口包含非法 split: {invalid_splits}")

    missing_splits = sorted(set(VALID_RECOMMENDATION_SPLITS) - set(windows["split"]))
    if missing_splits:
        raise ValueError(f"推荐窗口缺少 split: {missing_splits}")

    invalid_order = windows["cutoff_week"] >= windows["label_week"]
    if invalid_order.any():
        raise ValueError("推荐窗口必须满足 cutoff_week < label_week")

    invalid_label = windows["label_week"] != windows["cutoff_week"] + 1
    if invalid_label.any():
        raise ValueError("推荐窗口必须满足 label_week == cutoff_week + 1")


def _sort_windows(windows: pd.DataFrame) -> pd.DataFrame:
    split_order = {
        split: index for index, split in enumerate(VALID_RECOMMENDATION_SPLITS)
    }
    sorted_windows = windows.assign(_split_order=windows["split"].map(split_order))
    sorted_windows = sorted_windows.sort_values(
        ["_split_order", "cutoff_week"],
        kind="mergesort",
    )
    return sorted_windows.drop(columns="_split_order").reset_index(drop=True)
