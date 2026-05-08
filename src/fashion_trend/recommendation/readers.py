from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_recommendation_result(result_path: Path) -> pd.DataFrame:
    """读取推荐阶段写出的 Top-N 推荐结果表。"""
    if not result_path.exists():
        raise FileNotFoundError(f"推荐结果文件不存在: {result_path}")
    return pd.read_csv(result_path)
