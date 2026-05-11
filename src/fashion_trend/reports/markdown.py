from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def markdown_table(
    dataframe: pd.DataFrame,
    *,
    columns: Sequence[str],
    float_format: str = "{:.6f}",
) -> str:
    """Render a small DataFrame as a GitHub Flavored Markdown pipe table.

    This deliberately avoids pandas.DataFrame.to_markdown(), which depends on
    the optional tabulate package that this project does not declare.
    """
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Markdown 表格缺少列: {missing_columns}")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(
            _format_markdown_cell(row[column], float_format=float_format)
            for column in columns
        )
        + " |"
        for _, row in dataframe.loc[:, list(columns)].iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def _format_markdown_cell(value: Any, *, float_format: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        text = float_format.format(value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")
