from __future__ import annotations

import pandas as pd


def filter_seen_items(items: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Remove candidates purchased by the same customer at or before cutoff."""
    original_columns = items.columns
    if items.empty or transactions.empty:
        return items.copy()

    items_with_order = _with_string_ids(items).copy()
    items_with_order["_original_order"] = range(len(items_with_order))
    transactions = _with_string_ids(transactions)

    frames: list[pd.DataFrame] = []
    for window in (
        items_with_order[["split", "cutoff_week", "label_week"]]
        .drop_duplicates()
        .to_dict("records")
    ):
        window_items = _items_for_window(items_with_order, window)
        seen = transactions.loc[
            pd.to_numeric(transactions["week_id"], errors="raise")
            <= int(window["cutoff_week"]),
            ["customer_id", "article_id"],
        ].drop_duplicates()
        filtered = window_items.merge(
            seen.assign(_seen=True),
            on=["customer_id", "article_id"],
            how="left",
        )
        frames.append(filtered.loc[filtered["_seen"].isna(), window_items.columns])

    if not frames:
        return items.loc[[], original_columns].copy()
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("_original_order").reset_index(drop=True)
    return result.loc[:, original_columns]


def filter_seen_items_by_source_policy(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply source-level seen policy after enhanced candidate flags are attached."""
    required_columns = ("is_seen", "allow_seen", "has_reorder_source")
    missing = [
        column for column in required_columns if column not in candidates.columns
    ]
    if missing:
        raise ValueError(f"candidates missing source-level seen columns: {missing}")
    if candidates.empty:
        return candidates.copy()

    is_seen = candidates["is_seen"].fillna(False).astype(bool)
    allow_seen = candidates["allow_seen"].fillna(False).astype(bool)
    keep = (~is_seen) | allow_seen
    return candidates.loc[keep, candidates.columns].reset_index(drop=True)


def _items_for_window(
    items: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    mask = (
        (items["split"] == window["split"])
        & (items["cutoff_week"] == window["cutoff_week"])
        & (items["label_week"] == window["label_week"])
    )
    return items.loc[mask].copy()


def _with_string_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("article_id", "customer_id"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result
