from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from fashion_trend.presentation.contracts import (
    DEFAULT_DEMO_CASE_LIMIT,
    MIN_DEMO_CASE_COUNT,
)
from fashion_trend.recommendation.contracts import (
    RECOMMENDATION_SCORE_COLUMNS,
    RECOMMENDATION_TOP_K,
)

CASE_KEY_COLUMNS = ("customer_id", "split", "cutoff_week", "label_week")
ITEM_REQUIRED_COLUMNS = (
    *CASE_KEY_COLUMNS,
    "article_id",
    "rank",
    "score",
    *RECOMMENDATION_SCORE_COLUMNS,
    "candidate_sources",
)
LABEL_REQUIRED_COLUMNS = (*CASE_KEY_COLUMNS, "article_id")
PROFILE_REQUIRED_COLUMNS = (
    *CASE_KEY_COLUMNS,
    "attr_id",
    "attr_type",
    "attr_value",
    "preference_score",
    "purchase_count",
    "last_purchase_week",
)


def build_demo_case_payloads(
    *,
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    user_profile: pd.DataFrame,
    min_case_count: int = MIN_DEMO_CASE_COUNT,
    max_case_count: int = DEFAULT_DEMO_CASE_LIMIT,
) -> list[dict[str, Any]]:
    """Select high-quality recommendation windows for the defense demo app."""
    if min_case_count <= 0:
        raise ValueError("min_case_count must be positive")
    if max_case_count < min_case_count:
        raise ValueError(
            "max_case_count must be greater than or equal to min_case_count"
        )

    items = _normalize_recommendation_items(recommendation_items)
    labels = _normalize_labels(evaluation_labels)
    profiles = _normalize_user_profile(user_profile)
    candidates = _quality_case_stats(items, labels, profiles)

    if len(candidates) < min_case_count:
        raise ValueError(
            f"不足 {min_case_count} 个高质量演示用户案例: available={len(candidates)}"
        )

    selected = candidates.sort_values(
        [
            "hit_count",
            "profile_count",
            "trend_score_mean",
            "score_mean",
            "customer_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).head(max_case_count)
    label_keys = set(_iter_label_keys(labels))
    return [
        _build_case_payload(
            _case_key_from_row(row),
            items=items,
            labels=label_keys,
            profiles=profiles,
            hit_count=int(row.hit_count),
        )
        for row in selected.itertuples(index=False)
    ]


def _quality_case_stats(
    items: pd.DataFrame,
    labels: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    test_items = items.loc[
        items["split"].eq("test") & items["rank"].le(RECOMMENDATION_TOP_K)
    ].copy()
    if test_items.empty:
        return _empty_case_stats()

    test_items["_complete_item"] = _complete_item_mask(test_items)
    stats = (
        test_items.groupby(list(CASE_KEY_COLUMNS), as_index=False)
        .agg(
            item_count=("article_id", "size"),
            rank_count=("rank", "nunique"),
            article_count=("article_id", "nunique"),
            complete_item_count=("_complete_item", "sum"),
            trend_score_mean=("trend_score", "mean"),
            score_mean=("score", "mean"),
        )
        .merge(_rank_set_stats(test_items), on=list(CASE_KEY_COLUMNS), how="inner")
    )
    hit_counts = _hit_counts(test_items, labels)
    profile_counts = _profile_counts(profiles)
    candidates = stats.merge(hit_counts, on=list(CASE_KEY_COLUMNS), how="left").merge(
        profile_counts,
        on=list(CASE_KEY_COLUMNS),
        how="inner",
    )
    candidates["hit_count"] = candidates["hit_count"].fillna(0).astype(int)
    return candidates.loc[
        candidates["item_count"].eq(RECOMMENDATION_TOP_K)
        & candidates["rank_count"].eq(RECOMMENDATION_TOP_K)
        & candidates["article_count"].eq(RECOMMENDATION_TOP_K)
        & candidates["complete_item_count"].eq(RECOMMENDATION_TOP_K)
        & candidates["has_top12_ranks"]
        & candidates["profile_count"].gt(0)
    ].copy()


def _normalize_recommendation_items(dataframe: pd.DataFrame) -> pd.DataFrame:
    _require_columns(dataframe, "recommendation_items", ITEM_REQUIRED_COLUMNS)
    frame = dataframe.loc[:, list(ITEM_REQUIRED_COLUMNS)].copy()
    for column in ("customer_id", "split", "article_id"):
        frame[column] = frame[column].astype(str)
    frame["candidate_sources"] = (
        frame["candidate_sources"].astype("string").fillna("").astype(str)
    )
    for column in ("cutoff_week", "label_week", "rank"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in ("score", *RECOMMENDATION_SCORE_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame


def _normalize_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    _require_columns(dataframe, "evaluation_labels", LABEL_REQUIRED_COLUMNS)
    frame = dataframe.loc[:, list(LABEL_REQUIRED_COLUMNS)].copy()
    for column in ("customer_id", "split", "article_id"):
        frame[column] = frame[column].astype(str)
    for column in ("cutoff_week", "label_week"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame.drop_duplicates(list(LABEL_REQUIRED_COLUMNS))


def _normalize_user_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    _require_columns(dataframe, "user_profile", PROFILE_REQUIRED_COLUMNS)
    frame = dataframe.loc[:, list(PROFILE_REQUIRED_COLUMNS)].copy()
    for column in ("customer_id", "split", "attr_id", "attr_type", "attr_value"):
        frame[column] = frame[column].astype(str)
    for column in ("cutoff_week", "label_week", "purchase_count", "last_purchase_week"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["preference_score"] = pd.to_numeric(
        frame["preference_score"],
        errors="raise",
    ).astype(float)
    return frame


def _complete_item_mask(items: pd.DataFrame) -> pd.Series:
    score_values = items.loc[:, ("score", *RECOMMENDATION_SCORE_COLUMNS)].to_numpy(
        dtype=float
    )
    return pd.Series(
        np.isfinite(score_values).all(axis=1)
        & items["candidate_sources"].astype(str).str.strip().ne("").to_numpy(),
        index=items.index,
    )


def _rank_set_stats(items: pd.DataFrame) -> pd.DataFrame:
    expected = set(range(1, RECOMMENDATION_TOP_K + 1))
    rows = [
        {
            **dict(zip(CASE_KEY_COLUMNS, case_key, strict=True)),
            "has_top12_ranks": set(ranks.astype(int)) == expected,
        }
        for case_key, ranks in items.groupby(list(CASE_KEY_COLUMNS))["rank"]
    ]
    return pd.DataFrame(rows)


def _hit_counts(items: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(columns=[*CASE_KEY_COLUMNS, "hit_count"])
    hits = items.merge(
        labels,
        on=[*CASE_KEY_COLUMNS, "article_id"],
        how="inner",
    )
    if hits.empty:
        return pd.DataFrame(columns=[*CASE_KEY_COLUMNS, "hit_count"])
    return (
        hits.groupby(list(CASE_KEY_COLUMNS), as_index=False)
        .size()
        .rename(columns={"size": "hit_count"})
    )


def _profile_counts(profiles: pd.DataFrame) -> pd.DataFrame:
    return (
        profiles.loc[profiles["split"].eq("test")]
        .groupby(list(CASE_KEY_COLUMNS), as_index=False)
        .size()
        .rename(columns={"size": "profile_count"})
    )


def _build_case_payload(
    case_key: tuple[str, str, int, int],
    *,
    items: pd.DataFrame,
    labels: set[tuple[str, str, int, int, str]],
    profiles: pd.DataFrame,
    hit_count: int,
) -> dict[str, Any]:
    customer_id, split, cutoff_week, label_week = case_key
    case_items = items.loc[_case_mask(items, case_key)].sort_values(
        "rank", kind="mergesort"
    )
    case_profile = profiles.loc[_case_mask(profiles, case_key)].sort_values(
        ["preference_score", "purchase_count", "attr_type", "attr_value"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return {
        "customer_id": customer_id,
        "split": split,
        "cutoff_week": cutoff_week,
        "label_week": label_week,
        "window_id": f"{split}:{cutoff_week}:{label_week}",
        "hit_count": hit_count,
        "profile": _profile_rows(case_profile),
        "recommendations": _recommendation_rows(case_items, labels=labels),
    }


def _profile_rows(profile: pd.DataFrame, *, top_n: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "attr_id": str(row.attr_id),
            "attr_type": str(row.attr_type),
            "attr_value": str(row.attr_value),
            "preference_score": float(row.preference_score),
            "purchase_count": int(row.purchase_count),
            "last_purchase_week": int(row.last_purchase_week),
        }
        for row in profile.head(top_n).itertuples(index=False)
    ]


def _recommendation_rows(
    items: pd.DataFrame,
    *,
    labels: set[tuple[str, str, int, int, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(row.rank),
            "article_id": str(row.article_id),
            "is_hit": _item_label_key(row) in labels,
            "candidate_sources": str(row.candidate_sources),
            "score_decomposition": {
                "score": float(row.score),
                "pop_score": float(row.pop_score),
                "sim_score": float(row.sim_score),
                "trend_score": float(row.trend_score),
                "recent_score": float(row.recent_score),
            },
        }
        for row in items.itertuples(index=False)
    ]


def _case_mask(
    dataframe: pd.DataFrame,
    case_key: tuple[str, str, int, int],
) -> pd.Series:
    customer_id, split, cutoff_week, label_week = case_key
    return (
        dataframe["customer_id"].eq(customer_id)
        & dataframe["split"].eq(split)
        & dataframe["cutoff_week"].eq(cutoff_week)
        & dataframe["label_week"].eq(label_week)
    )


def _iter_label_keys(
    labels: pd.DataFrame,
) -> list[tuple[str, str, int, int, str]]:
    return [
        (
            str(row.customer_id),
            str(row.split),
            int(row.cutoff_week),
            int(row.label_week),
            str(row.article_id),
        )
        for row in labels.itertuples(index=False)
    ]


def _item_label_key(row: Any) -> tuple[str, str, int, int, str]:
    return (
        str(row.customer_id),
        str(row.split),
        int(row.cutoff_week),
        int(row.label_week),
        str(row.article_id),
    )


def _case_key_from_row(row: Any) -> tuple[str, str, int, int]:
    return (
        str(row.customer_id),
        str(row.split),
        int(row.cutoff_week),
        int(row.label_week),
    )


def _empty_case_stats() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *CASE_KEY_COLUMNS,
            "item_count",
            "rank_count",
            "article_count",
            "complete_item_count",
            "trend_score_mean",
            "score_mean",
            "has_top12_ranks",
            "hit_count",
            "profile_count",
        ]
    )


def _require_columns(
    dataframe: pd.DataFrame,
    artifact_name: str,
    required_columns: Sequence[str],
) -> None:
    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{artifact_name} 缺少必需列: {missing}")
