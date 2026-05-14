from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from fashion_trend.recommendation.contracts import SOURCE_ORDER
from fashion_trend.recommendation.features.cache import build_candidate_seen_flags
from fashion_trend.recommendation.ranking.filters import (
    filter_seen_items_by_source_policy,
)

WINDOW_COLUMNS = ["split", "cutoff_week", "label_week"]
USER_KEY_COLUMNS = [*WINDOW_COLUMNS, "customer_id"]
ITEM_KEY_COLUMNS = [*USER_KEY_COLUMNS, "article_id"]


def compute_candidate_recall(
    candidates: pd.DataFrame,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    split: str,
) -> dict[str, float]:
    """Measure candidate recall against all target-user label items for a split."""
    split_targets = _split_frame(target_users, split)
    split_labels = _label_items_for_targets(split_targets, labels, split)
    denominator = float(len(split_labels))
    if denominator == 0.0:
        return {
            "target_user_count": float(
                len(split_targets.drop_duplicates(USER_KEY_COLUMNS))
            ),
            "label_item_count": 0.0,
            "hit_label_item_count": 0.0,
            "candidate_recall": 0.0,
        }

    split_candidates = _candidate_items(candidates, split)
    hits = split_labels.merge(
        split_candidates.assign(_candidate_hit=True),
        on=ITEM_KEY_COLUMNS,
        how="left",
    )
    hit_count = float(hits["_candidate_hit"].fillna(False).sum())
    return {
        "target_user_count": float(
            len(split_targets.drop_duplicates(USER_KEY_COLUMNS))
        ),
        "label_item_count": denominator,
        "hit_label_item_count": hit_count,
        "candidate_recall": hit_count / denominator,
    }


def compute_source_hit_contribution(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    split: str,
) -> dict[str, object]:
    """Credit every source on a hit equally; primary source is diagnostic only."""
    split_labels = _split_frame(labels, split)
    if split_labels.empty or candidates.empty or "candidate_sources" not in candidates:
        return {
            "hit_label_item_count": 0,
            "source_credit": {},
            "primary_source_hit_count": {},
        }

    label_items = split_labels.loc[:, ITEM_KEY_COLUMNS].drop_duplicates()
    split_candidates = _split_frame(candidates, split).drop_duplicates(ITEM_KEY_COLUMNS)
    hits = label_items.merge(split_candidates, on=ITEM_KEY_COLUMNS, how="inner")
    source_credit: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()

    for row in hits.itertuples(index=False):
        sources = _source_tuple(getattr(row, "candidate_sources"))
        if not sources:
            continue
        credit = 1.0 / len(sources)
        for source in sources:
            source_credit[source] += credit
        primary_source = getattr(row, "primary_source", None)
        if primary_source is not None and not pd.isna(primary_source):
            primary_counts[str(primary_source)] += 1

    return {
        "hit_label_item_count": int(len(hits)),
        "source_credit": dict(sorted(source_credit.items(), key=lambda item: item[0])),
        "primary_source_hit_count": dict(
            sorted(primary_counts.items(), key=lambda item: item[0])
        ),
    }


def filter_candidate_sources_for_ablation(
    candidates: pd.DataFrame,
    *,
    dropped_sources: set[str],
    strategy: str,
    allow_all_seen: bool = False,
) -> pd.DataFrame:
    """Return candidates after dropping sources and recomputing source fields."""
    _validate_sources(dropped_sources)
    if candidates.empty:
        return _with_recomputed_source_scores(candidates.copy())
    if "candidate_sources" not in candidates.columns:
        raise ValueError("candidates missing candidate_sources")

    result = candidates.copy()
    source_lookup = _source_tuple_lookup(
        result["candidate_sources"],
        dropped_sources=dropped_sources,
    )
    source_sets = result["candidate_sources"].map(source_lookup)
    keep_mask = source_sets.map(bool)
    result = result.loc[keep_mask].copy().reset_index(drop=True)
    source_sets = source_sets.loc[keep_mask].tolist()
    if result.empty:
        return _empty_like_with_source_columns(candidates)

    result["strategy"] = strategy
    result["candidate_sources"] = ["|".join(sources) for sources in source_sets]
    result["primary_source"] = [sources[0] for sources in source_sets]
    result["best_source_rank"] = _recomputed_best_source_rank(result)
    has_reorder = ["reorder" in sources for sources in source_sets]
    result["has_reorder_source"] = pd.Series(has_reorder, dtype=object)
    result["allow_seen"] = pd.Series(
        [False if allow_all_seen else value for value in has_reorder],
        dtype=object,
    )
    return _with_recomputed_source_scores(result)


def filter_seen_candidates_for_diagnostics(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the enhanced source-level seen policy in memory."""
    required = {"allow_seen", "has_reorder_source", *ITEM_KEY_COLUMNS, "strategy"}
    if (
        candidates.empty
        or transactions.empty
        or not required.issubset(candidates.columns)
    ):
        return candidates.copy()

    seen_flags = build_candidate_seen_flags(candidates, transactions)
    if seen_flags.empty:
        result = candidates.copy()
        result["is_seen"] = False
        return result.loc[:, candidates.columns].reset_index(drop=True)

    join_columns = [*ITEM_KEY_COLUMNS]
    join_columns.insert(3, "strategy")
    marker = seen_flags.loc[:, [*join_columns, "is_seen"]].drop_duplicates()
    merged = candidates.merge(marker, on=join_columns, how="left")
    merged["is_seen"] = merged["is_seen"].fillna(False).astype(bool)
    return (
        filter_seen_items_by_source_policy(merged)
        .loc[:, candidates.columns]
        .reset_index(drop=True)
    )


def build_candidate_diagnostics_payload(
    *,
    candidates: pd.DataFrame,
    post_seen_candidates: pd.DataFrame,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, object]:
    recall_pre: dict[str, dict[str, float]] = {}
    recall_post: dict[str, dict[str, float]] = {}
    contribution_pre: dict[str, dict[str, object]] = {}
    contribution_post: dict[str, dict[str, object]] = {}
    avg_candidates: dict[str, dict[str, float]] = {"pre_seen": {}, "post_seen": {}}
    coverage: dict[str, dict[str, object]] = {"pre_seen": {}, "post_seen": {}}

    for split in _diagnostic_splits(target_users, labels, candidates):
        recall_pre[split] = compute_candidate_recall(
            candidates,
            target_users,
            labels,
            split=split,
        )
        recall_post[split] = compute_candidate_recall(
            post_seen_candidates,
            target_users,
            labels,
            split=split,
        )
        contribution_pre[split] = compute_source_hit_contribution(
            candidates,
            labels,
            split=split,
        )
        contribution_post[split] = compute_source_hit_contribution(
            post_seen_candidates,
            labels,
            split=split,
        )
        avg_candidates["pre_seen"][split] = _avg_candidates_per_user(
            candidates,
            target_users,
            split,
        )
        avg_candidates["post_seen"][split] = _avg_candidates_per_user(
            post_seen_candidates,
            target_users,
            split,
        )
        coverage["pre_seen"][split] = _source_coverage(candidates, split)
        coverage["post_seen"][split] = _source_coverage(post_seen_candidates, split)

    return {
        "candidate_recall_pre_seen": recall_pre,
        "candidate_recall_post_seen": recall_post,
        "source_hit_contribution_pre_seen": contribution_pre,
        "source_hit_contribution_post_seen": contribution_post,
        "avg_candidates_per_user": avg_candidates,
        "source_coverage": coverage,
    }


def _label_items_for_targets(
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    if target_users.empty or labels.empty:
        return pd.DataFrame(columns=ITEM_KEY_COLUMNS)
    split_labels = _split_frame(labels, split)
    if split_labels.empty:
        return pd.DataFrame(columns=ITEM_KEY_COLUMNS)
    target_keys = target_users.loc[:, USER_KEY_COLUMNS].drop_duplicates()
    return (
        target_keys.merge(split_labels.loc[:, ITEM_KEY_COLUMNS], on=USER_KEY_COLUMNS)
        .drop_duplicates(ITEM_KEY_COLUMNS)
        .reset_index(drop=True)
    )


def _candidate_items(candidates: pd.DataFrame, split: str) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=ITEM_KEY_COLUMNS)
    return (
        _split_frame(candidates, split)
        .loc[:, ITEM_KEY_COLUMNS]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _split_frame(dataframe: pd.DataFrame, split: str) -> pd.DataFrame:
    if dataframe.empty or "split" not in dataframe.columns:
        return dataframe.iloc[0:0].copy()
    return dataframe.loc[dataframe["split"].astype(str) == split].reset_index(drop=True)


def _source_tuple(value: Any) -> tuple[str, ...]:
    sources = [source for source in str(value).split("|") if source]
    _validate_sources(set(sources))
    return tuple(sorted(set(sources), key=SOURCE_ORDER.__getitem__))


def _source_tuple_lookup(
    values: pd.Series,
    *,
    dropped_sources: set[str] | None = None,
) -> dict[object, tuple[str, ...]]:
    dropped_sources = dropped_sources or set()
    return {
        value: tuple(
            source for source in _source_tuple(value) if source not in dropped_sources
        )
        for value in values.drop_duplicates()
    }


def _validate_sources(sources: set[str]) -> None:
    unknown = sorted(sources - set(SOURCE_ORDER))
    if unknown:
        raise ValueError(f"unknown candidate source: {unknown}")


def _recomputed_best_source_rank(candidates: pd.DataFrame) -> pd.Series:
    if "best_source_rank" not in candidates.columns:
        return pd.Series([1] * len(candidates), dtype=int)
    ranks = pd.to_numeric(candidates["best_source_rank"], errors="coerce").fillna(1)
    return ranks.clip(lower=1).astype(int)


def _with_recomputed_source_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    if "best_source_rank" in result.columns:
        source_rank = result["best_source_rank"]
    else:
        source_rank = pd.Series([1] * len(result), index=result.index)
    rank_values = 1.0 / pd.to_numeric(
        source_rank,
        errors="coerce",
    ).fillna(1)
    result["source_rank_score"] = _minmax_by_user_window(
        result,
        rank_values,
    )
    if "candidate_sources" in result.columns:
        source_lookup = _source_tuple_lookup(result["candidate_sources"])
        source_counts = result["candidate_sources"].map(
            {value: len(sources) for value, sources in source_lookup.items()}
        )
    else:
        source_counts = pd.Series([0] * len(result), index=result.index)
    result["source_count_score"] = _minmax_by_user_window(
        result,
        source_counts.astype(float),
    )
    return result.reset_index(drop=True)


def _minmax_by_user_window(
    candidates: pd.DataFrame,
    values: pd.Series,
) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)
    if not set(USER_KEY_COLUMNS).issubset(candidates.columns):
        return _minmax_values(values)
    grouped = values.groupby([candidates[column] for column in USER_KEY_COLUMNS])
    min_value = grouped.transform("min")
    max_value = grouped.transform("max")
    denominator = (max_value - min_value).replace(0, pd.NA)
    return ((values - min_value) / denominator).fillna(0.0).astype(float)


def _minmax_values(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return pd.Series([0.0] * len(values), index=values.index, dtype=float)
    return ((values - min_value) / (max_value - min_value)).astype(float)


def _empty_like_with_source_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.iloc[0:0].copy()
    for column in (
        "candidate_sources",
        "primary_source",
        "best_source_rank",
        "has_reorder_source",
        "allow_seen",
        "source_rank_score",
        "source_count_score",
    ):
        if column not in result.columns:
            result[column] = pd.Series(dtype=object)
    return result.reset_index(drop=True)


def _diagnostic_splits(*frames: pd.DataFrame) -> list[str]:
    splits: set[str] = set()
    for frame in frames:
        if not frame.empty and "split" in frame.columns:
            splits.update(frame["split"].dropna().astype(str))
    return sorted(splits or {"valid", "test"})


def _avg_candidates_per_user(
    candidates: pd.DataFrame,
    target_users: pd.DataFrame,
    split: str,
) -> float:
    targets = _split_frame(target_users, split)
    target_count = len(targets.drop_duplicates(USER_KEY_COLUMNS))
    if target_count == 0:
        return 0.0
    return float(len(_split_frame(candidates, split))) / float(target_count)


def _source_coverage(candidates: pd.DataFrame, split: str) -> dict[str, object]:
    split_candidates = _split_frame(candidates, split)
    if split_candidates.empty or "candidate_sources" not in split_candidates.columns:
        return {"candidate_rows": int(len(split_candidates)), "source_rows": {}}
    source_counts: Counter[str] = Counter()
    source_values = (
        split_candidates["candidate_sources"].astype("string").value_counts()
    )
    for value, count in source_values.items():
        for source in _source_tuple(value):
            source_counts[source] += int(count)
    return {
        "candidate_rows": int(len(split_candidates)),
        "source_rows": dict(sorted(source_counts.items(), key=lambda item: item[0])),
    }
