from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from fashion_trend.recommendation.contracts import (
    RECOMMENDATION_BACKFILL_CANDIDATES_PER_WINDOW,
)
from fashion_trend.recommendation.ranking.filters import filter_seen_items
from fashion_trend.recommendation.retrieval.candidates import build_candidate_items
from fashion_trend.recommendation.retrieval.popularity import (
    build_popularity_candidates,
    build_recent_popularity_candidates,
)

if TYPE_CHECKING:
    from fashion_trend.recommendation.methods.base import RecommendationContext


def append_backfill_items(
    context: RecommendationContext,
    candidates: pd.DataFrame,
    ranked: pd.DataFrame,
    weights: dict[str, float],
    backfill_mode: str | None,
) -> pd.DataFrame:
    if backfill_mode is None:
        return ranked
    missing_targets = _missing_target_users(context.target_users, ranked, context.top_k)
    if missing_targets.empty:
        return ranked

    strategy = _candidate_strategy_for_backfill(candidates)
    source = _build_backfill_source(backfill_mode, context, missing_targets)
    backfill_candidates = build_candidate_items(strategy, [source])
    if backfill_candidates.empty:
        return ranked

    if context.exclude_seen:
        backfill_candidates = filter_seen_items(
            backfill_candidates,
            context.transactions,
        )
    backfill_candidates = _drop_existing_candidates(backfill_candidates, ranked)
    backfill_items = _select_backfill_items(
        context.method,
        ranked,
        backfill_candidates,
        weights,
        backfill_mode,
        context.top_k,
    )
    if backfill_items.empty:
        return ranked
    result = pd.concat([ranked, backfill_items], ignore_index=True)
    return result.sort_values(
        ["customer_id", "split", "cutoff_week", "label_week", "rank"],
    ).reset_index(drop=True)


def _missing_target_users(
    target_users: pd.DataFrame,
    ranked: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    key_columns = ["split", "cutoff_week", "label_week", "customer_id"]
    counts = ranked.groupby(key_columns, as_index=False).size()
    underfilled = target_users.merge(counts, on=key_columns, how="left")
    underfilled["size"] = underfilled["size"].fillna(0).astype(int)
    return underfilled.loc[underfilled["size"] < top_k, target_users.columns].copy()


def _candidate_strategy_for_backfill(candidates: pd.DataFrame) -> str:
    if candidates.empty:
        return "popularity"
    strategies = candidates["strategy"].dropna().astype(str).unique().tolist()
    return strategies[0] if strategies else "popularity"


def _build_backfill_source(
    backfill_mode: str,
    context: RecommendationContext,
    missing_targets: pd.DataFrame,
) -> pd.DataFrame:
    builder = (
        build_popularity_candidates
        if backfill_mode == "popularity"
        else build_recent_popularity_candidates
    )
    return builder(
        context.transactions,
        context.windows,
        missing_targets,
        top_n=max(context.top_k, RECOMMENDATION_BACKFILL_CANDIDATES_PER_WINDOW),
    )


def _drop_existing_candidates(
    backfill_features: pd.DataFrame,
    existing_features: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["split", "cutoff_week", "label_week", "customer_id", "article_id"]
    existing_keys = existing_features.loc[:, key_columns].drop_duplicates()
    merged = backfill_features.merge(
        existing_keys.assign(_existing=True),
        on=key_columns,
        how="left",
    )
    return merged.loc[merged["_existing"].isna(), backfill_features.columns]


def _select_backfill_items(
    method_name: str,
    ranked: pd.DataFrame,
    candidates: pd.DataFrame,
    weights: dict[str, float],
    backfill_mode: str,
    top_k: int,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=ranked.columns)

    key_columns = ["split", "cutoff_week", "label_week", "customer_id"]
    counts = ranked.groupby(key_columns, as_index=False).size()
    counts = counts.rename(columns={"size": "_existing_count"})
    ordered = candidates.sort_values(
        [*key_columns, "best_source_rank", "article_id"],
        ascending=[True, True, True, True, True, True],
    )
    ordered = ordered.merge(counts, on=key_columns, how="left")
    ordered["_existing_count"] = ordered["_existing_count"].fillna(0).astype(int)
    ordered["_fill_rank"] = ordered.groupby(key_columns).cumcount() + 1
    ordered["_needed"] = top_k - ordered["_existing_count"]
    selected = ordered.loc[ordered["_fill_rank"] <= ordered["_needed"]].copy()
    if selected.empty:
        return pd.DataFrame(columns=ranked.columns)

    selected["method"] = method_name
    selected["rank"] = selected["_existing_count"] + selected["_fill_rank"]
    for column in ("pop_score", "recent_score", "sim_score", "trend_score"):
        selected[column] = 0.0

    source_score = _source_rank_score(selected["best_source_rank"])
    if backfill_mode == "popularity":
        selected["pop_score"] = source_score
    else:
        selected["recent_score"] = source_score
    selected["score"] = 0.0
    for feature, weight in weights.items():
        selected["score"] = selected["score"] + selected[feature] * weight

    columns = [
        "customer_id",
        "split",
        "cutoff_week",
        "label_week",
        "strategy",
        "article_id",
        "candidate_sources",
        "primary_source",
        "best_source_rank",
        "method",
        "rank",
        "score",
        "pop_score",
        "recent_score",
        "sim_score",
        "trend_score",
    ]
    output = selected.loc[:, columns].copy()
    if ranked.empty:
        return output
    return output.loc[:, ranked.columns]


def _source_rank_score(source_rank: pd.Series) -> pd.Series:
    rank = pd.to_numeric(source_rank, errors="raise").astype(float)
    max_value = max(float(rank.max()), 1.0)
    return ((max_value + 1.0 - rank) / max_value).clip(lower=0.0, upper=1.0)
