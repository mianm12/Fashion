from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic, write_parquet_atomic
from fashion_trend.recommendation.contracts import (
    CANDIDATE_ITEM_COLUMNS,
    ENHANCED_CANDIDATE_ITEM_COLUMNS,
    ENHANCED_CANDIDATE_SOURCE_CAPS,
    RECOMMENDATION_CANDIDATE_STRATEGIES,
    RECOMMENDATION_CANDIDATES_PER_SOURCE,
    SOURCE_ORDER,
)
from fashion_trend.recommendation.freshness import build_artifact_metadata
from fashion_trend.recommendation.paths import candidate_items_path
from fashion_trend.recommendation.retrieval.attributes import (
    build_attribute_similarity_candidates,
)
from fashion_trend.recommendation.retrieval.customer_segments import (
    build_age_popularity_candidates,
)
from fashion_trend.recommendation.retrieval.popularity import (
    build_recent_popularity_candidates,
)
from fashion_trend.recommendation.retrieval.preference_popularity import (
    build_preference_popularity_candidates,
)
from fashion_trend.recommendation.retrieval.product_variants import (
    build_product_variant_candidates,
)
from fashion_trend.recommendation.retrieval.reorder import build_reorder_candidates
from fashion_trend.recommendation.retrieval.trend import build_trend_candidates

SOURCE_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "article_id",
    "source",
    "source_rank",
)
GROUP_COLUMNS = ("split", "cutoff_week", "label_week", "customer_id", "article_id")
CANDIDATE_INPUT_KEYS_BY_STRATEGY = {
    "popularity": ("weekly_transactions", "time_windows", "target_users"),
    "similarity": (
        "article_attributes",
        "time_windows",
        "target_users",
        "user_profile",
    ),
    "trend_union": (
        "article_attributes",
        "trend_predictions",
        "time_windows",
        "target_users",
    ),
    "default": (
        "weekly_transactions",
        "article_attributes",
        "trend_predictions",
        "time_windows",
        "target_users",
        "user_profile",
    ),
    "enhanced_default": (
        "weekly_transactions",
        "article_attributes",
        "trend_predictions",
        "time_windows",
        "target_users",
        "user_profile",
        "customer_profile",
        "article_product_map",
    ),
}


def validate_candidate_strategy(strategy: str) -> None:
    if strategy not in RECOMMENDATION_CANDIDATE_STRATEGIES:
        choices = ", ".join(RECOMMENDATION_CANDIDATE_STRATEGIES)
        raise ValueError(f"未知候选 strategy: {strategy}. 可用 strategy: {choices}")


def candidate_input_paths_for_strategy(
    strategy: str,
    input_paths: dict[str, str] | None,
) -> dict[str, str]:
    validate_candidate_strategy(strategy)
    available_paths = dict(input_paths or {})
    return {
        key: available_paths[key]
        for key in CANDIDATE_INPUT_KEYS_BY_STRATEGY[strategy]
        if key in available_paths
    }


def build_candidate_items(
    strategy: str,
    source_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    validate_candidate_strategy(strategy)
    non_empty_frames = [frame for frame in source_frames if not frame.empty]
    if strategy == "trend_union" and not non_empty_frames:
        raise FileNotFoundError("trend_union strategy requires trend source candidates")
    if not non_empty_frames:
        columns = (
            ENHANCED_CANDIDATE_ITEM_COLUMNS
            if strategy == "enhanced_default"
            else CANDIDATE_ITEM_COLUMNS
        )
        return pd.DataFrame(columns=columns)

    sources = pd.concat(non_empty_frames, ignore_index=True)
    sources = _prepare_source_frame(sources)
    sorted_sources = sources.sort_values(
        [
            "split",
            "cutoff_week",
            "label_week",
            "customer_id",
            "article_id",
            "source_rank",
            "_source_order",
        ],
        kind="mergesort",
    )
    result = _aggregate_candidate_sources(sorted_sources, strategy)
    if strategy == "enhanced_default":
        result["allow_seen"] = result["has_reorder_source"]
        return result.loc[:, list(ENHANCED_CANDIDATE_ITEM_COLUMNS)]

    return result.loc[:, list(CANDIDATE_ITEM_COLUMNS)]


def build_source_frames_for_frames(
    strategy: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    customer_profile: pd.DataFrame | None = None,
    article_product_map: pd.DataFrame | None = None,
) -> list[pd.DataFrame]:
    validate_candidate_strategy(strategy)
    if strategy == "enhanced_default":
        _require_enhanced_inputs(
            article_attributes=article_attributes,
            trend_predictions=trend_predictions,
            user_profile=user_profile,
            customer_profile=customer_profile,
            article_product_map=article_product_map,
        )

    frames: list[pd.DataFrame] = []
    if strategy in {"popularity", "default", "enhanced_default"}:
        frames.append(
            build_recent_popularity_candidates(
                transactions,
                windows,
                target_users,
                top_n=_source_cap("popularity", "top_n"),
            )
        )
    if strategy in {"similarity", "default", "enhanced_default"}:
        if user_profile is None or article_attributes is None:
            raise FileNotFoundError(
                "similarity strategy requires user profile and article attributes"
            )
        frames.append(
            build_attribute_similarity_candidates(
                user_profile,
                article_attributes,
                windows,
                target_users,
                top_n=_source_cap("similarity", "top_n"),
            )
        )
    if strategy in {"trend_union", "default", "enhanced_default"}:
        if trend_predictions is None or article_attributes is None:
            raise FileNotFoundError(
                "trend strategy requires trend predictions and article attributes"
            )
        frames.append(
            build_trend_candidates(
                trend_predictions,
                article_attributes,
                windows,
                target_users,
                top_n=_source_cap("trend", "top_n"),
            )
        )
    if strategy == "enhanced_default":
        reorder_candidates = build_reorder_candidates(
            transactions,
            windows,
            target_users,
            top_n=_source_cap("reorder", "top_n"),
        )
        frames.append(reorder_candidates)
        frames.append(
            build_product_variant_candidates(
                reorder_candidates,
                transactions,
                article_product_map,
                windows,
                seed_top_n=_source_cap("product_variant", "seed_top_n"),
                per_seed_top_n=_source_cap("product_variant", "per_seed_top_n"),
                top_n=_source_cap("product_variant", "top_n"),
            )
        )
        frames.append(
            build_age_popularity_candidates(
                transactions,
                customer_profile,
                windows,
                target_users,
                pool_top_n=_source_cap("age_popularity", "pool_top_n"),
                per_user_top_n=_source_cap("age_popularity", "per_user_top_n"),
                recent_weeks=_source_cap("age_popularity", "recent_weeks"),
            )
        )
        frames.append(
            build_preference_popularity_candidates(
                transactions,
                article_attributes,
                user_profile,
                windows,
                target_users,
                top_attributes=_source_cap("preference_popularity", "top_attributes"),
                per_attribute_top_n=_source_cap(
                    "preference_popularity",
                    "per_attribute_top_n",
                ),
                per_user_top_n=_source_cap(
                    "preference_popularity",
                    "per_user_top_n",
                ),
                recent_weeks=_source_cap("preference_popularity", "recent_weeks"),
            )
        )
    return frames


def build_and_write_candidate_items(
    strategy: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    input_paths: dict[str, str] | None = None,
    customer_profile: pd.DataFrame | None = None,
    article_product_map: pd.DataFrame | None = None,
) -> Path:
    candidate_input_paths = candidate_input_paths_for_strategy(strategy, input_paths)
    source_frames = build_source_frames_for_frames(
        strategy=strategy,
        transactions=transactions,
        article_attributes=article_attributes,
        trend_predictions=trend_predictions,
        windows=windows,
        target_users=target_users,
        user_profile=user_profile,
        customer_profile=customer_profile,
        article_product_map=article_product_map,
    )
    candidates = build_candidate_items(strategy, source_frames)
    output_path = candidate_items_path(strategy)
    write_parquet_atomic(candidates, output_path)
    candidate_rows = int(len(candidates))
    write_json_atomic(
        {
            **build_artifact_metadata(
                name="recommendation_candidates",
                input_artifacts=candidate_input_paths,
                output_artifacts={"candidate_items": str(output_path)},
                schema_version=1,
                algorithm_version="recommendation-candidates-v1",
                config={
                    "strategy": strategy,
                    "candidates_per_source": RECOMMENDATION_CANDIDATES_PER_SOURCE,
                    **_enhanced_config(strategy),
                },
                row_counts={"candidate_rows": candidate_rows},
            ),
            "strategy": strategy,
            "candidate_rows": candidate_rows,
        },
        output_path.with_name("metadata.json"),
    )
    return output_path


def _prepare_source_frame(source_frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(source_frame.columns))
    if missing_columns:
        raise ValueError(f"候选 source frame 缺少必要字段: {missing_columns}")
    prepared = source_frame.loc[:, SOURCE_COLUMNS].copy()
    prepared["article_id"] = prepared["article_id"].astype(str)
    prepared["customer_id"] = prepared["customer_id"].astype(str)
    prepared["source"] = prepared["source"].astype(str)
    unknown_sources = sorted(set(prepared["source"]) - set(SOURCE_ORDER))
    if unknown_sources:
        raise ValueError(f"未知候选 source: {unknown_sources}")
    prepared["source_rank"] = pd.to_numeric(
        prepared["source_rank"],
        errors="raise",
    ).astype(int)
    prepared["_source_order"] = prepared["source"].map(SOURCE_ORDER).astype(int)
    return prepared


def _aggregate_candidate_sources(
    sorted_sources: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    first_rows = sorted_sources.drop_duplicates(list(GROUP_COLUMNS), keep="first")
    result = first_rows.loc[:, list(GROUP_COLUMNS) + ["source", "source_rank"]].rename(
        columns={"source": "primary_source", "source_rank": "best_source_rank"}
    )
    result.insert(3, "strategy", strategy)

    source_flags = _source_presence_flags(sorted_sources)
    result = result.merge(
        source_flags,
        on=list(GROUP_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result["_source_mask"] = result["_source_mask"].fillna(0).astype("int64")
    result["candidate_sources"] = _compose_candidate_sources(result)
    if strategy == "enhanced_default":
        result["has_reorder_source"] = _has_source(result["_source_mask"], "reorder")
    return result


def _source_presence_flags(sorted_sources: pd.DataFrame) -> pd.DataFrame:
    presence = sorted_sources.loc[
        :, list(GROUP_COLUMNS) + ["source", "_source_order"]
    ].drop_duplicates(list(GROUP_COLUMNS) + ["source"])
    presence["_source_bit"] = (2 ** presence["_source_order"].astype("int64")).astype(
        "int64"
    )
    return (
        presence.groupby(list(GROUP_COLUMNS), sort=False, as_index=False)["_source_bit"]
        .sum()
        .rename(columns={"_source_bit": "_source_mask"})
    )


def _compose_candidate_sources(flag_frame: pd.DataFrame) -> pd.Series:
    candidate_sources = pd.Series("", index=flag_frame.index, dtype=object)
    for source in SOURCE_ORDER:
        present = _has_source(flag_frame["_source_mask"], source)
        prefix = candidate_sources.where(
            candidate_sources.eq(""), candidate_sources + "|"
        )
        candidate_sources = candidate_sources.mask(present, prefix + source)
    return candidate_sources


def _has_source(source_mask: pd.Series, source: str) -> pd.Series:
    source_bit = 1 << SOURCE_ORDER[source]
    return (source_mask.astype("int64") & source_bit).ne(0)


def _require_enhanced_inputs(
    *,
    article_attributes: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    user_profile: pd.DataFrame | None,
    customer_profile: pd.DataFrame | None,
    article_product_map: pd.DataFrame | None,
) -> None:
    missing = []
    if article_attributes is None:
        missing.append("article_attributes")
    if trend_predictions is None:
        missing.append("trend_predictions")
    if user_profile is None:
        missing.append("user_profile")
    if customer_profile is None:
        missing.append("customer_profile")
    if article_product_map is None:
        missing.append("article_product_map")
    if missing:
        raise FileNotFoundError(
            "enhanced_default strategy requires input artifacts: "
            f"{', '.join(missing)}"
        )


def _source_cap(source: str, key: str) -> int:
    return int(ENHANCED_CANDIDATE_SOURCE_CAPS[source][key])


def _enhanced_config(strategy: str) -> dict[str, object]:
    if strategy != "enhanced_default":
        return {}
    return {
        "source_caps": ENHANCED_CANDIDATE_SOURCE_CAPS,
        "seen_policy": "source_level_reorder_only",
        "include_seen_for_reorder": True,
        "source_order": SOURCE_ORDER,
    }
