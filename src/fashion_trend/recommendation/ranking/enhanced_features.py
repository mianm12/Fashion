from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fashion_trend.recommendation.contracts import (
    ENHANCED_CANDIDATE_SOURCE_CAPS,
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
    RECOMMENDATION_CORE_ATTR_TYPES,
)
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

WINDOW_COLUMNS = ["split", "cutoff_week", "label_week"]
SOURCE_COLUMNS = [
    *WINDOW_COLUMNS,
    "customer_id",
    "article_id",
    "source",
    "source_rank",
]
ENHANCED_SCORE_COLUMNS = list(ENHANCED_RECOMMENDATION_SCORE_COLUMNS[4:])
ENHANCED_GROUP_COLUMNS = ["customer_id", *WINDOW_COLUMNS]
REORDER_RANK_CAP = ENHANCED_CANDIDATE_SOURCE_CAPS["reorder"]["top_n"]
VARIANT_RANK_CAP = ENHANCED_CANDIDATE_SOURCE_CAPS["product_variant"]["top_n"]
VARIANT_SEED_RANK_CAP = ENHANCED_CANDIDATE_SOURCE_CAPS["product_variant"]["seed_top_n"]
AGE_POPULARITY_RANK_CAP = ENHANCED_CANDIDATE_SOURCE_CAPS["age_popularity"][
    "per_user_top_n"
]
PREFERENCE_ATTRIBUTE_RANK_CAP = ENHANCED_CANDIDATE_SOURCE_CAPS["preference_popularity"][
    "per_attribute_top_n"
]
PREFERENCE_TOP_ATTRIBUTES = ENHANCED_CANDIDATE_SOURCE_CAPS["preference_popularity"][
    "top_attributes"
]
PREFERENCE_RECENT_WEEKS = ENHANCED_CANDIDATE_SOURCE_CAPS["preference_popularity"][
    "recent_weeks"
]
AGE_POPULARITY_RECENT_WEEKS = ENHANCED_CANDIDATE_SOURCE_CAPS["age_popularity"][
    "recent_weeks"
]
SOURCE_RANK_CAP_BY_SOURCE = {
    "popularity": ENHANCED_CANDIDATE_SOURCE_CAPS["popularity"]["top_n"],
    "similarity": ENHANCED_CANDIDATE_SOURCE_CAPS["similarity"]["top_n"],
    "trend": ENHANCED_CANDIDATE_SOURCE_CAPS["trend"]["top_n"],
    "reorder": REORDER_RANK_CAP,
    "product_variant": VARIANT_RANK_CAP,
    "age_popularity": AGE_POPULARITY_RANK_CAP,
    "preference_popularity": ENHANCED_CANDIDATE_SOURCE_CAPS["preference_popularity"][
        "per_user_top_n"
    ],
}


def add_enhanced_scores(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    customer_profile: pd.DataFrame | None,
    article_product_map: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add enhanced source-derived scores and keep legacy candidates at zero."""
    result = _with_string_ids(candidates)
    if result.empty:
        for column in ENHANCED_SCORE_COLUMNS:
            result[column] = 0.0
        return result

    result = add_reorder_score(result, transactions)
    result = add_variant_score(result, transactions, article_product_map)
    result = add_age_pop_score(result, transactions, customer_profile)
    result = add_preference_pop_score(
        result,
        transactions,
        article_attributes,
        user_profile,
    )
    result = add_source_rank_score(
        result,
        transactions,
        article_attributes,
        user_profile,
        trend_predictions,
        customer_profile,
        article_product_map,
    )
    result = add_source_count_score(result)
    return result


def add_reorder_score(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    result = _with_string_ids(candidates)
    raw_column = "_reorder_raw"
    if transactions.empty or not _has_source(result, "reorder").any():
        return _with_zero_normalized_score(result, raw_column, "reorder_score")

    transactions = _with_string_ids(transactions)
    frames: list[pd.DataFrame] = []
    for window in result[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        window_candidates = _frame_for_window(result, window)
        reorder_candidates = window_candidates.loc[
            _has_source(window_candidates, "reorder")
        ].copy()
        if reorder_candidates.empty:
            continue
        history = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
            recent_weeks=None,
        )
        ranks = _reorder_rank_frame(history, reorder_candidates)
        if not ranks.empty:
            frames.append(ranks)

    if not frames:
        return _with_zero_normalized_score(result, raw_column, "reorder_score")

    scores = pd.concat(frames, ignore_index=True)
    scores[raw_column] = 0.7 * _rank_norm(
        scores["last_purchase_rank"], REORDER_RANK_CAP
    ) + 0.3 * _rank_norm(scores["purchase_count_rank"], REORDER_RANK_CAP)
    return _merge_and_normalize_score(
        result,
        scores,
        raw_column=raw_column,
        output_column="reorder_score",
    )


def add_variant_score(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_product_map: pd.DataFrame | None,
) -> pd.DataFrame:
    result = _with_string_ids(candidates)
    raw_column = "_variant_raw"
    if (
        article_product_map is None
        or article_product_map.empty
        or transactions.empty
        or not _has_source(result, "product_variant").any()
    ):
        return _with_zero_normalized_score(result, raw_column, "variant_score")

    transactions = _with_string_ids(transactions)
    product_map = _clean_product_map(article_product_map)
    if product_map.empty:
        return _with_zero_normalized_score(result, raw_column, "variant_score")

    frames: list[pd.DataFrame] = []
    for window in result[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        window_candidates = _frame_for_window(result, window)
        variant_candidates = window_candidates.loc[
            _has_source(window_candidates, "product_variant")
        ].copy()
        if variant_candidates.empty:
            continue
        history = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
            recent_weeks=None,
        )
        scored = _variant_rank_frame(variant_candidates, history, product_map)
        if not scored.empty:
            frames.append(scored)

    if not frames:
        return _with_zero_normalized_score(result, raw_column, "variant_score")

    scores = pd.concat(frames, ignore_index=True)
    scores[raw_column] = 0.7 * _rank_norm(
        scores["variant_pop_rank"], VARIANT_RANK_CAP
    ) + 0.3 * _rank_norm(scores["seed_reorder_rank"], VARIANT_SEED_RANK_CAP)
    return _merge_and_normalize_score(
        result,
        scores,
        raw_column=raw_column,
        output_column="variant_score",
    )


def add_age_pop_score(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    customer_profile: pd.DataFrame | None,
) -> pd.DataFrame:
    result = _with_string_ids(candidates)
    raw_column = "_age_pop_raw"
    if (
        customer_profile is None
        or customer_profile.empty
        or transactions.empty
        or not _has_source(result, "age_popularity").any()
    ):
        return _with_zero_normalized_score(result, raw_column, "age_pop_score")

    profile = _clean_customer_profile(customer_profile)
    if profile.empty:
        return _with_zero_normalized_score(result, raw_column, "age_pop_score")

    transactions = _with_string_ids(transactions)
    frames: list[pd.DataFrame] = []
    for window in result[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        window_candidates = _frame_for_window(result, window)
        age_candidates = window_candidates.loc[
            _has_source(window_candidates, "age_popularity")
        ].copy()
        if age_candidates.empty:
            continue
        history = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
            recent_weeks=AGE_POPULARITY_RECENT_WEEKS,
        )
        scored = _age_popularity_rank_frame(age_candidates, history, profile)
        if not scored.empty:
            frames.append(scored)

    if not frames:
        return _with_zero_normalized_score(result, raw_column, "age_pop_score")

    scores = pd.concat(frames, ignore_index=True)
    scores[raw_column] = _rank_norm(
        scores["age_bucket_pop_rank"],
        AGE_POPULARITY_RANK_CAP,
    )
    return _merge_and_normalize_score(
        result,
        scores,
        raw_column=raw_column,
        output_column="age_pop_score",
    )


def add_preference_pop_score(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
) -> pd.DataFrame:
    result = _with_string_ids(candidates)
    raw_column = "_preference_pop_raw"
    if (
        user_profile is None
        or user_profile.empty
        or article_attributes.empty
        or transactions.empty
        or not _has_source(result, "preference_popularity").any()
    ):
        return _with_zero_normalized_score(
            result,
            raw_column,
            "preference_pop_score",
        )

    transactions = _with_string_ids(transactions)
    attributes = _core_article_attributes(article_attributes)
    if attributes.empty:
        return _with_zero_normalized_score(
            result,
            raw_column,
            "preference_pop_score",
        )

    profile = _with_string_ids(user_profile)
    frames: list[pd.DataFrame] = []
    for window in result[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        window_candidates = _frame_for_window(result, window)
        preference_candidates = window_candidates.loc[
            _has_source(window_candidates, "preference_popularity")
        ].copy()
        if preference_candidates.empty:
            continue
        history = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
            recent_weeks=PREFERENCE_RECENT_WEEKS,
        )
        scored = _preference_popularity_score_frame(
            preference_candidates,
            history,
            attributes,
            profile,
            window,
        )
        if not scored.empty:
            frames.append(scored)

    if not frames:
        return _with_zero_normalized_score(
            result,
            raw_column,
            "preference_pop_score",
        )

    scores = pd.concat(frames, ignore_index=True)
    scores[raw_column] = scores["preference_pop_value"]
    return _merge_and_normalize_score(
        result,
        scores,
        raw_column=raw_column,
        output_column="preference_pop_score",
    )


def add_source_rank_score(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    customer_profile: pd.DataFrame | None,
    article_product_map: pd.DataFrame | None,
) -> pd.DataFrame:
    result = _with_string_ids(candidates)
    raw_column = "_source_rank_raw"
    if (
        "candidate_sources" not in result.columns
        or "best_source_rank" not in result.columns
    ):
        return _with_zero_normalized_score(result, raw_column, "source_rank_score")

    source_sets = _candidate_source_sets(result)
    source_rank_cap = max(
        (
            _source_rank_cap(source)
            for sources in source_sets
            for source in sources
            if source in SOURCE_RANK_CAP_BY_SOURCE
        ),
        default=1,
    )
    result[raw_column] = _rank_norm(result["best_source_rank"], source_rank_cap)
    return _normalize_score_column(result, raw_column, "source_rank_score")


def add_source_count_score(candidates: pd.DataFrame) -> pd.DataFrame:
    result = _with_string_ids(candidates)
    raw_column = "_source_count_raw"
    if "candidate_sources" not in result.columns:
        return _with_zero_normalized_score(result, raw_column, "source_count_score")

    source_sets = _candidate_source_sets(result)
    source_count_cap = max(
        1,
        len({source for sources in source_sets for source in sources}),
    )
    result[raw_column] = [
        min(len(sources), source_count_cap) / source_count_cap
        for sources in source_sets
    ]
    return _normalize_score_column(result, raw_column, "source_count_score")


def _source_rank_score_frame(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    customer_profile: pd.DataFrame | None,
    article_product_map: pd.DataFrame | None,
    *,
    raw_column: str,
) -> pd.DataFrame:
    key_columns = _score_key_columns(candidates)
    output_columns = [*key_columns, raw_column]
    source_frame = _source_rank_source_frame(
        candidates,
        transactions,
        article_attributes,
        user_profile,
        trend_predictions,
        customer_profile,
        article_product_map,
    )
    if source_frame.empty:
        return pd.DataFrame(columns=output_columns)

    join_columns = [column for column in key_columns if column != "strategy"]
    candidate_keys = candidates.loc[
        :,
        [*key_columns, "candidate_sources"],
    ].drop_duplicates()
    matched = candidate_keys.merge(
        source_frame.loc[:, [*join_columns, "source", "source_rank"]],
        on=join_columns,
        how="inner",
    )
    if matched.empty:
        return pd.DataFrame(columns=output_columns)

    matched = matched.loc[
        [
            str(source) in set(str(sources).split("|"))
            for source, sources in zip(
                matched["source"],
                matched["candidate_sources"],
                strict=False,
            )
        ]
    ].copy()
    if matched.empty:
        return pd.DataFrame(columns=output_columns)

    matched[raw_column] = [
        _rank_norm(pd.Series([rank]), _source_rank_cap(str(source))).iloc[0]
        for source, rank in zip(
            matched["source"],
            matched["source_rank"],
            strict=False,
        )
    ]
    return (
        matched.groupby(key_columns, as_index=False)[raw_column]
        .max()
        .loc[:, output_columns]
    )


def _source_rank_source_frame(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    customer_profile: pd.DataFrame | None,
    article_product_map: pd.DataFrame | None,
) -> pd.DataFrame:
    source_names = _candidate_source_name_set(candidates)
    if not source_names:
        return _empty_source_frame()

    windows = candidates.loc[:, WINDOW_COLUMNS].drop_duplicates().reset_index(drop=True)
    target_users = candidates.loc[
        :,
        [*WINDOW_COLUMNS, "customer_id"],
    ].drop_duplicates()
    frames: list[pd.DataFrame] = []
    reorder_candidates = _empty_source_frame()

    if "popularity" in source_names and not transactions.empty:
        frames.append(
            build_recent_popularity_candidates(
                transactions,
                windows,
                target_users,
                top_n=_source_cap("popularity", "top_n"),
            )
        )
    if (
        "similarity" in source_names
        and user_profile is not None
        and not user_profile.empty
        and not article_attributes.empty
    ):
        frames.append(
            build_attribute_similarity_candidates(
                user_profile,
                article_attributes,
                windows,
                target_users,
                top_n=_source_cap("similarity", "top_n"),
            )
        )
    if (
        "trend" in source_names
        and trend_predictions is not None
        and not trend_predictions.empty
        and not article_attributes.empty
    ):
        frames.append(
            build_trend_candidates(
                trend_predictions,
                article_attributes,
                windows,
                target_users,
                top_n=_source_cap("trend", "top_n"),
            )
        )
    if ("reorder" in source_names or "product_variant" in source_names) and (
        not transactions.empty
    ):
        reorder_candidates = build_reorder_candidates(
            transactions,
            windows,
            target_users,
            top_n=_source_cap("reorder", "top_n"),
        )
        if "reorder" in source_names:
            frames.append(reorder_candidates)
    if (
        "product_variant" in source_names
        and article_product_map is not None
        and not article_product_map.empty
        and not reorder_candidates.empty
        and not transactions.empty
    ):
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
    if (
        "age_popularity" in source_names
        and customer_profile is not None
        and not customer_profile.empty
        and not transactions.empty
    ):
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
    if (
        "preference_popularity" in source_names
        and user_profile is not None
        and not user_profile.empty
        and not article_attributes.empty
        and not transactions.empty
    ):
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
    return _concat_source_frames(frames)


def _reorder_rank_frame(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [
        *_score_key_columns(candidates),
        "last_purchase_rank",
        "purchase_count_rank",
    ]
    if history.empty:
        return pd.DataFrame(columns=output_columns)

    candidate_pairs = candidates.loc[
        :,
        ["customer_id", "article_id"],
    ].drop_duplicates()
    history = history.merge(candidate_pairs.loc[:, ["customer_id"]].drop_duplicates())
    if history.empty:
        return pd.DataFrame(columns=output_columns)

    grouped = (
        history.assign(week_id=pd.to_numeric(history["week_id"], errors="raise"))
        .groupby(["customer_id", "article_id"], as_index=False)
        .agg(
            last_purchase_week=("week_id", "max"),
            purchase_count=("week_id", "size"),
        )
    )
    if grouped.empty:
        return pd.DataFrame(columns=output_columns)

    last_ranked = grouped.sort_values(
        ["customer_id", "last_purchase_week", "purchase_count", "article_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).copy()
    last_ranked["last_purchase_rank"] = (
        last_ranked.groupby("customer_id").cumcount() + 1
    )
    count_ranked = grouped.sort_values(
        ["customer_id", "purchase_count", "last_purchase_week", "article_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).copy()
    count_ranked["purchase_count_rank"] = (
        count_ranked.groupby("customer_id").cumcount() + 1
    )
    ranks = last_ranked.loc[
        :,
        ["customer_id", "article_id", "last_purchase_rank"],
    ].merge(
        count_ranked.loc[
            :,
            ["customer_id", "article_id", "purchase_count_rank"],
        ],
        on=["customer_id", "article_id"],
        how="inner",
    )
    return candidates.merge(ranks, on=["customer_id", "article_id"], how="inner").loc[
        :,
        output_columns,
    ]


def _variant_rank_frame(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    product_map: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [
        *_score_key_columns(candidates),
        "variant_pop_rank",
        "seed_reorder_rank",
    ]
    if history.empty:
        return pd.DataFrame(columns=output_columns)

    seed_ranks = _reorder_seed_ranks(history, candidates, product_map)
    if seed_ranks.empty:
        return pd.DataFrame(columns=output_columns)

    candidate_products = candidates.merge(product_map, on="article_id", how="inner")
    if candidate_products.empty:
        return pd.DataFrame(columns=output_columns)

    variants = candidate_products.merge(
        seed_ranks,
        on=["customer_id", "product_code"],
        how="inner",
    )
    variants = variants.loc[variants["article_id"] != variants["seed_article_id"]]
    if variants.empty:
        return pd.DataFrame(columns=output_columns)

    popularity = (
        history.groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "article_popularity"})
    )
    variants = variants.merge(popularity, on="article_id", how="left")
    variants["article_popularity"] = variants["article_popularity"].fillna(0)
    scored = (
        variants.groupby(_score_key_columns(candidates), as_index=False)
        .agg(
            article_popularity=("article_popularity", "max"),
            seed_reorder_rank=("seed_reorder_rank", "min"),
        )
        .sort_values(
            [
                "customer_id",
                "article_popularity",
                "seed_reorder_rank",
                "article_id",
            ],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
    )
    scored["variant_pop_rank"] = scored.groupby("customer_id").cumcount() + 1
    return scored.loc[:, output_columns]


def _reorder_seed_ranks(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    product_map: pd.DataFrame,
) -> pd.DataFrame:
    candidate_customers = candidates.loc[:, ["customer_id"]].drop_duplicates()
    history = history.merge(candidate_customers, on="customer_id", how="inner")
    if history.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "product_code",
                "seed_article_id",
                "seed_reorder_rank",
            ]
        )

    ranked = (
        history.assign(week_id=pd.to_numeric(history["week_id"], errors="raise"))
        .groupby(["customer_id", "article_id"], as_index=False)
        .agg(
            last_purchase_week=("week_id", "max"),
            purchase_count=("week_id", "size"),
        )
        .sort_values(
            ["customer_id", "last_purchase_week", "purchase_count", "article_id"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
    )
    ranked["seed_reorder_rank"] = ranked.groupby("customer_id").cumcount() + 1
    seeds = ranked.loc[ranked["seed_reorder_rank"] <= VARIANT_SEED_RANK_CAP].copy()
    seeds = seeds.rename(columns={"article_id": "seed_article_id"})
    return seeds.merge(
        product_map.rename(columns={"article_id": "seed_article_id"}),
        on="seed_article_id",
        how="inner",
    ).loc[
        :,
        ["customer_id", "product_code", "seed_article_id", "seed_reorder_rank"],
    ]


def _age_popularity_rank_frame(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [*_score_key_columns(candidates), "age_bucket_pop_rank"]
    if history.empty:
        return pd.DataFrame(columns=output_columns)

    candidate_buckets = candidates.merge(profile, on="customer_id", how="inner")
    if candidate_buckets.empty:
        return pd.DataFrame(columns=output_columns)

    history_with_bucket = history.merge(profile, on="customer_id", how="inner")
    if history_with_bucket.empty:
        return pd.DataFrame(columns=output_columns)

    ranked = (
        history_with_bucket.groupby(["age_bucket", "article_id"], as_index=False)
        .size()
        .rename(columns={"size": "purchase_count"})
        .sort_values(
            ["age_bucket", "purchase_count", "article_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    ranked["age_bucket_pop_rank"] = ranked.groupby("age_bucket").cumcount() + 1
    return candidate_buckets.merge(
        ranked.loc[:, ["age_bucket", "article_id", "age_bucket_pop_rank"]],
        on=["age_bucket", "article_id"],
        how="inner",
    ).loc[:, output_columns]


def _preference_popularity_score_frame(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    output_columns = [*_score_key_columns(candidates), "preference_pop_value"]
    if history.empty:
        return pd.DataFrame(columns=output_columns)

    profile = _top_preference_profile(user_profile, candidates, window)
    if profile.empty:
        return pd.DataFrame(columns=output_columns)

    popularity = (
        history.groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "article_popularity"})
    )
    attribute_popularity = article_attributes.merge(
        popularity,
        on="article_id",
        how="inner",
    )
    if attribute_popularity.empty:
        return pd.DataFrame(columns=output_columns)

    ranked_popularity = attribute_popularity.sort_values(
        ["attr_type", "attr_value", "article_popularity", "article_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    ranked_popularity["attribute_pop_rank"] = (
        ranked_popularity.groupby(["attr_type", "attr_value"]).cumcount() + 1
    )
    ranked_popularity = ranked_popularity.loc[
        ranked_popularity["attribute_pop_rank"] <= PREFERENCE_ATTRIBUTE_RANK_CAP
    ]
    candidate_attributes = candidates.merge(
        article_attributes,
        on="article_id",
        how="inner",
    )
    matched = candidate_attributes.merge(
        profile,
        on=["customer_id", "attr_type", "attr_value"],
        how="inner",
    ).merge(
        ranked_popularity.loc[
            :,
            ["attr_type", "attr_value", "article_id", "attribute_pop_rank"],
        ],
        on=["attr_type", "attr_value", "article_id"],
        how="inner",
    )
    if matched.empty:
        return pd.DataFrame(columns=output_columns)

    matched["preference_pop_value"] = pd.to_numeric(
        matched["preference_score"],
        errors="raise",
    ) * _rank_norm(matched["attribute_pop_rank"], PREFERENCE_ATTRIBUTE_RANK_CAP)
    return (
        matched.groupby(_score_key_columns(candidates), as_index=False)[
            "preference_pop_value"
        ]
        .max()
        .loc[:, output_columns]
    )


def _top_preference_profile(
    user_profile: pd.DataFrame,
    candidates: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    mask = (
        (user_profile["split"] == window["split"])
        & (user_profile["cutoff_week"] == window["cutoff_week"])
        & (user_profile["label_week"] == window["label_week"])
        & (user_profile["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES))
    )
    profile = user_profile.loc[
        mask,
        ["customer_id", "attr_type", "attr_value", "preference_score"],
    ].copy()
    if profile.empty:
        return profile

    profile = profile.merge(
        candidates.loc[:, ["customer_id"]].drop_duplicates(),
        on="customer_id",
        how="inner",
    )
    profile["preference_score"] = pd.to_numeric(
        profile["preference_score"],
        errors="raise",
    )
    profile = profile.sort_values(
        ["customer_id", "preference_score", "attr_type", "attr_value"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return profile.groupby("customer_id", group_keys=False).head(
        PREFERENCE_TOP_ATTRIBUTES
    )


def _clean_product_map(article_product_map: pd.DataFrame) -> pd.DataFrame:
    _require_columns(article_product_map, ("article_id", "product_code"), "product map")
    product_map = article_product_map.loc[:, ["article_id", "product_code"]].copy()
    product_map = product_map.loc[product_map["product_code"].notna()].copy()
    product_map["article_id"] = product_map["article_id"].astype(str)
    product_map["product_code"] = product_map["product_code"].astype(str)
    product_map = product_map.loc[product_map["product_code"] != ""]
    return product_map.drop_duplicates().reset_index(drop=True)


def _clean_customer_profile(customer_profile: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        customer_profile,
        ("customer_id", "age_bucket"),
        "customer profile",
    )
    profile = customer_profile.loc[:, ["customer_id", "age_bucket"]].copy()
    profile = profile.loc[profile["age_bucket"].notna()].copy()
    profile["customer_id"] = profile["customer_id"].astype(str)
    profile["age_bucket"] = profile["age_bucket"].astype(str)
    profile = profile.loc[profile["age_bucket"] != ""]
    return profile.drop_duplicates("customer_id").reset_index(drop=True)


def _core_article_attributes(article_attributes: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        article_attributes,
        ("article_id", "attr_type", "attr_value"),
        "article attributes",
    )
    attributes = article_attributes.loc[
        article_attributes["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES),
        ["article_id", "attr_type", "attr_value"],
    ].copy()
    return _with_string_ids(attributes).drop_duplicates().reset_index(drop=True)


def _merge_and_normalize_score(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    raw_column: str,
    output_column: str,
) -> pd.DataFrame:
    result = candidates.copy()
    key_columns = _score_key_columns(result)
    if scores.empty:
        return _with_zero_normalized_score(result, raw_column, output_column)
    result = result.merge(
        scores.loc[:, [*key_columns, raw_column]].drop_duplicates(key_columns),
        on=key_columns,
        how="left",
    )
    result[raw_column] = result[raw_column].fillna(0.0)
    return _normalize_score_column(result, raw_column, output_column)


def _with_zero_normalized_score(
    candidates: pd.DataFrame,
    raw_column: str,
    output_column: str,
) -> pd.DataFrame:
    result = candidates.copy()
    result[raw_column] = 0.0
    return _normalize_score_column(result, raw_column, output_column)


def _normalize_score_column(
    dataframe: pd.DataFrame,
    raw_column: str,
    output_column: str,
) -> pd.DataFrame:
    result = _minmax_normalize_by_group(
        dataframe,
        value_column=raw_column,
        output_column=output_column,
        group_columns=ENHANCED_GROUP_COLUMNS,
    )
    return result.drop(columns=[raw_column])


def _minmax_normalize_by_group(
    dataframe: pd.DataFrame,
    value_column: str,
    output_column: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    result = dataframe.copy()
    values = pd.to_numeric(result[value_column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{value_column} contains non-finite values")

    grouped = values.groupby([result[column] for column in group_columns])
    min_value = grouped.transform("min")
    max_value = grouped.transform("max")
    denominator = max_value - min_value
    result[output_column] = np.where(
        denominator == 0,
        0.0,
        (values - min_value) / denominator,
    )
    if not np.isfinite(result[output_column].to_numpy(dtype=float)).all():
        raise ValueError(f"{output_column} contains non-finite values")
    return result


def _rank_norm(rank: pd.Series, cap: int) -> pd.Series:
    values = pd.to_numeric(rank, errors="raise").astype(float)
    result = pd.Series(0.0, index=values.index, dtype=float)
    valid = values.notna()
    if not np.isfinite(values.loc[valid].to_numpy(dtype=float)).all():
        raise ValueError("rank contains non-finite values")
    if (values.loc[valid] < 1).any():
        raise ValueError("rank must be one-based")
    result.loc[valid] = ((cap - values.loc[valid] + 1.0) / cap).clip(
        lower=0.0,
        upper=1.0,
    )
    return result


def _has_source(dataframe: pd.DataFrame, source: str) -> pd.Series:
    if "candidate_sources" not in dataframe.columns:
        return pd.Series(False, index=dataframe.index)
    return (
        dataframe["candidate_sources"]
        .fillna("")
        .astype(str)
        .map(lambda value: source in value.split("|"))
    )


def _candidate_source_sets(dataframe: pd.DataFrame) -> list[tuple[str, ...]]:
    if "candidate_sources" not in dataframe.columns:
        return [tuple() for _index in dataframe.index]
    return [
        tuple(source for source in str(value).split("|") if source)
        for value in dataframe["candidate_sources"].fillna("")
    ]


def _candidate_source_name_set(dataframe: pd.DataFrame) -> set[str]:
    return {
        source for sources in _candidate_source_sets(dataframe) for source in sources
    }


def _source_rank_cap(source: str) -> int:
    if source not in SOURCE_RANK_CAP_BY_SOURCE:
        raise ValueError(f"unknown candidate source: {source}")
    return int(SOURCE_RANK_CAP_BY_SOURCE[source])


def _source_cap(source: str, key: str) -> int:
    return int(ENHANCED_CANDIDATE_SOURCE_CAPS[source][key])


def _score_key_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = [*WINDOW_COLUMNS, "customer_id", "article_id"]
    if "strategy" in dataframe.columns:
        columns.insert(3, "strategy")
    return columns


def _concat_source_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_source_frame()
    return pd.concat(non_empty, ignore_index=True).loc[:, SOURCE_COLUMNS]


def _empty_source_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SOURCE_COLUMNS)


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _frame_for_window(
    frame: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    mask = (
        (frame["split"] == window["split"])
        & (frame["cutoff_week"] == window["cutoff_week"])
        & (frame["label_week"] == window["label_week"])
    )
    return frame.loc[mask].copy()


def _transactions_for_window(
    transactions: pd.DataFrame,
    cutoff_week: int,
    recent_weeks: int | None,
) -> pd.DataFrame:
    week_id = pd.to_numeric(transactions["week_id"], errors="raise")
    mask = week_id <= cutoff_week
    if recent_weeks is not None:
        mask &= week_id > cutoff_week - recent_weeks
    return transactions.loc[mask].copy()


def _with_string_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("article_id", "customer_id", "attr_type", "attr_value", "split"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result
