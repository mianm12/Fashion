from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.contracts import RECOMMENDATION_CORE_ATTR_TYPES
from fashion_trend.recommendation.retrieval.popularity import SOURCE_COLUMNS


def build_preference_popularity_candidates(
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    *,
    top_attributes: int = 3,
    per_attribute_top_n: int = 4,
    per_user_top_n: int = 12,
    recent_weeks: int = 4,
) -> pd.DataFrame:
    """Return recent popular articles matching each user's top core preferences."""
    if (
        transactions.empty
        or article_attributes.empty
        or user_profile.empty
        or target_users.empty
    ):
        return _empty_source_frame()

    transactions = _with_string_ids(transactions)
    article_attributes = _core_article_attributes(article_attributes)
    user_profile = _with_string_ids(user_profile)
    target_users = _with_string_ids(target_users)
    if article_attributes.empty:
        return _empty_source_frame()

    frames: list[pd.DataFrame] = []
    for window in windows.to_dict("records"):
        window_targets = _target_users_for_window(target_users, window)
        if window_targets.empty:
            continue
        profile = _profile_for_window(
            user_profile,
            window,
            window_targets,
            top_attributes,
        )
        if profile.empty:
            continue
        popularity = _attribute_article_popularity(
            transactions,
            article_attributes,
            cutoff_week=int(window["cutoff_week"]),
            recent_weeks=recent_weeks,
            per_attribute_top_n=per_attribute_top_n,
        )
        if popularity.empty:
            continue
        matched = profile.merge(
            popularity,
            on=["attr_type", "attr_value"],
            how="inner",
        )
        ranked = _rank_preference_matches(matched, per_user_top_n)
        if ranked.empty:
            continue
        ranked.insert(0, "label_week", window["label_week"])
        ranked.insert(0, "cutoff_week", window["cutoff_week"])
        ranked.insert(0, "split", window["split"])
        frames.append(ranked.loc[:, SOURCE_COLUMNS])
    return _concat_source_frames(frames)


def _core_article_attributes(article_attributes: pd.DataFrame) -> pd.DataFrame:
    attributes = article_attributes.loc[
        article_attributes["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES),
        ["article_id", "attr_type", "attr_value"],
    ].copy()
    attributes["article_id"] = attributes["article_id"].astype(str)
    attributes["attr_type"] = attributes["attr_type"].astype(str)
    attributes["attr_value"] = attributes["attr_value"].astype(str)
    return attributes.drop_duplicates().reset_index(drop=True)


def _profile_for_window(
    user_profile: pd.DataFrame,
    window: dict[str, object],
    window_targets: pd.DataFrame,
    top_attributes: int,
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
    profile = profile.merge(window_targets, on="customer_id", how="inner")
    if profile.empty:
        return profile
    profile["preference_score"] = pd.to_numeric(
        profile["preference_score"],
        errors="raise",
    )
    sorted_profile = profile.sort_values(
        ["customer_id", "preference_score", "attr_type", "attr_value"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return sorted_profile.groupby("customer_id", group_keys=False).head(top_attributes)


def _attribute_article_popularity(
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    *,
    cutoff_week: int,
    recent_weeks: int,
    per_attribute_top_n: int,
) -> pd.DataFrame:
    week_id = pd.to_numeric(transactions["week_id"], errors="raise")
    mask = (week_id <= cutoff_week) & (week_id > cutoff_week - recent_weeks)
    history = transactions.loc[mask].copy()
    if history.empty:
        return pd.DataFrame(
            columns=["attr_type", "attr_value", "article_id", "article_popularity"]
        )

    popularity = (
        history.groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "article_popularity"})
    )
    matched = article_attributes.merge(popularity, on="article_id", how="inner")
    if matched.empty:
        return pd.DataFrame(
            columns=["attr_type", "attr_value", "article_id", "article_popularity"]
        )

    sorted_matches = matched.sort_values(
        ["attr_type", "attr_value", "article_popularity", "article_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    return sorted_matches.groupby(
        ["attr_type", "attr_value"],
        group_keys=False,
    ).head(per_attribute_top_n)


def _rank_preference_matches(
    matched: pd.DataFrame,
    per_user_top_n: int,
) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame(
            columns=["customer_id", "article_id", "source", "source_rank"]
        )

    scored = (
        matched.groupby(["customer_id", "article_id"], as_index=False)
        .agg(
            preference_score=("preference_score", "max"),
            article_popularity=("article_popularity", "max"),
        )
        .sort_values(
            ["customer_id", "preference_score", "article_popularity", "article_id"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
    )
    ranked = scored.groupby("customer_id", group_keys=False).head(per_user_top_n).copy()
    ranked["source"] = "preference_popularity"
    ranked["source_rank"] = ranked.groupby("customer_id").cumcount() + 1
    return ranked.loc[:, ["customer_id", "article_id", "source", "source_rank"]]


def _target_users_for_window(
    target_users: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    mask = (
        (target_users["split"] == window["split"])
        & (target_users["cutoff_week"] == window["cutoff_week"])
        & (target_users["label_week"] == window["label_week"])
    )
    return target_users.loc[mask, ["customer_id"]].drop_duplicates().copy()


def _concat_source_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_source_frame()
    result = pd.concat(non_empty, ignore_index=True)
    return _with_string_ids(result).loc[:, SOURCE_COLUMNS]


def _empty_source_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SOURCE_COLUMNS)


def _with_string_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("article_id", "customer_id", "attr_type", "attr_value"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result
