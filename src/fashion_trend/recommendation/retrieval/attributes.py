from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.contracts import (
    RECOMMENDATION_CORE_ATTR_TYPES,
    RECOMMENDATION_PROFILE_TOP_ATTRIBUTES,
)
from fashion_trend.recommendation.retrieval.popularity import SOURCE_COLUMNS


def build_attribute_similarity_candidates(
    user_profile: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return user-specific candidates ranked by profile-attribute matches."""
    if user_profile.empty or article_attributes.empty:
        return _empty_source_frame()

    user_profile = _with_string_ids(user_profile)
    article_attributes = _with_string_ids(article_attributes)
    article_attributes = _limit_articles_per_attribute(article_attributes, top_n)
    target_users = _with_string_ids(target_users)
    frames: list[pd.DataFrame] = []
    for window in windows.to_dict("records"):
        window_targets = _target_users_for_window(target_users, window)
        if window_targets.empty:
            continue
        profile = _profile_for_window(user_profile, window, window_targets)
        if profile.empty:
            continue
        matched = profile.merge(
            article_attributes.loc[:, ["article_id", "attr_type", "attr_value"]],
            on=["attr_type", "attr_value"],
            how="inner",
        )
        if matched.empty:
            continue
        scored = _rank_similarity_matches(matched, top_n)
        scored.insert(0, "label_week", window["label_week"])
        scored.insert(0, "cutoff_week", window["cutoff_week"])
        scored.insert(0, "split", window["split"])
        frames.append(scored.loc[:, SOURCE_COLUMNS])
    if not frames:
        return _empty_source_frame()
    return pd.concat(frames, ignore_index=True).loc[:, SOURCE_COLUMNS]


def _profile_for_window(
    user_profile: pd.DataFrame,
    window: dict[str, object],
    window_targets: pd.DataFrame,
) -> pd.DataFrame:
    mask = (
        (user_profile["split"] == window["split"])
        & (user_profile["cutoff_week"] == window["cutoff_week"])
        & (user_profile["label_week"] == window["label_week"])
    )
    profile = user_profile.loc[
        mask,
        ["customer_id", "attr_type", "attr_value", "preference_score"],
    ].copy()
    profile = profile.loc[
        profile["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES)
    ].copy()
    profile = profile.merge(window_targets, on="customer_id", how="inner")
    return _limit_profile_attributes(profile)


def _limit_profile_attributes(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return profile
    sorted_profile = profile.sort_values(
        ["customer_id", "preference_score", "attr_type", "attr_value"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return sorted_profile.groupby("customer_id", group_keys=False).head(
        RECOMMENDATION_PROFILE_TOP_ATTRIBUTES
    )


def _limit_articles_per_attribute(
    article_attributes: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    attributes = article_attributes.loc[
        article_attributes["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES),
        ["article_id", "attr_type", "attr_value"],
    ].drop_duplicates()
    sorted_attributes = attributes.sort_values(
        ["attr_type", "attr_value", "article_id"],
        kind="mergesort",
    )
    return sorted_attributes.groupby(
        ["attr_type", "attr_value"],
        group_keys=False,
    ).head(top_n)


def _rank_similarity_matches(matched: pd.DataFrame, top_n: int) -> pd.DataFrame:
    scores = (
        matched.assign(
            preference_score=pd.to_numeric(
                matched["preference_score"],
                errors="raise",
            )
        )
        .groupby(["customer_id", "article_id"], as_index=False)["preference_score"]
        .sum()
        .rename(columns={"preference_score": "similarity_score"})
        .sort_values(
            ["customer_id", "similarity_score", "article_id"],
            ascending=[True, False, True],
        )
    )
    ranked = scores.groupby("customer_id", group_keys=False).head(top_n).copy()
    ranked["source_rank"] = ranked.groupby("customer_id").cumcount() + 1
    ranked["source"] = "similarity"
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


def _empty_source_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SOURCE_COLUMNS)


def _with_string_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("article_id", "customer_id", "attr_type", "attr_value"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result
