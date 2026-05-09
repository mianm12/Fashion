from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fashion_trend.recommendation.contracts import (
    RECOMMENDATION_CORE_ATTR_TYPES,
    RECOMMENDATION_TREND_ATTR_WEIGHTS,
)

WINDOW_COLUMNS = ["split", "cutoff_week", "label_week"]
SCORE_COLUMNS = ["pop_score", "recent_score", "sim_score", "trend_score"]


def minmax_normalize_by_group(
    dataframe: pd.DataFrame,
    value_column: str,
    output_column: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Normalize a numeric column within groups; constant groups become 0.0."""
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


def build_ranking_features(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build bounded ranking features for candidate items."""
    feature_frame = candidates.copy()
    feature_frame = add_pop_score(feature_frame, transactions)
    feature_frame = add_recent_score(feature_frame, transactions)
    feature_frame = add_sim_score(feature_frame, article_attributes, user_profile)
    feature_frame = add_trend_score(
        feature_frame,
        article_attributes,
        trend_predictions,
    )
    _validate_score_bounds(feature_frame, SCORE_COLUMNS)
    return feature_frame


def add_pop_score(candidates: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative popularity up to cutoff week."""
    return _add_article_count_score(
        candidates,
        transactions,
        score_column="pop_score",
        recent_weeks=None,
    )


def add_recent_score(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Add popularity from the last four weeks up to cutoff week."""
    return _add_article_count_score(
        candidates,
        transactions,
        score_column="recent_score",
        recent_weeks=4,
    )


def add_sim_score(
    candidates: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add customer-article similarity from profile attribute preferences."""
    result = _with_string_ids(candidates)
    if user_profile is None or user_profile.empty or article_attributes.empty:
        result["sim_score"] = 0.0
        return result

    raw_scores = _build_similarity_scores(
        result,
        _with_string_ids(article_attributes),
        _with_string_ids(user_profile),
    )
    result = result.merge(
        raw_scores,
        on=[*WINDOW_COLUMNS, "customer_id", "article_id"],
        how="left",
    )
    result["sim_value"] = result["sim_value"].fillna(0.0)
    result = minmax_normalize_by_group(
        result,
        value_column="sim_value",
        output_column="sim_score",
        group_columns=(*WINDOW_COLUMNS, "customer_id"),
    )
    return result.drop(columns=["sim_value"])


def add_trend_score(
    candidates: pd.DataFrame,
    article_attributes: pd.DataFrame,
    trend_predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add article trend scores from cutoff-week attribute predictions."""
    result = _with_string_ids(candidates)
    if trend_predictions is None or trend_predictions.empty or article_attributes.empty:
        result["trend_score"] = 0.0
        return result

    windows = result.loc[:, WINDOW_COLUMNS].drop_duplicates().reset_index(drop=True)
    article_scores = build_article_trend_scores(
        trend_predictions,
        article_attributes,
        windows,
    )

    result = result.merge(
        article_scores,
        on=[*WINDOW_COLUMNS, "article_id"],
        how="left",
    )
    result["trend_score"] = result["trend_score"].fillna(0.0)
    return result


def build_article_trend_scores(
    predictions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Map cutoff-week attribute trend predictions to normalized article scores."""
    _require_columns(
        predictions,
        ("split", "week_id", "attr_type", "attr_value", "pred_target_growth"),
        "trend predictions",
    )
    _require_columns(
        article_attributes,
        ("article_id", "attr_type", "attr_value"),
        "article attributes",
    )
    _require_columns(windows, WINDOW_COLUMNS, "windows")

    output_columns = [*WINDOW_COLUMNS, "article_id", "trend_score"]
    if predictions.empty or article_attributes.empty or windows.empty:
        return pd.DataFrame(columns=output_columns)

    prediction_frame = _with_attr_join_ids(predictions)
    attribute_frame = _with_attr_join_ids(_with_string_ids(article_attributes))
    window_frame = windows.loc[:, WINDOW_COLUMNS].drop_duplicates().copy()

    prediction_frame = prediction_frame.loc[
        prediction_frame["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES)
    ].copy()
    if prediction_frame.empty:
        return pd.DataFrame(columns=output_columns)

    window_predictions = window_frame.merge(
        prediction_frame,
        left_on=["split", "cutoff_week"],
        right_on=["split", "week_id"],
        how="left",
    )
    window_predictions = window_predictions.dropna(subset=["pred_target_growth"])
    if window_predictions.empty:
        return pd.DataFrame(columns=output_columns)

    window_predictions = minmax_normalize_by_group(
        window_predictions,
        value_column="pred_target_growth",
        output_column="attr_trend_score",
        group_columns=["split", "cutoff_week", "attr_type"],
    )
    join_columns = _attribute_join_columns(window_predictions, attribute_frame)
    matched = attribute_frame.merge(
        window_predictions.loc[
            :,
            [*WINDOW_COLUMNS, *join_columns, "attr_trend_score"],
        ],
        on=join_columns,
        how="inner",
    )
    if matched.empty:
        return pd.DataFrame(columns=output_columns)

    matched["attr_weight"] = matched["attr_type"].map(RECOMMENDATION_TREND_ATTR_WEIGHTS)
    matched["weighted_score"] = matched["attr_trend_score"] * matched["attr_weight"]
    scores = matched.groupby([*WINDOW_COLUMNS, "article_id"], as_index=False).agg(
        weighted_score=("weighted_score", "sum"),
        matched_weight=("attr_weight", "sum"),
    )
    scores["trend_score"] = np.where(
        scores["matched_weight"] > 0.0,
        scores["weighted_score"] / scores["matched_weight"],
        0.0,
    )
    scores["trend_score"] = scores["trend_score"].clip(lower=0.0, upper=1.0)
    if not np.isfinite(scores["trend_score"].to_numpy(dtype=float)).all():
        raise ValueError("trend_score contains non-finite values")
    scores["article_id"] = scores["article_id"].astype(str)
    return scores.loc[:, output_columns]


def _add_article_count_score(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    score_column: str,
    recent_weeks: int | None,
) -> pd.DataFrame:
    result = _with_string_ids(candidates)
    if result.empty or transactions.empty:
        result[score_column] = 0.0
        return result

    transactions = _with_string_ids(transactions)
    frames: list[pd.DataFrame] = []
    for window in result[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        window_candidates = _frame_for_window(result, window)
        window_transactions = _transactions_for_window(
            transactions,
            cutoff_week=int(window["cutoff_week"]),
            recent_weeks=recent_weeks,
        )
        counts = (
            window_transactions.groupby("article_id", as_index=False)
            .size()
            .rename(columns={"size": "_count"})
        )
        scored = window_candidates.merge(counts, on="article_id", how="left")
        scored["_count"] = scored["_count"].fillna(0.0)
        frames.append(scored)

    merged = pd.concat(frames, ignore_index=True) if frames else result.copy()
    merged = minmax_normalize_by_group(
        merged,
        value_column="_count",
        output_column=score_column,
        group_columns=WINDOW_COLUMNS,
    )
    return merged.drop(columns=["_count"])


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


def _build_similarity_scores(
    candidates: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window in candidates[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        window_candidates = _frame_for_window(candidates, window)
        profile = _frame_for_window(user_profile, window)
        if profile.empty:
            continue
        candidate_pairs = window_candidates.loc[
            :,
            ["customer_id", "article_id"],
        ].drop_duplicates()
        candidate_attributes = candidate_pairs.merge(
            article_attributes.loc[
                article_attributes["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES),
                ["article_id", "attr_type", "attr_value"],
            ].drop_duplicates(),
            on="article_id",
            how="inner",
        )
        matched = candidate_attributes.merge(
            profile.loc[
                profile["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES),
                ["customer_id", "attr_type", "attr_value", "preference_score"],
            ],
            on=["customer_id", "attr_type", "attr_value"],
            how="inner",
        )
        if matched.empty:
            continue
        scored = (
            matched.assign(
                preference_score=pd.to_numeric(
                    matched["preference_score"],
                    errors="raise",
                )
            )
            .groupby(["customer_id", "article_id"], as_index=False)["preference_score"]
            .sum()
            .rename(columns={"preference_score": "sim_value"})
        )
        for column in WINDOW_COLUMNS:
            scored[column] = window[column]
        frames.append(
            scored.loc[
                :,
                [*WINDOW_COLUMNS, "customer_id", "article_id", "sim_value"],
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[*WINDOW_COLUMNS, "customer_id", "article_id", "sim_value"]
        )
    return pd.concat(frames, ignore_index=True)


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _attribute_join_columns(
    predictions: pd.DataFrame,
    article_attributes: pd.DataFrame,
) -> list[str]:
    columns = ["attr_type", "attr_value"]
    if "attr_id" in predictions.columns and "attr_id" in article_attributes.columns:
        return ["attr_id", *columns]
    return columns


def _with_attr_join_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    if "attr_id" in result.columns:
        result["attr_id"] = result["attr_id"].astype(str)
    for column in ("attr_type", "attr_value"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result


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


def _validate_score_bounds(frame: pd.DataFrame, score_columns: Sequence[str]) -> None:
    for column in score_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{column} must be within [0, 1]")


def _with_string_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("article_id", "customer_id", "attr_type", "attr_value"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result
