from __future__ import annotations

import pandas as pd

from fashion_trend.recommendation.contracts import RECOMMENDATION_CORE_ATTR_TYPES
from fashion_trend.recommendation.retrieval.popularity import SOURCE_COLUMNS


def build_trend_candidates(
    predictions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return trend candidates using predictions at each cutoff week."""
    if predictions.empty or article_attributes.empty:
        return _empty_source_frame()

    predictions = _with_string_ids(predictions)
    article_attributes = _with_string_ids(article_attributes)
    target_users = _with_string_ids(target_users)
    frames: list[pd.DataFrame] = []
    for window in windows.to_dict("records"):
        window_predictions = _predictions_for_window(predictions, window)
        if window_predictions.empty:
            continue
        ranked_articles = _rank_trend_articles(
            window_predictions,
            article_attributes,
            top_n,
        )
        window_targets = _target_users_for_window(target_users, window)
        if ranked_articles.empty or window_targets.empty:
            continue
        frame = window_targets.merge(ranked_articles, how="cross")
        frame.insert(0, "label_week", window["label_week"])
        frame.insert(0, "cutoff_week", window["cutoff_week"])
        frame.insert(0, "split", window["split"])
        frames.append(frame.loc[:, SOURCE_COLUMNS])
    if not frames:
        return _empty_source_frame()
    return pd.concat(frames, ignore_index=True).loc[:, SOURCE_COLUMNS]


def _predictions_for_window(
    predictions: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    week_id = pd.to_numeric(predictions["week_id"], errors="raise")
    mask = (
        (predictions["split"] == window["split"])
        & (week_id == int(window["cutoff_week"]))
        & predictions["attr_type"].isin(RECOMMENDATION_CORE_ATTR_TYPES)
    )
    return predictions.loc[
        mask,
        ["attr_type", "attr_value", "pred_target_growth"],
    ].copy()


def _rank_trend_articles(
    predictions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    matched = article_attributes.merge(
        predictions,
        on=["attr_type", "attr_value"],
        how="inner",
    )
    if matched.empty:
        return pd.DataFrame(columns=["article_id", "source", "source_rank"])
    scored = (
        matched.assign(
            pred_target_growth=pd.to_numeric(
                matched["pred_target_growth"],
                errors="raise",
            )
        )
        .groupby("article_id", as_index=False)["pred_target_growth"]
        .max()
        .rename(columns={"pred_target_growth": "trend_score"})
        .sort_values(["trend_score", "article_id"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    scored["source"] = "trend"
    scored["source_rank"] = scored.index + 1
    return scored.loc[:, ["article_id", "source", "source_rank"]]


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
    for column in ("article_id", "customer_id", "attr_type", "attr_value", "split"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result
