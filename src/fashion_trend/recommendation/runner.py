from __future__ import annotations

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic
from fashion_trend.recommendation.contracts import RECOMMENDATION_TOP_K
from fashion_trend.recommendation.methods.base import (
    RecommendationContext,
    RecommendationResult,
)
from fashion_trend.recommendation.outputs import (
    RecommendationResultChunkWriter,
    write_recommendation_result,
)
from fashion_trend.recommendation.registry import get_recommendation_method

WINDOW_COLUMNS = ("split", "cutoff_week", "label_week")


def run_recommendation_method(
    method_name: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    user_profile: pd.DataFrame | None = None,
    trend_predictions: pd.DataFrame | None = None,
    exclude_seen: bool = True,
    weights: dict[str, float] | None = None,
) -> RecommendationResult:
    method = get_recommendation_method(method_name)
    context = RecommendationContext(
        method=method_name,
        top_k=RECOMMENDATION_TOP_K,
        exclude_seen=exclude_seen,
        transactions=transactions,
        article_attributes=article_attributes,
        windows=windows,
        target_users=target_users,
        candidates=candidates,
        user_profile=user_profile,
        trend_predictions=trend_predictions,
        weights=weights,
    )
    result = method.build_recommendations(context)
    write_recommendation_result(result)
    return result


def run_recommendation_method_by_window(
    method_name: str,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    windows: pd.DataFrame,
    target_users: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    user_profile: pd.DataFrame | None = None,
    trend_predictions: pd.DataFrame | None = None,
    exclude_seen: bool = True,
    weights: dict[str, float] | None = None,
    collect_result: bool = True,
) -> RecommendationResult:
    """Run a recommendation method window-by-window and stream large CSV outputs."""
    method = get_recommendation_method(method_name)
    params = _params_for_method(method, method_name, exclude_seen, weights)
    metadata: dict[str, object] = {
        "method": method_name,
        "candidate_rows": 0,
        "recommendation_rows": 0,
        "recommendation_item_rows": 0,
        "window_count": 0,
        "window_summaries": [],
    }
    recommendation_chunks: list[pd.DataFrame] = []
    item_chunks: list[pd.DataFrame] = []

    with RecommendationResultChunkWriter(method_name) as writer:
        for window in windows.loc[:, WINDOW_COLUMNS].to_dict("records"):
            context = RecommendationContext(
                method=method_name,
                top_k=RECOMMENDATION_TOP_K,
                exclude_seen=exclude_seen,
                transactions=transactions,
                article_attributes=article_attributes,
                windows=pd.DataFrame([window], columns=list(WINDOW_COLUMNS)),
                target_users=_frame_for_window(target_users, window),
                candidates=_optional_frame_for_window(candidates, window),
                user_profile=_optional_frame_for_window(user_profile, window),
                trend_predictions=trend_predictions,
                weights=weights,
            )
            result = method.build_recommendations(context)
            writer.write_chunk(result)
            if collect_result:
                recommendation_chunks.append(result.recommendations)
                item_chunks.append(result.recommendation_items)
            _merge_window_metadata(metadata, window, result, context)
            params = result.params
        writer.publish()
        write_json_atomic(params, writer.output_paths.params)
        write_json_atomic(metadata, writer.output_paths.metadata)

    return RecommendationResult(
        recommendations=_concat_chunks(recommendation_chunks),
        recommendation_items=_concat_chunks(item_chunks),
        params=params,
        metadata=metadata,
    )


def _params_for_method(
    method,
    method_name: str,
    exclude_seen: bool,
    weights: dict[str, float] | None,
) -> dict[str, object]:
    return {
        "method": method_name,
        "method_type": method.method_type,
        "top_k": RECOMMENDATION_TOP_K,
        "exclude_seen": exclude_seen,
        "weights": dict(weights if weights is not None else method.default_weights),
    }


def _merge_window_metadata(
    metadata: dict[str, object],
    window: dict[str, object],
    result: RecommendationResult,
    context: RecommendationContext,
) -> None:
    candidate_rows = int(result.metadata.get("candidate_rows", 0))
    summary = {
        **window,
        "target_user_rows": int(len(context.target_users)),
        "candidate_rows": candidate_rows,
        "recommendation_rows": int(len(result.recommendations)),
        "recommendation_item_rows": int(len(result.recommendation_items)),
    }
    metadata["candidate_rows"] = int(metadata["candidate_rows"]) + candidate_rows
    metadata["recommendation_rows"] = int(metadata["recommendation_rows"]) + int(
        len(result.recommendations)
    )
    metadata["recommendation_item_rows"] = int(
        metadata["recommendation_item_rows"]
    ) + int(len(result.recommendation_items))
    metadata["window_count"] = int(metadata["window_count"]) + 1
    metadata["window_summaries"].append(summary)


def _frame_for_window(frame: pd.DataFrame, window: dict[str, object]) -> pd.DataFrame:
    mask = (
        (frame["split"] == window["split"])
        & (frame["cutoff_week"] == window["cutoff_week"])
        & (frame["label_week"] == window["label_week"])
    )
    return frame.loc[mask].reset_index(drop=True)


def _optional_frame_for_window(
    frame: pd.DataFrame | None,
    window: dict[str, object],
) -> pd.DataFrame | None:
    if frame is None:
        return None
    return _frame_for_window(frame, window)


def _concat_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_chunks = [chunk for chunk in chunks if not chunk.empty]
    if not non_empty_chunks:
        return pd.DataFrame()
    return pd.concat(non_empty_chunks, ignore_index=True)
