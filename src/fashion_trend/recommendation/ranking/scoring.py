from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fashion_trend.recommendation.contracts import ENHANCED_RECOMMENDATION_SCORE_COLUMNS
from fashion_trend.recommendation.ranking.weights import validate_score_weights

RANK_GROUP_COLUMNS = ["customer_id", "split", "cutoff_week", "label_week"]
_RANK_BASE_COLUMNS = [*RANK_GROUP_COLUMNS, "method", "article_id", "candidate_sources"]


def rank_candidate_items(
    feature_frame: pd.DataFrame,
    weights: dict[str, float],
    top_k: int,
    required_features: Sequence[str],
) -> pd.DataFrame:
    """Score and rank candidates with deterministic tie-breaking."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    validated_weights = validate_score_weights(weights, required_features)
    missing_features = set(validated_weights) - set(feature_frame.columns)
    if missing_features:
        raise ValueError(f"missing ranking feature columns: {sorted(missing_features)}")

    score_columns = [
        column
        for column in ENHANCED_RECOMMENDATION_SCORE_COLUMNS
        if column in feature_frame.columns
    ]
    result_columns = list(
        dict.fromkeys([*_RANK_BASE_COLUMNS, *score_columns, *validated_weights])
    )
    missing_columns = set(result_columns) - set(feature_frame.columns)
    if missing_columns:
        raise ValueError(f"missing ranking columns: {sorted(missing_columns)}")

    result = feature_frame.loc[:, result_columns].copy()
    score_values = np.zeros(len(result), dtype=float)
    for feature, weight in validated_weights.items():
        feature_values = pd.to_numeric(result[feature], errors="raise").to_numpy(
            dtype=float,
        )
        score_values = score_values + feature_values * weight

    if not np.isfinite(score_values).all():
        raise ValueError("score contains non-finite values")
    if ((score_values < 0.0) | (score_values > 1.0)).any():
        raise ValueError("score must be within [0, 1]")
    result["score"] = score_values

    result = result.sort_values(
        [*RANK_GROUP_COLUMNS, "score", "article_id"],
        ascending=[True, True, True, True, False, True],
    )
    result["rank"] = result.groupby(RANK_GROUP_COLUMNS).cumcount() + 1
    return result.loc[result["rank"] <= top_k].reset_index(drop=True)
