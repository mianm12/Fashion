from __future__ import annotations

import numpy as np
import pandas as pd

from fashion_trend.recommendation.ranking.weights import validate_score_weights


RANK_GROUP_COLUMNS = ["customer_id", "split", "cutoff_week", "label_week"]


def rank_candidate_items(
    feature_frame: pd.DataFrame,
    weights: dict[str, float],
    top_k: int,
) -> pd.DataFrame:
    """Score and rank candidates with deterministic tie-breaking."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    validated_weights = validate_score_weights(weights, tuple(weights))
    missing_features = set(validated_weights) - set(feature_frame.columns)
    if missing_features:
        raise ValueError(f"missing ranking feature columns: {sorted(missing_features)}")

    result = feature_frame.copy()
    result["score"] = 0.0
    for feature, weight in validated_weights.items():
        result["score"] = result["score"] + (
            pd.to_numeric(result[feature], errors="raise") * weight
        )

    score_values = result["score"].to_numpy(dtype=float)
    if not np.isfinite(score_values).all():
        raise ValueError("score contains non-finite values")
    if ((score_values < 0.0) | (score_values > 1.0)).any():
        raise ValueError("score must be within [0, 1]")

    result = result.sort_values(
        [*RANK_GROUP_COLUMNS, "score", "article_id"],
        ascending=[True, True, True, True, False, True],
    )
    result["rank"] = result.groupby(RANK_GROUP_COLUMNS).cumcount() + 1
    return result.loc[result["rank"] <= top_k].reset_index(drop=True)
