from __future__ import annotations

import pandas as pd

from fashion_trend.foundation.io import write_csv_atomic, write_json_atomic
from fashion_trend.recommendation.contracts import RECOMMENDATIONS_COLUMNS
from fashion_trend.recommendation.methods.base import RecommendationResult
from fashion_trend.recommendation.paths import method_output_paths


def build_recommendations_csv(
    recommendation_items: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    if recommendation_items.empty:
        return pd.DataFrame(columns=RECOMMENDATIONS_COLUMNS)
    ranked = recommendation_items.loc[
        recommendation_items["rank"] <= top_k
    ].sort_values(["customer_id", "split", "cutoff_week", "label_week", "rank"])
    predictions = ranked.groupby(
        ["customer_id", "split", "cutoff_week", "label_week", "method"],
        sort=False,
    )["article_id"].apply(lambda values: " ".join(values.astype(str)))
    return predictions.reset_index(name="prediction").loc[
        :,
        list(RECOMMENDATIONS_COLUMNS),
    ]


def write_recommendation_result(result: RecommendationResult) -> None:
    output_paths = method_output_paths(str(result.params["method"]))
    write_csv_atomic(result.recommendations, output_paths.recommendations)
    write_csv_atomic(result.recommendation_items, output_paths.recommendation_items)
    write_json_atomic(result.params, output_paths.params)
    write_json_atomic(result.metadata, output_paths.metadata)
