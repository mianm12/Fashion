from __future__ import annotations

import csv

import pandas as pd

from fashion_trend.foundation.io import (
    remove_file_if_exists,
    write_csv_atomic,
    write_json_atomic,
)
from fashion_trend.recommendation.contracts import (
    RECOMMENDATION_ITEMS_COLUMNS,
    RECOMMENDATIONS_COLUMNS,
)
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


class RecommendationResultChunkWriter:
    """Write large recommendation results through temp CSV chunks."""

    def __init__(self, method: str) -> None:
        self.output_paths = method_output_paths(method)
        self.recommendations_tmp = self.output_paths.recommendations.with_suffix(
            ".csv.tmp"
        )
        self.items_tmp = self.output_paths.recommendation_items.with_suffix(".csv.tmp")
        self._started = False

    def __enter__(self) -> "RecommendationResultChunkWriter":
        self.output_paths.output_dir.mkdir(parents=True, exist_ok=True)
        remove_file_if_exists(self.recommendations_tmp)
        remove_file_if_exists(self.items_tmp)
        _write_csv_header(self.recommendations_tmp, RECOMMENDATIONS_COLUMNS)
        _write_csv_header(self.items_tmp, RECOMMENDATION_ITEMS_COLUMNS)
        self._started = True
        return self

    def write_chunk(self, result: RecommendationResult) -> None:
        if not self._started:
            raise RuntimeError("chunk writer has not been opened")
        _append_csv_rows(result.recommendations, self.recommendations_tmp)
        _append_csv_rows(result.recommendation_items, self.items_tmp)

    def publish(self) -> None:
        if not self._started:
            raise RuntimeError("chunk writer has not been opened")
        self.recommendations_tmp.replace(self.output_paths.recommendations)
        self.items_tmp.replace(self.output_paths.recommendation_items)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            remove_file_if_exists(self.recommendations_tmp)
            remove_file_if_exists(self.items_tmp)


def _write_csv_header(path, columns: tuple[str, ...]) -> None:
    pd.DataFrame(columns=list(columns)).to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
    )


def _append_csv_rows(dataframe: pd.DataFrame, path) -> None:
    if dataframe.empty:
        return
    dataframe.to_csv(
        path,
        mode="a",
        header=False,
        index=False,
        quoting=csv.QUOTE_ALL,
    )
