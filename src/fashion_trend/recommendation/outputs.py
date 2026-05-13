from __future__ import annotations

import csv

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fashion_trend.foundation.io import (
    remove_file_if_exists,
    write_csv_atomic,
    write_json_atomic,
)
from fashion_trend.recommendation import paths as recommendation_paths
from fashion_trend.recommendation.contracts import (
    ENHANCED_RECOMMENDATION_SCORE_COLUMNS,
    RECOMMENDATION_ITEMS_COLUMNS,
    RECOMMENDATIONS_COLUMNS,
)
from fashion_trend.recommendation.methods.base import RecommendationResult

_RECOMMENDATION_ITEMS_ARROW_SCHEMA = pa.schema(
    [
        ("customer_id", pa.string()),
        ("split", pa.string()),
        ("cutoff_week", pa.int64()),
        ("label_week", pa.int64()),
        ("method", pa.string()),
        ("article_id", pa.string()),
        ("rank", pa.int64()),
        ("score", pa.float64()),
        ("pop_score", pa.float64()),
        ("sim_score", pa.float64()),
        ("trend_score", pa.float64()),
        ("recent_score", pa.float64()),
        ("reorder_score", pa.float64()),
        ("variant_score", pa.float64()),
        ("age_pop_score", pa.float64()),
        ("preference_pop_score", pa.float64()),
        ("source_rank_score", pa.float64()),
        ("source_count_score", pa.float64()),
        ("candidate_sources", pa.string()),
    ]
)


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


def format_recommendation_items(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame(columns=RECOMMENDATION_ITEMS_COLUMNS)
    result = _ensure_recommendation_item_columns(ranked)
    for column in ("customer_id", "article_id", "method", "candidate_sources"):
        result[column] = result[column].astype("string")
    return result.loc[:, list(RECOMMENDATION_ITEMS_COLUMNS)].reset_index(drop=True)


def write_recommendation_result(result: RecommendationResult) -> None:
    output_paths = recommendation_paths.method_output_paths(
        str(result.params["method"])
    )
    write_csv_atomic(result.recommendations, output_paths.recommendations)
    _write_recommendation_items_parquet_atomic(
        _normalize_recommendation_items(result.recommendation_items),
        output_paths.recommendation_items,
    )
    write_json_atomic(result.params, output_paths.params)
    write_json_atomic(result.metadata, output_paths.metadata)


class RecommendationResultChunkWriter:
    """Write large recommendation results through streaming temp artifacts."""

    def __init__(self, method: str) -> None:
        self.output_paths = recommendation_paths.method_output_paths(method)
        self.recommendations_tmp = self.output_paths.recommendations.with_suffix(
            ".csv.tmp"
        )
        self.items_tmp = self.output_paths.recommendation_items.with_suffix(
            ".parquet.tmp"
        )
        self._items_writer: pq.ParquetWriter | None = None
        self._started = False

    def __enter__(self) -> "RecommendationResultChunkWriter":
        self.output_paths.output_dir.mkdir(parents=True, exist_ok=True)
        remove_file_if_exists(self.recommendations_tmp)
        remove_file_if_exists(self.items_tmp)
        _write_csv_header(self.recommendations_tmp, RECOMMENDATIONS_COLUMNS)
        self._started = True
        return self

    def write_chunk(self, result: RecommendationResult) -> None:
        if not self._started:
            raise RuntimeError("chunk writer has not been opened")
        _append_csv_rows(result.recommendations, self.recommendations_tmp)
        self._append_item_rows(result.recommendation_items)

    def publish(self) -> None:
        if not self._started:
            raise RuntimeError("chunk writer has not been opened")
        self._close_items_writer()
        if not self.items_tmp.exists():
            _write_empty_items_parquet(self.items_tmp)
        self.recommendations_tmp.replace(self.output_paths.recommendations)
        self.items_tmp.replace(self.output_paths.recommendation_items)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._close_items_writer()
        if exc_type is not None:
            remove_file_if_exists(self.recommendations_tmp)
            remove_file_if_exists(self.items_tmp)

    def _append_item_rows(self, dataframe: pd.DataFrame) -> None:
        dataframe = _normalize_recommendation_items(dataframe)
        if dataframe.empty:
            return
        table = pa.Table.from_pandas(
            dataframe,
            schema=_RECOMMENDATION_ITEMS_ARROW_SCHEMA,
            preserve_index=False,
        )
        if self._items_writer is None:
            self._items_writer = pq.ParquetWriter(
                self.items_tmp,
                _RECOMMENDATION_ITEMS_ARROW_SCHEMA,
            )
        self._items_writer.write_table(table)

    def _close_items_writer(self) -> None:
        if self._items_writer is not None:
            self._items_writer.close()
            self._items_writer = None


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


def _normalize_recommendation_items(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = _ensure_recommendation_item_columns(dataframe)
    return normalized.loc[:, list(RECOMMENDATION_ITEMS_COLUMNS)]


def _ensure_recommendation_item_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    fillable_columns = set(ENHANCED_RECOMMENDATION_SCORE_COLUMNS[4:])
    missing_columns = [
        column
        for column in RECOMMENDATION_ITEMS_COLUMNS
        if column not in result.columns
    ]
    unfillable_columns = [
        column for column in missing_columns if column not in fillable_columns
    ]
    if unfillable_columns:
        raise ValueError(
            "recommendation_items 列契约不匹配: "
            f"expected={RECOMMENDATION_ITEMS_COLUMNS}, "
            f"actual={tuple(dataframe.columns)}"
        )
    for column in missing_columns:
        result[column] = 0.0
    return result


def _write_recommendation_items_parquet_atomic(
    dataframe: pd.DataFrame,
    output_path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        table = pa.Table.from_pandas(
            dataframe,
            schema=_RECOMMENDATION_ITEMS_ARROW_SCHEMA,
            preserve_index=False,
        )
        with pq.ParquetWriter(tmp_path, _RECOMMENDATION_ITEMS_ARROW_SCHEMA) as writer:
            writer.write_table(table)
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)


def _write_empty_items_parquet(path) -> None:
    table = pa.Table.from_pandas(
        pd.DataFrame(columns=list(RECOMMENDATION_ITEMS_COLUMNS)),
        schema=_RECOMMENDATION_ITEMS_ARROW_SCHEMA,
        preserve_index=False,
    )
    with pq.ParquetWriter(path, _RECOMMENDATION_ITEMS_ARROW_SCHEMA) as writer:
        writer.write_table(table)
