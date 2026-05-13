from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.presentation.extractors import (
    _read_recommendation_metric_payloads,
    _read_report_tables,
    _read_trend_metric_payloads,
    filter_frame_to_case_keys,
)


def test_read_report_tables_fails_for_empty_directory(tmp_path: Path) -> None:
    table_dir = tmp_path / "tables"
    table_dir.mkdir()

    with pytest.raises(ValueError, match="report tables.*tables"):
        _read_report_tables(table_dir)


def test_read_trend_metric_payloads_fails_for_empty_directory(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    with pytest.raises(ValueError, match="trend metrics.*metrics"):
        _read_trend_metric_payloads(metrics_dir)


def test_read_recommendation_metric_payloads_fails_for_empty_directory(
    tmp_path: Path,
) -> None:
    recommendation_dir = tmp_path / "recommendation"
    recommendation_dir.mkdir()

    with pytest.raises(ValueError, match="recommendation metrics.*recommendation"):
        _read_recommendation_metric_payloads(recommendation_dir)


def test_filter_frame_to_case_keys_keeps_target_top12_and_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "customer_id": "000000abcdef123456",
                "split": "test",
                "cutoff_week": 10,
                "label_week": 11,
                "article_id": f"00000000{rank:02d}",
                "rank": rank,
                "extra": "drop-me",
            }
            for rank in range(1, 14)
        ]
        + [
            {
                "customer_id": "other-user",
                "split": "test",
                "cutoff_week": 10,
                "label_week": 11,
                "article_id": "9999999999",
                "rank": 1,
                "extra": "drop-me",
            }
        ]
    )

    result = filter_frame_to_case_keys(
        frame,
        [
            {
                "customer_id": "000000abcdef123456",
                "split": "test",
                "cutoff_week": 10,
                "label_week": 11,
            }
        ],
        columns=[
            "customer_id",
            "split",
            "cutoff_week",
            "label_week",
            "article_id",
            "rank",
        ],
        top_k=12,
    )

    assert result["customer_id"].unique().tolist() == ["000000abcdef123456"]
    assert result["rank"].tolist() == list(range(1, 13))
    assert result.columns.tolist() == [
        "customer_id",
        "split",
        "cutoff_week",
        "label_week",
        "article_id",
        "rank",
    ]
