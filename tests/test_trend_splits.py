from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.foundation.io import write_json_atomic, write_parquet_atomic
from fashion_trend.trend.schema import TREND_MODEL_SPLIT_COLUMNS
from fashion_trend.trend.splits import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    read_trend_model_split,
)
from tests.trend_samples import sample_trend_model_samples_for_split


class TestTrendModelSplitFrame:
    def test_build_trend_model_split_frames_uses_time_boundaries(self) -> None:
        samples = sample_trend_model_samples_for_split()

        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )

        assert set(split_frames) == {"train", "valid", "test"}
        assert split_frames["train"]["week_id"].min() == 4
        assert split_frames["train"]["week_id"].max() == 15
        assert split_frames["valid"]["week_id"].min() == 16
        assert split_frames["valid"]["week_id"].max() == 19
        assert split_frames["test"]["week_id"].min() == 20
        assert split_frames["test"]["week_id"].max() == 23
        assert set(split_frames["train"]["split"]) == {"train"}
        assert set(split_frames["valid"]["split"]) == {"valid"}
        assert set(split_frames["test"]["split"]) == {"test"}

    def test_build_trend_model_split_frames_rejects_too_few_weeks(self) -> None:
        samples = sample_trend_model_samples_for_split()
        samples = samples[samples["week_id"] < 10].copy()

        with pytest.raises(ValueError, match="样本周数不足"):
            build_trend_model_split_frames(samples, valid_weeks=4, test_weeks=4)

    def test_build_trend_model_split_metadata_reports_ranges(self) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )

        metadata = build_trend_model_split_metadata(
            split_frames,
            input_path=Path("data/processed/features/trend_model_samples.parquet"),
            output_paths={
                "train": Path(
                    "data/processed/features/trend_model_samples_train.parquet"
                ),
                "valid": Path(
                    "data/processed/features/trend_model_samples_valid.parquet"
                ),
                "test": Path(
                    "data/processed/features/trend_model_samples_test.parquet"
                ),
            },
            valid_weeks=4,
            test_weeks=4,
        )

        assert metadata["split_strategy"] == "time"
        assert metadata["valid_weeks"] == 4
        assert metadata["test_weeks"] == 4
        assert metadata["splits"]["train"]["week_min"] == 4
        assert metadata["splits"]["train"]["week_max"] == 15
        assert metadata["splits"]["train"]["rows"] == 24
        assert metadata["splits"]["valid"]["week_min"] == 16
        assert metadata["splits"]["test"]["week_max"] == 23

    def test_read_trend_model_split_preserves_columns_for_legal_parquet(
        self, tmp_path: Path
    ) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )
        input_path = tmp_path / "trend_model_samples_train.parquet"
        write_parquet_atomic(split_frames["train"], input_path)

        split = read_trend_model_split(input_path)

        assert split.columns.tolist() == list(TREND_MODEL_SPLIT_COLUMNS)
        assert set(split["split"]) == {"train"}

    def test_read_trend_model_split_rejects_invalid_split_value(
        self, tmp_path: Path
    ) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )
        invalid_split = split_frames["train"].copy()
        invalid_split.loc[invalid_split.index[0], "split"] = "holdout"
        input_path = tmp_path / "trend_model_samples_train.parquet"
        write_parquet_atomic(invalid_split, input_path)

        with pytest.raises(ValueError, match="非法 split"):
            read_trend_model_split(input_path)

    def test_read_trend_model_split_rejects_duplicate_week_attr(
        self, tmp_path: Path
    ) -> None:
        samples = sample_trend_model_samples_for_split()
        split_frames = build_trend_model_split_frames(
            samples,
            valid_weeks=4,
            test_weeks=4,
        )
        duplicate_split = pd.concat(
            [split_frames["train"], split_frames["train"].iloc[[0]]],
            ignore_index=True,
        )
        input_path = tmp_path / "trend_model_samples_train.parquet"
        write_parquet_atomic(duplicate_split, input_path)

        with pytest.raises(ValueError, match="week_id, attr_id"):
            read_trend_model_split(input_path)


class TestTrendModelSplitWrite:
    def test_write_json_atomic_creates_parent_and_writes_sorted_keys(
        self, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "nested" / "metadata.json"

        write_json_atomic({"b": 1, "a": 2}, output_path)

        assert output_path.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}\n'
