from __future__ import annotations

import math

import pytest

from fashion_trend.trend import (ATTRIBUTE_WEEK_TARGET_COLUMNS,
                                 build_attribute_week_target_frame,
                                 validate_attribute_week_target)
from tests.trend_samples import sample_attribute_week_heat


class TestAttributeWeekTargetFrame:
    def test_build_attribute_week_target_frame_calculates_next_week_targets(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())

        assert target.columns.tolist() == list(ATTRIBUTE_WEEK_TARGET_COLUMNS)
        assert len(target) == 6
        assert set(target["week_id"]) == {0}

        black = target[target["attr_id"] == "colour_group_name::Black"].iloc[0]
        assert int(black["heat_t"]) == 2
        assert int(black["heat_t1"]) == 1
        assert math.isclose(float(black["share_t"]), 2 / 3)
        assert math.isclose(float(black["share_t1"]), 1.0)
        assert math.isclose(
            float(black["target_growth"]),
            math.log((1.0 + 1e-6) / ((2 / 3) + 1e-6)),
        )
        assert math.isclose(float(black["target_log_heat_t1"]), math.log1p(1))
        assert int(black["target_rank_in_type_t1"]) == 1

    def test_validate_attribute_week_target_rejects_inconsistent_growth(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[0, "target_growth"] = 999.0

        with pytest.raises(ValueError, match="target_growth"):
            validate_attribute_week_target(target)

    def test_validate_attribute_week_target_rejects_non_positive_epsilon(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())

        with pytest.raises(ValueError, match="epsilon"):
            validate_attribute_week_target(target, epsilon=0)

    def test_validate_attribute_week_target_rejects_non_finite_numeric_values(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target["heat_t"] = target["heat_t"].astype("float64")
        target.loc[0, "heat_t"] = float("inf")

        with pytest.raises(ValueError, match="非有限"):
            validate_attribute_week_target(target)

    def test_validate_attribute_week_target_rejects_inconsistent_log_heat_t1(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[0, "target_log_heat_t1"] = 999.0

        with pytest.raises(ValueError, match="target_log_heat_t1"):
            validate_attribute_week_target(target)

    def test_validate_attribute_week_target_rejects_share_greater_than_one(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[0, "share_t"] = 1.1

        with pytest.raises(ValueError, match="share 大于 1"):
            validate_attribute_week_target(target)

    def test_validate_attribute_week_target_rejects_duplicate_week_attr(
        self,
    ) -> None:
        target = build_attribute_week_target_frame(sample_attribute_week_heat())
        target.loc[len(target)] = target.loc[0]

        with pytest.raises(ValueError, match="week_id, attr_id"):
            validate_attribute_week_target(target)
