from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.trend.models.base import MODEL_TYPE_SUPERVISED


LIGHTGBM_MODULE = "fashion_trend.trend.models.supervised.lightgbm"
LIGHTGBM_SOURCE = Path("src/fashion_trend/trend/models/supervised/lightgbm.py")


class TestLightGBMTrendModel:
    def test_lightgbm_module_does_not_import_native_package_at_top_level(self) -> None:
        source = LIGHTGBM_SOURCE.read_text(encoding="utf-8")
        top_level_source = source.split("def _fit_lightgbm_model", maxsplit=1)[0]

        assert "import lightgbm" not in top_level_source
        assert "from lightgbm" not in top_level_source

    def test_lightgbm_constants_are_stable(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)

        assert lightgbm_model.LIGHTGBM_MODEL_NAME == "lightgbm"
        assert lightgbm_model.LIGHTGBM_TARGET_COLUMN == "target_growth"
        assert lightgbm_model.LIGHTGBM_EPSILON == 1e-6
        assert lightgbm_model.LIGHTGBM_CATEGORICAL_FEATURES == ("attr_type",)
        assert lightgbm_model.LIGHTGBM_ALLOWED_OBJECTIVES == (
            "regression",
            "regression_l1",
        )
        assert lightgbm_model.LIGHTGBM_PARAMS == {
            "objective": "regression",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
        }
        assert lightgbm_model.LIGHTGBM_EARLY_STOPPING == {"stopping_rounds": 30}
        assert lightgbm_model.LIGHTGBM_NUMERIC_FEATURES == (
            "heat_t",
            "share_t",
            "log_heat_t",
            "rank_in_type_t",
            "heat_lag_1",
            "heat_lag_2",
            "heat_lag_3",
            "heat_lag_4",
            "share_lag_1",
            "share_lag_2",
            "share_lag_3",
            "share_lag_4",
            "growth_lag_1",
            "growth_lag_2",
            "acc_lag_1",
            "heat_ma_4",
            "share_ma_4",
            "share_std_4",
            "share_max_4",
            "share_min_4",
            "article_count",
            "is_core_attr",
            "parent_count",
            "child_count",
            "degree",
            "history_total_heat_t",
            "history_active_weeks_t",
            "is_trend_eligible_t",
            "week_index",
            "week_mod_52",
        )
        assert lightgbm_model.LIGHTGBM_EXCLUDED_COLUMNS == (
            "attr_id",
            "attr_value",
            "target_growth",
            "target_log_heat_t1",
            "target_rank_in_type_t1",
            "split",
        )

    def test_lightgbm_trainer_metadata(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        trainer = lightgbm_model.LightGBMTrendTrainer()

        assert trainer.name == "lightgbm"
        assert trainer.model_type == MODEL_TYPE_SUPERVISED
