from __future__ import annotations

import importlib
from pathlib import Path

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

    def test_prepare_feature_frame_uses_train_categories(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="train")

        prepared = lightgbm_model.prepare_lightgbm_feature_frame(samples)

        assert prepared.attr_type_categories == ("colour_group_name",)
        assert prepared.features.columns.tolist() == [
            *lightgbm_model.LIGHTGBM_NUMERIC_FEATURES,
            *lightgbm_model.LIGHTGBM_CATEGORICAL_FEATURES,
        ]
        assert str(prepared.features["attr_type"].dtype) == "category"
        assert list(prepared.features["attr_type"].cat.categories) == [
            "colour_group_name"
        ]

    def test_prepare_feature_frame_reuses_train_categories(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="valid")

        prepared = lightgbm_model.prepare_lightgbm_feature_frame(
            samples,
            attr_type_categories=("colour_group_name", "product_type_name"),
        )

        assert prepared.attr_type_categories == (
            "colour_group_name",
            "product_type_name",
        )
        assert list(prepared.features["attr_type"].cat.categories) == [
            "colour_group_name",
            "product_type_name",
        ]

    def test_prepare_feature_frame_rejects_unknown_attr_type(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="valid")

        with pytest.raises(ValueError, match="未知 attr_type"):
            lightgbm_model.prepare_lightgbm_feature_frame(
                samples,
                attr_type_categories=("product_type_name",),
            )

    def test_prepare_feature_frame_rejects_non_finite_numeric_feature(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="train")
        samples.loc[samples.index[0], "growth_lag_1"] = float("nan")

        with pytest.raises(ValueError, match="非有限|growth_lag_1"):
            lightgbm_model.prepare_lightgbm_feature_frame(samples)

    def test_prepare_feature_frame_rejects_missing_attr_type(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="train")
        samples.loc[samples.index[0], "attr_type"] = None

        with pytest.raises(ValueError, match="attr_type"):
            lightgbm_model.prepare_lightgbm_feature_frame(samples)

    def test_describe_target_distribution_returns_split_metric_mapping(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().assign(split="train")

        distribution = lightgbm_model.describe_target_distribution({"train": samples})

        assert set(distribution) == {"train"}
        assert set(distribution["train"]) == {
            "count",
            "min",
            "max",
            "mean",
            "std",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
            "abs_gt_2",
        }
        assert distribution["train"]["count"] == len(samples)

    def test_describe_target_distribution_rejects_empty_split(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from tests.trend_samples import sample_trend_model_samples_for_split

        samples = sample_trend_model_samples_for_split().head(0).assign(split="train")

        with pytest.raises(ValueError, match="lightgbm.*空"):
            lightgbm_model.describe_target_distribution({"train": samples})

    def test_describe_residual_distribution_returns_valid_test_only(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS

        samples = _sample_lightgbm_samples("valid")
        predictions = samples.loc[
            :,
            [
                "week_id",
                "attr_id",
                "attr_type",
                "attr_value",
                "split",
                "share_t",
                "target_growth",
                "target_rank_in_type_t1",
            ],
        ].copy()
        predictions.insert(4, "model_name", "lightgbm")
        predictions["pred_share_t1"] = [0.6, 0.4, 0.6, 0.4]
        predictions["pred_target_growth"] = [0.1, -0.1, 0.2, -0.2]
        predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]

        distribution = lightgbm_model.describe_residual_distribution(
            {"valid": predictions}
        )

        assert set(distribution) == {"valid"}
        assert set(distribution["valid"]) == {
            "count",
            "min",
            "max",
            "mean",
            "std",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
            "mae",
            "rmse",
        }

    def test_describe_residual_distribution_rejects_empty_predictions(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS

        samples = _sample_lightgbm_samples("valid").head(0)
        predictions = samples.loc[
            :,
            [
                "week_id",
                "attr_id",
                "attr_type",
                "attr_value",
                "split",
                "share_t",
                "target_growth",
                "target_rank_in_type_t1",
            ],
        ].copy()
        predictions.insert(4, "model_name", "lightgbm")
        predictions["pred_share_t1"] = []
        predictions["pred_target_growth"] = []
        predictions = predictions.loc[:, list(TREND_MODEL_PREDICTION_COLUMNS)]

        with pytest.raises(ValueError, match="lightgbm.*空"):
            lightgbm_model.describe_residual_distribution({"valid": predictions})

    def test_build_feature_importance_frame_normalizes_gain(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        booster = _FakeBooster(
            feature_names=["growth_lag_1", "attr_type"],
            split_importance=[3, 1],
            gain_importance=[2.0, 6.0],
        )

        importance = lightgbm_model.build_feature_importance_frame(booster)

        assert importance.to_dict(orient="records") == [
            {
                "feature": "growth_lag_1",
                "split_importance": 3,
                "gain_importance": 2.0,
                "normalized_gain_importance": 0.25,
            },
            {
                "feature": "attr_type",
                "split_importance": 1,
                "gain_importance": 6.0,
                "normalized_gain_importance": 0.75,
            },
        ]

    def test_build_feature_importance_frame_handles_zero_gain(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        booster = _FakeBooster(
            feature_names=["growth_lag_1", "attr_type"],
            split_importance=[0, 0],
            gain_importance=[0.0, 0.0],
        )

        importance = lightgbm_model.build_feature_importance_frame(booster)

        assert importance["normalized_gain_importance"].tolist() == [0.0, 0.0]


def _sample_lightgbm_samples(split: str):
    from tests.trend_samples import sample_trend_model_samples_for_split

    samples = sample_trend_model_samples_for_split().head(4).copy()
    samples["split"] = split
    return samples


class _FakeBooster:
    def __init__(
        self,
        feature_names: list[str],
        split_importance: list[int],
        gain_importance: list[float],
    ) -> None:
        self._feature_names = feature_names
        self._split_importance = split_importance
        self._gain_importance = gain_importance

    def feature_name(self) -> list[str]:
        return list(self._feature_names)

    def feature_importance(self, importance_type: str):
        if importance_type == "split":
            return list(self._split_importance)
        if importance_type == "gain":
            return list(self._gain_importance)
        raise AssertionError(f"unexpected importance_type={importance_type}")

    def model_to_string(self) -> str:
        return "fake lightgbm model"
