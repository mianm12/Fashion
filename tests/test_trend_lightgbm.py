from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
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
            "objective": "regression_l1",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 30,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_split_gain": 0.0,
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

    def test_lightgbm_default_params_enable_subsample_freq(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)

        assert lightgbm_model.LIGHTGBM_PARAMS["subsample"] == 0.8
        assert lightgbm_model.LIGHTGBM_PARAMS["subsample_freq"] == 1
        assert lightgbm_model.LIGHTGBM_PARAMS["objective"] == "regression_l1"
        assert lightgbm_model.LIGHTGBM_PARAMS["colsample_bytree"] == 0.6
        assert lightgbm_model.LIGHTGBM_PARAMS["min_child_samples"] == 30

    def test_resolve_lightgbm_config_merges_file_and_cli_overrides(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        params_path = tmp_path / "params.json"
        params_path.write_text(
            json.dumps(
                {
                    "lightgbm_params": {"learning_rate": 0.03, "num_leaves": 63},
                    "early_stopping": {"stopping_rounds": 50},
                }
            ),
            encoding="utf-8",
        )

        config = config_module.resolve_lightgbm_config(
            params_path=params_path,
            cli_params=["num_leaves=31", "early_stopping.stopping_rounds=80"],
        )

        assert config.lightgbm_params["learning_rate"] == 0.03
        assert config.lightgbm_params["num_leaves"] == 31
        assert config.early_stopping == {"stopping_rounds": 80}
        assert config.param_source["params_file"] == str(params_path)
        assert config.param_source["overrides"] == {
            "num_leaves": 31,
            "early_stopping.stopping_rounds": 80,
        }

    def test_resolve_lightgbm_config_from_stable_reads_complete_artifact(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = (
            tmp_path / "outputs" / "models" / "lightgbm" / "params.json"
        )
        stable_params_path.parent.mkdir(parents=True)
        lightgbm_params = dict(config_module.LIGHTGBM_DEFAULT_PARAMS)
        lightgbm_params.update({"learning_rate": 0.03, "num_leaves": 63})
        stable_params_path.write_text(
            json.dumps(
                {
                    "model_name": "lightgbm",
                    "model_type": "supervised",
                    "best_iteration": 21,
                    "lightgbm_params": lightgbm_params,
                    "early_stopping": {"stopping_rounds": 45},
                }
            ),
            encoding="utf-8",
        )

        config = config_module.resolve_lightgbm_config_from_stable_or_default(
            stable_params_path
        )

        assert config.lightgbm_params["learning_rate"] == 0.03
        assert config.lightgbm_params["num_leaves"] == 63
        assert config.lightgbm_params["subsample_freq"] == 1
        assert config.early_stopping == {"stopping_rounds": 45}
        assert config.param_source == {
            "default": "stable",
            "params_file": str(stable_params_path),
            "overrides": {},
        }

    def test_resolve_lightgbm_config_from_stable_missing_file_uses_builtin(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = (
            tmp_path / "outputs" / "models" / "lightgbm" / "params.json"
        )

        config = config_module.resolve_lightgbm_config_from_stable_or_default(
            stable_params_path
        )

        assert config.lightgbm_params == config_module.LIGHTGBM_DEFAULT_PARAMS
        assert config.early_stopping == config_module.LIGHTGBM_DEFAULT_EARLY_STOPPING
        assert config.param_source == {
            "default": "builtin",
            "params_file": None,
            "overrides": {},
        }

    def test_resolve_lightgbm_config_from_stable_rejects_missing_core_sections(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "params.json"
        stable_params_path.write_text(json.dumps({}), encoding="utf-8")

        with pytest.raises(ValueError, match="stable|params.json|lightgbm_params"):
            config_module.resolve_lightgbm_config_from_stable_or_default(
                stable_params_path
            )

    def test_resolve_lightgbm_config_from_stable_rejects_partial_params(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "params.json"
        partial_params = dict(config_module.LIGHTGBM_DEFAULT_PARAMS)
        partial_params.pop("learning_rate")
        stable_params_path.write_text(
            json.dumps(
                {
                    "lightgbm_params": partial_params,
                    "early_stopping": {"stopping_rounds": 30},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="stable|learning_rate"):
            config_module.resolve_lightgbm_config_from_stable_or_default(
                stable_params_path
            )

    def test_resolve_lightgbm_config_from_stable_rejects_missing_early_stopping_key(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "params.json"
        stable_params_path.write_text(
            json.dumps(
                {
                    "lightgbm_params": dict(config_module.LIGHTGBM_DEFAULT_PARAMS),
                    "early_stopping": {},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="stable|stopping_rounds"):
            config_module.resolve_lightgbm_config_from_stable_or_default(
                stable_params_path
            )

    def test_resolve_lightgbm_config_from_stable_rejects_invalid_json(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = tmp_path / "params.json"
        stable_params_path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(ValueError, match="stable|JSON|params.json"):
            config_module.resolve_lightgbm_config_from_stable_or_default(
                stable_params_path
            )

    def test_resolve_lightgbm_config_from_stable_rejects_invalid_payloads(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        valid_params = dict(config_module.LIGHTGBM_DEFAULT_PARAMS)
        cases = [
            ("top_level_array", [], "object"),
            (
                "lightgbm_params_type",
                {"lightgbm_params": [], "early_stopping": {"stopping_rounds": 30}},
                "lightgbm_params",
            ),
            (
                "early_stopping_type",
                {"lightgbm_params": valid_params, "early_stopping": []},
                "early_stopping",
            ),
            (
                "unknown_param",
                {
                    "lightgbm_params": {**valid_params, "unknown": 1},
                    "early_stopping": {"stopping_rounds": 30},
                },
                "unknown|允许清单",
            ),
            (
                "invalid_param_value",
                {
                    "lightgbm_params": {**valid_params, "learning_rate": 0},
                    "early_stopping": {"stopping_rounds": 30},
                },
                "learning_rate|大于 0",
            ),
        ]
        for case_name, payload, error_match in cases:
            stable_params_path = tmp_path / f"{case_name}.json"
            stable_params_path.write_text(json.dumps(payload), encoding="utf-8")

            with pytest.raises(ValueError, match=error_match):
                config_module.resolve_lightgbm_config_from_stable_or_default(
                    stable_params_path
                )

    def test_resolve_lightgbm_config_explicit_param_does_not_use_stable(
        self,
        tmp_path: Path,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        stable_params_path = (
            tmp_path / "outputs" / "models" / "lightgbm" / "params.json"
        )
        stable_params_path.parent.mkdir(parents=True)
        lightgbm_params = dict(config_module.LIGHTGBM_DEFAULT_PARAMS)
        lightgbm_params["num_leaves"] = 63
        stable_params_path.write_text(
            json.dumps(
                {
                    "lightgbm_params": lightgbm_params,
                    "early_stopping": {"stopping_rounds": 45},
                }
            ),
            encoding="utf-8",
        )

        config = config_module.resolve_lightgbm_config(
            cli_params=["learning_rate=0.03"]
        )

        assert config.lightgbm_params["learning_rate"] == 0.03
        assert config.lightgbm_params["num_leaves"] == 31
        assert config.early_stopping == {"stopping_rounds": 30}
        assert config.param_source == {
            "default": "builtin",
            "params_file": None,
            "overrides": {"learning_rate": 0.03},
        }

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {"unknown": {}},
            {"lightgbm_params": []},
            {"early_stopping": []},
        ],
    )
    def test_resolve_lightgbm_config_rejects_invalid_params_file_shape(
        self,
        tmp_path: Path,
        payload: object,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        params_path = tmp_path / "params.json"
        params_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="params|lightgbm_params|early_stopping"):
            config_module.resolve_lightgbm_config(params_path=params_path)

    @pytest.mark.parametrize(
        "cli_param",
        [
            "unknown=1",
            "objective=binary",
            "n_estimators=0",
            "learning_rate=0",
            "max_depth=0",
            "subsample=1.2",
            "subsample_freq=0",
            "colsample_bytree=0",
            "reg_alpha=-1",
            "reg_lambda=-1",
            "min_split_gain=-0.1",
            "early_stopping.stopping_rounds=0",
            "lightgbm_params.learning_rate=0.03",
        ],
    )
    def test_resolve_lightgbm_config_rejects_invalid_cli_param(
        self,
        cli_param: str,
    ) -> None:
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )

        with pytest.raises(ValueError, match="参数|objective|subsample|early_stopping"):
            config_module.resolve_lightgbm_config(cli_params=[cli_param])

    def test_lightgbm_trainer_metadata(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        trainer = lightgbm_model.LightGBMTrendTrainer()

        assert trainer.name == "lightgbm"
        assert trainer.model_type == MODEL_TYPE_SUPERVISED

    def test_native_lightgbm_import_is_deferred_until_fit(self, monkeypatch) -> None:
        import builtins

        for module_name in (
            LIGHTGBM_MODULE,
            "fashion_trend.trend.models.registry",
        ):
            sys.modules.pop(module_name, None)

        original_import = builtins.__import__
        blocked_imports: list[str] = []

        def block_lightgbm(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lightgbm" or name.startswith("lightgbm."):
                blocked_imports.append(name)
                raise ImportError("blocked lightgbm import")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", block_lightgbm)

        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        registry = importlib.import_module("fashion_trend.trend.models.registry")

        assert "lightgbm" in registry.list_trend_model_names()
        assert registry.get_trend_model_trainer("last_week").name == "last_week"
        assert registry.get_trend_model_trainer("previous_growth").name == (
            "previous_growth"
        )
        assert registry.get_trend_model_trainer("moving_average").name == (
            "moving_average"
        )
        assert blocked_imports == []

        with pytest.raises(ValueError, match="lightgbm|native runtime|原生运行时"):
            lightgbm_model._fit_lightgbm_model(
                _sample_lightgbm_samples("train").loc[:, ["growth_lag_1"]],
                _sample_lightgbm_samples("train")["target_growth"],
                _sample_lightgbm_samples("valid").loc[:, ["growth_lag_1"]],
                _sample_lightgbm_samples("valid")["target_growth"],
                config=lightgbm_model.resolve_lightgbm_config(),
            )

        assert blocked_imports == ["lightgbm"]

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

    def test_trainer_returns_standard_train_result(self, monkeypatch) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.models.base import TrendTrainContext, TrendTrainResult
        from fashion_trend.trend.schema import TREND_MODEL_PREDICTION_COLUMNS
        from fashion_trend.trend.splits import build_trend_model_split_frames
        from tests.trend_samples import sample_trend_model_samples_for_split

        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )

        def fake_fit(
            train_features, train_target, valid_features, valid_target, *, config
        ):
            assert config.lightgbm_params["subsample_freq"] == 1
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)

        result = lightgbm_model.LightGBMTrendTrainer().train(
            TrendTrainContext(
                model_name="lightgbm",
                split_frames=split_frames,
                input_paths={
                    "train": Path("train.parquet"),
                    "valid": Path("valid.parquet"),
                    "test": Path("test.parquet"),
                },
                output_dir=Path("outputs/models/lightgbm"),
            )
        )

        assert isinstance(result, TrendTrainResult)
        assert result.model_name == "lightgbm"
        assert result.model_type == MODEL_TYPE_SUPERVISED
        assert result.predictions.columns.tolist() == list(
            TREND_MODEL_PREDICTION_COLUMNS
        )
        assert set(result.predictions["model_name"]) == {"lightgbm"}
        assert result.params["objective"] == "regression_l1"
        assert result.params["best_iteration"] == 7
        assert result.metadata["attr_type_categories"] == ["colour_group_name"]
        assert set(result.metadata["target_distribution"]) == {"train", "valid", "test"}
        assert set(result.metadata["residual_distribution"]) == {"valid", "test"}
        assert [artifact.relative_path for artifact in result.artifacts] == [
            "feature_importance.csv",
            "model.txt",
        ]

    def test_trainer_uses_context_lightgbm_config(self, monkeypatch) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        config_module = importlib.import_module(
            "fashion_trend.trend.models.supervised.lightgbm_config"
        )
        from fashion_trend.trend.models.base import TrendTrainContext
        from fashion_trend.trend.splits import build_trend_model_split_frames
        from tests.trend_samples import sample_trend_model_samples_for_split

        captured: dict[str, object] = {}

        def fake_fit(
            train_features,
            train_target,
            valid_features,
            valid_target,
            *,
            config,
        ):
            captured["params"] = dict(config.lightgbm_params)
            captured["early_stopping"] = dict(config.early_stopping)
            return _FakeLightGBMModel(train_features.columns.tolist())

        monkeypatch.setattr(lightgbm_model, "_fit_lightgbm_model", fake_fit)
        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        config = config_module.resolve_lightgbm_config(
            cli_params=["learning_rate=0.03", "early_stopping.stopping_rounds=50"]
        )

        result = lightgbm_model.LightGBMTrendTrainer().train(
            TrendTrainContext(
                model_name="lightgbm",
                split_frames=split_frames,
                input_paths={
                    "train": Path("train.parquet"),
                    "valid": Path("valid.parquet"),
                    "test": Path("test.parquet"),
                },
                output_dir=Path("outputs/models/lightgbm/runs/custom"),
                trainer_options={"lightgbm_config": config},
            )
        )

        assert captured["params"]["learning_rate"] == 0.03
        assert captured["early_stopping"] == {"stopping_rounds": 50}
        assert result.params["lightgbm_params"]["learning_rate"] == 0.03
        assert result.params["early_stopping"] == {"stopping_rounds": 50}
        assert result.metadata["param_source"]["overrides"] == {
            "learning_rate": 0.03,
            "early_stopping.stopping_rounds": 50,
        }

    def test_read_best_score_returns_json_serializable_values(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)

        class ScoreModel:
            best_score_ = {
                "valid_0": {
                    "l2": np.float32(0.12),
                    "history": [np.float64(0.2), (np.int64(3),)],
                }
            }

        best_score = lightgbm_model._read_best_score(ScoreModel())

        json.dumps(best_score)
        assert best_score == {
            "valid_0": {"l2": pytest.approx(0.12), "history": [0.2, [3]]}
        }

    def test_build_lightgbm_predictions_rejects_prediction_length_mismatch(
        self,
    ) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)

        with pytest.raises(ValueError, match="lightgbm.*(预测|行数)"):
            lightgbm_model._build_lightgbm_predictions(
                _sample_lightgbm_samples("valid"),
                [0.1],
            )

    def test_trainer_rejects_split_frame_with_mismatched_split_value(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.models.base import TrendTrainContext
        from fashion_trend.trend.splits import build_trend_model_split_frames
        from tests.trend_samples import sample_trend_model_samples_for_split

        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        split_frames["valid"] = split_frames["valid"].assign(split="train")

        with pytest.raises(ValueError, match="split.*valid|不一致"):
            lightgbm_model.LightGBMTrendTrainer().train(
                TrendTrainContext(
                    model_name="lightgbm",
                    split_frames=split_frames,
                    input_paths={
                        "train": Path("train.parquet"),
                        "valid": Path("valid.parquet"),
                        "test": Path("test.parquet"),
                    },
                    output_dir=Path("outputs/models/lightgbm"),
                )
            )

    def test_trainer_rejects_split_frame_missing_split_column(self) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)
        from fashion_trend.trend.models.base import TrendTrainContext
        from fashion_trend.trend.splits import build_trend_model_split_frames
        from tests.trend_samples import sample_trend_model_samples_for_split

        split_frames = build_trend_model_split_frames(
            sample_trend_model_samples_for_split(),
            valid_weeks=4,
            test_weeks=4,
        )
        split_frames["valid"] = split_frames["valid"].drop(columns=["split"])

        with pytest.raises(ValueError, match="lightgbm.*split.*缺少|缺少.*split"):
            lightgbm_model.LightGBMTrendTrainer().train(
                TrendTrainContext(
                    model_name="lightgbm",
                    split_frames=split_frames,
                    input_paths={
                        "train": Path("train.parquet"),
                        "valid": Path("valid.parquet"),
                        "test": Path("test.parquet"),
                    },
                    output_dir=Path("outputs/models/lightgbm"),
                )
            )

    def test_fit_lightgbm_model_wraps_native_import_errors(self, monkeypatch) -> None:
        lightgbm_model = importlib.import_module(LIGHTGBM_MODULE)

        def broken_import(name, *args, **kwargs):
            if name == "lightgbm":
                raise OSError("libomp.dylib not found")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", broken_import)

        with pytest.raises(ValueError, match="lightgbm|libomp"):
            lightgbm_model._fit_lightgbm_model(
                _sample_lightgbm_samples("train").loc[:, ["growth_lag_1"]],
                _sample_lightgbm_samples("train")["target_growth"],
                _sample_lightgbm_samples("valid").loc[:, ["growth_lag_1"]],
                _sample_lightgbm_samples("valid")["target_growth"],
                config=lightgbm_model.resolve_lightgbm_config(),
            )


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


class _FakeLightGBMModel:
    best_iteration_ = 7
    best_score_ = {"valid_0": {"l2": 0.12}}

    def __init__(self, feature_names: list[str]) -> None:
        self.booster_ = _FakeBooster(
            feature_names=feature_names,
            split_importance=[1 for _ in feature_names],
            gain_importance=[float(index + 1) for index, _ in enumerate(feature_names)],
        )

    def predict(self, features, num_iteration=None):
        return features["growth_lag_1"].astype(float).to_numpy()
