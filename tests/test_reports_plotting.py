from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from fashion_trend.reports.figures import (
    build_feature_importance_figure,
    build_recommendation_method_metrics_figure,
    build_trend_curve_examples_figure,
    build_trend_model_metrics_figure,
)
from fashion_trend.reports.plotting import (
    configure_matplotlib_for_reports,
    save_report_figure,
)


def test_configure_matplotlib_requires_cjk_font(monkeypatch) -> None:
    monkeypatch.setattr(
        "fashion_trend.reports.plotting.available_cjk_fonts", lambda: []
    )

    try:
        configure_matplotlib_for_reports()
    except RuntimeError as exc:
        assert "缺少可用中文字体" in str(exc)
    else:
        raise AssertionError("missing CJK font should fail")


def test_configure_matplotlib_sets_report_rc_params(monkeypatch) -> None:
    monkeypatch.setattr(
        "fashion_trend.reports.plotting.available_cjk_fonts",
        lambda: ["Test CJK Font"],
    )

    selected_font = configure_matplotlib_for_reports()

    assert selected_font == "Test CJK Font"
    assert plt.rcParams["font.sans-serif"] == ["Test CJK Font", "DejaVu Sans"]
    assert plt.rcParams["axes.unicode_minus"] is False
    assert plt.rcParams["figure.dpi"] == 140
    assert plt.rcParams["savefig.dpi"] == 180


@pytest.mark.parametrize(
    ("formats", "expected_suffixes", "unexpected_suffixes"),
    [
        (("svg",), ("svg",), ("png",)),
        (("png",), ("png",), ("svg",)),
        (("svg", "png"), ("svg", "png"), ()),
    ],
)
def test_save_report_figure_honors_requested_formats(
    tmp_path,
    monkeypatch,
    formats,
    expected_suffixes,
    unexpected_suffixes,
) -> None:
    monkeypatch.setattr(
        "fashion_trend.reports.plotting.available_cjk_fonts",
        lambda: ["DejaVu Sans"],
    )
    configure_matplotlib_for_reports()
    figure, axis = plt.subplots()
    axis.set_title("NDCG@12")
    axis.plot([-1, 0, 1], [1, 0, -1])
    paths = {"svg": tmp_path / "figure.svg", "png": tmp_path / "figure.png"}

    written = save_report_figure(figure, paths, formats=formats)

    assert written == [paths[suffix] for suffix in expected_suffixes]
    for suffix in expected_suffixes:
        assert paths[suffix].stat().st_size > 0
    for suffix in unexpected_suffixes:
        assert not paths[suffix].exists()


@pytest.mark.parametrize(
    ("formats", "path_keys", "match"),
    [
        ((), ("svg", "png"), "只支持"),
        (("pdf",), ("svg", "png"), "只支持"),
        (("svg", "svg"), ("svg", "png"), "不能重复"),
        (("svg", "png"), ("svg",), "缺少格式"),
    ],
)
def test_save_report_figure_closes_on_invalid_formats(
    tmp_path,
    monkeypatch,
    formats,
    path_keys,
    match,
) -> None:
    monkeypatch.setattr(
        "fashion_trend.reports.plotting.available_cjk_fonts",
        lambda: ["DejaVu Sans"],
    )
    configure_matplotlib_for_reports()
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    all_paths = {"svg": tmp_path / "figure.svg", "png": tmp_path / "figure.png"}
    paths = {key: all_paths[key] for key in path_keys}

    with pytest.raises(ValueError, match=match):
        save_report_figure(figure, paths, formats=formats)

    assert not plt.fignum_exists(figure.number)


def test_save_report_figure_cleans_partial_outputs_on_failure(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "fashion_trend.reports.plotting.available_cjk_fonts",
        lambda: ["DejaVu Sans"],
    )
    configure_matplotlib_for_reports()
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    paths = {"svg": tmp_path / "figure.svg", "png": tmp_path / "figure.png"}

    def fail_on_png(path, *args, **kwargs) -> None:
        path.write_text("partial", encoding="utf-8")
        if "png" in path.name:
            raise RuntimeError("png save failed")

    monkeypatch.setattr(figure, "savefig", fail_on_png)

    with pytest.raises(RuntimeError, match="png save failed"):
        save_report_figure(figure, paths, formats=("svg", "png"))

    assert not plt.fignum_exists(figure.number)
    assert not paths["svg"].exists()
    assert not paths["png"].exists()


@pytest.mark.parametrize(
    ("builder", "dataframe", "kwargs"),
    [
        (
            build_trend_model_metrics_figure,
            pd.DataFrame(columns=["model_name", "split", "ndcg_at_10"]),
            {},
        ),
        (
            build_recommendation_method_metrics_figure,
            pd.DataFrame(columns=["method", "split", "ndcg_at_12"]),
            {},
        ),
        (
            build_feature_importance_figure,
            pd.DataFrame(columns=["feature", "normalized_gain_importance"]),
            {},
        ),
        (
            build_trend_curve_examples_figure,
            pd.DataFrame(
                columns=[
                    "week_id",
                    "attr_id",
                    "attr_type",
                    "attr_value",
                    "heat_t",
                    "pred_share_t1",
                    "pred_target_growth",
                    "is_trend_eligible_t",
                ]
            ),
            {"week_id": 10},
        ),
    ],
)
def test_figure_builders_reject_empty_inputs(builder, dataframe, kwargs) -> None:
    with pytest.raises(ValueError, match="无可绘制数据"):
        builder(dataframe, **kwargs)


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (build_trend_model_metrics_figure, {}),
        (build_recommendation_method_metrics_figure, {}),
        (build_feature_importance_figure, {}),
        (build_trend_curve_examples_figure, {"week_id": 10}),
    ],
)
def test_figure_builders_reject_missing_columns(builder, kwargs) -> None:
    with pytest.raises(ValueError, match="缺少列"):
        builder(pd.DataFrame([{}]), **kwargs)


def test_trend_model_metrics_figure_rejects_duplicate_metric_key() -> None:
    metrics = pd.DataFrame(
        [
            {"model_name": "last_week", "split": "valid", "ndcg_at_10": 0.1},
            {"model_name": "last_week", "split": "valid", "ndcg_at_10": 0.2},
        ]
    )

    with pytest.raises(ValueError, match="trend_model_metrics.*last_week.*valid"):
        build_trend_model_metrics_figure(metrics)


def test_recommendation_metrics_figure_rejects_duplicate_metric_key() -> None:
    metrics = pd.DataFrame(
        [
            {"method": "pop_similarity", "split": "test", "ndcg_at_12": 0.1},
            {"method": "pop_similarity", "split": "test", "ndcg_at_12": 0.2},
        ]
    )

    with pytest.raises(ValueError, match="recommendation_method_metrics.*test"):
        build_recommendation_method_metrics_figure(metrics)


def test_trend_curve_examples_figure_rejects_duplicate_trend_view_key() -> None:
    trend_view = pd.DataFrame(
        [
            {
                "week_id": 10,
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 10,
                "pred_share_t1": 0.1,
                "pred_target_growth": 0.2,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": 10,
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 12,
                "pred_share_t1": 0.2,
                "pred_target_growth": 0.3,
                "is_trend_eligible_t": True,
            },
        ]
    )

    try:
        with pytest.raises(
            ValueError,
            match=(
                "trend_view.*attr_id.*1.*attr_type.*colour_group_name.*"
                "attr_value.*Black.*week_id.*10"
            ),
        ):
            build_trend_curve_examples_figure(trend_view, week_id=10)
    finally:
        plt.close("all")


def test_trend_curve_examples_figure_rejects_normalized_duplicate_key() -> None:
    trend_view = pd.DataFrame(
        [
            {
                "week_id": 10,
                "attr_id": 1,
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 10,
                "pred_share_t1": 0.1,
                "pred_target_growth": 0.2,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": "10",
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 12,
                "pred_share_t1": 0.2,
                "pred_target_growth": 0.3,
                "is_trend_eligible_t": True,
            },
        ]
    )

    try:
        with pytest.raises(
            ValueError,
            match=(
                "trend_view.*attr_id.*1.*attr_type.*colour_group_name.*"
                "attr_value.*Black.*week_id.*10"
            ),
        ):
            build_trend_curve_examples_figure(trend_view, week_id=10)
    finally:
        plt.close("all")


def test_trend_curve_examples_figure_rejects_non_integer_week_id() -> None:
    trend_view = pd.DataFrame(
        [
            {
                "week_id": "bad-week",
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 10,
                "pred_share_t1": 0.1,
                "pred_target_growth": 0.2,
                "is_trend_eligible_t": True,
            }
        ]
    )

    try:
        with pytest.raises(ValueError, match="trend_view.*week_id.*bad-week"):
            build_trend_curve_examples_figure(trend_view, week_id=10)
    finally:
        plt.close("all")


def test_trend_curve_example_uses_normalized_week_id_for_x_axis() -> None:
    trend_view = pd.DataFrame(
        [
            {
                "week_id": 8,
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 10,
                "pred_share_t1": 0.1,
                "pred_target_growth": 0.2,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": "9",
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 12,
                "pred_share_t1": 0.2,
                "pred_target_growth": 0.3,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": 10,
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 16,
                "pred_share_t1": 0.3,
                "pred_target_growth": 0.4,
                "is_trend_eligible_t": True,
            },
        ]
    )

    figure = build_trend_curve_examples_figure(
        trend_view,
        week_id=10,
        lookback_weeks=3,
        top_n=1,
    )

    try:
        assert list(figure.axes[0].lines[0].get_xdata()) == [8, 9, 10]
    finally:
        plt.close(figure)


def test_trend_curve_example_uses_separate_axes_and_dynamic_lookback_title() -> None:
    trend_view = pd.DataFrame(
        [
            {
                "week_id": 8,
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 10,
                "pred_share_t1": 0.1,
                "pred_target_growth": 0.2,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": 9,
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 12,
                "pred_share_t1": 0.2,
                "pred_target_growth": 0.3,
                "is_trend_eligible_t": True,
            },
            {
                "week_id": 10,
                "attr_id": "1",
                "attr_type": "colour_group_name",
                "attr_value": "Black",
                "heat_t": 16,
                "pred_share_t1": 0.3,
                "pred_target_growth": 0.4,
                "is_trend_eligible_t": True,
            },
        ]
    )

    figure = build_trend_curve_examples_figure(
        trend_view,
        week_id=10,
        lookback_weeks=3,
        top_n=1,
    )

    try:
        assert len(figure.axes) == 3
        assert [axis.get_ylabel() for axis in figure.axes] == [
            "heat_t",
            "pred_share_t1",
            "pred_target_growth",
        ]
        assert figure._suptitle is not None
        assert "最近 3 周" in figure._suptitle.get_text()
    finally:
        plt.close(figure)
