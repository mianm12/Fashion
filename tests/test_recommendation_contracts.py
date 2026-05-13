from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.recommendation import contracts, paths, readers


def test_public_recommendation_contract_constants() -> None:
    assert contracts.RECOMMENDATION_TOP_K == 12
    assert contracts.RECOMMENDATION_ARTICLE_ID_DTYPE == "string"
    assert contracts.CUSTOMER_AGE_BUCKETS == (
        "unknown",
        "0-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60+",
    )
    assert contracts.VALID_RECOMMENDATION_SPLITS == ("valid", "test")
    assert contracts.RECOMMENDATION_METHODS == (
        "global_popularity",
        "recent_popularity",
        "attribute_similarity",
        "pop_similarity",
        "pop_similarity_trend",
        "enhanced_pop_similarity_trend",
    )
    assert contracts.RECOMMENDATION_CANDIDATE_STRATEGIES == (
        "popularity",
        "similarity",
        "trend_union",
        "default",
        "enhanced_default",
    )
    assert contracts.RECOMMENDATION_SCORE_COLUMNS == (
        "pop_score",
        "sim_score",
        "trend_score",
        "recent_score",
    )
    assert contracts.ENHANCED_RECOMMENDATION_SCORE_COLUMNS == (
        "pop_score",
        "recent_score",
        "sim_score",
        "trend_score",
        "reorder_score",
        "variant_score",
        "age_pop_score",
        "preference_pop_score",
        "source_rank_score",
        "source_count_score",
    )
    assert contracts.RECOMMENDATION_CORE_ATTR_TYPES == (
        "product_type_name",
        "colour_group_name",
        "garment_group_name",
        "product_group_name",
        "graphical_appearance_name",
    )
    assert contracts.RECOMMENDATION_TREND_ATTR_WEIGHTS == {
        "product_type_name": 0.35,
        "colour_group_name": 0.25,
        "garment_group_name": 0.20,
        "product_group_name": 0.10,
        "graphical_appearance_name": 0.10,
    }
    assert contracts.TIME_WINDOW_KEY_COLUMNS == ("split", "cutoff_week", "label_week")
    assert contracts.TARGET_USER_KEY_COLUMNS == (
        "split",
        "cutoff_week",
        "label_week",
        "customer_id",
    )
    assert contracts.EVALUATION_LABEL_KEY_COLUMNS == (
        "split",
        "cutoff_week",
        "label_week",
        "customer_id",
        "article_id",
    )
    assert contracts.USER_PROFILE_KEY_COLUMNS == (
        "split",
        "cutoff_week",
        "label_week",
        "customer_id",
        "attr_id",
        "attr_type",
        "attr_value",
    )
    assert contracts.CUSTOMER_PROFILE_COLUMNS == (
        "customer_id",
        "age",
        "age_bucket",
        "club_member_status",
        "fashion_news_frequency",
    )
    assert contracts.ARTICLE_PRODUCT_MAP_COLUMNS == ("article_id", "product_code")
    assert contracts.CANDIDATE_ITEM_KEY_COLUMNS == (
        "split",
        "cutoff_week",
        "label_week",
        "strategy",
        "customer_id",
        "article_id",
    )
    assert contracts.RECOMMENDATIONS_KEY_COLUMNS == (
        "customer_id",
        "split",
        "cutoff_week",
        "label_week",
        "method",
    )
    assert contracts.RECOMMENDATION_ITEMS_KEY_COLUMNS == (
        "customer_id",
        "split",
        "cutoff_week",
        "label_week",
        "method",
        "article_id",
    )


def test_recommendation_paths_are_scoped_by_strategy_method_and_experiment() -> None:
    candidate_path = paths.candidate_items_path("default")
    assert str(candidate_path).endswith(
        "data/processed/recommend/candidates/default/candidate_items.parquet"
    )

    output_paths = paths.method_output_paths("pop_similarity_trend")
    assert str(output_paths.output_dir).endswith(
        "outputs/recommendation/pop_similarity_trend"
    )
    assert output_paths.recommendations == (
        output_paths.output_dir / "recommendations.csv"
    )
    assert output_paths.recommendation_items == (
        output_paths.output_dir / "recommendation_items.parquet"
    )
    assert output_paths.recommendation_items_csv == (
        output_paths.output_dir / "recommendation_items.csv"
    )
    assert output_paths.params == output_paths.output_dir / "params.json"
    assert output_paths.metadata == output_paths.output_dir / "metadata.json"
    assert output_paths.metrics == output_paths.output_dir / "metrics.json"

    experiment_run = paths.experiment_run_dir("exp_001", "run_001")
    assert str(experiment_run).endswith(
        "outputs/recommendation/experiments/exp_001/runs/run_001"
    )


@pytest.mark.parametrize("unsafe_segment", ["", ".", "..", "a/b", "a\\b"])
def test_recommendation_paths_reject_unsafe_segments(unsafe_segment: str) -> None:
    with pytest.raises(ValueError, match="安全"):
        paths.candidate_items_path(unsafe_segment)
    with pytest.raises(ValueError, match="安全"):
        paths.method_output_paths(unsafe_segment)
    with pytest.raises(ValueError, match="安全"):
        paths.experiment_dir(unsafe_segment)


def test_reader_helpers_validate_columns_and_duplicate_keys() -> None:
    dataframe = pd.DataFrame([{"split": "valid", "cutoff_week": 1}])

    readers.validate_columns(
        dataframe,
        ("split", "cutoff_week"),
        "time_windows",
    )

    with pytest.raises(ValueError, match="列契约不匹配"):
        readers.validate_columns(dataframe, ("cutoff_week", "split"), "time_windows")

    duplicate_dataframe = pd.DataFrame(
        [
            {"split": "valid", "cutoff_week": 1},
            {"split": "valid", "cutoff_week": 1},
        ]
    )
    with pytest.raises(ValueError, match="重复键"):
        readers.reject_duplicate_key(
            duplicate_dataframe,
            ("split", "cutoff_week"),
            "time_windows",
        )


def test_parquet_readers_validate_exact_columns_and_duplicates(tmp_path: Path) -> None:
    time_windows_path = tmp_path / "time_windows.parquet"
    target_users_path = tmp_path / "target_users.parquet"
    evaluation_labels_path = tmp_path / "evaluation_labels.parquet"
    user_profile_path = tmp_path / "user_profile.parquet"

    _write_parquet(
        time_windows_path,
        contracts.TIME_WINDOW_COLUMNS,
        [("valid", 104, 105)],
    )
    _write_parquet(
        target_users_path,
        contracts.TARGET_USER_COLUMNS,
        [("valid", 104, 105, "0000001", 3, 1)],
    )
    _write_parquet(
        evaluation_labels_path,
        contracts.EVALUATION_LABEL_COLUMNS,
        [("valid", 104, 105, "0000001", "0000123")],
    )
    _write_parquet(
        user_profile_path,
        contracts.USER_PROFILE_COLUMNS,
        [
            (
                "valid",
                104,
                105,
                "0000001",
                "product_type_name:dress",
                "product_type_name",
                "dress",
                0.7,
                2,
                103,
            )
        ],
    )

    assert tuple(readers.read_time_windows(time_windows_path).columns) == (
        contracts.TIME_WINDOW_COLUMNS
    )
    assert tuple(readers.read_target_users(target_users_path).columns) == (
        contracts.TARGET_USER_COLUMNS
    )
    assert tuple(readers.read_evaluation_labels(evaluation_labels_path).columns) == (
        contracts.EVALUATION_LABEL_COLUMNS
    )
    assert tuple(readers.read_user_profile(user_profile_path).columns) == (
        contracts.USER_PROFILE_COLUMNS
    )

    bad_path = tmp_path / "bad_time_windows.parquet"
    _write_parquet(
        bad_path,
        contracts.TIME_WINDOW_COLUMNS,
        [("valid", 104, 105), ("valid", 104, 105)],
    )
    with pytest.raises(ValueError, match="重复键"):
        readers.read_time_windows(bad_path)


def test_read_customer_profile_rejects_invalid_age_bucket(tmp_path: Path) -> None:
    customer_profile_path = tmp_path / "customer_profile.parquet"
    _write_parquet(
        customer_profile_path,
        contracts.CUSTOMER_PROFILE_COLUMNS,
        [("0000001", 20, "20s", "ACTIVE", "Regularly")],
    )

    with pytest.raises(ValueError, match="age_bucket"):
        readers.read_customer_profile(customer_profile_path)


def test_read_article_product_map_rejects_duplicate_article_id(tmp_path: Path) -> None:
    article_product_map_path = tmp_path / "article_product_map.parquet"
    _write_parquet(
        article_product_map_path,
        contracts.ARTICLE_PRODUCT_MAP_COLUMNS,
        [("0000000001", "001"), ("0000000001", "002")],
    )

    with pytest.raises(ValueError, match="重复键"):
        readers.read_article_product_map(article_product_map_path)


def test_candidate_items_reader_validates_strategy_from_path(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates" / "default" / "candidate_items.parquet"
    candidate_path.parent.mkdir(parents=True)
    _write_parquet(
        candidate_path,
        contracts.CANDIDATE_ITEM_COLUMNS,
        [("valid", 104, 105, "default", "0000001", "0000123", "pop", "pop", 1)],
    )

    dataframe = readers.read_candidate_items(candidate_path)

    assert tuple(dataframe.columns) == contracts.CANDIDATE_ITEM_COLUMNS

    mismatch_path = tmp_path / "candidates" / "similarity" / "candidate_items.parquet"
    mismatch_path.parent.mkdir(parents=True)
    _write_parquet(
        mismatch_path,
        contracts.CANDIDATE_ITEM_COLUMNS,
        [("valid", 104, 105, "default", "0000001", "0000123", "pop", "pop", 1)],
    )
    with pytest.raises(ValueError, match="strategy"):
        readers.read_candidate_items(mismatch_path)


def test_candidate_items_reader_accepts_enhanced_strategy_columns(
    tmp_path: Path,
) -> None:
    candidate_path = (
        tmp_path / "candidates" / "enhanced_default" / "candidate_items.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    _write_parquet(
        candidate_path,
        contracts.ENHANCED_CANDIDATE_ITEM_COLUMNS,
        [
            (
                "valid",
                104,
                105,
                "enhanced_default",
                "0000001",
                "0000123",
                "reorder|trend",
                "reorder",
                1,
                True,
                True,
            )
        ],
    )

    dataframe = readers.read_candidate_items(candidate_path)

    assert tuple(dataframe.columns) == contracts.ENHANCED_CANDIDATE_ITEM_COLUMNS
    assert str(dataframe["customer_id"].dtype) == "string"
    assert str(dataframe["article_id"].dtype) == "string"


def test_recommendation_csv_readers_preserve_string_ids_and_validate_method(
    tmp_path: Path,
) -> None:
    method_dir = tmp_path / "pop_similarity_trend"
    method_dir.mkdir()
    recommendations_path = method_dir / "recommendations.csv"
    recommendation_items_path = method_dir / "recommendation_items.parquet"

    recommendations = pd.DataFrame(
        [
            {
                "customer_id": "0000001",
                "split": "valid",
                "cutoff_week": 104,
                "label_week": 105,
                "method": "pop_similarity_trend",
                "prediction": "0000123 0000456",
            }
        ],
        columns=contracts.RECOMMENDATIONS_COLUMNS,
    )
    recommendation_items = pd.DataFrame(
        [
            {
                "customer_id": "0000001",
                "split": "valid",
                "cutoff_week": 104,
                "label_week": 105,
                "method": "pop_similarity_trend",
                "article_id": "0000123",
                "rank": 1,
                "score": 1.0,
                "pop_score": 0.4,
                "sim_score": 0.3,
                "trend_score": 0.2,
                "recent_score": 0.1,
                "candidate_sources": "pop,trend",
            }
        ],
        columns=contracts.RECOMMENDATION_ITEMS_COLUMNS,
    )
    recommendations.to_csv(recommendations_path, index=False)
    recommendation_items.to_parquet(recommendation_items_path, index=False)

    read_recommendations = readers.read_recommendations(recommendations_path)
    read_recommendation_items = readers.read_recommendation_items(
        recommendation_items_path
    )

    assert str(read_recommendations["customer_id"].dtype) == "string"
    assert str(read_recommendations["prediction"].dtype) == "string"
    assert read_recommendations.loc[0, "customer_id"] == "0000001"
    assert read_recommendations.loc[0, "prediction"] == "0000123 0000456"
    assert str(read_recommendation_items["customer_id"].dtype) == "string"
    assert str(read_recommendation_items["article_id"].dtype) == "string"
    assert read_recommendation_items.loc[0, "customer_id"] == "0000001"
    assert read_recommendation_items.loc[0, "article_id"] == "0000123"
    legacy_read_recommendations = readers.read_recommendation_result(
        recommendations_path
    )
    assert str(legacy_read_recommendations["customer_id"].dtype) == "string"
    assert legacy_read_recommendations.loc[0, "customer_id"] == "0000001"

    mismatch_dir = tmp_path / "recent_popularity"
    mismatch_dir.mkdir()
    mismatch_path = mismatch_dir / "recommendations.csv"
    recommendations.to_csv(mismatch_path, index=False)
    with pytest.raises(ValueError, match="method"):
        readers.read_recommendations(mismatch_path)
    with pytest.raises(ValueError, match="method"):
        readers.read_recommendation_result(mismatch_path)

    duplicate_path = method_dir / "duplicate_recommendation_items.parquet"
    pd.concat(
        [recommendation_items, recommendation_items],
        ignore_index=True,
    ).to_parquet(duplicate_path, index=False)
    with pytest.raises(ValueError, match="重复键"):
        readers.read_recommendation_items(duplicate_path)

    invalid_rank_path = method_dir / "invalid_rank_recommendation_items.parquet"
    recommendation_items.assign(rank=13).to_parquet(invalid_rank_path, index=False)
    with pytest.raises(ValueError, match="Top-K"):
        readers.read_recommendation_items(invalid_rank_path)


def test_read_recommendation_items_backfills_legacy_enhanced_scores(
    tmp_path: Path,
) -> None:
    method_dir = tmp_path / "pop_similarity_trend"
    method_dir.mkdir()
    path = method_dir / "recommendation_items.parquet"
    legacy_columns = tuple(
        column
        for column in contracts.RECOMMENDATION_ITEMS_COLUMNS
        if column
        not in {
            "reorder_score",
            "variant_score",
            "age_pop_score",
            "preference_pop_score",
            "source_rank_score",
            "source_count_score",
        }
    )
    legacy_items = pd.DataFrame(
        [
            {
                "customer_id": "0000001",
                "split": "test",
                "cutoff_week": 104,
                "label_week": 105,
                "method": "pop_similarity_trend",
                "article_id": "0000123",
                "rank": 1,
                "score": 1.0,
                "pop_score": 0.4,
                "sim_score": 0.3,
                "trend_score": 0.2,
                "recent_score": 0.1,
                "candidate_sources": "popularity|trend",
            }
        ],
        columns=legacy_columns,
    )
    legacy_items.to_parquet(path, index=False)

    result = readers.read_recommendation_items(path)

    assert tuple(result.columns) == contracts.RECOMMENDATION_ITEMS_COLUMNS
    assert result.loc[0, "reorder_score"] == pytest.approx(0.0)
    assert result.loc[0, "source_count_score"] == pytest.approx(0.0)


def _write_parquet(
    path: Path, columns: tuple[str, ...], rows: list[tuple[object, ...]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_parquet(path, index=False)
