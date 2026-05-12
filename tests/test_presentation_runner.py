from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.presentation.contracts import (
    CORE_TREND_ATTR_TYPES,
    PRESENTATION_SCHEMA_VERSION,
)
from fashion_trend.presentation.schema import read_schema_version

CASE_ID = "demo_test_10_11_000000abcdef"
CUSTOMER_ID = "000000abcdef123456"


def test_write_presentation_database_creates_valid_sqlite_database(tmp_path: Path):
    from fashion_trend.presentation.sqlite_writer import write_presentation_database

    output_path = tmp_path / "fashion_demo.sqlite"

    write_presentation_database(_complete_tables(), output_path)

    assert output_path.exists()
    with sqlite3.connect(output_path) as connection:
        assert read_schema_version(connection) == PRESENTATION_SCHEMA_VERSION
        table_names = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type = 'table'"
            )
        }
        assert set(_complete_tables()) <= table_names
        assert _table_count(connection, "recommendation_items") == 12
        article_id = connection.execute("select article_id from articles").fetchone()[0]
        assert article_id == "0000000001"
        assert isinstance(article_id, str)


def test_write_presentation_database_rejects_missing_required_table_without_final_db(
    tmp_path: Path,
):
    from fashion_trend.presentation.sqlite_writer import write_presentation_database

    output_path = tmp_path / "fashion_demo.sqlite"
    tables = _complete_tables()
    del tables["demo_users"]

    with pytest.raises(ValueError, match="demo_users"):
        write_presentation_database(tables, output_path)

    assert not output_path.exists()
    assert not output_path.with_suffix(".sqlite.tmp").exists()


def test_write_presentation_database_rejects_empty_required_table_without_final_db(
    tmp_path: Path,
):
    from fashion_trend.presentation.sqlite_writer import write_presentation_database

    output_path = tmp_path / "fashion_demo.sqlite"
    tables = _complete_tables()
    tables["metrics_summary"] = tables["metrics_summary"].iloc[0:0].copy()

    with pytest.raises(ValueError, match="metrics_summary"):
        write_presentation_database(tables, output_path)

    assert not output_path.exists()
    assert not output_path.with_suffix(".sqlite.tmp").exists()


def test_write_presentation_database_rejects_empty_score_components_without_final_db(
    tmp_path: Path,
):
    from fashion_trend.presentation.sqlite_writer import write_presentation_database

    output_path = tmp_path / "fashion_demo.sqlite"
    tables = _complete_tables()
    tables["recommendation_score_components"] = (
        tables["recommendation_score_components"].iloc[0:0].copy()
    )

    with pytest.raises(ValueError, match="recommendation_score_components"):
        write_presentation_database(tables, output_path)

    assert not output_path.exists()
    assert not output_path.with_suffix(".sqlite.tmp").exists()


def test_write_presentation_database_rejects_missing_score_component_for_recommendation_item(
    tmp_path: Path,
):
    from fashion_trend.presentation.sqlite_writer import write_presentation_database

    output_path = tmp_path / "fashion_demo.sqlite"
    tables = _complete_tables()
    tables["recommendation_score_components"] = tables[
        "recommendation_score_components"
    ].query("article_id != '0000000012'")

    with pytest.raises(ValueError, match="recommendation.*score"):
        write_presentation_database(tables, output_path)

    assert not output_path.exists()
    assert not output_path.with_suffix(".sqlite.tmp").exists()


def test_write_presentation_database_rejects_incomplete_top12_case_without_final_db(
    tmp_path: Path,
):
    from fashion_trend.presentation.sqlite_writer import write_presentation_database

    output_path = tmp_path / "fashion_demo.sqlite"
    tables = _complete_tables()
    tables["recommendation_items"] = tables["recommendation_items"].query("rank < 12")

    with pytest.raises(ValueError, match="Top-12"):
        write_presentation_database(tables, output_path)

    assert not output_path.exists()
    assert not output_path.with_suffix(".sqlite.tmp").exists()


def test_write_presentation_database_keeps_existing_final_db_when_validation_fails(
    tmp_path: Path,
):
    from fashion_trend.presentation.sqlite_writer import write_presentation_database

    output_path = tmp_path / "fashion_demo.sqlite"
    with sqlite3.connect(output_path) as connection:
        connection.execute("create table old_marker(value TEXT)")
        connection.execute("insert into old_marker(value) values ('kept')")

    tables = _complete_tables()
    tables["recommendation_items"] = tables["recommendation_items"].query("rank < 12")

    with pytest.raises(ValueError, match="Top-12"):
        write_presentation_database(tables, output_path)

    with sqlite3.connect(output_path) as connection:
        assert (
            connection.execute("select value from old_marker").fetchone()[0] == "kept"
        )
    assert not output_path.with_suffix(".sqlite.tmp").exists()


def test_validate_database_rejects_missing_required_schema_table():
    from fashion_trend.presentation.sqlite_writer import validate_database

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "create table app_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "insert into app_metadata(key, value) values ('schema_version', ?)",
        (PRESENTATION_SCHEMA_VERSION,),
    )

    with pytest.raises(ValueError, match="demo_users"):
        validate_database(connection)


def test_run_defense_app_db_build_copies_report_assets_and_returns_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fashion_trend.presentation import runner

    source_asset = tmp_path / "source_figures" / "trend_board.svg"
    source_asset.parent.mkdir()
    source_asset.write_text("<svg></svg>", encoding="utf-8")
    output_path = tmp_path / "custom" / "fashion_demo.sqlite"
    tables = _complete_tables()
    tables["report_assets"] = pd.DataFrame(
        [
            {
                "asset_name": "trend_board.svg",
                "asset_type": "figure",
                "title": "trend board",
                "source_path": str(source_asset),
                "static_url": "/static/reports/trend_board.svg",
                "description": "demo figure",
            }
        ]
    )

    class FakeSources:
        source_artifacts = {
            "reports_manifest": {
                "path": str(tmp_path / "manifest.json"),
                "mtime": 1.0,
                "size": 2,
                "row_count": None,
            }
        }

    monkeypatch.setattr(runner, "load_presentation_sources", lambda: FakeSources())
    monkeypatch.setattr(runner, "build_presentation_tables", lambda sources: tables)

    payload = runner.run_defense_app_db_build(output_path=output_path)

    copied_asset = output_path.parent / "static" / "reports" / "trend_board.svg"
    assert copied_asset.read_text(encoding="utf-8") == "<svg></svg>"
    assert output_path.exists()
    assert payload["database_path"] == str(output_path)
    assert payload["table_counts"]["recommendation_items"] == 12
    assert payload["source_artifacts"] == FakeSources.source_artifacts
    assert payload["static_assets"] == [
        {
            "asset_name": "trend_board.svg",
            "source_path": str(source_asset),
            "static_path": str(copied_asset),
            "static_url": "/static/reports/trend_board.svg",
        }
    ]
    assert (
        tables["report_assets"].loc[0, "static_url"]
        == "/static/reports/trend_board.svg"
    )
    assert tables["report_assets"].loc[0, "source_path"] == str(source_asset)


def test_run_defense_app_db_build_does_not_replace_static_assets_when_database_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fashion_trend.presentation import runner

    source_asset = tmp_path / "source_figures" / "trend_board.svg"
    source_asset.parent.mkdir()
    source_asset.write_text("<svg>new</svg>", encoding="utf-8")
    output_path = tmp_path / "custom" / "fashion_demo.sqlite"
    old_asset = output_path.parent / "static" / "reports" / "trend_board.svg"
    old_asset.parent.mkdir(parents=True)
    old_asset.write_text("<svg>old</svg>", encoding="utf-8")
    tables = _complete_tables()
    tables["report_assets"] = _report_assets_frame(source_asset)

    class FakeSources:
        source_artifacts = {}

    def fail_write(*args, **kwargs):
        raise ValueError("database validation failed")

    monkeypatch.setattr(runner, "load_presentation_sources", lambda: FakeSources())
    monkeypatch.setattr(runner, "build_presentation_tables", lambda sources: tables)
    monkeypatch.setattr(runner, "write_presentation_database", fail_write)

    with pytest.raises(ValueError, match="database validation failed"):
        runner.run_defense_app_db_build(output_path=output_path)

    assert old_asset.read_text(encoding="utf-8") == "<svg>old</svg>"
    _assert_no_static_staging_dirs(output_path.parent)


def test_run_defense_app_db_build_does_not_replace_static_assets_when_asset_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fashion_trend.presentation import runner

    source_asset = tmp_path / "source_figures" / "trend_board.svg"
    source_asset.parent.mkdir()
    source_asset.write_text("<svg>new</svg>", encoding="utf-8")
    missing_asset = tmp_path / "source_figures" / "missing.svg"
    output_path = tmp_path / "custom" / "fashion_demo.sqlite"
    old_asset = output_path.parent / "static" / "reports" / "trend_board.svg"
    old_asset.parent.mkdir(parents=True)
    old_asset.write_text("<svg>old</svg>", encoding="utf-8")
    tables = _complete_tables()
    tables["report_assets"] = pd.concat(
        [
            _report_assets_frame(source_asset),
            _report_assets_frame(missing_asset, asset_name="missing.svg"),
        ],
        ignore_index=True,
    )

    class FakeSources:
        source_artifacts = {}

    monkeypatch.setattr(runner, "load_presentation_sources", lambda: FakeSources())
    monkeypatch.setattr(runner, "build_presentation_tables", lambda sources: tables)

    with pytest.raises(FileNotFoundError, match="missing.svg"):
        runner.run_defense_app_db_build(output_path=output_path)

    assert old_asset.read_text(encoding="utf-8") == "<svg>old</svg>"
    _assert_no_static_staging_dirs(output_path.parent)


def test_run_defense_app_db_build_restores_old_database_when_static_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fashion_trend.presentation import runner

    source_asset = tmp_path / "source_figures" / "trend_board.svg"
    source_asset.parent.mkdir()
    source_asset.write_text("<svg>new</svg>", encoding="utf-8")
    output_path = tmp_path / "custom" / "fashion_demo.sqlite"
    output_path.parent.mkdir(parents=True)
    with sqlite3.connect(output_path) as connection:
        connection.execute("create table old_marker(value TEXT)")
        connection.execute("insert into old_marker(value) values ('old-db')")
    old_asset = output_path.parent / "static" / "reports" / "trend_board.svg"
    old_asset.parent.mkdir(parents=True)
    old_asset.write_text("<svg>old</svg>", encoding="utf-8")
    tables = _complete_tables()
    tables["report_assets"] = _report_assets_frame(source_asset)

    class FakeSources:
        source_artifacts = {}

    def fail_publish(output_dir: Path) -> None:
        raise ValueError("static publish failed")

    monkeypatch.setattr(runner, "load_presentation_sources", lambda: FakeSources())
    monkeypatch.setattr(runner, "build_presentation_tables", lambda sources: tables)
    monkeypatch.setattr(runner, "publish_staged_report_assets", fail_publish)

    with pytest.raises(ValueError, match="static publish failed"):
        runner.run_defense_app_db_build(output_path=output_path)

    with sqlite3.connect(output_path) as connection:
        assert (
            connection.execute("select value from old_marker").fetchone()[0] == "old-db"
        )
    assert old_asset.read_text(encoding="utf-8") == "<svg>old</svg>"
    _assert_no_static_staging_dirs(output_path.parent)
    assert not output_path.with_suffix(".sqlite.backup").exists()


def _table_count(connection: sqlite3.Connection, table_name: str) -> int:
    return int(connection.execute(f"select count(*) from {table_name}").fetchone()[0])


def _report_assets_frame(
    source_path: Path,
    *,
    asset_name: str = "trend_board.svg",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset_name": asset_name,
                "asset_type": "figure",
                "title": "trend board",
                "source_path": str(source_path),
                "static_url": f"/static/reports/{asset_name}",
                "description": "demo figure",
            }
        ]
    )


def _assert_no_static_staging_dirs(output_dir: Path) -> None:
    static_dir = output_dir / "static"
    assert not (static_dir / "reports.tmp").exists()
    assert not (static_dir / "reports.backup").exists()


def _complete_tables() -> dict[str, pd.DataFrame]:
    rows = [_recommendation_row(rank) for rank in range(1, 13)]
    return {
        "app_metadata": pd.DataFrame(
            [
                {"key": "generated_at", "value": "2026-05-12T00:00:00+00:00"},
                {"key": "source_artifacts", "value": "{}"},
            ]
        ),
        "demo_users": pd.DataFrame(
            [
                {
                    "case_id": CASE_ID,
                    "customer_id": CUSTOMER_ID,
                    "split": "test",
                    "cutoff_week": 10,
                    "label_week": 11,
                    "hit_count": 1,
                    "primary_tags": "[]",
                    "profile_summary": "profile",
                    "recommendation_summary": "recommendations",
                }
            ]
        ),
        "user_profile_attributes": pd.DataFrame(
            [
                {
                    "case_id": CASE_ID,
                    "customer_id": CUSTOMER_ID,
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                    "preference_score": 0.8,
                    "purchase_count": 3,
                    "last_purchase_week": 10,
                }
            ]
        ),
        "recommendation_items": pd.DataFrame(rows),
        "recommendation_score_components": pd.DataFrame(
            [
                {
                    "case_id": CASE_ID,
                    "article_id": row["article_id"],
                    "pop_score": 0.1,
                    "sim_score": 0.2,
                    "trend_score": 0.3,
                    "recent_score": 0.4,
                    "final_score": row["score"],
                }
                for row in rows
            ]
        ),
        "articles": pd.DataFrame(
            [
                {
                    "article_id": "0000000001",
                    "prod_name": "Demo Shirt",
                    "product_group_name": "Garment Upper body",
                    "product_type_name": "Shirt",
                    "garment_group_name": "Jersey Basic",
                    "colour_group_name": "Black",
                    "graphical_appearance_name": "Solid",
                    "department_name": "Ladieswear",
                    "section_name": "Womens Everyday Basics",
                    "index_name": "Ladieswear",
                    "index_group_name": "Ladieswear",
                }
            ]
        ),
        "article_attributes": pd.DataFrame(
            [
                {
                    "article_id": "0000000001",
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                }
            ]
        ),
        "trend_attributes": _trend_attribute_rows(),
        "attribute_heat_series": pd.DataFrame(
            [
                {
                    "attr_id": "colour_group_name::Black",
                    "attr_type": "colour_group_name",
                    "attr_value": "Black",
                    "week_id": 10,
                    "heat": 100.0,
                    "actual_target_growth": 0.1,
                    "pred_target_growth": 0.2,
                    "pred_share_t1": 0.3,
                }
            ]
        ),
        "attribute_hierarchy_edges": pd.DataFrame(
            [
                {
                    "parent_attr_id": "garment_group_name::Jersey Basic",
                    "child_attr_id": "product_type_name::Shirt",
                    "parent_attr_type": "garment_group_name",
                    "parent_attr_value": "Jersey Basic",
                    "child_attr_type": "product_type_name",
                    "child_attr_value": "Shirt",
                    "relation_type": "contains",
                }
            ]
        ),
        "metrics_summary": pd.DataFrame(
            [
                {
                    "metric_domain": "trend",
                    "model_or_method": "lightgbm",
                    "split": "test",
                    "metric_name": "ndcg_at_10",
                    "metric_value": 0.5,
                    "display_order": 1,
                }
            ]
        ),
        "report_assets": pd.DataFrame(
            [
                {
                    "asset_name": "trend_board.svg",
                    "asset_type": "figure",
                    "title": "trend board",
                    "source_path": str(Path("outputs/reports/figures/trend_board.svg")),
                    "static_url": "/static/reports/trend_board.svg",
                    "description": "demo figure",
                }
            ]
        ),
    }


def _recommendation_row(rank: int) -> dict[str, object]:
    return {
        "case_id": CASE_ID,
        "customer_id": CUSTOMER_ID,
        "article_id": f"00000000{rank:02d}",
        "rank": rank,
        "score": 1.0 / rank,
        "is_hit": 1 if rank == 1 else 0,
        "candidate_sources": "popularity,trend_union",
    }


def _trend_attribute_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_week": 10,
                "target_week": 11,
                "attr_id": f"{attr_type}::demo",
                "attr_type": attr_type,
                "attr_value": "demo",
                "rank": 1,
                "heat_t": 100.0,
                "pred_share_t1": 0.2,
                "pred_target_growth": 0.3,
                "is_trend_eligible_t": 1,
            }
            for attr_type in CORE_TREND_ATTR_TYPES
        ]
    )
