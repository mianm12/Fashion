import sqlite3

from fashion_trend.foundation.paths import DATA_DIR, OUTPUT_DIR
from fashion_trend.presentation.contracts import (
    CORE_TREND_ATTR_TYPES,
    PRESENTATION_SCHEMA_VERSION,
)
from fashion_trend.presentation.paths import (
    DEFENSE_APP_DB_PATH,
    DEFENSE_APP_OUTPUT_DIR,
    DEFENSE_APP_STATIC_DIR,
    REPORTS_MANIFEST_PATH,
)
from fashion_trend.presentation.schema import apply_schema, read_schema_version


REQUIRED_TABLES = {
    "app_metadata",
    "demo_users",
    "user_profile_attributes",
    "recommendation_items",
    "recommendation_score_components",
    "articles",
    "article_attributes",
    "trend_attributes",
    "attribute_heat_series",
    "attribute_hierarchy_edges",
    "metrics_summary",
    "report_assets",
}


TEXT_IDENTITY_COLUMNS = {
    "demo_users": {"case_id", "customer_id"},
    "user_profile_attributes": {"case_id", "customer_id", "attr_id"},
    "recommendation_items": {"case_id", "customer_id", "article_id"},
    "recommendation_score_components": {"case_id", "article_id"},
    "articles": {"article_id"},
    "article_attributes": {"article_id", "attr_id"},
    "trend_attributes": {"attr_id"},
    "attribute_heat_series": {"attr_id"},
    "attribute_hierarchy_edges": {"parent_attr_id", "child_attr_id"},
}


def test_apply_schema_creates_required_tables():
    connection = sqlite3.connect(":memory:")
    apply_schema(connection)

    tables = {
        row[0]
        for row in connection.execute(
            "select name from sqlite_master where type = 'table'"
        )
    }

    assert REQUIRED_TABLES <= tables


def test_read_schema_version_returns_none_before_schema():
    connection = sqlite3.connect(":memory:")

    assert read_schema_version(connection) is None


def test_apply_schema_records_schema_version():
    connection = sqlite3.connect(":memory:")
    apply_schema(connection)

    assert read_schema_version(connection) == PRESENTATION_SCHEMA_VERSION


def test_read_schema_version_raises_when_metadata_table_is_malformed():
    connection = sqlite3.connect(":memory:")
    connection.execute("create table app_metadata(key TEXT PRIMARY KEY)")

    try:
        read_schema_version(connection)
    except sqlite3.OperationalError as error:
        assert "value" in str(error)
    else:
        raise AssertionError("read_schema_version should reject malformed metadata")


def test_apply_schema_rejects_existing_incompatible_schema_version():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "create table app_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "insert into app_metadata(key, value) values ('schema_version', 'old_version')"
    )

    try:
        apply_schema(connection)
    except ValueError as error:
        assert "old_version" in str(error)
        assert PRESENTATION_SCHEMA_VERSION in str(error)
    else:
        raise AssertionError("apply_schema should reject incompatible schema versions")

    assert read_schema_version(connection) == "old_version"


def test_apply_schema_is_idempotent_for_current_version():
    connection = sqlite3.connect(":memory:")
    apply_schema(connection)
    apply_schema(connection)

    assert read_schema_version(connection) == PRESENTATION_SCHEMA_VERSION


def test_core_trend_attr_types_match_design():
    assert CORE_TREND_ATTR_TYPES == (
        "colour_group_name",
        "product_type_name",
        "graphical_appearance_name",
        "garment_group_name",
    )


def test_presentation_paths_point_to_expected_roots():
    assert DEFENSE_APP_OUTPUT_DIR == OUTPUT_DIR / "defense_app"
    assert DEFENSE_APP_DB_PATH == DEFENSE_APP_OUTPUT_DIR / "fashion_demo.sqlite"
    assert DEFENSE_APP_STATIC_DIR == DEFENSE_APP_OUTPUT_DIR / "static"
    assert REPORTS_MANIFEST_PATH == OUTPUT_DIR / "reports" / "manifest.json"

    for path in (
        DEFENSE_APP_OUTPUT_DIR,
        DEFENSE_APP_DB_PATH,
        DEFENSE_APP_STATIC_DIR,
        REPORTS_MANIFEST_PATH,
    ):
        assert OUTPUT_DIR in path.parents

    assert DATA_DIR.name == "data"


def test_schema_declares_text_identity_columns():
    connection = sqlite3.connect(":memory:")
    apply_schema(connection)

    for table_name, column_names in TEXT_IDENTITY_COLUMNS.items():
        declared_types = {
            row[1]: row[2]
            for row in connection.execute(f"pragma table_info({table_name})")
        }

        assert column_names <= declared_types.keys()
        for column_name in column_names:
            assert declared_types[column_name].upper() == "TEXT"
