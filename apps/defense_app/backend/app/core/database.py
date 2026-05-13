from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fashion_trend.presentation.contracts import PRESENTATION_SCHEMA_VERSION

from app.core.config import get_database_path


class DefenseAppError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


def database_unavailable() -> DefenseAppError:
    return DefenseAppError(
        503,
        "database_unavailable",
        "展示库不存在，请先运行 src/18_build_defense_app_db.py",
    )


def schema_version_mismatch() -> DefenseAppError:
    return DefenseAppError(
        503,
        "schema_version_mismatch",
        "展示库 schema_version 与应用不兼容",
    )


def not_found(message: str) -> DefenseAppError:
    return DefenseAppError(404, "not_found", message)


def validation_error() -> DefenseAppError:
    return DefenseAppError(422, "validation_error", "请求参数不合法")


def open_database(database_path: Path | None = None) -> sqlite3.Connection:
    path = database_path or get_database_path()
    if not path.exists():
        raise database_unavailable()

    database_uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        _assert_schema_version(connection)
    except Exception:
        connection.close()
        raise
    return connection


def get_database() -> Iterator[sqlite3.Connection]:
    with database_connection() as connection:
        yield connection


@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    connection = open_database()
    try:
        yield connection
    finally:
        connection.close()


def _assert_schema_version(connection: sqlite3.Connection) -> None:
    try:
        row = connection.execute(
            "select value from app_metadata where key = ?", ("schema_version",)
        ).fetchone()
    except sqlite3.Error as exc:
        raise schema_version_mismatch() from exc
    if row is None or row["value"] != PRESENTATION_SCHEMA_VERSION:
        raise schema_version_mismatch()
