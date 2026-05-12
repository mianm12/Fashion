from __future__ import annotations

import sqlite3
from collections.abc import Iterable


def row_to_dict(row: sqlite3.Row | None) -> dict[str, object] | None:
    if row is None:
        return None
    return dict(row)


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, object]]:
    return [dict(row) for row in rows]


def like_literal(value: str) -> str:
    escaped = value.replace("~", "~~").replace("%", "~%").replace("_", "~_")
    return f"%{escaped}%"


def like_prefix_literal(value: str) -> str:
    escaped = value.replace("~", "~~").replace("%", "~%").replace("_", "~_")
    return f"{escaped}%"
