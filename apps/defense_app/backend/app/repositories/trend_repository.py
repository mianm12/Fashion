from __future__ import annotations

import sqlite3

from app.repositories.sql import row_to_dict, rows_to_dicts

RowDict = dict[str, object]


class TrendRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def default_source_week(self) -> int | None:
        row = self.connection.execute(
            "select value from app_metadata where key = ?",
            ("default_source_week",),
        ).fetchone()
        if row is None or row["value"] in (None, ""):
            return None
        return int(row["value"])

    def list_trends(
        self, source_week: int | None, attr_type: str | None, limit: int
    ) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select source_week, target_week, attr_id, attr_type, attr_value, rank,
                   heat_t, pred_share_t1, pred_target_growth, is_trend_eligible_t
            from trend_attributes
            where (? is null or source_week = ?)
              and (? is null or attr_type = ?)
            order by rank asc
            limit ?
            """,
            (source_week, source_week, attr_type, attr_type, limit),
        ).fetchall()
        return rows_to_dicts(rows)

    def for_attribute(
        self, attr_id: str, source_week: int | None = None
    ) -> RowDict | None:
        if source_week is None:
            return self.latest_for_attribute(attr_id)
        row = self.connection.execute(
            """
            select source_week, target_week, attr_id, attr_type, attr_value, rank,
                   heat_t, pred_share_t1, pred_target_growth, is_trend_eligible_t
            from trend_attributes
            where attr_id = ? and source_week = ?
            order by rank asc
            limit 1
            """,
            (attr_id, source_week),
        ).fetchone()
        return row_to_dict(row)

    def latest_for_attribute(self, attr_id: str) -> RowDict | None:
        row = self.connection.execute(
            """
            select source_week, target_week, attr_id, attr_type, attr_value, rank,
                   heat_t, pred_share_t1, pred_target_growth, is_trend_eligible_t
            from trend_attributes
            where attr_id = ?
            order by source_week desc, rank asc
            limit 1
            """,
            (attr_id,),
        ).fetchone()
        return row_to_dict(row)

    def latest_for_attributes(self, attr_ids: list[str]) -> list[RowDict]:
        if not attr_ids:
            return []
        rows: list[RowDict] = []
        for attr_id in attr_ids:
            trend = self.latest_for_attribute(attr_id)
            if trend is not None:
                rows.append(trend)
        return rows
