from __future__ import annotations

import sqlite3

from app.repositories.sql import like_literal, row_to_dict, rows_to_dicts

RowDict = dict[str, object]


class DemoUserRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def list_users(
        self, query: str | None, tag: str | None, limit: int
    ) -> list[RowDict]:
        normalized_query = (query or "").strip()
        normalized_tag = (tag or "").strip()
        like_query = like_literal(normalized_query)
        like_tag = like_literal(normalized_tag)
        rows = self.connection.execute(
            """
            select case_id, customer_id, split, cutoff_week, label_week, hit_count,
                   primary_tags, profile_summary, recommendation_summary
            from demo_users
            where (
                ? = ''
                or case_id like ? escape '~'
                or customer_id like ? escape '~'
            )
              and (? = '' or primary_tags like ? escape '~')
            order by hit_count desc, case_id asc
            limit ?
            """,
            (
                normalized_query,
                like_query,
                like_query,
                normalized_tag,
                like_tag,
                limit,
            ),
        ).fetchall()
        return rows_to_dicts(rows)

    def get(self, case_id: str) -> RowDict | None:
        row = self.connection.execute(
            """
            select case_id, customer_id, split, cutoff_week, label_week, hit_count,
                   primary_tags, profile_summary, recommendation_summary
            from demo_users
            where case_id = ?
            """,
            (case_id,),
        ).fetchone()
        return row_to_dict(row)

    def profile(self, case_id: str) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select case_id, customer_id, attr_id, attr_type, attr_value,
                   preference_score, purchase_count, last_purchase_week
            from user_profile_attributes
            where case_id = ?
            order by preference_score desc, attr_id asc
            """,
            (case_id,),
        ).fetchall()
        return rows_to_dicts(rows)
