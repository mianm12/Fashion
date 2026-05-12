from __future__ import annotations

import sqlite3

from app.repositories.sql import row_to_dict, rows_to_dicts

RowDict = dict[str, object]


class AttributeRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_identity(self, attr_id: str) -> RowDict | None:
        queries = (
            """
            select attr_id, attr_type, attr_value
            from trend_attributes
            where attr_id = ?
            order by source_week desc
            limit 1
            """,
            """
            select attr_id, attr_type, attr_value
            from attribute_heat_series
            where attr_id = ?
            order by week_id desc
            limit 1
            """,
            """
            select attr_id, attr_type, attr_value
            from article_attributes
            where attr_id = ?
            limit 1
            """,
        )
        for query in queries:
            row = self.connection.execute(query, (attr_id,)).fetchone()
            if row is not None:
                return row_to_dict(row)
        row = self.connection.execute(
            """
            select parent_attr_id as attr_id, parent_attr_type as attr_type,
                   parent_attr_value as attr_value
            from attribute_hierarchy_edges
            where parent_attr_id = ?
            limit 1
            """,
            (attr_id,),
        ).fetchone()
        if row is not None:
            return row_to_dict(row)
        row = self.connection.execute(
            """
            select child_attr_id as attr_id, child_attr_type as attr_type,
                   child_attr_value as attr_value
            from attribute_hierarchy_edges
            where child_attr_id = ?
            limit 1
            """,
            (attr_id,),
        ).fetchone()
        return row_to_dict(row)

    def heat_at_or_before(self, attr_id: str, source_week: int | None) -> RowDict | None:
        row = self.connection.execute(
            """
            select attr_id, attr_type, attr_value, week_id, heat,
                   actual_target_growth, pred_target_growth, pred_share_t1
            from attribute_heat_series
            where attr_id = ?
              and (? is null or week_id <= ?)
            order by week_id desc
            limit 1
            """,
            (attr_id, source_week, source_week),
        ).fetchone()
        return row_to_dict(row)

    def heat_series(
        self, attr_id: str, source_week: int | None, weeks: int
    ) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select attr_id, attr_type, attr_value, week_id, heat,
                   actual_target_growth, pred_target_growth, pred_share_t1
            from attribute_heat_series
            where attr_id = ?
              and (? is null or week_id <= ?)
            order by week_id desc
            limit ?
            """,
            (attr_id, source_week, source_week, weeks),
        ).fetchall()
        return list(reversed(rows_to_dicts(rows)))

    def related_articles(self, attr_id: str, limit: int) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select a.article_id, a.prod_name, a.product_group_name,
                   a.product_type_name, a.garment_group_name, a.colour_group_name,
                   a.graphical_appearance_name, a.department_name, a.section_name,
                   a.index_name, a.index_group_name
            from article_attributes aa
            join articles a on a.article_id = aa.article_id
            where aa.attr_id = ?
            order by a.article_id asc
            limit ?
            """,
            (attr_id, limit),
        ).fetchall()
        return rows_to_dicts(rows)

    def hierarchy_edges(self, attr_id: str) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select parent_attr_id, child_attr_id, parent_attr_type,
                   parent_attr_value, child_attr_type, child_attr_value,
                   relation_type
            from attribute_hierarchy_edges
            where parent_attr_id = ? or child_attr_id = ?
            order by parent_attr_id asc, child_attr_id asc, relation_type asc
            """,
            (attr_id, attr_id),
        ).fetchall()
        return rows_to_dicts(rows)
