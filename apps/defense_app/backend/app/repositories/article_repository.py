from __future__ import annotations

import sqlite3

from app.repositories.sql import (
    like_literal,
    like_prefix_literal,
    row_to_dict,
    rows_to_dicts,
)

RowDict = dict[str, object]


class ArticleRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def search(self, query: str | None, limit: int) -> list[RowDict]:
        normalized_query = (query or "").strip()
        like_query = like_literal(normalized_query)
        prefix_query = like_prefix_literal(normalized_query)
        rows = self.connection.execute(
            """
            select article_id, prod_name, product_group_name, product_type_name,
                   garment_group_name, colour_group_name, graphical_appearance_name,
                   department_name, section_name, index_name, index_group_name
            from articles
            where ? = ''
               or article_id like ? escape '~'
               or coalesce(prod_name, '') like ? escape '~'
               or coalesce(product_type_name, '') like ? escape '~'
               or coalesce(colour_group_name, '') like ? escape '~'
            order by
                case
                    when article_id like ? escape '~' then 0
                    when coalesce(prod_name, '') like ? escape '~' then 1
                    else 2
                end,
                article_id asc
            limit ?
            """,
            (
                normalized_query,
                like_query,
                like_query,
                like_query,
                like_query,
                prefix_query,
                prefix_query,
                limit,
            ),
        ).fetchall()
        return rows_to_dicts(rows)

    def get(self, article_id: str) -> RowDict | None:
        row = self.connection.execute(
            """
            select article_id, prod_name, product_group_name, product_type_name,
                   garment_group_name, colour_group_name, graphical_appearance_name,
                   department_name, section_name, index_name, index_group_name
            from articles
            where article_id = ?
            """,
            (article_id,),
        ).fetchone()
        return row_to_dict(row)

    def attributes(self, article_id: str) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select attr_id, attr_type, attr_value
            from article_attributes
            where article_id = ?
            order by attr_type asc, attr_id asc
            """,
            (article_id,),
        ).fetchall()
        return rows_to_dicts(rows)
