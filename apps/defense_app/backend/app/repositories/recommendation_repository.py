from __future__ import annotations

import sqlite3

from app.repositories.sql import row_to_dict, rows_to_dicts

RowDict = dict[str, object]


class RecommendationRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def recommendations(self, case_id: str) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select ri.case_id, ri.customer_id, ri.article_id, ri.rank, ri.score,
                   ri.is_hit, ri.candidate_sources,
                   a.prod_name, a.product_group_name, a.product_type_name,
                   a.garment_group_name, a.colour_group_name,
                   a.graphical_appearance_name, a.department_name, a.section_name,
                   a.index_name, a.index_group_name
            from recommendation_items ri
            join articles a on a.article_id = ri.article_id
            where ri.case_id = ?
            order by ri.rank asc
            """,
            (case_id,),
        ).fetchall()
        return rows_to_dicts(rows)

    def recommendation(self, case_id: str, article_id: str) -> RowDict | None:
        row = self.connection.execute(
            """
            select case_id, customer_id, article_id, rank, score, is_hit,
                   candidate_sources
            from recommendation_items
            where case_id = ? and article_id = ?
            """,
            (case_id, article_id),
        ).fetchone()
        return row_to_dict(row)

    def score_components(self, case_id: str, article_id: str) -> RowDict | None:
        row = self.connection.execute(
            """
            select pop_score, sim_score, trend_score, recent_score, final_score
            from recommendation_score_components
            where case_id = ? and article_id = ?
            """,
            (case_id, article_id),
        ).fetchone()
        return row_to_dict(row)
