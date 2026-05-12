from __future__ import annotations

import sqlite3

from app.repositories.sql import rows_to_dicts

RowDict = dict[str, object]


class MetricsRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def list_metrics(
        self, domain: str | None = None, split: str | None = None
    ) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select metric_domain, model_or_method, split, metric_name,
                   metric_value, display_order
            from metrics_summary
            where (? is null or metric_domain = ?)
              and (? is null or split = ?)
            order by metric_domain asc, display_order asc,
                     model_or_method asc, metric_name asc
            """,
            (domain, domain, split, split),
        ).fetchall()
        return rows_to_dicts(rows)

    def list_default_split_metrics(self, domain: str) -> list[RowDict]:
        rows = self.connection.execute(
            """
            select metric_domain, model_or_method, split, metric_name,
                   metric_value, display_order
            from metrics_summary
            where metric_domain = ?
              and split in (?, ?)
            order by metric_domain asc, display_order asc,
                     model_or_method asc, metric_name asc
            """,
            (domain, "valid", "test"),
        ).fetchall()
        return rows_to_dicts(rows)
