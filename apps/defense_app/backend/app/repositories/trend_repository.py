from __future__ import annotations

import sqlite3

from fashion_trend.presentation.contracts import CORE_TREND_ATTR_TYPES

from app.repositories.sql import row_to_dict, rows_to_dicts

RowDict = dict[str, object]
HIGH_CONFIDENCE_GROWTH_THRESHOLD = 0.2


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

    def available_source_weeks(self) -> list[int]:
        rows = self.connection.execute(
            """
            select distinct source_week
            from trend_attributes
            order by source_week asc
            """
        ).fetchall()
        return [int(row["source_week"]) for row in rows]

    def list_trends(
        self, source_week: int | None, attr_type: str | None, limit: int
    ) -> list[RowDict]:
        if attr_type is None:
            return self.list_core_trends(source_week, limit)

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

    def list_core_trends(self, source_week: int | None, limit: int) -> list[RowDict]:
        attr_type_placeholders = ", ".join("?" for _ in CORE_TREND_ATTR_TYPES)
        attr_type_order = " ".join(
            f"when attr_type = ? then {index}"
            for index, attr_type in enumerate(CORE_TREND_ATTR_TYPES)
        )
        rows = self.connection.execute(
            f"""
            select source_week, target_week, attr_id, attr_type, attr_value, rank,
                   heat_t, pred_share_t1, pred_target_growth, is_trend_eligible_t
            from (
                select source_week, target_week, attr_id, attr_type, attr_value, rank,
                       heat_t, pred_share_t1, pred_target_growth,
                       is_trend_eligible_t,
                       row_number() over (
                           partition by attr_type
                           order by rank asc
                       ) as type_rank
                from trend_attributes
                where (? is null or source_week = ?)
                  and attr_type in ({attr_type_placeholders})
            )
            where type_rank <= ?
            order by case {attr_type_order} else 999 end, rank asc
            """,
            (
                source_week,
                source_week,
                *CORE_TREND_ATTR_TYPES,
                limit,
                *CORE_TREND_ATTR_TYPES,
            ),
        ).fetchall()
        return rows_to_dicts(rows)

    def summary(self, source_week: int | None, limit: int) -> RowDict:
        items = self.list_core_trends(source_week, limit)
        growth_values = [
            float(item["pred_target_growth"])
            for item in items
            if item["pred_target_growth"] is not None
        ]
        rising_count = sum(value > 0 for value in growth_values)
        high_confidence_count = sum(
            value >= HIGH_CONFIDENCE_GROWTH_THRESHOLD for value in growth_values
        )
        return {
            "source_week": source_week,
            "target_week": _target_week(items),
            "rising_attribute_count": rising_count,
            "high_confidence_attribute_count": high_confidence_count,
            "top_k_average_pred_target_growth": (
                sum(growth_values) / len(growth_values) if growth_values else None
            ),
            "covered_article_count": self._covered_article_count(
                [str(item["attr_id"]) for item in items]
            ),
            "model_status": "LightGBM stable",
        }

    def score_distribution(self, source_week: int | None) -> list[RowDict]:
        buckets = {
            "<0%": 0,
            "0-10%": 0,
            "10-20%": 0,
            "20-30%": 0,
            "30-50%": 0,
            "50%+": 0,
            "--": 0,
        }
        for item in self.list_core_trends(source_week, 50):
            value = item["pred_target_growth"]
            if value is None:
                buckets["--"] += 1
                continue
            numeric = float(value)
            if numeric < 0:
                buckets["<0%"] += 1
            elif numeric < 0.1:
                buckets["0-10%"] += 1
            elif numeric < 0.2:
                buckets["10-20%"] += 1
            elif numeric < 0.3:
                buckets["20-30%"] += 1
            elif numeric < 0.5:
                buckets["30-50%"] += 1
            else:
                buckets["50%+"] += 1
        return [
            {"label": label, "count": count}
            for label, count in buckets.items()
            if count > 0
        ]

    def top_history(self, source_week: int | None, limit: int) -> list[RowDict]:
        top_items = self.list_core_trends(source_week, 1)[:limit]
        if not top_items:
            return []
        attr_ids = [str(item["attr_id"]) for item in top_items]
        attr_meta = {
            str(item["attr_id"]): {
                "attr_type": str(item["attr_type"]),
                "attr_value": str(item["attr_value"]),
            }
            for item in top_items
        }
        placeholders = ", ".join("?" for _ in attr_ids)
        rows = self.connection.execute(
            f"""
            select attr_id, attr_type, attr_value, week_id, heat,
                   actual_target_growth, pred_target_growth, pred_share_t1
            from attribute_heat_series
            where attr_id in ({placeholders})
              and (? is null or week_id <= ?)
            order by attr_id asc, week_id asc
            """,
            (*attr_ids, source_week, source_week),
        ).fetchall()
        history = rows_to_dicts(rows)
        for row in history:
            meta = attr_meta.get(str(row["attr_id"]))
            if meta is None:
                continue
            row["attr_type"] = meta["attr_type"]
            row["attr_value"] = meta["attr_value"]
        return history

    def new_high_potential(
        self, source_week: int | None, limit: int
    ) -> list[RowDict]:
        if source_week is None:
            return []
        current = self.list_core_trends(source_week, limit)
        new_items: list[RowDict] = []
        for item in current:
            previous = self.connection.execute(
                """
                select 1
                from trend_attributes
                where source_week < ?
                  and attr_id = ?
                  and rank <= ?
                limit 1
                """,
                (source_week, item["attr_id"], limit),
            ).fetchone()
            if previous is None:
                new_items.append(item)
        return new_items[:limit]

    def detail_rows(
        self, source_week: int | None, attr_type: str | None, limit: int
    ) -> list[RowDict]:
        return self.list_trends(source_week, attr_type, limit)

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

    def _covered_article_count(self, attr_ids: list[str]) -> int:
        if not attr_ids:
            return 0
        placeholders = ", ".join("?" for _ in attr_ids)
        row = self.connection.execute(
            f"""
            select count(distinct article_id) as article_count
            from article_attributes
            where attr_id in ({placeholders})
            """,
            attr_ids,
        ).fetchone()
        return 0 if row is None else int(row["article_count"])


def _target_week(items: list[RowDict]) -> int | None:
    if not items:
        return None
    value = items[0]["target_week"]
    return None if value is None else int(value)
