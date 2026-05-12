from __future__ import annotations

from pydantic import BaseModel


class TrendItem(BaseModel):
    source_week: int
    target_week: int
    attr_id: str
    attr_type: str
    attr_value: str
    rank: int
    heat_t: float
    pred_share_t1: float | None = None
    pred_target_growth: float | None = None
    is_trend_eligible_t: bool


class TrendListResponse(BaseModel):
    source_week: int | None = None
    target_week: int | None = None
    items: list[TrendItem]
