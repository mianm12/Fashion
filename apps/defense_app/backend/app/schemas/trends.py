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


class TrendSourceWeeksResponse(BaseModel):
    default_source_week: int | None = None
    items: list[int]


class TrendSummaryResponse(BaseModel):
    source_week: int | None = None
    target_week: int | None = None
    rising_attribute_count: int
    high_confidence_attribute_count: int
    top_k_average_pred_target_growth: float | None = None
    covered_article_count: int
    model_status: str


class TrendDistributionBucket(BaseModel):
    label: str
    count: int


class TrendHistoryPoint(BaseModel):
    attr_id: str
    attr_type: str
    attr_value: str
    week_id: int
    heat: float
    actual_target_growth: float | None = None
    pred_target_growth: float | None = None
    pred_share_t1: float | None = None


class TrendEvidenceResponse(BaseModel):
    source_week: int | None = None
    target_week: int | None = None
    distribution: list[TrendDistributionBucket]
    top_history: list[TrendHistoryPoint]
    new_high_potential: list[TrendItem]
