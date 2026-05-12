from __future__ import annotations

from pydantic import BaseModel


class MetricItem(BaseModel):
    metric_domain: str
    model_or_method: str
    split: str
    metric_name: str
    metric_value: float
    display_order: int


class MetricsListResponse(BaseModel):
    items: list[MetricItem]


class MetricsSummaryResponse(BaseModel):
    groups: dict[str, list[MetricItem]]
