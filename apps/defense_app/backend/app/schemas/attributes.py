from __future__ import annotations

from pydantic import BaseModel

from app.schemas.common import ArticleItem, GraphEdge, GraphNode
from app.schemas.trends import TrendItem


class HeatSeriesItem(BaseModel):
    attr_id: str
    attr_type: str
    attr_value: str
    week_id: int
    heat: float
    actual_target_growth: float | None = None
    pred_target_growth: float | None = None
    pred_share_t1: float | None = None


class AttributeDetailResponse(BaseModel):
    attr_id: str
    attr_type: str
    attr_value: str
    latest_trend: TrendItem | None = None
    latest_heat: HeatSeriesItem | None = None


class HeatSeriesResponse(BaseModel):
    attr_id: str
    points: list[HeatSeriesItem]


class AttributeArticlesResponse(BaseModel):
    attr_id: str
    items: list[ArticleItem]


class AttributeGraphResponse(BaseModel):
    attr_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
