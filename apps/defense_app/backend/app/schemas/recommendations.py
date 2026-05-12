from __future__ import annotations

from pydantic import BaseModel

from app.schemas.common import ArticleItem, AttributeItem
from app.schemas.demo_users import UserProfileAttribute
from app.schemas.trends import TrendItem


class RecommendationItem(BaseModel):
    case_id: str
    customer_id: str
    article_id: str
    rank: int
    score: float
    is_hit: bool
    candidate_sources: str
    article: ArticleItem


class RecommendationListResponse(BaseModel):
    case_id: str
    items: list[RecommendationItem]


class ScoreComponents(BaseModel):
    pop_score: float
    sim_score: float
    trend_score: float
    recent_score: float
    final_score: float


class RecommendationExplanationResponse(BaseModel):
    case_id: str
    article: ArticleItem
    user_profile: list[UserProfileAttribute]
    item_attributes: list[AttributeItem]
    score_components: ScoreComponents
    matching_trend_attributes: list[TrendItem]
