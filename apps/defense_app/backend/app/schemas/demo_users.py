from __future__ import annotations

from pydantic import BaseModel


class DemoUserItem(BaseModel):
    case_id: str
    customer_id: str
    split: str
    cutoff_week: int
    label_week: int
    hit_count: int
    primary_tags: str
    profile_summary: str
    recommendation_summary: str


class DemoUserListResponse(BaseModel):
    items: list[DemoUserItem]


class UserProfileAttribute(BaseModel):
    case_id: str
    customer_id: str
    attr_id: str
    attr_type: str
    attr_value: str
    preference_score: float
    purchase_count: int
    last_purchase_week: int


class UserProfileResponse(BaseModel):
    case_id: str
    items: list[UserProfileAttribute]
