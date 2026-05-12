from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.database import get_database, not_found
from app.repositories.article_repository import ArticleRepository
from app.repositories.demo_user_repository import DemoUserRepository
from app.repositories.recommendation_repository import RecommendationRepository
from app.repositories.trend_repository import TrendRepository
from app.schemas.recommendations import (
    RecommendationExplanationResponse,
    RecommendationListResponse,
)
from app.services.recommendation_explanation_service import (
    build_recommendation_explanation,
)

router = APIRouter(prefix="/demo-users", tags=["recommendations"])


@router.get("/{case_id}/recommendations", response_model=RecommendationListResponse)
def get_demo_user_recommendations(
    case_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    demo_users = DemoUserRepository(connection)
    if demo_users.get(case_id) is None:
        raise not_found("未找到指定演示用户")

    recommendations = RecommendationRepository(connection)
    items = [
        _build_recommendation_item(item)
        for item in recommendations.recommendations(case_id)
    ]
    return {"case_id": case_id, "items": items}


@router.get(
    "/{case_id}/recommendations/{article_id}/explanation",
    response_model=RecommendationExplanationResponse,
)
def get_recommendation_explanation(
    case_id: str,
    article_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    return build_recommendation_explanation(
        case_id,
        article_id,
        DemoUserRepository(connection),
        ArticleRepository(connection),
        RecommendationRepository(connection),
        TrendRepository(connection),
    )


def _build_recommendation_item(item: dict[str, object]) -> dict[str, object]:
    article_fields = (
        "article_id",
        "prod_name",
        "product_group_name",
        "product_type_name",
        "garment_group_name",
        "colour_group_name",
        "graphical_appearance_name",
        "department_name",
        "section_name",
        "index_name",
        "index_group_name",
    )
    recommendation_fields = (
        "case_id",
        "customer_id",
        "article_id",
        "rank",
        "score",
        "is_hit",
        "candidate_sources",
    )
    return {
        **{field: item[field] for field in recommendation_fields},
        "article": {field: item[field] for field in article_fields},
    }
