from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    articles,
    attributes,
    demo_users,
    metrics,
    recommendations,
    trends,
)

api_router = APIRouter(prefix="/api")
api_router.include_router(trends.router)
api_router.include_router(attributes.router)
api_router.include_router(articles.router)
api_router.include_router(demo_users.router)
api_router.include_router(recommendations.router)
api_router.include_router(metrics.router)
