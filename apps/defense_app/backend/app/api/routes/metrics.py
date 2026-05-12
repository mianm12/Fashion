from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.database import get_database
from app.repositories.metrics_repository import MetricsRepository
from app.schemas.metrics import MetricsListResponse, MetricsSummaryResponse
from app.services.metrics_service import group_metrics_by_domain

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/summary", response_model=MetricsSummaryResponse)
def get_metrics_summary(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    repository = MetricsRepository(connection)
    return {"groups": group_metrics_by_domain(repository.list_metrics())}


@router.get("/trend", response_model=MetricsListResponse)
def get_trend_metrics(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    split: str | None = None,
) -> dict[str, object]:
    repository = MetricsRepository(connection)
    normalized_split = _normalize_split(split)
    if normalized_split is None:
        return {"items": repository.list_default_split_metrics("trend")}
    return {"items": repository.list_metrics("trend", normalized_split)}


@router.get("/recommendation", response_model=MetricsListResponse)
def get_recommendation_metrics(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    split: str | None = None,
) -> dict[str, object]:
    repository = MetricsRepository(connection)
    normalized_split = _normalize_split(split)
    if normalized_split is None:
        return {"items": repository.list_default_split_metrics("recommendation")}
    return {"items": repository.list_metrics("recommendation", normalized_split)}


def _normalize_split(split: str | None) -> str | None:
    if split is None:
        return None
    normalized = split.strip()
    if not normalized:
        return None
    return normalized
