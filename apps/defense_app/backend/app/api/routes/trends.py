from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.database import get_database
from app.repositories.trend_repository import TrendRepository
from app.schemas.trends import (
    TrendEvidenceResponse,
    TrendListResponse,
    TrendSourceWeeksResponse,
    TrendSummaryResponse,
)

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("/source-weeks", response_model=TrendSourceWeeksResponse)
def get_trend_source_weeks(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    repository = TrendRepository(connection)
    return {
        "default_source_week": repository.default_source_week(),
        "items": repository.available_source_weeks(),
    }


@router.get("/summary", response_model=TrendSummaryResponse)
def get_trend_summary(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    source_week: int | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, object]:
    repository = TrendRepository(connection)
    effective_source_week = _effective_source_week(repository, source_week)
    return repository.summary(effective_source_week, limit)


@router.get("/evidence", response_model=TrendEvidenceResponse)
def get_trend_evidence(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    source_week: int | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, object]:
    repository = TrendRepository(connection)
    effective_source_week = _effective_source_week(repository, source_week)
    items = repository.list_core_trends(effective_source_week, 1)
    return {
        "source_week": effective_source_week,
        "target_week": items[0]["target_week"] if items else None,
        "distribution": repository.score_distribution(effective_source_week),
        "top_history": repository.top_history(effective_source_week, limit),
        "new_high_potential": repository.new_high_potential(
            effective_source_week,
            limit,
        ),
    }


@router.get("/detail", response_model=TrendListResponse)
def get_trend_detail(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    source_week: int | None = None,
    attr_type: str | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, object]:
    repository = TrendRepository(connection)
    effective_source_week = _effective_source_week(repository, source_week)
    items = repository.detail_rows(effective_source_week, attr_type, limit)
    target_week = items[0]["target_week"] if items else None
    return {
        "source_week": effective_source_week,
        "target_week": target_week,
        "items": items,
    }


@router.get("", response_model=TrendListResponse)
def get_trends(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    source_week: int | None = None,
    attr_type: str | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, object]:
    repository = TrendRepository(connection)
    effective_source_week = _effective_source_week(repository, source_week)
    items = repository.list_trends(effective_source_week, attr_type, limit)
    target_week = items[0]["target_week"] if items else None
    return {
        "source_week": effective_source_week,
        "target_week": target_week,
        "items": items,
    }


def _effective_source_week(
    repository: TrendRepository,
    source_week: int | None,
) -> int | None:
    return source_week if source_week is not None else repository.default_source_week()
