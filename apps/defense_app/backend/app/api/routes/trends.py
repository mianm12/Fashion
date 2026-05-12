from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.database import get_database
from app.repositories.trend_repository import TrendRepository
from app.schemas.trends import TrendListResponse

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("", response_model=TrendListResponse)
def get_trends(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    source_week: int | None = None,
    attr_type: str | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, object]:
    repository = TrendRepository(connection)
    effective_source_week = (
        source_week if source_week is not None else repository.default_source_week()
    )
    items = repository.list_trends(effective_source_week, attr_type, limit)
    target_week = items[0]["target_week"] if items else None
    return {
        "source_week": effective_source_week,
        "target_week": target_week,
        "items": items,
    }
