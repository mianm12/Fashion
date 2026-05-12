from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.database import get_database, not_found
from app.repositories.attribute_repository import AttributeRepository
from app.repositories.trend_repository import TrendRepository
from app.schemas.attributes import (
    AttributeArticlesResponse,
    AttributeDetailResponse,
    AttributeGraphResponse,
    HeatSeriesResponse,
)
from app.services.attribute_graph_service import build_attribute_graph

router = APIRouter(prefix="/attributes", tags=["attributes"])


@router.get("/{attr_id}", response_model=AttributeDetailResponse)
def get_attribute(
    attr_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    source_week: int | None = None,
) -> dict[str, object]:
    attributes = AttributeRepository(connection)
    attribute = attributes.get_identity(attr_id)
    if attribute is None:
        raise not_found("未找到指定属性")

    trends = TrendRepository(connection)
    effective_source_week = (
        source_week if source_week is not None else trends.default_source_week()
    )
    return {
        **attribute,
        "latest_trend": trends.for_attribute(attr_id, effective_source_week),
        "latest_heat": attributes.heat_at_or_before(attr_id, effective_source_week),
    }


@router.get("/{attr_id}/heat-series", response_model=HeatSeriesResponse)
def get_attribute_heat_series(
    attr_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    source_week: int | None = None,
    weeks: Annotated[int, Query(ge=1, le=16)] = 8,
) -> dict[str, object]:
    attributes = AttributeRepository(connection)
    if attributes.get_identity(attr_id) is None:
        raise not_found("未找到指定属性")
    effective_source_week = (
        source_week
        if source_week is not None
        else TrendRepository(connection).default_source_week()
    )
    return {
        "attr_id": attr_id,
        "points": attributes.heat_series(attr_id, effective_source_week, weeks),
    }


@router.get("/{attr_id}/articles", response_model=AttributeArticlesResponse)
def get_attribute_articles(
    attr_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict[str, object]:
    attributes = AttributeRepository(connection)
    if attributes.get_identity(attr_id) is None:
        raise not_found("未找到指定属性")
    return {"attr_id": attr_id, "items": attributes.related_articles(attr_id, limit)}


@router.get("/{attr_id}/graph", response_model=AttributeGraphResponse)
def get_attribute_graph(
    attr_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    attributes = AttributeRepository(connection)
    attribute = attributes.get_identity(attr_id)
    if attribute is None:
        raise not_found("未找到指定属性")
    return build_attribute_graph(attribute, attributes.hierarchy_edges(attr_id))
