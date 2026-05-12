from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.database import get_database, not_found, validation_error
from app.repositories.article_repository import ArticleRepository
from app.schemas.articles import (
    ArticleDetailResponse,
    ArticleGraphResponse,
    ArticleSearchResponse,
)
from app.services.attribute_graph_service import build_article_graph

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("/search", response_model=ArticleSearchResponse)
def search_articles(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    q: Annotated[str, Query(min_length=1)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, object]:
    if not q.strip():
        raise validation_error()
    repository = ArticleRepository(connection)
    return {"items": repository.search(q, limit)}


@router.get("/{article_id}", response_model=ArticleDetailResponse)
def get_article(
    article_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    repository = ArticleRepository(connection)
    article = repository.get(article_id)
    if article is None:
        raise not_found("未找到指定商品")
    return {"article": article, "attributes": repository.attributes(article_id)}


@router.get("/{article_id}/graph", response_model=ArticleGraphResponse)
def get_article_graph(
    article_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    repository = ArticleRepository(connection)
    article = repository.get(article_id)
    if article is None:
        raise not_found("未找到指定商品")
    return build_article_graph(article, repository.attributes(article_id))
