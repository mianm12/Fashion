from __future__ import annotations

from pydantic import BaseModel

from app.schemas.common import ArticleItem, AttributeItem, GraphEdge, GraphNode


class ArticleSearchResponse(BaseModel):
    items: list[ArticleItem]


class ArticleDetailResponse(BaseModel):
    article: ArticleItem
    attributes: list[AttributeItem]


class ArticleGraphResponse(BaseModel):
    article: GraphNode
    nodes: list[GraphNode]
    edges: list[GraphEdge]
