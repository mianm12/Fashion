from __future__ import annotations

from pydantic import BaseModel


class AttributeItem(BaseModel):
    attr_id: str
    attr_type: str
    attr_value: str


class ArticleItem(BaseModel):
    article_id: str
    prod_name: str | None = None
    product_group_name: str | None = None
    product_type_name: str | None = None
    garment_group_name: str | None = None
    colour_group_name: str | None = None
    graphical_appearance_name: str | None = None
    department_name: str | None = None
    section_name: str | None = None
    index_name: str | None = None
    index_group_name: str | None = None


class GraphNode(BaseModel):
    id: str
    label: str
    type: str


class GraphEdge(BaseModel):
    source: str
    target: str
    relation_type: str
