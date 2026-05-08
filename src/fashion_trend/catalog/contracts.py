from __future__ import annotations

ARTICLE_ATTRIBUTE_EDGE_COLUMNS: tuple[str, ...] = (
    "article_id",
    "attr_id",
    "attr_type",
    "attr_value",
)

ARTICLE_ATTRIBUTE_EDGE_DTYPES: dict[str, str] = {
    "article_id": "string",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
}

ATTRIBUTE_NODE_COLUMNS: tuple[str, ...] = (
    "attr_id",
    "attr_type",
    "attr_value",
    "article_count",
    "is_core_attr",
)

ATTRIBUTE_NODE_DTYPES: dict[str, str] = {
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "article_count": "int64",
    "is_core_attr": "int64",
}

ATTRIBUTE_HIERARCHY_EDGE_COLUMNS: tuple[str, ...] = (
    "parent_attr_id",
    "child_attr_id",
    "parent_attr_type",
    "child_attr_type",
    "relation_type",
    "edge_weight",
)

ATTRIBUTE_HIERARCHY_EDGE_DTYPES: dict[str, str] = {
    "parent_attr_id": "string",
    "child_attr_id": "string",
    "parent_attr_type": "string",
    "child_attr_type": "string",
    "relation_type": "string",
    "edge_weight": "int64",
}
