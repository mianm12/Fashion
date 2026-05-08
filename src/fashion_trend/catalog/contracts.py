from __future__ import annotations

# 从 `edges_article_attribute.csv` 读取和下游消费的列契约。
ARTICLE_ATTRIBUTE_EDGE_COLUMNS: tuple[str, ...] = (
    "article_id",
    "attr_id",
    "attr_type",
    "attr_value",
)

# 读取 `edges_article_attribute.csv` 时使用的稳定 dtype 契约。
ARTICLE_ATTRIBUTE_EDGE_DTYPES: dict[str, str] = {
    "article_id": "string",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
}

# 从 `nodes_attribute.csv` 读取和下游消费的列契约。
ATTRIBUTE_NODE_COLUMNS: tuple[str, ...] = (
    "attr_id",
    "attr_type",
    "attr_value",
    "article_count",
    "is_core_attr",
)

# 读取 `nodes_attribute.csv` 时使用的稳定 dtype 契约。
ATTRIBUTE_NODE_DTYPES: dict[str, str] = {
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "article_count": "int64",
    "is_core_attr": "int64",
}

# `edges_attribute_hierarchy.csv` 当前完整产物列契约。
ATTRIBUTE_HIERARCHY_EDGE_COLUMNS: tuple[str, ...] = (
    "parent_attr_id",
    "child_attr_id",
    "parent_attr_type",
    "child_attr_type",
    "relation_type",
    "edge_weight",
)

# 读取 `edges_attribute_hierarchy.csv` 时使用的稳定 dtype 契约。
ATTRIBUTE_HIERARCHY_EDGE_DTYPES: dict[str, str] = {
    "parent_attr_id": "string",
    "child_attr_id": "string",
    "parent_attr_type": "string",
    "child_attr_type": "string",
    "relation_type": "string",
    "edge_weight": "int64",
}
