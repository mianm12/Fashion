from __future__ import annotations

# 属性列在属性图层级中的稳定角色定义。
LEVEL_BY_ATTRIBUTE: dict[str, str] = {
    "product_group_name": "parent",
    "product_type_name": "child",
    "perceived_colour_master_name": "parent",
    "colour_group_name": "child",
    "index_group_name": "parent",
    "index_name": "parent_child",
    "section_name": "parent_child",
    "department_name": "child",
    "garment_group_name": "flat",
    "graphical_appearance_name": "flat",
}

# 属性层级边的稳定父子列和关系类型定义。
HIERARCHY_RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("product_group_name", "product_type_name", "product_group_contains_type"),
    (
        "perceived_colour_master_name",
        "colour_group_name",
        "colour_master_contains_colour",
    ),
    ("index_group_name", "index_name", "index_group_contains_index"),
    ("index_name", "section_name", "index_contains_section"),
    ("section_name", "department_name", "section_contains_department"),
)

# 属性图内存表名到最终 CSV 文件名的发布契约。
GRAPH_OUTPUT_FILENAMES: dict[str, str] = {
    "nodes_article": "nodes_article.csv",
    "nodes_attribute": "nodes_attribute.csv",
    "edges_article_attribute": "edges_article_attribute.csv",
    "edges_attribute_hierarchy": "edges_attribute_hierarchy.csv",
}


def make_attr_id(attr_type: str, attr_value: str) -> str:
    """按属性类型和值生成稳定属性节点标识符。"""
    return f"{attr_type}::{attr_value}"


def make_article_node_id(article_id: str) -> str:
    """按商品 ID 生成稳定商品节点标识符。"""
    return f"article_{article_id}"


def make_edge_type(attr_type: str) -> str:
    """按属性类型生成商品指向属性的稳定边类型。"""
    return "has_" + attr_type.removesuffix("_name")
