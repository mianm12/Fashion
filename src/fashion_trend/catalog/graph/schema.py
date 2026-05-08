from __future__ import annotations

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

GRAPH_OUTPUT_FILENAMES: dict[str, str] = {
    "nodes_article": "nodes_article.csv",
    "nodes_attribute": "nodes_attribute.csv",
    "edges_article_attribute": "edges_article_attribute.csv",
    "edges_attribute_hierarchy": "edges_attribute_hierarchy.csv",
}


def make_attr_id(attr_type: str, attr_value: str) -> str:
    return f"{attr_type}::{attr_value}"


def make_article_node_id(article_id: str) -> str:
    return f"article_{article_id}"


def make_edge_type(attr_type: str) -> str:
    return "has_" + attr_type.removesuffix("_name")
