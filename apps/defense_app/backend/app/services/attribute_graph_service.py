from __future__ import annotations

from app.schemas.common import GraphEdge, GraphNode


def build_attribute_graph(
    attribute: dict[str, object], edges: list[dict[str, object]]
) -> dict[str, object]:
    nodes: dict[str, GraphNode] = {
        str(attribute["attr_id"]): GraphNode(
            id=str(attribute["attr_id"]),
            label=str(attribute["attr_value"]),
            type=str(attribute["attr_type"]),
        )
    }
    graph_edges: list[GraphEdge] = []
    for edge in edges:
        parent_id = str(edge["parent_attr_id"])
        child_id = str(edge["child_attr_id"])
        nodes[parent_id] = GraphNode(
            id=parent_id,
            label=str(edge["parent_attr_value"]),
            type=str(edge["parent_attr_type"]),
        )
        nodes[child_id] = GraphNode(
            id=child_id,
            label=str(edge["child_attr_value"]),
            type=str(edge["child_attr_type"]),
        )
        graph_edges.append(
            GraphEdge(
                source=parent_id,
                target=child_id,
                relation_type=str(edge["relation_type"]),
            )
        )
    return {
        "attr_id": str(attribute["attr_id"]),
        "nodes": list(nodes.values()),
        "edges": graph_edges,
    }


def build_article_graph(
    article: dict[str, object], attributes: list[dict[str, object]]
) -> dict[str, object]:
    article_id = str(article["article_id"])
    article_node = GraphNode(
        id=f"article::{article_id}",
        label=str(article.get("prod_name") or article_id),
        type="article",
    )
    nodes = [
        GraphNode(
            id=str(attribute["attr_id"]),
            label=str(attribute["attr_value"]),
            type=str(attribute["attr_type"]),
        )
        for attribute in attributes
    ]
    edges = [
        GraphEdge(
            source=article_node.id,
            target=str(attribute["attr_id"]),
            relation_type="has_attribute",
        )
        for attribute in attributes
    ]
    return {"article": article_node, "nodes": nodes, "edges": edges}
