from __future__ import annotations

import pandas as pd

from fashion_trend.catalog.articles import (
    ARTICLE_ID_COLUMN,
    ATTRIBUTE_COLUMNS,
    CORE_ATTRIBUTE_COLUMNS,
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_values,
)
from fashion_trend.catalog.graph.schema import (
    HIERARCHY_RELATIONS,
    LEVEL_BY_ATTRIBUTE,
    make_article_node_id,
    make_attr_id,
    make_edge_type,
)


def build_article_nodes(clean_articles: pd.DataFrame) -> pd.DataFrame:
    """从完整清洗商品表构建 `nodes_article.csv` 商品节点表。"""
    required_columns = ["article_id", "product_code", "prod_name"]
    validate_required_columns(
        clean_articles.columns.tolist(),
        required_columns,
        source_name="articles_clean.csv",
    )
    validate_no_missing_values(
        clean_articles,
        required_columns,
        source_name="articles_clean.csv",
    )
    article_nodes = clean_articles.loc[
        :, ["article_id", "product_code", "prod_name"]
    ].copy()
    article_nodes["article_id"] = article_nodes["article_id"].astype("string")
    article_nodes.insert(
        1,
        "article_node_id",
        article_nodes["article_id"].map(make_article_node_id),
    )
    return article_nodes[["article_id", "article_node_id", "product_code", "prod_name"]]


def build_attribute_nodes(clean_articles: pd.DataFrame) -> pd.DataFrame:
    """从完整清洗商品表聚合属性值并构建 `nodes_attribute.csv`。"""
    validate_required_columns(
        clean_articles.columns.tolist(),
        ATTRIBUTE_COLUMNS,
        source_name="articles_clean.csv",
    )
    validate_no_missing_values(
        clean_articles,
        ATTRIBUTE_COLUMNS,
        source_name="articles_clean.csv",
    )

    frames: list[pd.DataFrame] = []
    for attr_type in ATTRIBUTE_COLUMNS:
        counts = (
            clean_articles.groupby(attr_type, dropna=False)
            .size()
            .reset_index(name="article_count")
            .rename(columns={attr_type: "attr_value"})
        )
        counts["attr_type"] = attr_type
        frames.append(counts)

    attribute_nodes = pd.concat(frames, ignore_index=True)
    attribute_nodes["attr_value"] = attribute_nodes["attr_value"].astype("string")
    attribute_nodes["attr_id"] = attribute_nodes.apply(
        lambda row: make_attr_id(row["attr_type"], row["attr_value"]),
        axis=1,
    )
    attribute_nodes["attr_node_id"] = attribute_nodes["attr_id"]
    attribute_nodes["is_core_attr"] = (
        attribute_nodes["attr_type"].isin(CORE_ATTRIBUTE_COLUMNS).astype("int8")
    )
    attribute_nodes["level"] = attribute_nodes["attr_type"].map(LEVEL_BY_ATTRIBUTE)
    return attribute_nodes[
        [
            "attr_id",
            "attr_type",
            "attr_value",
            "attr_node_id",
            "article_count",
            "is_core_attr",
            "level",
        ]
    ].sort_values(["attr_type", "attr_value"], ignore_index=True)


def build_article_attribute_edges(clean_articles: pd.DataFrame) -> pd.DataFrame:
    """从商品属性列展开 `edges_article_attribute.csv` 商品-属性边表。"""
    required_columns = ["article_id", *ATTRIBUTE_COLUMNS]
    validate_required_columns(
        clean_articles.columns.tolist(),
        required_columns,
        source_name="articles_clean.csv",
    )
    validate_no_missing_values(
        clean_articles,
        required_columns,
        source_name="articles_clean.csv",
    )

    edges: list[pd.DataFrame] = []
    for attr_type in ATTRIBUTE_COLUMNS:
        edge_frame = clean_articles.loc[:, ["article_id", attr_type]].copy()
        edge_frame = edge_frame.rename(columns={attr_type: "attr_value"})
        edge_frame["article_id"] = edge_frame["article_id"].astype("string")
        edge_frame["attr_value"] = edge_frame["attr_value"].astype("string")
        edge_frame["article_node_id"] = edge_frame["article_id"].map(
            make_article_node_id
        )
        edge_frame["attr_type"] = attr_type
        edge_frame["attr_id"] = edge_frame["attr_value"].map(
            lambda value: make_attr_id(attr_type, value)
        )
        edge_frame["edge_type"] = make_edge_type(attr_type)
        edge_frame["edge_weight"] = 1.0
        edges.append(edge_frame)

    return pd.concat(edges, ignore_index=True)[
        [
            "article_id",
            "article_node_id",
            "attr_id",
            "attr_type",
            "attr_value",
            "edge_type",
            "edge_weight",
        ]
    ]


def build_attribute_hierarchy_edges(clean_articles: pd.DataFrame) -> pd.DataFrame:
    """构建属性层级边表。

    参数:
        clean_articles: 完整商品清洗表，必须包含层级关系涉及的属性列。

    返回:
        按父子属性类型、节点标识和关系类型聚合后的属性层级边表。

    边界:
        关系定义完全来自 `HIERARCHY_RELATIONS`；边权重是商品表中对应
        父子属性组合的出现次数，不在此处新增或推断其他关系。
    """
    required_columns = tuple(
        dict.fromkeys(
            column for relation in HIERARCHY_RELATIONS for column in relation[:2]
        )
    )
    validate_required_columns(
        clean_articles.columns.tolist(),
        required_columns,
        source_name="articles_clean.csv",
    )
    validate_no_missing_values(
        clean_articles,
        required_columns,
        source_name="articles_clean.csv",
    )

    hierarchy_edges: list[pd.DataFrame] = []
    for parent_attr_type, child_attr_type, relation_type in HIERARCHY_RELATIONS:
        relation_counts = (
            clean_articles.groupby([parent_attr_type, child_attr_type], dropna=False)
            .size()
            .reset_index(name="edge_weight")
        )
        relation_counts["parent_attr_type"] = parent_attr_type
        relation_counts["child_attr_type"] = child_attr_type
        relation_counts["parent_attr_id"] = relation_counts[parent_attr_type].map(
            lambda value: make_attr_id(parent_attr_type, str(value))
        )
        relation_counts["child_attr_id"] = relation_counts[child_attr_type].map(
            lambda value: make_attr_id(child_attr_type, str(value))
        )
        relation_counts["relation_type"] = relation_type
        hierarchy_edges.append(relation_counts)

    return pd.concat(hierarchy_edges, ignore_index=True)[
        [
            "parent_attr_id",
            "child_attr_id",
            "parent_attr_type",
            "child_attr_type",
            "relation_type",
            "edge_weight",
        ]
    ].sort_values(
        ["parent_attr_type", "child_attr_type", "parent_attr_id", "child_attr_id"],
        ignore_index=True,
    )


def validate_graph_references(
    nodes_article: pd.DataFrame,
    nodes_attribute: pd.DataFrame,
    edges_article_attribute: pd.DataFrame,
    edges_attribute_hierarchy: pd.DataFrame,
) -> None:
    """校验属性图边表只引用已经构建出的商品节点和属性节点。

    参数:
        nodes_article: 商品节点表。
        nodes_attribute: 属性节点表。
        edges_article_attribute: 商品-属性边表。
        edges_attribute_hierarchy: 属性层级边表。

    返回:
        None: 校验通过时不返回业务数据。

    异常:
        RuntimeError: 任一边表引用缺失节点时抛出。
    """
    article_node_ids = set(nodes_article["article_node_id"])
    attr_ids = set(nodes_attribute["attr_id"])

    missing_article_nodes = (
        set(edges_article_attribute["article_node_id"]) - article_node_ids
    )
    if missing_article_nodes:
        raise RuntimeError("商品-属性边引用了不存在的商品节点。")

    missing_attr_nodes = set(edges_article_attribute["attr_id"]) - attr_ids
    if missing_attr_nodes:
        raise RuntimeError("商品-属性边引用了不存在的属性节点。")

    missing_parent_nodes = set(edges_attribute_hierarchy["parent_attr_id"]) - attr_ids
    missing_child_nodes = set(edges_attribute_hierarchy["child_attr_id"]) - attr_ids
    if missing_parent_nodes or missing_child_nodes:
        raise RuntimeError("属性层级边引用了不存在的属性节点。")


def build_attribute_graph_frames(
    clean_articles: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """从完整商品清洗表构建属性图的四张内存表。

    参数:
        clean_articles: `articles_clean.csv` 对应的完整商品清洗表。

    返回:
        以图产物逻辑名为键、DataFrame 为值的四张属性图表。

    边界:
        先校验商品主键唯一，再分别构建商品节点、属性节点、商品-属性边和
        属性层级边；所有边引用必须通过 `validate_graph_references()`。
    """
    validate_required_columns(
        clean_articles.columns.tolist(),
        [ARTICLE_ID_COLUMN],
        source_name="articles_clean.csv",
    )
    validate_no_missing_values(
        clean_articles,
        [ARTICLE_ID_COLUMN],
        source_name="articles_clean.csv",
    )
    validate_unique_values(
        clean_articles,
        [ARTICLE_ID_COLUMN],
        source_name="articles_clean.csv",
    )
    nodes_article = build_article_nodes(clean_articles)
    nodes_attribute = build_attribute_nodes(clean_articles)
    edges_article_attribute = build_article_attribute_edges(clean_articles)
    edges_attribute_hierarchy = build_attribute_hierarchy_edges(clean_articles)
    validate_graph_references(
        nodes_article,
        nodes_attribute,
        edges_article_attribute,
        edges_attribute_hierarchy,
    )
    return {
        "nodes_article": nodes_article,
        "nodes_attribute": nodes_attribute,
        "edges_article_attribute": edges_article_attribute,
        "edges_attribute_hierarchy": edges_attribute_hierarchy,
    }
