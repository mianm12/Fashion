from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from fashion_trend.catalog.articles import (
    ARTICLE_ID_COLUMN,
    ATTRIBUTE_COLUMNS,
    CORE_ATTRIBUTE_COLUMNS,
    validate_no_missing_values,
    validate_required_columns,
    validate_unique_values,
)
from fashion_trend.foundation.io import remove_file_if_exists

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
ARTICLE_ATTRIBUTE_EDGE_READER_COLUMNS: tuple[str, ...] = (
    "article_id",
    "attr_id",
    "attr_type",
    "attr_value",
)
ARTICLE_ATTRIBUTE_EDGE_READER_DTYPES: dict[str, str] = {
    "article_id": "string",
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
}
ATTRIBUTE_NODE_READER_COLUMNS: tuple[str, ...] = (
    "attr_id",
    "attr_type",
    "attr_value",
    "article_count",
    "is_core_attr",
)
ATTRIBUTE_NODE_READER_DTYPES: dict[str, str] = {
    "attr_id": "string",
    "attr_type": "string",
    "attr_value": "string",
    "article_count": "int64",
    "is_core_attr": "int64",
}


def make_attr_id(attr_type: str, attr_value: str) -> str:
    return f"{attr_type}::{attr_value}"


def make_article_node_id(article_id: str) -> str:
    return f"article_{article_id}"


def make_edge_type(attr_type: str) -> str:
    return "has_" + attr_type.removesuffix("_name")


def build_article_nodes(clean_articles: pd.DataFrame) -> pd.DataFrame:
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


def read_clean_articles(clean_articles_path: Path) -> pd.DataFrame:
    if not clean_articles_path.exists():
        raise FileNotFoundError(f"商品 clean 文件不存在: {clean_articles_path}")

    return pd.read_csv(
        clean_articles_path,
        dtype={
            "article_id": "string",
            "product_code": "string",
        },
    )


def read_article_attribute_edges(article_attribute_edges_path: Path) -> pd.DataFrame:
    if not article_attribute_edges_path.exists():
        raise FileNotFoundError(f"商品-属性边表不存在: {article_attribute_edges_path}")

    try:
        header = pd.read_csv(article_attribute_edges_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"无法读取商品-属性边表: {article_attribute_edges_path}"
        ) from exc

    missing_columns = sorted(
        set(ARTICLE_ATTRIBUTE_EDGE_READER_COLUMNS) - set(header.columns)
    )
    if missing_columns:
        raise ValueError(
            "商品-属性边表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {article_attribute_edges_path}"
        )

    try:
        return pd.read_csv(
            article_attribute_edges_path,
            usecols=list(ARTICLE_ATTRIBUTE_EDGE_READER_COLUMNS),
            dtype=ARTICLE_ATTRIBUTE_EDGE_READER_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"无法读取商品-属性边表: {article_attribute_edges_path}"
        ) from exc


def read_attribute_nodes(attribute_nodes_path: Path) -> pd.DataFrame:
    if not attribute_nodes_path.exists():
        raise FileNotFoundError(f"属性节点表不存在: {attribute_nodes_path}")

    try:
        header = pd.read_csv(attribute_nodes_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性节点表: {attribute_nodes_path}") from exc

    missing_columns = sorted(set(ATTRIBUTE_NODE_READER_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "属性节点表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_nodes_path}"
        )

    try:
        return pd.read_csv(
            attribute_nodes_path,
            usecols=list(ATTRIBUTE_NODE_READER_COLUMNS),
            dtype=ATTRIBUTE_NODE_READER_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性节点表: {attribute_nodes_path}") from exc


def validate_graph_references(
    nodes_article: pd.DataFrame,
    nodes_attribute: pd.DataFrame,
    edges_article_attribute: pd.DataFrame,
    edges_attribute_hierarchy: pd.DataFrame,
) -> None:
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


def cleanup_graph_publish_files(paths_by_name: dict[str, Path]) -> None:
    for path in paths_by_name.values():
        remove_file_if_exists(path)


def rollback_graph_outputs(
    output_paths: dict[str, Path],
    backup_paths: dict[str, Path],
) -> None:
    for graph_name, output_path in output_paths.items():
        if output_path.is_file():
            output_path.unlink()

        backup_path = backup_paths[graph_name]
        if backup_path.exists():
            backup_path.replace(output_path)


def write_graph_frame_temp(dataframe: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(tmp_output_path, index=False, quoting=csv.QUOTE_ALL)
    return tmp_output_path


def publish_graph_frames(
    graph_frames: dict[str, pd.DataFrame],
    graph_dir: Path,
) -> None:
    output_paths = {
        graph_name: graph_dir / GRAPH_OUTPUT_FILENAMES[graph_name]
        for graph_name in graph_frames
    }
    temp_paths = {
        graph_name: output_path.with_suffix(output_path.suffix + ".tmp")
        for graph_name, output_path in output_paths.items()
    }
    backup_paths = {
        graph_name: output_path.with_suffix(output_path.suffix + ".bak")
        for graph_name, output_path in output_paths.items()
    }

    try:
        for graph_name, graph_frame in graph_frames.items():
            temp_paths[graph_name] = write_graph_frame_temp(
                graph_frame, output_paths[graph_name]
            )

        for graph_name, output_path in output_paths.items():
            remove_file_if_exists(backup_paths[graph_name])
            if output_path.is_file():
                output_path.replace(backup_paths[graph_name])

        for graph_name, temp_path in temp_paths.items():
            temp_path.replace(output_paths[graph_name])
    except Exception:
        try:
            rollback_graph_outputs(output_paths, backup_paths)
        finally:
            cleanup_graph_publish_files(temp_paths)
            cleanup_graph_publish_files(backup_paths)
        raise

    cleanup_graph_publish_files(temp_paths)
    cleanup_graph_publish_files(backup_paths)


def build_attribute_graph_frames(
    clean_articles: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
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


def build_attribute_graph_files(
    clean_articles_path: Path,
    graph_dir: Path,
) -> dict[str, int]:
    clean_articles = read_clean_articles(clean_articles_path)
    graph_frames = build_attribute_graph_frames(clean_articles)
    graph_dir.mkdir(parents=True, exist_ok=True)

    publish_graph_frames(graph_frames, graph_dir)
    return {
        graph_name: len(graph_frame) for graph_name, graph_frame in graph_frames.items()
    }
