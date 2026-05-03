from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

ARTICLE_ID_COLUMN = "article_id"
PRODUCT_CODE_COLUMN = "product_code"

CORE_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "product_group_name",
    "product_type_name",
    "garment_group_name",
    "colour_group_name",
    "graphical_appearance_name",
)

HIERARCHY_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "perceived_colour_master_name",
    "index_group_name",
    "index_name",
    "section_name",
    "department_name",
)

MVP_ARTICLE_COLUMNS: tuple[str, ...] = (
    "article_id",
    "product_code",
    "prod_name",
    *CORE_ATTRIBUTE_COLUMNS,
)

CLEAN_ARTICLE_COLUMNS: tuple[str, ...] = (
    *MVP_ARTICLE_COLUMNS,
    *HIERARCHY_ATTRIBUTE_COLUMNS,
)

ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    *CORE_ATTRIBUTE_COLUMNS,
    *HIERARCHY_ATTRIBUTE_COLUMNS,
)

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
    ("perceived_colour_master_name", "colour_group_name", "colour_master_contains_colour"),
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


def validate_required_columns(
    actual_columns: Sequence[str],
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(actual_columns))
    if missing_columns:
        raise ValueError(f"{source_name} 缺少必要字段: " + ", ".join(missing_columns))


def validate_no_missing_values(
    articles: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    missing_columns = [
        column for column in required_columns if int(articles[column].isna().sum()) > 0
    ]
    if missing_columns:
        raise ValueError(f"{source_name} 存在缺失值字段: " + ", ".join(missing_columns))


def normalize_article_identifiers(articles: pd.DataFrame) -> pd.DataFrame:
    normalized = articles.copy()
    normalized[ARTICLE_ID_COLUMN] = normalized[ARTICLE_ID_COLUMN].astype("string")
    normalized[PRODUCT_CODE_COLUMN] = normalized[PRODUCT_CODE_COLUMN].astype("string")
    return normalized


def build_clean_article_frames(raw_articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_required_columns(
        raw_articles.columns.tolist(),
        CLEAN_ARTICLE_COLUMNS,
        source_name="原始 articles.csv",
    )
    validate_no_missing_values(
        raw_articles,
        CLEAN_ARTICLE_COLUMNS,
        source_name="原始 articles.csv",
    )

    normalized_articles = normalize_article_identifiers(raw_articles)
    mvp_articles = normalized_articles.loc[:, list(MVP_ARTICLE_COLUMNS)].copy()
    clean_articles = normalized_articles.loc[:, list(CLEAN_ARTICLE_COLUMNS)].copy()
    return mvp_articles, clean_articles


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
    attribute_nodes["is_core_attr"] = attribute_nodes["attr_type"].isin(
        CORE_ATTRIBUTE_COLUMNS
    ).astype("int8")
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
        edge_frame["article_node_id"] = edge_frame["article_id"].map(make_article_node_id)
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


def read_articles_csv(raw_articles_path: Path) -> pd.DataFrame:
    if not raw_articles_path.exists():
        raise FileNotFoundError(f"原始商品文件不存在: {raw_articles_path}")

    return pd.read_csv(
        raw_articles_path,
        usecols=list(CLEAN_ARTICLE_COLUMNS),
        dtype={
            "article_id": "string",
            "product_code": "string",
        },
    )


def write_csv_temp(dataframe: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(tmp_output_path, index=False)
    return tmp_output_path


def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


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


def validate_graph_references(
    nodes_article: pd.DataFrame,
    nodes_attribute: pd.DataFrame,
    edges_article_attribute: pd.DataFrame,
    edges_attribute_hierarchy: pd.DataFrame,
) -> None:
    article_node_ids = set(nodes_article["article_node_id"])
    attr_ids = set(nodes_attribute["attr_id"])

    missing_article_nodes = set(edges_article_attribute["article_node_id"]) - article_node_ids
    if missing_article_nodes:
        raise RuntimeError("商品-属性边引用了不存在的商品节点。")

    missing_attr_nodes = set(edges_article_attribute["attr_id"]) - attr_ids
    if missing_attr_nodes:
        raise RuntimeError("商品-属性边引用了不存在的属性节点。")

    missing_parent_nodes = set(edges_attribute_hierarchy["parent_attr_id"]) - attr_ids
    missing_child_nodes = set(edges_attribute_hierarchy["child_attr_id"]) - attr_ids
    if missing_parent_nodes or missing_child_nodes:
        raise RuntimeError("属性层级边引用了不存在的属性节点。")


def build_attribute_graph_frames(clean_articles: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
            temp_paths[graph_name] = write_csv_temp(graph_frame, output_paths[graph_name])

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


def build_attribute_graph_files(
    clean_articles_path: Path,
    graph_dir: Path,
) -> dict[str, int]:
    clean_articles = read_clean_articles(clean_articles_path)
    graph_frames = build_attribute_graph_frames(clean_articles)
    graph_dir.mkdir(parents=True, exist_ok=True)

    publish_graph_frames(graph_frames, graph_dir)
    return {graph_name: len(graph_frame) for graph_name, graph_frame in graph_frames.items()}


def restore_mvp_output(
    mvp_output_path: Path,
    mvp_backup_path: Path,
    mvp_had_previous_output: bool,
) -> None:
    remove_file_if_exists(mvp_output_path)
    if mvp_had_previous_output:
        mvp_backup_path.replace(mvp_output_path)


def clean_articles_file(
    raw_articles_path: Path,
    mvp_output_path: Path,
    clean_output_path: Path,
) -> int:
    raw_articles = read_articles_csv(raw_articles_path)
    mvp_articles, clean_articles = build_clean_article_frames(raw_articles)

    mvp_tmp_output_path = mvp_output_path.with_suffix(mvp_output_path.suffix + ".tmp")
    clean_tmp_output_path = clean_output_path.with_suffix(clean_output_path.suffix + ".tmp")
    mvp_backup_path = mvp_output_path.with_suffix(mvp_output_path.suffix + ".bak")
    mvp_had_previous_output = False
    mvp_final_replace_started = False
    try:
        mvp_tmp_output_path = write_csv_temp(mvp_articles, mvp_output_path)
        clean_tmp_output_path = write_csv_temp(clean_articles, clean_output_path)

        if len(mvp_articles) != len(clean_articles):
            raise RuntimeError(
                f"clean_mvp 与 clean 行数不一致: {len(mvp_articles)} != {len(clean_articles)}"
            )
        if set(mvp_articles["article_id"]) != set(clean_articles["article_id"]):
            raise RuntimeError("clean_mvp 与 clean 的 article_id 集合不一致。")

        remove_file_if_exists(mvp_backup_path)
        if mvp_output_path.exists():
            mvp_output_path.replace(mvp_backup_path)
            mvp_had_previous_output = True

        mvp_tmp_output_path.replace(mvp_output_path)
        mvp_final_replace_started = True
        clean_tmp_output_path.replace(clean_output_path)
        remove_file_if_exists(mvp_backup_path)
    except Exception:
        try:
            if mvp_final_replace_started:
                restore_mvp_output(
                    mvp_output_path,
                    mvp_backup_path,
                    mvp_had_previous_output,
                )
            elif mvp_had_previous_output:
                mvp_backup_path.replace(mvp_output_path)
        finally:
            remove_file_if_exists(mvp_tmp_output_path)
            remove_file_if_exists(clean_tmp_output_path)
            remove_file_if_exists(mvp_backup_path)
        raise

    return len(clean_articles)
