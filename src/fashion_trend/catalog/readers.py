from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashion_trend.catalog.contracts import (
    ARTICLE_ATTRIBUTE_EDGE_COLUMNS,
    ARTICLE_ATTRIBUTE_EDGE_DTYPES,
    ATTRIBUTE_HIERARCHY_EDGE_COLUMNS,
    ATTRIBUTE_HIERARCHY_EDGE_DTYPES,
    ATTRIBUTE_NODE_COLUMNS,
    ATTRIBUTE_NODE_DTYPES,
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

    missing_columns = sorted(set(ARTICLE_ATTRIBUTE_EDGE_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "商品-属性边表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {article_attribute_edges_path}"
        )

    try:
        return pd.read_csv(
            article_attribute_edges_path,
            usecols=list(ARTICLE_ATTRIBUTE_EDGE_COLUMNS),
            dtype=ARTICLE_ATTRIBUTE_EDGE_DTYPES,
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

    missing_columns = sorted(set(ATTRIBUTE_NODE_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "属性节点表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_nodes_path}"
        )

    try:
        return pd.read_csv(
            attribute_nodes_path,
            usecols=list(ATTRIBUTE_NODE_COLUMNS),
            dtype=ATTRIBUTE_NODE_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性节点表: {attribute_nodes_path}") from exc


def read_attribute_hierarchy_edges(
    attribute_hierarchy_edges_path: Path,
) -> pd.DataFrame:
    if not attribute_hierarchy_edges_path.exists():
        raise FileNotFoundError(f"属性层级边表不存在: {attribute_hierarchy_edges_path}")

    try:
        header = pd.read_csv(attribute_hierarchy_edges_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"无法读取属性层级边表: {attribute_hierarchy_edges_path}"
        ) from exc

    missing_columns = sorted(
        set(ATTRIBUTE_HIERARCHY_EDGE_COLUMNS) - set(header.columns)
    )
    if missing_columns:
        raise ValueError(
            "属性层级边表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_hierarchy_edges_path}"
        )

    try:
        return pd.read_csv(
            attribute_hierarchy_edges_path,
            usecols=list(ATTRIBUTE_HIERARCHY_EDGE_COLUMNS),
            dtype=ATTRIBUTE_HIERARCHY_EDGE_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"无法读取属性层级边表: {attribute_hierarchy_edges_path}"
        ) from exc
