from __future__ import annotations

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
