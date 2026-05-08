from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import pandas as pd

from fashion_trend.foundation.io import remove_file_if_exists

ARTICLE_ID_COLUMN = "article_id"
PRODUCT_CODE_COLUMN = "product_code"

# `articles_clean_mvp.csv` 和 `articles_clean.csv` 共享的核心商品属性列。
CORE_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "product_group_name",
    "product_type_name",
    "garment_group_name",
    "colour_group_name",
    "graphical_appearance_name",
)

# `articles_clean.csv` 额外保留的属性层级列。
HIERARCHY_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "perceived_colour_master_name",
    "index_group_name",
    "index_name",
    "section_name",
    "department_name",
)

# `articles_clean_mvp.csv` 的稳定列契约，用于早期商品清洗闭环。
MVP_ARTICLE_COLUMNS: tuple[str, ...] = (
    "article_id",
    "product_code",
    "prod_name",
    *CORE_ATTRIBUTE_COLUMNS,
)

# `articles_clean.csv` 的完整清洗商品列契约，供属性图构建使用。
CLEAN_ARTICLE_COLUMNS: tuple[str, ...] = (
    *MVP_ARTICLE_COLUMNS,
    *HIERARCHY_ATTRIBUTE_COLUMNS,
)

# 属性图从完整清洗商品表中抽取的属性列集合。
ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    *CORE_ATTRIBUTE_COLUMNS,
    *HIERARCHY_ATTRIBUTE_COLUMNS,
)


def validate_required_columns(
    actual_columns: Sequence[str],
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    """校验商品输入列集合覆盖目标清洗或属性图契约。"""
    missing_columns = sorted(set(required_columns) - set(actual_columns))
    if missing_columns:
        raise ValueError(f"{source_name} 缺少必要字段: " + ", ".join(missing_columns))


def validate_no_missing_values(
    articles: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    """校验商品清洗所需列不存在缺失值。"""
    missing_columns = [
        column for column in required_columns if int(articles[column].isna().sum()) > 0
    ]
    if missing_columns:
        raise ValueError(f"{source_name} 存在缺失值字段: " + ", ".join(missing_columns))


def validate_unique_values(
    articles: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> None:
    """校验商品表中指定列组合没有重复值。"""
    duplicate_mask = articles.duplicated(subset=list(columns), keep=False)
    if duplicate_mask.any():
        raise ValueError(f"{source_name} 存在重复字段值: " + ", ".join(columns))


def normalize_article_identifiers(articles: pd.DataFrame) -> pd.DataFrame:
    """把商品主键和产品编码标准化为 pandas string 类型。"""
    normalized = articles.copy()
    normalized[ARTICLE_ID_COLUMN] = normalized[ARTICLE_ID_COLUMN].astype("string")
    normalized[PRODUCT_CODE_COLUMN] = normalized[PRODUCT_CODE_COLUMN].astype("string")
    return normalized


def build_clean_article_frames(
    raw_articles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从原始商品表同时构建 MVP 清洗表和完整清洗表。"""
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
    validate_unique_values(
        raw_articles,
        [ARTICLE_ID_COLUMN],
        source_name="原始 articles.csv",
    )

    normalized_articles = normalize_article_identifiers(raw_articles)
    mvp_articles = normalized_articles.loc[:, list(MVP_ARTICLE_COLUMNS)].copy()
    clean_articles = normalized_articles.loc[:, list(CLEAN_ARTICLE_COLUMNS)].copy()
    return mvp_articles, clean_articles


def read_articles_csv(raw_articles_path: Path) -> pd.DataFrame:
    """读取原始 `articles.csv` 中商品清洗需要的稳定列集合。"""
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
    """把商品清洗结果写入目标路径旁的临时 CSV 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(tmp_output_path, index=False, quoting=csv.QUOTE_ALL)
    return tmp_output_path


def restore_mvp_output(
    mvp_output_path: Path,
    mvp_backup_path: Path,
    mvp_had_previous_output: bool,
) -> None:
    """在完整清洗表发布失败时恢复已替换的 MVP 清洗产物。"""
    remove_file_if_exists(mvp_output_path)
    if mvp_had_previous_output:
        mvp_backup_path.replace(mvp_output_path)


def clean_articles_file(
    raw_articles_path: Path,
    mvp_output_path: Path,
    clean_output_path: Path,
) -> int:
    """发布 MVP 和完整商品清洗产物。

    参数:
        raw_articles_path: H&M 原始 `articles.csv` 路径。
        mvp_output_path: 只含核心属性的 `articles_clean_mvp.csv` 输出路径。
        clean_output_path: 含层级属性的 `articles_clean.csv` 输出路径。

    返回:
        完整清洗商品表的行数。

    边界:
        MVP 产物和完整清洗产物一起发布；若完整清洗产物替换失败，
        回滚逻辑会尽量恢复已经替换过的 MVP 产物，降低半发布风险；
        两个 CSV 的跨文件可见性不具备原子保证。
    """
    raw_articles = read_articles_csv(raw_articles_path)
    mvp_articles, clean_articles = build_clean_article_frames(raw_articles)

    mvp_tmp_output_path = mvp_output_path.with_suffix(mvp_output_path.suffix + ".tmp")
    clean_tmp_output_path = clean_output_path.with_suffix(
        clean_output_path.suffix + ".tmp"
    )
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
