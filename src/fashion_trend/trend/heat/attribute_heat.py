from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fashion_trend.catalog.contracts import (
    ARTICLE_ATTRIBUTE_EDGE_COLUMNS,
    ATTRIBUTE_NODE_COLUMNS,
)
from fashion_trend.foundation.dataframe import (
    validate_no_missing_values,
    validate_non_negative_values,
    validate_positive_values,
    validate_required_columns,
    validate_unique_key,
)
from fashion_trend.trend.heat.article_sales import validate_article_week_sales
from fashion_trend.trend.schema import (
    ATTRIBUTE_WEEK_HEAT_COLUMNS,
    ATTRIBUTE_WEEK_HEAT_DTYPES,
)


def validate_article_attribute_edges_for_heat(
    article_attribute_edges: pd.DataFrame,
) -> None:
    """校验属性热度计算所需的商品-属性边表主键和属性元数据一致性。"""
    validate_required_columns(
        article_attribute_edges,
        ARTICLE_ATTRIBUTE_EDGE_COLUMNS,
        source_name="商品-属性边表",
    )
    validate_no_missing_values(
        article_attribute_edges,
        ARTICLE_ATTRIBUTE_EDGE_COLUMNS,
        source_name="商品-属性边表",
    )
    validate_unique_key(
        article_attribute_edges,
        ["article_id", "attr_id"],
        source_name="商品-属性边表",
    )

    attr_pairs = article_attribute_edges.loc[
        :, ["attr_id", "attr_type", "attr_value"]
    ].drop_duplicates()
    attr_pair_counts = attr_pairs.groupby("attr_id").size()
    inconsistent_attr_ids = attr_pair_counts[attr_pair_counts > 1].index.tolist()
    if inconsistent_attr_ids:
        attr_id = inconsistent_attr_ids[0]
        pairs = attr_pairs[attr_pairs["attr_id"] == attr_id].loc[
            :, ["attr_type", "attr_value"]
        ]
        pair_examples = ", ".join(
            f"{row.attr_type}={row.attr_value}" for row in pairs.itertuples()
        )
        raise ValueError(
            "商品-属性边表存在 attr_id 映射到多个 attr_type + attr_value: "
            f"{attr_id} -> {pair_examples}"
        )


def validate_all_sales_articles_have_attribute_edges(
    article_week_sales: pd.DataFrame,
    article_attribute_edges: pd.DataFrame,
) -> None:
    """校验商品周销量表中的所有商品都能映射到至少一条属性边。"""
    sales_article_ids = set(article_week_sales["article_id"].astype("string"))
    edge_article_ids = set(article_attribute_edges["article_id"].astype("string"))
    missing_article_ids = sorted(sales_article_ids - edge_article_ids)
    if missing_article_ids:
        examples = ", ".join(missing_article_ids[:5])
        raise ValueError(
            f"商品周销量表存在 {len(missing_article_ids)} 个 article_id "
            f"无法映射到属性边，例如: {examples}"
        )


def read_attribute_week_heat(attribute_week_heat_path: Path) -> pd.DataFrame:
    """读取 `attribute_week_heat.csv` 属性周热度表并保留契约列类型。"""
    if not attribute_week_heat_path.exists():
        raise FileNotFoundError(f"属性周热度表不存在: {attribute_week_heat_path}")

    try:
        header = pd.read_csv(attribute_week_heat_path, nrows=0)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性周热度表: {attribute_week_heat_path}") from exc

    missing_columns = sorted(set(ATTRIBUTE_WEEK_HEAT_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            "属性周热度表缺少必要字段: "
            + ", ".join(missing_columns)
            + f"。文件: {attribute_week_heat_path}"
        )

    try:
        return pd.read_csv(
            attribute_week_heat_path,
            usecols=list(ATTRIBUTE_WEEK_HEAT_COLUMNS),
            dtype=ATTRIBUTE_WEEK_HEAT_DTYPES,
        )
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"无法读取属性周热度表: {attribute_week_heat_path}") from exc


def validate_attribute_nodes_for_heat(attribute_nodes: pd.DataFrame) -> None:
    """校验属性热度和图特征依赖的属性节点表字段、主键和标志位。"""
    validate_required_columns(
        attribute_nodes,
        ATTRIBUTE_NODE_COLUMNS,
        source_name="属性节点表",
    )
    validate_no_missing_values(
        attribute_nodes,
        ATTRIBUTE_NODE_COLUMNS,
        source_name="属性节点表",
    )
    validate_unique_key(attribute_nodes, ["attr_id"], source_name="属性节点表")
    validate_non_negative_values(
        attribute_nodes,
        ["article_count"],
        source_name="属性节点表",
    )
    invalid_core_flags = sorted(set(attribute_nodes["is_core_attr"]) - {0, 1})
    if invalid_core_flags:
        raise ValueError("属性节点表存在非法 is_core_attr: " + str(invalid_core_flags))


def validate_attribute_edge_node_metadata_consistency(
    article_attribute_edges: pd.DataFrame,
    attribute_nodes: pd.DataFrame,
) -> None:
    """校验商品-属性边表与属性节点表中的属性类型和值保持一致。"""
    edge_attributes = article_attribute_edges.loc[
        :, ["attr_id", "attr_type", "attr_value"]
    ].drop_duplicates()
    node_attributes = attribute_nodes.loc[
        :, ["attr_id", "attr_type", "attr_value"]
    ].drop_duplicates()
    merged_attributes = edge_attributes.merge(
        node_attributes,
        on="attr_id",
        how="inner",
        suffixes=("_edge", "_node"),
    )

    for column in ["attr_type", "attr_value"]:
        mismatch_mask = (
            merged_attributes[f"{column}_edge"] != merged_attributes[f"{column}_node"]
        )
        if mismatch_mask.any():
            mismatch = merged_attributes[mismatch_mask].iloc[0]
            raise ValueError(
                "商品-属性边表与属性节点表存在元数据不一致: "
                f"attr_id={mismatch.attr_id}, "
                f"{column}_edge={mismatch[f'{column}_edge']}, "
                f"{column}_node={mismatch[f'{column}_node']}"
            )


def build_attribute_week_heat_frame(
    article_week_sales: pd.DataFrame,
    article_attribute_edges: pd.DataFrame,
    attribute_nodes: pd.DataFrame,
) -> pd.DataFrame:
    """构建完整属性周热度面板。

    Args:
        article_week_sales: 商品周销量表，提供每个商品在每周的销售次数。
        article_attribute_edges: 商品-属性边表，用于把商品销量分摊到属性。
        attribute_nodes: 属性节点表，提供完整属性全集和属性元数据。

    Returns:
        完整 `week_id x attr_id` 面板。未观测到销量的属性周填充为
        `heat_cnt=0`，再按 `week_id + attr_type` 计算 `type_total_heat`
        和 `heat_share`，用 `log1p(heat_cnt)` 生成 `log_heat`，并按
        `week_id`、`attr_type`、`heat_cnt` 降序、`attr_id` 升序生成稳定排名。

    Raises:
        ValueError: 当输入契约、商品映射、属性节点或元数据一致性不满足要求时抛出。
    """
    validate_article_week_sales(article_week_sales)
    validate_article_attribute_edges_for_heat(article_attribute_edges)
    validate_attribute_nodes_for_heat(attribute_nodes)
    validate_all_sales_articles_have_attribute_edges(
        article_week_sales,
        article_attribute_edges,
    )

    edge_attr_ids = set(article_attribute_edges["attr_id"].astype("string"))
    node_attr_ids = set(attribute_nodes["attr_id"].astype("string"))
    missing_node_attr_ids = sorted(edge_attr_ids - node_attr_ids)
    if missing_node_attr_ids:
        examples = ", ".join(missing_node_attr_ids[:5])
        raise ValueError(
            f"商品-属性边表存在 {len(missing_node_attr_ids)} 个 attr_id "
            f"无法映射到属性节点，例如: {examples}"
        )

    validate_attribute_edge_node_metadata_consistency(
        article_attribute_edges,
        attribute_nodes,
    )

    normalized_sales = article_week_sales.loc[
        :, ["week_id", "article_id", "sales_cnt"]
    ].copy()
    normalized_sales["article_id"] = normalized_sales["article_id"].astype("string")

    normalized_edges = article_attribute_edges.loc[
        :, list(ARTICLE_ATTRIBUTE_EDGE_COLUMNS)
    ].copy()
    normalized_edges["article_id"] = normalized_edges["article_id"].astype("string")
    normalized_edges["attr_id"] = normalized_edges["attr_id"].astype("string")
    normalized_edges["attr_type"] = normalized_edges["attr_type"].astype("string")
    normalized_edges["attr_value"] = normalized_edges["attr_value"].astype("string")

    joined = normalized_sales.merge(normalized_edges, on="article_id", how="inner")
    observed_heat = (
        joined.groupby(["week_id", "attr_id"], as_index=False)["sales_cnt"]
        .sum()
        .rename(columns={"sales_cnt": "heat_cnt"})
    )

    weeks = pd.DataFrame({"week_id": sorted(normalized_sales["week_id"].unique())})
    attributes = attribute_nodes.loc[:, ["attr_id", "attr_type", "attr_value"]].copy()
    # 完整面板保留零热度属性，避免下游标签和样本阶段隐式补缺。
    panel = weeks.merge(attributes, how="cross")
    heat = panel.merge(observed_heat, on=["week_id", "attr_id"], how="left")
    heat["heat_cnt"] = heat["heat_cnt"].fillna(0).astype("int64")
    heat["type_total_heat"] = heat.groupby(["week_id", "attr_type"])[
        "heat_cnt"
    ].transform("sum")
    heat["type_total_heat"] = heat["type_total_heat"].astype("int64")
    heat["heat_share"] = np.where(
        heat["type_total_heat"] > 0,
        heat["heat_cnt"] / heat["type_total_heat"],
        0.0,
    )
    heat["log_heat"] = np.log1p(heat["heat_cnt"])

    # 排名先按热度降序，再用 attr_id 升序稳定处理同热度属性。
    heat = heat.sort_values(
        ["week_id", "attr_type", "heat_cnt", "attr_id"],
        ascending=[True, True, False, True],
        ignore_index=True,
    )
    heat["rank_in_type"] = (
        heat.groupby(["week_id", "attr_type"]).cumcount().add(1).astype("int64")
    )

    return heat.loc[:, list(ATTRIBUTE_WEEK_HEAT_COLUMNS)].sort_values(
        ["week_id", "attr_type", "rank_in_type", "attr_id"],
        ignore_index=True,
    )


def validate_attribute_week_heat(
    attribute_week_heat: pd.DataFrame,
    expected_week_ids: Sequence[int] | None = None,
    expected_attribute_nodes: pd.DataFrame | None = None,
) -> None:
    """校验属性周热度表的完整性和派生字段一致性。

    Args:
        attribute_week_heat: 待校验的属性周热度表。
        expected_week_ids: 可选的预期周集合，用于校验完整面板周范围。
        expected_attribute_nodes: 可选的属性节点表，用于校验属性全集和行数。

    Raises:
        ValueError: 当列契约、主键、非负约束、占比、`log1p` 或排名不一致时抛出。
    """
    validate_required_columns(
        attribute_week_heat,
        ATTRIBUTE_WEEK_HEAT_COLUMNS,
        source_name="属性周热度表",
    )
    validate_no_missing_values(
        attribute_week_heat,
        ATTRIBUTE_WEEK_HEAT_COLUMNS,
        source_name="属性周热度表",
    )
    validate_unique_key(
        attribute_week_heat,
        ["week_id", "attr_id"],
        source_name="属性周热度表",
    )
    validate_non_negative_values(
        attribute_week_heat,
        ["heat_cnt", "type_total_heat", "heat_share", "log_heat"],
        source_name="属性周热度表",
    )
    validate_positive_values(
        attribute_week_heat,
        ["rank_in_type"],
        source_name="属性周热度表",
    )

    normalized_expected_week_ids = None
    if expected_week_ids is not None:
        normalized_expected_week_ids = set(expected_week_ids)
        actual_week_ids = set(attribute_week_heat["week_id"])
        if actual_week_ids != normalized_expected_week_ids:
            raise ValueError("属性周热度表 week_id 集合与预期不一致。")

    if expected_attribute_nodes is not None:
        validate_attribute_nodes_for_heat(expected_attribute_nodes)
        expected_attr_ids = set(expected_attribute_nodes["attr_id"].astype("string"))
        actual_attr_ids = set(attribute_week_heat["attr_id"].astype("string"))
        if actual_attr_ids != expected_attr_ids:
            raise ValueError("属性周热度表 attr_id 集合与属性节点表不一致。")
        if normalized_expected_week_ids is not None:
            expected_rows = len(normalized_expected_week_ids) * len(
                expected_attribute_nodes
            )
            if len(attribute_week_heat) != expected_rows:
                raise ValueError(
                    "属性周热度表行数与完整 week_id x attr_id 面板不一致。"
                )

    if (attribute_week_heat["type_total_heat"] < attribute_week_heat["heat_cnt"]).any():
        raise ValueError("属性周热度表存在 type_total_heat 小于 heat_cnt 的记录。")
    if (attribute_week_heat["heat_share"] > 1).any():
        raise ValueError("属性周热度表存在 heat_share 大于 1 的记录。")

    expected_type_total_heat = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "heat_cnt"
    ].transform("sum")
    if not (
        attribute_week_heat["type_total_heat"].to_numpy()
        == expected_type_total_heat.to_numpy()
    ).all():
        raise ValueError(
            "属性周热度表存在 type_total_heat 与 heat_cnt 分组求和不一致。"
        )

    share_totals = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "heat_share"
    ].sum()
    total_heat_by_type = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "heat_cnt"
    ].sum()
    expected_share_totals = pd.Series(
        np.where(total_heat_by_type > 0, 1.0, 0.0),
        index=total_heat_by_type.index,
    )
    invalid_share_totals = share_totals[
        ~np.isclose(share_totals, expected_share_totals, atol=1e-9, rtol=0)
    ]
    if not invalid_share_totals.empty:
        raise ValueError(
            "属性周热度表存在 week_id + attr_type 占比和不等于 1 或 0 的分组。"
        )

    expected_heat_share = np.where(
        attribute_week_heat["type_total_heat"] > 0,
        attribute_week_heat["heat_cnt"] / attribute_week_heat["type_total_heat"],
        0.0,
    )
    if not np.allclose(
        attribute_week_heat["heat_share"].to_numpy(dtype=float),
        expected_heat_share,
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError(
            "属性周热度表存在 heat_share 与 heat_cnt / type_total_heat 不一致。"
        )

    expected_log_heat = np.log1p(attribute_week_heat["heat_cnt"])
    if not np.allclose(
        attribute_week_heat["log_heat"].to_numpy(dtype=float),
        expected_log_heat.to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("属性周热度表存在 log_heat 与 log1p(heat_cnt) 不一致。")

    # 排名契约要求每个 week_id + attr_type 分组内从 1 开始且连续。
    rank_counts = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "rank_in_type"
    ].nunique()
    row_counts = attribute_week_heat.groupby(["week_id", "attr_type"]).size()
    if not rank_counts.equals(row_counts):
        raise ValueError("属性周热度表存在重复 rank_in_type。")
    min_ranks = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "rank_in_type"
    ].min()
    if (min_ranks != 1).any():
        raise ValueError("属性周热度表存在 rank_in_type 未从 1 开始的分组。")
    max_ranks = attribute_week_heat.groupby(["week_id", "attr_type"])[
        "rank_in_type"
    ].max()
    if not max_ranks.equals(row_counts):
        raise ValueError("属性周热度表存在 rank_in_type 不连续的分组。")

    ranked_heat = attribute_week_heat.sort_values(
        ["week_id", "attr_type", "heat_cnt", "attr_id"],
        ascending=[True, True, False, True],
    ).copy()
    expected_ranks = ranked_heat.groupby(["week_id", "attr_type"]).cumcount().add(1)
    if not (ranked_heat["rank_in_type"].to_numpy() == expected_ranks.to_numpy()).all():
        raise ValueError("属性周热度表存在 rank_in_type 排序不符合热度降序规则。")
