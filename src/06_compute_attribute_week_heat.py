from __future__ import annotations

from fashion_trend.catalog.readers import (
    read_article_attribute_edges,
    read_attribute_nodes,
)
from fashion_trend.catalog.paths import (
    GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH,
    GRAPH_NODES_ATTRIBUTE_PATH,
)
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.trend.article_sales import (
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    validate_all_sales_articles_have_attribute_edges,
    validate_attribute_edge_node_metadata_consistency,
    validate_attribute_week_heat,
)
from fashion_trend.trend.paths import (
    TREND_ARTICLE_WEEK_SALES_PATH,
    TREND_ATTRIBUTE_WEEK_HEAT_PATH,
)

LOG_SOURCE = "attribute-week-heat"


def compute_attribute_week_heat() -> dict[str, int]:
    log.info(f"输入商品周销量表: {TREND_ARTICLE_WEEK_SALES_PATH}", source=LOG_SOURCE)
    log.info(
        "业务阶段: 属性图 + article_week_sales.csv -> attribute_week_heat.csv",
        source=LOG_SOURCE,
    )
    article_week_sales = read_article_week_sales(TREND_ARTICLE_WEEK_SALES_PATH)
    validate_article_week_sales(article_week_sales)

    log.info(
        f"输入商品-属性边表: {GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH}",
        source=LOG_SOURCE,
    )
    article_attribute_edges = read_article_attribute_edges(
        GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH
    )

    log.info(f"输入属性节点表: {GRAPH_NODES_ATTRIBUTE_PATH}", source=LOG_SOURCE)
    attribute_nodes = read_attribute_nodes(GRAPH_NODES_ATTRIBUTE_PATH)
    validate_all_sales_articles_have_attribute_edges(
        article_week_sales,
        article_attribute_edges,
    )
    validate_attribute_edge_node_metadata_consistency(
        article_attribute_edges,
        attribute_nodes,
    )

    attribute_week_heat = build_attribute_week_heat_frame(
        article_week_sales,
        article_attribute_edges,
        attribute_nodes,
    )
    validate_attribute_week_heat(
        attribute_week_heat,
        expected_week_ids=sorted(article_week_sales["week_id"].unique()),
        expected_attribute_nodes=attribute_nodes,
    )
    write_csv_atomic(attribute_week_heat, TREND_ATTRIBUTE_WEEK_HEAT_PATH)

    return {
        "rows": len(attribute_week_heat),
        "weeks": int(attribute_week_heat["week_id"].nunique()),
        "attr_types": int(attribute_week_heat["attr_type"].nunique()),
        "attributes": int(attribute_week_heat["attr_id"].nunique()),
    }


def main() -> int:
    try:
        stats = compute_attribute_week_heat()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"属性周热度行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性类型数: {stats['attr_types']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {stats['attributes']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {TREND_ATTRIBUTE_WEEK_HEAT_PATH}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
