from __future__ import annotations

from fashion_trend import log
from fashion_trend.config import PATH
from fashion_trend.trend.article_sales import (
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    read_article_attribute_edges,
    read_attribute_nodes,
    validate_article_attribute_edges_for_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.io import write_trend_csv

LOG_SOURCE = "attribute-week-heat"


def compute_attribute_week_heat() -> dict[str, int]:
    log.info(f"输入商品周销量表: {PATH['trend_article_week_sales']}", source=LOG_SOURCE)
    article_week_sales = read_article_week_sales(PATH["trend_article_week_sales"])
    validate_article_week_sales(article_week_sales)

    log.info(
        f"输入商品-属性边表: {PATH['graph_edges_article_attribute']}",
        source=LOG_SOURCE,
    )
    article_attribute_edges = read_article_attribute_edges(
        PATH["graph_edges_article_attribute"]
    )
    validate_article_attribute_edges_for_heat(article_attribute_edges)

    log.info(f"输入属性节点表: {PATH['graph_nodes_attribute']}", source=LOG_SOURCE)
    attribute_nodes = read_attribute_nodes(PATH["graph_nodes_attribute"])
    validate_attribute_nodes_for_heat(attribute_nodes)

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
    write_trend_csv(attribute_week_heat, PATH["trend_attribute_week_heat"])

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
    log.info(f"输出文件: {PATH['trend_attribute_week_heat']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
