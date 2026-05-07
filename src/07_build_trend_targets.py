from __future__ import annotations

from fashion_trend.foundation import logging as log
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.foundation.paths import PATH
from fashion_trend.catalog.graph import read_attribute_nodes
from fashion_trend.trend.attribute_heat import (
    read_attribute_week_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.targets import (
    build_attribute_week_target_frame,
    validate_attribute_week_target,
)

LOG_SOURCE = "trend-targets"


def build_trend_targets() -> dict[str, int]:
    log.info(f"输入属性节点表: {PATH['graph_nodes_attribute']}", source=LOG_SOURCE)
    log.info(
        f"输入属性周热度表: {PATH['trend_attribute_week_heat']}", source=LOG_SOURCE
    )
    log.info(
        "业务阶段: attribute_week_heat.csv -> attribute_week_target.csv",
        source=LOG_SOURCE,
    )
    attribute_nodes = read_attribute_nodes(PATH["graph_nodes_attribute"])
    validate_attribute_nodes_for_heat(attribute_nodes)
    attribute_week_heat = read_attribute_week_heat(PATH["trend_attribute_week_heat"])
    validate_attribute_week_heat(
        attribute_week_heat,
        expected_week_ids=sorted(attribute_week_heat["week_id"].unique()),
        expected_attribute_nodes=attribute_nodes,
    )

    attribute_week_target = build_attribute_week_target_frame(attribute_week_heat)
    validate_attribute_week_target(
        attribute_week_target,
        expected_week_count=int(attribute_week_heat["week_id"].nunique()),
        expected_attribute_count=len(attribute_nodes),
    )
    write_csv_atomic(attribute_week_target, PATH["trend_attribute_week_target"])

    return {
        "rows": len(attribute_week_target),
        "weeks": int(attribute_week_target["week_id"].nunique()),
        "attributes": int(attribute_week_target["attr_id"].nunique()),
    }


def main() -> int:
    try:
        stats = build_trend_targets()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"趋势标签行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖当前周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {stats['attributes']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {PATH['trend_attribute_week_target']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
