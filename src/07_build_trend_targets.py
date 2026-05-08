from __future__ import annotations

from fashion_trend.catalog.paths import GRAPH_NODES_ATTRIBUTE_PATH
from fashion_trend.catalog.readers import read_attribute_nodes
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.io import write_csv_atomic
from fashion_trend.trend.heat.attribute_heat import (
    read_attribute_week_heat,
    validate_attribute_nodes_for_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.labels.targets import (
    build_attribute_week_target_frame,
    validate_attribute_week_target,
)
from fashion_trend.trend.paths import (
    TREND_ATTRIBUTE_WEEK_HEAT_PATH,
    TREND_ATTRIBUTE_WEEK_TARGET_PATH,
)

LOG_SOURCE = "trend-targets"


def build_trend_targets() -> dict[str, int]:
    """编排趋势标签构建流程。

    流程:
        1. 读取属性节点表并校验热度面板所需节点。
        2. 读取、校验 attribute_week_heat.csv。
        3. 构建、校验并写出 attribute_week_target.csv。
        4. 返回输出行数、覆盖周数和属性节点数摘要。

    Returns:
        dict[str, int]: 趋势标签产物的行数、周数和属性数。
    """
    log.info(f"输入属性节点表: {GRAPH_NODES_ATTRIBUTE_PATH}", source=LOG_SOURCE)
    log.info(f"输入属性周热度表: {TREND_ATTRIBUTE_WEEK_HEAT_PATH}", source=LOG_SOURCE)
    log.info(
        "业务阶段: attribute_week_heat.csv -> attribute_week_target.csv",
        source=LOG_SOURCE,
    )
    attribute_nodes = read_attribute_nodes(GRAPH_NODES_ATTRIBUTE_PATH)
    validate_attribute_nodes_for_heat(attribute_nodes)
    attribute_week_heat = read_attribute_week_heat(TREND_ATTRIBUTE_WEEK_HEAT_PATH)
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
    write_csv_atomic(attribute_week_target, TREND_ATTRIBUTE_WEEK_TARGET_PATH)

    return {
        "rows": len(attribute_week_target),
        "weeks": int(attribute_week_target["week_id"].nunique()),
        "attributes": int(attribute_week_target["attr_id"].nunique()),
    }


def main() -> int:
    """趋势标签阶段入口，稳定写出 attribute_week_target.csv。"""
    try:
        stats = build_trend_targets()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"趋势标签行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖当前周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {stats['attributes']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {TREND_ATTRIBUTE_WEEK_TARGET_PATH}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
