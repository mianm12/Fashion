from __future__ import annotations

from fashion_trend.catalog.paths import (
    GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH,
    GRAPH_NODES_ATTRIBUTE_PATH,
)
from fashion_trend.catalog.readers import (
    read_attribute_hierarchy_edges,
    read_attribute_nodes,
)
from fashion_trend.foundation import logging as log
from fashion_trend.foundation.io import write_parquet_atomic
from fashion_trend.trend.features.samples import (
    build_trend_model_samples_frame,
    validate_trend_model_samples,
)
from fashion_trend.trend.heat.attribute_heat import read_attribute_week_heat
from fashion_trend.trend.labels.targets import read_attribute_week_target
from fashion_trend.trend.paths import (
    TREND_ATTRIBUTE_WEEK_HEAT_PATH,
    TREND_ATTRIBUTE_WEEK_TARGET_PATH,
    TREND_MODEL_SAMPLES_PATH,
)

LOG_SOURCE = "trend-model-samples"


def build_trend_model_samples() -> dict[str, int]:
    """编排趋势模型样本构建流程。

    流程:
        1. 读取属性周热度、趋势标签、属性节点和属性层级边表。
        2. 构建趋势模型训练样本并校验训练契约。
        3. 写出 trend_model_samples.parquet。
        4. 返回输出行数、覆盖周数和属性节点数摘要。

    返回:
        dict[str, int]: 趋势样本产物的行数、周数和属性数。
    """
    log.info(f"输入属性周热度表: {TREND_ATTRIBUTE_WEEK_HEAT_PATH}", source=LOG_SOURCE)
    log.info(
        "业务阶段: 属性热度 + 趋势标签 + 属性图 -> trend_model_samples.parquet",
        source=LOG_SOURCE,
    )
    attribute_week_heat = read_attribute_week_heat(TREND_ATTRIBUTE_WEEK_HEAT_PATH)

    log.info(f"输入趋势标签表: {TREND_ATTRIBUTE_WEEK_TARGET_PATH}", source=LOG_SOURCE)
    attribute_week_target = read_attribute_week_target(TREND_ATTRIBUTE_WEEK_TARGET_PATH)

    log.info(f"输入属性节点表: {GRAPH_NODES_ATTRIBUTE_PATH}", source=LOG_SOURCE)
    attribute_nodes = read_attribute_nodes(GRAPH_NODES_ATTRIBUTE_PATH)

    log.info(
        f"输入属性层级边表: {GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH}",
        source=LOG_SOURCE,
    )
    attribute_hierarchy_edges = read_attribute_hierarchy_edges(
        GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH
    )

    samples = build_trend_model_samples_frame(
        attribute_week_heat,
        attribute_week_target,
        attribute_nodes,
        attribute_hierarchy_edges,
    )
    validate_trend_model_samples(samples)
    write_parquet_atomic(samples, TREND_MODEL_SAMPLES_PATH)

    return {
        "rows": len(samples),
        "weeks": int(samples["week_id"].nunique()),
        "attributes": int(samples["attr_id"].nunique()),
    }


def main() -> int:
    """趋势样本构建阶段入口，稳定写出 trend_model_samples.parquet。"""
    try:
        stats = build_trend_model_samples()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"趋势训练样本行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖样本周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {stats['attributes']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {TREND_MODEL_SAMPLES_PATH}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
