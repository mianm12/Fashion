from __future__ import annotations

from fashion_trend import log
from fashion_trend.config import PATH
from fashion_trend.trend import (
    build_trend_model_samples_frame,
    read_attribute_hierarchy_edges,
    read_attribute_nodes,
    read_attribute_week_heat,
    read_attribute_week_target,
    validate_trend_model_samples,
    write_trend_parquet,
)

LOG_SOURCE = "trend-model-samples"


def build_trend_model_samples() -> dict[str, int]:
    log.info(f"输入属性周热度表: {PATH['trend_attribute_week_heat']}", source=LOG_SOURCE)
    attribute_week_heat = read_attribute_week_heat(PATH["trend_attribute_week_heat"])

    log.info(f"输入趋势标签表: {PATH['trend_attribute_week_target']}", source=LOG_SOURCE)
    attribute_week_target = read_attribute_week_target(
        PATH["trend_attribute_week_target"]
    )

    log.info(f"输入属性节点表: {PATH['graph_nodes_attribute']}", source=LOG_SOURCE)
    attribute_nodes = read_attribute_nodes(PATH["graph_nodes_attribute"])

    log.info(
        f"输入属性层级边表: {PATH['graph_edges_attribute_hierarchy']}",
        source=LOG_SOURCE,
    )
    attribute_hierarchy_edges = read_attribute_hierarchy_edges(
        PATH["graph_edges_attribute_hierarchy"]
    )

    samples = build_trend_model_samples_frame(
        attribute_week_heat,
        attribute_week_target,
        attribute_nodes,
        attribute_hierarchy_edges,
    )
    validate_trend_model_samples(samples)
    write_trend_parquet(samples, PATH["features_trend_model_samples"])

    return {
        "rows": len(samples),
        "weeks": int(samples["week_id"].nunique()),
        "attributes": int(samples["attr_id"].nunique()),
    }


def main() -> int:
    try:
        stats = build_trend_model_samples()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(f"趋势训练样本行数: {stats['rows']:,}", source=LOG_SOURCE)
    log.info(f"覆盖样本周数: {stats['weeks']:,}", source=LOG_SOURCE)
    log.info(f"覆盖属性节点数: {stats['attributes']:,}", source=LOG_SOURCE)
    log.info(f"输出文件: {PATH['features_trend_model_samples']}", source=LOG_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
