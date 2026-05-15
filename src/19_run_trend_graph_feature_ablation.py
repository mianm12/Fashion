from __future__ import annotations

from collections.abc import Sequence

from experiments.trend_graph_feature_ablation.runner import (
    run_trend_graph_feature_ablation,
)
from fashion_trend.foundation import logging as log

LOG_SOURCE = "trend-graph-feature-ablation"


def main(argv: Sequence[str] | None = None) -> int:
    command = ["uv", "run", "python", "src/19_run_trend_graph_feature_ablation.py"]
    if argv:
        command.extend(str(part) for part in argv)
    try:
        payload = run_trend_graph_feature_ablation(command=command)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    log.info(
        "趋势图特征消融实验完成: "
        f"schema_version={payload['schema_version']}, "
        f"summary={payload['metrics_summary_path']}",
        source=LOG_SOURCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
