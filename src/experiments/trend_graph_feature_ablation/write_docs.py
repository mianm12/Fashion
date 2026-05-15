from __future__ import annotations

from collections.abc import Sequence

from experiments.trend_graph_feature_ablation.contracts import SCHEMA_VERSION


def render_experiment_doc(
    summary_markdown: str,
    command: str | Sequence[str] | None,
) -> str:
    """渲染趋势图特征消融实验文档，明确其非 stable 边界。"""

    command_text = _format_command(command)
    summary = summary_markdown.rstrip()
    return (
        "# 趋势图特征消融独立实验\n\n"
        f"- schema_version: `{SCHEMA_VERSION}`\n"
        f"- command: `{command_text}`\n\n"
        "## 边界说明\n\n"
        "- 本产物是非 stable 独立实验，只用于审查趋势图特征增强的离线效果。\n"
        "- 本实验不覆盖 `outputs/models/lightgbm/`，不改变默认 LightGBM stable "
        "训练契约。\n"
        "- 本实验不写 `outputs/reports/manifest.json`，不会进入默认论文素材导出。\n"
        "- 本实验不改变 defense app 数据源；答辩展示应用仍读取现有 "
        "`outputs/defense_app/fashion_demo.sqlite`。\n"
        "- 如后续采用本实验结论，应通过显式的非默认导出流程复制到 reports "
        "experimental 产物。\n\n"
        "## 指标汇总\n\n"
        f"{summary}\n"
    )


def _format_command(command: str | Sequence[str] | None) -> str:
    if command is None:
        return "uv run python src/19_run_trend_graph_feature_ablation.py"
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)
