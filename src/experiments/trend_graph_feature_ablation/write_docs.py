from __future__ import annotations

from collections.abc import Sequence

from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_VARIANTS,
    SCHEMA_VERSION,
)
from experiments.trend_graph_feature_ablation.feature_groups import (
    build_feature_groups,
    build_variant_feature_masks,
)


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
        "## 实验目的\n\n"
        "本实验用于隔离评估属性知识图谱增强特征对趋势 LightGBM 排序效果的影响。"
        "它复用默认趋势样本和默认时间切分，只在独立实验目录中生成增强样本、"
        "消融 run、指标汇总和实验说明，便于判断图结构上下文、兄弟竞争关系和轻量"
        "结构特征是否值得进入后续论文分析。\n\n"
        "## 输入 artifact\n\n"
        "- `data/processed/features/trend_model_samples.parquet`\n"
        "- `data/processed/features/trend_model_samples_train.parquet`\n"
        "- `data/processed/features/trend_model_samples_valid.parquet`\n"
        "- `data/processed/features/trend_model_samples_test.parquet`\n"
        "- `data/processed/graph/edges_attribute_hierarchy.csv`\n"
        "- `data/processed/graph/nodes_attribute.csv`，仅作为 input hash 记录。\n"
        "- `outputs/models/lightgbm/params.json`，可选输入，仅用于复用 stable "
        "LightGBM 参数。\n\n"
        "## feature groups 定义\n\n"
        f"{_render_feature_groups()}\n"
        "## 五个消融版本\n\n"
        f"{_render_variants()}\n"
        "## 运行命令\n\n"
        "```sh\n"
        f"{command_text}\n"
        "```\n\n"
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
        f"{summary}\n\n"
        "## 论文使用注意事项\n\n"
        "- 该实验只能作为趋势图特征消融证据，不能直接替代默认 LightGBM stable "
        "结果。\n"
        "- 指标解读应同时查看 valid 和 test，不应只根据单个 split 形成强结论。\n"
        "- 写入论文正文前，需要明确说明这是独立实验目录下的非默认产物。\n"
        "- 默认 reports 和 defense app 不会自动读取本实验输出；如需展示，应先通过"
        "非默认 experimental 导出复制。\n"
    )


def _format_command(command: str | Sequence[str] | None) -> str:
    if command is None:
        return "uv run python src/19_run_trend_graph_feature_ablation.py"
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def _render_feature_groups() -> str:
    lines: list[str] = []
    for group_name, features in build_feature_groups().items():
        feature_text = ", ".join(f"`{feature}`" for feature in features)
        lines.append(f"- `{group_name}`: {feature_text}")
    return "\n".join(lines) + "\n\n"


def _render_variants() -> str:
    masks = build_variant_feature_masks()
    descriptions = {
        "no_graph": "仅保留非图数值特征和类别特征，用作无图特征基线。",
        "current_coarse_graph": "复现当前 stable LightGBM 的粗粒度图特征组合。",
        "full_enhanced": "使用粗粒度图特征、层级上下文、兄弟竞争和轻量结构特征。",
        "wo_hierarchy_context": "从 full enhanced 中移除父子层级动态上下文特征。",
        "wo_sibling_competition": "从 full enhanced 中移除同父兄弟竞争特征。",
    }
    lines: list[str] = []
    for variant in ABLATION_VARIANTS:
        mask = masks[variant]
        feature_count = len(mask["numeric_features"]) + len(
            mask["categorical_features"]
        )
        lines.append(
            f"- `{variant}`: {descriptions[variant]} feature_count={feature_count}。"
        )
    return "\n".join(lines) + "\n\n"
