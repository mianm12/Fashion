from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
)


def available_cjk_fonts() -> list[str]:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [font_name for font_name in CJK_FONT_CANDIDATES if font_name in installed]


def configure_matplotlib_for_reports() -> str:
    fonts = available_cjk_fonts()
    if not fonts:
        raise RuntimeError(
            "缺少可用中文字体，无法可靠导出中文 SVG/PNG。"
            "请安装 PingFang SC、Noto Sans CJK SC 或 Microsoft YaHei 等字体。"
        )
    selected_font = fonts[0]
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    return selected_font


def save_report_figure(
    figure: Figure,
    output_paths: dict[str, Path],
    *,
    formats: tuple[str, ...] = ("svg", "png"),
) -> list[Path]:
    temp_paths: list[Path] = []
    written: list[Path] = []
    try:
        _validate_figure_formats(output_paths, formats=formats)
        for suffix in formats:
            output_path = output_paths[suffix]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = _temporary_output_path(output_path)
            if temp_path.exists():
                temp_path.unlink()
            temp_paths.append(temp_path)
            figure.savefig(temp_path, bbox_inches="tight", format=suffix)
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise ValueError(f"报告图表输出为空: {output_path}")
            written.append(output_path)
        for suffix, temp_path in zip(formats, temp_paths):
            temp_path.replace(output_paths[suffix])
        temp_paths.clear()
    except Exception:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()
        raise
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()
        plt.close(figure)
    return written


def _validate_figure_formats(
    output_paths: dict[str, Path],
    *,
    formats: tuple[str, ...],
) -> None:
    allowed_formats = {"svg", "png"}
    unknown_formats = sorted(set(formats) - allowed_formats)
    if not formats or unknown_formats:
        raise ValueError(f"figure formats 只支持 svg,png: {formats}")
    if len(set(formats)) != len(formats):
        raise ValueError(f"figure formats 不能重复: {formats}")
    missing_paths = sorted(set(formats) - set(output_paths))
    if missing_paths:
        raise ValueError(f"图表输出路径缺少格式: {missing_paths}")


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.tmp")
