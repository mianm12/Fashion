from __future__ import annotations

from pathlib import Path


def validate_safe_path_segment(segment: str, source_name: str) -> None:
    if not segment:
        raise ValueError(f"{source_name} 不能为空。")
    if segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise ValueError(f"{source_name} 不是安全的路径片段: {segment}")


def validate_output_parent_dirs(parent_path: Path, output_dir: Path) -> None:
    parent_path = parent_path.resolve()
    output_dir = output_dir.resolve()
    if not parent_path.is_relative_to(output_dir):
        raise ValueError(f"输出目录不在允许范围内: {parent_path}")
