from __future__ import annotations

from pathlib import Path


def validate_safe_path_segment(segment: str, source_name: str) -> None:
    """校验路径片段不为空且不能通过分隔符或 `..` 形成路径穿越。"""
    if not segment:
        raise ValueError(f"{source_name} 不能为空。")
    if segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise ValueError(f"{source_name} 不是安全的路径片段: {segment}")


def validate_output_parent_dirs(parent_path: Path, output_dir: Path) -> None:
    """校验产物父目录解析后仍位于允许的输出根目录内。"""
    parent_path = parent_path.resolve()
    output_dir = output_dir.resolve()
    if not parent_path.is_relative_to(output_dir):
        raise ValueError(f"输出目录不在允许范围内: {parent_path}")
