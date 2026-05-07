from __future__ import annotations

from pathlib import Path

import pytest

from fashion_trend.foundation.artifacts import validate_output_parent_dirs


def test_validate_output_parent_dirs_accepts_child_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "models"

    validate_output_parent_dirs(output_root / "model", output_root)


def test_validate_output_parent_dirs_rejects_outside_parent_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs" / "models"
    outside_parent = tmp_path / "outside"

    with pytest.raises(ValueError, match="输出目录不在允许范围内"):
        validate_output_parent_dirs(outside_parent, output_root)
