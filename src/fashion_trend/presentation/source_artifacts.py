from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def collect_source_artifact_metadata(
    paths: Mapping[str, Path],
    required: Iterable[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Collect stable source artifact audit metadata for the defense app."""
    required_keys = set(required or ())
    metadata: dict[str, dict[str, object]] = {}
    for key, path in paths.items():
        source_path = Path(path)
        if not source_path.exists():
            if key in required_keys:
                raise FileNotFoundError(
                    f"required source artifact missing: key={key}, path={source_path}"
                )
            continue
        stat = source_path.stat()
        metadata[key] = {
            "path": str(source_path),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "row_count": _row_count(source_path),
        }
    return metadata


def _row_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _csv_row_count(path)
    if suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    if suffix == ".json":
        return _json_row_count(path)
    return None


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def _json_row_count(path: Path) -> int | None:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, list):
        return len(payload)
    return None
