from __future__ import annotations

from pathlib import Path


def build_input_fingerprints(
    input_paths: dict[str, str] | None,
) -> dict[str, dict[str, object]]:
    """Build lightweight freshness fingerprints for recommendation inputs."""
    fingerprints: dict[str, dict[str, object]] = {}
    for name, path_value in sorted(dict(input_paths or {}).items()):
        path_text = str(path_value)
        path = Path(path_text)
        fingerprint: dict[str, object] = {
            "path": path_text,
            "exists": path.exists(),
        }
        if path.exists():
            stat = path.stat()
            fingerprint["size_bytes"] = int(stat.st_size)
            fingerprint["mtime_ns"] = int(stat.st_mtime_ns)
        fingerprints[name] = fingerprint
    return fingerprints
