from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTimer:
    stage: str
    rows: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._started_at = time.perf_counter()

    def finish(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "elapsed_seconds": time.perf_counter() - self._started_at,
            **self.details,
        }
        if self.rows is not None:
            payload["rows"] = int(self.rows)
        return payload


def format_stage_log(payload: dict[str, Any]) -> str:
    parts = [f"stage={payload['stage']}"]
    for key, value in payload.items():
        if key == "stage":
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)
