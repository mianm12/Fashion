from __future__ import annotations

import os
from pathlib import Path

DB_PATH_ENV = "DEFENSE_APP_DB_PATH"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def get_database_path() -> Path:
    configured_path = os.getenv(DB_PATH_ENV)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return get_repo_root() / "outputs" / "defense_app" / "fashion_demo.sqlite"
