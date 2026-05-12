from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from fashion_trend.presentation.builders import build_presentation_tables
from fashion_trend.presentation.extractors import load_presentation_sources
from fashion_trend.presentation.paths import DEFENSE_APP_DB_PATH
from fashion_trend.presentation.sqlite_writer import write_presentation_database

REPORT_ASSET_SUFFIXES = {".png", ".svg"}


def run_defense_app_db_build(
    output_path: Path = DEFENSE_APP_DB_PATH,
) -> dict[str, object]:
    """Build the read-only SQLite database for the local defense demo app."""
    sources = load_presentation_sources()
    tables = build_presentation_tables(sources)
    output_dir = output_path.parent
    db_backup_path: Path | None = None
    try:
        static_assets = stage_report_assets(tables["report_assets"], output_dir)
        db_backup_path = backup_database(output_path)
        write_presentation_database(tables, output_path)
        publish_staged_report_assets(output_dir)
    except Exception:
        cleanup_staged_report_assets(output_dir)
        restore_database_backup(output_path, db_backup_path)
        raise
    else:
        cleanup_database_backup(db_backup_path)
    return {
        "database_path": str(output_path),
        "table_counts": {
            table_name: int(len(frame)) for table_name, frame in tables.items()
        },
        "source_artifacts": sources.source_artifacts or {},
        "static_assets": static_assets,
    }


def stage_report_assets(
    report_assets: pd.DataFrame,
    output_dir: Path,
) -> list[dict[str, str]]:
    """Copy selected report image assets into a staging directory."""
    staging_dir = _staging_reports_dir(output_dir)
    final_reports_dir = _final_reports_dir(output_dir)
    copied_assets: list[dict[str, str]] = []
    _remove_dir(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        for row in report_assets.itertuples(index=False):
            source_path = Path(str(row.source_path))
            if source_path.suffix.lower() not in REPORT_ASSET_SUFFIXES:
                continue
            if not source_path.exists():
                raise FileNotFoundError(f"report asset not found: {source_path}")
            staged_path = staging_dir / source_path.name
            final_path = final_reports_dir / source_path.name
            shutil.copy2(source_path, staged_path)
            copied_assets.append(
                {
                    "asset_name": str(row.asset_name),
                    "source_path": str(source_path),
                    "static_path": str(final_path),
                    "static_url": str(row.static_url),
                }
            )
    except Exception:
        _remove_dir(staging_dir)
        raise
    return copied_assets


def publish_staged_report_assets(output_dir: Path) -> None:
    """Publish staged report assets, restoring the old directory on failure."""
    final_dir = _final_reports_dir(output_dir)
    staging_dir = _staging_reports_dir(output_dir)
    backup_dir = _backup_reports_dir(output_dir)
    _remove_dir(backup_dir)

    try:
        if final_dir.exists():
            final_dir.rename(backup_dir)
        staging_dir.rename(final_dir)
    except Exception:
        if final_dir.exists():
            _remove_dir(final_dir)
        if backup_dir.exists():
            backup_dir.rename(final_dir)
        raise
    else:
        _remove_dir(backup_dir)


def cleanup_staged_report_assets(output_dir: Path) -> None:
    """Remove temporary static asset directories left by a failed build."""
    _remove_dir(_staging_reports_dir(output_dir))
    _remove_dir(_backup_reports_dir(output_dir))


def backup_database(output_path: Path) -> Path | None:
    """Copy the current database aside so runner-level publish can roll back."""
    backup_path = _database_backup_path(output_path)
    if backup_path.exists():
        backup_path.unlink()
    if not output_path.exists():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, backup_path)
    return backup_path


def restore_database_backup(
    output_path: Path,
    backup_path: Path | None,
) -> None:
    """Restore the previous database or remove a newly published failed one."""
    if backup_path is None:
        if output_path.exists():
            output_path.unlink()
        return
    if output_path.exists():
        output_path.unlink()
    backup_path.replace(output_path)


def cleanup_database_backup(backup_path: Path | None) -> None:
    """Remove the runner database backup after a successful publish."""
    if backup_path is not None and backup_path.exists():
        backup_path.unlink()


def copy_report_assets(
    report_assets: pd.DataFrame,
    output_dir: Path,
) -> list[dict[str, str]]:
    """Stage and publish selected report image assets."""
    static_assets = stage_report_assets(report_assets, output_dir)
    publish_staged_report_assets(output_dir)
    return static_assets


def _final_reports_dir(output_dir: Path) -> Path:
    return output_dir / "static" / "reports"


def _staging_reports_dir(output_dir: Path) -> Path:
    return output_dir / "static" / "reports.tmp"


def _backup_reports_dir(output_dir: Path) -> Path:
    return output_dir / "static" / "reports.backup"


def _database_backup_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".backup")


def _remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
