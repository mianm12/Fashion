from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from fashion_trend.catalog.graph.schema import GRAPH_OUTPUT_FILENAMES
from fashion_trend.foundation.io import remove_file_if_exists


def cleanup_graph_publish_files(paths_by_name: dict[str, Path]) -> None:
    for path in paths_by_name.values():
        remove_file_if_exists(path)


def rollback_graph_outputs(
    output_paths: dict[str, Path],
    backup_paths: dict[str, Path],
    changed_graph_names: set[str],
) -> None:
    for graph_name in changed_graph_names:
        output_path = output_paths[graph_name]
        if output_path.is_file():
            output_path.unlink()

        backup_path = backup_paths[graph_name]
        if backup_path.exists():
            backup_path.replace(output_path)


def write_graph_frame_temp(dataframe: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(tmp_output_path, index=False, quoting=csv.QUOTE_ALL)
    return tmp_output_path


def publish_graph_frames(
    graph_frames: dict[str, pd.DataFrame],
    graph_dir: Path,
) -> None:
    output_paths = {
        graph_name: graph_dir / GRAPH_OUTPUT_FILENAMES[graph_name]
        for graph_name in graph_frames
    }
    temp_paths = {
        graph_name: output_path.with_suffix(output_path.suffix + ".tmp")
        for graph_name, output_path in output_paths.items()
    }
    backup_paths = {
        graph_name: output_path.with_suffix(output_path.suffix + ".bak")
        for graph_name, output_path in output_paths.items()
    }
    backed_up_graph_names: set[str] = set()
    published_graph_names: set[str] = set()

    try:
        for graph_name, graph_frame in graph_frames.items():
            temp_paths[graph_name] = write_graph_frame_temp(
                graph_frame, output_paths[graph_name]
            )

        for graph_name, output_path in output_paths.items():
            remove_file_if_exists(backup_paths[graph_name])
            if output_path.is_file():
                output_path.replace(backup_paths[graph_name])
                backed_up_graph_names.add(graph_name)

        for graph_name, temp_path in temp_paths.items():
            temp_path.replace(output_paths[graph_name])
            published_graph_names.add(graph_name)
    except Exception:
        try:
            rollback_graph_outputs(
                output_paths,
                backup_paths,
                changed_graph_names=backed_up_graph_names | published_graph_names,
            )
        finally:
            cleanup_graph_publish_files(temp_paths)
            cleanup_graph_publish_files(backup_paths)
        raise

    cleanup_graph_publish_files(temp_paths)
    cleanup_graph_publish_files(backup_paths)
