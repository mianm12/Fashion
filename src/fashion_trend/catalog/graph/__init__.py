from __future__ import annotations

from pathlib import Path

from fashion_trend.catalog.graph.builders import build_attribute_graph_frames
from fashion_trend.catalog.graph.publishing import publish_graph_frames
from fashion_trend.catalog.readers import read_clean_articles


def build_attribute_graph_files(
    clean_articles_path: Path,
    graph_dir: Path,
) -> dict[str, int]:
    clean_articles = read_clean_articles(clean_articles_path)
    graph_frames = build_attribute_graph_frames(clean_articles)
    graph_dir.mkdir(parents=True, exist_ok=True)
    publish_graph_frames(graph_frames, graph_dir)
    return {
        graph_name: len(graph_frame) for graph_name, graph_frame in graph_frames.items()
    }
