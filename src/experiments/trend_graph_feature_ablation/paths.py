from __future__ import annotations

from pathlib import Path

from experiments.trend_graph_feature_ablation.contracts import (
    ABLATION_EXPERIMENT_ID,
    ABLATION_VARIANTS,
    RUN_ARTIFACT_FILENAMES,
)
from fashion_trend.foundation.artifacts import validate_safe_path_segment
from fashion_trend.foundation.paths import OUTPUT_DIR

EXPERIMENT_ROOT = OUTPUT_DIR / "experiments" / ABLATION_EXPERIMENT_ID
FEATURES_DIR = EXPERIMENT_ROOT / "features"
RUNS_DIR = EXPERIMENT_ROOT / "runs"
DOCS_DIR = EXPERIMENT_ROOT / "docs"

FEATURE_GROUPS_PATH = FEATURES_DIR / "feature_groups.json"
FEATURE_SCHEMA_PATH = FEATURES_DIR / "feature_schema.json"
ROW_ALIGNMENT_CHECK_PATH = FEATURES_DIR / "row_alignment_check.json"
INPUT_HASHES_PATH = FEATURES_DIR / "input_hashes.json"
METRICS_SUMMARY_CSV_PATH = EXPERIMENT_ROOT / "metrics_summary.csv"
METRICS_SUMMARY_MD_PATH = EXPERIMENT_ROOT / "metrics_summary.md"
EXPERIMENT_DOC_PATH = EXPERIMENT_ROOT / "experiment.md"
MANIFEST_PATH = EXPERIMENT_ROOT / "manifest.json"

SAMPLE_SPLITS: tuple[str, ...] = ("all", "train", "valid", "test")


def enhanced_sample_path(split: str) -> Path:
    """返回增强样本 parquet 路径，不创建目录。"""

    _validate_split(split)
    return FEATURES_DIR / f"enhanced_samples_{split}.parquet"


def run_dir(variant: str) -> Path:
    """返回单个 ablation variant 的正式 run 目录。"""

    _validate_variant(variant)
    return RUNS_DIR / variant


def staging_run_dir(variant: str) -> Path:
    """返回单个 ablation variant 的临时 staging run 目录。"""

    _validate_variant(variant)
    return RUNS_DIR / f".{variant}.staging"


def run_artifact_path(variant: str, filename: str) -> Path:
    """返回单个 run artifact 路径，不创建目录。"""

    _validate_run_artifact_filename(filename)
    return run_dir(variant) / filename


def _validate_split(split: str) -> None:
    validate_safe_path_segment(split, "样本 split")
    if split not in SAMPLE_SPLITS:
        raise ValueError(f"未知趋势图消融样本 split: {split}")


def _validate_variant(variant: str) -> None:
    validate_safe_path_segment(variant, "消融 variant")
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"未知趋势图消融 variant: {variant}")


def _validate_run_artifact_filename(filename: str) -> None:
    validate_safe_path_segment(filename, "run artifact 文件名")
    if filename not in RUN_ARTIFACT_FILENAMES:
        raise ValueError(f"未知趋势图消融 run artifact 文件名: {filename}")
