from __future__ import annotations

from pandas import read_parquet

from fashion_trend.foundation import logging as log
from fashion_trend.foundation.io import write_json_atomic, write_parquet_atomic
from fashion_trend.trend.paths import (
    TREND_MODEL_SAMPLES_PATH,
    TREND_MODEL_SAMPLES_SPLIT_METADATA_PATH,
    TREND_MODEL_SAMPLES_TEST_PATH,
    TREND_MODEL_SAMPLES_TRAIN_PATH,
    TREND_MODEL_SAMPLES_VALID_PATH,
    TREND_SPLIT_TEST_WEEKS,
    TREND_SPLIT_VALID_WEEKS,
)
from fashion_trend.trend.splits import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    validate_trend_model_split_frames,
)

LOG_SOURCE = "trend-model-split"


def split_trend_model_samples() -> dict[str, object]:
    """编排趋势样本时间切分流程。

    流程:
        1. 读取 trend_model_samples.parquet。
        2. 按固定 valid/test 周数构建 train、valid、test 切分。
        3. 校验切分覆盖关系并写出 split parquet。
        4. 写出切分元数据并返回完整摘要。

    返回:
        dict[str, object]: 切分行数、周范围、输出路径和参数摘要。
    """
    input_path = TREND_MODEL_SAMPLES_PATH
    log.info(f"输入趋势样本表: {input_path}", source=LOG_SOURCE)
    log.info(
        f"关键参数: valid_weeks={TREND_SPLIT_VALID_WEEKS}, "
        f"test_weeks={TREND_SPLIT_TEST_WEEKS}",
        source=LOG_SOURCE,
    )
    log.info(
        "业务阶段: trend_model_samples.parquet -> train/valid/test split parquet",
        source=LOG_SOURCE,
    )
    if not input_path.exists():
        raise FileNotFoundError(f"趋势样本表不存在: {input_path}")

    trend_model_samples = read_parquet(input_path)
    split_frames = build_trend_model_split_frames(
        trend_model_samples,
        valid_weeks=TREND_SPLIT_VALID_WEEKS,
        test_weeks=TREND_SPLIT_TEST_WEEKS,
    )
    validate_trend_model_split_frames(split_frames, trend_model_samples)

    output_paths = {
        "train": TREND_MODEL_SAMPLES_TRAIN_PATH,
        "valid": TREND_MODEL_SAMPLES_VALID_PATH,
        "test": TREND_MODEL_SAMPLES_TEST_PATH,
    }
    for split_name, split_frame in split_frames.items():
        write_parquet_atomic(split_frame, output_paths[split_name])

    metadata = build_trend_model_split_metadata(
        split_frames,
        input_path=input_path,
        output_paths=output_paths,
        valid_weeks=TREND_SPLIT_VALID_WEEKS,
        test_weeks=TREND_SPLIT_TEST_WEEKS,
    )
    write_json_atomic(metadata, TREND_MODEL_SAMPLES_SPLIT_METADATA_PATH)
    return metadata


def main() -> int:
    """趋势样本切分阶段入口，稳定写出 train/valid/test parquet 和元数据 JSON。"""
    try:
        metadata = split_trend_model_samples()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log.error(f"处理失败: {exc}", source=LOG_SOURCE)
        return 1

    for split_name in ("train", "valid", "test"):
        split_stats = metadata["splits"][split_name]
        log.info(
            f"{split_name} 样本: rows={split_stats['rows']:,}, "
            f"weeks={split_stats['weeks']:,}, "
            f"week_range={split_stats['week_min']}..{split_stats['week_max']}",
            source=LOG_SOURCE,
        )
        log.info(f"{split_name} 输出文件: {split_stats['path']}", source=LOG_SOURCE)
    log.info(
        f"切分元数据: {TREND_MODEL_SAMPLES_SPLIT_METADATA_PATH}",
        source=LOG_SOURCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
