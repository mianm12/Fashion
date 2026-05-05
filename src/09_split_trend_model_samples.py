from __future__ import annotations

from fashion_trend import log
from fashion_trend.config import (
    PATH,
    TREND_SPLIT_TEST_WEEKS,
    TREND_SPLIT_VALID_WEEKS,
)
from fashion_trend.trend import (
    build_trend_model_split_frames,
    build_trend_model_split_metadata,
    validate_trend_model_split_frames,
    write_json,
    write_trend_parquet,
)

LOG_SOURCE = "trend-model-split"


def split_trend_model_samples() -> dict[str, object]:
    input_path = PATH["features_trend_model_samples"]
    log.info(f"输入趋势样本表: {input_path}", source=LOG_SOURCE)
    if not input_path.exists():
        raise FileNotFoundError(f"趋势样本表不存在: {input_path}")

    import pandas as pd

    trend_model_samples = pd.read_parquet(input_path)
    split_frames = build_trend_model_split_frames(
        trend_model_samples,
        valid_weeks=TREND_SPLIT_VALID_WEEKS,
        test_weeks=TREND_SPLIT_TEST_WEEKS,
    )
    validate_trend_model_split_frames(split_frames, trend_model_samples)

    output_paths = {
        "train": PATH["features_trend_model_samples_train"],
        "valid": PATH["features_trend_model_samples_valid"],
        "test": PATH["features_trend_model_samples_test"],
    }
    for split_name, split_frame in split_frames.items():
        write_trend_parquet(split_frame, output_paths[split_name])

    metadata = build_trend_model_split_metadata(
        split_frames,
        input_path=input_path,
        output_paths=output_paths,
        valid_weeks=TREND_SPLIT_VALID_WEEKS,
        test_weeks=TREND_SPLIT_TEST_WEEKS,
    )
    write_json(metadata, PATH["features_trend_model_samples_split_metadata"])
    return metadata


def main() -> int:
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
        f"切分元数据: {PATH['features_trend_model_samples_split_metadata']}",
        source=LOG_SOURCE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
