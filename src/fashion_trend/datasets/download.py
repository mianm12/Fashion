from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import kagglehub

from fashion_trend.foundation.paths import DEFAULT_COMPETITION, RAW_DIR

Downloader = Callable[..., str | Path]


def competition_target_dir(data_dir: Path, competition: str) -> Path:
    """根据数据根目录和 Kaggle 比赛 slug 生成最终保存目录。"""
    return data_dir / competition


def should_skip_download(destination: Path, force: bool) -> bool:
    """判断目标目录已有内容且未强制下载时是否应跳过下载。"""
    return not force and destination.exists() and any(destination.iterdir())


def extract_zip_files(destination: Path) -> list[Path]:
    """解压目标目录下所有 zip 文件，并返回解压得到的文件路径列表。"""
    extracted_paths: list[Path] = []
    destination_root = destination.resolve()

    for zip_path in sorted(destination.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination_root):
                    raise RuntimeError(
                        f"Refusing to extract unsafe zip member: {member.filename}"
                    )
            archive.extractall(destination)
            extracted_paths.extend(destination / name for name in archive.namelist())

    return extracted_paths


def kagglehub_competition_download(
    competition: str,
    output_dir: str,
    force_download: bool,
) -> str | Path:
    """调用 KaggleHub Python API 下载指定比赛数据集。"""
    return kagglehub.competition_download(
        competition,
        output_dir=output_dir,
        force_download=force_download,
    )


def download_competition(
    competition: str = DEFAULT_COMPETITION,
    data_dir: Path = RAW_DIR,
    unzip: bool = True,
    force: bool = False,
    downloader: Downloader = kagglehub_competition_download,
) -> Path:
    """下载指定 Kaggle 比赛数据集，并按配置执行解压和跳过逻辑。"""
    destination = competition_target_dir(data_dir, competition)
    destination.mkdir(parents=True, exist_ok=True)

    if should_skip_download(destination, force):
        print(
            f"Dataset already exists in {destination}. Use --force to download again."
        )
        if unzip:
            extract_zip_files(destination)
        return destination

    downloaded_path = Path(
        downloader(
            competition,
            output_dir=str(destination),
            force_download=force,
        )
    )

    if unzip:
        extract_zip_files(downloaded_path)

    return downloaded_path
