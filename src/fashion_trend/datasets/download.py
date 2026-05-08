from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import kagglehub

from fashion_trend.datasets.paths import DEFAULT_COMPETITION
from fashion_trend.foundation.paths import RAW_DIR

Downloader = Callable[..., str | Path]


def competition_target_dir(data_dir: Path, competition: str) -> Path:
    """根据数据根目录和 Kaggle 比赛 slug 派生原始数据保存目录。"""
    return data_dir / competition


def should_skip_download(destination: Path, force: bool) -> bool:
    """判断目标目录已有内容且未传入强制下载时是否跳过 Kaggle 下载。"""
    return not force and destination.exists() and any(destination.iterdir())


def extract_zip_files(destination: Path) -> list[Path]:
    """解压目标目录下所有 zip 文件，并拒绝逃逸目标目录的成员路径。"""
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
    """下载指定 Kaggle 比赛数据集到原始数据目录。

    参数:
        competition: Kaggle 比赛 slug，默认指向 H&M Personalized Fashion。
        data_dir: 原始数据根目录，比赛 slug 会作为其下一级目录。
        unzip: 下载或跳过下载后是否解压目录内 zip 文件。
        force: 是否强制重新调用下载器覆盖已有数据判断。
        downloader: 实际下载函数，测试可替换为本地桩函数。

    返回:
        下载器返回路径或已存在目标目录路径。

    边界:
        当目标目录已有文件且 `force=False` 时跳过下载；解压时复用
        `extract_zip_files()` 的 zip 成员路径逃逸防护。
    """
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
