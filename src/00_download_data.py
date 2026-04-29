from __future__ import annotations

import argparse
import sys
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import kagglehub


DEFAULT_COMPETITION = "h-and-m-personalized-fashion-recommendations"
DEFAULT_DATA_DIR = Path("data/raw")

Downloader = Callable[..., str | Path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，返回下载数据集所需的配置。"""
    parser = argparse.ArgumentParser(
        description="Download a Kaggle competition dataset into the project data directory."
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug. Default: {DEFAULT_COMPETITION}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_false",
        dest="unzip",
        help="Keep downloaded zip files without extracting them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the Kaggle download even when destination files already exist.",
    )
    parser.set_defaults(unzip=True)
    return parser.parse_args(argv)


def competition_target_dir(data_dir: Path, competition: str) -> Path:
    """根据数据根目录和 Kaggle 比赛 slug 生成最终保存目录。"""
    return data_dir / competition


def should_skip_download(destination: Path, force: bool) -> bool:
    """判断目标目录已有内容且未强制下载时，是否应跳过 Kaggle 下载。"""
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
                    raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")
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
    data_dir: Path = DEFAULT_DATA_DIR,
    unzip: bool = True,
    force: bool = False,
    downloader: Downloader = kagglehub_competition_download,
) -> Path:
    """下载指定 Kaggle 比赛数据集到数据目录，并按配置执行解压和跳过逻辑。"""
    destination = competition_target_dir(data_dir, competition)
    destination.mkdir(parents=True, exist_ok=True)

    if should_skip_download(destination, force):
        print(f"Dataset already exists in {destination}. Use --force to download again.")
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


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口函数：解析参数、执行下载，并返回进程退出码。"""
    args = parse_args(argv)
    try:
        destination = download_competition(
            competition=args.competition,
            data_dir=args.data_dir,
            unzip=args.unzip,
            force=args.force,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Dataset ready at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
