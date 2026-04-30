from __future__ import annotations

import argparse
import sys
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import kagglehub

from fashion_trend.config import DEFAULT_COMPETITION, RAW_DIR  # noqa: E402

Downloader = Callable[..., str | Path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，生成下载数据集所需的运行配置。

    Args:
        argv (Sequence[str] | None, optional): 命令行参数序列；为 None 时从
            sys.argv 读取实际命令行输入。Defaults to None.

    Returns:
        argparse.Namespace: 解析后的参数对象，包含 competition、data_dir、unzip
            和 force 等下载配置。
    """
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
        default=RAW_DIR,
        help=f"Base data directory. Default: {RAW_DIR}",
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
    """根据数据根目录和 Kaggle 比赛 slug 生成最终保存目录。

    Args:
        data_dir (Path): 用于保存原始数据的根目录。
        competition (str): Kaggle competition slug。

    Returns:
        Path: 指向该比赛数据集本地保存目录的路径。
    """
    return data_dir / competition


def should_skip_download(destination: Path, force: bool) -> bool:
    """判断目标目录已有内容且未强制下载时是否应跳过下载。

    Args:
        destination (Path): 数据集目标保存目录。
        force (bool): 是否强制重新执行 Kaggle 下载。

    Returns:
        bool: 如果目标目录已有内容且 force 为 False，则返回 True；否则返回
            False。
    """
    return not force and destination.exists() and any(destination.iterdir())


def extract_zip_files(destination: Path) -> list[Path]:
    """解压目标目录下所有 zip 文件，并返回解压得到的文件路径列表。

    Args:
        destination (Path): 包含 zip 文件且用于接收解压内容的目录。

    Returns:
        list[Path]: 每个 zip 归档内成员解压后的目标路径列表。

    Raises:
        RuntimeError: 当 zip 成员路径会逃逸出 destination 目录时抛出。
    """
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
    """调用 KaggleHub Python API 下载指定比赛数据集。

    Args:
        competition (str): Kaggle competition slug。
        output_dir (str): KaggleHub 写入下载文件的目标目录。
        force_download (bool): 是否强制重新下载数据集。

    Returns:
        str | Path: KaggleHub 返回的本地下载路径。
    """
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
    """下载指定 Kaggle 比赛数据集，并按配置执行解压和跳过逻辑。

    Args:
        competition (str, optional): Kaggle competition slug。Defaults to
            DEFAULT_COMPETITION.
        data_dir (Path, optional): 保存原始数据的根目录。Defaults to
            RAW_DIR.
        unzip (bool, optional): 下载后是否解压目标目录下的 zip 文件。Defaults
            to True.
        force (bool, optional): 是否忽略已有文件并强制重新下载。Defaults to
            False.
        downloader (Downloader, optional): 执行实际下载的可注入函数，便于测试
            或替换下载实现。Defaults to kagglehub_competition_download.

    Returns:
        Path: 数据集最终可用的本地路径。

    Raises:
        RuntimeError: 当解压阶段发现 zip 成员路径不安全时抛出。
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


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口函数，负责解析参数、执行下载并返回进程退出码。

    Args:
        argv (Sequence[str] | None, optional): 命令行参数序列；为 None 时使用
            sys.argv 中的实际输入。Defaults to None.

    Returns:
        int: 进程退出码；0 表示执行成功，1 表示下载或解压阶段出现可处理错误。
    """
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
