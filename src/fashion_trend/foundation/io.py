from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


def remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def write_json_atomic(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)


def write_csv_atomic(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_csv(tmp_path, index=False, quoting=csv.QUOTE_ALL)
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)


def write_parquet_atomic(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        dataframe.to_parquet(tmp_path, index=False)
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)


def write_binary_atomic(payload: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(output_path)
    finally:
        remove_file_if_exists(tmp_path)
