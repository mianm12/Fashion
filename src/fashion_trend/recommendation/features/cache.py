from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fashion_trend.foundation.io import write_json_atomic, write_parquet_atomic
from fashion_trend.recommendation.freshness import (
    assert_fresh_metadata,
    build_artifact_metadata,
)
from fashion_trend.recommendation.paths import (
    FEATURE_CACHE_METADATA_PATH,
    feature_cache_partition_metadata_path,
    feature_cache_partition_path,
)
from fashion_trend.recommendation.ranking.features import build_ranking_features

FEATURE_NAMES = (
    "popularity_scores",
    "recent_scores",
    "similarity_scores",
    "trend_scores",
    "candidate_seen_flags",
    "recommendable_pool",
)
WINDOW_COLUMNS = ["split", "cutoff_week", "label_week"]
CANDIDATE_KEY_COLUMNS = [
    *WINDOW_COLUMNS,
    "strategy",
    "customer_id",
    "article_id",
]
SEEN_FLAG_COLUMNS = [*CANDIDATE_KEY_COLUMNS, "seen"]
ENHANCED_SEEN_FLAG_COLUMNS = [
    *SEEN_FLAG_COLUMNS,
    "is_seen",
    "allow_seen",
    "has_reorder_source",
]
ARTICLE_SCORE_FEATURES = {
    "popularity_scores": "pop_score",
    "recent_scores": "recent_score",
    "trend_scores": "trend_score",
}
CUSTOMER_ARTICLE_SCORE_FEATURES = {"similarity_scores": "sim_score"}
FEATURE_INPUT_KEYS = {
    "candidate_seen_flags": (
        "weekly_transactions",
        "candidate_items",
        "candidate_metadata",
    ),
    "popularity_scores": (
        "weekly_transactions",
        "candidate_items",
        "candidate_metadata",
    ),
    "recent_scores": (
        "weekly_transactions",
        "candidate_items",
        "candidate_metadata",
    ),
    "similarity_scores": (
        "article_attributes",
        "user_profile",
        "candidate_items",
        "candidate_metadata",
    ),
    "trend_scores": (
        "article_attributes",
        "trend_predictions",
        "candidate_items",
        "candidate_metadata",
    ),
}
FEATURE_CACHE_SCHEMA_VERSION = 1
FEATURE_CACHE_ALGORITHM_VERSION = "recommendation-feature-cache-v1"
RECOMMENDABLE_POOL_SCHEMA_VERSION = 1
RECOMMENDABLE_POOL_ALGORITHM_VERSION = "recommendable-pool-cache-v1"
RECOMMENDABLE_POOL_MANIFEST_ALGORITHM_VERSION = (
    "recommendation-feature-cache-manifest-v1"
)
RECOMMENDABLE_POOL_MANIFEST_KEY = "feature:recommendable_pool:strategy:all"
RECOMMENDABLE_POOL_COLUMNS = ["split", "cutoff_week", "label_week", "article_id"]


def build_candidate_seen_flags(
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Return only candidate pairs seen by cutoff week, never full history."""
    _require_columns(candidates, CANDIDATE_KEY_COLUMNS, "candidates")
    seen_flag_columns = _seen_flag_columns(candidates)
    if _requires_source_seen_policy(candidates):
        _require_columns(
            candidates,
            ("allow_seen", "has_reorder_source"),
            "enhanced candidates",
        )
    _require_columns(
        transactions, ("customer_id", "article_id", "week_id"), "transactions"
    )

    if candidates.empty or transactions.empty:
        return pd.DataFrame(columns=seen_flag_columns)

    candidate_columns = list(CANDIDATE_KEY_COLUMNS)
    if _requires_source_seen_policy(candidates):
        candidate_columns.extend(["allow_seen", "has_reorder_source"])
    candidate_frame = _with_string_ids(candidates.loc[:, candidate_columns])
    candidate_frame["_candidate_order"] = range(len(candidate_frame))
    transaction_frame = _with_string_ids(transactions)
    transaction_frame["week_id"] = pd.to_numeric(
        transaction_frame["week_id"],
        errors="raise",
    ).astype(int)

    frames: list[pd.DataFrame] = []
    for window in candidate_frame[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        window_candidates = _frame_for_window(candidate_frame, window)
        seen_pairs = (
            transaction_frame.loc[
                transaction_frame["week_id"] <= int(window["cutoff_week"]),
                ["customer_id", "article_id"],
            ]
            .drop_duplicates()
            .assign(seen=True)
        )
        if seen_pairs.empty:
            continue
        matched = window_candidates.merge(
            seen_pairs,
            on=["customer_id", "article_id"],
            how="inner",
        )
        if not matched.empty:
            if _requires_source_seen_policy(candidates):
                matched["is_seen"] = matched["seen"]
            frames.append(matched)

    if not frames:
        return pd.DataFrame(columns=seen_flag_columns)

    result = pd.concat(frames, ignore_index=True).drop_duplicates(seen_flag_columns)
    result = result.sort_values("_candidate_order", kind="mergesort")
    return result.loc[:, seen_flag_columns].reset_index(drop=True)


def build_recommendable_pool(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-window article pools from history through each cutoff week."""
    _require_columns(transactions, ("article_id", "week_id"), "transactions")
    _require_columns(windows, WINDOW_COLUMNS, "windows")

    frames: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        active = (
            transactions.loc[
                transactions["week_id"] <= int(window.cutoff_week),
                ["article_id"],
            ]
            .drop_duplicates()
            .copy()
        )
        active["article_id"] = active["article_id"].astype("string")
        active = active.assign(
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
            label_week=int(window.label_week),
        )
        frames.append(active.loc[:, RECOMMENDABLE_POOL_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=RECOMMENDABLE_POOL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def recommendable_pool_partition_path(split: str, cutoff_week: int) -> Path:
    return feature_cache_partition_path(
        "recommendable_pool",
        strategy="all",
        split=split,
        cutoff_week=cutoff_week,
    )


def write_recommendable_pool_cache(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
    input_artifacts: dict[str, str],
) -> dict[str, object]:
    """Write recommendable pool partitions and merge their manifest entry."""
    _require_columns(windows, WINDOW_COLUMNS, "windows")
    partitions: list[dict[str, object]] = []
    output_artifacts: dict[str, str] = {
        "feature_cache_metadata": str(FEATURE_CACHE_METADATA_PATH),
    }
    row_counts: dict[str, int] = {}

    for index, window in enumerate(windows.itertuples(index=False)):
        window_frame = pd.DataFrame([window._asdict()])
        frame = build_recommendable_pool(transactions, window_frame)
        partition_path = recommendable_pool_partition_path(
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
        )
        metadata_path = feature_cache_partition_metadata_path(
            "recommendable_pool",
            strategy="all",
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
        )
        write_parquet_atomic(frame, partition_path)

        partition_metadata = build_artifact_metadata(
            name="recommendable_pool_cache",
            input_artifacts=dict(input_artifacts),
            output_artifacts={
                "partition": str(partition_path),
                "partition_metadata": str(metadata_path),
            },
            schema_version=RECOMMENDABLE_POOL_SCHEMA_VERSION,
            algorithm_version=RECOMMENDABLE_POOL_ALGORITHM_VERSION,
            config=_recommendable_pool_config(window),
            row_counts={"rows": int(len(frame))},
        )
        write_json_atomic(partition_metadata, metadata_path)

        partition_key = f"recommendable_pool_partition_{index:04d}"
        metadata_key = f"recommendable_pool_metadata_{index:04d}"
        output_artifacts[partition_key] = str(partition_path)
        output_artifacts[metadata_key] = str(metadata_path)
        row_counts[partition_key] = int(len(frame))
        partitions.append(
            {
                "split": str(window.split),
                "cutoff_week": int(window.cutoff_week),
                "label_week": int(window.label_week),
                "path": str(partition_path),
                "metadata_path": str(metadata_path),
                "rows": int(len(frame)),
            }
        )

    global_manifest = build_artifact_metadata(
        name="recommendation_feature_cache_manifest",
        input_artifacts=dict(input_artifacts),
        output_artifacts=output_artifacts,
        schema_version=RECOMMENDABLE_POOL_SCHEMA_VERSION,
        algorithm_version=RECOMMENDABLE_POOL_MANIFEST_ALGORITHM_VERSION,
        config={"feature_name": "recommendable_pool", "strategy": "all"},
        row_counts=row_counts,
    )
    global_manifest["manifest"] = {
        "feature_name": "recommendable_pool",
        "strategy": "all",
        "partitions": partitions,
    }
    return update_feature_cache_manifest(
        manifest_key=RECOMMENDABLE_POOL_MANIFEST_KEY,
        payload=global_manifest,
    )


def read_recommendable_pool_cache(windows: pd.DataFrame) -> pd.DataFrame:
    _require_columns(windows, WINDOW_COLUMNS, "windows")
    frames: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        path = recommendable_pool_partition_path(
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
        )
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=RECOMMENDABLE_POOL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def recommendable_pool_cache_exists(windows: pd.DataFrame) -> bool:
    _require_columns(windows, WINDOW_COLUMNS, "windows")
    return all(
        recommendable_pool_partition_path(
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
        ).exists()
        and feature_cache_partition_metadata_path(
            "recommendable_pool",
            strategy="all",
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
        ).exists()
        for window in windows.itertuples(index=False)
    )


def recommendable_pool_cache_fresh(
    windows: pd.DataFrame,
    input_artifacts: dict[str, str],
) -> bool:
    _require_columns(windows, WINDOW_COLUMNS, "windows")
    if not recommendable_pool_cache_exists(windows):
        return False

    for window in windows.itertuples(index=False):
        metadata_path = feature_cache_partition_metadata_path(
            "recommendable_pool",
            strategy="all",
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
        )
        partition_path = recommendable_pool_partition_path(
            split=str(window.split),
            cutoff_week=int(window.cutoff_week),
        )
        try:
            assert_fresh_metadata(
                metadata_path=metadata_path,
                expected_input_artifacts=dict(input_artifacts),
                expected_output_artifacts={
                    "partition": str(partition_path),
                    "partition_metadata": str(metadata_path),
                },
                expected_schema_version=RECOMMENDABLE_POOL_SCHEMA_VERSION,
                expected_algorithm_version=RECOMMENDABLE_POOL_ALGORITHM_VERSION,
                expected_config=_recommendable_pool_config(window),
                stale_message=lambda reason: reason,
            )
        except (json.JSONDecodeError, RuntimeError):
            return False
    return True


def update_feature_cache_manifest(
    manifest_key: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Merge one strategy manifest entry into the global feature cache manifest."""
    manifest = _read_manifest(FEATURE_CACHE_METADATA_PATH)
    entries = manifest.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("feature cache manifest entries must be an object")
    entries[manifest_key] = _json_compatible(dict(payload))
    write_json_atomic(manifest, FEATURE_CACHE_METADATA_PATH)
    return manifest


def build_and_write_feature_cache_for_strategy(
    strategy: str,
    candidates: pd.DataFrame,
    transactions: pd.DataFrame,
    article_attributes: pd.DataFrame,
    user_profile: pd.DataFrame | None,
    trend_predictions: pd.DataFrame | None,
    input_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build strategy/window-scoped feature cache partitions for candidates."""
    _require_columns(candidates, CANDIDATE_KEY_COLUMNS, "candidates")
    _validate_candidate_strategy(candidates, strategy)

    partitions: dict[str, list[dict[str, object]]] = {}
    output_artifacts: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    for window in candidates[WINDOW_COLUMNS].drop_duplicates().to_dict("records"):
        partition_candidates = _frame_for_window(candidates, window)
        if partition_candidates.empty:
            continue

        feature_frame = build_ranking_features(
            partition_candidates,
            transactions,
            article_attributes,
            user_profile,
            trend_predictions,
        )
        seen_flags = build_candidate_seen_flags(partition_candidates, transactions)
        if len(seen_flags) > len(partition_candidates):
            raise ValueError("candidate_seen_flags rows exceed candidate rows")

        partition_frames = _feature_partition_frames(feature_frame, seen_flags)
        for feature_name, frame in partition_frames.items():
            partition_path = feature_cache_partition_path(
                feature_name,
                strategy=strategy,
                split=str(window["split"]),
                cutoff_week=int(window["cutoff_week"]),
            )
            write_parquet_atomic(frame, partition_path)
            metadata_path = feature_cache_partition_metadata_path(
                feature_name,
                strategy=strategy,
                split=str(window["split"]),
                cutoff_week=int(window["cutoff_week"]),
            )
            feature_input_artifacts = _feature_input_artifacts(
                feature_name,
                input_paths or {},
            )
            metadata = build_artifact_metadata(
                name=f"recommendation_feature_cache_{feature_name}",
                input_artifacts=feature_input_artifacts,
                output_artifacts={
                    "partition": str(partition_path),
                    "partition_metadata": str(metadata_path),
                },
                schema_version=FEATURE_CACHE_SCHEMA_VERSION,
                algorithm_version=FEATURE_CACHE_ALGORITHM_VERSION,
                config={
                    "feature_name": feature_name,
                    "strategy": strategy,
                    "split": str(window["split"]),
                    "cutoff_week": int(window["cutoff_week"]),
                    "label_week": int(window["label_week"]),
                },
                row_counts={"rows": int(len(frame))},
            )
            write_json_atomic(metadata, metadata_path)

            partition_key = _partition_manifest_key(feature_name, window)
            partitions.setdefault(feature_name, []).append(
                {
                    "split": str(window["split"]),
                    "cutoff_week": int(window["cutoff_week"]),
                    "label_week": int(window["label_week"]),
                    "path": str(partition_path),
                    "metadata_path": str(metadata_path),
                    "rows": int(len(frame)),
                }
            )
            output_artifacts[f"{partition_key}:partition"] = str(partition_path)
            output_artifacts[f"{partition_key}:partition_metadata"] = str(metadata_path)
            row_counts[partition_key] = int(len(frame))

    output_artifacts["feature_cache_metadata"] = str(FEATURE_CACHE_METADATA_PATH)
    row_counts["candidate_rows"] = int(len(candidates))
    global_manifest = build_artifact_metadata(
        name="recommendation_feature_cache",
        input_artifacts=dict(input_paths or {}),
        output_artifacts=output_artifacts,
        schema_version=FEATURE_CACHE_SCHEMA_VERSION,
        algorithm_version=FEATURE_CACHE_ALGORITHM_VERSION,
        config={"strategy": strategy},
        row_counts=row_counts,
    )
    global_manifest["manifest"] = {
        "strategy": strategy,
        "features": list(FEATURE_NAMES),
        "partitions": partitions,
    }
    return update_feature_cache_manifest(
        manifest_key=f"strategy:{strategy}",
        payload=global_manifest,
    )


def _feature_partition_frames(
    feature_frame: pd.DataFrame,
    seen_flags: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frames = {
        feature_name: feature_frame.loc[
            :,
            [*WINDOW_COLUMNS, "strategy", "article_id", score_column],
        ].copy()
        for feature_name, score_column in ARTICLE_SCORE_FEATURES.items()
    }
    frames = {
        feature_name: frame.drop_duplicates().reset_index(drop=True)
        for feature_name, frame in frames.items()
    }
    frames.update(
        {
            feature_name: feature_frame.loc[
                :,
                [*CANDIDATE_KEY_COLUMNS, score_column],
            ]
            .drop_duplicates()
            .reset_index(drop=True)
            for feature_name, score_column in CUSTOMER_ARTICLE_SCORE_FEATURES.items()
        }
    )
    seen_flag_columns = _seen_flag_columns(seen_flags)
    frames["candidate_seen_flags"] = (
        seen_flags.loc[:, seen_flag_columns].drop_duplicates().reset_index(drop=True)
    )
    return frames


def _validate_candidate_strategy(candidates: pd.DataFrame, strategy: str) -> None:
    if candidates.empty:
        return
    candidate_strategies = set(candidates["strategy"].astype(str))
    if candidate_strategies != {strategy}:
        raise ValueError(
            "candidates strategy must match feature cache strategy: "
            f"expected {strategy}, found {sorted(candidate_strategies)}"
        )


def _partition_manifest_key(feature_name: str, window: dict[str, object]) -> str:
    return (
        f"{feature_name}:split={window['split']}:"
        f"cutoff_week={int(window['cutoff_week'])}"
    )


def _feature_input_artifacts(
    feature_name: str,
    input_paths: dict[str, str],
) -> dict[str, str]:
    keys = FEATURE_INPUT_KEYS.get(feature_name, ())
    return {key: input_paths[key] for key in keys if key in input_paths}


def _recommendable_pool_config(window) -> dict[str, object]:
    return {
        "feature_name": "recommendable_pool",
        "strategy": "all",
        "split": str(window.split),
        "cutoff_week": int(window.cutoff_week),
        "label_week": int(window.label_week),
    }


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"entries": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("feature cache manifest must be a JSON object")
    payload.setdefault("entries", {})
    return payload


def _require_columns(
    dataframe: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    name: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _seen_flag_columns(dataframe: pd.DataFrame) -> list[str]:
    if _requires_source_seen_policy(dataframe):
        return ENHANCED_SEEN_FLAG_COLUMNS
    return SEEN_FLAG_COLUMNS


def _requires_source_seen_policy(dataframe: pd.DataFrame) -> bool:
    if {"allow_seen", "has_reorder_source"}.issubset(dataframe.columns):
        return True
    if "strategy" not in dataframe.columns or dataframe.empty:
        return False
    return "enhanced_default" in set(dataframe["strategy"].astype(str))


def _frame_for_window(
    frame: pd.DataFrame,
    window: dict[str, object],
) -> pd.DataFrame:
    mask = (
        (frame["split"] == window["split"])
        & (frame["cutoff_week"] == window["cutoff_week"])
        & (frame["label_week"] == window["label_week"])
    )
    return frame.loc[mask].copy()


def _with_string_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in ("article_id", "customer_id"):
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
