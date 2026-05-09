from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import pandas as pd

EVALUATION_KEY_COLUMNS = ("split", "cutoff_week", "label_week", "customer_id")
WINDOW_KEY_COLUMNS = ("split", "cutoff_week", "label_week")


def parse_prediction_items(prediction: str, top_k: int) -> list[str]:
    """Parse a space-delimited recommendation string without changing IDs."""
    items = [item for item in prediction.split() if item]
    if len(items) > top_k:
        raise ValueError(f"prediction contains more than {top_k} items")
    if len(set(items)) != len(items):
        raise ValueError("prediction contains duplicate article_id values")
    return items


def apk(predicted: list[str], relevant: set[str], top_k: int) -> float:
    hits = 0
    score = 0.0
    for index, article_id in enumerate(predicted[:top_k], start=1):
        if article_id in relevant:
            hits += 1
            score += hits / index
    return score / min(len(relevant), top_k) if relevant else 0.0


def recall_at_k(predicted: list[str], relevant: set[str], top_k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(predicted[:top_k]) & relevant) / len(relevant)


def hit_rate_at_k(predicted: list[str], relevant: set[str], top_k: int) -> float:
    return float(bool(set(predicted[:top_k]) & relevant))


def ndcg_at_k(predicted: list[str], relevant: set[str], top_k: int) -> float:
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, article_id in enumerate(predicted[:top_k], start=1)
        if article_id in relevant
    )
    ideal_hits = min(len(relevant), top_k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    target_users: pd.DataFrame,
    labels: pd.DataFrame,
    recommendable_pool: pd.DataFrame,
    top_k: int,
    strict_missing_users: bool = False,
) -> dict[str, dict[str, object]]:
    """Evaluate Top-N recommendation rows against all eligible target users."""
    if target_users.empty:
        raise ValueError("target_users must not be empty")

    relevant_by_user = _build_relevant_sets(labels)
    predictions_by_user = _build_predictions_by_user(recommendations, top_k)
    coverage_by_window = _build_coverage_by_window(
        recommendations,
        recommendable_pool,
        top_k,
    )

    split_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "map_at_12": [],
            "recall_at_12": [],
            "hit_rate_at_12": [],
            "ndcg_at_12": [],
        }
    )
    user_counts: dict[str, int] = defaultdict(int)
    missing_counts: dict[str, int] = defaultdict(int)

    for row in target_users.itertuples(index=False):
        key = _user_key(row)
        split = str(key[0])
        user_counts[split] += 1
        predicted = predictions_by_user.get(key)
        if predicted is None:
            if strict_missing_users:
                raise ValueError(f"missing recommendation user: {key}")
            predicted = []
            missing_counts[split] += 1

        relevant = relevant_by_user.get(key, set())
        split_scores[split]["map_at_12"].append(apk(predicted, relevant, top_k))
        split_scores[split]["recall_at_12"].append(
            recall_at_k(predicted, relevant, top_k)
        )
        split_scores[split]["hit_rate_at_12"].append(
            hit_rate_at_k(predicted, relevant, top_k)
        )
        split_scores[split]["ndcg_at_12"].append(ndcg_at_k(predicted, relevant, top_k))

    metrics_by_split: dict[str, dict[str, object]] = {}
    for split in sorted(user_counts):
        split_windows = _sorted_target_windows(target_users, split)
        split_coverage_by_window = [
            {
                "cutoff_week": cutoff_week,
                "label_week": label_week,
                "coverage": coverage_by_window.get(
                    (split, cutoff_week, label_week),
                    0.0,
                ),
            }
            for cutoff_week, label_week in split_windows
        ]
        metrics_by_split[split] = {
            "map_at_12": _mean(split_scores[split]["map_at_12"]),
            "recall_at_12": _mean(split_scores[split]["recall_at_12"]),
            "hit_rate_at_12": _mean(split_scores[split]["hit_rate_at_12"]),
            "ndcg_at_12": _mean(split_scores[split]["ndcg_at_12"]),
            "coverage": _mean(
                [entry["coverage"] for entry in split_coverage_by_window]
            ),
            "user_count": user_counts[split],
            "missing_recommendation_user_count": missing_counts[split],
            "coverage_by_window": split_coverage_by_window,
        }
    return metrics_by_split


def _build_relevant_sets(labels: pd.DataFrame) -> dict[tuple[Any, ...], set[str]]:
    relevant_by_user: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    if labels.empty:
        return relevant_by_user

    deduped = labels.drop_duplicates(list(EVALUATION_KEY_COLUMNS) + ["article_id"])
    for row in deduped.itertuples(index=False):
        relevant_by_user[_user_key(row)].add(str(row.article_id))
    return relevant_by_user


def _build_predictions_by_user(
    recommendations: pd.DataFrame,
    top_k: int,
) -> dict[tuple[Any, ...], list[str]]:
    predictions_by_user: dict[tuple[Any, ...], list[str]] = {}
    if recommendations.empty:
        return predictions_by_user

    duplicated = recommendations.duplicated(list(EVALUATION_KEY_COLUMNS), keep=False)
    if duplicated.any():
        sample = (
            recommendations.loc[duplicated, list(EVALUATION_KEY_COLUMNS)]
            .head(3)
            .to_dict("records")
        )
        raise ValueError(f"recommendations contains duplicate user keys: {sample}")

    for row in recommendations.itertuples(index=False):
        predictions_by_user[_user_key(row)] = parse_prediction_items(
            str(row.prediction),
            top_k,
        )
    return predictions_by_user


def _build_coverage_by_window(
    recommendations: pd.DataFrame,
    recommendable_pool: pd.DataFrame,
    top_k: int,
) -> dict[tuple[Any, ...], float]:
    recommended_by_window: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in recommendations.itertuples(index=False):
        window_key = _window_key(row)
        recommended_by_window[window_key].update(
            parse_prediction_items(str(row.prediction), top_k)
        )

    pool_by_window: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in recommendable_pool.itertuples(index=False):
        pool_by_window[_window_key(row)].add(str(row.article_id))

    coverage_by_window: dict[tuple[Any, ...], float] = {}
    for window_key, pool_items in pool_by_window.items():
        coverage_by_window[window_key] = (
            len(recommended_by_window.get(window_key, set()) & pool_items)
            / len(pool_items)
            if pool_items
            else 0.0
        )
    return coverage_by_window


def _sorted_target_windows(
    target_users: pd.DataFrame,
    split: str,
) -> list[tuple[int, int]]:
    split_rows = target_users.loc[target_users["split"].astype(str) == split]
    windows = split_rows.loc[:, ["cutoff_week", "label_week"]].drop_duplicates()
    return [
        (int(row.cutoff_week), int(row.label_week))
        for row in windows.sort_values(["cutoff_week", "label_week"]).itertuples(
            index=False
        )
    ]


def _user_key(row: object) -> tuple[Any, ...]:
    return (
        str(getattr(row, "split")),
        int(getattr(row, "cutoff_week")),
        int(getattr(row, "label_week")),
        str(getattr(row, "customer_id")),
    )


def _window_key(row: object) -> tuple[Any, ...]:
    return (
        str(getattr(row, "split")),
        int(getattr(row, "cutoff_week")),
        int(getattr(row, "label_week")),
    )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0
