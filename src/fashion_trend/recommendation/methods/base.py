from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class RecommendationContext:
    method: str
    top_k: int
    exclude_seen: bool
    transactions: pd.DataFrame
    article_attributes: pd.DataFrame
    windows: pd.DataFrame
    target_users: pd.DataFrame
    candidates: pd.DataFrame | None = None
    user_profile: pd.DataFrame | None = None
    trend_predictions: pd.DataFrame | None = None
    weights: dict[str, float] | None = None
    input_paths: dict[str, str] | None = None
    trend_model_source: str | None = None


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: pd.DataFrame
    recommendation_items: pd.DataFrame
    params: dict[str, object]
    metadata: dict[str, object]


class RecommendationMethod(Protocol):
    name: str
    method_type: str
    default_candidate_strategy: str | None
    default_weights: dict[str, float]
    required_features: Sequence[str]

    def build_recommendations(
        self,
        context: RecommendationContext,
    ) -> RecommendationResult:
        raise NotImplementedError
