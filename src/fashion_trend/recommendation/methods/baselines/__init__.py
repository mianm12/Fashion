from __future__ import annotations

from fashion_trend.recommendation.methods.baselines.attribute_similarity import (
    AttributeSimilarityMethod,
)
from fashion_trend.recommendation.methods.baselines.global_popularity import (
    GlobalPopularityMethod,
)
from fashion_trend.recommendation.methods.baselines.pop_similarity import (
    PopSimilarityMethod,
)
from fashion_trend.recommendation.methods.baselines.recent_popularity import (
    RecentPopularityMethod,
)

__all__ = [
    "AttributeSimilarityMethod",
    "GlobalPopularityMethod",
    "PopSimilarityMethod",
    "RecentPopularityMethod",
]
