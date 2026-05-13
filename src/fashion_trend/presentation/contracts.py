from __future__ import annotations

PRESENTATION_SCHEMA_VERSION = "defense_app_v1"

DEFAULT_TOP_K = 10
MAX_TREND_LIMIT = 50
MAX_ARTICLE_LIMIT = 100
MIN_DEMO_CASE_COUNT = 20
MAX_DEMO_USER_LIMIT = 50
DEFAULT_DEMO_CASE_LIMIT = MAX_DEMO_USER_LIMIT

CORE_TREND_ATTR_TYPES = (
    "colour_group_name",
    "product_type_name",
    "graphical_appearance_name",
    "garment_group_name",
)

MAIN_RECOMMENDATION_METHOD = "pop_similarity_trend"
