from __future__ import annotations

RECOMMENDATION_TOP_K = 12
RECOMMENDATION_CANDIDATES_PER_SOURCE = RECOMMENDATION_TOP_K
RECOMMENDATION_BACKFILL_CANDIDATES_PER_WINDOW = 50
RECOMMENDATION_PROFILE_TOP_ATTRIBUTES = 3
RECOMMENDATION_ARTICLE_ID_DTYPE = "string"
CUSTOMER_AGE_BUCKETS = (
    "unknown",
    "0-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60+",
)

VALID_RECOMMENDATION_SPLITS = ("valid", "test")
RECOMMENDATION_METHODS = (
    "global_popularity",
    "recent_popularity",
    "attribute_similarity",
    "pop_similarity",
    "pop_similarity_trend",
)
RECOMMENDATION_CANDIDATE_STRATEGIES = (
    "popularity",
    "similarity",
    "trend_union",
    "default",
    "enhanced_default",
)
SOURCE_ORDER = {
    "popularity": 0,
    "similarity": 1,
    "trend": 2,
    "reorder": 3,
    "product_variant": 4,
    "age_popularity": 5,
    "preference_popularity": 6,
}
RECOMMENDATION_SCORE_COLUMNS = (
    "pop_score",
    "sim_score",
    "trend_score",
    "recent_score",
)
RECOMMENDATION_CORE_ATTR_TYPES = (
    "product_type_name",
    "colour_group_name",
    "garment_group_name",
    "product_group_name",
    "graphical_appearance_name",
)
RECOMMENDATION_TREND_ATTR_WEIGHTS = {
    "product_type_name": 0.35,
    "colour_group_name": 0.25,
    "garment_group_name": 0.20,
    "product_group_name": 0.10,
    "graphical_appearance_name": 0.10,
}

TIME_WINDOW_COLUMNS = ("split", "cutoff_week", "label_week")
TARGET_USER_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "history_purchase_count",
    "label_purchase_count",
)
EVALUATION_LABEL_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "article_id",
)
USER_PROFILE_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "attr_id",
    "attr_type",
    "attr_value",
    "preference_score",
    "purchase_count",
    "last_purchase_week",
)
CUSTOMER_PROFILE_COLUMNS = (
    "customer_id",
    "age",
    "age_bucket",
    "club_member_status",
    "fashion_news_frequency",
)
ARTICLE_PRODUCT_MAP_COLUMNS = ("article_id", "product_code")
CANDIDATE_ITEM_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "strategy",
    "customer_id",
    "article_id",
    "candidate_sources",
    "primary_source",
    "best_source_rank",
)
ENHANCED_CANDIDATE_ITEM_COLUMNS = (
    *CANDIDATE_ITEM_COLUMNS,
    "has_reorder_source",
    "allow_seen",
)
ENHANCED_CANDIDATE_SOURCE_CAPS = {
    "popularity": {"top_n": RECOMMENDATION_CANDIDATES_PER_SOURCE},
    "similarity": {"top_n": RECOMMENDATION_CANDIDATES_PER_SOURCE},
    "trend": {"top_n": RECOMMENDATION_CANDIDATES_PER_SOURCE},
    "reorder": {"top_n": RECOMMENDATION_CANDIDATES_PER_SOURCE},
    "product_variant": {
        "seed_top_n": 6,
        "per_seed_top_n": 3,
        "top_n": RECOMMENDATION_CANDIDATES_PER_SOURCE,
    },
    "age_popularity": {
        "pool_top_n": 50,
        "per_user_top_n": RECOMMENDATION_CANDIDATES_PER_SOURCE,
        "recent_weeks": 4,
    },
    "preference_popularity": {
        "top_attributes": RECOMMENDATION_PROFILE_TOP_ATTRIBUTES,
        "per_attribute_top_n": 4,
        "per_user_top_n": RECOMMENDATION_CANDIDATES_PER_SOURCE,
        "recent_weeks": 4,
    },
}
RECOMMENDATIONS_COLUMNS = (
    "customer_id",
    "split",
    "cutoff_week",
    "label_week",
    "method",
    "prediction",
)
RECOMMENDATION_ITEMS_COLUMNS = (
    "customer_id",
    "split",
    "cutoff_week",
    "label_week",
    "method",
    "article_id",
    "rank",
    "score",
    "pop_score",
    "sim_score",
    "trend_score",
    "recent_score",
    "candidate_sources",
)

TIME_WINDOW_KEY_COLUMNS = TIME_WINDOW_COLUMNS
TARGET_USER_KEY_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
)
EVALUATION_LABEL_KEY_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "article_id",
)
CANDIDATE_ITEM_KEY_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "strategy",
    "customer_id",
    "article_id",
)
RECOMMENDATIONS_KEY_COLUMNS = (
    "customer_id",
    "split",
    "cutoff_week",
    "label_week",
    "method",
)
RECOMMENDATION_ITEMS_KEY_COLUMNS = (
    "customer_id",
    "split",
    "cutoff_week",
    "label_week",
    "method",
    "article_id",
)
USER_PROFILE_KEY_COLUMNS = (
    "split",
    "cutoff_week",
    "label_week",
    "customer_id",
    "attr_id",
    "attr_type",
    "attr_value",
)
CUSTOMER_PROFILE_KEY_COLUMNS = ("customer_id",)
ARTICLE_PRODUCT_MAP_KEY_COLUMNS = ("article_id",)

RECOMMENDATION_TEXT_COLUMNS = (
    "split",
    "customer_id",
    "article_id",
    "product_code",
    "prediction",
    "strategy",
    "method",
    "candidate_sources",
    "primary_source",
    "attr_type",
    "attr_value",
    "age_bucket",
    "club_member_status",
    "fashion_news_frequency",
)
