from __future__ import annotations

from fashion_trend.foundation.paths import DATA_DIR, OUTPUT_DIR

DEFENSE_APP_OUTPUT_DIR = OUTPUT_DIR / "defense_app"
DEFENSE_APP_DB_PATH = DEFENSE_APP_OUTPUT_DIR / "fashion_demo.sqlite"
DEFENSE_APP_STATIC_DIR = DEFENSE_APP_OUTPUT_DIR / "static"

REPORTS_MANIFEST_PATH = OUTPUT_DIR / "reports" / "manifest.json"

REPORTS_TABLES_DIR = OUTPUT_DIR / "reports" / "tables"
REPORTS_CASE_STUDIES_DIR = OUTPUT_DIR / "reports" / "case_studies"
LIGHTGBM_PREDICTIONS_PATH = OUTPUT_DIR / "models" / "lightgbm" / "predictions.csv"
LIGHTGBM_FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIR / "models" / "lightgbm" / "feature_importance.csv"
)
TREND_METRICS_DIR = OUTPUT_DIR / "metrics"
RECOMMENDATION_OUTPUT_DIR = OUTPUT_DIR / "recommendation"
MAIN_RECOMMENDATION_ITEMS_PATH = (
    RECOMMENDATION_OUTPUT_DIR
    / "pop_similarity_trend"
    / "recommendation_items.parquet"
)
RECOMMENDATION_EXPERIMENT_PATH = (
    RECOMMENDATION_OUTPUT_DIR / "experiments" / "main" / "experiment.json"
)

RECOMMENDATION_DATA_DIR = DATA_DIR / "processed" / "recommend"
TIME_WINDOWS_PATH = RECOMMENDATION_DATA_DIR / "time_windows.parquet"
TARGET_USERS_PATH = RECOMMENDATION_DATA_DIR / "target_users.parquet"
EVALUATION_LABELS_PATH = RECOMMENDATION_DATA_DIR / "evaluation_labels.parquet"
USER_PROFILE_PATH = RECOMMENDATION_DATA_DIR / "user_profile.parquet"

TREND_DATA_DIR = DATA_DIR / "processed" / "trend"
ATTRIBUTE_WEEK_HEAT_PATH = TREND_DATA_DIR / "attribute_week_heat.csv"
TREND_MODEL_SAMPLES_PATH = (
    DATA_DIR / "processed" / "features" / "trend_model_samples.parquet"
)

GRAPH_DATA_DIR = DATA_DIR / "processed" / "graph"
GRAPH_NODES_ARTICLE_PATH = GRAPH_DATA_DIR / "nodes_article.csv"
GRAPH_NODES_ATTRIBUTE_PATH = GRAPH_DATA_DIR / "nodes_attribute.csv"
GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH = GRAPH_DATA_DIR / "edges_article_attribute.csv"
GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH = GRAPH_DATA_DIR / "edges_attribute_hierarchy.csv"

ARTICLES_CLEAN_PATH = DATA_DIR / "interim" / "articles_clean.csv"
