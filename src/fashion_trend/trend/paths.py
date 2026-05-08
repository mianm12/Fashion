from fashion_trend.foundation.paths import OUTPUT_DIR, PROCESSED_DIR

TREND_DIR = PROCESSED_DIR / "trend"
FEATURES_DIR = PROCESSED_DIR / "features"
OUTPUT_MODELS_DIR = OUTPUT_DIR / "models"
OUTPUT_METRICS_DIR = OUTPUT_DIR / "metrics"

TREND_ARTICLE_WEEK_SALES_PATH = TREND_DIR / "article_week_sales.csv"
TREND_ATTRIBUTE_WEEK_HEAT_PATH = TREND_DIR / "attribute_week_heat.csv"
TREND_ATTRIBUTE_WEEK_TARGET_PATH = TREND_DIR / "attribute_week_target.csv"
TREND_MODEL_SAMPLES_PATH = FEATURES_DIR / "trend_model_samples.parquet"
TREND_MODEL_SAMPLES_TRAIN_PATH = FEATURES_DIR / "trend_model_samples_train.parquet"
TREND_MODEL_SAMPLES_VALID_PATH = FEATURES_DIR / "trend_model_samples_valid.parquet"
TREND_MODEL_SAMPLES_TEST_PATH = FEATURES_DIR / "trend_model_samples_test.parquet"
TREND_MODEL_SAMPLES_SPLIT_METADATA_PATH = (
    FEATURES_DIR / "trend_model_samples_split_metadata.json"
)
TREND_SPLIT_VALID_WEEKS = 8
TREND_SPLIT_TEST_WEEKS = 8
