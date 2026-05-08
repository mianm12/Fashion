from fashion_trend.foundation.paths import OUTPUT_DIR, PROCESSED_DIR

# 趋势确定性流水线的中间表输出根目录。
TREND_DIR = PROCESSED_DIR / "trend"
# 趋势样本和时间切分特征的输出根目录。
FEATURES_DIR = PROCESSED_DIR / "features"
# 趋势模型训练产物根目录，按 `outputs/models/<model>/` 组织。
OUTPUT_MODELS_DIR = OUTPUT_DIR / "models"
# 趋势评价产物根目录，按 `outputs/metrics/<model>/` 组织。
OUTPUT_METRICS_DIR = OUTPUT_DIR / "metrics"

# Phase 4 商品周销量阶段输出。
TREND_ARTICLE_WEEK_SALES_PATH = TREND_DIR / "article_week_sales.csv"
# Phase 4 属性周热度阶段输出。
TREND_ATTRIBUTE_WEEK_HEAT_PATH = TREND_DIR / "attribute_week_heat.csv"
# Phase 5 趋势标签阶段输出。
TREND_ATTRIBUTE_WEEK_TARGET_PATH = TREND_DIR / "attribute_week_target.csv"
# Phase 5 趋势训练样本阶段输出。
TREND_MODEL_SAMPLES_PATH = FEATURES_DIR / "trend_model_samples.parquet"
# 时间切分阶段的训练集输出。
TREND_MODEL_SAMPLES_TRAIN_PATH = FEATURES_DIR / "trend_model_samples_train.parquet"
# 时间切分阶段的验证集输出。
TREND_MODEL_SAMPLES_VALID_PATH = FEATURES_DIR / "trend_model_samples_valid.parquet"
# 时间切分阶段的测试集输出。
TREND_MODEL_SAMPLES_TEST_PATH = FEATURES_DIR / "trend_model_samples_test.parquet"
# 时间切分阶段的元数据 JSON 输出。
TREND_MODEL_SAMPLES_SPLIT_METADATA_PATH = (
    FEATURES_DIR / "trend_model_samples_split_metadata.json"
)
# 验证集按最后测试窗口之前的连续周数留出。
TREND_SPLIT_VALID_WEEKS = 8
# 测试集按样本末尾连续周数留出。
TREND_SPLIT_TEST_WEEKS = 8
