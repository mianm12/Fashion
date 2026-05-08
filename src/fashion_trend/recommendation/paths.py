from fashion_trend.foundation.paths import OUTPUT_DIR, PROCESSED_DIR

# 尚未扩展的推荐阶段中间产物目录与特征目录。
RECOMMEND_DIR = PROCESSED_DIR / "recommend"
RECOMMEND_FEATURES_DIR = RECOMMEND_DIR / "features"

# 尚未扩展的推荐阶段最终输出目录。
OUTPUT_RECOMMENDATION_DIR = OUTPUT_DIR / "recommendation"

# 尚未扩展的推荐阶段预留中间产物位置。
USER_PROFILE_PATH = RECOMMEND_DIR / "user_profile.parquet"
RECOMMEND_CANDIDATES_PATH = RECOMMEND_DIR / "candidate_items.parquet"

# 尚未扩展的推荐阶段预留结果与评价输出位置。
RECOMMENDATION_RESULT_PATH = OUTPUT_RECOMMENDATION_DIR / "recommendation_result.csv"
RECOMMENDATION_METRICS_PATH = OUTPUT_RECOMMENDATION_DIR / "recommendation_metrics.json"
