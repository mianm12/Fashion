from fashion_trend.foundation.paths import RAW_DIR

# H&M Personalized Fashion Kaggle 比赛的稳定 slug。
DEFAULT_COMPETITION = "h-and-m-personalized-fashion-recommendations"
# H&M 原始数据集解压后的目录。
RAW_HM_DIR = RAW_DIR / DEFAULT_COMPETITION
# H&M 原始交易明细 CSV 文件。
RAW_TRANSACTIONS_PATH = RAW_HM_DIR / "transactions_train.csv"
# H&M 原始商品目录 CSV 文件。
RAW_ARTICLES_PATH = RAW_HM_DIR / "articles.csv"
# H&M 原始客户 CSV 文件。
RAW_CUSTOMERS_PATH = RAW_HM_DIR / "customers.csv"
