from pathlib import Path

# ============================================================
# 0. Competition and dataset configuration
# ============================================================

DEFAULT_COMPETITION = "h-and-m-personalized-fashion-recommendations"


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_HM_DIR = RAW_DIR / "h-and-m-personalized-fashion-recommendations"

INTERIM_DIR = DATA_DIR / "interim"

PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"

MODEL_DIR = DATA_DIR / "models"


# ============================================================
# 2. File paths
# ============================================================

PATH = {
    # ---------------- Raw H&M data ----------------
    "raw_transactions": RAW_HM_DIR / "transactions_train.csv",
    "raw_articles": RAW_HM_DIR / "articles.csv",
    "raw_customers": RAW_HM_DIR / "customers.csv",
    # ---------------- Interim data ----------------
    "interim_transactions_weekly": INTERIM_DIR / "transactions_train_weekly.parquet",
    "interim_articles_clean_mvp": INTERIM_DIR / "articles_clean_mvp.csv",
    "interim_articles_clean": INTERIM_DIR / "articles_clean.csv",
}
