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
TREND_DIR = PROCESSED_DIR / "trend"

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
    # ---------------- Processed graph data ----------------
    "graph_nodes_article": GRAPH_DIR / "nodes_article.csv",
    "graph_nodes_attribute": GRAPH_DIR / "nodes_attribute.csv",
    "graph_edges_article_attribute": GRAPH_DIR / "edges_article_attribute.csv",
    "graph_edges_attribute_hierarchy": GRAPH_DIR / "edges_attribute_hierarchy.csv",
    # ---------------- Processed trend data ----------------
    "trend_article_week_sales": TREND_DIR / "article_week_sales.csv",
    "trend_attribute_week_heat": TREND_DIR / "attribute_week_heat.csv",
}
