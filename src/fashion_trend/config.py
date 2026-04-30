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
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"

MODEL_DIR = DATA_DIR / "models"
