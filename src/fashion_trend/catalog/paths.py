from fashion_trend.foundation.paths import INTERIM_DIR, PROCESSED_DIR

GRAPH_DIR = PROCESSED_DIR / "graph"
ARTICLES_CLEAN_MVP_PATH = INTERIM_DIR / "articles_clean_mvp.csv"
ARTICLES_CLEAN_PATH = INTERIM_DIR / "articles_clean.csv"
GRAPH_NODES_ARTICLE_PATH = GRAPH_DIR / "nodes_article.csv"
GRAPH_NODES_ATTRIBUTE_PATH = GRAPH_DIR / "nodes_attribute.csv"
GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH = GRAPH_DIR / "edges_article_attribute.csv"
GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH = GRAPH_DIR / "edges_attribute_hierarchy.csv"
