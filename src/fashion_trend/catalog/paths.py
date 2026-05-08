from fashion_trend.foundation.paths import INTERIM_DIR, PROCESSED_DIR

# 属性图 CSV 产物所在目录。
GRAPH_DIR = PROCESSED_DIR / "graph"
# 只含核心属性的 MVP 商品清洗 CSV 产物。
ARTICLES_CLEAN_MVP_PATH = INTERIM_DIR / "articles_clean_mvp.csv"
# 含完整属性层级的商品清洗 CSV 产物。
ARTICLES_CLEAN_PATH = INTERIM_DIR / "articles_clean.csv"
# 属性图商品节点 CSV 产物。
GRAPH_NODES_ARTICLE_PATH = GRAPH_DIR / "nodes_article.csv"
# 属性图属性节点 CSV 产物。
GRAPH_NODES_ATTRIBUTE_PATH = GRAPH_DIR / "nodes_attribute.csv"
# 属性图商品-属性边 CSV 产物。
GRAPH_EDGES_ARTICLE_ATTRIBUTE_PATH = GRAPH_DIR / "edges_article_attribute.csv"
# 属性图属性层级边 CSV 产物。
GRAPH_EDGES_ATTRIBUTE_HIERARCHY_PATH = GRAPH_DIR / "edges_attribute_hierarchy.csv"
