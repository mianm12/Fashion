from pathlib import Path

# 项目仓库根路径，仅提供无业务语义的定位基准。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 项目数据根目录，仅按存储层级组织数据文件。
DATA_DIR = PROJECT_ROOT / "data"
# 原始数据根目录，不编码具体业务产物语义。
RAW_DIR = DATA_DIR / "raw"
# 中间数据根目录，不编码具体业务产物语义。
INTERIM_DIR = DATA_DIR / "interim"
# 处理后数据根目录，不编码具体业务产物语义。
PROCESSED_DIR = DATA_DIR / "processed"
# 模型、指标等输出根目录，不编码具体业务产物语义。
OUTPUT_DIR = PROJECT_ROOT / "outputs"
