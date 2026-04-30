# Fashion

时尚推荐相关实验项目，当前主要提供 Kaggle H&M 个性化时尚推荐数据集的下载入口和基础目录约定。

## 数据集

项目使用 Kaggle 比赛数据集：`h-and-m-personalized-fashion-recommendations`。

该数据集来自 H&M 个性化时尚推荐比赛，主要用于根据用户历史交易、商品信息和用户信息预测用户未来可能购买的商品。下载后通常会包含以下类型的数据文件：

- `transactions_train.csv`：用户历史交易记录。
- `articles.csv`：商品信息，包括商品编号、类别、颜色、描述等字段。
- `customers.csv`：用户信息。
- `sample_submission.csv`：Kaggle 提交格式示例。
- 商品图片文件：部分数据包中会包含按商品编号组织的图片资源。

数据文件体积较大，`data/` 目录默认不会包含到版本库。

## 下载数据集

下载入口位于：

```sh
src/00_download_data.py
```

默认配置集中在 `src/fashion_trend/config.py`：

- `DEFAULT_COMPETITION`：`h-and-m-personalized-fashion-recommendations`
- `RAW_DIR`：项目根目录下的 `data/raw`

默认会下载 `h-and-m-personalized-fashion-recommendations`，并保存到项目根目录下的：

```sh
data/raw/h-and-m-personalized-fashion-recommendations/
```

脚本会自动创建目标目录；如果目标目录已经存在且非空，默认会跳过重新下载。需要重新下载时使用 `--force`。

### 1. 准备环境

项目使用 Python 3.13+，并通过 `uv` 管理依赖。首次运行前先安装依赖：

```sh
uv sync
```

首次下载 Kaggle 数据前，需要确保当前环境已经具备 Kaggle 访问权限，并且已在 Kaggle 页面接受该比赛的数据使用规则。

如果需要配置 Kaggle API 凭据，请只放在本机安全位置或环境变量中，不要将 `kaggle.json`、Token、密码等敏感信息提交到仓库。

### 2. 执行下载

```sh
uv run python src/00_download_data.py
```

脚本会自动创建目标目录，并默认解压下载得到的 zip 文件。

### 3. 常用参数

指定其他 Kaggle competition slug：

```sh
uv run python src/00_download_data.py --competition h-and-m-personalized-fashion-recommendations
```

指定数据保存根目录：

```sh
uv run python src/00_download_data.py --data-dir data/raw
```

仅下载，不自动解压：

```sh
uv run python src/00_download_data.py --no-unzip
```

即使目标目录已有文件，也重新下载：

```sh
uv run python src/00_download_data.py --force
```

查看完整命令帮助：

```sh
uv run python src/00_download_data.py --help
```

### 4. 下载行为说明

- 默认目标目录：项目根目录下的 `data/raw/<competition>/`。
- 默认行为：目标目录不存在或为空时下载；目标目录已有内容时跳过下载。
- `--force`：忽略已有内容，重新调用 Kaggle 下载。
- `--no-unzip`：保留下载得到的 zip 文件，不执行解压。
- 默认解压：脚本会解压目标目录下的 zip 文件，并拒绝解压会逃逸出目标目录的异常路径。

## 数据目录结构

下载完成后，目录大致如下：

```text
data/
+-- raw/
    +-- h-and-m-personalized-fashion-recommendations/
        +-- articles.csv
        +-- customers.csv
        +-- sample_submission.csv
        +-- transactions_train.csv
        +-- ...
```

## 项目结构

```text
src/
+-- 00_download_data.py
+-- fashion_trend/
    +-- __init__.py
    +-- config.py
```

- `src/00_download_data.py`：命令行入口，负责解析参数、下载 Kaggle 数据、跳过重复下载和解压 zip 文件。
- `src/fashion_trend/config.py`：集中维护默认 Kaggle competition slug 和默认数据根目录。
