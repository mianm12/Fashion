# 仓库指南

## 项目范围与当前状态

这是一个 Python 3.10-3.12 项目，用于 H&M 时尚趋势预测和轻量推荐实验。当前已实现的流水线覆盖周级数据准备、商品清洗、属性图构建、商品周销量、属性周热度、趋势标签、趋势样本、基于时间的样本切分、三类趋势 baseline、LightGBM 主模型和趋势评价。

推荐生成和推荐评价尚未实现。除非对应实现和产物已经存在，否则不要把它们描述为可运行功能。

大体量数据集和生成产物位于 `data/` 和 `outputs/`。这些内容应视为运行时产物，不应作为源代码文件处理。

运行时依赖以 `pyproject.toml` 为准，当前包含 `kagglehub`、`lightgbm`、`numpy`、`pandas`、`pyarrow` 和 `scikit-learn`。LightGBM 是原生依赖，导入失败时应优先检查本机 native runtime，例如 macOS 上的 `libomp`，不要把缺少 native runtime 误判为 baseline 或 registry 问题。

## 项目结构与模块组织

`src/00_*.py` 到 `src/11_*.py` 的编号脚本是工作流入口。它们应保持为清晰的编排层：解析参数、按顺序调用包函数、记录日志并返回稳定退出码。核心计算、校验、读取、写入和 artifact 处理应放在 `src/fashion_trend/` 下。

包按领域组织：

- `foundation/`：项目根路径、日志、DataFrame 工具、安全 IO 和 artifact helper。
- `datasets/`：Kaggle 下载和原始数据集 profile。
- `transactions/`：交易路径、契约、reader 和周级聚合。
- `catalog/`：商品清洗、catalog 契约和 reader，以及 `catalog/graph/` 下的 builder 与 publisher。
- `trend/`：趋势 schema、路径、reader、预测契约、热度、标签、特征、切分、训练、评价和模型实现。
- `recommendation/`：为后续下游工作预留的推荐契约、路径和 reader。
- `reports/`：只读报表路径和边界。

在 `trend/` 内部保持职责清晰：

- `heat/`：商品周销量和属性周热度。
- `labels/`：趋势标签生成。
- `features/`：趋势模型样本生成。
- `splits/`：基于时间的 train/valid/test 切分逻辑。
- `models/baselines/`：确定性 baseline，例如 `last_week`、`previous_growth` 和 `moving_average`。
- `models/supervised/`：监督模型，例如 `lightgbm`。
- `models/registry.py`：模型注册和查找。
- `training/`：训练 runner、输出路径和 run artifact 契约。
- `evaluation/`：指标、payload、runner 和指标 artifact 契约。

不要重新引入历史根模块，例如 `fashion_trend.training`、`fashion_trend.evaluation`、`fashion_trend.models`、`fashion_trend.articles` 或 `fashion_trend.data_loader`。当存在具体子模块导入路径时，不要通过 `fashion_trend.trend` facade 导入；架构测试会强制检查直接导入。

## 构建、测试与开发命令

- `uv sync`：根据 `pyproject.toml` 和 `uv.lock` 安装依赖。
- `uv run pytest`：使用 `src` 作为 `PYTHONPATH` 运行完整 pytest 测试套件。
- `uv run pytest tests/test_trend_training.py tests/test_trend_lightgbm.py tests/test_trend_evaluation.py`：针对趋势训练、LightGBM 和评价改动的聚焦验证。
- `uv run pytest tests/test_architecture_boundaries.py`：修改包结构、导入路径或 facade 边界时的聚焦验证。
- `uv run black --check src tests`：检查 Python 格式。
- `uv run isort --check-only src tests`：使用 Black profile 检查 import 排序。
- `uv run python -m compileall -q src`：在导入或 CLI 边界发生变化时，对包和编号脚本做编译检查。

验证 artifact 时，按编号顺序运行已实现流水线：

```sh
uv run python src/00_download_data.py
uv run python src/01_data_check.py
uv run python src/02_build_weekly_transactions.py
uv run python src/03_clean_articles.py
uv run python src/04_build_attribute_graph.py
uv run python src/05_compute_article_week_sales.py
uv run python src/06_compute_attribute_week_heat.py
uv run python src/07_build_trend_targets.py
uv run python src/08_build_trend_model_samples.py
uv run python src/09_split_trend_model_samples.py
```

通过共享入口训练和评价已注册的趋势模型：

```sh
uv run python src/10_train_trend_model.py --model last_week
uv run python src/10_train_trend_model.py --model previous_growth
uv run python src/10_train_trend_model.py --model moving_average
uv run python src/10_train_trend_model.py --model lightgbm
uv run python src/11_eval_trend_model.py --model last_week
uv run python src/11_eval_trend_model.py --model previous_growth
uv run python src/11_eval_trend_model.py --model moving_average
uv run python src/11_eval_trend_model.py --model lightgbm
```

默认运行 `uv run python src/10_train_trend_model.py --model lightgbm` 时，训练参数优先读取 `outputs/models/lightgbm/params.json` 中已经发布的 stable 参数；如果 stable 参数文件不存在，才回退到源码内置默认参数。这条默认训练路径会生成自动 `run_id`，并默认发布到 stable，因此会更新 `outputs/models/lightgbm/`。默认运行 `uv run python src/11_eval_trend_model.py --model lightgbm` 时评价当前 stable 预测，并写入 `outputs/metrics/lightgbm/trend_metrics.json`。

LightGBM 调参 run 是 run-scoped，不应意外替换 stable 主结果：

```sh
uv run python src/10_train_trend_model.py --model lightgbm --run-id smoke-lightgbm --no-promote
uv run python src/11_eval_trend_model.py --model lightgbm --run-id smoke-lightgbm
uv run python src/10_train_trend_model.py --model lightgbm --promote-run smoke-lightgbm
```

`--run-id`、`--params`、`--param`、`--promote`、`--no-promote` 和 `--promote-run` 只适用于 LightGBM。Baseline 遇到这些选项时必须拒绝，而不是静默忽略。带显式 `run_id`、参数文件或 CLI 参数覆盖的实验默认不 promotion；需要保留实验时显式使用 `--no-promote`。`--promote` 只在训练后发布模型 artifact，不会自动运行评价；要让 stable 模型 artifact 与 stable metrics 对齐到同一个已评估 run，应先评价 run，再使用 `--promote-run <run_id>`。

## Artifact 契约

关键数据 artifact：

- `data/interim/transactions_train_weekly.parquet`
- `data/interim/articles_clean_mvp.csv`
- `data/interim/articles_clean.csv`
- `data/processed/graph/nodes_article.csv`
- `data/processed/graph/nodes_attribute.csv`
- `data/processed/graph/edges_article_attribute.csv`
- `data/processed/graph/edges_attribute_hierarchy.csv`
- `data/processed/trend/article_week_sales.csv`
- `data/processed/trend/attribute_week_heat.csv`
- `data/processed/trend/attribute_week_target.csv`
- `data/processed/features/trend_model_samples.parquet`
- `data/processed/features/trend_model_samples_train.parquet`
- `data/processed/features/trend_model_samples_valid.parquet`
- `data/processed/features/trend_model_samples_test.parquet`
- `data/processed/features/trend_model_samples_split_metadata.json`

Baseline 和 stable 模型输出使用：

- `outputs/models/<model>/predictions.csv`
- `outputs/models/<model>/params.json`
- `outputs/models/<model>/metadata.json`
- `outputs/metrics/<model>/trend_metrics.json`

LightGBM stable 输出额外包含 `feature_importance.csv` 和 `model.txt`。LightGBM run 输出使用：

- `outputs/models/lightgbm/runs/<run_id>/predictions.csv`
- `outputs/models/lightgbm/runs/<run_id>/params.json`
- `outputs/models/lightgbm/runs/<run_id>/metadata.json`
- `outputs/models/lightgbm/runs/<run_id>/feature_importance.csv`
- `outputs/models/lightgbm/runs/<run_id>/model.txt`
- `outputs/models/lightgbm/runs/index.jsonl`
- `outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json`
- `outputs/metrics/lightgbm/runs/evaluations.jsonl`

LightGBM stable 目录代表当前主结果。Run 目录代表保留的实验。Stable `metadata.json` 和 `trend_metrics.json` 应记录发布来源 `run_id`；`--promote-run` 发布后，stable 核心模型 artifact 与 run artifact 应一致，但 stable metadata 的 `output_dir`、`prediction_path` 和 `params_path` 会改写为 stable 路径，这是预期行为。

## 编码风格与架构边界

使用 Black 格式、`profile = "black"` 的 isort，以及清晰的 snake_case 模块名、函数名、变量名和测试 helper 名。

优先复用项目已有 helper，不要新增临时工具：

- 使用 `foundation.paths` 和各领域 `paths.py` 模块维护路径常量。
- 使用 `foundation.logging` 记录 CLI 日志。
- 使用 `foundation.io` 和 artifact helper 进行安全写入以及 JSON/CSV/parquet IO。
- 将校验逻辑放在拥有该契约的领域包附近。

架构边界会被测试强制检查：

- `foundation` 不能导入业务领域。
- `datasets`、`transactions` 和 `catalog` 只能依赖允许的下层工具。
- `trend` 可以依赖稳定输入领域，但不能依赖 `datasets`、`recommendation` 或 `reports`。
- `recommendation` 和 `reports` 只能读取上游公开契约和 reader，不能依赖核心计算模块。

新增功能时，将其放在拥有对应业务事实的领域中。不要让编号脚本成为可复用逻辑的来源。

## 测试指南

项目使用 pytest。测试文件命名为 `tests/test_*.py`，新增测试应对齐真实流水线阶段：foundation artifact、商品清洗、属性图、商品销量、属性热度、标签、样本、切分、训练、LightGBM、评价或架构边界。

除非明确说明是 artifact 验证，否则测试不应依赖真实 H&M 数据集。优先使用小型内存 fixture，以及 `tests/trend_samples.py` 或 `tests/__init__.py` 中的共享 helper。

修复 bug 时，添加或更新一个修复前会失败的回归测试。修改模型或 artifact 契约时，同时验证 happy path 和边界失败，例如非法模型名、非法 `run_id`、缺失 split 列、不安全路径，以及 prediction/metrics payload 不匹配。

## 文档指南

当命令语法、artifact 路径、模型语义或架构边界变化时，保持 `README.md`、`docs/gpt-research/implementation-plan.md`，以及相关 `docs/superpowers/specs/` 或 `docs/superpowers/plans/` 与 as-built 行为一致。

不要留下暗示已实现模块仍位于已删除根文件中的历史设计文本。当前趋势训练和评价代码位于 `src/fashion_trend/trend/` 下。

## Commit 与 Pull Request 指南

近期历史使用 Conventional Commit 前缀和简洁中文摘要，例如 `fix(trend): 修正预测校验和 LightGBM 日志`、`docs: 说明 LightGBM run 调参流程`。

按功能、阶段或契约边界保持 commit 粒度。提交前检查 `git diff`，确认改动范围，并运行相关验证命令。不要提交生成的数据集、模型输出、凭据、缓存或无关格式化改动。

Pull request 应说明变更的阶段或契约，列出验证命令，在相关时说明 artifact 路径，并指出数据或配置假设。

## 安全与配置提示

Kaggle 凭据、API token、`.env` 文件、原始数据集、生成输出、模型 artifact 和本机会话文件都不能进入提交。使用环境变量或仓库外的凭据文件访问数据。

不要在日志、文档、测试、快照、metadata 或提交信息中写入敏感信息。
