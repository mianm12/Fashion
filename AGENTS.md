# 仓库指南

## 项目范围与当前状态

这是一个 Python 3.10-3.12 项目，用于 H&M 时尚趋势预测和轻量推荐实验。当前已实现的流水线覆盖周级数据准备、商品清洗、属性图构建、商品周销量、属性周热度、趋势标签、趋势样本、基于时间的样本切分、三类趋势 baseline、LightGBM 主模型、趋势评价、轻量离线 Top-N 推荐生成、推荐评价、论文图表/表格/案例导出、只读 SQLite 展示库构建，以及本地答辩展示应用。

推荐模块已实现为离线实验层，用于把趋势预测结果映射回商品推荐并验证趋势分贡献。第一阶段推荐增强已作为独立可选实验实现，但默认 main、stable、reports 和 defense app 边界不变。不要把它描述成在线服务、深度推荐模型、向量召回系统或完整生产推荐平台。

报告导出模块已实现为只读论文素材层，用于从稳定 artifact 导出静态图表、Markdown/CSV 表格、推荐解释案例和 manifest。不要把它描述成在线 dashboard、交互式展示系统或会重新训练模型的流程。

presentation 模块已实现为本地答辩展示库构建层，用于从稳定 artifact 抽取趋势、属性图、推荐、reports 图表和案例数据，写入 `outputs/defense_app/fashion_demo.sqlite` 并发布展示所需静态图表副本。不要把它描述成训练、推荐重跑、候选构建或在线查询系统。

`apps/defense_app/` 是本地只读展示应用：后端是 FastAPI 查询层，前端是 Vue/Vite 桌面展示界面。它依赖已经构建好的 SQLite 展示库，不直接读取原始 H&M 数据、Parquet/CSV 上游 artifact、训练 run 目录或历史推荐 CSV。

大体量数据集和生成产物位于 `data/` 和 `outputs/`。这些内容应视为运行时产物，不应作为源代码文件处理。

运行时依赖以 `pyproject.toml` 为准。核心依赖当前包含 `kagglehub`、`lightgbm`、`matplotlib`、`numpy`、`pandas`、`pyarrow` 和 `scikit-learn`；`app` 依赖组包含 FastAPI 后端所需的 `fastapi` 和 `uvicorn`；`dev` 依赖组包含 `pytest`、`httpx`、`black` 和 `isort`。前端依赖位于 `apps/defense_app/frontend/package.json`，不要在仓库根目录新增 Node 项目。LightGBM 是原生依赖，导入失败时应优先检查本机 native runtime，例如 macOS 上的 `libomp`，不要把缺少 native runtime 误判为 baseline 或 registry 问题。

## 项目结构与模块组织

`src/00_*.py` 到 `src/18_*.py` 的编号脚本是工作流入口。它们应保持为清晰的编排层：解析参数、按顺序调用包函数、记录日志并返回稳定退出码。核心计算、校验、读取、写入和 artifact 处理应放在 `src/fashion_trend/` 下。

包按领域组织：

- `foundation/`：项目根路径、日志、DataFrame 工具、安全 IO 和 artifact helper。
- `datasets/`：Kaggle 下载和原始数据集 profile。
- `transactions/`：交易路径、契约、reader 和周级聚合。
- `catalog/`：商品清洗、catalog 契约和 reader，以及 `catalog/graph/` 下的 builder 与 publisher。
- `trend/`：趋势 schema、路径、reader、预测契约、热度、标签、特征、切分、训练、评价和模型实现。
- `recommendation/`：推荐契约、路径、reader、输入构建、候选召回、重排序方法、评价、实验编排和输出契约。
- `reports/`：只读论文素材导出，包括报表路径、输入加载、图表、表格、案例、manifest 和导出 runner。
- `presentation/`：只读答辩展示库构建，包括展示 schema、source artifact 记录、上游 artifact 抽取、表构建、SQLite 写入、静态 reports 资产发布和构建 runner。

本地展示应用位于 `apps/defense_app/`：

- `backend/`：FastAPI 只读 API、Pydantic schema、SQLite repository 和轻量服务层。
- `frontend/`：Vue 3、TypeScript 和 Vite 桌面展示应用。
- `README.md`：展示库构建、后端运行、前端运行和视觉 QA 说明。

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

在 `recommendation/` 内部保持职责清晰：

- `contracts.py`：推荐列契约、Top-K、method、strategy、split、核心属性类型、权重和 metrics payload 字段。
- `paths.py`：推荐输入、候选、method 输出和 experiment 输出路径。
- `readers.py`：推荐 artifact 的严格读取和校验。
- `time_windows.py`：推荐 `cutoff_week`、`label_week` 和 split 窗口逻辑。
- `retrieval/`：候选召回能力，例如 popularity、attribute similarity 和 trend union。
- `ranking/`：用户画像、特征、过滤、打分和权重组合。
- `methods/`：可注册推荐方法，包括 `global_popularity`、`recent_popularity`、`attribute_similarity`、`pop_similarity`、`pop_similarity_trend` 和可选增强方法 `enhanced_pop_similarity_trend`。
- `evaluation/`：Top-N 离线指标、payload 和评价 runner。
- `experiments/`：权重搜索、消融和主实验编排。

在 `reports/` 内部保持职责清晰：

- `paths.py`：稳定输入 artifact 和 `outputs/reports/` 输出路径。
- `loaders.py`：只读加载趋势、推荐、图和数据规模 artifact，并整理报告表输入。
- `figures.py`、`plotting.py`：使用 `matplotlib` 生成静态 SVG/PNG 图表，并在缺少 CJK 字体时 fail-fast。
- `tables.py`、`markdown.py`：生成 CSV 与 Markdown 表格，不引入额外表格渲染依赖。
- `cases.py`：选择和渲染推荐解释案例。
- `manifest.py`：记录参数、输入、输出、行数、案例用户和 warnings。
- `runner.py`：编排论文素材导出；只能读取稳定 artifact，不能训练模型、重跑推荐方法或构建上游数据。

在 `presentation/` 内部保持职责清晰：

- `paths.py`：展示库输出路径和稳定上游 artifact 路径。
- `contracts.py`：展示库 schema version、展示限制和主推荐 method。
- `source_artifacts.py`：记录 source 路径、mtime、size 和 row count。
- `extractors.py`：只读加载 reports、趋势、推荐、图和商品 artifact。
- `builders.py`：构建 SQLite 展示表，不写文件。
- `sqlite_writer.py`：创建 schema、写表、校验和索引。
- `runner.py`：编排 SQLite 发布和静态 reports 图表复制；失败时回滚旧数据库和旧静态资源。

不要重新引入历史根模块，例如 `fashion_trend.training`、`fashion_trend.evaluation`、`fashion_trend.models`、`fashion_trend.articles` 或 `fashion_trend.data_loader`。当存在具体子模块导入路径时，不要通过 `fashion_trend.trend` facade 导入；架构测试会强制检查直接导入。

## 构建、测试与开发命令

- `uv sync`：根据 `pyproject.toml` 和 `uv.lock` 安装依赖。
- `uv run pytest`：使用 `src` 作为 `PYTHONPATH` 运行完整 pytest 测试套件。
- `uv run pytest tests/test_trend_training.py tests/test_trend_lightgbm.py tests/test_trend_evaluation.py`：针对趋势训练、LightGBM 和评价改动的聚焦验证。
- `uv run pytest tests/test_recommendation_*.py tests/test_architecture_boundaries.py`：针对推荐输入、召回、排序、方法、评价、实验和架构边界改动的聚焦验证。
- `uv run pytest tests/test_reports_*.py tests/test_architecture_boundaries.py`：针对论文素材导出、reports 只读边界和架构边界改动的聚焦验证。
- `uv run pytest tests/test_presentation_*.py tests/test_architecture_boundaries.py`：针对展示库构建、presentation 只读边界和 18 号入口改动的聚焦验证。
- `uv run --group app pytest apps/defense_app/backend/tests`：针对 FastAPI 后端只读 API、schema、repository 和 service 改动的聚焦验证。
- `uv run --group app python -m compileall -q apps/defense_app/backend`：后端导入边界或 API 结构改动后的编译检查。
- `npm --prefix apps/defense_app/frontend run typecheck`：前端 TypeScript 类型检查。
- `npm --prefix apps/defense_app/frontend run build`：前端生产构建检查。
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

通过推荐入口生成候选、重排序、评价和主实验：

```sh
uv run python src/12_build_recommendation_inputs.py
uv run python src/13_build_recommend_candidates.py --strategy popularity
uv run python src/13_build_recommend_candidates.py --strategy trend_union
uv run python src/13_build_recommend_candidates.py --strategy default
uv run python src/13_build_recommend_candidates.py --strategy enhanced_default
uv run python src/13_build_recommend_candidates.py --strategy similarity
uv run python src/14_rerank_recommendations.py --method global_popularity
uv run python src/14_rerank_recommendations.py --method recent_popularity
uv run python src/14_rerank_recommendations.py --method attribute_similarity
uv run python src/14_rerank_recommendations.py --method pop_similarity
uv run python src/14_rerank_recommendations.py --method pop_similarity_trend
uv run python src/14_rerank_recommendations.py --method enhanced_pop_similarity_trend
uv run python src/15_eval_recommendations.py --method pop_similarity_trend
uv run python src/16_run_recommendation_experiment.py --experiment main
uv run python src/16_run_recommendation_experiment.py --experiment recommendation_enhanced
```

推荐阶段依赖已发布的 LightGBM stable 预测、周级交易和商品属性边。`12_build_recommendation_inputs.py` 默认读取 `outputs/models/lightgbm/predictions.csv`，因此如果该文件缺失，应先运行 LightGBM 训练和评价链路，而不是在推荐层添加 fallback。

`recommendation_enhanced` 是独立可选实验，只写入 `outputs/recommendation/experiments/recommendation_enhanced/`，不覆盖 `outputs/recommendation/experiments/main/experiment.json`，也不默认替换 `outputs/recommendation/pop_similarity_trend/` stable 结果。reports 和 defense app 仍消费稳定默认输出，除非后续显式调整其默认输入。

通过 reports 入口导出论文图表、表格、案例和 manifest：

```sh
uv run python src/17_export_paper_assets.py
```

reports 阶段依赖已发布的稳定趋势、推荐、图和数据 artifact。它只读取现有 artifact 并写入 `outputs/reports/`，不训练模型、不重跑推荐方法、不构建上游数据。如果本机缺少可用 CJK 字体，图表导出应 fail-fast，避免生成缺字中文图。

通过 presentation 入口构建本地答辩展示库：

```sh
uv run python src/18_build_defense_app_db.py
```

presentation 阶段依赖已发布的稳定趋势、推荐、reports、图和商品 artifact。它只读取现有 artifact，写入 `outputs/defense_app/fashion_demo.sqlite`，并复制展示所需 reports 图表到 `outputs/defense_app/static/reports/`。它不能训练模型、重跑推荐方法、构建候选、重导出 reports 或直接读取原始交易 CSV。

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

推荐中间产物使用：

- `data/processed/recommend/time_windows.parquet`
- `data/processed/recommend/target_users.parquet`
- `data/processed/recommend/evaluation_labels.parquet`
- `data/processed/recommend/user_profile.parquet`
- `data/processed/recommend/customer_profile.parquet`
- `data/processed/recommend/article_product_map.parquet`
- `data/processed/recommend/metadata.json`
- `data/processed/recommend/candidates/<strategy>/candidate_items.parquet`
- `data/processed/recommend/candidates/enhanced_default/candidate_items.parquet`
- `data/processed/recommend/features/<feature_name>/strategy=<strategy>/split=<split>/cutoff_week=<week>/part.parquet`

推荐 method stable 输出使用：

- `outputs/recommendation/<method>/recommendations.csv`
- `outputs/recommendation/<method>/recommendation_items.parquet`
- `outputs/recommendation/<method>/params.json`
- `outputs/recommendation/<method>/metadata.json`
- `outputs/recommendation/<method>/metrics.json`

第一阶段增强 method 输出使用 `outputs/recommendation/enhanced_pop_similarity_trend/`，但这不是默认 stable 主推荐输出。默认主推荐输出仍是 `outputs/recommendation/pop_similarity_trend/`。

推荐实验输出使用：

- `outputs/recommendation/experiments/<experiment_id>/experiment.json`
- `outputs/recommendation/experiments/<experiment_id>/runs/<run_id>/recommendations.csv`
- `outputs/recommendation/experiments/<experiment_id>/runs/<run_id>/recommendation_items.parquet`
- `outputs/recommendation/experiments/<experiment_id>/runs/<run_id>/params.json`
- `outputs/recommendation/experiments/<experiment_id>/runs/<run_id>/metadata.json`

第一阶段增强实验输出使用 `outputs/recommendation/experiments/recommendation_enhanced/`。该目录独立于 `outputs/recommendation/experiments/main/`，用于验证增强候选和增强打分，不作为 reports 或 defense app 的默认输入。

推荐结果必须保留 `customer_id`、`article_id` 和 `prediction` 的字符串语义，尤其不能丢失前导 0。`recommendations.csv` 是每个用户窗口一行的 Top-12 短表；`recommendation_items.parquet` 是用于解释、审计和评价的默认内部长表。同一用户窗口内推荐商品不能重复。`recommendation_items.csv` 只有显式导出时才应出现，不能作为默认产物或默认 reader 来源。

论文素材导出使用：

- `outputs/reports/figures/*.svg`
- `outputs/reports/figures/*.png`
- `outputs/reports/tables/*.csv`
- `outputs/reports/tables/*.md`
- `outputs/reports/case_studies/*.json`
- `outputs/reports/case_studies/*.md`
- `outputs/reports/manifest.json`

默认 reports 导出应包含核心图表的 SVG/PNG 双格式、CSV/Markdown 双格式表格、推荐解释案例和 `manifest.json`。Manifest 应记录导出参数、输入 artifact、输出 artifact、表格行数、案例用户和 warnings，方便论文素材审计。

本地答辩展示应用使用：

- `outputs/defense_app/fashion_demo.sqlite`
- `outputs/defense_app/static/reports/*.svg`
- `outputs/defense_app/static/reports/*.png`

这些文件都是生成产物，不应提交。展示库应记录 source artifact 的路径、mtime、size 和 row count，并保留 `article_id`、`customer_id`、`attr_id` 的字符串语义。后端默认只读取 `outputs/defense_app/fashion_demo.sqlite`，允许通过 `DEFENSE_APP_DB_PATH` 覆盖路径。

`16_run_recommendation_experiment.py` 的推荐实验 force 语义必须保持精确：

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
uv run python src/16_run_recommendation_experiment.py --experiment main --force-method pop_similarity
uv run python src/16_run_recommendation_experiment.py --experiment main --force-cache
uv run python src/16_run_recommendation_experiment.py --experiment main --force-candidates
uv run python src/16_run_recommendation_experiment.py --experiment main --force-rebuild-all
```

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
- `recommendation` 不能导入 trend training、trend models、trend evaluation runner 或 catalog graph builder；它只能消费上游公开 reader、contract 和已发布 artifact。
- `reports` 不能导入训练 runner、推荐实验 runner、候选构建、重排序实现或图构建实现；它只能通过 reader、paths 和稳定 artifact 做只读汇总。
- `presentation` 只能通过公开 reader、contract、reports loaders/paths 和稳定 artifact 做只读抽取；不能导入 datasets 下载、transactions weekly 构建、catalog graph builder、trend training/models/evaluation runner、recommendation runner/retrieval/ranking/experiments 或 reports runner。
- `apps/defense_app/backend` 只能通过 SQLite repository 查询展示库，不应直接读取 `data/`、`outputs/models/`、`outputs/recommendation/`、`outputs/reports/` 或原始 H&M 文件。
- `apps/defense_app/frontend` 只调用 FastAPI `/api` 接口，不应直接读取 SQLite、CSV、Parquet、本地文件路径或上游 artifact。

新增功能时，将其放在拥有对应业务事实的领域中。不要让编号脚本成为可复用逻辑的来源。

## 测试指南

项目使用 pytest。测试文件命名为 `tests/test_*.py`，新增测试应对齐真实流水线阶段：foundation artifact、商品清洗、属性图、商品销量、属性热度、标签、样本、切分、训练、LightGBM、趋势评价、推荐输入、推荐召回、推荐排序、推荐方法、推荐评价、推荐实验、reports 导出、presentation 展示库、defense app 后端 API 或架构边界。前端变更应至少运行 TypeScript typecheck；影响构建输出、路由或共享组件时运行 frontend build。

除非明确说明是 artifact 验证，否则测试不应依赖真实 H&M 数据集。优先使用小型内存 fixture，以及 `tests/trend_samples.py` 或 `tests/__init__.py` 中的共享 helper。

修复 bug 时，添加或更新一个修复前会失败的回归测试。修改模型、推荐方法、reports 导出、presentation 展示库或 artifact 契约时，同时验证 happy path 和边界失败，例如非法模型名、非法 method、非法 strategy、非法 `run_id`、缺失 split 列、不安全路径、缺失 eligible 用户、重复推荐商品、Top-K 越界、figure format 重复或非法、缺失 source artifact、SQLite schema 不匹配，以及 prediction/metrics payload 不匹配。

## 文档指南

当命令语法、artifact 路径、模型语义、展示应用行为或架构边界变化时，保持 `README.md`、`apps/defense_app/README.md`、`docs/gpt-research/implementation-plan.md`，以及相关 `docs/superpowers/specs/` 或 `docs/superpowers/plans/` 与 as-built 行为一致。

不要留下暗示已实现模块仍位于已删除根文件中的历史设计文本。当前趋势训练和评价代码位于 `src/fashion_trend/trend/` 下，推荐代码位于 `src/fashion_trend/recommendation/` 下，论文素材导出代码位于 `src/fashion_trend/reports/` 下，答辩展示库构建代码位于 `src/fashion_trend/presentation/` 下，本地展示应用位于 `apps/defense_app/` 下。

## Commit 与 Pull Request 指南

近期历史使用 Conventional Commit 前缀和简洁中文摘要，例如 `fix(trend): 修正预测校验和 LightGBM 日志`、`docs: 说明 LightGBM run 调参流程`。

按功能、阶段或契约边界保持 commit 粒度。提交前检查 `git diff`，确认改动范围，并运行相关验证命令。不要提交生成的数据集、模型输出、凭据、缓存或无关格式化改动。

Pull request 应说明变更的阶段或契约，列出验证命令，在相关时说明 artifact 路径，并指出数据或配置假设。

## 安全与配置提示

Kaggle 凭据、API token、`.env` 文件、原始数据集、生成输出、模型 artifact 和本机会话文件都不能进入提交。使用环境变量或仓库外的凭据文件访问数据。

不要在日志、文档、测试、快照、metadata 或提交信息中写入敏感信息。
