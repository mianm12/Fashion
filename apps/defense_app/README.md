# Defense App

本目录是本地答辩展示应用，用于把已经生成的趋势预测、属性图、推荐结果和论文案例整理成只读演示界面。它不是在线推荐服务、生产推荐平台或实时个性化系统。

## 数据库构建

先确保上游稳定 artifact 已存在，包括清洗商品、属性图、趋势预测、推荐结果、推荐案例和 reports 输出。然后构建展示库：

```sh
uv run python src/18_build_defense_app_db.py
```

默认输出：

```text
outputs/defense_app/fashion_demo.sqlite
```

该 SQLite 文件是生成产物，不应提交到版本库。

## 后端

后端位于 `apps/defense_app/backend/`，通过 FastAPI 只读查询 SQLite。默认读取 `outputs/defense_app/fashion_demo.sqlite`；如需覆盖路径，设置 `DEFENSE_APP_DB_PATH`。

```sh
uv run --group app uvicorn app.main:app --reload --app-dir apps/defense_app/backend
```

验证：

```sh
uv run --group app pytest apps/defense_app/backend/tests
uv run --group app python -m compileall -q apps/defense_app/backend
```

## 前端

前端位于 `apps/defense_app/frontend/`，只调用 FastAPI `/api` 接口，不直接读取 SQLite、CSV、Parquet 或上游 artifact。

```sh
cd apps/defense_app/frontend
npm install
npm run dev
```

验证：

```sh
npm run typecheck
npm run build
```

`node_modules/` 和 `dist/` 是本地依赖与构建产物，不应提交。

## 页面范围

- `/`：趋势看板，展示四类核心属性趋势榜和摘要指标。
- `/attributes/:attrId`：属性详情，展示 8 周热度曲线、预测值、属性关系和关联商品。
- `/graph/articles/:articleId`：商品属性图，展示商品到属性的只读连接。
- `/graph/articles?attr_id={attrId}`：从属性入口选择代表商品并进入商品属性图。
- `/recommendations`：演示用户选择和 Top-12 推荐。
- `/recommendations/:caseId/:articleId`：推荐解释，展示用户画像、商品属性、匹配趋势属性和 score breakdown。

## 视觉 QA

本应用按桌面答辩场景验证，建议使用 `1440x900` 或更宽视口。完整本地检查顺序：

```sh
uv run python src/18_build_defense_app_db.py
sqlite3 outputs/defense_app/fashion_demo.sqlite "pragma integrity_check;"
uv run --group app pytest apps/defense_app/backend/tests
uv run pytest tests/test_presentation_builders.py tests/test_presentation_runner.py tests/test_architecture_boundaries.py
cd apps/defense_app/frontend
npm run typecheck
npm run build
```

本地预览：

```sh
uv run --group app uvicorn app.main:app --reload --app-dir apps/defense_app/backend
cd apps/defense_app/frontend
npm run dev
```

预览时依次走查 `/`、`/attributes/:attrId`、`/graph/articles/:articleId`、`/recommendations` 和 `/recommendations/:caseId/:articleId`。重点确认左侧 shell、页面工具栏、趋势榜、属性关系树、图画布、Top-12 推荐网格、推荐理由分数条和 Top-12 上下文都能在桌面视口中完整阅读。

## 边界

- 后端只读 SQLite，不训练模型、不重跑推荐、不构建候选。
- 前端只调用后端 API，不读取本地数据文件。
- 展示库依赖稳定上游产物，不替代 `src/00_*.py` 到 `src/17_*.py` 的流水线。
- UI 文案应保持“本地答辩展示应用、离线趋势预测、轻量 Top-N 推荐实验、推荐解释”定位。
