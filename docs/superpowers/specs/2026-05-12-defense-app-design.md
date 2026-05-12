# 答辩展示应用设计

## 范围

本设计覆盖毕业答辩用的本地 Web 展示应用。应用目标是把当前已经完成的 H&M 服装属性趋势预测和轻量 Top-N 推荐实验组织成一个正式、可操作、可解释的演示系统。

应用主线是：

```text
趋势看板
  -> 属性详情
  -> 属性图展示
  -> 推荐展示
  -> 推荐理由与用户画像
```

它是离线研究流水线的下游展示层：

```text
现有稳定 artifacts
  -> 构建只读 SQLite 展示库
  -> FastAPI 查询 API
  -> Vue 答辩展示应用
```

当前离线研究流水线仍然是主系统。答辩应用只消费已经发布的稳定产物，不训练模型、不重跑推荐、不重新构建候选、不修改实验输出。

## 非目标

本应用不实现以下内容：

- 在线推荐服务。
- 模型训练、调参、评价或 promotion。
- 任意全量用户实时推荐计算。
- 全量 `transactions_train.csv` 查询。
- 完整用户画像系统。
- 登录、权限、多用户协作或后台管理。
- PostgreSQL、Neo4j、向量数据库或外部服务部署。
- 将推荐模块包装成生产级推荐平台。

应用允许输入或搜索 `customer_id`，但第一版只支持预置高质量演示用户池。这样可以保留正式应用的交互感，同时避免现场输入稀疏用户导致推荐解释质量不稳定。

## 技术栈

采用：

```text
Frontend: Vue 3 + TypeScript + Vite
Backend: FastAPI
Database: SQLite
Charts: ECharts / vue-echarts
Icons: lucide-vue-next
Python DB access: sqlite3 标准库
```

选择 Vue 的原因：

- 页面数量固定，主要是看板、筛选、详情、图表和推荐卡片。
- Vue 单文件组件适合把展示页面的 template、logic 和 style 放在一起维护。
- 比 React 更少样板代码，适合答辩应用的中等复杂度。
- Vue Router、ECharts 和 TypeScript 已足够覆盖本项目需求。

选择 FastAPI 的原因：

- API 契约可以用类型标注和 Pydantic response model 明确表达。
- 自动 OpenAPI 文档能体现正式 API 工程边界。
- 适合薄查询层：读取 SQLite，返回结构化 JSON。
- 比 Flask 更适合多接口、强 response schema 和参数校验的场景。

选择 SQLite 的原因：

- 展示库是只读、可重建 artifact，不需要独立数据库服务。
- 支持搜索、筛选、关联查询和 explain 页面，比纯 JSON 更像正式应用。
- 部署和答辩风险低。
- 不需要 SQLAlchemy 或迁移系统。第一版使用 `sqlite3` 标准库即可。

## 依赖边界

答辩应用依赖不应无条件进入核心研究流水线的最小运行环境。后续实现时建议：

- Python 侧把 `fastapi`、`uvicorn` 放入单独的 `app` dependency group。
- 不把 `fastapi`、`uvicorn` 作为趋势训练、推荐实验和 reports 导出的必需依赖。
- 第一版不引入 SQLAlchemy、Alembic、DuckDB、PostgreSQL driver 或额外 ORM。
- 前端依赖只放在 `apps/defense_app/frontend/package.json`，不在仓库根目录新增 `package.json`。
- 前端第一版依赖控制在 `vue`、`vue-router`、`echarts`、`vue-echarts`、`lucide-vue-next`、`vite`、`typescript`、`@vitejs/plugin-vue` 和 `vue-tsc`。

这样可以保持已有 `uv run pytest`、趋势训练、推荐实验和 reports 导出的依赖边界清晰。需要运行答辩应用时，再显式安装应用依赖。

## 目录组织

新增应用代码放在仓库顶层 `apps/defense_app/`，展示库构建逻辑放在 `src/fashion_trend/presentation/`。两者职责分离。

目标结构：

```text
Fashion/
  apps/
    defense_app/
      README.md
      backend/
        app/
          main.py
          core/
            config.py
            database.py
          api/
            router.py
            routes/
              trends.py
              attributes.py
              articles.py
              demo_users.py
              recommendations.py
              metrics.py
          schemas/
            trends.py
            attributes.py
            articles.py
            demo_users.py
            recommendations.py
            metrics.py
          repositories/
            trend_repository.py
            attribute_repository.py
            article_repository.py
            demo_user_repository.py
            recommendation_repository.py
            metrics_repository.py
          services/
            trend_service.py
            attribute_graph_service.py
            recommendation_explanation_service.py
            metrics_service.py
        tests/
      frontend/
        package.json
        vite.config.ts
        tsconfig.json
        index.html
        src/
          main.ts
          router/
          api/
          views/
            TrendDashboardView.vue
            AttributeDetailView.vue
            AttributeGraphView.vue
            RecommendationView.vue
            UserProfileReasonView.vue
          components/
            layout/
            trend/
            attribute/
            graph/
            recommendation/
            metrics/
          charts/
          types/
          styles/
  src/
    18_build_defense_app_db.py
    fashion_trend/
      presentation/
        __init__.py
        contracts.py
        paths.py
        schema.py
        extractors.py
        builders.py
        sqlite_writer.py
  outputs/
    defense_app/
      fashion_demo.sqlite
      static/
  tests/
    test_presentation_*.py
```

边界规则：

- `src/fashion_trend/presentation/` 只负责从稳定 artifact 抽取展示数据并写 SQLite。
- `apps/defense_app/backend/` 只读 SQLite，不直接读取原始 CSV、Parquet 或模型输出。
- `apps/defense_app/frontend/` 只调用 FastAPI，不知道底层 artifact 路径。
- `outputs/defense_app/` 是生成产物目录，不应提交数据库文件。
- `reports` 仍是论文素材导出层，不承载 Web 应用。

## 数据构建流程

新增编号入口：

```sh
uv run python src/18_build_defense_app_db.py
```

该入口只做编排：

```text
读取现有 artifacts
  -> 抽取趋势看板数据
  -> 抽取属性详情数据
  -> 抽取属性图数据
  -> 筛选演示用户
  -> 抽取用户画像、推荐命中和解释字段
  -> 写入 outputs/defense_app/fashion_demo.sqlite
  -> 写入构建 metadata
```

展示库构建可以读取：

- `outputs/reports/manifest.json`
- `outputs/reports/tables/*.csv`
- `outputs/reports/case_studies/*.json`
- `outputs/models/lightgbm/predictions.csv`
- `outputs/models/lightgbm/feature_importance.csv`
- `outputs/metrics/*/trend_metrics.json`
- `outputs/recommendation/*/metrics.json`
- `outputs/recommendation/pop_similarity_trend/recommendation_items.parquet`
- `outputs/recommendation/experiments/main/experiment.json`
- `data/processed/recommend/time_windows.parquet`
- `data/processed/recommend/target_users.parquet`
- `data/processed/recommend/evaluation_labels.parquet`
- `data/processed/recommend/user_profile.parquet`
- `data/processed/trend/attribute_week_heat.csv`
- `data/processed/features/trend_model_samples.parquet`
- `data/processed/graph/nodes_article.csv`
- `data/processed/graph/nodes_attribute.csv`
- `data/processed/graph/edges_article_attribute.csv`
- `data/processed/graph/edges_attribute_hierarchy.csv`
- `data/interim/articles_clean.csv`

展示库不读取：

- 原始 `transactions_train.csv`
- 原始 `customers.csv`
- 全量商品图片目录
- 训练过程临时 run
- 历史 `recommendation_items.csv`

构建不变量：

- 先写临时 SQLite，校验通过后再原子替换 `outputs/defense_app/fashion_demo.sqlite`。
- 记录 source artifact 的路径、mtime、size 和 row count，以 JSON 写入 `app_metadata.source_artifacts`。
- 缺失必需 source、schema version 不匹配、关键行数为 0、Top-12 不完整时 fail-fast。
- 所有 `article_id`、`customer_id` 和 `attr_id` 在读取和写入时都按字符串处理，保留前导 0。
- 推荐命中 `is_hit` 从 `evaluation_labels.parquet` 与推荐 Top-12 按 user-window 和 `article_id` join 得出，不从推荐长表假定已有字段。
- 用户画像从 `user_profile.parquet` 读取，不从 case markdown 或推荐长表反推。

## SQLite 展示库 Schema

### `app_metadata`

记录展示库来源和构建信息。

```text
key TEXT PRIMARY KEY
value TEXT NOT NULL
```

典型字段：

- `schema_version`
- `generated_at`
- `source_manifest_path`
- `source_artifacts`
- `default_source_week`
- `case_count`
- `artifact_warnings`

### `demo_users`

预置演示用户池。

```text
case_id TEXT PRIMARY KEY
customer_id TEXT NOT NULL
split TEXT NOT NULL
cutoff_week INTEGER NOT NULL
label_week INTEGER NOT NULL
hit_count INTEGER NOT NULL
primary_tags TEXT NOT NULL
profile_summary TEXT NOT NULL
recommendation_summary TEXT NOT NULL
UNIQUE (customer_id, split, cutoff_week, label_week)
```

`primary_tags` 使用 JSON 字符串存储，例如：

```json
["高命中", "颜色偏好明显", "趋势解释清晰"]
```

`case_id` 是展示层稳定主键，建议格式为：

```text
demo_<split>_<cutoff_week>_<label_week>_<customer_id 前 12 位>
```

同一 `customer_id` 可以在不同 split 或不同窗口中形成不同演示 case。

演示用户筛选原则：

- Top-12 推荐中至少有明确命中或清晰负例价值。
- 用户历史偏好属性可解释。
- 推荐商品包含可解释的商品属性。
- 分数分解中 `pop_score`、`sim_score`、`trend_score`、`recent_score` 至少有一个明显主导信号。

第一版可以从已导出的 3 个 case 起步，后续扩展到 20-50 个高质量用户。

### `user_profile_attributes`

用户画像属性。

```text
case_id TEXT NOT NULL
customer_id TEXT NOT NULL
attr_id TEXT NOT NULL
attr_type TEXT NOT NULL
attr_value TEXT NOT NULL
preference_score REAL NOT NULL
purchase_count INTEGER NOT NULL
last_purchase_week INTEGER NOT NULL
PRIMARY KEY (case_id, attr_id)
```

### `recommendation_items`

Top-12 推荐商品。

```text
case_id TEXT NOT NULL
customer_id TEXT NOT NULL
article_id TEXT NOT NULL
rank INTEGER NOT NULL
score REAL NOT NULL
is_hit INTEGER NOT NULL
candidate_sources TEXT NOT NULL
PRIMARY KEY (case_id, rank)
```

`article_id` 必须保持字符串语义，保留前导 0。

### `recommendation_score_components`

推荐分数分解。

```text
case_id TEXT NOT NULL
article_id TEXT NOT NULL
pop_score REAL NOT NULL
sim_score REAL NOT NULL
trend_score REAL NOT NULL
recent_score REAL NOT NULL
final_score REAL NOT NULL
PRIMARY KEY (case_id, article_id)
```

### `articles`

展示商品核心字段。

```text
article_id TEXT PRIMARY KEY
prod_name TEXT
product_group_name TEXT
product_type_name TEXT
garment_group_name TEXT
colour_group_name TEXT
graphical_appearance_name TEXT
department_name TEXT
section_name TEXT
index_name TEXT
index_group_name TEXT
```

第一版不要求商品图片。如果本机图片路径可用，可以后续增加 `image_path` 字段，但页面必须在无图片时正常展示。

### `article_attributes`

商品-属性边。

```text
article_id TEXT NOT NULL
attr_id TEXT NOT NULL
attr_type TEXT NOT NULL
attr_value TEXT NOT NULL
PRIMARY KEY (article_id, attr_id)
```

用于属性图展示和推荐商品解释。

### `trend_attributes`

趋势看板榜单。

```text
source_week INTEGER NOT NULL
target_week INTEGER NOT NULL
attr_id TEXT NOT NULL
attr_type TEXT NOT NULL
attr_value TEXT NOT NULL
rank INTEGER NOT NULL
heat_t REAL NOT NULL
pred_share_t1 REAL
pred_target_growth REAL
is_trend_eligible_t INTEGER NOT NULL
PRIMARY KEY (source_week, attr_type, rank)
```

`source_week` 是模型产生预测时所在的当前周，也等价于展示推荐窗口里的 cutoff week；`target_week` 是被解释为“下一周趋势”的目标周，通常为 `source_week + 1`。页面文案应展示“预测第 `target_week` 周上升趋势”，避免把当前周和下一周混淆。

第一版默认展示核心属性类型：

- `colour_group_name`
- `product_type_name`
- `graphical_appearance_name`
- `garment_group_name`

也可以保留 `department_name` 等扩展类型，前端默认不放在主看板第一屏。

### `attribute_heat_series`

属性详情页最近 8 周曲线。

```text
attr_id TEXT NOT NULL
attr_type TEXT NOT NULL
attr_value TEXT NOT NULL
week_id INTEGER NOT NULL
heat REAL NOT NULL
actual_target_growth REAL
pred_target_growth REAL
pred_share_t1 REAL
PRIMARY KEY (attr_id, week_id)
```

用于属性详情页展示：

- 最近 8 周热度。
- 预测增长。
- 可用时展示真实结果。
- 预测与真实的方向对比。

### `attribute_hierarchy_edges`

属性层级边。

```text
parent_attr_id TEXT NOT NULL
child_attr_id TEXT NOT NULL
parent_attr_type TEXT NOT NULL
parent_attr_value TEXT NOT NULL
child_attr_type TEXT NOT NULL
child_attr_value TEXT NOT NULL
relation_type TEXT NOT NULL
PRIMARY KEY (parent_attr_id, child_attr_id, relation_type)
```

### `metrics_summary`

趋势模型和推荐方法指标摘要。

```text
metric_domain TEXT NOT NULL
model_or_method TEXT NOT NULL
split TEXT NOT NULL
metric_name TEXT NOT NULL
metric_value REAL NOT NULL
display_order INTEGER NOT NULL
PRIMARY KEY (metric_domain, model_or_method, split, metric_name)
```

`metric_domain` 取值：

- `trend`
- `recommendation`

### `report_assets`

记录可复用的报告图表。

```text
asset_name TEXT PRIMARY KEY
asset_type TEXT NOT NULL
title TEXT NOT NULL
source_path TEXT NOT NULL
static_url TEXT
description TEXT NOT NULL
```

`source_path` 只用于审计，不直接暴露给浏览器。若前端需要展示 SVG，构建步骤应把选中的图表复制到 `outputs/defense_app/static/reports/`，FastAPI 挂载为 `/static/reports/...`，并把可访问路径写入 `static_url`。第一版页面以数据库查询为主，SVG 图表作为辅助。

## 后端 API 设计

API 只读，统一前缀：

```text
/api
```

第一版 API 契约控制在页面所需的最小集合，不在设计文档中展开完整 JSON 示例。每个接口实现时应有 Pydantic response model。

| Endpoint | 参数 | 默认与上限 | 返回字段摘要 | 主要表 |
| --- | --- | --- | --- | --- |
| `GET /api/trends` | `source_week?: int`, `attr_type?: str`, `limit?: int` | `source_week=default_source_week`, `limit=10`, `max=50` | `source_week`, `target_week`, `groups[]` 或 `items[]` | `trend_attributes` |
| `GET /api/attributes/{attr_id}` | `attr_id: str`, `source_week?: int` | `source_week=default_source_week` | 属性基本信息、rank、source/target week | `trend_attributes` |
| `GET /api/attributes/{attr_id}/heat-series` | `attr_id: str`, `source_week?: int`, `weeks?: int` | `source_week=default_source_week`, `weeks=8`, `max=16` | `points[]`，含 `week_id`, `heat`, `pred_target_growth`, `actual_target_growth` | `attribute_heat_series` |
| `GET /api/attributes/{attr_id}/articles` | `attr_id: str`, `limit?: int` | `limit=20`, `max=100` | 关联商品摘要 | `article_attributes`, `articles` |
| `GET /api/attributes/{attr_id}/graph` | `attr_id: str` | 无 | 父子属性节点和边 | `attribute_hierarchy_edges` |
| `GET /api/articles/search` | `q: str`, `limit?: int` | `limit=10`, `max=50` | 商品搜索结果 | `articles` |
| `GET /api/articles/{article_id}` | `article_id: str` | 无 | 商品展示字段 | `articles` |
| `GET /api/articles/{article_id}/graph` | `article_id: str` | 无 | 商品节点、属性节点和边 | `articles`, `article_attributes` |
| `GET /api/demo-users` | `q?: str`, `tag?: str`, `limit?: int` | `limit=20`, `max=50` | 演示用户列表、标签、命中数摘要 | `demo_users` |
| `GET /api/demo-users/{case_id}` | `case_id: str` | 无 | 演示用户详情 | `demo_users` |
| `GET /api/demo-users/{case_id}/profile` | `case_id: str` | 无 | 用户偏好属性列表 | `user_profile_attributes` |
| `GET /api/demo-users/{case_id}/recommendations` | `case_id: str` | 无 | Top-12 推荐、商品摘要、命中标记 | `recommendation_items`, `articles` |
| `GET /api/demo-users/{case_id}/recommendations/{article_id}/explanation` | `case_id: str`, `article_id: str` | 无 | 用户偏好、商品属性、分数分解、趋势相关属性 | `recommendation_score_components`, `user_profile_attributes`, `article_attributes`, `trend_attributes` |
| `GET /api/metrics/summary` | 无 | 无 | 趋势和推荐核心指标摘要 | `metrics_summary` |
| `GET /api/metrics/trend` | `split?: str` | `split` 为空返回 valid/test | 趋势模型指标 | `metrics_summary` |
| `GET /api/metrics/recommendation` | `split?: str` | `split` 为空返回 valid/test | 推荐方法指标 | `metrics_summary` |

排序规则：

- 趋势榜按 `rank ASC`。
- 推荐结果按 `rank ASC`。
- 演示用户默认按 `hit_count DESC, case_id ASC`。
- 搜索结果默认按精确前缀命中优先，再按 id 升序。

错误 payload 统一为：

```json
{
  "detail": {
    "code": "not_found",
    "message": "未找到指定演示用户"
  }
}
```

## 前端页面设计

前端路由：

| Route | 页面 | 关键参数 |
| --- | --- | --- |
| `/` | 趋势看板 | `source_week`, `attr_type` query |
| `/attributes/:attrId` | 属性详情 | `attrId` path |
| `/graph/articles/:articleId?` | 属性图展示 | `articleId` path 或搜索 query |
| `/recommendations` | 推荐展示 | `case_id`, `tag`, `q` query |
| `/recommendations/:caseId/:articleId?` | 推荐理由与用户画像 | `caseId`, `articleId` path |

### 趋势看板

目标：展示“下一周哪些颜色、品类、图案、服装组上升”。

主要区域：

- 顶部指标摘要：数据规模、趋势主模型、推荐主方法。
- 四列趋势榜：颜色、品类、图案、服装组。
- 属性类型筛选。
- Top-K 控制。
- 点击属性进入属性详情。

数据来源：

- `trend_attributes`
- `metrics_summary`

### 属性详情

目标：解释某个属性为什么被认为是趋势属性。

主要区域：

- 属性基本信息。
- 最近 8 周热度曲线。
- 预测增长与真实增长。
- 关联商品样例。
- 所属父级或子级属性。

数据来源：

- `trend_attributes`
- `attribute_heat_series`
- `article_attributes`
- `articles`
- `attribute_hierarchy_edges`

### 属性图展示

目标：展示商品如何连接到颜色、品类、图案、服装组和部门等属性，并支持从属性反查代表商品。

主要区域：

- 商品搜索。
- 商品中心图。
- 属性节点分组。
- 关系说明。
- 代表商品列表。

数据来源：

- `articles`
- `article_attributes`
- `attribute_hierarchy_edges`

图展示第一版可以用轻量自定义节点布局，不必引入复杂图数据库。需要更强交互时再评估图组件库。

### 推荐展示

目标：选择一个演示用户，输出 Top-12 推荐商品。

主要区域：

- 用户搜索框。
- 用户标签筛选。
- 演示用户列表。
- Top-12 推荐商品卡片。
- 命中标记。
- 商品属性摘要。

数据来源：

- `demo_users`
- `recommendation_items`
- `articles`
- `article_attributes`

### 推荐理由与用户画像

目标：解释“为什么推荐这个商品”。

主要区域：

- 用户偏好属性：偏好分、购买次数、最近购买周。
- 推荐商品属性：颜色、品类、图案、服装组。
- 趋势相关属性。
- `pop_score`、`sim_score`、`trend_score`、`recent_score` 分数分解。
- 最终分数。

数据来源：

- `user_profile_attributes`
- `recommendation_items`
- `recommendation_score_components`
- `articles`
- `article_attributes`
- `trend_attributes`

## 前后端职责边界

后端负责：

- SQLite 连接和只读查询。
- 参数校验。
- response schema。
- 简单聚合和 explain payload 拼装。
- 明确错误码。

前端负责：

- 页面路由。
- UI 状态。
- 筛选控件。
- 图表渲染。
- 用户操作流程。
- 空状态和加载状态。

前端不直接读取 SQLite、CSV、Parquet 或 `outputs/` 路径。

后端不读取全量原始 artifact，不做 pandas 查询，不重新计算推荐分数。

## 错误处理

展示库不存在时：

```json
{
  "detail": {
    "code": "database_unavailable",
    "message": "展示库不存在，请先运行 src/18_build_defense_app_db.py"
  }
}
```

查询不到用户、属性或商品时：

```json
{
  "detail": {
    "code": "not_found",
    "message": "未找到指定演示用户"
  }
}
```

非法参数时：

```json
{
  "detail": {
    "code": "validation_error",
    "message": "请求参数不合法"
  }
}
```

数据库 schema 版本不匹配时：

```json
{
  "detail": {
    "code": "schema_version_mismatch",
    "message": "展示库 schema_version 与应用不兼容"
  }
}
```

前端需要有对应空状态：

- 无匹配演示用户。
- 无关联商品。
- 无属性热度曲线。
- 推荐解释缺少某个分数组件。

## 测试策略

### 展示库构建测试

放在 `tests/test_presentation_*.py`：

- 生成临时 SQLite。
- 校验必需表存在。
- 校验 `article_id` 和 `customer_id` 保持字符串语义。
- 校验 demo user 至少包含 Top-12 推荐。
- 校验推荐分数组件完整。
- 校验趋势看板至少包含四类核心属性榜单。
- 校验缺失 artifact 时 fail-fast，不生成半成品数据库。

### 架构边界测试

新增 `fashion_trend.presentation` 后，需要同步更新 `tests/test_architecture_boundaries.py`：

- 将 `presentation` 纳入业务域集合。
- 允许它只读依赖 `foundation`、各领域 public contracts/readers、稳定路径契约和推荐 strict readers。
- 禁止它导入 trend training、trend models、trend evaluation runner、recommendation runner、candidate builder、reranker 或 reports runner。

### 后端测试

放在 `apps/defense_app/backend/tests/`：

- 使用临时 SQLite fixture。
- 测试每个 API 的 happy path。
- 测试 404、422、schema version mismatch。
- 测试推荐解释 payload 包含用户画像、商品属性、分数分解。

### 前端测试

第一版只要求轻量验证：

- TypeScript 编译通过。
- 关键页面能在 mock API 下渲染。
- 趋势看板、推荐展示、推荐理由页没有空白崩溃。

如果后续引入端到端测试，优先覆盖：

- 首页进入属性详情。
- 搜索演示用户并打开 Top-12 推荐。
- 展开某个推荐商品解释。

## 开发与运行方式

数据库构建：

```sh
uv run python src/18_build_defense_app_db.py
```

后端开发：

```sh
uv run --group app uvicorn app.main:app --reload --app-dir apps/defense_app/backend
```

前端开发：

```sh
cd apps/defense_app/frontend
npm install
npm run dev
```

第一版可以使用 Vite dev server 代理 `/api` 到 FastAPI。最终答辩前可以增加 build 模式，由 FastAPI 静态服务前端 `dist/`，但这不是第一阶段必须项。

## 验收标准

设计实现完成后，应满足：

- 可以一条命令构建 `outputs/defense_app/fashion_demo.sqlite`。
- FastAPI 启动后 `/docs` 可查看所有 API。
- Vue 应用包含趋势看板、属性详情、属性图展示、推荐展示、推荐理由五个页面。
- 趋势看板能展示颜色、品类、图案、服装组的 Top-K 上升属性。
- 属性详情能展示最近 8 周热度曲线和预测值。
- 属性图页能展示某商品连接到哪些属性。
- 推荐展示能搜索或选择预置演示用户，并展示 Top-12 商品。
- 推荐理由页能展示用户偏好属性、商品属性、趋势属性和分数分解。
- 应用不依赖原始大 CSV，也不在启动时扫描大型 Parquet。
- 应用文案不把推荐模块表述成生产级在线推荐系统。

## 后续实施建议

建议按以下提交边界推进：

1. 展示库 schema、构建入口和最小 SQLite 生成。
2. FastAPI 只读查询层和 API tests。
3. Vue 项目骨架、路由和基础布局。
4. 趋势看板与属性详情。
5. 属性图展示。
6. 推荐展示与推荐理由。
7. 文档、运行命令和最终验收。

每个阶段完成后运行对应最小验证，避免最后形成混在一起的大 diff。
