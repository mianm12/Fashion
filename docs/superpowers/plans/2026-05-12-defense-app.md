# Defense App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local defense demo application that turns existing stable Fashion trend, recommendation, graph, reports, and metrics artifacts into a read-only SQLite-backed FastAPI + Vue presentation system.

**Architecture:** Add a `fashion_trend.presentation` layer that builds `outputs/defense_app/fashion_demo.sqlite` from stable artifacts only. Add `apps/defense_app/backend` as a thin read-only FastAPI API over SQLite, and `apps/defense_app/frontend` as a Vue 3 + TypeScript + Vite app that only calls `/api`. No model training, recommendation reranking, candidate generation, upstream artifact mutation, raw transaction scanning, or production-service claims belong in this app.

**Tech Stack:** Python 3.10-3.12, pandas, pyarrow, sqlite3, FastAPI, uvicorn, pytest, Vue 3, TypeScript, Vite, Vue Router, ECharts, vue-echarts, lucide-vue-next.

---

## Execution Mode

Use **serial subagent-driven execution beginning with Task 1**, with the main thread acting as integrator and reviewer after every task. Do not run multiple code-writing workers in parallel because the tasks share contracts, generated schemas, API payload names, and frontend types.

If subagents are unavailable or too much context must remain local, use inline execution with the same task gates. In either mode:

- Review `git diff` after every task.
- Run the task-specific test command before moving on.
- Fix issues immediately in the same task boundary.
- Do not commit unless the user explicitly requests commits.
- Do not submit generated SQLite files, frontend `node_modules`, build outputs, caches, or large artifacts.

## File Structure

Create or modify these files:

- Modify: `pyproject.toml`
  - Add optional `app` dependency group with `fastapi` and `uvicorn`.
- Modify: `uv.lock`
  - Update only if dependency resolution is run.
- Create: `src/18_build_defense_app_db.py`
  - Thin CLI orchestration entrypoint for SQLite build.
- Create: `src/fashion_trend/presentation/__init__.py`
  - Package marker and public version constant.
- Create: `src/fashion_trend/presentation/contracts.py`
  - Schema version, core attr types, table names, limits, metric names, and artifact keys.
- Create: `src/fashion_trend/presentation/paths.py`
  - Defense app output paths and stable source artifact path helpers.
- Create: `src/fashion_trend/presentation/schema.py`
  - SQLite DDL and schema validation helpers.
- Create: `src/fashion_trend/presentation/source_artifacts.py`
  - Source artifact metadata collection: path, mtime, size, row count.
- Create: `src/fashion_trend/presentation/extractors.py`
  - Read stable artifacts into typed DataFrames and Python payloads.
- Create: `src/fashion_trend/presentation/builders.py`
  - Convert extracted artifacts into SQLite-ready tables.
- Create: `src/fashion_trend/presentation/sqlite_writer.py`
  - Atomic SQLite creation, table writes, indexes, validation, and replacement.
- Create: `src/fashion_trend/presentation/runner.py`
  - Orchestrate extraction, table build, static report asset copy, and final SQLite publish.
- Modify: `tests/test_architecture_boundaries.py`
  - Add `presentation` as a domain and enforce read-only upstream imports.
- Create: `tests/test_presentation_schema.py`
- Create: `tests/test_presentation_builders.py`
- Create: `tests/test_presentation_runner.py`
- Create: `apps/defense_app/README.md`
- Create: `apps/defense_app/backend/app/main.py`
- Create: `apps/defense_app/backend/app/core/config.py`
- Create: `apps/defense_app/backend/app/core/database.py`
- Create: `apps/defense_app/backend/app/api/router.py`
- Create: `apps/defense_app/backend/app/api/routes/trends.py`
- Create: `apps/defense_app/backend/app/api/routes/attributes.py`
- Create: `apps/defense_app/backend/app/api/routes/articles.py`
- Create: `apps/defense_app/backend/app/api/routes/demo_users.py`
- Create: `apps/defense_app/backend/app/api/routes/recommendations.py`
- Create: `apps/defense_app/backend/app/api/routes/metrics.py`
- Create: `apps/defense_app/backend/app/schemas/common.py`
- Create: `apps/defense_app/backend/app/schemas/trends.py`
- Create: `apps/defense_app/backend/app/schemas/attributes.py`
- Create: `apps/defense_app/backend/app/schemas/articles.py`
- Create: `apps/defense_app/backend/app/schemas/demo_users.py`
- Create: `apps/defense_app/backend/app/schemas/recommendations.py`
- Create: `apps/defense_app/backend/app/schemas/metrics.py`
- Create: `apps/defense_app/backend/app/repositories/sql.py`
- Create: `apps/defense_app/backend/app/repositories/trend_repository.py`
- Create: `apps/defense_app/backend/app/repositories/attribute_repository.py`
- Create: `apps/defense_app/backend/app/repositories/article_repository.py`
- Create: `apps/defense_app/backend/app/repositories/demo_user_repository.py`
- Create: `apps/defense_app/backend/app/repositories/recommendation_repository.py`
- Create: `apps/defense_app/backend/app/repositories/metrics_repository.py`
- Create: `apps/defense_app/backend/app/services/attribute_graph_service.py`
- Create: `apps/defense_app/backend/app/services/recommendation_explanation_service.py`
- Create: `apps/defense_app/backend/app/services/metrics_service.py`
- Create: `apps/defense_app/backend/tests/conftest.py`
- Create: `apps/defense_app/backend/tests/test_api_trends.py`
- Create: `apps/defense_app/backend/tests/test_api_attributes.py`
- Create: `apps/defense_app/backend/tests/test_api_articles.py`
- Create: `apps/defense_app/backend/tests/test_api_demo_users.py`
- Create: `apps/defense_app/backend/tests/test_api_recommendations.py`
- Create: `apps/defense_app/backend/tests/test_api_metrics.py`
- Create: `apps/defense_app/frontend/package.json`
- Create: `apps/defense_app/frontend/vite.config.ts`
- Create: `apps/defense_app/frontend/tsconfig.json`
- Create: `apps/defense_app/frontend/index.html`
- Create: `apps/defense_app/frontend/src/main.ts`
- Create: `apps/defense_app/frontend/src/router/index.ts`
- Create: `apps/defense_app/frontend/src/api/client.ts`
- Create: `apps/defense_app/frontend/src/api/defenseApi.ts`
- Create: `apps/defense_app/frontend/src/types/api.ts`
- Create: `apps/defense_app/frontend/src/views/TrendDashboardView.vue`
- Create: `apps/defense_app/frontend/src/views/AttributeDetailView.vue`
- Create: `apps/defense_app/frontend/src/views/AttributeGraphView.vue`
- Create: `apps/defense_app/frontend/src/views/RecommendationView.vue`
- Create: `apps/defense_app/frontend/src/views/UserProfileReasonView.vue`
- Create: `apps/defense_app/frontend/src/components/layout/AppShell.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendMetricStrip.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendBoard.vue`
- Create: `apps/defense_app/frontend/src/components/attribute/AttributeHeatChart.vue`
- Create: `apps/defense_app/frontend/src/components/graph/ArticleAttributeGraph.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/DemoUserList.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/RecommendationGrid.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/ScoreBreakdown.vue`
- Create: `apps/defense_app/frontend/src/styles/main.css`
- Modify: `README.md`
  - Add the defense app build/run commands and boundary statement.
- Modify: `docs/gpt-research/project-status-summary.md`
  - Add the defense demo app as a downstream presentation artifact after it is verified.

## Task 1: Presentation Contracts, Paths, and SQLite Schema

**Files:**
- Create: `src/fashion_trend/presentation/__init__.py`
- Create: `src/fashion_trend/presentation/contracts.py`
- Create: `src/fashion_trend/presentation/paths.py`
- Create: `src/fashion_trend/presentation/schema.py`
- Create: `tests/test_presentation_schema.py`

- [ ] **Step 1: Write schema tests**

Create tests that build an in-memory SQLite database, apply the DDL, and assert all required tables exist:

```python
import sqlite3

from fashion_trend.presentation.contracts import PRESENTATION_SCHEMA_VERSION
from fashion_trend.presentation.schema import apply_schema, read_schema_version


def test_apply_schema_creates_required_tables():
    connection = sqlite3.connect(":memory:")
    apply_schema(connection)

    tables = {
        row[0]
        for row in connection.execute(
            "select name from sqlite_master where type = 'table'"
        )
    }

    assert {
        "app_metadata",
        "demo_users",
        "user_profile_attributes",
        "recommendation_items",
        "recommendation_score_components",
        "articles",
        "article_attributes",
        "trend_attributes",
        "attribute_heat_series",
        "attribute_hierarchy_edges",
        "metrics_summary",
        "report_assets",
    } <= tables
    assert read_schema_version(connection) == PRESENTATION_SCHEMA_VERSION
```

- [ ] **Step 2: Add stable contracts**

Define:

```python
PRESENTATION_SCHEMA_VERSION = "defense_app_v1"
DEFAULT_TOP_K = 10
MAX_TREND_LIMIT = 50
MAX_ARTICLE_LIMIT = 100
MAX_DEMO_USER_LIMIT = 50
CORE_TREND_ATTR_TYPES = (
    "colour_group_name",
    "product_type_name",
    "graphical_appearance_name",
    "garment_group_name",
)
MAIN_RECOMMENDATION_METHOD = "pop_similarity_trend"
```

- [ ] **Step 3: Add paths**

Use `foundation.paths.OUTPUT_DIR`, `DATA_DIR`, and explicit stable artifact paths only. Include:

```python
DEFENSE_APP_OUTPUT_DIR = OUTPUT_DIR / "defense_app"
DEFENSE_APP_DB_PATH = DEFENSE_APP_OUTPUT_DIR / "fashion_demo.sqlite"
DEFENSE_APP_STATIC_DIR = DEFENSE_APP_OUTPUT_DIR / "static"
REPORTS_MANIFEST_PATH = OUTPUT_DIR / "reports" / "manifest.json"
```

- [ ] **Step 4: Add SQLite schema DDL**

Implement `apply_schema(connection)` and `read_schema_version(connection)`. The DDL must match the design document table names and primary keys, store ids as `TEXT`, and write `PRESENTATION_SCHEMA_VERSION` into `app_metadata`.

- [ ] **Step 5: Run focused verification**

Run:

```sh
uv run pytest tests/test_presentation_schema.py
uv run python -m compileall -q src
```

Expected: both commands pass.

Review gate: inspect `git diff -- src/fashion_trend/presentation tests/test_presentation_schema.py` and confirm this task only adds contracts, paths, and schema helpers.

## Task 2: Artifact Extractors and SQLite Table Builders

**Files:**
- Create: `src/fashion_trend/presentation/source_artifacts.py`
- Create: `src/fashion_trend/presentation/extractors.py`
- Create: `src/fashion_trend/presentation/builders.py`
- Create: `tests/test_presentation_builders.py`

- [ ] **Step 1: Write builder tests with tiny fixtures**

Use small in-memory DataFrames and JSON-like dicts to verify:

- `article_id`, `customer_id`, and `attr_id` stay strings.
- Demo users get stable `case_id` values.
- Top-12 validation fails when a case has fewer than 12 recommendation items.
- `is_hit` is computed by joining recommendations to labels on `customer_id`, `split`, `cutoff_week`, `label_week`, and `article_id`.
- Core trend board output contains the four required attr types.

- [ ] **Step 2: Implement source artifact metadata**

For every source path, collect:

```python
{
    "path": str(path),
    "mtime": path.stat().st_mtime,
    "size": path.stat().st_size,
    "row_count": row_count_or_null,
}
```

Fail fast with `FileNotFoundError` for required artifacts.

- [ ] **Step 3: Implement extractors**

Read these stable artifacts:

- `outputs/reports/manifest.json`
- `outputs/reports/tables/*.csv`
- `outputs/models/lightgbm/predictions.csv`
- `outputs/models/lightgbm/feature_importance.csv`
- `outputs/metrics/*/trend_metrics.json`
- `outputs/recommendation/*/metrics.json`
- `outputs/recommendation/pop_similarity_trend/recommendation_items.parquet`
- `outputs/recommendation/experiments/main/experiment.json`
- `data/processed/recommend/evaluation_labels.parquet`
- `data/processed/recommend/user_profile.parquet`
- `data/processed/trend/attribute_week_heat.csv`
- `data/processed/features/trend_model_samples.parquet`
- `data/processed/graph/nodes_article.csv`
- `data/processed/graph/nodes_attribute.csv`
- `data/processed/graph/edges_article_attribute.csv`
- `data/processed/graph/edges_attribute_hierarchy.csv`
- `data/interim/articles_clean.csv`

Demo users are selected in the presentation layer from
`recommendation_items.parquet`, `evaluation_labels.parquet`, and
`user_profile.parquet`; the paper `case_studies` export remains independent from the
defense app user pool.

Do not read raw `transactions_train.csv`, raw `customers.csv`, training run directories, or historical `recommendation_items.csv`.

- [ ] **Step 4: Implement builders**

Build these DataFrames exactly:

- `app_metadata`: JSON-encoded source artifact audit data, generated timestamp, default source week, case count, artifact warnings.
- `demo_users`: from report cases, enriched with hit count, tags, profile summary, and recommendation summary.
- `user_profile_attributes`: from `user_profile.parquet`, filtered to demo cases.
- `recommendation_items`: Top-12 from parquet, filtered to demo cases, joined to labels for `is_hit`.
- `recommendation_score_components`: score columns from recommendation parquet.
- `articles`: display fields from `articles_clean.csv`.
- `article_attributes`: from `edges_article_attribute.csv` joined to attribute node labels when needed.
- `trend_attributes`: from LightGBM predictions and trend samples, ranked by `pred_target_growth` within each `source_week` and `attr_type`.
- `attribute_heat_series`: last 8 weeks per trend attribute.
- `attribute_hierarchy_edges`: from hierarchy edge artifact with parent/child labels.
- `metrics_summary`: trend and recommendation metrics flattened to valid/test summary rows.
- `report_assets`: selected SVG/PNG report assets with static URLs under `/static/reports/`.

- [ ] **Step 5: Run focused verification**

Run:

```sh
uv run pytest tests/test_presentation_builders.py
uv run python -m compileall -q src
```

Expected: tests pass and compile succeeds.

Review gate: inspect builder diff for raw artifact reads, id dtype handling, and fail-fast checks.

## Task 3: SQLite Writer, CLI Entrypoint, and Architecture Boundary

**Files:**
- Create: `src/fashion_trend/presentation/sqlite_writer.py`
- Create: `src/fashion_trend/presentation/runner.py`
- Create: `src/18_build_defense_app_db.py`
- Modify: `tests/test_architecture_boundaries.py`
- Create: `tests/test_presentation_runner.py`

- [ ] **Step 1: Write runner tests**

Use temporary source artifacts and temporary output paths to verify:

- A complete fixture writes a SQLite database with all required tables.
- Missing required source raises a clear exception and does not leave the final database path.
- A case with fewer than 12 recommendations fails validation.
- Atomic replacement writes a temp database first and replaces only after validation.

- [ ] **Step 2: Implement SQLite writer**

Use `sqlite3` only. Implement:

```python
def write_presentation_database(tables: Mapping[str, pd.DataFrame], output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    apply_schema(connection)
    for table_name, frame in tables.items():
        frame.to_sql(table_name, connection, if_exists="append", index=False)
    validate_database(connection)
    tmp_path.replace(output_path)
```

Ensure `finally` removes the temp file on failure.

- [ ] **Step 3: Implement runner**

`run_defense_app_db_build(output_path: Path = DEFENSE_APP_DB_PATH) -> dict[str, object]` should:

1. Load stable sources.
2. Build table DataFrames.
3. Copy selected report SVG/PNG assets into `outputs/defense_app/static/reports/`.
4. Write SQLite atomically.
5. Return a payload with `database_path`, `table_counts`, `source_artifacts`, and `static_assets`.

- [ ] **Step 4: Add numbered CLI**

`src/18_build_defense_app_db.py` should parse `--output-path`, call the runner, log table counts, and return a stable exit code. It should not contain business logic.

- [ ] **Step 5: Update architecture tests**

Add `presentation` to `BUSINESS_DOMAINS`. Add an allowlist test that permits only public read-only contracts/readers/paths and forbids training, model, evaluation runner, recommendation runner, candidate builder, reranker, and reports runner imports.

- [ ] **Step 6: Run focused verification**

Run:

```sh
uv run pytest tests/test_presentation_*.py tests/test_architecture_boundaries.py
uv run python src/18_build_defense_app_db.py
uv run python -m compileall -q src
```

Expected: tests pass, compile succeeds, and the CLI builds `outputs/defense_app/fashion_demo.sqlite`.

Review gate: inspect `sqlite3 outputs/defense_app/fashion_demo.sqlite ".tables"` and a few row counts before proceeding.

## Task 4: FastAPI Backend Dependencies and Read-Only Query Layer

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create backend files under `apps/defense_app/backend/app/`
- Create backend tests under `apps/defense_app/backend/tests/`

- [ ] **Step 1: Add app dependency group**

Add:

```toml
[dependency-groups]
app = [
    "fastapi>=0.116.0",
    "uvicorn>=0.35.0",
]
dev = [
    "black>=26.3.1",
    "isort>=8.0.1",
    "pytest>=9.0.3",
]
```

If backend request-level tests need `fastapi.testclient` and the installed FastAPI package does not provide `httpx`, add `httpx` to `dev` with a short note in the task review.

- [ ] **Step 2: Write backend tests**

Use a temp SQLite fixture seeded with the same tables from Task 1. Cover:

- `GET /api/trends`
- `GET /api/attributes/{attr_id}`
- `GET /api/attributes/{attr_id}/heat-series`
- `GET /api/articles/search`
- `GET /api/articles/{article_id}/graph`
- `GET /api/demo-users`
- `GET /api/demo-users/{case_id}/recommendations`
- `GET /api/demo-users/{case_id}/recommendations/{article_id}/explanation`
- `GET /api/metrics/summary`
- 404 for missing user, attribute, and article
- schema version mismatch error

- [ ] **Step 3: Implement core config and database**

Read the SQLite path from `DEFENSE_APP_DB_PATH` env var when set; otherwise default to `outputs/defense_app/fashion_demo.sqlite`. Open connections read-only with SQLite URI mode:

```python
sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
```

If the file does not exist, raise the `database_unavailable` payload.

- [ ] **Step 4: Implement repositories**

Repository methods should contain SQL only, return dicts/lists, bind all parameters with `?`, and enforce limits after validating maximums. Do not read CSV, Parquet, or files from repositories.

- [ ] **Step 5: Implement services**

Services should assemble graph and explanation payloads from repository results:

- Attribute graph: current attribute, parent/child attributes, and edges.
- Article graph: article node, attribute nodes, and article-attribute edges.
- Recommendation explanation: user profile attributes, item attributes, score components, and matching trend attributes.

- [ ] **Step 6: Implement routes and response schemas**

Every endpoint in the design document must have a Pydantic response model. Error payloads should use:

```json
{"detail": {"code": "not_found", "message": "未找到指定演示用户"}}
```

- [ ] **Step 7: Run focused verification**

Run:

```sh
uv sync --group app
uv run --group app pytest apps/defense_app/backend/tests
uv run --group app python -m compileall -q apps/defense_app/backend
```

Expected: tests pass and backend compiles.

Review gate: inspect backend diff for direct artifact reads, SQL string interpolation, and missing response models.

## Task 5: Vue Frontend Skeleton, Routing, API Types, and Layout

**Files:**
- Create: `apps/defense_app/frontend/package.json`
- Create: `apps/defense_app/frontend/vite.config.ts`
- Create: `apps/defense_app/frontend/tsconfig.json`
- Create: `apps/defense_app/frontend/index.html`
- Create: `apps/defense_app/frontend/src/main.ts`
- Create: `apps/defense_app/frontend/src/router/index.ts`
- Create: `apps/defense_app/frontend/src/api/client.ts`
- Create: `apps/defense_app/frontend/src/api/defenseApi.ts`
- Create: `apps/defense_app/frontend/src/types/api.ts`
- Create: `apps/defense_app/frontend/src/components/layout/AppShell.vue`
- Create: `apps/defense_app/frontend/src/styles/main.css`

- [ ] **Step 1: Add frontend package**

Use only the frontend dependencies from the design:

```json
{
  "scripts": {
    "dev": "vite --host 127.0.0.1",
    "build": "vue-tsc --noEmit && vite build",
    "typecheck": "vue-tsc --noEmit"
  },
  "dependencies": {
    "@vitejs/plugin-vue": "^6.0.0",
    "echarts": "^6.0.0",
    "lucide-vue-next": "^0.468.0",
    "typescript": "^5.9.0",
    "vite": "^7.0.0",
    "vue": "^3.5.0",
    "vue-echarts": "^8.0.0",
    "vue-router": "^4.5.0",
    "vue-tsc": "^3.0.0"
  },
  "devDependencies": {}
}
```

Do not create a root `package.json`.

- [ ] **Step 2: Add Vite proxy**

Configure `/api` to proxy to `http://127.0.0.1:8000`.

- [ ] **Step 3: Add API types and client**

Types should mirror backend response models and keep ids as strings:

```ts
export interface TrendAttribute {
  source_week: number;
  target_week: number;
  attr_id: string;
  attr_type: string;
  attr_value: string;
  rank: number;
  heat_t: number;
  pred_share_t1: number | null;
  pred_target_growth: number | null;
  is_trend_eligible_t: boolean;
}
```

- [ ] **Step 4: Add routes**

Create routes:

- `/`
- `/attributes/:attrId`
- `/graph/articles/:articleId?`
- `/recommendations`
- `/recommendations/:caseId/:articleId?`

- [ ] **Step 5: Add app shell**

Use a dense, presentation-focused layout with top navigation, clear active state, and content widths that work on desktop and mobile. Avoid marketing hero sections and nested cards.

- [ ] **Step 6: Run focused verification**

Run:

```sh
cd apps/defense_app/frontend
npm install
npm run typecheck
```

Expected: dependencies install and TypeScript passes.

Review gate: inspect frontend scaffold diff for root-level package files, one-note palettes, layout overlap risks, and missing API type fields.

## Task 6: Trend Dashboard, Attribute Detail, and Attribute Graph Pages

**Files:**
- Create: `apps/defense_app/frontend/src/views/TrendDashboardView.vue`
- Create: `apps/defense_app/frontend/src/views/AttributeDetailView.vue`
- Create: `apps/defense_app/frontend/src/views/AttributeGraphView.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendMetricStrip.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendBoard.vue`
- Create: `apps/defense_app/frontend/src/components/attribute/AttributeHeatChart.vue`
- Create: `apps/defense_app/frontend/src/components/graph/ArticleAttributeGraph.vue`

- [ ] **Step 1: Implement dashboard**

Render:

- Metric strip from `/api/metrics/summary`.
- Four trend columns for `colour_group_name`, `product_type_name`, `graphical_appearance_name`, and `garment_group_name`.
- Top-K control with stable dimensions.
- Attribute links to `/attributes/:attrId`.

- [ ] **Step 2: Implement attribute detail**

Render:

- Attribute metadata from `/api/attributes/:attrId`.
- ECharts line chart for 8 week heat and growth series.
- Associated article summaries.
- Parent/child attribute list.

- [ ] **Step 3: Implement graph page**

Render:

- Product search.
- Article-centered attribute graph with custom SVG or CSS layout.
- Attribute groups and relation labels.
- Representative article list for searched or selected attribute.

- [ ] **Step 4: Add empty and loading states**

Add states for:

- No trend attributes.
- No heat series.
- No article search results.
- No graph edges.

- [ ] **Step 5: Run focused verification**

Run:

```sh
cd apps/defense_app/frontend
npm run typecheck
npm run build
```

Expected: frontend typecheck and production build pass.

Review gate: run a browser smoke check against the local backend if dependencies are available; otherwise document the skipped browser check and keep the task open until it can be verified.

## Task 7: Recommendation Pages and Explanation Flow

**Files:**
- Create: `apps/defense_app/frontend/src/views/RecommendationView.vue`
- Create: `apps/defense_app/frontend/src/views/UserProfileReasonView.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/DemoUserList.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/RecommendationGrid.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/ScoreBreakdown.vue`

- [ ] **Step 1: Implement recommendation selection page**

Render:

- User search by `customer_id`, `case_id`, and tags.
- Tag filter.
- Demo user list sorted by hit count.
- Top-12 recommendation cards with rank, article id, title, core attributes, score, and hit marker.

- [ ] **Step 2: Implement explanation page**

Render:

- User profile attributes with preference score, purchase count, and last purchase week.
- Selected article display fields.
- Product attributes.
- Matching trend attributes.
- Score breakdown for `pop_score`, `sim_score`, `trend_score`, `recent_score`, and `final_score`.

- [ ] **Step 3: Preserve app wording boundary**

All UI copy must describe the system as:

- local defense demo
- offline trend prediction
- lightweight Top-N recommendation experiment
- recommendation explanation

Do not describe it as:

- online recommendation service
- production recommendation platform
- real-time personalization engine
- vector retrieval or deep recommendation system

- [ ] **Step 4: Run focused verification**

Run:

```sh
cd apps/defense_app/frontend
npm run typecheck
npm run build
```

Expected: frontend typecheck and build pass.

Review gate: inspect recommendation UI for duplicate cards, missing hit marker, missing score components, and unclear empty states.

## Task 8: Documentation and End-to-End Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/gpt-research/project-status-summary.md`
- Modify: `apps/defense_app/README.md`

- [ ] **Step 1: Document build and run commands**

Document:

```sh
uv run python src/18_build_defense_app_db.py
uv run --group app uvicorn app.main:app --reload --app-dir apps/defense_app/backend
cd apps/defense_app/frontend
npm install
npm run dev
```

Also document:

- `outputs/defense_app/fashion_demo.sqlite` is generated and not committed.
- Backend only reads SQLite.
- Frontend only calls FastAPI.
- App depends on stable upstream artifacts.

- [ ] **Step 2: Run Python verification**

Run:

```sh
uv run pytest tests/test_presentation_*.py tests/test_architecture_boundaries.py
uv run --group app pytest apps/defense_app/backend/tests
uv run python -m compileall -q src
uv run --group app python -m compileall -q apps/defense_app/backend
```

Expected: all pass.

- [ ] **Step 3: Run database build verification**

Run:

```sh
uv run python src/18_build_defense_app_db.py
```

Expected:

- `outputs/defense_app/fashion_demo.sqlite` exists.
- SQLite contains all required tables.
- `demo_users` has at least 3 rows.
- Each demo case has 12 recommendation rows.
- `trend_attributes` contains the four core attr types.

- [ ] **Step 4: Run frontend verification**

Run:

```sh
cd apps/defense_app/frontend
npm run typecheck
npm run build
```

Expected: both pass.

- [ ] **Step 5: Browser smoke**

Start backend and frontend dev servers. Verify:

- `/docs` loads and lists all `/api` endpoints.
- `/` shows trend dashboard and the four attr groups.
- `/attributes/:attrId` shows a heat chart.
- `/graph/articles/:articleId` shows article attributes.
- `/recommendations` shows demo users and Top-12 items.
- `/recommendations/:caseId/:articleId` shows profile, item attributes, matching trend attrs, and score breakdown.

- [ ] **Step 6: Final diff review**

Run:

```sh
git status --short
git diff --check
git diff --stat
```

Confirm no generated SQLite, `node_modules`, frontend `dist`, caches, credentials, raw data, or unrelated files are staged or left as intended source changes.

## Completion Audit Checklist

Before declaring the objective complete, verify each item with concrete evidence:

- Design plan exists: `docs/superpowers/plans/2026-05-12-defense-app.md`.
- SQLite build entrypoint exists and runs: `src/18_build_defense_app_db.py`.
- SQLite artifact builds: `outputs/defense_app/fashion_demo.sqlite`.
- Presentation layer is under `src/fashion_trend/presentation/`.
- Backend app is under `apps/defense_app/backend/`.
- Frontend app is under `apps/defense_app/frontend/`.
- FastAPI `/docs` exposes the intended API surface.
- Vue routes cover the five required pages.
- Trend dashboard displays four required core attr groups.
- Attribute detail displays 8-week series and prediction values.
- Attribute graph displays article-to-attribute connections.
- Recommendation page displays searchable/selectable demo users and Top-12 recommendations.
- Explanation page displays user profile, item attrs, trend attrs, and score components.
- Backend reads SQLite only after startup; frontend reads API only.
- No raw `transactions_train.csv`, raw `customers.csv`, training run directories, or historical `recommendation_items.csv` are used.
- Tests and verification commands above pass or any skipped command is explicitly explained with reason.
