# Defense App Visual Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the current SQLite-backed FastAPI + Vue defense demo app so its information architecture, desktop shell, page density, visual language, and verification flow closely match `docs/superpowers/specs/2026-05-13-defense-app-visual-design-contract.md`.

**Architecture:** Keep the existing boundary: stable artifacts are transformed into `outputs/defense_app/fashion_demo.sqlite`, FastAPI remains a thin read-only query layer, and Vue renders the full desktop presentation UI through `/api`. Add only display-oriented API/data refinements needed by the visual contract; do not introduce online recommendation, model training, reranking, graph databases, or mobile-first behavior.

**Tech Stack:** Python 3.10-3.12, pandas, sqlite3, FastAPI, pytest, Vue 3, TypeScript, Vite, Vue Router, ECharts/vue-echarts, lucide-vue-next, browser visual QA.

---

## Current Baseline

The current app already has the correct broad stack and a working MVP:

- `src/fashion_trend/presentation/` builds a read-only SQLite display database.
- `outputs/defense_app/fashion_demo.sqlite` exists and contains the visual-design core tables.
- `apps/defense_app/backend/` exposes read-only FastAPI endpoints over SQLite.
- `apps/defense_app/frontend/` renders the five required routes.
- The current UI is still closer to a functional prototype than the accepted visual contract: top navigation instead of left shell, incomplete trend evidence area, relation list instead of tree, graph page missing the right inspector density, recommendation pages not yet at the reference-screen hierarchy, and no browser screenshot fidelity ledger.

Do not revert or overwrite current uncommitted implementation changes. Work with the existing files and review `git diff` at every task boundary.

## Execution Rules

- Execute serially. Do not let multiple workers edit overlapping frontend files in parallel.
- Do not commit unless the user explicitly asks for commits.
- Do not submit generated SQLite files, `node_modules`, `dist`, cache files, screenshots, or temporary QA artifacts unless the user explicitly asks to keep them.
- Preserve the app positioning: local read-only research demo, not online recommendation, not production platform, not training UI.
- Desktop only: target `1440 x 900`, minimum useful width `1280px`; remove mobile-nav assumptions from the visual implementation path.
- Every task ends with a diff review and at least one focused verification command.

## File Responsibility Map

### Data and API

- Modify: `src/fashion_trend/presentation/builders.py`
  - Ensure the presentation DB includes source weeks needed by the default trend view and demo recommendation cases.
  - Keep id columns as strings and retain existing fail-fast checks.
- Modify: `tests/test_presentation_builders.py`
  - Cover multi-week trend extraction and heat-series coverage for the selected weeks.
- Modify: `apps/defense_app/backend/app/repositories/trend_repository.py`
  - Add display-focused query helpers for source weeks, grouped trend summary, score distribution, top attribute history, new high-potential attributes, and detail rows.
- Modify: `apps/defense_app/backend/app/api/routes/trends.py`
  - Add endpoints used by the visual trend dashboard.
- Modify: `apps/defense_app/backend/app/schemas/trends.py`
  - Add Pydantic response models for the new trend display endpoints.
- Modify: `apps/defense_app/backend/tests/test_api_trends.py`
  - Cover the new endpoints and source-week behavior.
- Modify: `apps/defense_app/backend/app/repositories/attribute_repository.py`
  - Add helper methods only if the attribute tree or representative graph entry needs richer data than current endpoints return.
- Modify: `apps/defense_app/backend/tests/test_api_attributes.py`
  - Cover any attribute graph/tree response changes.
- Modify: `apps/defense_app/backend/app/repositories/metrics_repository.py`
  - Reuse existing metrics for display tiles; add no synthetic metrics.
- Modify: `apps/defense_app/backend/tests/test_api_metrics.py`
  - Cover any status or summary refinements.

### Shared Frontend System

- Modify: `apps/defense_app/frontend/src/styles/main.css`
  - Replace prototype CSS with visual-contract tokens, shell layout, desktop grid rules, panel primitives, table primitives, chart sizing, graph canvas styles, and recommendation card styles.
- Modify: `apps/defense_app/frontend/src/components/layout/AppShell.vue`
  - Implement fixed left navigation, brand block, bottom system status, and main content area.
- Create: `apps/defense_app/frontend/src/components/layout/PageToolbar.vue`
  - Shared page toolbar with title, context, actions, and status slots.
- Create: `apps/defense_app/frontend/src/components/ui/Panel.vue`
  - Shared panel wrapper with optional heading and actions.
- Create: `apps/defense_app/frontend/src/components/ui/MetricTile.vue`
  - Shared metric tile matching the visual contract.
- Create: `apps/defense_app/frontend/src/components/ui/DataTable.vue`
  - Shared dense table shell for detail rows, article rows, and trend rows.
- Create: `apps/defense_app/frontend/src/components/ui/StatusBlock.vue`
  - Shared loading, empty, and error states.
- Create: `apps/defense_app/frontend/src/utils/formatters.ts`
  - Percent, score, integer, week, and JSON tag formatting helpers.
- Modify: `apps/defense_app/frontend/src/api/defenseApi.ts`
  - Add API wrappers for new display endpoints.
- Modify: `apps/defense_app/frontend/src/types/api.ts`
  - Add TypeScript interfaces for new display payloads.

### Page Features

- Modify: `apps/defense_app/frontend/src/views/TrendDashboardView.vue`
- Modify: `apps/defense_app/frontend/src/components/trend/TrendMetricStrip.vue`
- Modify: `apps/defense_app/frontend/src/components/trend/TrendBoard.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendRankMatrix.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendEvidenceArea.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendDetailTable.vue`
- Modify: `apps/defense_app/frontend/src/views/AttributeDetailView.vue`
- Modify: `apps/defense_app/frontend/src/components/attribute/AttributeHeatChart.vue`
- Create: `apps/defense_app/frontend/src/components/attribute/AttributeSummaryBand.vue`
- Create: `apps/defense_app/frontend/src/components/attribute/AttributeRelationTree.vue`
- Create: `apps/defense_app/frontend/src/components/attribute/AttributeWeekDetailTable.vue`
- Modify: `apps/defense_app/frontend/src/views/AttributeGraphView.vue`
- Modify: `apps/defense_app/frontend/src/components/graph/ArticleAttributeGraph.vue`
- Create: `apps/defense_app/frontend/src/components/graph/GraphCanvas.vue`
- Create: `apps/defense_app/frontend/src/components/graph/GraphNodeInspector.vue`
- Create: `apps/defense_app/frontend/src/components/graph/GraphEdgeTable.vue`
- Modify: `apps/defense_app/frontend/src/views/RecommendationView.vue`
- Modify: `apps/defense_app/frontend/src/components/recommendation/DemoUserList.vue`
- Modify: `apps/defense_app/frontend/src/components/recommendation/RecommendationGrid.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/RecommendationMetricsStrip.vue`
- Modify: `apps/defense_app/frontend/src/views/UserProfileReasonView.vue`
- Modify: `apps/defense_app/frontend/src/components/recommendation/ScoreBreakdown.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/PreferenceMatchMatrix.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/Top12ContextStrip.vue`

### Verification and Docs

- Modify: `apps/defense_app/README.md`
  - Add visual QA runbook after implementation is verified.
- Optional create during QA only, then remove before final handoff unless the user asks to keep it: `apps/defense_app/frontend/tmp-visual-qa/`

## Task 0: Baseline Snapshot and Guardrails

**Files:**
- Read-only: repository state, current visual contract, existing defense app files.

- [x] **Step 1: Capture current dirty state**

Run:

```sh
git status --short apps/defense_app src/fashion_trend/presentation src/18_build_defense_app_db.py docs/superpowers/specs/2026-05-13-defense-app-visual-design-contract.md
```

Expected: current uncommitted implementation work is visible. Treat it as user or prior work unless you made it in the active task.

- [x] **Step 2: Confirm SQLite baseline**

Run:

```sh
sqlite3 outputs/defense_app/fashion_demo.sqlite "pragma integrity_check;"
sqlite3 -header -column outputs/defense_app/fashion_demo.sqlite "select name from sqlite_master where type='table' order by name;"
sqlite3 -header -column outputs/defense_app/fashion_demo.sqlite "select attr_type, count(*) as trend_count from trend_attributes group by attr_type order by attr_type;"
sqlite3 -header -column outputs/defense_app/fashion_demo.sqlite "select case_id, count(*) as top12_count, sum(is_hit) as hit_count from recommendation_items group by case_id order by case_id;"
```

Expected:

- `integrity_check` returns `ok`.
- Required visual tables exist.
- Core trend types have rows.
- Every demo case has `top12_count = 12`.

- [x] **Step 3: Confirm current tests before refactor**

Run:

```sh
uv run --group app pytest apps/defense_app/backend/tests
uv run pytest tests/test_presentation_builders.py tests/test_presentation_runner.py tests/test_architecture_boundaries.py
cd apps/defense_app/frontend && npm run typecheck
cd apps/defense_app/frontend && npm run build
```

Expected: commands pass. If a command fails because of local sandbox/cache access, rerun with approved escalation rather than changing code to dodge the environment.

- [x] **Step 4: Record visual contract acceptance targets**

Use these concept files as the source of truth during browser QA:

```text
docs/superpowers/assets/defense-app-global-system-preview.png
docs/superpowers/assets/defense-app-trend-dashboard-preview.png
docs/superpowers/assets/defense-app-attribute-detail-preview.png
docs/superpowers/assets/defense-app-attribute-graph-preview.png
docs/superpowers/assets/defense-app-recommendation-preview.png
docs/superpowers/assets/defense-app-recommendation-reason-preview.png
```

Review gate: no code changes in this task.

## Task 1: Data and API Refinements for Visual Pages

**Files:**
- Modify: `src/fashion_trend/presentation/builders.py`
- Modify: `tests/test_presentation_builders.py`
- Modify: `apps/defense_app/backend/app/repositories/trend_repository.py`
- Modify: `apps/defense_app/backend/app/api/routes/trends.py`
- Modify: `apps/defense_app/backend/app/schemas/trends.py`
- Modify: `apps/defense_app/backend/tests/test_api_trends.py`
- Modify if needed: `apps/defense_app/backend/app/repositories/metrics_repository.py`
- Modify if needed: `apps/defense_app/backend/tests/test_api_metrics.py`

- [x] **Step 1: Add failing builder coverage for visual source weeks**

Add tests proving that `build_presentation_tables()` includes:

- the latest default trend source week;
- each demo case `cutoff_week` when trend predictions exist for that week;
- heat-series rows for attributes selected into those source weeks;
- no duplicate `trend_attributes` primary-key combinations.

Run:

```sh
uv run pytest tests/test_presentation_builders.py -k "trend or heat"
```

Expected before implementation: at least one new assertion fails because only the default source week is represented.

- [x] **Step 2: Implement multi-week trend table build**

Update `build_trend_attributes()` or add a helper that builds per-week trend rows for:

- the default source week from current logic;
- every `cutoff_week` in `report_cases` that exists in `prediction_sample_view`;
- each selected week limited to `limit_per_type=50`.

Keep ordering stable:

```text
source_week ASC, attr_type ASC, rank ASC
```

Do not add fallback rows for missing weeks. Missing source weeks should be recorded as metadata warnings or omitted with explicit tests.

- [x] **Step 3: Add trend display repository methods**

Add methods to `TrendRepository`:

```python
available_source_weeks() -> list[int]
summary(source_week: int | None, limit: int) -> dict[str, object]
score_distribution(source_week: int | None) -> list[dict[str, object]]
top_history(source_week: int | None, limit: int) -> list[dict[str, object]]
new_high_potential(source_week: int | None, limit: int) -> list[dict[str, object]]
detail_rows(source_week: int | None, attr_type: str | None, limit: int) -> list[dict[str, object]]
```

The methods should only read `trend_attributes` and `attribute_heat_series`. Do not read raw artifacts from the backend.

- [x] **Step 4: Add trend display schemas and endpoints**

Add endpoints:

```text
GET /api/trends/source-weeks
GET /api/trends/summary?source_week=&limit=
GET /api/trends/evidence?source_week=&limit=
GET /api/trends/detail?source_week=&attr_type=&limit=
```

Response payloads should be compact and display-oriented. Include only values the frontend needs for the visual contract:

- available source weeks;
- target week;
- rising attribute count;
- high-confidence count using a documented threshold;
- Top-K average predicted growth;
- covered article count when it can be computed from `article_attributes`;
- distribution buckets;
- Top-1 history points;
- new high-potential rows;
- dense detail table rows.

Do not invent model metrics if the DB lacks them. Missing optional values should be `null`.

- [x] **Step 5: Add backend endpoint tests**

Cover:

- default source week selection;
- explicit `source_week`;
- invalid `limit` boundaries;
- grouped summary uses existing rows only;
- `source-weeks` returns sorted integers;
- evidence endpoint returns stable keys even if a section is empty.

Run:

```sh
uv run --group app pytest apps/defense_app/backend/tests/test_api_trends.py
uv run --group app pytest apps/defense_app/backend/tests/test_api_metrics.py
uv run pytest tests/test_presentation_builders.py
```

Expected: all pass.

- [x] **Step 6: Rebuild and inspect SQLite**

Run:

```sh
uv run python src/18_build_defense_app_db.py
sqlite3 outputs/defense_app/fashion_demo.sqlite "pragma integrity_check;"
sqlite3 -header -column outputs/defense_app/fashion_demo.sqlite "select source_week, attr_type, count(*) from trend_attributes group by source_week, attr_type order by source_week, attr_type;"
```

Expected: DB integrity is `ok`; trend rows include the default week and available demo cutoff weeks where source predictions exist.

Review gate: backend and presentation diffs only. No frontend visual changes in this task.

## Task 2: Shared Desktop Shell and Visual System

**Files:**
- Modify: `apps/defense_app/frontend/src/components/layout/AppShell.vue`
- Create: `apps/defense_app/frontend/src/components/layout/PageToolbar.vue`
- Create: `apps/defense_app/frontend/src/components/ui/Panel.vue`
- Create: `apps/defense_app/frontend/src/components/ui/MetricTile.vue`
- Create: `apps/defense_app/frontend/src/components/ui/DataTable.vue`
- Create: `apps/defense_app/frontend/src/components/ui/StatusBlock.vue`
- Create: `apps/defense_app/frontend/src/utils/formatters.ts`
- Modify: `apps/defense_app/frontend/src/styles/main.css`
- Modify: `apps/defense_app/frontend/src/router/index.ts`

- [x] **Step 1: Implement visual tokens**

Replace prototype tokens with the visual-contract token names:

```css
--app-bg: #F6F7F5;
--surface: #FFFFFF;
--surface-muted: #EEF1ED;
--text: #1B1D1F;
--muted: #66706A;
--line: #D8DED7;
--trend-red: #B42318;
--trend-red-soft: #FCE9E6;
--signal-green: #0F766E;
--rank-gold: #9A6700;
--graph-blue: #3A5A6A;
```

Set the font stack to:

```css
Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif
```

Keep `letter-spacing: 0`.

- [x] **Step 2: Implement fixed desktop AppShell**

`AppShell.vue` should render:

- left sidebar width `220px`;
- brand `Fashion Trend Lab`;
- subtitle `Defense Demo`;
- nav items in this order: 趋势看板, 属性详情, 属性图展示, 推荐展示, 推荐理由;
- bottom statuses: `SQLite ready`, `LightGBM stable`, `pop_similarity_trend`;
- main workspace with no top sticky marketing-style header.

Use lucide icons at `18px`. Active nav uses a `3px` red left rail.

- [x] **Step 3: Add shared PageToolbar**

`PageToolbar.vue` should accept:

```ts
defineProps<{
  title: string;
  context?: string;
  status?: string;
}>();
```

Use slots:

```text
actions
```

The toolbar height should be `64px` and should not introduce global search.

- [x] **Step 4: Add UI primitives**

Add `Panel`, `MetricTile`, `DataTable`, and `StatusBlock` so page implementations stop copying one-off panel/status/table markup. Keep these components simple and slot-based.

- [x] **Step 5: Remove mobile-first collapse from the app path**

Set a desktop minimum shell width near `1280px`. Remove hamburger/mobile navigation assumptions and avoid media queries that turn the accepted desktop layout into a different product. If a narrow viewport is opened, horizontal scrolling is acceptable for this defense demo.

- [x] **Step 6: Verify frontend types and build**

Run:

```sh
cd apps/defense_app/frontend && npm run typecheck
cd apps/defense_app/frontend && npm run build
```

Expected: both pass.

Review gate: open the diff for `AppShell.vue`, new shared components, router labels, and `main.css`. Confirm no page-specific behavior was changed except what is required to compile against the new shell primitives.

## Task 3: Trend Dashboard Visual Refactor

**Files:**
- Modify: `apps/defense_app/frontend/src/views/TrendDashboardView.vue`
- Modify: `apps/defense_app/frontend/src/components/trend/TrendMetricStrip.vue`
- Modify: `apps/defense_app/frontend/src/components/trend/TrendBoard.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendRankMatrix.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendEvidenceArea.vue`
- Create: `apps/defense_app/frontend/src/components/trend/TrendDetailTable.vue`
- Modify: `apps/defense_app/frontend/src/api/defenseApi.ts`
- Modify: `apps/defense_app/frontend/src/types/api.ts`

- [x] **Step 1: Wire trend display APIs**

Add API client functions:

```ts
listTrendSourceWeeks()
getTrendSummary(params)
getTrendEvidence(params)
getTrendDetail(params)
```

Add matching TypeScript interfaces. Keep nullable fields nullable; do not coerce missing API values to `0`.

- [x] **Step 2: Build PageToolbar actions**

Trend toolbar must show:

- title `趋势看板`;
- context `预测第 {target_week} 周上升趋势`;
- source-week select;
- Top-K select with `5 / 10 / 20 / 50`;
- attr-type filter;
- refresh button.

Changing source week or Top-K updates data without changing page structure.

- [x] **Step 3: Build MetricStrip from trend summary**

Display 4-5 tiles:

- 上升属性总数;
- 高置信属性数;
- Top-K 平均预测分;
- 覆盖商品数;
- 模型/稳定 artifact 状态.

If an optional value is missing, display `--`, not a fabricated number.

- [x] **Step 4: Build four-column TrendRankMatrix**

Fixed order:

```text
颜色
品类
图案
服装组
```

Each row shows rank, attribute name, weak attr id/type, predicted growth or score, and a horizontal intensity bar. Row click goes to `/attributes/:attrId` with the current `source_week`.

- [x] **Step 5: Build lower evidence area**

`TrendEvidenceArea` should include:

- distribution chart;
- Top-1 trend history chart;
- new high-potential attributes table.

Use ECharts for charts and keep a compact table for new attributes. Empty subsections should use `StatusBlock`, not blank panels.

- [x] **Step 6: Build dense trend detail table**

`TrendDetailTable` should show source week, target week, attr type, attr value, rank, heat, predicted share, predicted growth, and action links.

- [x] **Step 7: Verify**

Run:

```sh
uv run --group app pytest apps/defense_app/backend/tests/test_api_trends.py
cd apps/defense_app/frontend && npm run typecheck
cd apps/defense_app/frontend && npm run build
```

Expected: all pass.

Review gate: compare the trend page against `docs/superpowers/assets/defense-app-trend-dashboard-preview.png` for shell, first-screen density, four-column rank matrix, evidence area, and absence of marketing hero layout.

## Task 4: Attribute Detail Visual Refactor

**Files:**
- Modify: `apps/defense_app/frontend/src/views/AttributeDetailView.vue`
- Modify: `apps/defense_app/frontend/src/components/attribute/AttributeHeatChart.vue`
- Create: `apps/defense_app/frontend/src/components/attribute/AttributeSummaryBand.vue`
- Create: `apps/defense_app/frontend/src/components/attribute/AttributeRelationTree.vue`
- Create: `apps/defense_app/frontend/src/components/attribute/AttributeWeekDetailTable.vue`
- Modify if needed: `apps/defense_app/backend/app/repositories/attribute_repository.py`
- Modify if needed: `apps/defense_app/backend/tests/test_api_attributes.py`

- [x] **Step 1: Rebuild page structure**

Use:

```text
PageToolbar
AttributeSummaryBand
AttributeHeatEvidence
RelatedEvidence
WeekDetailTable
```

The chart is the main evidence block. Do not reintroduce a separate `预测证据` panel.

- [x] **Step 2: Implement AttributeSummaryBand**

Show:

- rank;
- current heat;
- predicted growth;
- predicted share;
- trend candidate state.

Missing values render as `--`.

- [x] **Step 3: Refine heat chart**

Keep lines for:

- `heat`;
- `pred_target_growth`;
- `actual_target_growth` when available.

Mark the current prediction window visually with a subtle vertical band or marker. Tooltip includes `week_id`, `heat`, `pred_target_growth`, `actual_target_growth`, and `pred_share_t1`.

- [x] **Step 4: Replace relation list with tree**

`AttributeRelationTree` builds a lightweight tree from `/api/attributes/{attr_id}/graph`:

- current node: soft red fill and red border;
- parent nodes: blue-gray emphasis;
- child/sibling nodes: green or neutral emphasis;
- thin gray connector lines;
- no fake nodes for missing hierarchy levels.

Clicking a node opens that attribute detail route.

- [x] **Step 5: Add week detail table**

Rows come from `heatSeries.points`. Columns:

```text
week_id, heat, pred_target_growth, actual_target_growth, pred_share_t1, direction_pred, direction_actual, direction_match
```

Direction fields are frontend-derived from numeric signs. If actual growth is missing, show `--`.

- [x] **Step 6: Verify**

Run:

```sh
uv run --group app pytest apps/defense_app/backend/tests/test_api_attributes.py
cd apps/defense_app/frontend && npm run typecheck
cd apps/defense_app/frontend && npm run build
```

Expected: all pass.

Review gate: compare against `docs/superpowers/assets/defense-app-attribute-detail-preview.png`; verify the relation area is a tree, not a text edge list.

## Task 5: Attribute Graph Workspace Refactor

**Files:**
- Modify: `apps/defense_app/frontend/src/views/AttributeGraphView.vue`
- Modify: `apps/defense_app/frontend/src/components/graph/ArticleAttributeGraph.vue`
- Create: `apps/defense_app/frontend/src/components/graph/GraphCanvas.vue`
- Create: `apps/defense_app/frontend/src/components/graph/GraphNodeInspector.vue`
- Create: `apps/defense_app/frontend/src/components/graph/GraphEdgeTable.vue`
- Modify: `apps/defense_app/frontend/src/router/index.ts`

- [x] **Step 1: Support article and attribute entry contexts**

Keep `/graph/articles/:articleId?`. Also allow query entry with `attr_id`:

```text
/graph/articles?attr_id={attrId}
```

When `attr_id` is provided and no article is selected, load representative articles through the existing attribute articles endpoint and select the first returned article. If none exists, show an empty state that explains no representative article is available.

- [x] **Step 2: Build three-zone workspace**

Use the visual contract structure:

```text
left: search and selected article summary
center: graph canvas
right: node inspector, attribute groups, relation summary
bottom: edge table
```

Keep the graph canvas un-nested and visually treated as a workspace, not a decorative card inside another card.

- [x] **Step 3: Refine graph layout**

Group nodes by attribute type around the article. Use stable positions so the graph does not reshuffle on hover or inspector selection. Make labels readable at `1440 x 900`.

- [x] **Step 4: Add node inspector**

Clicking a node selects it and updates the right inspector:

- node label;
- node type;
- relation count;
- navigation action for attributes.

Do not add graph editing, save, drag persistence, or Neo4j-style controls.

- [x] **Step 5: Add edge table**

`GraphEdgeTable` shows source, target, relation type, and grouped attribute type. It is a supporting evidence table below the canvas.

- [x] **Step 6: Verify**

Run:

```sh
uv run --group app pytest apps/defense_app/backend/tests/test_api_articles.py apps/defense_app/backend/tests/test_api_attributes.py
cd apps/defense_app/frontend && npm run typecheck
cd apps/defense_app/frontend && npm run build
```

Expected: all pass.

Review gate: compare against `docs/superpowers/assets/defense-app-attribute-graph-preview.png`; verify the page reads as a graph workspace, not a search page plus rough SVG prototype.

## Task 6: Recommendation Display Refactor

**Files:**
- Modify: `apps/defense_app/frontend/src/views/RecommendationView.vue`
- Modify: `apps/defense_app/frontend/src/components/recommendation/DemoUserList.vue`
- Modify: `apps/defense_app/frontend/src/components/recommendation/RecommendationGrid.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/RecommendationMetricsStrip.vue`
- Modify: `apps/defense_app/frontend/src/api/defenseApi.ts`
- Modify: `apps/defense_app/frontend/src/types/api.ts`

- [x] **Step 1: Rebuild recommendation workspace**

Use:

```text
PageToolbar
RecommendationWorkspace
RecommendationEvidence
```

Left side contains demo user search, tag filtering, and selected user summary. Right side contains Top-12 recommendations.

- [x] **Step 2: Keep demo users controlled**

Use the existing `/api/demo-users` flow. Do not add arbitrary online recommendation calculation. Searching filters the preset case pool only.

- [x] **Step 3: Build Top-12 grid**

Use a 4x3 recommendation grid at desktop width. Each card shows:

- rank;
- article id or product name;
- score;
- hit/miss;
- core attributes;
- candidate sources as compact chips or short text.

Card click opens `/recommendations/:caseId/:articleId`.

- [x] **Step 4: Add recommendation metrics strip**

Use `/api/metrics/summary` or `/api/metrics/recommendation` to show:

- MAP@12;
- Recall@12;
- Hit Rate@12;
- Method `pop_similarity_trend`.

Only display metrics present in `metrics_summary`.

- [x] **Step 5: Add recommendation detail preview**

Below the grid, add a dense table preview for Top-12 rows. It helps answer questions without leaving the page.

- [x] **Step 6: Verify**

Run:

```sh
uv run --group app pytest apps/defense_app/backend/tests/test_api_demo_users.py apps/defense_app/backend/tests/test_api_recommendations.py apps/defense_app/backend/tests/test_api_metrics.py
cd apps/defense_app/frontend && npm run typecheck
cd apps/defense_app/frontend && npm run build
```

Expected: all pass.

Review gate: compare against `docs/superpowers/assets/defense-app-recommendation-preview.png`; verify the UI communicates preset demo cases and offline Top-12 output.

## Task 7: Recommendation Reason and User Profile Refactor

**Files:**
- Modify: `apps/defense_app/frontend/src/views/UserProfileReasonView.vue`
- Modify: `apps/defense_app/frontend/src/components/recommendation/ScoreBreakdown.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/PreferenceMatchMatrix.vue`
- Create: `apps/defense_app/frontend/src/components/recommendation/Top12ContextStrip.vue`
- Modify if needed: `apps/defense_app/backend/app/services/recommendation_explanation_service.py`
- Modify if needed: `apps/defense_app/backend/tests/test_api_recommendations.py`

- [x] **Step 1: Rebuild the three-column reason workspace**

Use:

```text
left: user profile attributes
middle: recommended item attributes
right: score breakdown
```

The top toolbar context should include `case_id`, selected `article_id`, and final score when available.

- [x] **Step 2: Build score breakdown bars**

Show:

```text
pop_score, sim_score, trend_score, recent_score, final_score
```

Use comparable horizontal bars. Do not show a hard-coded score formula unless the backend payload contains formula metadata.

- [x] **Step 3: Build preference match matrix**

Compute matches client-side from `user_profile` and `item_attributes`:

- exact `attr_id` match is a strong match;
- same `attr_type` but different value is a weak contextual relation;
- no matches show a clear empty state.

This matrix must not imply unavailable model internals.

- [x] **Step 4: Build matching trend attributes**

Use `matching_trend_attributes` from the explanation payload. If the array is empty, show an empty state and do not fabricate trend evidence.

- [x] **Step 5: Add Top-12 context strip**

Show the selected item within the case's Top-12 list, with rank, score, hit/miss, and quick navigation between items.

- [x] **Step 6: Verify**

Run:

```sh
uv run --group app pytest apps/defense_app/backend/tests/test_api_recommendations.py
cd apps/defense_app/frontend && npm run typecheck
cd apps/defense_app/frontend && npm run build
```

Expected: all pass.

Review gate: compare against `docs/superpowers/assets/defense-app-recommendation-reason-preview.png`; verify the page explains the selected recommendation without inventing online personalization.

## Task 8: Browser Fidelity QA and Final Hardening

**Files:**
- Modify: `apps/defense_app/README.md`
- Temporary only: browser screenshots in a temp directory outside tracked source, removed before final handoff unless user asks to keep them.

- [x] **Step 1: Run full backend and presentation verification**

Run:

```sh
uv run pytest tests/test_presentation_builders.py tests/test_presentation_runner.py tests/test_architecture_boundaries.py
uv run --group app pytest apps/defense_app/backend/tests
uv run python src/18_build_defense_app_db.py
sqlite3 outputs/defense_app/fashion_demo.sqlite "pragma integrity_check;"
```

Expected: all tests pass and SQLite integrity is `ok`.

- [x] **Step 2: Run full frontend verification**

Run:

```sh
cd apps/defense_app/frontend && npm run typecheck
cd apps/defense_app/frontend && npm run build
```

Expected: both pass.

- [x] **Step 3: Start local servers**

Backend:

```sh
uv run --group app uvicorn app.main:app --reload --app-dir apps/defense_app/backend
```

Frontend:

```sh
cd apps/defense_app/frontend && npm run dev
```

Use another port only if the default Vite port is occupied.

- [x] **Step 4: Browser walkthrough**

Use the in-app Browser first. Verify:

- `/` trend dashboard loads and source-week/Top-K controls update visible data;
- clicking a trend row opens attribute detail and preserves `source_week`;
- attribute relation tree node navigation works;
- graph page search selects an article and node inspector updates on click;
- recommendation page selects a demo user and shows Top-12;
- recommendation card click opens the reason page;
- reason page Top-12 context switches selected recommendation.

- [x] **Step 5: Visual fidelity ledger**

Compare implementation screenshots against the accepted concept images with `view_image`:

```text
global shell -> defense-app-global-system-preview.png
trend dashboard -> defense-app-trend-dashboard-preview.png
attribute detail -> defense-app-attribute-detail-preview.png
attribute graph -> defense-app-attribute-graph-preview.png
recommendation display -> defense-app-recommendation-preview.png
recommendation reason -> defense-app-recommendation-reason-preview.png
```

For each page record at least five checks:

- shell/nav placement;
- toolbar copy and control density;
- typography scale;
- color tokens;
- panel/grid proportions;
- chart/table readability;
- icon treatment;
- empty/error state treatment;
- navigation flow.

Fix any visible drift that would be a design-review comment.

- [x] **Step 6: Update README runbook**

Add a short visual QA section to `apps/defense_app/README.md` with:

- DB build command;
- backend command;
- frontend command;
- expected routes;
- verification commands.

Do not describe the app as online recommendation or production service.

- [x] **Step 7: Final diff review**

Run:

```sh
git diff -- apps/defense_app src/fashion_trend/presentation src/18_build_defense_app_db.py tests docs/superpowers/plans/2026-05-13-defense-app-visual-refactor.md
git status --short
```

Expected:

- source changes match the planned tasks;
- no SQLite DB, build output, `node_modules`, screenshots, cache files, or unrelated artifacts are staged or left as accidental new files;
- unrelated dirty files are identified and left untouched.

## Acceptance Criteria

The refactor is complete only when all of these are true:

- The app uses a fixed desktop left-shell layout with the visual-contract nav order and system status.
- All five routes match the accepted information architecture:
  - 趋势看板;
  - 属性详情;
  - 属性图展示;
  - 推荐展示;
  - 推荐理由.
- The trend page includes metrics, four-column trend ranking, evidence area, and detail table.
- The attribute detail page includes summary band, heat/growth chart, related article table, relation tree, and week detail table.
- The graph page reads as a graph workspace with search, canvas, node inspector, grouping summary, and edge table.
- The recommendation page uses controlled demo users and a Top-12 grid, not arbitrary online recommendation.
- The recommendation reason page shows user profile, item attributes, score bars, preference matching, trend matches, and Top-12 context.
- Missing values display as `--` or explicit empty states; no fake scores, fake metrics, or fabricated trend evidence.
- Backend tests, presentation tests, architecture tests, frontend typecheck, and frontend build pass.
- Browser QA confirms the accepted concept images and implementation are visually aligned at the desktop target.

## Suggested Task Boundaries for Later Commits

Only commit if the user explicitly asks. Suggested commit boundaries:

1. `feat(defense-app): 补齐视觉看板数据接口`
2. `feat(defense-app): 重构桌面 App Shell 和视觉系统`
3. `feat(defense-app): 重构趋势看板页面`
4. `feat(defense-app): 重构属性详情页面`
5. `feat(defense-app): 重构属性图展示页面`
6. `feat(defense-app): 重构推荐展示页面`
7. `feat(defense-app): 重构推荐理由页面`
8. `docs(defense-app): 补充视觉验收说明`

## Self-Review Checklist

- Visual contract scope is covered by Tasks 2-8.
- SQLite/FastAPI gaps needed by the visual contract are covered by Task 1.
- No task asks for online recommendation, training, reranking, PostgreSQL, Neo4j, or mobile-first implementation.
- Every task has file paths, focused steps, and verification commands.
- Execution can stop safely after any task with a reviewable diff.
