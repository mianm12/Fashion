# Recommendation Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-stage recommendation enhancement from `docs/superpowers/specs/2026-05-14-recommendation-enhancement-design.md`: `enhanced_default` candidates, `enhanced_pop_similarity_trend` method, and `recommendation_enhanced` experiment.

**Architecture:** Keep the existing `strategy -> method -> experiment` boundary. New input artifacts are created by the recommendation input stage, enhanced retrieval writes a separate `enhanced_default` candidate artifact, enhanced ranking uses strategy-scoped feature cache partitions, and the enhanced experiment writes only under `outputs/recommendation/experiments/recommendation_enhanced/`.

**Tech Stack:** Python, pandas, numpy, pyarrow, pytest, existing project artifact/freshness helpers. No new runtime dependency is allowed in this phase.

---

## Execution Rules

- Work on one task at a time. After each task: inspect `git diff`, run the task validation commands, run required architecture or compile checks, then commit only that task.
- Use the repository style for commit messages, for example `feat(recommendation): 构建增强候选输入`.
- Do not commit `data/` or `outputs/` generated artifacts. Tests must use in-memory frames or `tmp_path`.
- Do not run or overwrite `outputs/recommendation/experiments/main/experiment.json` or `outputs/recommendation/pop_similarity_trend/` as part of default enhanced experiment work.
- If `uv` fails because of cache permissions, rerun the same command with `UV_CACHE_DIR=/private/tmp/uv-cache`.
- If a task exposes a conflict between this plan, the spec, and current implementation contracts, stop and report the conflict before continuing.

## Planned File Map

- `src/fashion_trend/recommendation/contracts.py`: add new strategy, method, score, candidate, customer profile, product map, source, and diagnostic constants.
- `src/fashion_trend/recommendation/paths.py`: add `CUSTOMER_PROFILE_PATH`, `ARTICLE_PRODUCT_MAP_PATH`, and reuse existing strategy/method/experiment path helpers.
- `src/fashion_trend/recommendation/readers.py`: strict readers for new artifacts and enhanced candidate/method contracts.
- `src/fashion_trend/recommendation/inputs.py`: build and write `customer_profile.parquet` and `article_product_map.parquet` with metadata.
- `src/12_build_recommendation_inputs.py`: pass `customers.csv` and `articles_clean.csv` into the input builder.
- `src/13_build_recommend_candidates.py`: support `enhanced_default` inputs without changing existing default strategy behavior.
- `src/14_rerank_recommendations.py`: allow `enhanced_pop_similarity_trend` and read its enhanced candidate/cache inputs.
- `src/16_run_recommendation_experiment.py`: dispatch `recommendation_enhanced` without changing `main`.
- `src/fashion_trend/recommendation/retrieval/reorder.py`: reorder source.
- `src/fashion_trend/recommendation/retrieval/product_variants.py`: product-code variant source.
- `src/fashion_trend/recommendation/retrieval/customer_segments.py`: age-bucket popularity source.
- `src/fashion_trend/recommendation/retrieval/preference_popularity.py`: preference attribute popularity source.
- `src/fashion_trend/recommendation/retrieval/candidates.py`: source order, enhanced merge, source metadata, and candidate writer metadata.
- `src/fashion_trend/recommendation/features/cache.py`: enhanced score partitions, source-level seen flags, freshness metadata.
- `src/fashion_trend/recommendation/ranking/features.py`: raw enhanced score builders and group normalization.
- `src/fashion_trend/recommendation/ranking/filters.py`: source-level seen filter helper.
- `src/fashion_trend/recommendation/ranking/scoring.py`: existing weighted score path should work after score constants are expanded.
- `src/fashion_trend/recommendation/ranking/weights.py`: validate the expanded enhanced weight vector.
- `src/fashion_trend/recommendation/methods/trend_aware/enhanced_pop_similarity_trend.py`: enhanced method class.
- `src/fashion_trend/recommendation/methods/trend_aware/__init__.py` and `src/fashion_trend/recommendation/registry.py`: method registration.
- `src/fashion_trend/recommendation/experiments/enhanced_grid_search.py`: bounded enhanced weight grid and valid selection.
- `src/fashion_trend/recommendation/experiments/enhanced_diagnostics.py`: candidate recall and source contribution diagnostics.
- `src/fashion_trend/recommendation/experiments/enhanced_runner.py`: enhanced experiment orchestration.
- `src/fashion_trend/recommendation/experiments/runner.py`: thin dispatch to preserve `main`.
- Tests: update existing `tests/test_recommendation_*.py` files; add focused tests only when an existing file becomes too broad.
- Docs: update `README.md`, `AGENTS.md`, and `docs/gpt-research/implementation-plan.md` only after implementation behavior is verified.

---

### Task 1: Recommendation Input Artifacts

**Files:**
- Modify: `src/fashion_trend/recommendation/contracts.py`
- Modify: `src/fashion_trend/recommendation/paths.py`
- Modify: `src/fashion_trend/recommendation/readers.py`
- Modify: `src/fashion_trend/recommendation/inputs.py`
- Modify: `src/12_build_recommendation_inputs.py`
- Test: `tests/test_recommendation_inputs.py`
- Test: `tests/test_recommendation_contracts.py`

- [ ] **Step 1: Add failing tests for customer profile and product map contracts**

  Add tests named:
  - `test_customer_profile_buckets_age_and_preserves_unknowns`
  - `test_customer_profile_rejects_duplicate_customer_id`
  - `test_article_product_map_preserves_string_ids_and_rejects_missing_product_code`
  - `test_build_and_write_inputs_records_customer_and_product_artifacts`
  - `test_read_customer_profile_rejects_invalid_age_bucket`
  - `test_read_article_product_map_rejects_duplicate_article_id`

  The tests should assert this concrete contract:

  ```python
  assert profile.columns.tolist() == [
      "customer_id",
      "age",
      "age_bucket",
      "club_member_status",
      "fashion_news_frequency",
  ]
  assert profile["age_bucket"].tolist() == [
      "unknown",
      "0-19",
      "20-29",
      "30-39",
      "40-49",
      "50-59",
      "60+",
  ]
  assert product_map.columns.tolist() == ["article_id", "product_code"]
  assert product_map["article_id"].astype(str).tolist() == ["0000000001"]
  ```

- [ ] **Step 2: Run tests and verify the expected failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_inputs.py tests/test_recommendation_contracts.py
  ```

  Expected: FAIL because `CUSTOMER_PROFILE_COLUMNS`, `ARTICLE_PRODUCT_MAP_COLUMNS`, builders, paths, and readers do not exist yet.

- [ ] **Step 3: Implement the input contracts**

  Add constants:

  ```python
  CUSTOMER_AGE_BUCKETS = ("unknown", "0-19", "20-29", "30-39", "40-49", "50-59", "60+")
  CUSTOMER_PROFILE_COLUMNS = (
      "customer_id",
      "age",
      "age_bucket",
      "club_member_status",
      "fashion_news_frequency",
  )
  ARTICLE_PRODUCT_MAP_COLUMNS = ("article_id", "product_code")
  ```

  Add paths:

  ```python
  CUSTOMER_PROFILE_PATH = RECOMMEND_DIR / "customer_profile.parquet"
  ARTICLE_PRODUCT_MAP_PATH = RECOMMEND_DIR / "article_product_map.parquet"
  ```

  Add builders with these exact signatures: `build_customer_profile(customers: pd.DataFrame) -> pd.DataFrame` and `build_article_product_map(clean_articles: pd.DataFrame) -> pd.DataFrame`.

  Required behavior:
  - `customer_id` and `article_id` must remain string-like and preserve leading zeroes.
  - `age` is numeric nullable; missing or unparsable age stays null.
  - `age_bucket` uses the fixed bucket enum and never becomes null.
  - `club_member_status` and `fashion_news_frequency` missing values become `unknown`.
  - `product_code` must be string-like and non-null when built from `articles_clean.csv`.
  - Duplicate `customer_id` or `article_id` raises `ValueError`.

- [ ] **Step 4: Wire inputs into metadata and CLI**

  Extend `RecommendationInputArtifacts` with `customer_profile` and `article_product_map`. Extend `build_and_write_recommendation_inputs()` with optional explicit DataFrame inputs:

  ```python
  customers: pd.DataFrame | None = None
  clean_articles: pd.DataFrame | None = None
  ```

  When provided, write the two new parquet artifacts and include them in:
  - `output_artifacts`
  - `row_counts`
  - metadata `config` with schema and bucket algorithm versions
  - `input_artifacts` and `input_fingerprints`

  Update `src/12_build_recommendation_inputs.py` to read:
  - `RAW_CUSTOMERS_PATH` from `fashion_trend.datasets.paths`
  - `ARTICLES_CLEAN_PATH` through `fashion_trend.catalog.readers.read_clean_articles`

  The recommendation package must not import `fashion_trend.datasets.paths`.

- [ ] **Step 5: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_inputs.py tests/test_recommendation_contracts.py
  uv run pytest tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 6: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/contracts.py src/fashion_trend/recommendation/paths.py src/fashion_trend/recommendation/readers.py src/fashion_trend/recommendation/inputs.py src/12_build_recommendation_inputs.py tests/test_recommendation_inputs.py tests/test_recommendation_contracts.py
  git commit -m "feat(recommendation): 增加增强推荐输入契约"
  ```

---

### Task 2: Reorder And Product Variant Sources

**Files:**
- Create: `src/fashion_trend/recommendation/retrieval/reorder.py`
- Create: `src/fashion_trend/recommendation/retrieval/product_variants.py`
- Modify: `src/fashion_trend/recommendation/retrieval/candidates.py`
- Modify: `src/fashion_trend/recommendation/contracts.py`
- Test: `tests/test_recommendation_retrieval.py`

- [ ] **Step 1: Add failing retrieval tests**

  Add tests named:
  - `test_reorder_candidates_use_cutoff_history_and_stable_ranking`
  - `test_reorder_candidates_cap_each_user_window_at_top_12`
  - `test_product_variant_candidates_use_reorder_top_6_seeds`
  - `test_product_variant_candidates_skip_self_and_missing_product_code`
  - `test_enhanced_source_order_rejects_unknown_source`

  The source frame contract must stay:

  ```python
  SOURCE_COLUMNS = (
      "split",
      "cutoff_week",
      "label_week",
      "customer_id",
      "article_id",
      "source",
      "source_rank",
  )
  ```

  The tests should assert `source == "reorder"` and `source == "product_variant"` and one-based `source_rank`.

- [ ] **Step 2: Run tests and verify failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_retrieval.py
  ```

  Expected: FAIL because source modules and `SOURCE_ORDER` entries are missing.

- [ ] **Step 3: Implement `reorder` source**

  Add `build_reorder_candidates(transactions: pd.DataFrame, windows: pd.DataFrame, target_users: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame`.

  Ranking per `customer_id + split + cutoff_week + label_week`:
  1. `last_purchase_week` descending.
  2. `purchase_count` descending.
  3. `article_id` ascending.

  Only rows with `week_id <= cutoff_week` are eligible.

- [ ] **Step 4: Implement `product_variant` source**

  Add `build_product_variant_candidates(reorder_candidates: pd.DataFrame, transactions: pd.DataFrame, article_product_map: pd.DataFrame, windows: pd.DataFrame, *, seed_top_n: int = 6, per_seed_top_n: int = 3, top_n: int = 12) -> pd.DataFrame`.

  Required behavior:
  - Use only reorder seeds with `source_rank <= 6`.
  - Join seeds to `product_code`, then candidate articles sharing that `product_code`.
  - Exclude the seed article itself.
  - Ignore missing `product_code`; do not group missing values as `unknown`.
  - Rank by cutoff-history article popularity, then seed rank, then article ID.

- [ ] **Step 5: Register enhanced source names**

  Update `SOURCE_ORDER` so it explicitly includes:

  ```python
  {
      "popularity": 0,
      "similarity": 1,
      "trend": 2,
      "reorder": 3,
      "product_variant": 4,
      "age_popularity": 5,
      "preference_popularity": 6,
  }
  ```

  This task only implements the first two new sources; the remaining names are registered now so unknown source detection is stable before candidate merging.

- [ ] **Step 6: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_retrieval.py
  uv run pytest tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 7: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/contracts.py src/fashion_trend/recommendation/retrieval/candidates.py src/fashion_trend/recommendation/retrieval/reorder.py src/fashion_trend/recommendation/retrieval/product_variants.py tests/test_recommendation_retrieval.py
  git commit -m "feat(recommendation): 增加复购和同款变体召回"
  ```

---

### Task 3: Age And Preference Popularity Sources

**Files:**
- Create: `src/fashion_trend/recommendation/retrieval/customer_segments.py`
- Create: `src/fashion_trend/recommendation/retrieval/preference_popularity.py`
- Modify: `src/fashion_trend/recommendation/retrieval/candidates.py`
- Test: `tests/test_recommendation_retrieval.py`

- [ ] **Step 1: Add failing tests for segment and preference sources**

  Add tests named:
  - `test_age_popularity_uses_age_bucket_and_recent_four_weeks`
  - `test_age_popularity_does_not_backfill_global_popularity`
  - `test_preference_popularity_uses_top_3_core_attributes`
  - `test_preference_popularity_caps_each_attribute_at_top_4_and_window_at_top_12`
  - `test_preference_popularity_ignores_non_core_attributes`

  Assert the source names exactly:

  ```python
  assert set(age_candidates["source"]) == {"age_popularity"}
  assert set(pref_candidates["source"]) == {"preference_popularity"}
  ```

- [ ] **Step 2: Run tests and verify failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_retrieval.py
  ```

  Expected: FAIL because modules are missing.

- [ ] **Step 3: Implement age-bucket popularity**

  Add `build_age_popularity_candidates(transactions: pd.DataFrame, customer_profile: pd.DataFrame, windows: pd.DataFrame, target_users: pd.DataFrame, *, pool_top_n: int = 50, per_user_top_n: int = 12, recent_weeks: int = 4) -> pd.DataFrame`.

  Use only `week_id <= cutoff_week` and `week_id > cutoff_week - 4`. If a bucket has fewer than 12 candidates, return fewer than 12 for that source.

- [ ] **Step 4: Implement preference popularity**

  Add `build_preference_popularity_candidates(transactions: pd.DataFrame, article_attributes: pd.DataFrame, user_profile: pd.DataFrame, windows: pd.DataFrame, target_users: pd.DataFrame, *, top_attributes: int = 3, per_attribute_top_n: int = 4, per_user_top_n: int = 12, recent_weeks: int = 4) -> pd.DataFrame`.

  Use only `RECOMMENDATION_CORE_ATTR_TYPES`. Ranking should use preference score descending, attribute recent popularity descending, then article ID ascending.

- [ ] **Step 5: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_retrieval.py
  uv run pytest tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 6: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/retrieval/customer_segments.py src/fashion_trend/recommendation/retrieval/preference_popularity.py src/fashion_trend/recommendation/retrieval/candidates.py tests/test_recommendation_retrieval.py
  git commit -m "feat(recommendation): 增加用户分群和偏好热门召回"
  ```

---

### Task 4: Enhanced Default Candidate Strategy And Source-Level Seen Semantics

**Files:**
- Modify: `src/fashion_trend/recommendation/contracts.py`
- Modify: `src/fashion_trend/recommendation/retrieval/candidates.py`
- Modify: `src/fashion_trend/recommendation/features/cache.py`
- Modify: `src/fashion_trend/recommendation/ranking/filters.py`
- Modify: `src/13_build_recommend_candidates.py`
- Test: `tests/test_recommendation_retrieval.py`
- Test: `tests/test_recommendation_feature_cache.py`

- [ ] **Step 1: Add failing tests for enhanced merge and seen flags**

  Add tests named:
  - `test_enhanced_default_candidates_merge_all_sources_with_stable_source_order`
  - `test_enhanced_default_candidates_keep_reorder_allow_seen_after_dedup`
  - `test_source_level_seen_filter_keeps_seen_reorder_and_filters_other_seen_sources`
  - `test_candidate_seen_flags_include_is_seen_allow_seen_and_has_reorder_source`
  - `test_enhanced_candidate_metadata_records_source_caps_and_seen_policy`

  Expected enhanced candidate columns:

  ```python
  ENHANCED_CANDIDATE_ITEM_COLUMNS = (
      *CANDIDATE_ITEM_COLUMNS,
      "has_reorder_source",
      "allow_seen",
  )
  ```

  Existing `default` candidates must keep the old `CANDIDATE_ITEM_COLUMNS` contract.

- [ ] **Step 2: Run tests and verify failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_retrieval.py tests/test_recommendation_feature_cache.py
  ```

  Expected: FAIL because `enhanced_default` is not registered and source-level flags do not exist.

- [ ] **Step 3: Implement enhanced source assembly**

  Add `enhanced_default` to `RECOMMENDATION_CANDIDATE_STRATEGIES` and `CANDIDATE_INPUT_KEYS_BY_STRATEGY`.

  Extend `build_source_frames_for_frames()` so `enhanced_default` assembles:
  - existing `popularity`
  - existing `similarity`
  - existing `trend`
  - `reorder`
  - `product_variant`
  - `age_popularity`
  - `preference_popularity`

  The function signature should accept:

  ```python
  customer_profile: pd.DataFrame | None = None
  article_product_map: pd.DataFrame | None = None
  ```

  Missing required inputs for `enhanced_default` must raise `FileNotFoundError`.

- [ ] **Step 4: Implement enhanced merge columns**

  Add an internal merge helper that computes:
  - `candidate_sources`: pipe-separated all-source list ordered by `SOURCE_ORDER`.
  - `primary_source`: source with best `source_rank`, then `SOURCE_ORDER`.
  - `best_source_rank`: min source rank.
  - `has_reorder_source`: true when `candidate_sources` contains `reorder`.
  - `allow_seen`: true only when `has_reorder_source` is true.

  Calling `build_candidate_items` with `strategy="default"` must still return the existing column list. Calling it with `strategy="enhanced_default"` must return enhanced columns.

- [ ] **Step 5: Implement source-level seen filtering**

  Keep `build_candidate_seen_flags()` compatible with old strategies and extend enhanced partitions to include:

  ```python
  "is_seen"
  "allow_seen"
  "has_reorder_source"
  ```

  Add `filter_seen_items_by_source_policy(candidates: pd.DataFrame) -> pd.DataFrame`.

  It must filter rows where `is_seen == True and allow_seen == False`, and retain rows where `is_seen == True and allow_seen == True`.

- [ ] **Step 6: Write enhanced candidate metadata**

  For `enhanced_default`, metadata `config` must include:
  - source caps from the spec.
  - `seen_policy`: `source_level_reorder_only`.
  - `include_seen_for_reorder`: `true`.
  - `source_order`.

  Metadata input artifacts must include `weekly_transactions`, `article_attributes`, `trend_predictions`, `time_windows`, `target_users`, `user_profile`, `customer_profile`, and `article_product_map` when paths are available.

- [ ] **Step 7: Wire CLI candidate build**

  Update `src/13_build_recommend_candidates.py` to read `CUSTOMER_PROFILE_PATH` and `ARTICLE_PRODUCT_MAP_PATH` only for `enhanced_default`.

- [ ] **Step 8: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_retrieval.py tests/test_recommendation_feature_cache.py
  uv run pytest tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 9: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/contracts.py src/fashion_trend/recommendation/retrieval/candidates.py src/fashion_trend/recommendation/features/cache.py src/fashion_trend/recommendation/ranking/filters.py src/13_build_recommend_candidates.py tests/test_recommendation_retrieval.py tests/test_recommendation_feature_cache.py
  git commit -m "feat(recommendation): 构建增强默认候选策略"
  ```

---

### Task 5: Enhanced Ranking Feature Cache And Score Normalization

**Files:**
- Modify: `src/fashion_trend/recommendation/contracts.py`
- Modify: `src/fashion_trend/recommendation/features/cache.py`
- Modify: `src/fashion_trend/recommendation/ranking/features.py`
- Modify: `src/fashion_trend/recommendation/runner.py`
- Test: `tests/test_recommendation_ranking.py`
- Test: `tests/test_recommendation_feature_cache.py`

- [ ] **Step 1: Add failing tests for enhanced score features**

  Add tests named:
  - `test_enhanced_rank_norm_scores_are_group_normalized_to_unit_interval`
  - `test_enhanced_scores_fill_missing_sources_with_zero`
  - `test_source_rank_score_uses_filtered_candidate_sources`
  - `test_source_count_score_uses_filtered_source_count_cap`
  - `test_enhanced_feature_cache_writes_strategy_scoped_partitions`
  - `test_enhanced_feature_cache_rejects_default_strategy_reuse`
  - `test_enhanced_feature_metadata_records_algorithm_and_strategy`

  Required feature name mapping:

  ```python
  {
      "reorder_scores": "reorder_score",
      "variant_scores": "variant_score",
      "age_popularity_scores": "age_pop_score",
      "preference_popularity_scores": "preference_pop_score",
      "source_rank_scores": "source_rank_score",
      "source_count_scores": "source_count_score",
  }
  ```

- [ ] **Step 2: Run tests and verify failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_ranking.py tests/test_recommendation_feature_cache.py
  ```

  Expected: FAIL because enhanced score columns and cache partitions are missing.

- [ ] **Step 3: Add enhanced score constants and builders**

  Add enhanced score columns to contracts:

  ```python
  ENHANCED_RECOMMENDATION_SCORE_COLUMNS = (
      "pop_score",
      "recent_score",
      "sim_score",
      "trend_score",
      "reorder_score",
      "variant_score",
      "age_pop_score",
      "preference_pop_score",
      "source_rank_score",
      "source_count_score",
  )
  ```

  Add functions that compute raw signals exactly from the spec and then call existing `minmax_normalize_by_group()` with group columns:

  ```python
  ("customer_id", "split", "cutoff_week", "label_week")
  ```

  Constant groups must output `0.0`; non-finite values must raise `ValueError`.

- [ ] **Step 4: Extend feature cache**

  Add enhanced feature names to `FEATURE_NAMES`, `FEATURE_INPUT_KEYS`, and `FEATURE_JOIN_SPECS`.

  Feature partitions for enhanced scores must use:

  ```text
  data/processed/recommend/features/<feature_name>/strategy=enhanced_default/split=<split>/cutoff_week=<week>/part.parquet
  ```

  Metadata must include the strategy, feature name, schema version, algorithm version, and all input artifacts that affect that score.

- [ ] **Step 5: Ensure method window runner can join enhanced features**

  Update `build_cached_feature_frame_for_window()` so any required feature in `FEATURE_JOIN_SPECS` can be joined. Unknown required features must still raise `ValueError`.

- [ ] **Step 6: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_ranking.py tests/test_recommendation_feature_cache.py
  uv run pytest tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 7: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/contracts.py src/fashion_trend/recommendation/features/cache.py src/fashion_trend/recommendation/ranking/features.py src/fashion_trend/recommendation/runner.py tests/test_recommendation_ranking.py tests/test_recommendation_feature_cache.py
  git commit -m "feat(recommendation): 增加增强排序特征缓存"
  ```

---

### Task 6: Enhanced Pop Similarity Trend Method

**Files:**
- Create: `src/fashion_trend/recommendation/methods/trend_aware/enhanced_pop_similarity_trend.py`
- Modify: `src/fashion_trend/recommendation/methods/trend_aware/__init__.py`
- Modify: `src/fashion_trend/recommendation/contracts.py`
- Modify: `src/fashion_trend/recommendation/registry.py`
- Modify: `src/fashion_trend/recommendation/runner.py`
- Modify: `src/fashion_trend/recommendation/outputs.py`
- Modify: `src/14_rerank_recommendations.py`
- Test: `tests/test_recommendation_methods.py`

- [ ] **Step 1: Add failing method tests**

  Add tests named:
  - `test_enhanced_pop_similarity_trend_contract`
  - `test_enhanced_method_uses_enhanced_default_candidates`
  - `test_enhanced_method_disables_method_level_backfill`
  - `test_enhanced_method_filters_seen_by_source_policy`
  - `test_enhanced_method_records_underfilled_diagnostics`
  - `test_enhanced_items_preserve_enhanced_score_columns`

  Expected default weights:

  ```python
  {
      "pop_score": 0.14,
      "recent_score": 0.30,
      "sim_score": 0.10,
      "trend_score": 0.08,
      "reorder_score": 0.16,
      "variant_score": 0.08,
      "age_pop_score": 0.04,
      "preference_pop_score": 0.04,
      "source_rank_score": 0.03,
      "source_count_score": 0.03,
  }
  ```

  The weights sum to 1.0 and are a starting point only; experiment search can select a different valid-best vector.

- [ ] **Step 2: Run tests and verify failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_methods.py tests/test_recommendation_ranking.py
  ```

  Expected: FAIL because the method is not registered and output contracts do not include enhanced score columns.

- [ ] **Step 3: Implement and register the method**

  Add a frozen dataclass named `EnhancedPopSimilarityTrendMethod` with `name="enhanced_pop_similarity_trend"`, `method_type="trend_aware"`, `default_candidate_strategy="enhanced_default"`, and `required_features=ENHANCED_RECOMMENDATION_SCORE_COLUMNS`.

  Register it in `registry.py` and add the method name to `RECOMMENDATION_METHODS`.

- [ ] **Step 4: Disable backfill for the enhanced method**

  Ensure `BACKFILL_MODE_BY_METHOD` has no entry for `enhanced_pop_similarity_trend`.

  Method metadata must record:
  - `backfill_mode: None`
  - `underfilled_user_count`
  - `still_underfilled_user_count`
  - `candidate_strategy: enhanced_default`
  - `source_level_seen_policy: reorder_only`

- [ ] **Step 5: Preserve enhanced score columns in long items**

  Extend `RECOMMENDATION_ITEMS_COLUMNS` and output formatting so enhanced method outputs include the new score columns while existing methods remain readable. If preserving a single global item schema is simpler, set missing enhanced columns to `0.0` for legacy methods and update reader tests accordingly.

- [ ] **Step 6: Wire CLI reranking**

  Update `src/14_rerank_recommendations.py` so enhanced method reads:
  - `enhanced_default` candidates
  - `CUSTOMER_PROFILE_PATH`
  - `ARTICLE_PRODUCT_MAP_PATH`
  - `FEATURE_CACHE_METADATA_PATH`
  - LightGBM trend predictions

  It must not treat enhanced method as `pop_similarity_trend` when choosing trend inputs.

- [ ] **Step 7: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_methods.py tests/test_recommendation_ranking.py tests/test_recommendation_feature_cache.py
  uv run pytest tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 8: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/methods/trend_aware/enhanced_pop_similarity_trend.py src/fashion_trend/recommendation/methods/trend_aware/__init__.py src/fashion_trend/recommendation/contracts.py src/fashion_trend/recommendation/registry.py src/fashion_trend/recommendation/runner.py src/fashion_trend/recommendation/outputs.py src/14_rerank_recommendations.py tests/test_recommendation_methods.py tests/test_recommendation_ranking.py tests/test_recommendation_feature_cache.py
  git commit -m "feat(recommendation): 增加增强趋势感知排序方法"
  ```

---

### Task 7: Recommendation Enhanced Experiment With Valid/Test Separation

**Files:**
- Create: `src/fashion_trend/recommendation/experiments/enhanced_grid_search.py`
- Create: `src/fashion_trend/recommendation/experiments/enhanced_runner.py`
- Modify: `src/fashion_trend/recommendation/experiments/runner.py`
- Modify: `src/16_run_recommendation_experiment.py`
- Test: `tests/test_recommendation_experiments.py`

- [ ] **Step 1: Add failing experiment dispatch and search tests**

  Add tests named:
  - `test_enhanced_weight_grid_has_at_most_32_normalized_rows`
  - `test_enhanced_selects_best_weights_by_valid_map_then_ndcg`
  - `test_recommendation_enhanced_dispatch_does_not_call_main_runner`
  - `test_recommendation_enhanced_payload_records_valid_test_metrics`
  - `test_recommendation_enhanced_does_not_publish_pop_similarity_trend_stable`

  Selection contract:

  ```python
  assert payload["selection_metric"] == "map_at_12"
  assert payload["tie_break"] == "ndcg_at_12"
  assert "valid" in payload["metrics"]["enhanced_pop_similarity_trend"]
  assert "test" in payload["metrics"]["enhanced_pop_similarity_trend"]
  ```

- [ ] **Step 2: Run tests and verify failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_experiments.py
  ```

  Expected: FAIL because enhanced experiment modules and dispatch are missing.

- [ ] **Step 3: Implement enhanced weight grid**

  Add `iter_enhanced_weight_grid()` with at most 32 fixed, non-negative, normalized weight dictionaries over all enhanced score columns.

  Add `select_best_enhanced_weights(results: list[dict[str, Any]]) -> dict[str, float]`.

  Sort by `valid_metrics["map_at_12"]` descending, then `valid_metrics["ndcg_at_12"]` descending, then `grid_index` ascending.

- [ ] **Step 4: Implement enhanced experiment runner**

  `run_recommendation_enhanced_experiment()` must:
  - ensure recommendation inputs and `enhanced_default` candidates.
  - ensure enhanced feature cache for `enhanced_default`.
  - evaluate baselines `recent_popularity`, `pop_similarity`, and `pop_similarity_trend` as comparison rows.
  - search enhanced weights on valid split only.
  - evaluate the selected enhanced weights on valid and test.
  - write only `outputs/recommendation/experiments/recommendation_enhanced/experiment.json`.
  - not call `publish_trend_method_with_weights()` for `pop_similarity_trend`.
  - not overwrite `outputs/recommendation/experiments/main/experiment.json`.

- [ ] **Step 5: Wire CLI dispatch**

  In `run_recommendation_experiment()`, dispatch `experiment_id == "recommendation_enhanced"` to the enhanced runner. Keep the existing `main` path untouched.

- [ ] **Step 6: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_experiments.py tests/test_recommendation_methods.py tests/test_recommendation_feature_cache.py
  uv run pytest tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 7: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/experiments/enhanced_grid_search.py src/fashion_trend/recommendation/experiments/enhanced_runner.py src/fashion_trend/recommendation/experiments/runner.py src/16_run_recommendation_experiment.py tests/test_recommendation_experiments.py
  git commit -m "feat(recommendation): 增加增强推荐实验编排"
  ```

---

### Task 8: Source-Level Ablation And Candidate Diagnostics

**Files:**
- Create: `src/fashion_trend/recommendation/experiments/enhanced_diagnostics.py`
- Modify: `src/fashion_trend/recommendation/experiments/enhanced_runner.py`
- Modify: `src/fashion_trend/recommendation/retrieval/candidates.py`
- Modify: `src/fashion_trend/recommendation/features/cache.py`
- Test: `tests/test_recommendation_experiments.py`
- Test: `tests/test_recommendation_evaluation.py`

- [ ] **Step 1: Add failing diagnostics and ablation tests**

  Add tests named:
  - `test_candidate_recall_counts_full_target_user_labels_pre_and_post_seen`
  - `test_source_hit_contribution_uses_fractional_all_source_credit`
  - `test_source_level_ablation_recomputes_source_fields_after_filter`
  - `test_enhanced_seen_filtered_filters_all_seen_items`
  - `test_enhanced_payload_contains_required_ablation_rows`
  - `test_source_level_ablation_does_not_write_enhanced_default_artifact`

  Required rows:

  ```python
  {
      "Full Model",
      "enhanced_w/o Trend Score",
      "enhanced_w/o Trend Source+Score",
      "enhanced_w/o Reorder/Variant",
      "enhanced_w/o Customer Segment",
      "enhanced_seen_filtered",
  }
  ```

- [ ] **Step 2: Run tests and verify failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_experiments.py tests/test_recommendation_evaluation.py
  ```

  Expected: FAIL because diagnostics and source-level ablation are missing.

- [ ] **Step 3: Implement candidate recall diagnostics**

  Add `compute_candidate_recall(candidates: pd.DataFrame, target_users: pd.DataFrame, labels: pd.DataFrame, *, split: str) -> dict[str, float]`.

  Denominator is all label items for full target users in the split. Users with no candidates remain in the denominator.

- [ ] **Step 4: Implement source contribution diagnostics**

  Add `compute_source_hit_contribution(candidates: pd.DataFrame, labels: pd.DataFrame, *, split: str) -> dict[str, object]`.

  For a label hit with `candidate_sources == "reorder|trend"`, each source gets `0.5` contribution. `primary_source` is emitted only as secondary diagnostics.

- [ ] **Step 5: Implement source-level candidate filtering**

  Add pure function `filter_candidate_sources_for_ablation(candidates: pd.DataFrame, *, dropped_sources: set[str], strategy: str, allow_all_seen: bool = False) -> pd.DataFrame`. The function must not write files.

  It must recompute `candidate_sources`, `primary_source`, `best_source_rank`, `has_reorder_source`, `allow_seen`, `source_rank_score`, and `source_count_score` from the filtered source set. It must not overwrite `candidate_items_path("enhanced_default")`.

- [ ] **Step 6: Add enhanced ablation rows to experiment payload**

  Enhanced payload must include:
  - `candidate_recall_pre_seen`
  - `candidate_recall_post_seen`
  - `source_hit_contribution_pre_seen`
  - `source_hit_contribution_post_seen`
  - `avg_candidates_per_user`
  - `source_coverage`
  - `named_ablation`
  - each ablation row's `source_filter`, `weights`, `metrics`, `candidate_rows`, and `lineage`

  `enhanced_w/o Trend Score` drops and renormalizes only `trend_score`.
  `enhanced_w/o Trend Source+Score` drops source `trend` and weight `trend_score`.
  `enhanced_w/o Reorder/Variant` drops sources `reorder` and `product_variant` and their score weights.
  `enhanced_w/o Customer Segment` drops source `age_popularity` and `age_pop_score`.
  `enhanced_seen_filtered` keeps sources and weights but filters all seen items.

- [ ] **Step 7: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_experiments.py tests/test_recommendation_evaluation.py tests/test_recommendation_feature_cache.py
  uv run pytest tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 8: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/experiments/enhanced_diagnostics.py src/fashion_trend/recommendation/experiments/enhanced_runner.py src/fashion_trend/recommendation/retrieval/candidates.py src/fashion_trend/recommendation/features/cache.py tests/test_recommendation_experiments.py tests/test_recommendation_evaluation.py
  git commit -m "feat(recommendation): 增加增强实验诊断和消融"
  ```

---

### Task 9: Freshness, Reader Contracts, And Architecture Boundaries

**Files:**
- Modify: `src/fashion_trend/recommendation/freshness.py` only if a shared helper is required.
- Modify: `src/fashion_trend/recommendation/readers.py`
- Modify: `src/fashion_trend/recommendation/experiments/enhanced_runner.py`
- Modify: `src/fashion_trend/recommendation/features/cache.py`
- Modify: `tests/test_architecture_boundaries.py`
- Test: `tests/test_recommendation_freshness.py`
- Test: `tests/test_recommendation_contracts.py`
- Test: `tests/test_recommendation_feature_cache.py`
- Test: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: Add failing freshness and boundary tests**

  Add tests named:
  - `test_enhanced_candidate_freshness_requires_customer_profile_and_product_map`
  - `test_enhanced_feature_cache_rejects_changed_candidate_metadata`
  - `test_enhanced_method_output_metadata_includes_enhanced_feature_partitions`
  - `test_recommendation_enhanced_payload_records_candidate_and_cache_fingerprints`
  - `test_recommendation_package_does_not_import_datasets_paths`
  - `test_retrieval_and_ranking_do_not_read_raw_csv_paths`

- [ ] **Step 2: Run tests and verify failures**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_freshness.py tests/test_recommendation_contracts.py tests/test_recommendation_feature_cache.py tests/test_architecture_boundaries.py
  ```

  Expected: FAIL if any enhanced path misses a fingerprint, reader validation, or import boundary.

- [ ] **Step 3: Harden reader contracts**

  Ensure strict readers reject:
  - missing `customer_profile` columns.
  - duplicate `customer_id`.
  - null or invalid `age_bucket`.
  - missing `article_product_map` columns.
  - duplicate `article_id`.
  - null `product_code`.
  - enhanced candidates without `has_reorder_source` or `allow_seen`.
  - enhanced recommendation items missing required enhanced score columns.

- [ ] **Step 4: Harden freshness metadata**

  Ensure metadata and expected fingerprint checks include:
  - weekly transactions.
  - article attributes.
  - trend predictions.
  - time windows.
  - target users.
  - user profile.
  - customer profile.
  - article product map.
  - candidate metadata.
  - feature cache partitions and partition metadata.

  Error text should point to the existing precise rebuild options: `--force-cache`, `--force-candidates`, or `--force-rebuild-all`.

- [ ] **Step 5: Harden architecture tests**

  Extend architecture tests so:
  - `src/fashion_trend/recommendation/` does not import `fashion_trend.datasets`.
  - retrieval and ranking modules do not import raw path modules or call `pd.read_csv`.
  - `recomd` is not imported anywhere.

- [ ] **Step 6: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_freshness.py tests/test_recommendation_contracts.py tests/test_recommendation_feature_cache.py tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 7: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add src/fashion_trend/recommendation/freshness.py src/fashion_trend/recommendation/readers.py src/fashion_trend/recommendation/experiments/enhanced_runner.py src/fashion_trend/recommendation/features/cache.py tests/test_recommendation_freshness.py tests/test_recommendation_contracts.py tests/test_recommendation_feature_cache.py tests/test_architecture_boundaries.py
  git commit -m "test(recommendation): 收紧增强推荐契约和边界"
  ```

  If `freshness.py` is unchanged, do not stage it.

---

### Task 10: Documentation And Project Status Sync

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/gpt-research/implementation-plan.md`
- Modify: `apps/defense_app/README.md` only if wording would otherwise imply defense app consumes enhanced outputs by default.
- Test: `tests/test_recommendation_contracts.py`
- Test: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: Add or update doc guard tests if needed**

  If existing tests do not guard the default-output boundary, add a contract test named `test_enhanced_experiment_does_not_change_reports_or_defense_default_method`.

  It should assert code-level defaults still point to `pop_similarity_trend` and `main` where reports or defense app loaders expect stable recommendation artifacts.

- [ ] **Step 2: Update README command and artifact sections**

  Add the new optional commands:

  ```sh
  uv run python src/13_build_recommend_candidates.py --strategy enhanced_default
  uv run python src/14_rerank_recommendations.py --method enhanced_pop_similarity_trend
  uv run python src/16_run_recommendation_experiment.py --experiment recommendation_enhanced
  ```

  State explicitly:
  - `recommendation_enhanced` is optional and separate from `main`.
  - it does not overwrite `outputs/recommendation/experiments/main/experiment.json`.
  - it does not default-replace `outputs/recommendation/pop_similarity_trend/`.
  - reports and defense app remain stable-output consumers unless explicitly changed later.

- [ ] **Step 3: Update AGENTS.md project guidance**

  Add the new artifacts and module boundaries:
  - `data/processed/recommend/customer_profile.parquet`
  - `data/processed/recommend/article_product_map.parquet`
  - `data/processed/recommend/candidates/enhanced_default/`
  - `outputs/recommendation/enhanced_pop_similarity_trend/`
  - `outputs/recommendation/experiments/recommendation_enhanced/`

  Keep the wording clear that this is an offline experiment layer, not an online service or deep recommendation system.

- [ ] **Step 4: Update research status document**

  In `docs/gpt-research/implementation-plan.md`, record the enhanced experiment as first-stage recommendation enhancement and include the paper boundary:
  - success is relative improvement over `pop_similarity_trend` in MAP@12 or NDCG@12.
  - no claim that it exceeds `recent_popularity` unless artifact metrics prove that.
  - trend contribution must be described cautiously if strict trend source+score ablation is weak.

- [ ] **Step 5: Run focused validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_contracts.py tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  ```

  Expected: PASS.

- [ ] **Step 6: Review and commit**

  Run:

  ```sh
  git diff
  git status --short
  ```

  Commit:

  ```sh
  git add README.md AGENTS.md docs/gpt-research/implementation-plan.md apps/defense_app/README.md tests/test_recommendation_contracts.py
  git commit -m "docs(recommendation): 同步增强推荐实验说明"
  ```

  If `apps/defense_app/README.md` or `tests/test_recommendation_contracts.py` is unchanged, do not stage it.

---

### Task 11: Final Verification And Result Boundary Check

**Files:**
- No source change expected.
- Test: `tests/test_recommendation_*.py`
- Test: `tests/test_architecture_boundaries.py`

- [ ] **Step 1: Run required final validation**

  Run:

  ```sh
  uv run pytest tests/test_recommendation_*.py tests/test_architecture_boundaries.py
  uv run python -m compileall -q src
  uv run black --check src tests
  uv run isort --check-only src tests
  ```

  Expected: PASS. If `uv` cache permission fails, rerun the exact command with:

  ```sh
  UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_recommendation_*.py tests/test_architecture_boundaries.py
  UV_CACHE_DIR=/private/tmp/uv-cache uv run python -m compileall -q src
  UV_CACHE_DIR=/private/tmp/uv-cache uv run black --check src tests
  UV_CACHE_DIR=/private/tmp/uv-cache uv run isort --check-only src tests
  ```

- [ ] **Step 2: Verify default artifacts are not targeted by the enhanced experiment**

  Inspect code and diff:

  ```sh
  rg -n "recommendation_enhanced|enhanced_pop_similarity_trend|enhanced_default|pop_similarity_trend|experiments/main" src README.md AGENTS.md docs/gpt-research/implementation-plan.md
  git diff --stat
  git status --short
  ```

  Confirm:
  - no generated files under `data/` or `outputs/` are staged.
  - `recommendation_enhanced` writes to its own experiment dir.
  - `main` experiment behavior is still available.
  - reports and defense app defaults are unchanged.

- [ ] **Step 3: Optional artifact smoke run only if inputs already exist**

  If the local stable upstream artifacts already exist and runtime cost is acceptable, run:

  ```sh
  uv run python src/16_run_recommendation_experiment.py --experiment recommendation_enhanced --force-experiment
  ```

  Then inspect `outputs/recommendation/experiments/recommendation_enhanced/experiment.json` without staging it. If the artifact run is skipped because required local data is missing or runtime is too high, state that in the final summary.

- [ ] **Step 4: Final review**

  Run:

  ```sh
  git log --oneline -n 8
  git status --short
  ```

  Ensure each task has a separate commit and the worktree contains no unintended generated artifacts.

---

## Spec Coverage Checklist

- customer profile and article product map input artifacts: Task 1.
- `enhanced_default` candidate strategy and source metadata: Tasks 2, 3, and 4.
- `reorder`, `product_variant`, `age_popularity`, and `preference_popularity`: Tasks 2 and 3.
- source-level seen semantics, `allow_seen`, and `has_reorder_source`: Task 4.
- ranking feature cache and score normalization: Task 5.
- `enhanced_pop_similarity_trend` method: Task 6.
- `recommendation_enhanced` experiment with valid/test separation: Task 7.
- source-level ablation, candidate recall pre/post seen, and source contribution pre/post seen: Task 8.
- freshness metadata, reader contract, and architecture boundaries: Task 9.
- docs, README, and project status sync: Task 10.
- final validation gate: Task 11.

## Go/No-Go Notes For Paper Results

- Go condition: `enhanced_pop_similarity_trend` improves over `pop_similarity_trend` on MAP@12 or NDCG@12, and `enhanced_default` improves candidate recall over `default`.
- No-Go for strong trend claim: if `enhanced_w/o Trend Source+Score` does not drop clearly, write trend as an explanatory auxiliary signal.
- `recent_popularity` can remain stronger; the enhanced recommendation experiment is an application validation extension, not the paper's main contribution.
