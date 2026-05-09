# Repository Guidelines

## Project Scope & Current State

This is a Python 3.10-3.12 project for H&M fashion trend prediction and lightweight recommendation experiments. The current implemented pipeline reaches weekly data preparation, article cleaning, attribute graph construction, article weekly sales, attribute weekly heat, trend labels, trend samples, time-based splits, three baseline trend models, the LightGBM main model, and trend evaluation.

Recommendation generation and recommendation evaluation are not implemented yet. Do not describe them as working code unless the implementation and artifacts exist.

Large datasets and generated artifacts live under `data/` and `outputs/`. Treat them as runtime artifacts, not source files.

## Project Structure & Module Organization

Numbered scripts in `src/00_*.py` through `src/11_*.py` are workflow entrypoints. Keep them as readable orchestration layers: parse arguments, call package functions in order, log progress, and return stable exit codes. Core calculations, validation, readers, writers, and artifact handling belong under `src/fashion_trend/`.

The package is organized by domain:

- `foundation/`: project roots, logging, dataframe utilities, safe IO, and artifact helpers.
- `datasets/`: Kaggle download and raw dataset profiling.
- `transactions/`: transaction paths, contracts, readers, and weekly aggregation.
- `catalog/`: article cleaning, catalog contracts/readers, and `catalog/graph/` builders and publishers.
- `trend/`: trend schema, paths, readers, predictions, heat, labels, features, splits, training, evaluation, and model implementations.
- `recommendation/`: recommendation contracts, paths, and readers for future downstream work.
- `reports/`: read-only reporting paths and boundaries.

Inside `trend/`, keep responsibilities explicit:

- `heat/`: article weekly sales and attribute weekly heat.
- `labels/`: trend target generation.
- `features/`: trend model sample generation.
- `splits/`: time-based train/valid/test split logic.
- `models/baselines/`: deterministic baselines such as `last_week`, `previous_growth`, and `moving_average`.
- `models/supervised/`: supervised models such as `lightgbm`.
- `models/registry.py`: model registration and lookup.
- `training/`: training runner, output paths, and run artifact contracts.
- `evaluation/`: metrics, payloads, runner, and metric artifact contracts.

Do not reintroduce historical root modules such as `fashion_trend.training`, `fashion_trend.evaluation`, `fashion_trend.models`, `fashion_trend.articles`, or `fashion_trend.data_loader`. Do not import through the `fashion_trend.trend` facade when a concrete submodule import is available; architecture tests enforce direct imports.

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the full pytest suite with `src` on `PYTHONPATH`.
- `uv run pytest tests/test_trend_training.py tests/test_trend_lightgbm.py tests/test_trend_evaluation.py`: focused validation for trend training, LightGBM, and evaluation changes.
- `uv run black --check src tests`: check Python formatting.
- `uv run isort --check-only src tests`: check import ordering using the Black profile.
- `uv run python -m compileall -q src`: compile-check package and numbered scripts when import or CLI boundaries change.

Run the implemented pipeline in numbered order when validating artifacts:

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

Train and evaluate registered trend models through the shared entrypoints:

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

LightGBM tuning runs are run-scoped and should not accidentally replace the stable main result:

```sh
uv run python src/10_train_trend_model.py --model lightgbm --run-id smoke-lightgbm --no-promote
uv run python src/11_eval_trend_model.py --model lightgbm --run-id smoke-lightgbm
uv run python src/10_train_trend_model.py --model lightgbm --promote-run smoke-lightgbm
```

`--run-id`, `--params`, `--param`, `--promote`, `--no-promote`, and `--promote-run` are LightGBM-only options. Baselines must reject those options rather than silently ignoring them.

## Artifact Contracts

Key data artifacts:

- `data/interim/transactions_train_weekly.parquet`
- `data/interim/articles_clean_mvp.csv`
- `data/interim/articles_clean.csv`
- `data/interim/nodes_article.csv`
- `data/interim/nodes_attribute.csv`
- `data/interim/edges_article_attribute.csv`
- `data/interim/edges_attribute_hierarchy.csv`
- `data/interim/article_week_sales.csv`
- `data/interim/attribute_week_heat.csv`
- `data/interim/attribute_week_target.csv`
- `data/processed/trend_model_samples.parquet`
- `data/processed/trend_model_samples_train.parquet`
- `data/processed/trend_model_samples_valid.parquet`
- `data/processed/trend_model_samples_test.parquet`

Baseline and stable model outputs use:

- `outputs/models/<model>/predictions.csv`
- `outputs/models/<model>/params.json`
- `outputs/models/<model>/metadata.json`
- `outputs/metrics/<model>/trend_metrics.json`

LightGBM stable outputs additionally include `feature_importance.csv` and `model.txt`. LightGBM run outputs use `outputs/models/lightgbm/runs/<run_id>/...` and `outputs/metrics/lightgbm/runs/<run_id>/trend_metrics.json`.

The stable LightGBM directory represents the current main result. Run directories represent preserved experiments. A parameterized or explicit run should default to not promoting. `--promote` publishes model artifacts only and does not run evaluation. `--promote-run` publishes an already evaluated run and must keep stable model artifacts and stable metrics aligned to the same `run_id`.

## Coding Style & Architecture Boundaries

Use Black formatting, isort with `profile = "black"`, and clear snake_case names for modules, functions, variables, and test helpers.

Prefer existing project helpers over new ad hoc utilities:

- Use `foundation.paths` and domain `paths.py` modules for path constants.
- Use `foundation.logging` for CLI logs.
- Use `foundation.io` and artifact helpers for safe writes and JSON/CSV/parquet IO.
- Keep validation close to the domain package that owns the contract.

Architecture boundaries are intentionally tested:

- `foundation` must not import business domains.
- `datasets`, `transactions`, and `catalog` depend only on allowed lower-level utilities.
- `trend` may depend on stable input domains, but not on `datasets`, `recommendation`, or `reports`.
- `recommendation` and `reports` may read only public upstream contracts/readers, not core computation modules.

When adding functionality, place it in the domain that owns the business fact. Do not make numbered scripts the source of reusable logic.

## Testing Guidelines

The project uses pytest. Name test files `tests/test_*.py` and align new tests with real pipeline stages: foundation artifacts, article cleaning, attribute graph, article sales, attribute heat, targets, samples, splits, training, LightGBM, evaluation, or architecture boundaries.

Tests should not require the real H&M dataset unless explicitly called out as artifact validation. Prefer small in-memory fixtures and shared helpers in `tests/trend_samples.py` or `tests/__init__.py`.

For bug fixes, add or update a regression test that fails before the fix. For model or artifact contract changes, validate both happy paths and boundary failures such as invalid model names, invalid `run_id`, missing split columns, unsafe paths, and mismatched prediction/metrics payloads.

## Documentation Guidelines

Keep `README.md`, `docs/gpt-research/implementation-plan.md`, and relevant `docs/superpowers/specs/` or `docs/superpowers/plans/` aligned with as-built behavior when command syntax, artifact paths, model semantics, or architecture boundaries change.

Do not leave historical design text implying that implemented modules still live in removed root files. Current trend training and evaluation code lives under `src/fashion_trend/trend/`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit prefixes with concise Chinese summaries, for example `fix(trend): 修正预测校验和 LightGBM 日志` and `docs: 说明 LightGBM run 调参流程`.

Keep commits scoped by feature, stage, or contract. Before committing, inspect `git diff`, confirm the change scope, and run the relevant validation commands. Do not commit generated datasets, model outputs, credentials, caches, or unrelated formatting churn.

Pull requests should describe the changed stage or contract, list validation commands, mention artifact paths when relevant, and call out data or configuration assumptions.

## Security & Configuration Tips

Kaggle credentials, API tokens, `.env` files, raw datasets, generated outputs, model artifacts, and local session files must stay out of commits. Use environment variables or credential files outside the repository for data access.

Do not log secrets or embed credentials in docs, tests, snapshots, metadata, or commit messages.
