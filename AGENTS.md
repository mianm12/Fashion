# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10-3.12 project for H&M fashion trend and recommendation experiments. Numbered scripts in `src/00_*.py` through `src/11_*.py` are workflow entrypoints; keep them as readable orchestration layers. Reusable logic belongs under `src/fashion_trend/`, organized by domain:

- `foundation/`: paths, logging, IO, dataframe, and artifact helpers.
- `datasets/`, `transactions/`, `catalog/`, `trend/`: pipeline business domains.
- `recommendation/` and `reports/`: downstream contracts and reporting boundaries.

Tests live in `tests/` and are split by pipeline stage. Large artifacts are under `data/` and `outputs/`; do not treat datasets, model outputs, or credentials as source files.

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the full pytest suite with `src` on `PYTHONPATH`.
- `uv run black --check src tests`: check Python formatting.
- `uv run isort --check-only src tests`: check import ordering using the Black profile.
- `uv run python src/10_train_trend_model.py --model last_week`: train a registered trend model and write `outputs/models/<model>/`.
- `uv run python src/11_eval_trend_model.py --model last_week`: evaluate predictions and write `outputs/metrics/<model>/trend_metrics.json`.

## Coding Style & Naming Conventions

Use Black formatting, isort with `profile = "black"`, and clear snake_case names for modules, functions, variables, and test helpers. Keep numbered scripts thin: parse arguments, call package functions, log progress, and exit. Put core calculations, validation, readers, and writers in the relevant `fashion_trend` domain package. Preserve the existing architecture boundary: avoid importing compatibility facades when a concrete submodule import is available.

## Testing Guidelines

The project uses pytest. Name test files `tests/test_*.py` and align new tests with real pipeline stages such as article sales, attribute heat, targets, samples, splits, training, evaluation, or architecture boundaries. Prefer focused fixtures and helpers in `tests/trend_samples.py` or `tests/__init__.py`. For bug fixes, add a regression test before or alongside the fix.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit prefixes with concise Chinese summaries, for example `feat(trend): 添加 previous_growth 基线`, `test(trend): 覆盖三类基线训练评价`, and `docs: 对齐三类趋势基线语义`. Keep commits scoped by feature or stage, and verify before committing. Pull requests should describe the changed stage, list validation commands, mention artifact paths when relevant, and call out data or configuration assumptions.

## Security & Configuration Tips

Kaggle credentials, API tokens, `.env` files, raw datasets, and generated outputs must stay out of commits. Use environment variables or local credential files outside the repository for data access.
