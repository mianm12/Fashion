# Kaggle Download Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable Kaggle competition download script for placing datasets under `data/raw`.

**Architecture:** Keep the behavior in `src/00_download_data.py` with small pure helpers for path calculation, skip detection, KaggleHub downloader invocation, and zip extraction. Tests import the script by file path and inject a fake downloader so verification does not need Kaggle credentials or network access.

**Tech Stack:** Python 3.13, `kagglehub`, standard library `argparse`, `zipfile`, `unittest`.

---

### Task 1: Tests

**Files:**
- Create: `tests/test_download_data.py`

- [ ] Write failing `unittest` coverage for default arguments, target directory construction, skip logic, KaggleHub downloader invocation, and zip extraction.
- [ ] Run `python -m unittest tests.test_download_data -v` and confirm tests fail because `src/00_download_data.py` has no implementation yet.

### Task 2: Script

**Files:**
- Modify: `src/00_download_data.py`

- [ ] Implement CLI parsing with defaults.
- [ ] Implement `download_competition()` using `kagglehub.competition_download`.
- [ ] Implement default unzip behavior.
- [ ] Implement helpful `ImportError` handling for missing `kagglehub`.
- [ ] Run `python -m unittest tests.test_download_data -v` and confirm tests pass.
- [ ] Run `python src/00_download_data.py --help` and confirm the CLI is usable.
