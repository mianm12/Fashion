# Recommendation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement recommendation-layer named ablation and trend bucket representative metrics from `docs/superpowers/specs/2026-05-13-recommendation-ablation-design.md`.

**Architecture:** Keep recommendation experiment logic inside `fashion_trend.recommendation.experiments`; do not register new recommendation methods and do not write new method-scoped stable output directories. Extend `experiment.json` with self-describing `named_ablation` and `trend_bucket_best_by_valid`, then let `reports` read those artifact fields without importing recommendation experiment runner internals.

**Tech Stack:** Python 3.10-3.12, pandas, pytest, existing recommendation candidates / feature cache / evaluation runner, existing reports table flattening.

---

## File Structure

- Modify: `src/fashion_trend/recommendation/experiments/ablation.py`
  - Add pure helpers for strict drop-and-renormalize weights, named ablation row construction, trend bucket selection, and finite metric validation.
- Modify: `src/fashion_trend/recommendation/experiments/runner.py`
  - Evaluate strict in-memory variants, build `named_ablation`, build `trend_bucket_best_by_valid`, and write both into `experiment.json`.
- Modify: `src/fashion_trend/reports/runner.py`
  - Flatten `named_ablation` and `trend_bucket_best_by_valid` into `recommendation_experiment_summary`; update strict `w/o Recent` warning detection.
- Modify: `tests/test_recommendation_experiments.py`
  - Cover dynamic `best_weights` derivation, metadata fields, `selection_split`, trend bucket selection, and payload integration.
- Modify: `tests/test_reports_runner.py`
  - Cover reports flattening and warning behavior for new fields.
- Modify: `README.md`
  - Document strict named ablation and trend bucket representative output.
- Modify: `docs/gpt-research/implementation-plan.md`
  - Clarify that `w/o Graph`, `w/o Growth`, and `w/o Rank` remain trend-model feature ablations outside this implementation.
- Modify: `docs/gpt-research/project-status-summary.md`
  - Update after real artifact verification with actual metrics and remaining cautions.

Execution mode for this implementation: inline. The changed files share tight contracts, so the main thread will implement each task, inspect diff, run focused tests, and commit before moving to the next task.

---

### Task 1: Pure Ablation Helpers

**Files:**
- Modify: `src/fashion_trend/recommendation/experiments/ablation.py`
- Test: `tests/test_recommendation_experiments.py`

- [ ] **Step 1: Write failing tests for strict weight derivation and metadata**

Append tests near the existing `build_ablation_summary` tests in `tests/test_recommendation_experiments.py`:

```python
def test_drop_and_renormalize_weights_derive_from_best_weights() -> None:
    from fashion_trend.recommendation.experiments.ablation import (
        derive_strict_ablation_weights,
    )

    best_weights = {
        "pop_score": 0.2,
        "sim_score": 0.2,
        "trend_score": 0.1,
        "recent_score": 0.5,
    }

    without_trend = derive_strict_ablation_weights(best_weights, "trend_score")
    assert without_trend == pytest.approx(
        {
            "pop_score": 2 / 9,
            "sim_score": 2 / 9,
            "trend_score": 0.0,
            "recent_score": 5 / 9,
        }
    )

    without_similarity = derive_strict_ablation_weights(best_weights, "sim_score")
    assert without_similarity == pytest.approx(
        {
            "pop_score": 0.25,
            "sim_score": 0.0,
            "trend_score": 0.125,
            "recent_score": 0.625,
        }
    )

    without_recent = derive_strict_ablation_weights(best_weights, "recent_score")
    assert without_recent == pytest.approx(
        {
            "pop_score": 0.4,
            "sim_score": 0.4,
            "trend_score": 0.2,
            "recent_score": 0.0,
        }
    )


def test_drop_and_renormalize_rejects_invalid_weights() -> None:
    from fashion_trend.recommendation.experiments.ablation import (
        derive_strict_ablation_weights,
    )

    with pytest.raises(ValueError, match="best_weights"):
        derive_strict_ablation_weights({"pop_score": 0.0, "trend_score": 0.0}, "trend_score")

    with pytest.raises(ValueError, match="unknown_score"):
        derive_strict_ablation_weights({"pop_score": 1.0}, "unknown_score")


def test_build_named_ablation_rows_keeps_audit_fields_and_selection_split() -> None:
    from fashion_trend.recommendation.experiments.ablation import (
        build_named_ablation_rows,
    )

    best_weights = {
        "pop_score": 0.2,
        "sim_score": 0.2,
        "trend_score": 0.1,
        "recent_score": 0.5,
    }
    rows = build_named_ablation_rows(
        best_weights=best_weights,
        strict_metrics={
            "without_trend_in_rec": {"valid": {"ndcg_at_12": 0.1}, "test": {"ndcg_at_12": 0.2}},
            "without_similarity": {"valid": {"ndcg_at_12": 0.3}, "test": {"ndcg_at_12": 0.4}},
            "without_recent": {"valid": {"ndcg_at_12": 0.5}, "test": {"ndcg_at_12": 0.6}},
        },
        full_model_metrics={"valid": {"ndcg_at_12": 0.7}, "test": {"ndcg_at_12": 0.8}},
        stable_baseline_metrics={
            "recent_only_baseline": {
                "method": "recent_popularity",
                "display_name": "Recent Only",
                "metrics": {"valid": {"ndcg_at_12": 0.9}, "test": {"ndcg_at_12": 1.0}},
            },
            "pop_similarity_baseline": {
                "method": "pop_similarity",
                "display_name": "Pop + Similarity baseline",
                "metrics": {"valid": {"ndcg_at_12": 1.1}, "test": {"ndcg_at_12": 1.2}},
            },
        },
    )

    by_id = {row["variant_id"]: row for row in rows}
    assert tuple(by_id) == (
        "full_model",
        "without_trend_in_rec",
        "without_similarity",
        "without_recent",
        "recent_only_baseline",
        "pop_similarity_baseline",
    )
    assert by_id["without_trend_in_rec"]["selection_split"] == "valid"
    assert by_id["without_trend_in_rec"]["weight_policy"] == (
        "strict_drop_and_renormalize_from_full"
    )
    assert by_id["without_trend_in_rec"]["weights"] == pytest.approx(
        {
            "pop_score": 2 / 9,
            "sim_score": 2 / 9,
            "trend_score": 0.0,
            "recent_score": 5 / 9,
        }
    )
    assert by_id["recent_only_baseline"]["selection_split"] == "not_applicable"
    assert by_id["recent_only_baseline"]["weight_policy"] == "stable_method_baseline"
```

- [ ] **Step 2: Run tests to confirm failure**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py::test_drop_and_renormalize_weights_derive_from_best_weights tests/test_recommendation_experiments.py::test_build_named_ablation_rows_keeps_audit_fields_and_selection_split -q
```

Expected: fails because `derive_strict_ablation_weights` and `build_named_ablation_rows` do not exist.

- [ ] **Step 3: Implement pure helpers**

Add to `src/fashion_trend/recommendation/experiments/ablation.py`:

```python
from __future__ import annotations

import math
from typing import Any

SCORE_FEATURES = ("pop_score", "sim_score", "trend_score", "recent_score")
STRICT_VARIANTS = (
    ("without_trend_in_rec", "w/o Trend in Rec", "trend_score"),
    ("without_similarity", "w/o Similarity", "sim_score"),
    ("without_recent", "w/o Recent", "recent_score"),
)


def derive_strict_ablation_weights(
    best_weights: dict[str, float],
    dropped_feature: str,
) -> dict[str, float]:
    weights = _read_weights(best_weights, context="best_weights")
    if dropped_feature not in SCORE_FEATURES:
        raise ValueError(f"未知 strict ablation feature: {dropped_feature}")
    if dropped_feature not in weights:
        raise ValueError(f"best_weights 缺少 {dropped_feature}")

    remaining_total = sum(
        value for feature, value in weights.items() if feature != dropped_feature
    )
    if remaining_total <= 0.0 or not math.isfinite(remaining_total):
        raise ValueError("best_weights 删除目标组件后无法归一化。")

    derived = {
        feature: (
            0.0
            if feature == dropped_feature
            else weights.get(feature, 0.0) / remaining_total
        )
        for feature in SCORE_FEATURES
    }
    _validate_weight_sum(derived, context=f"strict ablation {dropped_feature}")
    return derived


def build_named_ablation_rows(
    *,
    best_weights: dict[str, float],
    strict_metrics: dict[str, dict[str, dict[str, float]]],
    full_model_metrics: dict[str, dict[str, float]],
    stable_baseline_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    weights = _read_weights(best_weights, context="best_weights")
    rows: list[dict[str, Any]] = [
        {
            "variant_id": "full_model",
            "display_name": "Full Model",
            "method": "pop_similarity_trend",
            "base_method": "pop_similarity_trend",
            "candidate_strategy": "default",
            "weight_policy": "stable_full_model",
            "selection_split": "valid",
            "metrics_source": "stable_method_output",
            "weights": weights,
            "metrics": _read_metrics_by_split(full_model_metrics, "full_model"),
        }
    ]
    for variant_id, display_name, dropped_feature in STRICT_VARIANTS:
        rows.append(
            {
                "variant_id": variant_id,
                "display_name": display_name,
                "method": "pop_similarity_trend",
                "base_method": "pop_similarity_trend",
                "candidate_strategy": "default",
                "weight_policy": "strict_drop_and_renormalize_from_full",
                "selection_split": "valid",
                "metrics_source": "in_memory_evaluation",
                "weights": derive_strict_ablation_weights(weights, dropped_feature),
                "metrics": _read_metrics_by_split(strict_metrics[variant_id], variant_id),
            }
        )
    for variant_id in ("recent_only_baseline", "pop_similarity_baseline"):
        baseline = stable_baseline_metrics[variant_id]
        rows.append(
            {
                "variant_id": variant_id,
                "display_name": str(baseline["display_name"]),
                "method": str(baseline["method"]),
                "base_method": str(baseline["method"]),
                "candidate_strategy": "not_applicable",
                "weight_policy": "stable_method_baseline",
                "selection_split": "not_applicable",
                "metrics_source": "stable_method_output",
                "weights": {},
                "metrics": _read_metrics_by_split(dict(baseline["metrics"]), variant_id),
            }
        )
    return rows
```

Also add private helpers:

```python
def _read_weights(raw: dict[str, Any], *, context: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for feature in SCORE_FEATURES:
        try:
            value = float(raw.get(feature, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} 的 {feature} 不是有限数值。") from exc
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{context} 的 {feature} 不是非负有限数值。")
        weights[feature] = value
    _validate_weight_sum(weights, context=context)
    return weights


def _validate_weight_sum(weights: dict[str, float], *, context: str) -> None:
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{context} 权重和必须为 1: {total}")


def _read_metrics_by_split(
    metrics_by_split: dict[str, dict[str, float]],
    context: str,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for split in ("valid", "test"):
        if split not in metrics_by_split:
            raise ValueError(f"{context} 缺少 {split} metrics。")
        metrics[split] = {
            key: _finite_number(value, context=f"{context}.{split}.{key}")
            for key, value in dict(metrics_by_split[split]).items()
        }
    return metrics


def _finite_number(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} 不是有限数值。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} 不是有限数值。")
    return number
```

Keep the existing `build_ablation_summary()` function in the same file.

- [ ] **Step 4: Run focused tests**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py::test_drop_and_renormalize_weights_derive_from_best_weights tests/test_recommendation_experiments.py::test_drop_and_renormalize_rejects_invalid_weights tests/test_recommendation_experiments.py::test_build_named_ablation_rows_keeps_audit_fields_and_selection_split -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Review and commit**

Run:

```sh
git diff -- src/fashion_trend/recommendation/experiments/ablation.py tests/test_recommendation_experiments.py
git diff --check
```

Confirm the helpers are pure and do not read/write files. Then commit:

```sh
git add src/fashion_trend/recommendation/experiments/ablation.py tests/test_recommendation_experiments.py
git commit -m "feat(recommendation): 添加命名消融权重 helpers"
```

---

### Task 2: Experiment Runner Integration

**Files:**
- Modify: `src/fashion_trend/recommendation/experiments/runner.py`
- Modify: `tests/test_recommendation_experiments.py`

- [ ] **Step 1: Write failing tests for payload integration**

Add tests in `tests/test_recommendation_experiments.py` near existing experiment payload tests:

```python
def test_build_experiment_payload_includes_named_ablation_and_trend_buckets() -> None:
    payload = experiment_runner.build_experiment_payload(
        "main",
        baseline_payloads=[
            {"method": "recent_popularity", "metrics": {"valid": {"ndcg_at_12": 0.1}, "test": {"ndcg_at_12": 0.2}}},
            {"method": "pop_similarity", "metrics": {"valid": {"ndcg_at_12": 0.3}, "test": {"ndcg_at_12": 0.4}}},
        ],
        search_results=[
            {
                "grid_index": 0,
                "weights": {"pop_score": 0.2, "sim_score": 0.2, "trend_score": 0.1, "recent_score": 0.5},
                "valid_metrics": {"ndcg_at_12": 0.7},
            }
        ],
        trend_payload={
            "method": "pop_similarity_trend",
            "metrics": {"valid": {"ndcg_at_12": 0.8}, "test": {"ndcg_at_12": 0.9}},
        },
        named_ablation=[{"variant_id": "full_model"}],
        trend_bucket_best_by_valid=[{"variant_id": "trend_bucket_0_1"}],
    )

    assert payload["named_ablation"] == [{"variant_id": "full_model"}]
    assert payload["trend_bucket_best_by_valid"] == [{"variant_id": "trend_bucket_0_1"}]
```

Add a test for trend bucket selection:

```python
def test_select_trend_bucket_best_by_valid_uses_ndcg_and_grid_order() -> None:
    from fashion_trend.recommendation.experiments.ablation import (
        select_trend_bucket_representatives,
    )

    rows = select_trend_bucket_representatives(
        [
            {
                "grid_index": 2,
                "weights": {"pop_score": 0.3, "sim_score": 0.2, "trend_score": 0.1, "recent_score": 0.4},
                "valid_metrics": {"ndcg_at_12": 0.4},
            },
            {
                "grid_index": 1,
                "weights": {"pop_score": 0.2, "sim_score": 0.2, "trend_score": 0.1, "recent_score": 0.5},
                "valid_metrics": {"ndcg_at_12": 0.4},
            },
            {
                "grid_index": 3,
                "weights": {"pop_score": 0.4, "sim_score": 0.2, "trend_score": 0.2, "recent_score": 0.2},
                "valid_metrics": {"ndcg_at_12": 0.5},
            },
        ],
        required_trend_scores=(0.1, 0.2),
    )

    assert [row["trend_score"] for row in rows] == [0.1, 0.2]
    assert rows[0]["grid_index"] == 1
```

- [ ] **Step 2: Run tests to confirm failure**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py::test_build_experiment_payload_includes_named_ablation_and_trend_buckets tests/test_recommendation_experiments.py::test_select_trend_bucket_best_by_valid_uses_ndcg_and_grid_order -q
```

Expected: failures because the payload arguments and helper do not exist yet.

- [ ] **Step 3: Add trend bucket helper**

In `src/fashion_trend/recommendation/experiments/ablation.py`, add:

```python
def select_trend_bucket_representatives(
    search_results: list[dict[str, Any]],
    *,
    required_trend_scores: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4),
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for trend_score in required_trend_scores:
        bucket = [
            result
            for result in search_results
            if math.isclose(
                float(dict(result["weights"])["trend_score"]),
                trend_score,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        if not bucket:
            raise ValueError(f"trend bucket 缺少代表组合: {trend_score}")
        best = min(
            bucket,
            key=lambda result: (
                -float(dict(result["valid_metrics"])["ndcg_at_12"]),
                int(result["grid_index"]),
            ),
        )
        selected.append(
            {
                "variant_id": _trend_bucket_variant_id(trend_score),
                "display_name": f"trend_score={trend_score:g} valid-best",
                "trend_score": trend_score,
                "grid_index": int(best["grid_index"]),
                "base_method": "pop_similarity_trend",
                "candidate_strategy": "default",
                "weight_policy": "trend_bucket_best_by_valid_ndcg_at_12",
                "selection_split": "valid",
                "metrics_source": "in_memory_evaluation",
                "weights": _read_weights(dict(best["weights"]), context="trend bucket weights"),
                "metrics": {"valid": dict(best["valid_metrics"])},
            }
        )
    return selected


def _trend_bucket_variant_id(trend_score: float) -> str:
    return f"trend_bucket_{str(trend_score).replace('.', '_')}"
```

- [ ] **Step 4: Extend payload builder signature**

Change `build_experiment_payload()` in `src/fashion_trend/recommendation/experiments/runner.py` to accept and include the new fields:

```python
def build_experiment_payload(
    experiment_id: str,
    baseline_payloads: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    trend_payload: dict[str, Any],
    stage_status: list[dict[str, Any]] | None = None,
    force: dict[str, object] | None = None,
    timings: list[dict[str, Any]] | None = None,
    named_ablation: list[dict[str, Any]] | None = None,
    trend_bucket_best_by_valid: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "experiment_path": str(experiment_dir(experiment_id) / "experiment.json"),
        "best_weights": select_best_weights(search_results),
        "search_results": search_results,
        "ablation": build_ablation_summary([*baseline_payloads, trend_payload]),
        "named_ablation": list(named_ablation or []),
        "trend_bucket_best_by_valid": list(trend_bucket_best_by_valid or []),
        "stage_status": list(stage_status or []),
        "force": dict(force or {}),
        "timings": list(timings or []),
    }
```

- [ ] **Step 5: Add in-memory evaluation helper for one weight set**

In `src/fashion_trend/recommendation/experiments/runner.py`, add:

```python
def evaluate_weight_variant_by_split(
    *,
    weights: dict[str, float],
    context: RecommendationExperimentContext,
    inputs: RecommendationInputArtifacts,
    candidates: pd.DataFrame,
    force: bool = False,
) -> dict[str, dict[str, float]]:
    metrics_by_split: dict[str, dict[str, float]] = {}
    for split in ("valid", "test"):
        result = build_recommendation_result_in_memory(
            method_name=TREND_METHOD,
            weights=weights,
            split_filter=split,
            context=context,
            inputs=inputs,
            candidates=candidates,
        )
        payload = evaluate_result_for_experiment(
            TREND_METHOD,
            result,
            context,
            inputs,
            force=force,
        )
        metrics_by_split[split] = dict(payload["metrics"][split])
    return metrics_by_split
```

- [ ] **Step 6: Build named ablation and trend buckets in the runner**

After `trend_payload = _publish_or_reuse_trend_method(...)` in `run_recommendation_experiment()`, compute:

```python
from fashion_trend.recommendation.experiments.ablation import (
    build_named_ablation_rows,
    derive_strict_ablation_weights,
    select_trend_bucket_representatives,
)

strict_metrics = {
    variant_id: evaluate_weight_variant_by_split(
        weights=derive_strict_ablation_weights(best_weights, dropped_feature),
        context=context,
        inputs=inputs,
        candidates=default_candidates,
        force=force_cache or force_rebuild_all,
    )
    for variant_id, _display_name, dropped_feature in STRICT_VARIANTS
}
named_ablation = build_named_ablation_rows(
    best_weights=best_weights,
    strict_metrics=strict_metrics,
    full_model_metrics=dict(trend_payload["metrics"]),
    stable_baseline_metrics=_baseline_metrics_for_named_ablation(baseline_payloads),
)
trend_bucket_best_by_valid = []
for row in select_trend_bucket_representatives(search_results):
    metrics = evaluate_weight_variant_by_split(
        weights=dict(row["weights"]),
        context=context,
        inputs=inputs,
        candidates=default_candidates,
        force=force_cache or force_rebuild_all,
    )
    trend_bucket_best_by_valid.append({**row, "metrics": metrics})
```

Add a private helper:

```python
def _baseline_metrics_for_named_ablation(
    baseline_payloads: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_method = {str(payload["method"]): payload for payload in baseline_payloads}
    return {
        "recent_only_baseline": {
            "method": "recent_popularity",
            "display_name": "Recent Only",
            "metrics": dict(by_method["recent_popularity"]["metrics"]),
        },
        "pop_similarity_baseline": {
            "method": "pop_similarity",
            "display_name": "Pop + Similarity baseline",
            "metrics": dict(by_method["pop_similarity"]["metrics"]),
        },
    }
```

Pass both fields into `build_experiment_payload(...)`.

- [ ] **Step 7: Run focused experiment tests**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py -q
```

Expected: all recommendation experiment tests pass.

- [ ] **Step 8: Review and commit**

Run:

```sh
git diff -- src/fashion_trend/recommendation/experiments/ablation.py src/fashion_trend/recommendation/experiments/runner.py tests/test_recommendation_experiments.py
git diff --check
```

Confirm no new method registry entries and no new output directories are introduced. Then commit:

```sh
git add src/fashion_trend/recommendation/experiments/ablation.py src/fashion_trend/recommendation/experiments/runner.py tests/test_recommendation_experiments.py
git commit -m "feat(recommendation): 输出命名消融实验结果"
```

---

### Task 3: Reports Flattening and Warning Semantics

**Files:**
- Modify: `src/fashion_trend/reports/runner.py`
- Modify: `tests/test_reports_runner.py`

- [ ] **Step 1: Write failing reports tests**

In `tests/test_reports_runner.py`, add:

```python
def test_flatten_recommendation_experiment_rows_includes_named_ablation_and_trend_buckets() -> None:
    from fashion_trend.reports import runner

    payload = {
        "best_weights": {"pop_score": 0.2, "sim_score": 0.2, "trend_score": 0.1, "recent_score": 0.5},
        "search_results": [],
        "ablation": [],
        "named_ablation": [
            {
                "variant_id": "without_recent",
                "display_name": "w/o Recent",
                "weights": {"pop_score": 0.4, "sim_score": 0.4, "trend_score": 0.2, "recent_score": 0.0},
                "metrics": {
                    "valid": {"map_at_12": 0.1, "recall_at_12": 0.2, "hit_rate_at_12": 0.3, "ndcg_at_12": 0.4, "coverage": 0.5},
                    "test": {"map_at_12": 0.6, "recall_at_12": 0.7, "hit_rate_at_12": 0.8, "ndcg_at_12": 0.9, "coverage": 1.0},
                },
            }
        ],
        "trend_bucket_best_by_valid": [
            {
                "variant_id": "trend_bucket_0_1",
                "display_name": "trend_score=0.1 valid-best",
                "weights": {"pop_score": 0.2, "sim_score": 0.2, "trend_score": 0.1, "recent_score": 0.5},
                "metrics": {
                    "valid": {"map_at_12": 0.11, "recall_at_12": 0.12, "hit_rate_at_12": 0.13, "ndcg_at_12": 0.14, "coverage": 0.15},
                    "test": {"map_at_12": 0.21, "recall_at_12": 0.22, "hit_rate_at_12": 0.23, "ndcg_at_12": 0.24, "coverage": 0.25},
                },
            }
        ],
    }

    rows = runner.flatten_recommendation_experiment_rows(payload)

    assert [row["section"] for row in rows] == [
        "named_ablation",
        "named_ablation",
        "trend_bucket_best_by_valid",
        "trend_bucket_best_by_valid",
    ]
    assert rows[0]["method"] == "w/o Recent"
    assert rows[0]["split"] == "valid"
    assert rows[1]["split"] == "test"
    assert rows[2]["method"] == "trend_score=0.1 valid-best"
```

Update the existing manifest warning test to include a strict `named_ablation` `w/o Recent` row and assert the warning is absent.

- [ ] **Step 2: Run reports tests to confirm failure**

Run:

```sh
uv run pytest tests/test_reports_runner.py::test_flatten_recommendation_experiment_rows_includes_named_ablation_and_trend_buckets -q
```

Expected: fails because reports only flattens `search_results` and `ablation`.

- [ ] **Step 3: Implement flattening for nested metrics**

In `src/fashion_trend/reports/runner.py`, extend `flatten_recommendation_experiment_rows()`:

```python
    rows.extend(
        _flatten_named_experiment_rows(
            payload.get("named_ablation", []),
            section="named_ablation",
        )
    )
    rows.extend(
        _flatten_named_experiment_rows(
            payload.get("trend_bucket_best_by_valid", []),
            section="trend_bucket_best_by_valid",
        )
    )
```

Add helper:

```python
def _flatten_named_experiment_rows(
    values: object,
    *,
    section: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if values is None:
        return rows
    if not isinstance(values, list):
        raise ValueError(f"{section} 必须是列表")
    for rank, result in enumerate(values, start=1):
        if not isinstance(result, dict):
            raise ValueError(f"{section}[{rank - 1}] 必须是对象")
        weights = _required_mapping(result, "weights", f"{section}[{rank - 1}]")
        metrics_by_split = _required_mapping(result, "metrics", f"{section}[{rank - 1}]")
        display_name = str(result.get("display_name") or result.get("variant_id") or "")
        for split in ("valid", "test"):
            metrics = _required_mapping(
                metrics_by_split,
                split,
                f"{section}[{rank - 1}].metrics",
            )
            rows.append(
                _recommendation_experiment_row(
                    section=section,
                    rank=rank,
                    method=display_name,
                    split=split,
                    weights=weights,
                    metrics=metrics,
                    blank_missing_weights=False,
                )
            )
    return rows
```

Use existing `_required_mapping()` and `_recommendation_experiment_row()`.

- [ ] **Step 4: Update strict w/o Recent warning detection**

Change `_has_strict_without_recent_ablation()` to check `named_ablation` first:

```python
def _has_strict_without_recent_ablation(payload: dict[str, object]) -> bool:
    for row in payload.get("named_ablation", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("display_name", "")).lower() != "w/o recent":
            continue
        if row.get("weight_policy") != "strict_drop_and_renormalize_from_full":
            continue
        weights = row.get("weights")
        metrics = row.get("metrics")
        if not isinstance(weights, dict) or not isinstance(metrics, dict):
            continue
        try:
            recent_score = float(weights.get("recent_score", -1.0))
        except (TypeError, ValueError):
            continue
        if recent_score == 0.0 and {"valid", "test"} <= set(metrics):
            return True
    return False
```

Do not treat a bare `recent_score=0` in `search_results` as strict w/o Recent.

- [ ] **Step 5: Run focused reports tests**

Run:

```sh
uv run pytest tests/test_reports_runner.py tests/test_reports_tables.py -q
```

Expected: reports runner and tables tests pass.

- [ ] **Step 6: Review and commit**

Run:

```sh
git diff -- src/fashion_trend/reports/runner.py tests/test_reports_runner.py
git diff --check
```

Confirm `reports` still does not import recommendation experiment runner. Then commit:

```sh
git add src/fashion_trend/reports/runner.py tests/test_reports_runner.py
git commit -m "feat(reports): 展开推荐命名消融表格"
```

---

### Task 4: Documentation Sync

**Files:**
- Modify: `README.md`
- Modify: `docs/gpt-research/implementation-plan.md`

- [ ] **Step 1: Update README recommendation section**

In `README.md` section `### 14. 轻量离线推荐实验`, add a concise paragraph after the current tuned weight summary:

```markdown
`main` 实验还会写出 `named_ablation` 和 `trend_bucket_best_by_valid`：

- `named_ablation` 包含 `Full Model`、严格 `w/o Trend in Rec`、严格 `w/o Similarity`、严格 `w/o Recent`、`Recent Only` 和 `Pop + Similarity baseline`。严格消融权重从当次 `best_weights` 动态派生，不注册为新的正式 method，也不写入新的 stable method 目录。
- `trend_bucket_best_by_valid` 记录 `trend_score=0.0/0.1/0.2/0.3/0.4` 下按 valid `NDCG@12` 选出的代表组合及其 valid/test 指标；它不是单因素 controlled sweep。
```

- [ ] **Step 2: Update implementation plan ablation section**

In `docs/gpt-research/implementation-plan.md` section `18.3 消融实验`, add:

```markdown
当前实现把推荐层消融与趋势模型特征消融分开处理。已实现的推荐层消融包括 Full Model、严格 `w/o Trend in Rec`、严格 `w/o Similarity`、严格 `w/o Recent` 和稳定 baseline 对照。`w/o Graph`、`w/o Growth`、`w/o Rank` 属于趋势模型特征消融，需要重训 LightGBM 变体，不属于本轮推荐消融实现。
```

- [ ] **Step 3: Run documentation checks**

Run:

```sh
rg -n "w/o Trend|w/o Graph|trend_bucket_best_by_valid|named_ablation" README.md docs/gpt-research/implementation-plan.md
git diff --check
```

Expected: grep output shows the new text; diff check passes.

- [ ] **Step 4: Review and commit**

Run:

```sh
git diff -- README.md docs/gpt-research/implementation-plan.md
```

Confirm docs do not claim trend-model feature ablations are implemented. Then commit:

```sh
git add README.md docs/gpt-research/implementation-plan.md
git commit -m "docs: 同步推荐消融实验说明"
```

---

### Task 5: Focused Verification and Code Review

**Files:**
- No planned source edits unless review finds a defect.

- [ ] **Step 1: Run focused verification**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py tests/test_reports_runner.py tests/test_reports_tables.py tests/test_architecture_boundaries.py
uv run python -m compileall -q src
```

Expected: all tests pass and compileall returns no output.

- [ ] **Step 2: Review staged implementation diff**

Run:

```sh
git diff HEAD
rg -n "named_ablation|trend_bucket_best_by_valid|strict_drop_and_renormalize_from_full|not_applicable" src tests README.md docs/gpt-research/implementation-plan.md
```

Review checklist:

- `named_ablation` is produced from current `best_weights`, not hardcoded constants.
- `stable_method_baseline` rows use `selection_split=not_applicable`.
- No new recommendation method names are registered.
- No code path writes new ablation-specific stable directories such as `outputs/recommendation/without_recent/`.
- `reports` still only reads experiment artifact fields and does not import experiment runner.

- [ ] **Step 3: Commit review fixes when review finds defects**

If review finds code or docs defects, fix them in the relevant files from this plan and commit:

```sh
git add src/fashion_trend/recommendation/experiments/ablation.py src/fashion_trend/recommendation/experiments/runner.py src/fashion_trend/reports/runner.py tests/test_recommendation_experiments.py tests/test_reports_runner.py README.md docs/gpt-research/implementation-plan.md
git commit -m "fix(recommendation): 修正消融实验审查问题"
```

If review finds no defects, do not create an empty commit.

---

### Task 6: Real Artifact Verification

**Files:**
- Generated outputs are not committed.
- Modify: `docs/gpt-research/project-status-summary.md` only after real metrics are available.

- [ ] **Step 1: Rebuild experiment payload**

Run:

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main --force-experiment
```

Expected: command succeeds and rewrites `outputs/recommendation/experiments/main/experiment.json`. If freshness or cache changes make the command fail, rerun with:

```sh
uv run python src/16_run_recommendation_experiment.py --experiment main --force-rebuild-all
```

- [ ] **Step 2: Inspect experiment artifact**

Run:

```sh
jq '{named_ablation_count:(.named_ablation|length), trend_bucket_count:(.trend_bucket_best_by_valid|length), named_ablation:.named_ablation[].variant_id, trend_buckets:[.trend_bucket_best_by_valid[].trend_score]}' outputs/recommendation/experiments/main/experiment.json
```

Expected:

```text
named_ablation_count: 6
trend_bucket_count: 5
named_ablation includes full_model, without_trend_in_rec, without_similarity, without_recent, recent_only_baseline, pop_similarity_baseline
trend_buckets includes 0, 0.1, 0.2, 0.3, 0.4
```

- [ ] **Step 3: Re-export reports**

Run:

```sh
uv run python src/17_export_paper_assets.py
```

Expected: command succeeds and rewrites `outputs/reports/`.

- [ ] **Step 4: Inspect reports outputs**

Run:

```sh
rg -n "w/o Recent|trend_bucket_best_by_valid|named_ablation" outputs/reports/tables/recommendation_experiment_summary.md
jq '.warnings' outputs/reports/manifest.json
```

Expected: table contains `w/o Recent` and `trend_bucket_best_by_valid`; manifest warnings do not include “缺少严格 w/o Recent 消融行”. The valid-only grid warning may remain.

- [ ] **Step 5: Update status summary with real metrics**

If real artifact verification succeeds, update `docs/gpt-research/project-status-summary.md` section `6.5 消融实验与权重搜索`:

- Replace the “当前缺失” statement for strict `w/o Recent` with the actual verified result.
- Add a short note that `trend_bucket_best_by_valid` is valid-selected representative analysis, not a single-factor sweep.
- Keep the cautious recommendation conclusion about strong `recent_popularity`.

- [ ] **Step 6: Review and commit status docs**

Run:

```sh
git diff -- docs/gpt-research/project-status-summary.md
git diff --check
```

Confirm generated outputs under `outputs/` are not staged. Then commit:

```sh
git add docs/gpt-research/project-status-summary.md
git commit -m "docs: 更新推荐消融实验结果"
```

---

### Task 7: Completion Audit

**Files:**
- No planned edits unless audit finds a gap.

- [ ] **Step 1: Run final verification**

Run:

```sh
uv run pytest tests/test_recommendation_experiments.py tests/test_reports_runner.py tests/test_reports_tables.py tests/test_architecture_boundaries.py
uv run python -m compileall -q src
git status --short
```

Expected:

- Tests pass.
- Compileall returns no output.
- Only expected untracked user file may remain: `docs/gpt-research/defense-stable-roadmap.md`.
- No generated `outputs/` files are staged.

- [ ] **Step 2: Prompt-to-artifact audit**

Verify each objective item:

| Requirement | Evidence |
| --- | --- |
| Current design doc committed | `git log --oneline` contains `a202432 docs: 明确消融权重派生语义` or later doc commits. |
| Implementation follows spec | `experiment.json` contains `named_ablation` and `trend_bucket_best_by_valid` with required metadata. |
| Every step proactively committed | `git log --oneline` shows separate plan, helper, runner, reports, docs/status commits. |
| Review after stages | Each task includes diff review and focused verification before commit. |
| No ignored issues | Completion audit checks actual artifacts, tests, docs, and untracked/staged state. |
| Correct execution mode | Inline execution used because implementation is tightly coupled. |

- [ ] **Step 3: Final response**

Summarize:

- Commits created.
- Files changed.
- Verification commands and results.
- Real artifact paths inspected.
- Remaining risk discovered by the completion audit, or state that no unresolved implementation risk remains.
