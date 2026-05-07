from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "fashion_trend"

BUSINESS_DOMAINS = {
    "datasets",
    "transactions",
    "catalog",
    "trend",
    "recommendation",
    "reports",
}

HISTORICAL_ROOT_MODULES = {
    "articles.py",
    "config.py",
    "data_loader.py",
    "evaluation.py",
    "log.py",
    "training.py",
}


def iter_python_files(package_name: str) -> list[Path]:
    package_path = PACKAGE_ROOT / package_name
    assert package_path.exists(), f"package missing: fashion_trend.{package_name}"
    return sorted(
        path
        for path in package_path.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def assert_package_does_not_import(
    package_name: str,
    forbidden_modules: set[str],
) -> None:
    offenders: list[str] = []
    for path in iter_python_files(package_name):
        for module_name in imported_modules(path):
            for forbidden in forbidden_modules:
                if module_name == forbidden or module_name.startswith(forbidden + "."):
                    relative_path = path.relative_to(PACKAGE_ROOT.parents[0])
                    offenders.append(f"{relative_path}: {module_name}")
    assert not offenders, "\n".join(offenders)


def test_foundation_has_no_business_domain_imports() -> None:
    forbidden = {f"fashion_trend.{name}" for name in BUSINESS_DOMAINS}
    assert_package_does_not_import("foundation", forbidden)


def test_catalog_does_not_depend_on_trend_or_recommendation() -> None:
    assert_package_does_not_import(
        "catalog",
        {"fashion_trend.trend", "fashion_trend.recommendation"},
    )


def test_transactions_does_not_depend_on_catalog_trend_or_recommendation() -> None:
    assert_package_does_not_import(
        "transactions",
        {
            "fashion_trend.catalog",
            "fashion_trend.trend",
            "fashion_trend.recommendation",
        },
    )


def test_trend_does_not_depend_on_recommendation_or_reports() -> None:
    assert_package_does_not_import(
        "trend",
        {"fashion_trend.recommendation", "fashion_trend.reports"},
    )


def test_recommendation_does_not_depend_on_trend_model_internals() -> None:
    assert_package_does_not_import(
        "recommendation",
        {"fashion_trend.trend.models"},
    )


def test_historical_root_modules_are_removed() -> None:
    existing = sorted(
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_file() and path.name in HISTORICAL_ROOT_MODULES
    )
    assert existing == []
