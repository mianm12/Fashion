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
        elif isinstance(node, ast.ImportFrom):
            modules.update(imported_from_modules(path, node))
    return modules


def imported_from_modules(path: Path, node: ast.ImportFrom) -> set[str]:
    base_module = import_from_base_module(path, node)
    modules: set[str] = set()
    if base_module:
        modules.add(base_module)
    for alias in node.names:
        if alias.name == "*":
            continue
        modules.add(f"{base_module}.{alias.name}" if base_module else alias.name)
    return modules


def import_from_base_module(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = package_parts_for_path(path)
    parent_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        parent_parts.extend(node.module.split("."))
    return ".".join(parent_parts)


def package_parts_for_path(path: Path) -> list[str]:
    package_root = package_root_for_path(path)
    relative_module = path.relative_to(package_root.parent).with_suffix("")
    parts = list(relative_module.parts)
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def package_root_for_path(path: Path) -> Path:
    try:
        path.relative_to(PACKAGE_ROOT)
    except ValueError:
        parts = path.resolve().parts
        package_indexes = [
            index for index, part in enumerate(parts) if part == "fashion_trend"
        ]
        assert package_indexes, f"path is not inside a fashion_trend package: {path}"
        return Path(*parts[: package_indexes[-1] + 1])
    return PACKAGE_ROOT


def test_imported_modules_resolves_absolute_from_import_aliases(tmp_path) -> None:
    module_path = tmp_path / "example.py"
    module_path.write_text(
        "\n".join(
            [
                "from fashion_trend import catalog",
                "from fashion_trend.trend import models",
                "import fashion_trend.catalog.graph",
            ]
        ),
        encoding="utf-8",
    )

    assert imported_modules(module_path) >= {
        "fashion_trend.catalog",
        "fashion_trend.trend.models",
        "fashion_trend.catalog.graph",
    }


def test_imported_modules_resolves_relative_import_aliases(tmp_path) -> None:
    package_root = tmp_path / "src" / "fashion_trend"
    module_path = package_root / "recommendation" / "ranker.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "\n".join(
            [
                "from .. import catalog",
                "from ..trend import models",
            ]
        ),
        encoding="utf-8",
    )

    assert imported_modules(module_path) >= {
        "fashion_trend.catalog",
        "fashion_trend.trend.models",
    }


def test_imported_modules_resolves_init_relative_parent_alias(tmp_path) -> None:
    package_root = tmp_path / "src" / "fashion_trend"
    module_path = package_root / "recommendation" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("from .. import catalog\n", encoding="utf-8")

    assert "fashion_trend.catalog" in imported_modules(module_path)


def test_imported_modules_resolves_init_relative_sibling_alias(tmp_path) -> None:
    package_root = tmp_path / "src" / "fashion_trend"
    module_path = package_root / "recommendation" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("from ..trend import models\n", encoding="utf-8")

    assert "fashion_trend.trend.models" in imported_modules(module_path)


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
        {"fashion_trend.models", "fashion_trend.trend.models"},
    )


def test_historical_root_modules_are_removed() -> None:
    existing = sorted(
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_file() and path.name in HISTORICAL_ROOT_MODULES
    )
    assert existing == []
