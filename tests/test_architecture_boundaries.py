from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "fashion_trend"
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
PACKAGE_NAME = PACKAGE_ROOT.name

BUSINESS_DOMAINS = {
    "datasets",
    "transactions",
    "catalog",
    "trend",
    "recommendation",
    "reports",
}

HISTORICAL_ROOT_MODULE_NAMES = {
    "articles",
    "config",
    "data_loader",
    "evaluation",
    "log",
    "training",
}

HISTORICAL_ROOT_MODULES = {
    f"{module_name}.py" for module_name in HISTORICAL_ROOT_MODULE_NAMES
}

HISTORICAL_ROOT_PACKAGES = {
    "models",
}

HISTORICAL_ROOT_IMPORTS = {
    f"{PACKAGE_NAME}.{module_name}"
    for module_name in HISTORICAL_ROOT_MODULE_NAMES | HISTORICAL_ROOT_PACKAGES
}

RECOMMENDATION_PUBLIC_UPSTREAM_IMPORTS = {
    "fashion_trend.transactions.contracts",
    "fashion_trend.transactions.readers",
    "fashion_trend.catalog.contracts",
    "fashion_trend.catalog.readers",
    "fashion_trend.trend.schema",
    "fashion_trend.trend.predictions",
    "fashion_trend.trend.readers",
}

REPORTS_PUBLIC_IMPORTS = {
    "fashion_trend.transactions.contracts",
    "fashion_trend.transactions.readers",
    "fashion_trend.catalog.contracts",
    "fashion_trend.catalog.readers",
    "fashion_trend.trend.schema",
    "fashion_trend.trend.predictions",
    "fashion_trend.trend.readers",
    "fashion_trend.recommendation.contracts",
    "fashion_trend.recommendation.readers",
}

FOUNDATION_PATH_ALLOWED_EXPORTS = {
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "OUTPUT_DIR",
}


def iter_python_files(package_name: str) -> list[Path]:
    package_path = PACKAGE_ROOT / package_name
    assert package_path.exists(), f"package missing: fashion_trend.{package_name}"
    return sorted(
        path for path in package_path.rglob("*.py") if "__pycache__" not in path.parts
    )


def iter_architecture_python_files() -> list[Path]:
    return sorted(
        path
        for root in (PACKAGE_ROOT, TESTS_ROOT)
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def imported_modules(path: Path) -> set[str]:
    # These architecture checks intentionally cover static import statements only.
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
                f"from {PACKAGE_NAME}.trend import models",
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


def test_trend_facade_import_offenders_detects_absolute_from_imports(tmp_path) -> None:
    module_path = tmp_path / "example.py"
    trend_package = f"{PACKAGE_NAME}.trend"
    module_path.write_text(
        f"from {trend_package} import article_sales\n", encoding="utf-8"
    )

    assert trend_facade_import_offenders([module_path]) == [
        f"{module_path}: {trend_package}.article_sales"
    ]


def trend_facade_import_offenders(paths: list[Path]) -> list[str]:
    trend_package = f"{PACKAGE_NAME}.trend"
    offenders: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 0 or node.module != trend_package:
                continue
            for alias in node.names:
                imported_name = "*" if alias.name == "*" else alias.name
                offenders.append(f"{path}: {trend_package}.{imported_name}")
    return offenders


def test_no_trend_package_facade_from_imports() -> None:
    offenders = trend_facade_import_offenders(iter_architecture_python_files())
    assert offenders == []


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


def package_upstream_import_offenders(
    paths: list[Path],
    upstream_roots: set[str],
    allowed_modules: set[str],
) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        for module_name in sorted(imported_modules(path)):
            matched_root = next(
                (
                    root
                    for root in upstream_roots
                    if module_name == root or module_name.startswith(root + ".")
                ),
                None,
            )
            if matched_root is None:
                continue
            is_allowed = any(
                module_name == allowed or module_name.startswith(allowed + ".")
                for allowed in allowed_modules
            )
            if not is_allowed:
                try:
                    display_path = path.relative_to(PACKAGE_ROOT.parents[0])
                except ValueError:
                    display_path = path
                offenders.append(f"{display_path}: {module_name}")
    return offenders


def assert_package_imports_only_allowed_upstream(
    package_name: str,
    upstream_roots: set[str],
    allowed_modules: set[str],
) -> None:
    offenders = package_upstream_import_offenders(
        iter_python_files(package_name),
        upstream_roots,
        allowed_modules,
    )
    assert offenders == []


def test_allowlist_rejects_recommendation_importing_catalog_graph(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "src" / "fashion_trend"
    module_path = package_root / "recommendation" / "ranker.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "from fashion_trend.catalog.graph import read_attribute_nodes\n",
        encoding="utf-8",
    )

    offenders = package_upstream_import_offenders(
        [module_path],
        {"fashion_trend.catalog"},
        {"fashion_trend.catalog.readers"},
    )

    assert offenders == [
        f"{module_path}: fashion_trend.catalog.graph",
        f"{module_path}: fashion_trend.catalog.graph.read_attribute_nodes",
    ]


def test_allowlist_rejects_core_computation_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "fashion_trend"
    module_path = package_root / "reports" / "summary.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "\n".join(
            [
                "from fashion_trend.transactions.weekly import build_weekly_transactions",
                "from fashion_trend.catalog.graph.builders import build_attribute_nodes",
                "from fashion_trend.trend.features.samples import build_trend_model_samples_frame",
                "from fashion_trend.trend.training import run_trend_model_training",
                "from fashion_trend.trend.evaluation import run_trend_model_evaluation",
            ]
        ),
        encoding="utf-8",
    )

    offenders = package_upstream_import_offenders(
        [module_path],
        {
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
        },
        REPORTS_PUBLIC_IMPORTS,
    )

    assert offenders == [
        f"{module_path}: fashion_trend.catalog.graph.builders",
        f"{module_path}: fashion_trend.catalog.graph.builders.build_attribute_nodes",
        f"{module_path}: fashion_trend.transactions.weekly",
        f"{module_path}: fashion_trend.transactions.weekly.build_weekly_transactions",
        f"{module_path}: fashion_trend.trend.evaluation",
        f"{module_path}: fashion_trend.trend.evaluation.run_trend_model_evaluation",
        f"{module_path}: fashion_trend.trend.features.samples",
        f"{module_path}: fashion_trend.trend.features.samples.build_trend_model_samples_frame",
        f"{module_path}: fashion_trend.trend.training",
        f"{module_path}: fashion_trend.trend.training.run_trend_model_training",
    ]


def forbidden_imports(*module_names: str) -> set[str]:
    return HISTORICAL_ROOT_IMPORTS | set(module_names)


def test_foundation_has_no_business_domain_imports() -> None:
    forbidden = {f"fashion_trend.{name}" for name in BUSINESS_DOMAINS}
    assert_package_does_not_import("foundation", HISTORICAL_ROOT_IMPORTS | forbidden)


def test_foundation_paths_exports_only_project_roots() -> None:
    import fashion_trend.foundation.paths as paths

    exported_names = {
        name for name in vars(paths) if name.isupper() and not name.startswith("_")
    }
    assert exported_names == FOUNDATION_PATH_ALLOWED_EXPORTS


def test_datasets_depends_only_on_foundation() -> None:
    assert_package_does_not_import(
        "datasets",
        forbidden_imports(
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
            "fashion_trend.recommendation",
            "fashion_trend.reports",
        ),
    )


def test_catalog_depends_only_on_foundation() -> None:
    assert_package_does_not_import(
        "catalog",
        forbidden_imports(
            "fashion_trend.datasets",
            "fashion_trend.transactions",
            "fashion_trend.trend",
            "fashion_trend.recommendation",
            "fashion_trend.reports",
        ),
    )


def test_transactions_depends_only_on_foundation() -> None:
    assert_package_does_not_import(
        "transactions",
        forbidden_imports(
            "fashion_trend.datasets",
            "fashion_trend.catalog",
            "fashion_trend.trend",
            "fashion_trend.recommendation",
            "fashion_trend.reports",
        ),
    )


def test_trend_depends_only_on_stable_input_domains() -> None:
    assert_package_does_not_import(
        "trend",
        forbidden_imports(
            "fashion_trend.datasets",
            "fashion_trend.recommendation",
            "fashion_trend.reports",
        ),
    )


def test_recommendation_imports_only_public_upstream_surfaces() -> None:
    assert_package_imports_only_allowed_upstream(
        "recommendation",
        {
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
        },
        RECOMMENDATION_PUBLIC_UPSTREAM_IMPORTS,
    )


def test_reports_imports_only_public_read_only_surfaces() -> None:
    assert_package_imports_only_allowed_upstream(
        "reports",
        {
            "fashion_trend.datasets",
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
            "fashion_trend.recommendation",
        },
        REPORTS_PUBLIC_IMPORTS,
    )


def test_historical_root_modules_are_removed() -> None:
    existing = sorted(
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_file() and path.name in HISTORICAL_ROOT_MODULES
    )
    assert existing == []


def test_historical_root_packages_are_removed() -> None:
    existing = sorted(
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name in HISTORICAL_ROOT_PACKAGES
    )
    assert existing == []
