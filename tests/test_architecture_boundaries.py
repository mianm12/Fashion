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
    "presentation",
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

PRESENTATION_PUBLIC_IMPORTS = {
    "fashion_trend.transactions.contracts",
    "fashion_trend.transactions.readers",
    "fashion_trend.catalog.contracts",
    "fashion_trend.catalog.readers",
    "fashion_trend.trend.schema",
    "fashion_trend.trend.predictions",
    "fashion_trend.trend.readers",
    "fashion_trend.recommendation.contracts",
    "fashion_trend.recommendation.readers",
    "fashion_trend.reports.loaders",
    "fashion_trend.reports.paths",
}

FOUNDATION_PATH_ALLOWED_EXPORTS = {
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "OUTPUT_DIR",
}

RAW_PATH_MODULE_IMPORTS = {
    "fashion_trend.datasets.paths",
    "fashion_trend.foundation.paths",
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
    """解析文件中的静态 import 语句并返回完整模块候选集合。

    该 helper 只覆盖 AST 中的 import/from import 语句；动态 import 不属于
    架构边界测试目标。from import 会同时记录基础模块和别名展开后的模块名，
    供后续白名单与前缀匹配使用。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.update(imported_from_modules(path, node))
    return modules


def imported_from_modules(path: Path, node: ast.ImportFrom) -> set[str]:
    """展开 from import 节点的基础模块和具名导入模块。

    解析绝对导入和相对导入后，返回可用于架构白名单检查的模块集合；
    星号导入只保留基础模块，避免生成无法静态定位的成员路径。
    """
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
    """把 from import 节点解析为绝对基础模块名。"""
    if node.level == 0:
        return node.module or ""
    package_parts = package_parts_for_path(path)
    parent_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        parent_parts.extend(node.module.split("."))
    return ".".join(parent_parts)


def package_parts_for_path(path: Path) -> list[str]:
    """根据文件路径推导所属 Python 包路径片段。"""
    package_root = package_root_for_path(path)
    relative_module = path.relative_to(package_root.parent).with_suffix("")
    parts = list(relative_module.parts)
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def pandas_read_csv_call_locations(paths: list[Path]) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "read_csv":
                continue
            value = func.value
            if isinstance(value, ast.Name) and value.id == "pd":
                try:
                    display_path = path.relative_to(PACKAGE_ROOT.parents[0])
                except ValueError:
                    display_path = path
                offenders.append(f"{display_path}:{node.lineno}: pd.read_csv")
    return offenders


def package_root_for_path(path: Path) -> Path:
    """定位文件路径对应的 fashion_trend 包根目录。"""
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
    """找出仍从 trend facade 直接导入子模块的违规位置。

    只检查 `from fashion_trend.trend import ...` 形式，按 AST 静态解析展开
    导入别名，并把每个违规项格式化为路径加模块名，便于断言输出精确定位。
    """
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
    """找出跨上游业务域导入中未命中白名单的模块。

    先用 `upstream_roots` 做根模块前缀匹配，筛出需要检查的上游导入；
    再用 `allowed_modules` 做精确或子模块前缀白名单匹配，剩余项会保留
    相对路径和模块名，供边界测试输出可读的违规列表。
    """
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
            "fashion_trend.presentation",
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
            "fashion_trend.presentation",
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
            "fashion_trend.presentation",
        ),
    )


def test_trend_depends_only_on_stable_input_domains() -> None:
    assert_package_does_not_import(
        "trend",
        forbidden_imports(
            "fashion_trend.datasets",
            "fashion_trend.recommendation",
            "fashion_trend.reports",
            "fashion_trend.presentation",
        ),
    )


def test_recommendation_imports_only_public_upstream_surfaces() -> None:
    assert_package_imports_only_allowed_upstream(
        "recommendation",
        {
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
            "fashion_trend.presentation",
        },
        RECOMMENDATION_PUBLIC_UPSTREAM_IMPORTS,
    )


def test_recommendation_package_does_not_import_datasets_paths() -> None:
    assert_package_does_not_import(
        "recommendation",
        {"fashion_trend.datasets", "fashion_trend.datasets.paths"},
    )


def test_retrieval_and_ranking_do_not_read_raw_csv_paths() -> None:
    paths = [
        *iter_python_files("recommendation/retrieval"),
        *iter_python_files("recommendation/ranking"),
    ]
    import_offenders = package_upstream_import_offenders(
        paths,
        RAW_PATH_MODULE_IMPORTS,
        allowed_modules=set(),
    )
    read_csv_offenders = pandas_read_csv_call_locations(paths)

    assert import_offenders == []
    assert read_csv_offenders == []


def test_no_recomd_imports_anywhere() -> None:
    offenders: list[str] = []
    for path in iter_architecture_python_files():
        for module_name in imported_modules(path):
            if module_name == "recomd" or module_name.startswith("recomd."):
                try:
                    display_path = path.relative_to(PACKAGE_ROOT.parents[0])
                except ValueError:
                    display_path = path
                offenders.append(f"{display_path}: {module_name}")

    assert offenders == []


def test_recommendation_does_not_import_trend_training_or_models() -> None:
    assert_package_does_not_import(
        "recommendation",
        {
            "fashion_trend.trend.training",
            "fashion_trend.trend.evaluation.runner",
            "fashion_trend.trend.models",
            "fashion_trend.catalog.graph.builders",
            "fashion_trend.presentation",
        },
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
            "fashion_trend.presentation",
        },
        REPORTS_PUBLIC_IMPORTS,
    )


def test_presentation_is_tracked_as_business_domain() -> None:
    assert "presentation" in BUSINESS_DOMAINS


def test_presentation_imports_only_public_read_only_surfaces() -> None:
    assert_package_imports_only_allowed_upstream(
        "presentation",
        {
            "fashion_trend.datasets",
            "fashion_trend.transactions",
            "fashion_trend.catalog",
            "fashion_trend.trend",
            "fashion_trend.recommendation",
            "fashion_trend.reports",
        },
        PRESENTATION_PUBLIC_IMPORTS,
    )


def test_presentation_does_not_import_core_runners_or_builders() -> None:
    assert_package_does_not_import(
        "presentation",
        {
            "fashion_trend.datasets",
            "fashion_trend.transactions.weekly",
            "fashion_trend.catalog.graph.builders",
            "fashion_trend.trend.training",
            "fashion_trend.trend.models",
            "fashion_trend.trend.evaluation.runner",
            "fashion_trend.recommendation.runner",
            "fashion_trend.recommendation.retrieval",
            "fashion_trend.recommendation.ranking",
            "fashion_trend.recommendation.experiments",
            "fashion_trend.reports.runner",
        },
    )


def test_trend_graph_feature_ablation_19_is_not_in_default_readme_pipeline() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    marker = "当前已实现流水线可按下面顺序"
    start = readme.index(marker)
    block_start = readme.index("```sh", start)
    block_end = readme.index("```", block_start + 1)
    default_pipeline_block = readme[block_start:block_end]

    assert "src/18_build_defense_app_db.py" in default_pipeline_block
    assert "src/19_run_trend_graph_feature_ablation.py" not in default_pipeline_block


def test_trend_graph_feature_ablation_write_modules_have_no_default_output_literals() -> (
    None
):
    module_paths = [
        PROJECT_ROOT / "src" / "experiments" / "trend_graph_feature_ablation" / filename
        for filename in (
            "runner.py",
            "train_runs.py",
            "evaluate.py",
            "artifact_io.py",
        )
    ]
    module_paths.append(PROJECT_ROOT / "src" / "19_run_trend_graph_feature_ablation.py")

    forbidden_output_literals = (
        "outputs/models/lightgbm/",
        "outputs/reports/",
        "outputs/defense_app/",
        "apps/defense_app/",
        "data/processed/features/",
    )
    offenders: list[str] = []
    for module_path in module_paths:
        source = module_path.read_text(encoding="utf-8")
        for literal in forbidden_output_literals:
            if literal in source:
                offenders.append(f"{module_path.relative_to(PROJECT_ROOT)}: {literal}")

    assert offenders == []


def test_trend_graph_feature_ablation_default_path_semantics_are_guarded() -> None:
    module_paths = [
        PROJECT_ROOT / "src" / "experiments" / "trend_graph_feature_ablation" / filename
        for filename in (
            "runner.py",
            "train_runs.py",
            "evaluate.py",
            "artifact_io.py",
        )
    ]
    module_paths.append(PROJECT_ROOT / "src" / "19_run_trend_graph_feature_ablation.py")
    forbidden_path_semantics = (
        'OUTPUT_DIR / "models" / "lightgbm"',
        'OUTPUT_DIR / "metrics" / "lightgbm"',
        'OUTPUT_DIR / "reports"',
        'OUTPUT_DIR / "defense_app"',
        'PROJECT_ROOT / "apps" / "defense_app"',
        'DATA_DIR / "processed" / "features"',
    )
    allowed_lines = {
        "src/experiments/trend_graph_feature_ablation/train_runs.py": {
            'STABLE_LIGHTGBM_PARAMS_PATH = OUTPUT_DIR / "models" / "lightgbm" / "params.json"'
        },
        "src/experiments/trend_graph_feature_ablation/artifact_io.py": {
            'OUTPUT_DIR / "models" / "lightgbm",',
            'OUTPUT_DIR / "metrics" / "lightgbm",',
            'OUTPUT_DIR / "reports",',
            'OUTPUT_DIR / "defense_app",',
            'PROJECT_ROOT / "apps" / "defense_app",',
            'DATA_DIR / "processed" / "features",',
        },
    }

    offenders: list[str] = []
    for module_path in module_paths:
        relative_path = str(module_path.relative_to(PROJECT_ROOT))
        source_lines = module_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(source_lines, start=1):
            stripped = line.strip()
            if not any(semantic in stripped for semantic in forbidden_path_semantics):
                continue
            if stripped in allowed_lines.get(relative_path, set()):
                continue
            offenders.append(f"{relative_path}:{line_number}: {stripped}")

    assert offenders == []


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
