# pytest 迁移与测试拆分设计

## 范围

本轮将项目测试从标准库 `unittest` 迁移到 `pytest`。迁移采用分阶段方式：

1. 引入 `pytest` 作为正式测试入口。
2. 将较小的测试文件迁移为 pytest idiom。
3. 将过大的 `tests/test_trend.py` 按 README 和 `docs/gpt-research/implementation-plan.md` 中的数据处理与趋势建模流程拆分。

本轮不修改生产代码行为，不重写业务逻辑，不新增模型能力，也不改变现有 147 个测试覆盖点的语义。迁移后的正式验证命令为：

```sh
uv run pytest
```

旧的 `unittest discover` 命令可以作为临时兼容参考，但 README 和日常开发入口应切换到 pytest。

## 设计结论

采用分阶段 pytest 迁移方案。

先让项目拥有稳定 pytest runner，再迁移小文件，最后拆分 `tests/test_trend.py`。拆分时按项目主线组织测试文件，而不是简单按原类名切块。测试目录应直接反映当前数据流水线：

```text
articles clean
attribute graph
article week sales
attribute week heat
trend targets
trend samples
trend splits
trend training
trend evaluation
```

这样做的目标是让后续新增 EWMA、LightGBM、推荐模块测试时有自然落点，避免继续向单个大文件追加测试。

## 文件组织

迁移后测试目录目标结构：

```text
tests/
    __init__.py
    trend_samples.py
    test_articles_clean.py
    test_attribute_graph.py
    test_trend_article_sales.py
    test_trend_attribute_heat.py
    test_trend_targets.py
    test_trend_samples.py
    test_trend_splits.py
    test_trend_training.py
    test_trend_evaluation.py
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `tests/__init__.py` | 让共享测试 helper 可以通过 `tests.trend_samples` 显式导入 |
| `tests/trend_samples.py` | 只放跨多个测试文件复用的样本数据 builder，供拆分后的测试显式导入 |
| `tests/test_articles_clean.py` | articles 清洗字段、缺失值、重复 `article_id` 和双输出写出回滚 |
| `tests/test_attribute_graph.py` | 属性节点、商品-属性边、属性层级边、图文件写出和回滚 |
| `tests/test_trend_article_sales.py` | 周交易读取、商品周销量聚合、商品周销量读取、趋势 CSV 写出 |
| `tests/test_trend_attribute_heat.py` | 商品属性边读取、属性节点校验、属性周热度完整面板和派生字段 |
| `tests/test_trend_targets.py` | `attribute_week_target.csv` 目标计算、增长公式和异常输入 |
| `tests/test_trend_samples.py` | 趋势样本 lag、移动窗口、图特征、目标合入和 stale target 防护 |
| `tests/test_trend_splits.py` | train/valid/test 时间切分、split 读取、metadata、JSON/Parquet 写出 |
| `tests/test_trend_training.py` | `last_week`、`moving_average`、registry、runner、训练输出契约和训练 CLI |
| `tests/test_trend_evaluation.py` | 预测读取、评价输入校验、分组指标、payload、写出边界和评价 CLI |

`tests/test_trend.py` 在拆分完成后删除，不保留空壳 wrapper。

## pytest 迁移规则

测试代码应使用 pytest idiom：

- `unittest.TestCase` 改为普通测试类或模块级测试函数。
- `self.assertEqual(a, b)` 改为 `assert a == b`。
- `self.assertTrue(expr)` 改为 `assert expr`。
- `self.assertFalse(expr)` 改为 `assert not expr`。
- `self.assertIn(a, b)` 改为 `assert a in b`。
- `self.assertNotIn(a, b)` 改为 `assert a not in b`。
- `self.assertRaisesRegex(Error, pattern)` 改为 `pytest.raises(Error, match=pattern)`。
- `TemporaryDirectory()` 优先改为 `tmp_path`。
- `self.subTest(...)` 优先改为 `pytest.mark.parametrize`。

对 pandas 表格的精确相等断言，如当前只是检查列名、长度、集合、单元格值，继续使用裸 `assert`。只有需要比较完整 DataFrame 时，才引入 `pandas.testing.assert_frame_equal`。

## 共享样本数据

共享样本数据保持克制。

如果样本 builder 只被一个测试文件使用，保留在该测试文件内。如果同一个 builder 被多个拆分后的趋势测试文件复用，则移动到 `tests/trend_samples.py`，并命名为普通函数，通过 `tests.trend_samples` 显式导入。

初始迁移可优先移动这些跨文件样本：

- `sample_article_attribute_edges()`
- `sample_attribute_nodes()`
- `sample_attribute_week_heat()`
- `sample_long_attribute_week_heat()`
- `sample_attribute_hierarchy_edges()`
- `sample_trend_model_samples_for_split()`
- `sample_trend_predictions_for_evaluation()`

不把所有测试数据一次性集中到 helper 文件，避免形成新的大文件。

## 依赖与配置

在 `pyproject.toml` 的 dev dependency 中加入 `pytest`。

增加最小 pytest 配置：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`pythonpath = ["src"]` 让 `uv run pytest` 能稳定导入 `fashion_trend`，不要求调用者手动设置 `PYTHONPATH`。不引入额外 pytest 插件，除非后续验证证明确有必要。

如果 `uv sync` 或 `uv run pytest` 因 sandbox 内无法访问 `~/.cache/uv` 失败，验证时应使用已批准的 `uv run` 权限重试。该问题属于本地执行权限，不是测试迁移失败。

## 迁移顺序

第一阶段：引入 pytest runner。

- 更新 `pyproject.toml` dev dependency。
- 增加 pytest 配置。
- 确认未迁移的 `unittest.TestCase` 测试仍可由 pytest 收集并运行。
- 更新 README 验证命令。

第二阶段：迁移小文件。

- 将 `tests/test_articles_clean.py` 迁移为 pytest idiom。
- 将 `tests/test_attribute_graph.py` 迁移为 pytest idiom。
- 使用 `tmp_path` 替换临时目录。
- 使用 `pytest.raises` 和裸 `assert` 替换 unittest 断言。

第三阶段：拆分并迁移 `tests/test_trend.py`。

- 先按职责创建目标测试文件。
- 每次移动一个职责区域，迁移该区域断言为 pytest idiom。
- 每次移动后运行对应测试文件或 pytest 选中子集。
- 拆分完成后删除原 `tests/test_trend.py`。

第四阶段：收口验证与文档同步。

- 全量运行 `uv run pytest`。
- 使用 `rg` 确认测试文件内没有残留 `unittest`、`TestCase`、`self.assert` 和 `TemporaryDirectory`。
- 使用 `rg` 确认 README 和当前实现计划不再把正式测试入口写成 `unittest discover`。

## 错误处理与边界

迁移不得通过跳过测试、弱化断言或吞掉异常来获得通过结果。

如果迁移后测试失败，先判断失败属于三类中的哪一类：

1. 迁移错误，例如断言写反、regex 转义不一致、fixture 生命周期改变。
2. pytest 收集或导入错误，例如 `pythonpath` 配置不正确。
3. 原测试暴露出的真实生产代码问题。

只有第一类和第二类属于本轮迁移范围。第三类需要单独定位根因，不能用静默 fallback 或 mock 成功路径掩盖。

CLI 测试中由 `argparse` 打印的 usage、项目日志输出和非零退出码都保留原行为，只迁移断言形式，不改变生产代码。

## 验收标准

迁移完成需要满足：

- `uv run pytest` 通过。
- pytest 收集到的测试数量不低于当前 147 个测试语义覆盖点。
- `tests/test_trend.py` 已拆分到职责明确的多个测试文件中。
- 测试目录不再依赖 `unittest.TestCase`。
- README 的正式验证命令更新为 `uv run pytest`。
- `pyproject.toml` 声明 pytest dev dependency 和最小 pytest 配置。
- 未留下临时调试代码、一次性脚本、无关日志或中间产物。

## 非目标

本轮不做以下事情：

- 不修改生产代码行为。
- 不新增或删除趋势模型。
- 不引入 pytest 插件生态。
- 不迁移到 coverage、tox、nox 或 CI。
- 不把所有测试样本集中成一个大型 helper 文件。
- 不对测试做无关格式化或大范围风格重写。
