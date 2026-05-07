# 趋势模块拆分实施计划

## 状态

本计划最初用于把过大的趋势共享实现拆成 `fashion_trend.trend` 子包。早期计划中的兼容入口策略已经被后续领域驱动架构收敛取代。

当前以 Task 4 后的边界为准：

- `fashion_trend.trend` 只是趋势领域包标记。
- `src/fashion_trend/trend/__init__.py` 只保留包说明 docstring，不再重新导出核心 API。
- 仓库内代码必须直接导入具体模块。
- 周级交易稳定产物 reader 归 `fashion_trend.transactions.weekly`。
- 商品目录图 reader 归 `fashion_trend.catalog.graph`。
- 架构目标由 `tests/test_architecture_boundaries.py` 覆盖。

## 当前文件结构

```text
src/fashion_trend/
    catalog/
        graph.py
    transactions/
        weekly.py
    trend/
        __init__.py
        article_sales.py
        attribute_heat.py
        targets.py
        samples.py
        splits.py
        predictions.py
        schema.py
```

`src/fashion_trend/trend.py` 已移除。

## 当前导入原则

不要从 `fashion_trend.trend` 包入口导入具体函数、reader、validator 或 schema 常量。按职责直接导入具体模块：

```python
from fashion_trend.transactions.weekly import read_weekly_transactions
from fashion_trend.catalog.graph import read_article_attribute_edges, read_attribute_nodes
from fashion_trend.trend.article_sales import (
    build_article_week_sales_frame,
    read_article_week_sales,
    validate_article_week_sales,
)
from fashion_trend.trend.attribute_heat import (
    build_attribute_week_heat_frame,
    read_attribute_week_heat,
    validate_attribute_week_heat,
)
from fashion_trend.trend.targets import build_attribute_week_target_frame
from fashion_trend.trend.samples import build_trend_model_samples_frame
from fashion_trend.trend.splits import build_trend_model_split_frames
from fashion_trend.trend.predictions import validate_trend_model_predictions
```

## Task 4 后续验证

确定性趋势流水线边界验证：

```sh
uv run pytest tests/test_trend_article_sales.py tests/test_trend_attribute_heat.py tests/test_trend_targets.py tests/test_trend_samples.py tests/test_trend_splits.py tests/test_architecture_boundaries.py -q
```

在 Task 5 完成前，`tests/test_architecture_boundaries.py` 可能仍只因历史 root `evaluation.py`、`training.py` 和 root `models/` 未迁移而失败。

编译验证：

```sh
uv run python -m py_compile \
  src/fashion_trend/trend/article_sales.py \
  src/fashion_trend/trend/attribute_heat.py \
  src/fashion_trend/trend/targets.py \
  src/fashion_trend/trend/samples.py \
  src/fashion_trend/trend/splits.py \
  src/fashion_trend/trend/predictions.py \
  src/fashion_trend/transactions/weekly.py \
  src/fashion_trend/catalog/graph.py \
  src/05_compute_article_week_sales.py \
  src/06_compute_attribute_week_heat.py \
  src/07_build_trend_targets.py \
  src/08_build_trend_model_samples.py \
  src/09_split_trend_model_samples.py
```

## 后续边界

Task 5 负责迁移趋势模型、训练和评价相关模块，不在 Task 4 文档修正中移动或重写：

- `src/fashion_trend/training.py`
- `src/fashion_trend/evaluation.py`
- `src/fashion_trend/models/`

## 完成标准

- 文档不再指导保留或测试聚合入口。
- 文档验证命令不再引用已删除的兼容测试文件。
- 文档明确说明 `trend/__init__.py` 只是包标记。
- 文档只描述当前领域边界，不引导修改 Task 5 范围内代码。
