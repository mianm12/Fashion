# 趋势共享模块拆分设计

## 状态

本设计记录的是 2026-05-07 初始趋势模块拆分方案。该方案中的兼容策略已经被后续领域驱动架构收敛取代。

当前兼容政策以 Task 4 后的实现为准：

- `fashion_trend.trend` 现在只是趋势领域 package marker。
- `src/fashion_trend/trend/__init__.py` 只保留包说明 docstring，不再作为 re-export facade。
- 仓库内 Python import 必须直接依赖具体模块，例如 `fashion_trend.trend.article_sales`、`fashion_trend.trend.attribute_heat`、`fashion_trend.trend.targets`、`fashion_trend.trend.samples`、`fashion_trend.trend.splits`、`fashion_trend.trend.predictions`。
- 周级交易稳定产物 reader 属于 `fashion_trend.transactions.weekly`。
- 商品目录图 reader 属于 `fashion_trend.catalog.graph`。

旧方案曾为了降低早期拆分风险而保留顶层 re-export 入口；领域包边界稳定后，内部 Python 导入允许破坏旧聚合入口，以换取更清晰的领域所有权和架构测试边界。

## 范围

本轮重构最初针对过大的趋势共享实现。原实现同时承担趋势数据流水线、通用 IO、通用校验、时间切分和预测表契约等多重职责，影响后续审查依赖边界、时间泄漏风险和模型训练契约。

最终目标仍是把趋势共享实现拆成职责明确的 `fashion_trend.trend` 子包，并让生产代码直接依赖职责明确的子模块。拆分不改变数据口径、算法行为、输出 schema、文件路径或用户运行命令。

本设计不实现新 baseline，不实现 LightGBM，不进入推荐模块，不调整特征公式，不修改趋势评价指标口径，也不做与拆分无关的格式化。

## 当前设计结论

采用“领域边界优先 + 具体模块导入”的方案。

`src/fashion_trend/trend.py` 已被 `src/fashion_trend/trend/` 包替代。`trend/__init__.py` 不承载业务实现，也不重新导出核心 API。仓库内生产代码和测试都应直接从具体模块导入，使真实运行路径体现新的依赖边界。

这种方式兼顾两个目标：

- 依赖边界清楚：脚本、训练、评价和模型代码不再从一个聚合入口任意取函数。
- 所有权明确：稳定交易产物读取归 `transactions`，目录图读取归 `catalog`，趋势包只保留确定性趋势流水线逻辑。

## 包结构

当前结构：

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

职责划分：

| 模块 | 职责 |
| --- | --- |
| `catalog.graph` | 商品目录图构建、商品-属性边读取、属性节点读取 |
| `transactions.weekly` | 原始交易周级化，以及周级交易稳定产物读取 |
| `trend/__init__.py` | 趋势领域包标记，只保留包说明 |
| `trend/schema.py` | 列契约、dtype、split 值、预测表常量 |
| `trend/article_sales.py` | 商品周销量构建和商品周销量校验 |
| `trend/attribute_heat.py` | 属性周热度构建、heat 专属输入校验和输出校验 |
| `trend/targets.py` | 属性趋势标签构建、读取、公式校验和 heat-target 一致性校验 |
| `trend/samples.py` | 属性图特征、lag/rolling 特征、趋势训练样本构建和样本校验 |
| `trend/splits.py` | train/valid/test 时间切分、split 读取和 split metadata |
| `trend/predictions.py` | 趋势预测表契约、`pred_share_t1` 派生和预测表校验 |

## 依赖方向

依赖方向保持单向、可审查：

- `catalog` 和 `transactions` 不依赖 `trend`。
- `trend.article_sales` 可以读取 `transactions.weekly.read_weekly_transactions`，但商品周销量构建逻辑仍归趋势阶段。
- `trend.attribute_heat` 使用目录图 reader 的输出，但商品目录图读取归 `catalog.graph`。
- `targets.py` 依赖 `attribute_heat.py` 的 heat 表契约。
- `samples.py` 可以依赖 `attribute_heat.py` 与 `targets.py`，因为它需要校验目标表来自当前 heat 表。
- `splits.py` 只依赖样本 schema 和样本校验，不反向依赖训练 runner。
- `predictions.py` 承载训练和评价共享的预测契约，不依赖具体模型 trainer 或评价指标。

这套方向让时间泄漏审查集中在 `targets.py` 和 `samples.py`：标签可以使用 `t+1`，特征只能使用当前周和历史周。

## 生产代码导入原则

新增和维护代码不得从 `fashion_trend.trend` 聚合入口导入具体函数或常量。应按职责直接导入：

```python
from fashion_trend.transactions.weekly import read_weekly_transactions
from fashion_trend.catalog.graph import read_article_attribute_edges, read_attribute_nodes
from fashion_trend.trend.article_sales import build_article_week_sales_frame
from fashion_trend.trend.attribute_heat import build_attribute_week_heat_frame
from fashion_trend.trend.targets import build_attribute_week_target_frame
from fashion_trend.trend.samples import build_trend_model_samples_frame
from fashion_trend.trend.splits import build_trend_model_split_frames
from fashion_trend.trend.predictions import validate_trend_model_predictions
```

## 测试策略

测试按当前阶段文件组织：

```text
tests/test_trend_article_sales.py
tests/test_trend_attribute_heat.py
tests/test_trend_targets.py
tests/test_trend_samples.py
tests/test_trend_splits.py
tests/test_trend_training.py
tests/test_trend_evaluation.py
tests/test_architecture_boundaries.py
```

趋势测试应直接导入具体模块。旧聚合入口兼容测试已删除；架构目标由 `tests/test_architecture_boundaries.py` 覆盖。

## 行为不变约束

以下契约必须保持不变：

- `article_week_sales.csv` 输出列、排序、聚合口径不变。
- `attribute_week_heat.csv` 继续是完整 `week_id x attr_id` 面板。
- `heat_cnt` 继续只基于商品周销量中的 `sales_cnt`。
- `target_growth = log((share_t1 + epsilon) / (share_t + epsilon))` 不变。
- `trend_model_samples.parquet` 特征只使用当前周和历史周，目标字段来自标签表。
- train/valid/test 继续按时间顺序切分，且互不重叠。
- `TREND_MODEL_PREDICTION_COLUMNS` 列顺序不变。
- `pred_share_t1` 继续在 `split/week_id/attr_type` 内归一化。
- 趋势评价只写 `outputs/metrics/<model>/trend_metrics.json`，不改写训练产物。

如果实施时发现必须改变行为，应停止并单独提出变更原因，不把行为变化混进拆分重构。

## 错误处理

拆分不得降低现有错误质量。

现有可预期错误仍应抛出可定位异常，例如缺文件、缺列、缺失值、重复键、非法 split、非有限数值、stale target、预测表列顺序不一致和 `pred_share_t1` 未归一化。不得用静默 fallback、空 `except` 或宽松 schema 掩盖问题。

## 验证命令

确定性趋势流水线边界验证：

```sh
uv run pytest tests/test_trend_article_sales.py tests/test_trend_attribute_heat.py tests/test_trend_targets.py tests/test_trend_samples.py tests/test_trend_splits.py tests/test_architecture_boundaries.py -q
```

历史说明：Task 5 完成前，该架构测试曾可能因 root `evaluation.py`、`training.py` 和 root `models/` 未迁移而失败。当前这些 root 路径已迁移到 `fashion_trend.trend.evaluation`、`fashion_trend.trend.training` 和 `fashion_trend.trend.models`。

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

如果实现期间没有改动数据算法，不要求重新生成真实数据产物。若测试或审查显示可能影响输出口径，再补跑相关 CLI。

## 验收标准

完成后应满足：

- `src/fashion_trend/trend.py` 不再作为大实现文件存在。
- `src/fashion_trend/trend/` 子包存在，且模块职责符合当前领域边界。
- `trend/__init__.py` 只包含包说明，不包含 builder、validator、reader、schema re-export 或 IO 具体实现。
- 生产代码直接导入具体模块，不依赖 `fashion_trend.trend` 聚合入口获取具体函数。
- 趋势测试直接导入具体模块。
- 确定性趋势测试通过；架构测试只允许保留 Task 5 范围内的历史 root 模块失败。
- 编译验证通过。
- diff 中没有无关格式化、算法口径改动、输出路径改动或临时文件。

## 非目标

本轮不做以下事项：

- 不新增趋势模型。
- 不实现 LightGBM。
- 不实现推荐模块或推荐评价。
- 不改变 README 中的用户运行命令。
