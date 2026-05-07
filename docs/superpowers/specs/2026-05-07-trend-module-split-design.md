# 趋势共享模块拆分设计

## 范围

本轮重构 `src/fashion_trend/trend.py`。当前该文件超过 1600 行，同时承担趋势数据流水线、通用 IO、通用校验、时间切分和预测表契约等多重职责，已经影响后续审查依赖边界、时间泄漏风险和模型训练契约。

本轮目标是把趋势共享实现拆成 `fashion_trend.trend` 子包，并让生产代码直接依赖职责明确的子模块。拆分只改变模块组织和导入路径，不改变数据口径、算法行为、输出 schema、文件路径或用户运行命令。

本轮不实现新 baseline，不实现 LightGBM，不进入推荐模块，不调整特征公式，不修改趋势评价指标口径，也不做与拆分无关的格式化。

## 设计结论

采用“兼容 facade + 生产代码优先迁移”的方案。

现有 `src/fashion_trend/trend.py` 替换为 `src/fashion_trend/trend/` 包。`trend/__init__.py` 短期作为兼容 facade，继续 re-export 旧 `fashion_trend.trend` 入口，避免已有外部调用突然断掉。仓库内生产代码改为直接从新子模块导入，使真实运行路径体现新的依赖边界。

这种方式兼顾两个目标：

- 行为风险可控：旧导入路径仍可用，测试可以覆盖兼容入口。
- 依赖边界清楚：脚本、训练、评价和模型代码不再从一个大模块里任意取函数。

## 包结构

目标结构：

```text
src/fashion_trend/
    trend/
        __init__.py
        schema.py
        validation.py
        io.py
        article_sales.py
        attribute_heat.py
        targets.py
        samples.py
        splits.py
        predictions.py
```

职责划分：

| 模块 | 职责 |
| --- | --- |
| `trend/__init__.py` | 兼容 facade，只 re-export 旧 `fashion_trend.trend` API，不承载业务实现 |
| `trend/schema.py` | 列契约、dtype、split 值、预测表常量 |
| `trend/validation.py` | 通用 DataFrame 校验原语，例如必需列、缺失值、唯一键、正数和非负数 |
| `trend/io.py` | 共享 CSV、Parquet、JSON 读写和原子写入辅助函数 |
| `trend/article_sales.py` | 周级交易读取、商品周销量构建和商品周销量校验 |
| `trend/attribute_heat.py` | 商品-属性边读取、属性节点读取、属性周热度构建和校验 |
| `trend/targets.py` | 属性趋势标签构建、读取、公式校验和 heat-target 一致性校验 |
| `trend/samples.py` | 属性图特征、lag/rolling 特征、趋势训练样本构建和样本校验 |
| `trend/splits.py` | train/valid/test 时间切分、split 读取和 split metadata |
| `trend/predictions.py` | 趋势预测表契约、`pred_share_t1` 派生和预测表校验 |

不采用 `fashion_trend/trend_*.py` 平铺结构。所有被拆分模块都属于趋势子域，放在 `fashion_trend.trend.*` 下能保持根包清晰，也方便后续继续扩展趋势相关实现。

## 依赖方向

依赖方向保持单向、可审查：

```text
schema
    -> validation
    -> io

article_sales
    -> attribute_heat
    -> targets
    -> samples
    -> splits

predictions
```

更具体地说：

- `schema.py` 不依赖其他趋势子模块。
- `validation.py` 可以依赖 `schema.py` 中的常量类型，但不依赖具体业务阶段。
- `io.py` 可以依赖 `schema.py`、`validation.py` 和文件读写库，但不依赖 builder。
- 阶段模块可以依赖 `schema.py`、`validation.py`、`io.py` 和上游阶段模块。
- `samples.py` 可以依赖 `attribute_heat.py` 与 `targets.py`，因为它需要校验目标表来自当前 heat 表。
- `splits.py` 只依赖样本 schema 和样本校验，不反向依赖训练 runner。
- `predictions.py` 承载训练和评价共享的预测契约，不依赖具体模型 trainer 或评价指标。
- `training.py`、`evaluation.py` 和 `models/*` 依赖 `splits.py`、`predictions.py`、`io.py` 或 `validation.py`，不依赖 facade。

这套方向让时间泄漏审查集中在 `targets.py` 和 `samples.py`：标签可以使用 `t+1`，特征只能使用当前周和历史周。

## 生产代码迁移

第一阶段迁移仓库内生产代码导入，不要求一次性改完所有测试导入。

迁移目标：

| 文件 | 新依赖 |
| --- | --- |
| `src/05_compute_article_week_sales.py` | `trend.article_sales`、`trend.io` |
| `src/06_compute_attribute_week_heat.py` | `trend.article_sales`、`trend.attribute_heat`、`trend.io` |
| `src/07_build_trend_targets.py` | `trend.attribute_heat`、`trend.targets`、`trend.io` |
| `src/08_build_trend_model_samples.py` | `trend.attribute_heat`、`trend.targets`、`trend.samples`、`trend.io` |
| `src/09_split_trend_model_samples.py` | `trend.splits`、`trend.io` |
| `src/fashion_trend/training.py` | `trend.splits`、`trend.predictions`、`trend.io` |
| `src/fashion_trend/evaluation.py` | `trend.schema`、`trend.predictions`、`trend.validation`、`trend.io` |
| `src/fashion_trend/models/last_week.py` | `trend.schema`、`trend.predictions`、`trend.validation` |
| `src/fashion_trend/models/moving_average.py` | `trend.schema`、`trend.predictions`、`trend.validation` |
| `src/10_train_trend_model.py` | `trend.schema`，只在确实需要 split 常量时导入 |

`trend/__init__.py` 保留旧 API re-export，主要服务兼容和外部调用。新增生产代码不应从 `fashion_trend.trend` facade 导入具体函数。

## 测试策略

测试仍按当前阶段文件组织，不重做测试架构：

```text
tests/test_trend_article_sales.py
tests/test_trend_attribute_heat.py
tests/test_trend_targets.py
tests/test_trend_samples.py
tests/test_trend_splits.py
tests/test_trend_training.py
tests/test_trend_evaluation.py
```

第一阶段只做必要导入调整，避免把测试重写和模块拆分混在一起。允许部分测试继续通过 `fashion_trend.trend` facade 校验旧入口，但应增加或保留一个明确的兼容测试，证明旧导入路径仍导出核心 API。

后续第二阶段可以把测试导入也迁移到子模块，并把 facade 测试缩小到兼容面。本轮 spec 不要求第二阶段立即完成。

## 迁移顺序

建议按以下顺序实施：

1. 创建 `src/fashion_trend/trend/` 包骨架。
2. 先抽出 `schema.py`、`validation.py`、`io.py`，建立底层依赖。
3. 按数据流抽出 `article_sales.py`、`attribute_heat.py`、`targets.py`、`samples.py`。
4. 抽出 `splits.py` 和 `predictions.py`，稳定训练与评价共享契约。
5. 用 `trend/__init__.py` re-export 旧 API，并移除旧 `trend.py` 文件。
6. 迁移生产代码 import 到新子模块。
7. 运行测试和编译验证，确认没有行为漂移。
8. 如 README 或 implementation plan 需要说明模块位置变化，只补充最小文档说明，不重写用户流程。

每一步都应检查 diff，确认只是结构拆分和导入迁移，没有混入算法修改。

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

`io.py` 中的原子写入行为必须保留：写 CSV、Parquet、JSON 时先写临时文件，成功后替换目标文件，失败时清理临时文件。

## 验证命令

最小验证：

```sh
uv run pytest
```

由于本轮涉及模块替换，还应运行编译验证：

```sh
uv run python -m py_compile \
  src/fashion_trend/trend/__init__.py \
  src/fashion_trend/trend/schema.py \
  src/fashion_trend/trend/validation.py \
  src/fashion_trend/trend/io.py \
  src/fashion_trend/trend/article_sales.py \
  src/fashion_trend/trend/attribute_heat.py \
  src/fashion_trend/trend/targets.py \
  src/fashion_trend/trend/samples.py \
  src/fashion_trend/trend/splits.py \
  src/fashion_trend/trend/predictions.py \
  src/05_compute_article_week_sales.py \
  src/06_compute_attribute_week_heat.py \
  src/07_build_trend_targets.py \
  src/08_build_trend_model_samples.py \
  src/09_split_trend_model_samples.py \
  src/10_train_trend_model.py \
  src/11_eval_trend_model.py
```

如果实现期间没有改动数据算法，不要求重新生成真实数据产物。若测试或审查显示可能影响输出口径，再补跑相关 CLI。

## 验收标准

完成后应满足：

- `src/fashion_trend/trend.py` 不再作为大实现文件存在。
- `src/fashion_trend/trend/` 子包存在，且模块职责符合本设计。
- `trend/__init__.py` 只做兼容 re-export，不包含 builder、validator 或 IO 具体实现。
- 生产代码直接导入 `fashion_trend.trend.*` 子模块，不依赖 facade 获取具体函数。
- 旧入口 `from fashion_trend.trend import ...` 对核心 API 仍可用。
- `uv run pytest` 通过。
- 编译验证通过。
- diff 中没有无关格式化、算法口径改动、输出路径改动或临时文件。

## 非目标

本轮不做以下事项：

- 不新增趋势模型。
- 不实现 LightGBM。
- 不实现推荐模块或推荐评价。
- 不改变 README 中的用户运行命令。
- 不调整测试框架。
- 不批量重写测试风格。
- 不删除兼容 facade。
- 不提交或推送，除非用户明确要求。
