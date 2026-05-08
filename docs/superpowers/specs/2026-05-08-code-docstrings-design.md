# 代码注释与 Docstring 补齐设计

## 范围

本设计用于为 `src/` 和 `tests/` 补齐面向理解与维护的 docstring 和必要注释。目标是让读者能更快理解项目代码的业务流程、函数契约、常用结构体、schema 常量和复杂逻辑边界。

本次工作只做可读性增强，不改变业务行为、导入边界、产物路径、测试断言、数据处理公式或模型评价口径。

## 非目标

- 不重构代码结构。
- 不调整函数签名、返回值、异常类型或错误文案。
- 不修改 schema 常量值、路径常量、模型参数、指标计算公式或测试断言。
- 不对全仓做格式化、批量替换、codemod 或自动注释生成。
- 不为每个测试方法机械添加重复 docstring。

## 当前基线

仓库当前已经按业务域分为 `foundation`、`datasets`、`transactions`、`catalog`、`trend`、`recommendation` 和 `reports`。编号脚本 `src/00_*.py` 到 `src/11_*.py` 是业务流程索引，核心计算位于 `src/fashion_trend/` 下的业务包。

现有注释分布不均：

- `datasets.download`、`transactions.weekly`、`foundation.logging`、部分训练和评价入口已有简短中文 docstring。
- `trend` 领域中的 schema、样本构造、属性周热度、训练输出、模型接口、评价指标等复杂模块仍缺少足够说明。
- `TrendArtifact`、`TrendTrainContext`、`TrendTrainResult` 等常用 dataclass 还缺少字段契约说明。
- `tests/` 中共享样本、AST import 检查 helper 和复杂回滚测试 helper 需要说明，但普通测试方法可继续依赖清晰测试名表达意图。

## 注释策略

采用“public API 优先、复杂逻辑加深”的规则。

简单函数使用一句简短中文 docstring，说明函数做什么。例如读取、校验、派生路径、输出日志这类单一职责函数，保持一句话即可。

复杂函数使用结构化 docstring，覆盖：

- 功能目的。
- 参数含义。
- 返回值。
- 可能抛出的异常。
- 关键边界或不变量。

复杂函数包括但不限于：

- complete panel 构建。
- lag 和 rolling 特征构造。
- 趋势标签与样本契约校验。
- `pred_share_t1` 派生和归一化校验。
- atomic output 发布、回滚和 payload 校验。
- valid/test-only 趋势评价聚合。
- AST import 边界检查。

dataclass、Protocol 和常用 tuple schema 应补说明：

- dataclass / Protocol 使用类 docstring 解释字段角色和消费方。
- schema tuple 上方使用短注释解释该列集合对应的产物或契约。
- 结构体字段不逐行重复类型表面含义，只解释业务语义、稳定性要求和下游依赖。

private helper 只在逻辑不直观时补说明。回滚、路径安全、整数周校验、严格 JSON 校验、指标折现、缺失值边界等 helper 可以加 docstring 或局部注释；简单转发或显而易见的 helper 不强制补。

测试方法不机械补 docstring。普通 `test_*` 方法继续用测试名表达行为；共享样本函数、架构边界 helper、复杂测试支撑函数和不直观的 fixture 构造需要补说明。

## 实施顺序

按业务域推进，并结合克制的 public API 覆盖原则。

### 第一阶段：基础和上游领域

覆盖：

- `foundation`
- `datasets`
- `transactions`
- `catalog`

重点说明通用 DataFrame 校验、原子写入、安全路径、Kaggle 下载、周级交易构建、商品清洗和属性图构建。

### 第二阶段：趋势领域

覆盖 `trend` 下的核心模块：

- schema 和路径契约。
- 属性周热度与商品周销量。
- 趋势标签。
- 趋势样本和图特征。
- 时间切分。
- 预测契约。
- 模型接口、registry 和 baseline。
- 训练 runner、输出 payload、artifact 发布和回滚。
- 趋势评价 metrics、payload 和 runner。

这是本次注释工作的重点。结构化 docstring 应优先投放到数据流、指标流和 artifact 写盘边界。

### 第三阶段：编号 CLI

覆盖 `src/00_*.py` 到 `src/11_*.py`。

编号 CLI 的 docstring 应强调它们是业务流程索引，说明高层编排顺序、输入输出和退出码语义。不把 CLI docstring 写成详细算法文档，算法细节归业务域模块说明。

### 第四阶段：测试

覆盖 `tests/` 中的共享 helper 和复杂测试基础设施：

- `tests/trend_samples.py` 中的样本构造函数。
- `tests/test_architecture_boundaries.py` 中的 AST import 解析、allowlist 和历史导入检查 helper。
- 复杂回滚、原子写入、payload 校验相关测试 helper。

普通测试方法不强制添加 docstring，避免重复测试名。

## 审查边界

最终 diff 应只包含注释、docstring 和必要空行调整。审查时重点确认没有出现下列行为性变化：

- import 路径变化。
- 常量值变化。
- 函数签名变化。
- pandas 表达式、排序、聚合、merge、rolling、指标公式变化。
- 异常类型或错误文案变化。
- 测试断言变化。
- 输出路径、产物文件名或 schema 列顺序变化。

如果补 docstring 时发现函数职责或命名存在问题，本次只记录，不顺手重构。只有当当前结构明显阻碍准确注释时，才单独提出并等待确认。

## 验证

完成实现后运行：

```sh
uv run python -m compileall src tests
uv run pytest
git diff --check
```

并做人工 diff 审查，确认改动范围只涉及 docstring、注释和必要空行。

必要时用 `rg` 抽查 public 函数、类和 schema 常量覆盖情况，避免遗漏重要理解入口。

## 风险与控制

主要风险是注释工作会产生较大 diff，容易混入非注释改动。控制方式：

- 逐业务域、逐文件直接编辑。
- 不使用批量脚本、自动注释生成、全仓格式化或 codemod。
- 每阶段检查 diff，确认没有行为变化。
- 对复杂函数使用结构化 docstring，但避免为了凑格式写空泛段落。

另一个风险是注释和代码漂移。控制方式是只描述稳定契约、当前公式和当前边界，不写尚未落地的实现承诺；对未实现模块只说明当前已有职责，不补不存在的行为。
