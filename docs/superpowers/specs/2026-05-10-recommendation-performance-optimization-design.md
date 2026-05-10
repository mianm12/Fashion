# 推荐性能优化设计

## 背景

推荐阶段已经完成轻量离线 Top-N 闭环，但全量运行耗时过长。近期在 Mac M4 Pro 上按现有流程重跑真实 H&M 全量产物时，推荐阶段暴露出结构性性能问题：`default` 候选约 4,333 万行，每个推荐 method 产出约 120 万行 `recommendations.csv` 和约 1,444 万行 `recommendation_items.csv`，单个 `recommendation_items.csv` 约 2.9GB 到 3.4GB。五个 method 的推荐输出目录合计超过 15GB。

这不是硬件异常，而是当前推荐流程把大量中间排序候选和解释长表作为 CSV 稳定产物反复生成、读取和重建。优化目标是让真实全量推荐生产运行更快，而不是只增加小样本 smoke 模式。

## 目标

- 不新增 runtime 依赖，继续使用现有 `pandas`、`numpy`、`pyarrow` 和标准库。
- 将全量 `12 -> 16` 推荐生产流程尽量压到 1 小时以内。
- 将 `recommendation_items` 主产物从 CSV 迁移到 Parquet。
- 允许推荐算法变化，但必须保持无时间泄漏、Top-K 合法、指标口径稳定和产物可审计。
- 让 `16_run_recommendation_experiment.py` 成为智能编排入口，而不是默认全链路强制重跑入口。

## 非目标

- 不引入 DuckDB、Polars、FAISS、Annoy、LightFM、implicit、向量数据库或深度推荐依赖。
- 不实现在线服务、实时推荐、双塔模型或近似向量召回。
- 不要求新推荐结果与当前实现逐行一致。
- 不提交生成数据、模型输出或推荐 artifact。

## 根因分析

当前慢主要来自五类问题：

1. 输出格式过重：`recommendation_items.csv` 是内部稳定长表，但 CSV 对千万级字符串行、浮点列和全字段引用不适合作为生产内部格式。
2. 重复计算过多：多个 method 重复扫描交易、用户画像、趋势预测和候选表，重复构建 popularity、recent、similarity、trend feature。
3. 编排语义过重：`16 --force` 会重新构建输入、候选、baseline method、主 method 和实验结果，容易把前面已跑过的 `12-15` 再做一遍。
4. 按 window 循环切大表：多处逻辑按 16 个推荐窗口对大 DataFrame 做 boolean mask、merge、groupby、sort 和 dedupe。
5. 产物语义偏中间态：稳定长表保存过多排序前或可复算信息，而生产推荐真正需要的是 Top-12 结果和必要解释。

## 性能预算

以本机真实 H&M 全量数据为验收基准：

- `12` 推荐输入：分钟级，可重建但不应被 `16` 默认反复重建。
- `13` 候选生成：全量必要 strategy 合计目标 15 到 20 分钟。
- `14` method 产出：五个 method 合计目标 25 到 30 分钟。
- `15` 推荐评价：五个 method 合计目标 5 到 10 分钟。
- `16` 默认实验编排：已有新鲜产物时目标 10 到 15 分钟。
- 默认全量生产链路 `12 -> 16`：目标 1 小时以内。
- `--force-rebuild-all` 是显式慢路径，可以超过 1 小时，但必须记录阶段耗时。

## Artifact 契约

推荐 artifact 分成生产主产物、缓存产物和可选导出产物。

### 生产主产物

继续保留：

```text
outputs/recommendation/<method>/recommendations.csv
outputs/recommendation/<method>/params.json
outputs/recommendation/<method>/metadata.json
outputs/recommendation/<method>/metrics.json
outputs/recommendation/experiments/<experiment_id>/experiment.json
```

新增并作为主长表产物：

```text
outputs/recommendation/<method>/recommendation_items.parquet
```

`recommendation_items.parquet` 只保存最终 Top-12 item 解释行，不保存排序前全部候选评分。列契约保持：

```text
customer_id
split
cutoff_week
label_week
method
article_id
rank
score
pop_score
sim_score
trend_score
recent_score
candidate_sources
```

`recommendations.csv` 继续作为最终短表和外部可读主结果。

### 候选与缓存产物

候选继续 strategy-scoped：

```text
data/processed/recommend/candidates/<strategy>/candidate_items.parquet
data/processed/recommend/candidates/<strategy>/metadata.json
```

新增可复用 feature cache：

```text
data/processed/recommend/features/popularity_scores.parquet
data/processed/recommend/features/recent_scores.parquet
data/processed/recommend/features/similarity_scores.parquet
data/processed/recommend/features/trend_scores.parquet
data/processed/recommend/features/seen_items.parquet
data/processed/recommend/features/recommendable_pool.parquet
data/processed/recommend/features/metadata.json
```

feature cache metadata 必须记录：

- `input_artifacts`
- `input_fingerprints`
- `algorithm_version`
- `schema_version`
- `config`
- `row_counts`
- `generated_at`

输入、算法版本或配置变化时，下游必须识别 stale。默认行为可以报错并给出重建命令，也可以在 `16` 中只重建必要 stale 节点，但不能静默混用旧产物。

### 可选导出产物

`recommendation_items.csv` 降级为显式导出产物。默认不写。如果需要人工审阅或外部交付，后续可通过独立导出入口生成，例如：

```sh
uv run python src/17_export_recommendation_artifacts.py --method pop_similarity_trend --items-csv
```

或通过 `14` 的显式参数生成：

```sh
uv run python src/14_rerank_recommendations.py --method pop_similarity_trend --export-items-csv
```

第一阶段不必须实现 `17`，但文档和 reader 不能再把 CSV 当作默认内部读取来源。

## 推荐计算数据流

### `12` 推荐输入

`12_build_recommendation_inputs.py` 保持职责不变：

```text
weekly transactions + article attributes + lightgbm predictions
-> time_windows
-> target_users
-> evaluation_labels
-> user_profile
```

它不生成候选或排序 feature，但需要写出更完整 metadata，供 `13`、feature cache、`14`、`15`、`16` 做 freshness 检查。

### `13` 候选生成

`13_build_recommend_candidates.py` 仍按 strategy 写候选，但内部应避免重复构建 source：

```text
popularity_source
similarity_source
trend_source
```

`default` strategy 只做 source union 和去重，不重新计算 popularity、similarity、trend source。`popularity`、`similarity`、`trend_union` 可以复用 source cache 或 materialized source parquet。

### Feature cache

新增 feature cache builder。它可以先作为推荐包内函数落地，再按需要提升为编号脚本。职责是 materialize 跨 method 共享的计算：

```text
candidate_items + transactions + article_attributes + user_profile + trend_predictions
-> pop_score
-> recent_score
-> sim_score
-> trend_score
-> seen_items
-> recommendable_pool
```

缓存边界：

- `pop_score` 和 `recent_score` 按 `split + cutoff_week + label_week + article_id` 预计算。
- `sim_score` 只针对候选中实际出现的 `customer_id + article_id + window` 预计算。
- `trend_score` 按 `split + cutoff_week + label_week + article_id` 预计算，所有用户共享。
- `seen_items` 按 `split + cutoff_week + label_week + customer_id + article_id` 或等价 join key 预计算。
- `recommendable_pool` 按窗口预计算，供所有 method 的评价复用。

### `14` Method 产出

`14_rerank_recommendations.py` 不再从原始 transactions、profile 和 predictions 重建全部 ranking features。它读取：

```text
method candidate set
+ feature cache
+ seen_items
-> score
-> filter seen
-> Top-12
-> recommendations.csv
-> recommendation_items.parquet
-> params.json
-> metadata.json
```

每个 method 只负责声明：

- candidate strategy
- required score columns
- score weights
- exclude seen 行为
- backfill 行为

### `15` 推荐评价

`15_eval_recommendations.py` 读取：

```text
recommendations.csv
target_users
evaluation_labels
recommendable_pool cache
```

它不再从 transactions 为每个 method 重新构建 recommendable pool。

### `16` 实验编排

`16_run_recommendation_experiment.py` 默认执行：

```text
ensure inputs fresh
ensure candidates fresh
ensure feature cache fresh
ensure baseline outputs fresh or rebuild selected stale method
run valid weight search using cached features
publish best pop_similarity_trend
write experiment.json
```

默认不重建新鲜的 `12` 输入、`13` 候选、feature cache 或 baseline method。

## 候选与排序算法调整

优化后的推荐不要求逐行匹配旧版结果。候选池应更贴近 Top-N 生产语义。

### 候选池

保留三类 source：

- popularity：全局或近期热门兜底。
- similarity：用户属性画像召回。
- trend：趋势属性召回。

`default` 仍是主策略，但每个 source 的召回上限独立配置。第一版默认建议：

```text
popularity source: top 24
similarity source: top 24
trend source: top 24
final recommendation: top 12
```

后续可通过 valid metrics 比较 `12/24/36/48` 等配置。配置必须写入 metadata。

### Top-K 产物语义

`recommendation_items.parquet` 只保存最终 Top-12 解释项。排序前候选评分默认不落盘。如果需要审计，可通过实验 scoped debug artifact 写出：

```text
outputs/recommendation/experiments/<experiment_id>/runs/<run_id>/candidate_scores.parquet
```

### Score 计算

保留四类 score：

- `pop_score`
- `recent_score`
- `sim_score`
- `trend_score`

变化点：

- `trend_score` 是重排序信号，trend source 只负责补充候选。
- `similarity` 只对候选内 user-article pair 计算。
- `pop_score` 和 `recent_score` 只按 article-window 计算一次，再 join 到候选。

### Backfill

Backfill 策略：

1. 先从 method candidate pool 中选择 Top-12。
2. 如果用户窗口不足 Top-12，再用预计算 popularity/recent source 补齐。
3. Backfill 只进入最终 Top-12 产物，不保存所有补齐候选。
4. metadata 记录 `underfilled_user_count`、`backfilled_user_count` 和 `still_underfilled_user_count`。

### 权重搜索

`pop_similarity_trend` 的 valid 权重搜索不写 stable method 产物。它使用缓存特征在 valid split 内生成 Top-12 临时结果并评价，最后只发布最佳权重对应的 stable method 结果。

## `16` 重建语义

将重建模式拆成明确参数：

```sh
--force-experiment
```

只重跑权重搜索和实验汇总，不重建输入、候选、cache 或 baseline。

```sh
--force-method <method>
```

只重建指定 method，可重复使用。

```sh
--force-cache
```

重建 feature cache，但复用输入和候选。

```sh
--force-candidates
```

重建候选和 feature cache，但不重建 `12` 输入。

```sh
--force-rebuild-all
```

从 `12` 输入、候选、feature cache、baseline、主方法、metrics、experiment 全部重建。这个命令是显式慢路径。

旧 `--force` 不应继续表示全链路重建。它可以作为 `--force-experiment` 的 alias，也可以废弃并报错提示新参数，避免误用。

`experiment.json` 需要记录：

- 输入 fingerprints。
- candidate/cache/method artifact paths。
- baseline metrics。
- valid weight search grid。
- best weights。
- best valid metrics。
- final test metrics。
- 每个阶段 reused 或 rebuilt。
- 使用的 force 参数。
- 总耗时和阶段耗时。

## 性能日志

所有 `12-16` 入口都应记录阶段耗时和行数。使用 `time.perf_counter()` 即可，不新增依赖。日志示例：

```text
stage=input_build rows=<row_count> elapsed=<seconds>
stage=candidate_build strategy=default rows=<row_count> elapsed=<seconds>
stage=feature_cache name=sim_score rows=<row_count> elapsed=<seconds>
stage=method method=pop_similarity_trend rows=<row_count> elapsed=<seconds>
stage=evaluation method=<method_name> elapsed=<seconds>
stage=experiment experiment=main elapsed=<seconds>
```

性能优化后必须能从日志判断瓶颈是否从 CSV IO 转移到 feature join/rank。

## 正确性验证

新增或更新测试覆盖：

- `recommendation_items.parquet` schema、dtype、key 唯一性、rank 范围和 Top-K 无重复。
- `recommendations.csv` 与 `recommendation_items.parquet` 一致。
- Cache freshness：输入 fingerprint、algorithm version 或 config 变化时识别 stale。
- `16` 默认复用新鲜产物，不重建 `12`、`13`、cache 或 baseline。
- `--force-cache`、`--force-candidates`、`--force-method`、`--force-rebuild-all` 只影响声明范围。
- `--force-rebuild-all` 后关键 artifact 更新时间戳更新。
- 时间泄漏检查：trend score 使用 `cutoff_week`，评价 labels 使用 `label_week`。
- 推荐算法变化后，Top-K、无重复、eligible denominator 和 metrics payload 仍合法。

## 真实 Artifact 验收

全量跑完后必须审计：

- `recommendations.csv` 行数等于 eligible user-window 覆盖数，或 metadata 明确记录缺失。
- `recommendation_items.parquet` 行数不超过 `eligible user-window * 12`。
- 每个 method 默认不生成 `recommendation_items.csv`。
- `recommendation_items.parquet` 体积显著小于历史 2.9GB 到 3.4GB CSV。
- `outputs/recommendation/experiments/main/experiment.json` 记录 reused/rebuilt、阶段耗时和 best weights。
- `git status` 不出现源码以外的意外改动；生成产物不提交。

## 实施切分

建议按 5 个阶段实施，避免一次大改：

1. 观测与耗时日志：给 `12-16` 加阶段计时和 row count，不改算法，建立 baseline。
2. Artifact 契约迁移：将 `recommendation_items.csv` 主产物迁到 `recommendation_items.parquet`，更新 reader、evaluator、docs 和 tests。
3. Feature cache 层：引入 `data/processed/recommend/features/`，缓存 pop/recent/sim/trend/seen/recommendable_pool，并接入 freshness metadata。
4. Method runner 重构：`14` 改为读取 candidate 和 feature cache，只输出 Top-12 parquet 和 short CSV。
5. Experiment 编排重构：改造 `16` 的 force 语义、新鲜度复用、权重搜索和 experiment payload，最后跑真实全量性能验收。

每个阶段都应独立提交，避免把 artifact 契约、算法变化和实验编排混成一个大 diff。
