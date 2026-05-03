# articles.csv 属性图设计

## 范围

本轮先从 `articles.csv` 生成两份过滤清洗后的中间表，再基于中间表构建静态商品属性图。

本轮产出两类文件：

- `data/interim/` 下的 articles 清洗中间表。
- `data/processed/graph/` 下的属性图节点表和边表。

本轮不计算商品周销量、属性周热度、趋势标签、趋势特征、模型结果或推荐结果。

articles 清洗中间表落地为 2 张 CSV 表：

- `data/interim/articles_clean_mvp.csv`
- `data/interim/articles_clean.csv`

属性图落地为 4 张 CSV 表：

- `data/processed/graph/nodes_article.csv`
- `data/processed/graph/nodes_attribute.csv`
- `data/processed/graph/edges_article_attribute.csv`
- `data/processed/graph/edges_attribute_hierarchy.csv`

## 属性字段范围

本轮使用 `articles.csv` 中的 10 个属性字段。

核心属性字段用于后续趋势主线：

| 字段 | 角色 |
| --- | --- |
| `product_group_name` | 商品品类父级 |
| `product_type_name` | 商品品类子级 |
| `garment_group_name` | 扁平服装组属性 |
| `colour_group_name` | 具体颜色子级 |
| `graphical_appearance_name` | 扁平图案或外观属性 |

层级增强字段用于构造属性父子边：

| 字段 | 角色 |
| --- | --- |
| `perceived_colour_master_name` | 颜色父级 |
| `index_group_name` | 业务大类父级 |
| `index_name` | 业务线，既可作为父级也可作为子级 |
| `section_name` | 商品区域，既可作为父级也可作为子级 |
| `department_name` | 部门子级 |

`detail_desc` 不进入本轮处理范围。

## articles 清洗中间表

先对原始 `articles.csv` 做字段过滤、类型统一和缺失值校验，生成两份中间表。后续属性图构建不直接读取原始 `articles.csv`，而是读取这两份中间表。

### MVP 版中间表

输出文件：`data/interim/articles_clean_mvp.csv`

MVP 版只保留商品标识、展示字段和 5 个核心属性字段：

| 字段 | 用途 |
| --- | --- |
| `article_id` | 商品唯一 ID，按字符串保留前导 0 |
| `product_code` | 商品款式族编号，用于审查和后续扩展 |
| `prod_name` | 商品名称，用于展示 |
| `product_group_name` | 核心属性，商品品类父级 |
| `product_type_name` | 核心属性，商品品类子级 |
| `garment_group_name` | 核心属性，服装组 |
| `colour_group_name` | 核心属性，具体颜色 |
| `graphical_appearance_name` | 核心属性，图案或外观 |

这份表用于最小可审查版本，也可作为后续只分析核心属性趋势时的输入。

### 稳妥版中间表

输出文件：`data/interim/articles_clean.csv`

稳妥版在 MVP 版字段基础上增加 5 个层级增强字段：

| 字段 | 用途 |
| --- | --- |
| `article_id` | 商品唯一 ID，按字符串保留前导 0 |
| `product_code` | 商品款式族编号，用于审查和后续扩展 |
| `prod_name` | 商品名称，用于展示 |
| `product_group_name` | 核心属性，商品品类父级 |
| `product_type_name` | 核心属性，商品品类子级 |
| `garment_group_name` | 核心属性，服装组 |
| `colour_group_name` | 核心属性，具体颜色 |
| `graphical_appearance_name` | 核心属性，图案或外观 |
| `perceived_colour_master_name` | 层级增强字段，颜色父级 |
| `index_group_name` | 层级增强字段，业务大类父级 |
| `index_name` | 层级增强字段，业务线 |
| `section_name` | 层级增强字段，商品区域 |
| `department_name` | 层级增强字段，部门 |

这份表用于本轮属性图构建。商品-属性边覆盖其中 10 个属性字段，属性层级边使用其中 5 组父子字段。

### 清洗规则

中间表生成时执行以下规则：

- `article_id` 按字符串读取，保留前导 0。
- `product_code` 按字符串保留，避免后续同款族分析时类型不一致。
- 只保留文档定义的字段，不携带 `detail_desc`、编号映射字段或图片字段。
- 对输出字段做缺失值校验；任一必需字段存在缺失值时直接失败。
- 输出列顺序固定，便于后续审查和测试。

## 商品节点表

输出文件：`data/processed/graph/nodes_article.csv`

| 字段 | 生成规则 |
| --- | --- |
| `article_id` | 来自 `data/interim/articles_clean.csv.article_id` |
| `article_node_id` | `article_` + `article_id` |
| `product_code` | 来自 `data/interim/articles_clean.csv.product_code`，用于审查和同款族扩展 |
| `prod_name` | 来自 `data/interim/articles_clean.csv.prod_name`，用于展示 |

每一行 `data/interim/articles_clean.csv` 中的商品生成一个商品节点。

## 属性节点表

输出文件：`data/processed/graph/nodes_attribute.csv`

| 字段 | 生成规则 |
| --- | --- |
| `attr_id` | `attr_type + "::" + attr_value` |
| `attr_type` | `data/interim/articles_clean.csv` 中的属性字段名 |
| `attr_value` | 属性字段对应的取值 |
| `attr_node_id` | 与 `attr_id` 保持一致 |
| `article_count` | 关联到该属性的商品数量 |
| `is_core_attr` | 核心 5 个字段为 `1`，其他层级增强字段为 `0` |
| `level` | 属性在层级图中的角色 |

`level` 取值规则：

| 字段 | `level` |
| --- | --- |
| `product_group_name` | `parent` |
| `product_type_name` | `child` |
| `perceived_colour_master_name` | `parent` |
| `colour_group_name` | `child` |
| `index_group_name` | `parent` |
| `index_name` | `parent_child` |
| `section_name` | `parent_child` |
| `department_name` | `child` |
| `garment_group_name` | `flat` |
| `graphical_appearance_name` | `flat` |

## 商品-属性边表

输出文件：`data/processed/graph/edges_article_attribute.csv`

| 字段 | 生成规则 |
| --- | --- |
| `article_id` | 原始商品 ID |
| `article_node_id` | `article_` + `article_id` |
| `attr_id` | `attr_type + "::" + attr_value` |
| `attr_type` | 属性字段名 |
| `attr_value` | 属性字段取值 |
| `edge_type` | `has_` + 去掉 `_name` 后缀的属性类型 |
| `edge_weight` | 固定为 `1.0` |

这张表基于 `data/interim/articles_clean.csv` 构建，覆盖全部 10 个属性字段。后续趋势主线如果只分析核心属性，可以通过 `nodes_attribute.csv` 中的 `is_core_attr = 1` 过滤，也可以直接使用 `data/interim/articles_clean_mvp.csv` 作为输入重新生成 MVP 图。

## 属性层级边表

输出文件：`data/processed/graph/edges_attribute_hierarchy.csv`

| 字段 | 生成规则 |
| --- | --- |
| `parent_attr_id` | 父属性的 `attr_type::attr_value` |
| `child_attr_id` | 子属性的 `attr_type::attr_value` |
| `parent_attr_type` | 父属性字段名 |
| `child_attr_type` | 子属性字段名 |
| `relation_type` | 固定关系名 |
| `edge_weight` | 父子属性组合在商品表中的共现商品数 |

本轮构建 5 类层级关系：

| 父字段 | 子字段 | `relation_type` |
| --- | --- | --- |
| `product_group_name` | `product_type_name` | `product_group_contains_type` |
| `perceived_colour_master_name` | `colour_group_name` | `colour_master_contains_colour` |
| `index_group_name` | `index_name` | `index_group_contains_index` |
| `index_name` | `section_name` | `index_contains_section` |
| `section_name` | `department_name` | `section_contains_department` |

如果同一个子属性在数据中对应多个父属性，本轮保留多条父子边，并使用 `edge_weight` 记录共现强度。后续如果需要简化为树结构，可以再按 `edge_weight` 选择最强父边。

## 校验规则

清洗脚本遇到以下情况应直接失败：

- `articles.csv` 不存在。
- 必需源字段缺失。
- `article_id` 存在缺失值。
- 任一输出字段存在缺失值。

清洗脚本写出后应验证：

- `articles_clean_mvp.csv` 和 `articles_clean.csv` 行数都等于原始 `articles.csv` 行数。
- 两份中间表的 `article_id` 集合一致。
- 两份中间表的输出列与设计文档完全一致。

构建图脚本遇到以下情况应直接失败：

- `data/interim/articles_clean.csv` 不存在。
- 构建图所需字段缺失。
- `article_id` 存在缺失值。
- 任一配置属性字段存在缺失值。

构建图脚本写出后应验证：

- 商品节点数量等于 `data/interim/articles_clean.csv` 行数。
- 每条商品-属性边都能引用到已存在的商品节点和属性节点。
- 每条属性层级边都能引用到已存在的属性节点。
- 输出 CSV 文件都写入 `data/processed/graph/`。
