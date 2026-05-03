# Articles Attribute Graph Design

## Scope

Build the static article attribute graph from `articles.csv`.

This round produces graph files only. It does not compute article weekly sales,
attribute weekly heat, trend labels, model features, or recommendations.

The graph is stored as CSV tables:

- `data/processed/graph/nodes_article.csv`
- `data/processed/graph/nodes_attribute.csv`
- `data/processed/graph/edges_article_attribute.csv`
- `data/processed/graph/edges_attribute_hierarchy.csv`

## Attribute Field Set

Use 10 fields from `articles.csv`.

Core attributes for the trend main line:

| Field | Role |
| --- | --- |
| `product_group_name` | Product category parent |
| `product_type_name` | Product category child |
| `garment_group_name` | Flat garment group attribute |
| `colour_group_name` | Specific color child |
| `graphical_appearance_name` | Flat pattern or appearance attribute |

Hierarchy attributes used to build parent-child edges:

| Field | Role |
| --- | --- |
| `perceived_colour_master_name` | Color parent |
| `index_group_name` | Business group parent |
| `index_name` | Business line parent and child |
| `section_name` | Section parent and child |
| `department_name` | Department child |

`detail_desc` is out of scope for this round.

## Article Nodes

Output: `data/processed/graph/nodes_article.csv`

| Column | Rule |
| --- | --- |
| `article_id` | `articles.csv.article_id`, read as string to preserve leading zeros |
| `article_node_id` | `article_` + `article_id` |
| `product_code` | `articles.csv.product_code`, retained for inspection |
| `prod_name` | `articles.csv.prod_name`, retained for display |

One input article produces one article node.

## Attribute Nodes

Output: `data/processed/graph/nodes_attribute.csv`

| Column | Rule |
| --- | --- |
| `attr_id` | `attr_type + "::" + attr_value` |
| `attr_type` | Source column name from `articles.csv` |
| `attr_value` | Source column value |
| `attr_node_id` | Same as `attr_id` |
| `article_count` | Number of articles associated with this attribute |
| `is_core_attr` | `1` for the 5 core fields, otherwise `0` |
| `level` | Attribute hierarchy role |

`level` values:

| Field | level |
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

## Article-Attribute Edges

Output: `data/processed/graph/edges_article_attribute.csv`

| Column | Rule |
| --- | --- |
| `article_id` | Source article ID |
| `article_node_id` | `article_` + `article_id` |
| `attr_id` | `attr_type + "::" + attr_value` |
| `attr_type` | Source attribute column name |
| `attr_value` | Source attribute value |
| `edge_type` | `has_` + attribute type without the `_name` suffix |
| `edge_weight` | Constant `1.0` |

The table covers all 10 attribute fields. Later trend steps may filter to
`is_core_attr = 1` when computing the main trend target.

## Attribute Hierarchy Edges

Output: `data/processed/graph/edges_attribute_hierarchy.csv`

| Column | Rule |
| --- | --- |
| `parent_attr_id` | Parent `attr_type::attr_value` |
| `child_attr_id` | Child `attr_type::attr_value` |
| `parent_attr_type` | Parent source column name |
| `child_attr_type` | Child source column name |
| `relation_type` | Fixed relation name |
| `edge_weight` | Number of articles where the parent-child pair co-occurs |

Build these hierarchy relations:

| Parent field | Child field | relation_type |
| --- | --- | --- |
| `product_group_name` | `product_type_name` | `product_group_contains_type` |
| `perceived_colour_master_name` | `colour_group_name` | `colour_master_contains_colour` |
| `index_group_name` | `index_name` | `index_group_contains_index` |
| `index_name` | `section_name` | `index_contains_section` |
| `section_name` | `department_name` | `section_contains_department` |

The graph keeps multiple parent edges for a child when the data contains them.
The `edge_weight` records the observed co-occurrence strength, so later steps
can either keep all parent links or select the strongest parent if needed.

## Validation Rules

The build script should fail fast when:

- `articles.csv` is missing.
- Required source columns are missing.
- `article_id` contains missing values.
- Any configured attribute field contains missing values.

The build script should verify after writing:

- Article node count equals the number of rows in `articles.csv`.
- Every article-attribute edge references an existing article node and
  attribute node.
- Every hierarchy edge references existing attribute nodes.
- Output CSV files are written under `data/processed/graph/`.
