from __future__ import annotations

from typing import Any

import pandas as pd

CASE_KEY_COLUMNS = ("customer_id", "split", "cutoff_week", "label_week")
CASE_EXPLANATION_COLUMNS = (
    "pop_score",
    "sim_score",
    "trend_score",
    "recent_score",
    "candidate_sources",
)
CASE_EXPLANATORY_PROFILE_ATTR_TYPES = (
    "graphical_appearance_name",
    "product_group_name",
    "colour_group_name",
)
CORE_ARTICLE_ATTR_TYPES = (
    "product_group_name",
    "product_type_name",
    "graphical_appearance_name",
    "colour_group_name",
    "department_name",
)


def select_recommendation_cases(
    *,
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    user_profile: pd.DataFrame,
    case_count: int,
) -> list[tuple[str, str, int, int]]:
    """Select reproducible recommendation windows for paper case studies."""
    if case_count <= 0:
        raise ValueError("case_count 必须为正整数。")
    _require_columns(recommendation_items, "recommendation_items", _item_columns())
    _require_columns(
        evaluation_labels, "evaluation_labels", (*CASE_KEY_COLUMNS, "article_id")
    )
    _require_columns(user_profile, "user_profile", _profile_columns())

    test_items = _test_split(recommendation_items).copy()
    _reject_duplicate_top_articles(test_items)
    test_labels = _test_split(evaluation_labels)
    explanatory_profiles = _test_split(user_profile).loc[
        lambda frame: frame["attr_type"]
        .astype(str)
        .isin(CASE_EXPLANATORY_PROFILE_ATTR_TYPES)
    ]

    hits = test_items.merge(
        test_labels.loc[:, [*CASE_KEY_COLUMNS, "article_id"]],
        on=[*CASE_KEY_COLUMNS, "article_id"],
        how="left",
        indicator=True,
    )
    hits["is_hit"] = hits["_merge"] == "both"
    case_stats = (
        hits.groupby(list(CASE_KEY_COLUMNS), as_index=False)
        .agg(
            hit_count=("is_hit", "sum"),
            recommendation_count=("article_id", "size"),
        )
        .merge(_complete_explanation_counts(test_items), on=list(CASE_KEY_COLUMNS))
    )
    profile_counts = (
        explanatory_profiles.groupby(list(CASE_KEY_COLUMNS), as_index=False)
        .size()
        .rename(columns={"size": "explanatory_profile_count"})
    )
    candidates = case_stats.merge(
        profile_counts, on=list(CASE_KEY_COLUMNS), how="inner"
    )
    candidates = candidates.loc[
        (candidates["recommendation_count"] == candidates["complete_item_count"])
        & (candidates["explanatory_profile_count"] > 0)
    ]
    candidates = candidates.sort_values(
        ["hit_count", "explanatory_profile_count", "customer_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if len(candidates) < case_count:
        raise ValueError(f"不足 {case_count} 个推荐案例。")
    return [_case_key_from_row(row) for row in candidates.head(case_count).itertuples()]


def build_case_payload(
    *,
    case_key: tuple[str, str, int, int],
    recommendation_items: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    user_profile: pd.DataFrame,
    article_attributes: pd.DataFrame,
    representative_trends: pd.DataFrame,
) -> dict[str, Any]:
    """Build a JSON-serializable case payload for one recommendation window."""
    _require_columns(recommendation_items, "recommendation_items", _item_columns())
    _require_columns(
        evaluation_labels, "evaluation_labels", (*CASE_KEY_COLUMNS, "article_id")
    )
    _require_columns(user_profile, "user_profile", _profile_columns())

    items = recommendation_items.loc[_case_mask(recommendation_items, case_key)].copy()
    if items.empty:
        raise ValueError(f"推荐案例缺少推荐商品: {case_key}")
    _reject_duplicate_top_articles(items)
    labels = set(
        evaluation_labels.loc[
            _case_mask(evaluation_labels, case_key), "article_id"
        ].astype(str)
    )
    profile = user_profile.loc[_case_mask(user_profile, case_key)].sort_values(
        ["preference_score", "purchase_count", "attr_type", "attr_value"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    recommendations = _recommendation_rows(
        items.sort_values("rank", kind="mergesort"),
        labels=labels,
        attrs_by_article=_article_attributes_by_article(article_attributes),
    )
    customer_id, split, cutoff_week, label_week = case_key
    return {
        "customer_id": customer_id,
        "split": split,
        "cutoff_week": cutoff_week,
        "label_week": label_week,
        "window_id": f"{split}:{cutoff_week}:{label_week}",
        "hit_count": sum(1 for row in recommendations if row["is_hit"]),
        "profile": _profile_rows(profile),
        "representative_trends": _representative_trend_rows(
            representative_trends,
            week_id=cutoff_week,
        ),
        "recommendations": recommendations,
    }


def render_case_markdown(payload: dict[str, Any]) -> str:
    """Render a case payload as Markdown for paper and defense material reuse."""
    lines = [
        f"# 推荐案例 {payload['customer_id']}",
        "",
        f"- split: {payload['split']}",
        f"- cutoff_week: {payload['cutoff_week']}",
        f"- label_week: {payload['label_week']}",
        f"- hit_count: {payload['hit_count']}",
        "",
        "## 用户偏好属性",
    ]
    lines.extend(_profile_markdown_lines(payload["profile"]))
    lines.extend(["", "## 代表性趋势属性"])
    lines.extend(_trend_markdown_lines(payload["representative_trends"]))
    lines.extend(["", "## 推荐商品与解释"])
    lines.extend(_recommendation_markdown_lines(payload["recommendations"]))
    lines.extend(["", "## 简短案例解读", _case_summary(payload)])
    return "\n".join(lines) + "\n"


def _item_columns() -> tuple[str, ...]:
    return (
        *CASE_KEY_COLUMNS,
        "method",
        "article_id",
        "rank",
        "score",
        *CASE_EXPLANATION_COLUMNS,
    )


def _profile_columns() -> tuple[str, ...]:
    return (
        *CASE_KEY_COLUMNS,
        "attr_id",
        "attr_type",
        "attr_value",
        "preference_score",
        "purchase_count",
        "last_purchase_week",
    )


def _test_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.loc[dataframe["split"].astype(str) == "test"]


def _complete_explanation_counts(items: pd.DataFrame) -> pd.DataFrame:
    complete = items.loc[:, list(CASE_EXPLANATION_COLUMNS)].notna().all(axis=1)
    return (
        items.assign(_complete_item=complete)
        .groupby(list(CASE_KEY_COLUMNS), as_index=False)["_complete_item"]
        .sum()
        .rename(columns={"_complete_item": "complete_item_count"})
    )


def _case_key_from_row(row: Any) -> tuple[str, str, int, int]:
    return (
        str(row.customer_id),
        str(row.split),
        int(row.cutoff_week),
        int(row.label_week),
    )


def _case_mask(
    dataframe: pd.DataFrame, case_key: tuple[str, str, int, int]
) -> pd.Series:
    customer_id, split, cutoff_week, label_week = case_key
    return (
        (dataframe["customer_id"].astype(str) == customer_id)
        & (dataframe["split"].astype(str) == split)
        & (dataframe["cutoff_week"].astype(int) == cutoff_week)
        & (dataframe["label_week"].astype(int) == label_week)
    )


def _recommendation_rows(
    items: pd.DataFrame,
    *,
    labels: set[str],
    attrs_by_article: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in items.itertuples(index=False):
        article_id = str(row.article_id)
        rows.append(
            {
                "rank": int(row.rank),
                "article_id": article_id,
                "is_hit": article_id in labels,
                "candidate_sources": str(row.candidate_sources),
                "score_decomposition": {
                    "score": float(row.score),
                    "pop_score": float(row.pop_score),
                    "sim_score": float(row.sim_score),
                    "trend_score": float(row.trend_score),
                    "recent_score": float(row.recent_score),
                },
                "attributes": attrs_by_article.get(article_id, {}),
            }
        )
    return rows


def _profile_rows(profile: pd.DataFrame, *, top_n: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "attr_type": str(row.attr_type),
            "attr_value": str(row.attr_value),
            "preference_score": float(row.preference_score),
            "purchase_count": int(row.purchase_count),
            "last_purchase_week": int(row.last_purchase_week),
        }
        for row in profile.head(top_n).itertuples(index=False)
    ]


def _article_attributes_by_article(
    article_attributes: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    _require_columns(
        article_attributes,
        "article_attributes",
        ("article_id", "attr_type", "attr_value"),
    )
    scoped = article_attributes.loc[
        article_attributes["attr_type"].astype(str).isin(CORE_ARTICLE_ATTR_TYPES)
    ]
    result: dict[str, dict[str, str]] = {}
    for row in scoped.itertuples(index=False):
        result.setdefault(str(row.article_id), {})[str(row.attr_type)] = str(
            row.attr_value
        )
    return result


def _representative_trend_rows(
    representative_trends: pd.DataFrame,
    *,
    week_id: int,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    required = ("week_id", "attr_type", "attr_value", "pred_target_growth", "heat_t")
    _require_columns(representative_trends, "representative_trends", required)
    scoped = representative_trends.loc[
        representative_trends["week_id"].astype(int) == week_id
    ].sort_values("pred_target_growth", ascending=False, kind="mergesort")
    if scoped.empty:
        raise ValueError(f"week_id={week_id} 缺少代表性趋势属性。")
    return [
        {
            "attr_type": str(row.attr_type),
            "attr_value": str(row.attr_value),
            "pred_target_growth": float(row.pred_target_growth),
            "heat_t": float(row.heat_t),
        }
        for row in scoped.head(top_n).itertuples(index=False)
    ]


def _reject_duplicate_top_articles(items: pd.DataFrame) -> None:
    top_items = items.loc[items["rank"].astype(int) <= 12]
    duplicated = top_items.duplicated([*CASE_KEY_COLUMNS, "article_id"], keep=False)
    if duplicated.any():
        sample = top_items.loc[duplicated, [*CASE_KEY_COLUMNS, "article_id"]]
        raise ValueError(
            f"推荐案例 Top-12 商品存在重复: {sample.head(3).to_dict('records')}"
        )


def _require_columns(
    dataframe: pd.DataFrame,
    artifact_name: str,
    required_columns: tuple[str, ...],
) -> None:
    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{artifact_name} 缺少列: {missing}")


def _profile_markdown_lines(rows: list[dict[str, Any]]) -> list[str]:
    return [
        "- {attr_type}: {attr_value} "
        "(score={preference_score:.4f}, count={purchase_count})".format(**row)
        for row in rows
    ]


def _trend_markdown_lines(rows: list[dict[str, Any]]) -> list[str]:
    return [
        "- {attr_type}: {attr_value} "
        "(pred_growth={pred_target_growth:.4f}, heat_t={heat_t:.2f})".format(**row)
        for row in rows
    ]


def _recommendation_markdown_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        scores = row["score_decomposition"]
        attrs = ", ".join(
            f"{name}={value}" for name, value in sorted(row["attributes"].items())
        )
        lines.append(
            "- rank {rank}: {article_id} hit={is_hit} score={score:.4f}; "
            "pop={pop_score:.4f}, sim={sim_score:.4f}, trend={trend_score:.4f}, "
            "recent={recent_score:.4f}; sources={candidate_sources}; "
            "商品属性: {attrs}".format(attrs=attrs or "未补全", **row, **scores)
        )
    return lines


def _case_summary(payload: dict[str, Any]) -> str:
    return (
        f"该用户窗口命中 {payload['hit_count']} 个目标商品，推荐解释同时展示用户历史偏好、"
        "代表性趋势属性、商品属性与分数分解。"
    )
