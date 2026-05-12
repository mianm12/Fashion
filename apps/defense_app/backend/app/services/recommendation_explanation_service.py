from __future__ import annotations

from app.core.database import not_found
from app.repositories.article_repository import ArticleRepository
from app.repositories.demo_user_repository import DemoUserRepository
from app.repositories.recommendation_repository import RecommendationRepository
from app.repositories.trend_repository import TrendRepository


def build_recommendation_explanation(
    case_id: str,
    article_id: str,
    demo_users: DemoUserRepository,
    articles: ArticleRepository,
    recommendations: RecommendationRepository,
    trends: TrendRepository,
) -> dict[str, object]:
    if demo_users.get(case_id) is None:
        raise not_found("未找到指定演示用户")
    if recommendations.recommendation(case_id, article_id) is None:
        raise not_found("未找到指定推荐商品")

    article = articles.get(article_id)
    score_components = recommendations.score_components(case_id, article_id)
    if article is None or score_components is None:
        raise not_found("未找到指定推荐商品")

    item_attributes = articles.attributes(article_id)
    matching_trends = trends.latest_for_attributes(
        [str(attribute["attr_id"]) for attribute in item_attributes]
    )
    return {
        "case_id": case_id,
        "article": article,
        "user_profile": demo_users.profile(case_id),
        "item_attributes": item_attributes,
        "score_components": score_components,
        "matching_trend_attributes": matching_trends,
    }
