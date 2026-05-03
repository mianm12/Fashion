from __future__ import annotations

import unittest

import pandas as pd

from fashion_trend.articles import (
    CLEAN_ARTICLE_COLUMNS,
    MVP_ARTICLE_COLUMNS,
    build_clean_article_frames,
    validate_required_columns,
)


def sample_raw_articles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["0108775015", "0108775044"],
            "product_code": ["0108775", "0108775"],
            "prod_name": ["Strap top", "Strap top"],
            "product_group_name": ["Garment Upper body", "Garment Upper body"],
            "product_type_name": ["Vest top", "Vest top"],
            "garment_group_name": ["Jersey Basic", "Jersey Basic"],
            "colour_group_name": ["Black", "White"],
            "graphical_appearance_name": ["Solid", "Solid"],
            "perceived_colour_master_name": ["Black", "White"],
            "index_group_name": ["Ladieswear", "Ladieswear"],
            "index_name": ["Ladieswear", "Ladieswear"],
            "section_name": ["Womens Everyday Basics", "Womens Everyday Basics"],
            "department_name": ["Jersey Basic", "Jersey Basic"],
            "detail_desc": ["Ignored text", "Ignored text"],
        }
    )


class CleanArticleFrameTests(unittest.TestCase):
    def test_build_clean_article_frames_returns_fixed_columns(self) -> None:
        mvp_articles, clean_articles = build_clean_article_frames(sample_raw_articles())

        self.assertEqual(list(mvp_articles.columns), list(MVP_ARTICLE_COLUMNS))
        self.assertEqual(list(clean_articles.columns), list(CLEAN_ARTICLE_COLUMNS))
        self.assertEqual(len(mvp_articles), 2)
        self.assertEqual(len(clean_articles), 2)
        self.assertEqual(mvp_articles["article_id"].tolist(), ["0108775015", "0108775044"])
        self.assertNotIn("detail_desc", mvp_articles.columns)
        self.assertNotIn("detail_desc", clean_articles.columns)

    def test_build_clean_article_frames_casts_identifier_columns_to_string(self) -> None:
        raw_articles = sample_raw_articles()
        raw_articles["article_id"] = pd.Series(["0108775015", "0108775044"], dtype="string")
        raw_articles["product_code"] = pd.Series(["0108775", "0108775"], dtype="string")

        mvp_articles, clean_articles = build_clean_article_frames(raw_articles)

        self.assertEqual(mvp_articles["article_id"].dtype.name, "string")
        self.assertEqual(mvp_articles["product_code"].dtype.name, "string")
        self.assertEqual(clean_articles["article_id"].dtype.name, "string")
        self.assertEqual(clean_articles["product_code"].dtype.name, "string")

    def test_validate_required_columns_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少必要字段: product_type_name"):
            validate_required_columns(
                ["article_id", "product_group_name"],
                ["article_id", "product_group_name", "product_type_name"],
                source_name="测试 articles 表",
            )

    def test_build_clean_article_frames_rejects_missing_values(self) -> None:
        raw_articles = sample_raw_articles()
        raw_articles.loc[0, "colour_group_name"] = pd.NA

        with self.assertRaisesRegex(ValueError, "colour_group_name"):
            build_clean_article_frames(raw_articles)


if __name__ == "__main__":
    unittest.main()
