from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fashion_trend.articles import (
    CLEAN_ARTICLE_COLUMNS,
    MVP_ARTICLE_COLUMNS,
    build_clean_article_frames,
    clean_articles_file,
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

    def test_build_clean_article_frames_rejects_duplicate_article_ids(self) -> None:
        raw_articles = sample_raw_articles()
        raw_articles.loc[1, "article_id"] = raw_articles.loc[0, "article_id"]

        with self.assertRaisesRegex(ValueError, "article_id"):
            build_clean_article_frames(raw_articles)


class CleanArticleFileTests(unittest.TestCase):
    def test_clean_articles_file_writes_mvp_and_clean_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path = tmp_path / "articles.csv"
            mvp_output_path = tmp_path / "articles_clean_mvp.csv"
            clean_output_path = tmp_path / "articles_clean.csv"
            sample_raw_articles().to_csv(raw_path, index=False)

            row_count = clean_articles_file(
                raw_articles_path=raw_path,
                mvp_output_path=mvp_output_path,
                clean_output_path=clean_output_path,
            )

            self.assertEqual(row_count, 2)
            csv_dtype = {"article_id": "string", "product_code": "string"}
            mvp_articles = pd.read_csv(mvp_output_path, dtype=csv_dtype)
            clean_articles = pd.read_csv(clean_output_path, dtype=csv_dtype)
            self.assertEqual(list(mvp_articles.columns), list(MVP_ARTICLE_COLUMNS))
            self.assertEqual(list(clean_articles.columns), list(CLEAN_ARTICLE_COLUMNS))
            self.assertEqual(mvp_articles["article_id"].tolist(), ["0108775015", "0108775044"])
            self.assertEqual(clean_articles["article_id"].tolist(), ["0108775015", "0108775044"])
            self.assertEqual(mvp_articles["product_code"].tolist(), ["0108775", "0108775"])
            self.assertEqual(clean_articles["product_code"].tolist(), ["0108775", "0108775"])

    def test_clean_articles_file_does_not_replace_either_output_when_second_write_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path = tmp_path / "articles.csv"
            mvp_output_path = tmp_path / "mvp" / "articles_clean_mvp.csv"
            clean_output_path = tmp_path / "blocked" / "articles_clean.csv"
            sample_raw_articles().to_csv(raw_path, index=False)
            mvp_output_path.parent.mkdir()
            mvp_output_path.write_text("previous mvp output\n", encoding="utf-8")
            clean_output_path.parent.write_text("not a directory\n", encoding="utf-8")

            with self.assertRaises(OSError):
                clean_articles_file(
                    raw_articles_path=raw_path,
                    mvp_output_path=mvp_output_path,
                    clean_output_path=clean_output_path,
                )

            self.assertEqual(mvp_output_path.read_text(encoding="utf-8"), "previous mvp output\n")
            self.assertFalse(mvp_output_path.with_suffix(".csv.tmp").exists())
            self.assertFalse(clean_output_path.with_suffix(".csv.tmp").exists())

    def test_clean_articles_file_restores_mvp_when_second_final_replace_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path = tmp_path / "articles.csv"
            mvp_output_path = tmp_path / "articles_clean_mvp.csv"
            clean_output_path = tmp_path / "articles_clean.csv"
            sample_raw_articles().to_csv(raw_path, index=False)
            mvp_output_path.write_text("previous mvp output\n", encoding="utf-8")
            clean_output_path.mkdir()

            with self.assertRaises(OSError):
                clean_articles_file(
                    raw_articles_path=raw_path,
                    mvp_output_path=mvp_output_path,
                    clean_output_path=clean_output_path,
                )

            self.assertEqual(mvp_output_path.read_text(encoding="utf-8"), "previous mvp output\n")
            self.assertTrue(clean_output_path.is_dir())
            self.assertFalse(mvp_output_path.with_suffix(".csv.tmp").exists())
            self.assertFalse(clean_output_path.with_suffix(".csv.tmp").exists())
            self.assertFalse(mvp_output_path.with_suffix(".csv.bak").exists())

    def test_clean_articles_file_fails_when_input_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "原始商品文件不存在"):
                clean_articles_file(
                    raw_articles_path=tmp_path / "missing.csv",
                    mvp_output_path=tmp_path / "articles_clean_mvp.csv",
                    clean_output_path=tmp_path / "articles_clean.csv",
                )


if __name__ == "__main__":
    unittest.main()
