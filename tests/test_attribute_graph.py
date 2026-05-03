from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fashion_trend.articles import (
    ATTRIBUTE_COLUMNS,
    build_article_attribute_edges,
    build_article_nodes,
    build_attribute_hierarchy_edges,
    build_attribute_graph_files,
    build_attribute_nodes,
)


def sample_clean_articles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["0108775015", "0108775044", "0110065001"],
            "product_code": ["0108775", "0108775", "0110065"],
            "prod_name": ["Strap top", "Strap top", "Bra"],
            "product_group_name": ["Garment Upper body", "Garment Upper body", "Underwear"],
            "product_type_name": ["Vest top", "T-shirt", "Bra"],
            "garment_group_name": ["Jersey Basic", "Jersey Basic", "Under-, Nightwear"],
            "colour_group_name": ["Black", "Black", "White"],
            "graphical_appearance_name": ["Solid", "Stripe", "Solid"],
            "perceived_colour_master_name": ["Black", "Black", "White"],
            "index_group_name": ["Ladieswear", "Ladieswear", "Ladieswear"],
            "index_name": ["Ladieswear", "Ladieswear", "Lingeries/Tights"],
            "section_name": [
                "Womens Everyday Basics",
                "Womens Everyday Basics",
                "Womens Lingerie",
            ],
            "department_name": ["Jersey Basic", "Jersey Basic", "Clean Lingerie"],
        }
    )


class AttributeGraphBuilderTests(unittest.TestCase):
    def test_build_article_nodes_returns_one_node_per_article(self) -> None:
        nodes_article = build_article_nodes(sample_clean_articles())

        self.assertEqual(
            nodes_article.columns.tolist(),
            ["article_id", "article_node_id", "product_code", "prod_name"],
        )
        self.assertEqual(len(nodes_article), 3)
        self.assertEqual(nodes_article.loc[0, "article_node_id"], "article_0108775015")

    def test_build_attribute_nodes_counts_articles_and_marks_core_fields(self) -> None:
        nodes_attribute = build_attribute_nodes(sample_clean_articles())

        black_node = nodes_attribute.set_index("attr_id").loc["colour_group_name::Black"]
        self.assertEqual(int(black_node["article_count"]), 2)
        self.assertEqual(int(black_node["is_core_attr"]), 1)
        self.assertEqual(black_node["level"], "child")

        index_node = nodes_attribute.set_index("attr_id").loc["index_name::Ladieswear"]
        self.assertEqual(int(index_node["is_core_attr"]), 0)
        self.assertEqual(index_node["level"], "parent_child")

    def test_build_article_attribute_edges_returns_one_edge_per_article_attribute(self) -> None:
        edges = build_article_attribute_edges(sample_clean_articles())

        self.assertEqual(len(edges), 3 * len(ATTRIBUTE_COLUMNS))
        first_edge = edges[
            (edges["article_id"] == "0108775015")
            & (edges["attr_id"] == "product_group_name::Garment Upper body")
        ].iloc[0]
        self.assertEqual(first_edge["article_node_id"], "article_0108775015")
        self.assertEqual(first_edge["edge_type"], "has_product_group")
        self.assertEqual(float(first_edge["edge_weight"]), 1.0)

    def test_build_attribute_hierarchy_edges_counts_parent_child_cooccurrence(self) -> None:
        hierarchy_edges = build_attribute_hierarchy_edges(sample_clean_articles())

        colour_edge = hierarchy_edges[
            (
                hierarchy_edges["parent_attr_id"]
                == "perceived_colour_master_name::Black"
            )
            & (hierarchy_edges["child_attr_id"] == "colour_group_name::Black")
        ].iloc[0]
        self.assertEqual(colour_edge["relation_type"], "colour_master_contains_colour")
        self.assertEqual(int(colour_edge["edge_weight"]), 2)

        section_edge = hierarchy_edges[
            (hierarchy_edges["parent_attr_id"] == "section_name::Womens Everyday Basics")
            & (hierarchy_edges["child_attr_id"] == "department_name::Jersey Basic")
        ].iloc[0]
        self.assertEqual(section_edge["relation_type"], "section_contains_department")
        self.assertEqual(int(section_edge["edge_weight"]), 2)

    def test_graph_builders_reject_missing_attribute_values(self) -> None:
        builders = (
            build_attribute_nodes,
            build_article_attribute_edges,
            build_attribute_hierarchy_edges,
        )

        for builder in builders:
            with self.subTest(builder=builder.__name__):
                clean_articles = sample_clean_articles()
                clean_articles.loc[0, "colour_group_name"] = pd.NA

                with self.assertRaisesRegex(ValueError, "colour_group_name"):
                    builder(clean_articles)


class AttributeGraphFileTests(unittest.TestCase):
    def test_build_attribute_graph_files_writes_all_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            clean_articles_path = tmp_path / "articles_clean.csv"
            output_dir = tmp_path / "graph"
            sample_clean_articles().to_csv(clean_articles_path, index=False)

            output_counts = build_attribute_graph_files(
                clean_articles_path=clean_articles_path,
                graph_dir=output_dir,
            )

            self.assertEqual(output_counts["nodes_article"], 3)
            self.assertEqual(
                output_counts["edges_article_attribute"],
                3 * len(ATTRIBUTE_COLUMNS),
            )
            self.assertTrue((output_dir / "nodes_article.csv").exists())
            self.assertTrue((output_dir / "nodes_attribute.csv").exists())
            self.assertTrue((output_dir / "edges_article_attribute.csv").exists())
            self.assertTrue((output_dir / "edges_attribute_hierarchy.csv").exists())

            nodes_attribute = pd.read_csv(output_dir / "nodes_attribute.csv")
            edges = pd.read_csv(output_dir / "edges_article_attribute.csv")
            self.assertTrue(set(edges["attr_id"]).issubset(set(nodes_attribute["attr_id"])))

    def test_build_attribute_graph_files_fails_when_clean_input_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "商品 clean 文件不存在"):
                build_attribute_graph_files(
                    clean_articles_path=tmp_path / "articles_clean.csv",
                    graph_dir=tmp_path / "graph",
                )

    def test_build_attribute_graph_files_rolls_back_partial_publish_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            clean_articles_path = tmp_path / "articles_clean.csv"
            output_dir = tmp_path / "graph"
            output_dir.mkdir()
            sample_clean_articles().to_csv(clean_articles_path, index=False)

            nodes_article_path = output_dir / "nodes_article.csv"
            nodes_attribute_path = output_dir / "nodes_attribute.csv"
            edges_article_attribute_path = output_dir / "edges_article_attribute.csv"
            edges_attribute_hierarchy_path = (
                output_dir / "edges_attribute_hierarchy.csv"
            )
            old_nodes_article = "article_id,article_node_id\nold,article_old\n"
            old_nodes_attribute = "attr_id,attr_type\nold_attr,old_type\n"
            nodes_article_path.write_text(old_nodes_article)
            nodes_attribute_path.write_text(old_nodes_attribute)
            edges_article_attribute_path.mkdir()

            with self.assertRaises(OSError):
                build_attribute_graph_files(
                    clean_articles_path=clean_articles_path,
                    graph_dir=output_dir,
                )

            self.assertEqual(nodes_article_path.read_text(), old_nodes_article)
            self.assertEqual(nodes_attribute_path.read_text(), old_nodes_attribute)
            self.assertTrue(edges_article_attribute_path.is_dir())
            self.assertFalse(edges_attribute_hierarchy_path.exists())
            self.assertEqual(list(output_dir.glob("*.tmp")), [])
            self.assertEqual(list(output_dir.glob("*.bak")), [])


if __name__ == "__main__":
    unittest.main()
