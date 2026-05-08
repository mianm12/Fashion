from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fashion_trend.catalog.articles import ATTRIBUTE_COLUMNS
from fashion_trend.catalog.graph import build_attribute_graph_files
from fashion_trend.catalog.graph.builders import (
    build_article_attribute_edges,
    build_article_nodes,
    build_attribute_graph_frames,
    build_attribute_hierarchy_edges,
    build_attribute_nodes,
)
from fashion_trend.catalog.graph.publishing import publish_graph_frames


def sample_clean_articles() -> pd.DataFrame:
    """构造属性图阶段使用的 clean articles 输入样本。"""
    return pd.DataFrame(
        {
            "article_id": ["0108775015", "0108775044", "0110065001"],
            "product_code": ["0108775", "0108775", "0110065"],
            "prod_name": ["Strap top", "Strap top", "Bra"],
            "product_group_name": [
                "Garment Upper body",
                "Garment Upper body",
                "Underwear",
            ],
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


class TestAttributeGraphBuilder:
    def test_build_article_nodes_returns_one_node_per_article(self) -> None:
        nodes_article = build_article_nodes(sample_clean_articles())

        assert nodes_article.columns.tolist() == [
            "article_id",
            "article_node_id",
            "product_code",
            "prod_name",
        ]
        assert len(nodes_article) == 3
        assert nodes_article.loc[0, "article_node_id"] == "article_0108775015"

    def test_build_attribute_nodes_counts_articles_and_marks_core_fields(self) -> None:
        nodes_attribute = build_attribute_nodes(sample_clean_articles())

        black_node = nodes_attribute.set_index("attr_id").loc[
            "colour_group_name::Black"
        ]
        assert int(black_node["article_count"]) == 2
        assert int(black_node["is_core_attr"]) == 1
        assert black_node["level"] == "child"

        index_node = nodes_attribute.set_index("attr_id").loc["index_name::Ladieswear"]
        assert int(index_node["is_core_attr"]) == 0
        assert index_node["level"] == "parent_child"

    def test_build_article_attribute_edges_returns_one_edge_per_article_attribute(
        self,
    ) -> None:
        edges = build_article_attribute_edges(sample_clean_articles())

        assert len(edges) == 3 * len(ATTRIBUTE_COLUMNS)
        first_edge = edges[
            (edges["article_id"] == "0108775015")
            & (edges["attr_id"] == "product_group_name::Garment Upper body")
        ].iloc[0]
        assert first_edge["article_node_id"] == "article_0108775015"
        assert first_edge["edge_type"] == "has_product_group"
        assert float(first_edge["edge_weight"]) == 1.0

    def test_build_attribute_hierarchy_edges_counts_parent_child_cooccurrence(
        self,
    ) -> None:
        hierarchy_edges = build_attribute_hierarchy_edges(sample_clean_articles())

        colour_edge = hierarchy_edges[
            (hierarchy_edges["parent_attr_id"] == "perceived_colour_master_name::Black")
            & (hierarchy_edges["child_attr_id"] == "colour_group_name::Black")
        ].iloc[0]
        assert colour_edge["relation_type"] == "colour_master_contains_colour"
        assert int(colour_edge["edge_weight"]) == 2

        section_edge = hierarchy_edges[
            (
                hierarchy_edges["parent_attr_id"]
                == "section_name::Womens Everyday Basics"
            )
            & (hierarchy_edges["child_attr_id"] == "department_name::Jersey Basic")
        ].iloc[0]
        assert section_edge["relation_type"] == "section_contains_department"
        assert int(section_edge["edge_weight"]) == 2

    @pytest.mark.parametrize(
        "builder",
        [
            build_attribute_nodes,
            build_article_attribute_edges,
            build_attribute_hierarchy_edges,
        ],
    )
    def test_graph_builders_reject_missing_attribute_values(self, builder) -> None:
        clean_articles = sample_clean_articles()
        clean_articles.loc[0, "colour_group_name"] = pd.NA

        with pytest.raises(ValueError, match="colour_group_name"):
            builder(clean_articles)

    def test_build_attribute_graph_frames_rejects_duplicate_article_ids(self) -> None:
        clean_articles = sample_clean_articles()
        clean_articles.loc[1, "article_id"] = clean_articles.loc[0, "article_id"]

        with pytest.raises(ValueError, match="article_id"):
            build_attribute_graph_frames(clean_articles)


class TestAttributeGraphFile:
    def test_build_attribute_graph_files_writes_all_outputs(
        self, tmp_path: Path
    ) -> None:
        clean_articles_path = tmp_path / "articles_clean.csv"
        output_dir = tmp_path / "graph"
        sample_clean_articles().to_csv(clean_articles_path, index=False)

        output_counts = build_attribute_graph_files(
            clean_articles_path=clean_articles_path,
            graph_dir=output_dir,
        )

        assert output_counts["nodes_article"] == 3
        assert output_counts["edges_article_attribute"] == 3 * len(ATTRIBUTE_COLUMNS)
        assert (output_dir / "nodes_article.csv").exists()
        assert (output_dir / "nodes_attribute.csv").exists()
        assert (output_dir / "edges_article_attribute.csv").exists()
        assert (output_dir / "edges_attribute_hierarchy.csv").exists()

        nodes_attribute = pd.read_csv(output_dir / "nodes_attribute.csv")
        edges = pd.read_csv(output_dir / "edges_article_attribute.csv")
        assert set(edges["attr_id"]).issubset(set(nodes_attribute["attr_id"]))

    def test_graph_outputs_quote_all_fields_for_csv_auto_detection(
        self, tmp_path: Path
    ) -> None:
        clean_articles_path = tmp_path / "articles_clean.csv"
        output_dir = tmp_path / "graph"
        sample_clean_articles().to_csv(clean_articles_path, index=False)

        build_attribute_graph_files(
            clean_articles_path=clean_articles_path,
            graph_dir=output_dir,
        )

        edges_lines = (
            (output_dir / "edges_article_attribute.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        comma_value_line = next(
            line for line in edges_lines if "Under-, Nightwear" in line
        )
        assert (
            edges_lines[0]
            == '"article_id","article_node_id","attr_id","attr_type","attr_value","edge_type","edge_weight"'
        )
        assert '"garment_group_name::Under-, Nightwear"' in comma_value_line
        assert '"Under-, Nightwear"' in comma_value_line

    def test_build_attribute_graph_files_fails_when_clean_input_missing(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="商品 clean 文件不存在"):
            build_attribute_graph_files(
                clean_articles_path=tmp_path / "articles_clean.csv",
                graph_dir=tmp_path / "graph",
            )

    def test_build_attribute_graph_files_rolls_back_partial_publish_failure(
        self, tmp_path: Path
    ) -> None:
        clean_articles_path = tmp_path / "articles_clean.csv"
        output_dir = tmp_path / "graph"
        output_dir.mkdir()
        sample_clean_articles().to_csv(clean_articles_path, index=False)

        nodes_article_path = output_dir / "nodes_article.csv"
        nodes_attribute_path = output_dir / "nodes_attribute.csv"
        edges_article_attribute_path = output_dir / "edges_article_attribute.csv"
        edges_attribute_hierarchy_path = output_dir / "edges_attribute_hierarchy.csv"
        old_nodes_article = "article_id,article_node_id\nold,article_old\n"
        old_nodes_attribute = "attr_id,attr_type\nold_attr,old_type\n"
        nodes_article_path.write_text(old_nodes_article)
        nodes_attribute_path.write_text(old_nodes_attribute)
        edges_article_attribute_path.mkdir()

        # 预置目录占用其中一个目标路径，强制发布中途失败并覆盖回滚路径。
        with pytest.raises(OSError):
            build_attribute_graph_files(
                clean_articles_path=clean_articles_path,
                graph_dir=output_dir,
            )

        assert nodes_article_path.read_text() == old_nodes_article
        assert nodes_attribute_path.read_text() == old_nodes_attribute
        assert edges_article_attribute_path.is_dir()
        assert not edges_attribute_hierarchy_path.exists()
        assert list(output_dir.glob("*.tmp")) == []
        assert list(output_dir.glob("*.bak")) == []

    def test_publish_graph_frames_keeps_unpublished_outputs_on_backup_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_dir = tmp_path / "graph"
        output_dir.mkdir()
        output_paths = {
            "nodes_article": output_dir / "nodes_article.csv",
            "nodes_attribute": output_dir / "nodes_attribute.csv",
            "edges_article_attribute": output_dir / "edges_article_attribute.csv",
            "edges_attribute_hierarchy": output_dir / "edges_attribute_hierarchy.csv",
        }
        old_contents = {
            graph_name: f"old {graph_name}\n" for graph_name in output_paths
        }
        for graph_name, output_path in output_paths.items():
            output_path.write_text(old_contents[graph_name], encoding="utf-8")

        graph_frames = {
            graph_name: pd.DataFrame({"value": [f"new {graph_name}"]})
            for graph_name in output_paths
        }
        original_replace = Path.replace

        def fail_nodes_attribute_backup(self: Path, target: Path) -> Path:
            if (
                self.name == "nodes_attribute.csv"
                and target.name == "nodes_attribute.csv.bak"
            ):
                raise OSError("simulated backup failure")
            return original_replace(self, target)

        # 只让第二个备份动作失败，验证未发布文件和旧文件都能保持原状。
        monkeypatch.setattr(Path, "replace", fail_nodes_attribute_backup)

        with pytest.raises(OSError, match="simulated backup failure"):
            publish_graph_frames(graph_frames, output_dir)

        for graph_name, output_path in output_paths.items():
            assert output_path.read_text(encoding="utf-8") == old_contents[graph_name]
        assert list(output_dir.glob("*.tmp")) == []
        assert list(output_dir.glob("*.bak")) == []
