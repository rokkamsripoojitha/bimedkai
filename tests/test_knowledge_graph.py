"""Tests for Knowledge Graph traversal and Hypergraph expansion operations."""

import pytest
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph


class TestBiomedicalKnowledgeGraph:

    @pytest.fixture
    def populated_kg(self, tmp_path):
        kg = BiomedicalKnowledgeGraph(kg_path=str(tmp_path / "test_kg.json"))
        kg.add_relation("bph", "causes", "urinary obstruction", "BPH blocks urine flow.")
        kg.add_relation("urinary obstruction", "causes", "hydronephrosis", "Backpressure dilates pelvis.")
        kg.add_relation("hydronephrosis", "causes", "renal atrophy", "Pressure causes parenchyma atrophy.")

        kg.add_hyperedge(
            hyperedge_id="he_bph_syndrome",
            entity_ids=["bph", "urinary obstruction", "hydronephrosis", "renal atrophy"],
            relation_type="obstructive_syndrome",
            evidence_text="Full disease progression.",
        )
        return kg

    def test_seed_extraction(self, populated_kg):
        text = "Does chronic bph lead to severe complications?"
        seeds = populated_kg.extract_seed_entities(text)
        assert "bph" in seeds

    def test_traversal_depths(self, populated_kg):
        # Depth 1 from bph should reach urinary obstruction
        res1 = populated_kg.traverse_subgraph(["bph"], depth=1)
        assert "urinary obstruction" in res1["visited_nodes"]
        assert "renal atrophy" not in res1["visited_nodes"]

        # Depth 3 from bph should reach renal atrophy
        res3 = populated_kg.traverse_subgraph(["bph"], depth=3)
        assert "renal atrophy" in res3["visited_nodes"]

    def test_hyperedge_expansion(self, populated_kg):
        res = populated_kg.expand_hypergraph(["bph"], depth=1)
        assert len(res["hyperedges"]) > 0
        assert res["hyperedges"][0]["hyperedge_id"] == "he_bph_syndrome"
        assert "hydronephrosis" in res["total_entities_covered"]

    def test_save_and_reload(self, populated_kg, tmp_path):
        save_file = str(tmp_path / "saved_kg.json")
        populated_kg.save(save_file)

        reloaded = BiomedicalKnowledgeGraph(kg_path=save_file)
        assert reloaded.graph.number_of_nodes() == populated_kg.graph.number_of_nodes()
        assert len(reloaded.hyperedges) == len(populated_kg.hyperedges)
