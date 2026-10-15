"""Tests for PrimeKG CSV dataset loader, schema normalization, canonical ID preservation, and error handling."""

import os
import csv
import pytest
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph


class TestPrimeKGLoader:

    @pytest.fixture
    def synthetic_primekg_csv(self, tmp_path):
        """Create a valid synthetic PrimeKG CSV following official Harvard Dataverse schema."""
        csv_file = str(tmp_path / "primekg_synthetic.csv")
        header = [
            "relation", "display_relation", "x_index", "x_id", "x_type", "x_name", "x_source",
            "y_index", "y_id", "y_type", "y_name", "y_source"
        ]
        rows = [
            # disease -> gene
            ["disease_phenotype_positive", "phenotype positive", "0", "mondo:0005015", "disease", "diabetes mellitus", "MONDO", "1", "ncbigene:7157", "gene/protein", "TP53", "NCBI"],
            # disease -> drug
            ["indication", "indication", "2", "mondo:0005015", "disease", "diabetes mellitus", "MONDO", "3", "db00331", "drug", "metformin", "DrugBank"],
            # disease -> phenotype
            ["disease_phenotype_positive", "phenotype positive", "4", "mondo:0005015", "disease", "diabetes mellitus", "MONDO", "5", "hp:0000822", "effect/phenotype", "hypertension", "HPO"],
            # drug -> gene
            ["target", "target", "6", "db00331", "drug", "metformin", "DrugBank", "7", "ncbigene:5562", "gene/protein", "PRKAA1", "NCBI"],
            # malformed row with missing columns
            ["broken_row", "incomplete"],
            # empty row
            [],
            # row with missing IDs
            ["", "", "", "", "", "", "", "", "", "", "", ""],
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return csv_file

    def test_primekg_loading_and_stats(self, synthetic_primekg_csv):
        kg = BiomedicalKnowledgeGraph(kg_path=None)
        stats = kg.load_primekg_csv(csv_path=synthetic_primekg_csv, build_hyperedges=True)

        assert stats["nodes_added"] >= 4
        assert stats["edges_added"] == 4
        assert stats["skipped_rows"] >= 2
        assert stats["indexed_names_count"] >= 4

        # Verify canonical IDs are preserved in graph
        assert kg.graph.has_node("mondo:0005015")
        assert kg.graph.has_node("db00331")
        assert kg.graph.has_node("ncbigene:7157")
        assert kg.graph.has_node("hp:0000822")

        # Verify attributes
        node_data = kg.graph.nodes["mondo:0005015"]
        assert node_data["name"] == "diabetes mellitus"
        assert node_data["entity_type"] == "disease"

    def test_entity_lookup_from_loaded_primekg(self, synthetic_primekg_csv):
        kg = BiomedicalKnowledgeGraph(kg_path=None)
        kg.load_primekg_csv(csv_path=synthetic_primekg_csv)

        # Lookup by human-readable entity name
        seeds_name = kg.extract_seed_entities("Patient presents with diabetes mellitus and requires intervention.")
        assert "mondo:0005015" in seeds_name

        # Lookup by drug name
        seeds_drug = kg.extract_seed_entities("Evaluate therapeutic efficacy of Metformin.")
        assert "db00331" in seeds_drug

    def test_multi_entity_disease_hyperedge_creation(self, synthetic_primekg_csv):
        kg = BiomedicalKnowledgeGraph(kg_path=None)
        kg.load_primekg_csv(csv_path=synthetic_primekg_csv, build_hyperedges=True)

        # mondo:0005015 has 3 connections (TP53, metformin, hypertension) -> should form hyperedge
        assert "he_primekg_mondo:0005015" in kg.hyperedges
        he = kg.hyperedges["he_primekg_mondo:0005015"]
        assert "mondo:0005015" in he["entities"]
        assert len(he["entities"]) >= 3

    def test_missing_file_raises_actionable_error(self, tmp_path):
        kg = BiomedicalKnowledgeGraph()
        non_existent_path = str(tmp_path / "non_existent_primekg.csv")
        with pytest.raises(FileNotFoundError) as exc_info:
            kg.load_primekg_csv(csv_path=non_existent_path)

        assert "PrimeKG dataset file not found" in str(exc_info.value)
        assert "Harvard Dataverse" in str(exc_info.value)
        assert "--smoke-test" in str(exc_info.value)
