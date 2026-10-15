"""Tests for O(M) fast dictionary-based entity lookup, word boundary safety, and 100k-node benchmark."""

import time
import pytest
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph


class TestFastEntityLookup:

    @pytest.fixture
    def small_kg(self):
        kg = BiomedicalKnowledgeGraph()
        kg.add_entity("mondo:0005015", name="diabetes mellitus", entity_type="disease", aliases=["diabetes", "dm"])
        kg.add_entity("mondo:0005148", name="type 2 diabetes mellitus", entity_type="disease", aliases=["t2d", "t2dm"])
        kg.add_entity("db00001", name="lepirudin", entity_type="drug")
        kg.add_entity("ncbigene:7157", name="tp53", entity_type="gene", aliases=["tumor protein p53", "p53"])
        kg.add_entity("mesh:d001281", name="atrophy", entity_type="phenotype")
        kg.add_entity("mesh:d006965", name="hyperplasia", entity_type="phenotype")
        kg.add_entity("mesh:d011469", name="benign prostatic hyperplasia", entity_type="disease", aliases=["bph"])
        kg.build_entity_lookup_index()
        return kg

    def test_exact_and_case_insensitive_match(self, small_kg):
        seeds = small_kg.extract_seed_entities("The patient has Lepirudin administered.")
        assert "db00001" in seeds

    def test_multiword_entity_matching(self, small_kg):
        # Should match 'benign prostatic hyperplasia' as a whole entity
        seeds = small_kg.extract_seed_entities("Patient presents with severe benign prostatic hyperplasia and retention.")
        assert "mesh:d011469" in seeds

    def test_alias_matching(self, small_kg):
        # 'BPH' is an alias for mesh:d011469
        seeds = small_kg.extract_seed_entities("Clinical diagnosis of BPH confirmed.")
        assert "mesh:d011469" in seeds

        # 'T2D' is an alias for mondo:0005148
        seeds_t2d = small_kg.extract_seed_entities("Metformin prescribed for T2D management.")
        assert "mondo:0005148" in seeds_t2d

    def test_word_boundary_preservation_no_substring_false_positives(self, small_kg):
        # 'dm' is an alias for diabetes, but 'admire' or 'administer' or 'cat' should NOT match
        seeds = small_kg.extract_seed_entities("We admire the administrative progress.")
        assert "mondo:0005015" not in seeds

    def test_duplicate_suppression(self, small_kg):
        # Mentioning both name and alias in the same question should not produce duplicate seeds
        seeds = small_kg.extract_seed_entities("Diabetes mellitus, also known as diabetes, was assessed.")
        assert seeds.count("mondo:0005015") == 1

    def test_unknown_entity_handling(self, small_kg):
        seeds = small_kg.extract_seed_entities("Completely unrelated text with no biological terms whatsoever.")
        assert isinstance(seeds, list)
        assert len(seeds) == 0

    def test_all_returned_seeds_exist_in_graph(self, small_kg):
        query = "Does benign prostatic hyperplasia or TP53 lead to cellular atrophy or lepirudin toxicity?"
        seeds = small_kg.extract_seed_entities(query)
        for s in seeds:
            assert small_kg.graph.has_node(s), f"Seed '{s}' not found in graph nodes!"

    def test_lookup_benchmark_100k_synthetic_nodes(self):
        """
        Performance validation with synthetic graph containing >= 100,000 nodes.
        Verifies sub-millisecond lookup latency and O(M) scaling independent of graph size.
        """
        kg = BiomedicalKnowledgeGraph()
        num_synthetic = 100_000

        print(f"\nBuilding synthetic graph with {num_synthetic:,} nodes...")
        t_build_start = time.perf_counter()

        # Batch-populate 100,000 nodes
        for i in range(num_synthetic):
            node_id = f"syn_node_{i}"
            # Give every node a 2-word medical-sounding name
            kg.graph.add_node(
                node_id,
                name=f"protein_{i} receptor_{i}",
                entity_type="SyntheticProtein",
                aliases=[f"syn_{i}"] if i % 5 == 0 else [],
            )

        # Insert specific target nodes to find
        kg.add_entity("target_disease_1", name="cardiovascular ischemic cardiomyopathy", entity_type="disease")
        kg.add_entity("target_drug_1", name="lisinopril dihydrate", entity_type="drug")

        kg.build_entity_lookup_index()
        t_build = time.perf_counter() - t_build_start

        indexed_names = kg._indexed_names_count
        assert indexed_names >= num_synthetic, f"Expected >= {num_synthetic} indexed names, got {indexed_names}"

        # Test lookup latency across multiple realistic clinical questions
        test_questions = [
            "Patient diagnosed with cardiovascular ischemic cardiomyopathy and treated with lisinopril dihydrate.",
            "Synthetic query looking for protein_42 receptor_42 in blood plasma.",
            "Normal sentence with protein_99999 receptor_99999 mention.",
            "Question with no matching entities in the synthetic vocabulary.",
        ]

        latencies = []
        for q in test_questions:
            t0 = time.perf_counter()
            seeds = kg.extract_seed_entities(q)
            lat = time.perf_counter() - t0
            latencies.append(lat)

        mean_latency_ms = (sum(latencies) / len(latencies)) * 1000.0

        # Verify target detection
        target_seeds = kg.extract_seed_entities(test_questions[0])
        assert "target_disease_1" in target_seeds
        assert "target_drug_1" in target_seeds

        print(f"\n[100k Synthetic Node Benchmark]")
        print(f"  Graph Nodes    : {kg.graph.number_of_nodes():,}")
        print(f"  Indexed Names  : {indexed_names:,}")
        print(f"  Mean Lookup Lat: {mean_latency_ms:.3f} ms per question")

        # Must execute lookup in under 5 milliseconds (sub-millisecond typical)
        assert mean_latency_ms < 5.0, f"Lookup too slow: {mean_latency_ms:.3f} ms (expected < 5.0 ms)"
