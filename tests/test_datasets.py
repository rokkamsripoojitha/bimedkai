"""Tests for dataset loading, normalization, and reproducible sampling."""

import pytest
from src.shared.datasets import BiomedicalDatasetLoader


class TestBiomedicalDatasetLoader:

    @pytest.fixture(scope="class")
    def loader(self, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("data_test")
        return BiomedicalDatasetLoader(raw_dir=str(tmp_dir / "raw"), processed_dir=str(tmp_dir / "processed"))

    def test_medmcqa_sample_structure(self, loader):
        sample = loader.get_sample("medmcqa", sample_size=5, seed=42)
        assert len(sample) == 5
        item = sample[0]
        assert "id" in item
        assert item["dataset"] == "medmcqa"
        assert "question" in item
        assert "options" in item
        assert set(item["options"].keys()) == {"A", "B", "C", "D"}
        assert item["reference_answer"] in ["A", "B", "C", "D"]

    def test_pubmedqa_sample_structure(self, loader):
        sample = loader.get_sample("pubmedqa", sample_size=5, seed=42)
        assert len(sample) == 5
        item = sample[0]
        assert "id" in item
        assert item["dataset"] == "pubmedqa"
        assert item["reference_answer"] in ["yes", "no", "maybe"]
        assert "context" in item

    def test_sampling_reproducibility(self, loader):
        s1 = loader.get_sample("medmcqa", sample_size=10, seed=123)
        s2 = loader.get_sample("medmcqa", sample_size=10, seed=123)
        assert [x["id"] for x in s1] == [x["id"] for x in s2]
