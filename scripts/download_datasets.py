"""Download, inspect, and prepare deterministic evaluation splits for MedMCQA and PubMedQA."""

import sys
import os
# Ensure root workspace directory is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.datasets import BiomedicalDatasetLoader
from src.shared.logging_utils import setup_logger

logger = setup_logger("biomedkai.download")


def download_and_prepare_datasets(seeds=[42, 123, 999], sample_size=300, pilot_size=20):
    loader = BiomedicalDatasetLoader()

    logger.info("Loading full MedMCQA validation dataset...")
    medmcqa_full = loader.load_medmcqa(split="validation")
    logger.info(f"Loaded {len(medmcqa_full)} valid MedMCQA samples.")

    logger.info("Loading full PubMedQA labeled dataset...")
    pubmedqa_full = loader.load_pubmedqa(split="train")
    logger.info(f"Loaded {len(pubmedqa_full)} valid PubMedQA samples.")

    # Generate pilot splits
    logger.info(f"Generating pilot splits (N={pilot_size})...")
    pilot_med = loader.get_sample("medmcqa", sample_size=pilot_size, seed=42)
    pilot_pub = loader.get_sample("pubmedqa", sample_size=pilot_size, seed=42)
    logger.info(f"Pilot samples generated: MedMCQA={len(pilot_med)}, PubMedQA={len(pilot_pub)}")

    # Generate full experiment evaluation splits for each seed
    for seed in seeds:
        logger.info(f"Generating evaluation splits for seed={seed} (target N={sample_size})...")
        s_med = loader.get_sample("medmcqa", sample_size=sample_size, seed=seed)
        s_pub = loader.get_sample("pubmedqa", sample_size=sample_size, seed=seed)
        logger.info(f"Seed {seed}: MedMCQA={len(s_med)} items, PubMedQA={len(s_pub)} items.")

    logger.info("Dataset preparation complete! All sample splits persisted to data/processed/")


if __name__ == "__main__":
    download_and_prepare_datasets()
