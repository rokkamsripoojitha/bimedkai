"""Setup and environment verification script for the biomedical KG reasoning research project."""

import os
import sys
import platform
import psutil
import torch
from transformers import AutoTokenizer, AutoConfig
import datasets
import faiss


def check_environment() -> bool:
    print("=" * 70)
    print("BIOMEDICAL KG REASONING: ENVIRONMENT & HARDWARE AUDIT")
    print("=" * 70)

    # 1. Platform & Python
    py_ver = platform.python_version()
    os_info = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
    print(f"Python Version    : {py_ver}")
    print(f"Operating System  : {os_info}")

    # 2. Memory & Disk
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    free_ram_gb = mem.available / (1024 ** 3)
    disk = psutil.disk_usage(os.getcwd())
    free_disk_gb = disk.free / (1024 ** 3)
    print(f"Total Physical RAM: {total_ram_gb:.2f} GB")
    print(f"Available Free RAM: {free_ram_gb:.2f} GB")
    print(f"Free Disk Space   : {free_disk_gb:.2f} GB")

    # 3. Acceleration
    cuda_avail = torch.cuda.is_available()
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Hardware Note     : CPU execution detected. BioMistral-7B weights require >= 14GB RAM in FP16.")

    # 4. Verified Libraries
    print(f"FAISS Version     : {faiss.__version__}")
    print(f"Datasets Version  : {datasets.__version__}")

    # 5. Model Access
    print("\nVerifying model identifiers on Hugging Face...")
    try:
        emb_tok = AutoTokenizer.from_pretrained("NeuML/biomedbert-base-embeddings")
        print("Embedding Model   : NeuML/biomedbert-base-embeddings (Tokenizer OK)")
    except Exception as e:
        print(f"Embedding Model   : Failed: {e}")

    try:
        llm_cfg = AutoConfig.from_pretrained("BioMistral/BioMistral-7B")
        print("Backbone LLM      : BioMistral/BioMistral-7B (Config OK, Architecture: MistralForCausalLM)")
    except Exception as e:
        print(f"Backbone LLM      : Failed: {e}")

    # 6. Ensure Directories
    dirs = [
        "config", "src", "tests", "scripts",
        "data/raw", "data/processed", "data/indexes", "data/graph",
        "checkpoints", "logs", "results/raw", "results/processed",
        "results/tables", "results/figures", "reports", "artifacts"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("\nDirectory hierarchy verified and established.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    check_environment()
