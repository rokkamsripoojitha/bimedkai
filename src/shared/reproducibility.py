"""Reproducibility management, deterministic seeding, and experiment manifest generation."""

import os
import random
import platform
import hashlib
import json
from typing import Dict, Any
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds across standard library, numpy, and PyTorch for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_environment_info() -> Dict[str, Any]:
    """Capture comprehensive environment details for auditability and reproducibility."""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    return info


def hash_config(config: Dict[str, Any]) -> str:
    """Compute a deterministic MD5 hash of an experiment configuration dictionary."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()
