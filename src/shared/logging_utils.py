"""Structured logging and console messaging utilities."""

import os
import sys
import logging
from typing import Optional


def setup_logger(
    name: str = "biomedkai",
    log_dir: str = "logs",
    log_file: Optional[str] = "experiment.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a structured logger writing to both console and a log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Formatter with timestamp, level, and logger name
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (use stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
