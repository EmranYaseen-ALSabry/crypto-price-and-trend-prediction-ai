"""Structured logging utility for Crypto Price Prediction AI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "crypto_prediction", log_file: str | Path | None = None) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger.
        log_file: Optional file path to persist logs.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
