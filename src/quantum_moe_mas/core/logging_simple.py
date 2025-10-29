"""
Simplified logging for validation.

This is a temporary simplified version for testing purposes.
"""

import logging
from typing import Any


def setup_logging() -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
