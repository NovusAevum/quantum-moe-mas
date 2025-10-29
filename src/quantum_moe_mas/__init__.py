"""
Quantum-Infused Mixture of Experts Multi-Agent System (MAS)

A sophisticated AI powerhouse that leverages quantum-inspired routing,
multi-modal RAG, and domain-specialized agents for enterprise-grade
AI orchestration and automation.

Author: Wan Mohamad Hanis bin Wan Hassan
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Wan Mohamad Hanis bin Wan Hassan"
__email__ = "hanis@example.com"
__description__ = "Quantum-Infused MoE Multi-Agent System with Adaptive Multi-Modal RAG"

# Core imports for easy access
from quantum_moe_mas.config.settings_simple import Settings, get_settings
from quantum_moe_mas.core.logging_simple import get_logger, setup_logging

# Initialize logging on import
setup_logging()

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "Settings",
    "get_settings",
    "get_logger",
    "setup_logging",
]