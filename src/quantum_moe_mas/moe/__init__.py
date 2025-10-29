"""
Mixture of Experts (MoE) module for quantum-inspired routing and expert management.

This module provides the core MoE routing functionality with quantum-probabilistic
gating, expert management, and performance tracking.
"""

from quantum_moe_mas.moe.router import QuantumMoERouter
from quantum_moe_mas.moe.expert import Expert, ExpertType, ExpertStatus
from quantum_moe_mas.moe.metrics import RoutingMetrics, RoutingDecision
from quantum_moe_mas.moe.expert_manager import (
    ExpertPoolManager,
    ExpertHealthMonitor,
    FailoverStrategy,
)
from quantum_moe_mas.moe.analytics import (
    RoutingAnalytics,
    EfficiencyReport,
    VisualizationData,
)

__all__ = [
    "QuantumMoERouter",
    "Expert",
    "ExpertType",
    "ExpertStatus",
    "RoutingMetrics",
    "RoutingDecision",
    "ExpertPoolManager",
    "ExpertHealthMonitor",
    "FailoverStrategy",
    "RoutingAnalytics",
    "EfficiencyReport",
    "VisualizationData",
]
