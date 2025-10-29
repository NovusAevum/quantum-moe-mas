"""
Auto-Scaling and Performance Optimization Module

This module provides comprehensive auto-scaling mechanisms including:
- Kubernetes HPA (Horizontal Pod Autoscaler) integration
- Intelligent Redis caching with 40%+ API call reduction
- Load balancing with health checks
- Predictive scaling based on ML patterns
- Container orchestration with Docker and Kubernetes

Requirements: 8.1, 8.3, 8.4
"""

from .kubernetes_scaler import KubernetesScaler, ScalingMetrics
from .redis_cache import IntelligentCache, CacheStrategy
from .load_balancer import LoadBalancer, HealthChecker
from .predictive_scaler import PredictiveScaler, UsagePredictor
from .container_orchestrator import ContainerOrchestrator, DeploymentManager

__all__ = [
    "KubernetesScaler",
    "ScalingMetrics",
    "IntelligentCache", 
    "CacheStrategy",
    "LoadBalancer",
    "HealthChecker",
    "PredictiveScaler",
    "UsagePredictor",
    "ContainerOrchestrator",
    "DeploymentManager",
]