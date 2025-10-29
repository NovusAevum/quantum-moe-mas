"""
API Integration and Orchestration module for Quantum MoE MAS.

This module provides comprehensive API orchestration capabilities including:
- Unified interface for 30+ free AI APIs
- API key management and authentication
- Rate limiting and quota management
- Health monitoring and failover mechanisms
- Usage tracking and cost management
- Performance analysis and optimization
- Automated signup and key management
"""

from quantum_moe_mas.api.orchestrator import API_Orchestrator
from quantum_moe_mas.api.key_manager import APIKeyManager, APICredentials
from quantum_moe_mas.api.rate_limiter import RateLimiter, QuotaManager
from quantum_moe_mas.api.health_monitor import APIHealthMonitor
from quantum_moe_mas.api.integrations.base import BaseAPIIntegration, APIResponse, APIError
from quantum_moe_mas.api.integration_registry import APIIntegrationRegistry, get_integration_registry

# New usage tracking and cost management components
from quantum_moe_mas.api.usage_tracker import UsageTracker, UsageMetricType, TimeWindow
from quantum_moe_mas.api.cost_manager import CostManager, BudgetPeriod, AlertType
from quantum_moe_mas.api.performance_analyzer import PerformanceAnalyzer, PerformanceMetric
from quantum_moe_mas.api.signup_manager import SignupManager, ProviderCategory
from quantum_moe_mas.api.usage_cost_integration import IntegratedAPIManager, IntegratedAPIMetrics

__all__ = [
    # Core orchestration
    "API_Orchestrator",
    "APIKeyManager",
    "APICredentials",
    "RateLimiter",
    "QuotaManager",
    "APIHealthMonitor",
    "BaseAPIIntegration",
    "APIResponse",
    "APIError",
    "APIIntegrationRegistry",
    "get_integration_registry",
    
    # Usage tracking and cost management
    "UsageTracker",
    "UsageMetricType",
    "TimeWindow",
    "CostManager",
    "BudgetPeriod",
    "AlertType",
    "PerformanceAnalyzer",
    "PerformanceMetric",
    "SignupManager",
    "ProviderCategory",
    
    # Integrated management
    "IntegratedAPIManager",
    "IntegratedAPIMetrics",
]

__version__ = "0.1.0"