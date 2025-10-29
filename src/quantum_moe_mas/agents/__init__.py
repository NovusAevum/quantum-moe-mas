"""
Domain-Specialized Agent Framework

This module provides the base agent architecture and specialized agents
for different domains including cybersecurity, cloud computing, marketing,
and quantum computing.

Author: Wan Mohamad Hanis bin Wan Hassan
"""

from quantum_moe_mas.agents.base_agent import (
    BaseAgent,
    AgentState,
    AgentMessage,
    AgentContext,
    AgentCapability,
    AgentMetrics,
    MessageType,
)
from quantum_moe_mas.agents.communication import (
    MessageBus,
    CommunicationProtocol,
    MessagePriority,
    MessageStatus,
    MessageRoute,
    MessageDeliveryReceipt,
    MessageFilter,
)
from quantum_moe_mas.agents.monitoring import (
    AgentMonitor,
    HealthStatus,
    PerformanceMetrics,
    HealthCheck,
    Alert,
    AlertSeverity,
    MetricCollector,
)

# Domain-specialized agents
from quantum_moe_mas.agents.cyber_agent import (
    CyberAgent,
    ThreatLevel,
    VulnerabilityType,
    OSINTSource,
    ThreatData,
    Vulnerability,
    OSINTResult,
    ThreatAnalysis,
    VulnerabilityReport,
    OSINTReport,
    SecurityRecommendation,
)

from quantum_moe_mas.agents.cloud_agent import (
    CloudAgent,
    CloudProvider,
    ResourceType,
    DeploymentStatus,
    ScalingAction,
    CloudResource,
    InfrastructureConfig,
    DeploymentResult,
    ResourceMetrics,
    ScalingDecision,
    CostOptimization,
    HealthReport,
)

from quantum_moe_mas.agents.marketing_agent import (
    MarketingAgent,
    CampaignType,
    CampaignStatus,
    LeadStage,
    ContentType,
    Campaign,
    Lead,
    CampaignData,
    CampaignAnalysis,
    ROIMetrics,
    TargetingStrategy,
    GeneratedContent,
)

from quantum_moe_mas.agents.quantum_agent import (
    QuantumAgent,
    QuantumAlgorithm,
    QuantumBackend,
    OptimizationProblem,
    QuantumCircuit,
    QuantumData,
    SimulationResult,
    QuantumSolution,
    QMLModel,
    AdvantageAnalysis,
)

__all__ = [
    # Base agent components
    "BaseAgent",
    "AgentState",
    "AgentMessage",
    "AgentContext",
    "AgentCapability",
    "AgentMetrics",
    "MessageType",
    
    # Communication components
    "MessageBus",
    "CommunicationProtocol",
    "MessagePriority",
    "MessageStatus",
    "MessageRoute",
    "MessageDeliveryReceipt",
    "MessageFilter",
    
    # Monitoring components
    "AgentMonitor",
    "HealthStatus",
    "PerformanceMetrics",
    "HealthCheck",
    "Alert",
    "AlertSeverity",
    "MetricCollector",
    
    # Domain-specialized agents
    "CyberAgent",
    "CloudAgent", 
    "MarketingAgent",
    "QuantumAgent",
    
    # Cyber agent components
    "ThreatLevel",
    "VulnerabilityType",
    "OSINTSource",
    "ThreatData",
    "Vulnerability",
    "OSINTResult",
    "ThreatAnalysis",
    "VulnerabilityReport",
    "OSINTReport",
    "SecurityRecommendation",
    
    # Cloud agent components
    "CloudProvider",
    "ResourceType",
    "DeploymentStatus",
    "ScalingAction",
    "CloudResource",
    "InfrastructureConfig",
    "DeploymentResult",
    "ResourceMetrics",
    "ScalingDecision",
    "CostOptimization",
    "HealthReport",
    
    # Marketing agent components
    "CampaignType",
    "CampaignStatus",
    "LeadStage",
    "ContentType",
    "Campaign",
    "Lead",
    "CampaignData",
    "CampaignAnalysis",
    "ROIMetrics",
    "TargetingStrategy",
    "GeneratedContent",
    
    # Quantum agent components
    "QuantumAlgorithm",
    "QuantumBackend",
    "OptimizationProblem",
    "QuantumCircuit",
    "QuantumData",
    "SimulationResult",
    "QuantumSolution",
    "QMLModel",
    "AdvantageAnalysis",
]