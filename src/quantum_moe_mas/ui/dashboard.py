"""
Main Dashboard Controller for Quantum MoE MAS

This module provides the central dashboard controller that coordinates
all system components and provides unified access for the UI.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.moe.router import QuantumMoERouter
from quantum_moe_mas.moe.expert_manager import ExpertPoolManager, FailoverStrategy
from quantum_moe_mas.moe.analytics import RoutingAnalytics, EfficiencyReport, VisualizationData
from quantum_moe_mas.moe.expert import Expert, ExpertType, ExpertStatus
from quantum_moe_mas.api.orchestrator import API_Orchestrator, APIProvider
from quantum_moe_mas.rag.adaptive_rag import AdaptiveMultiModalRAG, AdaptiveRAGConfig
from quantum_moe_mas.agents.base_agent import BaseAgent, AgentState
from quantum_moe_mas.orchestration.mas_orchestrator import MASOrchestrator
from quantum_moe_mas.config.settings_simple import get_settings

logger = get_logger(__name__)


@dataclass
class SystemStatus:
    """System status information."""
    healthy: bool
    components: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Any]
    last_updated: datetime
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "healthy": self.healthy,
            "components": self.components,
            "metrics": self.metrics,
            "last_updated": self.last_updated.isoformat(),
            "uptime_seconds": self.uptime_seconds,
        }


@dataclass
class QueryRequest:
    """Request object for system queries."""
    query_text: str
    query_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing."""
        return {
            "query_text": self.query_text,
            "query_type": self.query_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class QueryResponse:
    """Response object for system queries."""
    response_text: str
    confidence_score: float
    expert_used: str
    processing_time: float
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class DashboardController:
    """Main dashboard controller for Quantum MoE MAS system."""
    
    def __init__(self):
        """Initialize the dashboard controller."""
        self.logger = logger
        self.settings = get_settings()
        self.start_time = time.time()
        
        # Initialize core components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize MoE Router
            self.moe_router = QuantumMoERouter()
            
            # Initialize Expert Manager
            self.expert_manager = ExpertPoolManager(
                failover_strategy=FailoverStrategy.ROUND_ROBIN
            )
            
            # Initialize Analytics
            self.analytics = RoutingAnalytics()
            
            # Initialize API Orchestrator
            self.api_orchestrator = API_Orchestrator()
            
            # Initialize RAG System
            rag_config = AdaptiveRAGConfig()
            self.rag_system = AdaptiveMultiModalRAG(rag_config)
            
            # Initialize MAS Orchestrator
            self.mas_orchestrator = MASOrchestrator()
            
            self.logger.info("Dashboard controller initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard controller: {e}")
            raise
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status."""
        try:
            components = {}
            
            # Check MoE Router status
            components["moe_router"] = {
                "status": "healthy" if self.moe_router else "unavailable",
                "active_experts": len(self.expert_manager.get_active_experts()) if self.expert_manager else 0
            }
            
            # Check Expert Manager status
            components["expert_manager"] = {
                "status": "healthy" if self.expert_manager else "unavailable",
                "total_experts": len(self.expert_manager.experts) if self.expert_manager else 0
            }
            
            # Check API Orchestrator status
            components["api_orchestrator"] = {
                "status": "healthy" if self.api_orchestrator else "unavailable",
                "active_providers": len(self.api_orchestrator.get_active_providers()) if self.api_orchestrator else 0
            }
            
            # Check RAG System status
            components["rag_system"] = {
                "status": "healthy" if self.rag_system else "unavailable"
            }
            
            # Calculate overall health
            healthy_components = sum(1 for comp in components.values() if comp["status"] == "healthy")
            overall_healthy = healthy_components == len(components)
            
            # Get system metrics
            metrics = await self._get_system_metrics()
            
            return SystemStatus(
                healthy=overall_healthy,
                components=components,
                metrics=metrics,
                last_updated=datetime.now(),
                uptime_seconds=time.time() - self.start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return SystemStatus(
                healthy=False,
                components={},
                metrics={},
                last_updated=datetime.now(),
                uptime_seconds=time.time() - self.start_time
            )
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            metrics = {}
            
            if self.analytics:
                efficiency_report = await self.analytics.get_efficiency_report()
                metrics.update({
                    "routing_efficiency": efficiency_report.overall_efficiency,
                    "avg_confidence": efficiency_report.average_confidence,
                    "total_queries": efficiency_report.total_queries
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query through the MoE system."""
        start_time = time.time()
        
        try:
            # Route query through MoE system
            routing_result = await self.moe_router.route_query(
                request.query_text,
                request.query_type
            )
            
            # Process with selected expert
            expert_response = await routing_result.expert.process_query(
                request.query_text,
                request.metadata
            )
            
            processing_time = time.time() - start_time
            
            return QueryResponse(
                response_text=expert_response.text,
                confidence_score=routing_result.confidence,
                expert_used=routing_result.expert.name,
                processing_time=processing_time,
                metadata={
                    "routing_decision": routing_result.reasoning,
                    "expert_metadata": expert_response.metadata
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            processing_time = time.time() - start_time
            
            return QueryResponse(
                response_text=f"Error processing query: {str(e)}",
                confidence_score=0.0,
                expert_used="error_handler",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )