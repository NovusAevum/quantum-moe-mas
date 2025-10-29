"""
Routing metrics and analytics for the MoE system.

This module provides comprehensive metrics tracking for routing decisions,
performance analysis, and efficiency calculations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4


@dataclass
class RoutingDecision:
    """
    Represents a single routing decision made by the MoE router.
    
    Attributes:
        id: Unique identifier for this decision
        query_id: ID of the query being routed
        query_text: Text of the query
        domain: Domain classification of the query
        selected_expert_ids: List of selected expert IDs
        expert_scores: Confidence scores for each expert
        quantum_state: Quantum-inspired state representation
        routing_confidence: Overall confidence in routing decision (0-100)
        decision_rationale: Explanation of routing decision
        latency_ms: Time taken to make routing decision
        timestamp: When the decision was made
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    query_id: str = ""
    query_text: str = ""
    domain: str = "general"
    selected_expert_ids: List[str] = field(default_factory=list)
    expert_scores: Dict[str, float] = field(default_factory=dict)
    quantum_state: Dict[str, float] = field(default_factory=dict)
    routing_confidence: float = 0.0
    decision_rationale: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert routing decision to dictionary representation."""
        return {
            "id": self.id,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "domain": self.domain,
            "selected_expert_ids": self.selected_expert_ids,
            "expert_scores": self.expert_scores,
            "quantum_state": self.quantum_state,
            "routing_confidence": self.routing_confidence,
            "decision_rationale": self.decision_rationale,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RoutingMetrics:
    """
    Comprehensive metrics for MoE routing performance.
    
    Tracks routing decisions, performance, efficiency gains, and system health.
    """
    
    # Request metrics
    total_requests: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    
    # Latency metrics
    total_routing_latency_ms: float = 0.0
    average_routing_latency_ms: float = 0.0
    min_routing_latency_ms: float = float('inf')
    max_routing_latency_ms: float = 0.0
    
    # Confidence metrics
    average_routing_confidence: float = 0.0
    high_confidence_routes: int = 0  # >= 80%
    medium_confidence_routes: int = 0  # 60-80%
    low_confidence_routes: int = 0  # < 60%
    
    # Expert utilization
    expert_selection_counts: Dict[str, int] = field(default_factory=dict)
    expert_success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency metrics
    baseline_latency_ms: float = 1000.0  # Baseline for comparison
    efficiency_gain_percentage: float = 0.0
    tokens_saved: int = 0
    cost_saved: float = 0.0
    
    # Domain distribution
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quantum state metrics
    quantum_entropy: float = 0.0
    superposition_utilization: float = 0.0
    
    # Time tracking
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Calculate routing success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_routes / self.total_requests) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate routing failure rate percentage."""
        return 100.0 - self.success_rate
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate system uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    @property
    def requests_per_second(self) -> float:
        """Calculate average requests per second."""
        uptime = self.uptime_seconds
        if uptime == 0:
            return 0.0
        return self.total_requests / uptime
    
    def record_routing_decision(
        self,
        decision: RoutingDecision,
        success: bool = True
    ) -> None:
        """
        Record a routing decision and update metrics.
        
        Args:
            decision: The routing decision to record
            success: Whether the routing was successful
        """
        self.total_requests += 1
        
        if success:
            self.successful_routes += 1
        else:
            self.failed_routes += 1
        
        # Update latency metrics
        latency = decision.latency_ms
        self.total_routing_latency_ms += latency
        self.average_routing_latency_ms = (
            self.total_routing_latency_ms / self.total_requests
        )
        self.min_routing_latency_ms = min(self.min_routing_latency_ms, latency)
        self.max_routing_latency_ms = max(self.max_routing_latency_ms, latency)
        
        # Update confidence metrics
        confidence = decision.routing_confidence
        total = self.total_requests
        self.average_routing_confidence = (
            (self.average_routing_confidence * (total - 1) + confidence) / total
        )
        
        if confidence >= 80:
            self.high_confidence_routes += 1
        elif confidence >= 60:
            self.medium_confidence_routes += 1
        else:
            self.low_confidence_routes += 1
        
        # Update expert selection counts
        for expert_id in decision.selected_expert_ids:
            self.expert_selection_counts[expert_id] = (
                self.expert_selection_counts.get(expert_id, 0) + 1
            )
        
        # Update domain distribution
        domain = decision.domain
        self.domain_distribution[domain] = (
            self.domain_distribution.get(domain, 0) + 1
        )
        
        # Calculate efficiency gain
        if self.baseline_latency_ms > 0:
            self.efficiency_gain_percentage = (
                (self.baseline_latency_ms - self.average_routing_latency_ms) /
                self.baseline_latency_ms * 100.0
            )
        
        self.last_update = datetime.utcnow()
    
    def update_expert_success_rate(
        self,
        expert_id: str,
        success_rate: float
    ) -> None:
        """
        Update the success rate for a specific expert.
        
        Args:
            expert_id: ID of the expert
            success_rate: Success rate percentage (0-100)
        """
        self.expert_success_rates[expert_id] = success_rate
        self.last_update = datetime.utcnow()
    
    def calculate_efficiency_metrics(
        self,
        baseline_latency_ms: float,
        tokens_saved: int = 0,
        cost_saved: float = 0.0
    ) -> None:
        """
        Calculate and update efficiency metrics.
        
        Args:
            baseline_latency_ms: Baseline latency for comparison
            tokens_saved: Number of tokens saved through sparse activation
            cost_saved: Cost saved in USD
        """
        self.baseline_latency_ms = baseline_latency_ms
        self.tokens_saved = tokens_saved
        self.cost_saved = cost_saved
        
        if baseline_latency_ms > 0:
            self.efficiency_gain_percentage = (
                (baseline_latency_ms - self.average_routing_latency_ms) /
                baseline_latency_ms * 100.0
            )
        
        self.last_update = datetime.utcnow()
    
    def get_top_experts(self, n: int = 5) -> List[tuple[str, int]]:
        """
        Get the top N most frequently selected experts.
        
        Args:
            n: Number of top experts to return
        
        Returns:
            List of (expert_id, selection_count) tuples
        """
        sorted_experts = sorted(
            self.expert_selection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_experts[:n]
    
    def get_domain_distribution_percentages(self) -> Dict[str, float]:
        """
        Get domain distribution as percentages.
        
        Returns:
            Dictionary mapping domain to percentage
        """
        if self.total_requests == 0:
            return {}
        
        return {
            domain: (count / self.total_requests) * 100.0
            for domain, count in self.domain_distribution.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "total_requests": self.total_requests,
            "successful_routes": self.successful_routes,
            "failed_routes": self.failed_routes,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "average_routing_latency_ms": self.average_routing_latency_ms,
            "min_routing_latency_ms": self.min_routing_latency_ms if self.min_routing_latency_ms != float('inf') else 0.0,
            "max_routing_latency_ms": self.max_routing_latency_ms,
            "average_routing_confidence": self.average_routing_confidence,
            "high_confidence_routes": self.high_confidence_routes,
            "medium_confidence_routes": self.medium_confidence_routes,
            "low_confidence_routes": self.low_confidence_routes,
            "efficiency_gain_percentage": self.efficiency_gain_percentage,
            "tokens_saved": self.tokens_saved,
            "cost_saved": self.cost_saved,
            "top_experts": self.get_top_experts(),
            "domain_distribution": self.get_domain_distribution_percentages(),
            "uptime_seconds": self.uptime_seconds,
            "requests_per_second": self.requests_per_second,
            "start_time": self.start_time.isoformat(),
            "last_update": self.last_update.isoformat(),
        }
