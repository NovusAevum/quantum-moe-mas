"""
Expert dataclass and management for the MoE system.

This module defines the Expert entity with API endpoint configuration,
health monitoring, and performance tracking capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4


class ExpertType(Enum):
    """Types of AI experts available in the system."""
    
    LANGUAGE_MODEL = "language_model"
    VISION_MODEL = "vision_model"
    CODE_MODEL = "code_model"
    REASONING_MODEL = "reasoning_model"
    MULTIMODAL_MODEL = "multimodal_model"
    EMBEDDING_MODEL = "embedding_model"


class ExpertStatus(Enum):
    """Health status of an expert."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ExpertPerformanceMetrics:
    """Performance metrics for an expert."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    average_confidence: float = 0.0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    last_request_timestamp: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        return 100.0 - self.success_rate


@dataclass
class Expert:
    """
    Expert entity representing an AI model/API in the MoE system.
    
    Attributes:
        id: Unique identifier for the expert
        name: Human-readable name
        type: Type of expert (language, vision, code, etc.)
        api_endpoint: API endpoint URL
        api_key_env_var: Environment variable name for API key
        capabilities: List of capabilities this expert provides
        cost_per_token: Cost per token in USD
        max_tokens: Maximum tokens supported
        confidence_score: Current confidence score (0-100)
        load_factor: Current load factor (0-1)
        status: Current health status
        priority: Priority level for routing (higher = more preferred)
        metadata: Additional metadata
        performance: Performance metrics
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    type: ExpertType = ExpertType.LANGUAGE_MODEL
    api_endpoint: str = ""
    api_key_env_var: str = ""
    capabilities: List[str] = field(default_factory=list)
    cost_per_token: float = 0.0
    max_tokens: int = 4096
    confidence_score: float = 80.0
    load_factor: float = 0.0
    status: ExpertStatus = ExpertStatus.HEALTHY
    priority: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance: ExpertPerformanceMetrics = field(default_factory=ExpertPerformanceMetrics)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate expert configuration after initialization."""
        if not self.name:
            raise ValueError("Expert name cannot be empty")
        if not self.api_endpoint:
            raise ValueError("Expert API endpoint cannot be empty")
        if not 0 <= self.confidence_score <= 100:
            raise ValueError("Confidence score must be between 0 and 100")
        if not 0 <= self.load_factor <= 1:
            raise ValueError("Load factor must be between 0 and 1")
    
    def update_confidence(self, new_confidence: float) -> None:
        """
        Update the expert's confidence score.
        
        Args:
            new_confidence: New confidence score (0-100)
        
        Raises:
            ValueError: If confidence score is out of range
        """
        if not 0 <= new_confidence <= 100:
            raise ValueError("Confidence score must be between 0 and 100")
        self.confidence_score = new_confidence
        self.updated_at = datetime.utcnow()
    
    def update_load(self, new_load: float) -> None:
        """
        Update the expert's load factor.
        
        Args:
            new_load: New load factor (0-1)
        
        Raises:
            ValueError: If load factor is out of range
        """
        if not 0 <= new_load <= 1:
            raise ValueError("Load factor must be between 0 and 1")
        self.load_factor = new_load
        self.updated_at = datetime.utcnow()
    
    def update_status(self, new_status: ExpertStatus) -> None:
        """
        Update the expert's health status.
        
        Args:
            new_status: New health status
        """
        self.status = new_status
        self.updated_at = datetime.utcnow()
    
    def record_request(
        self,
        success: bool,
        latency_ms: float,
        confidence: float,
        tokens_used: int = 0,
        cost: float = 0.0
    ) -> None:
        """
        Record a request to this expert for performance tracking.
        
        Args:
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
            confidence: Confidence score of the response
            tokens_used: Number of tokens used
            cost: Cost of the request
        """
        self.performance.total_requests += 1
        
        if success:
            self.performance.successful_requests += 1
        else:
            self.performance.failed_requests += 1
        
        # Update running averages
        total = self.performance.total_requests
        self.performance.average_latency_ms = (
            (self.performance.average_latency_ms * (total - 1) + latency_ms) / total
        )
        self.performance.average_confidence = (
            (self.performance.average_confidence * (total - 1) + confidence) / total
        )
        
        self.performance.total_tokens_used += tokens_used
        self.performance.total_cost += cost
        self.performance.last_request_timestamp = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def is_available(self) -> bool:
        """
        Check if the expert is available for routing.
        
        Returns:
            True if expert is healthy and not overloaded
        """
        return (
            self.status == ExpertStatus.HEALTHY and
            self.load_factor < 0.95  # Not overloaded
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert expert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "api_endpoint": self.api_endpoint,
            "capabilities": self.capabilities,
            "cost_per_token": self.cost_per_token,
            "max_tokens": self.max_tokens,
            "confidence_score": self.confidence_score,
            "load_factor": self.load_factor,
            "status": self.status.value,
            "priority": self.priority,
            "performance": {
                "total_requests": self.performance.total_requests,
                "success_rate": self.performance.success_rate,
                "average_latency_ms": self.performance.average_latency_ms,
                "average_confidence": self.performance.average_confidence,
                "total_cost": self.performance.total_cost,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
