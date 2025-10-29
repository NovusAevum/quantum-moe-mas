"""
Performance Bottleneck Identification and Analysis System

Provides automated bottleneck detection, root cause analysis, and optimization
recommendations for the Quantum MoE MAS system.

Requirements: 8.1, 8.2
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict, Counter

import structlog

from .latency_tracker import LatencyTracker, LatencyCategory, LatencyStats
from .resource_monitor import ResourceMonitor, ResourceType, ResourceStats

logger = structlog.get_logger(__name__)


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    DATABASE_BOUND = "database_bound"
    API_RATE_LIMITED = "api_rate_limited"
    CACHE_MISS = "cache_miss"
    ROUTING_INEFFICIENCY = "routing_inefficiency"
    EXPERT_OVERLOAD = "expert_overload"


class Severity(Enum):
    """Bottleneck severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BottleneckDetection:
    """Detected performance bottleneck."""
    
    bottleneck_type: BottleneckType
    severity: Severity
    confidence: float  # 0.0 to 1.0
    description: str
    affected_operations: List[str]
    metrics: Dict[str, Any]
    recommendations: List[str]
    detected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bottleneck_type': self.bottleneck_type.value,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'description': self.description,
            'affected_operations': self.affected_operations,
            'metrics': self.metrics,
            'recommendations': self.recommendations,
            'detected_at': self.detected_at.isoformat()
        }


@dataclass
class AnalysisResult:
    """Complete bottleneck analysis result."""
    
    bottlenecks: List[BottleneckDetection]
    system_health_score: float  # 0.0 to 1.0
    primary_bottleneck: Optional[BottleneckDetection]
    optimization_priority: List[BottleneckDetection]
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bottlenecks': [b.to_dict() for b in self.bottlenecks],
            'system_health_score': self.system_health_score,
            'primary_bottleneck': self.primary_bottleneck.to_dict() if self.primary_bottleneck else None,
            'optimization_priority': [b.to_dict() for b in self.optimization_priority],
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }


class BottleneckAnalyzer:
    """
    Automated performance bottleneck detection and analysis system.
    
    Analyzes latency patterns, resource utilization, and system metrics to
    identify performance bottlenecks and provide optimization recommendations.
    """
    
    def __init__(self,
                 latency_tracker: LatencyTracker,
                 resource_monitor: ResourceMonitor,
                 analysis_window: timedelta = timedelta(minutes=15),
                 confidence_threshold: float = 0.7):
        """Initialize bottleneck analyzer."""
        
        self.latency_tracker = latency_tracker
        self.resource_monitor = resource_monitor
        self.analysis_window = analysis_window
        self.confidence_threshold = confidence_threshold
        
        # Detection thresholds
        self.cpu_bottleneck_threshold = 85.0  # %
        self.memory_bottleneck_threshold = 90.0  # %
        self.latency_degradation_threshold = 2.0  # 2x normal latency
        self.cache_miss_threshold = 0.6  # 60% cache miss rate
        
        # Historical analysis data
        self._historical_detections: List[BottleneckDetection] = []
        self._detection_patterns: Dict[str, int] = defaultdict(int)
        
        logger.info("BottleneckAnalyzer initialized",
                   analysis_window_minutes=analysis_window.total_seconds() / 60,
                   confidence_threshold=confidence_threshold)
    
    async def analyze_system_performance(self) -> AnalysisResult:
        """Perform comprehensive system performance analysis."""
        
        logger.info("Starting comprehensive bottleneck analysis")
        
        # Collect current metrics
        latency_stats = self.latency_tracker.get_all_category_stats(self.analysis_window)
        resource_stats = self.resource_monitor.get_all_resource_stats(self.analysis_window)
        
        # Run all bottleneck detection algorithms
        bottlenecks = []
        
        # Resource-based bottleneck detection
        bottlenecks.extend(await self._detect_resource_bottlenecks(resource_stats))
        
        # Latency-based bottleneck detection
        bottlenecks.extend(await self._detect_latency_bottlenecks(latency_stats))
        
        # Pattern-based bottleneck detection
        bottlenecks.extend(await self._detect_pattern_bottlenecks(latency_stats, resource_stats))
        
        # System-specific bottleneck detection
        bottlenecks.extend(await self._detect_system_specific_bottlenecks(latency_stats, resource_stats))
        
        # Filter by confidence threshold
        high_confidence_bottlenecks = [
            b for b in bottlenecks 
            if b.confidence >= self.confidence_threshold
        ]
        
        # Calculate system health score
        health_score = self._calculate_system_health_score(latency_stats, resource_stats, high_confidence_bottlenecks)
        
        # Identify primary bottleneck
        primary_bottleneck = self._identify_primary_bottleneck(high_confidence_bottlenecks)
        
        # Prioritize optimizations
        optimization_priority = self._prioritize_optimizations(high_confidence_bottlenecks)
        
        # Store historical data
        self._historical_detections.extend(high_confidence_bottlenecks)
        for bottleneck in high_confidence_bottlenecks:
            self._detection_patterns[bottleneck.bottleneck_type.value] += 1
        
        result = AnalysisResult(
            bottlenecks=high_confidence_bottlenecks,
            system_health_score=health_score,
            primary_bottleneck=primary_bottleneck,
            optimization_priority=optimization_priority
        )
        
        logger.info("Bottleneck analysis completed",
                   bottlenecks_found=len(high_confidence_bottlenecks),
                   health_score=health_score,
                   primary_bottleneck=primary_bottleneck.bottleneck_type.value if primary_bottleneck else None)
        
        return result
    
    async def _detect_resource_bottlenecks(self, 
                                         resource_stats: Dict[ResourceType, ResourceStats]) -> List[BottleneckDetection]:
        """Detect bottlenecks based on resource utilization."""
        
        bottlenecks = []
        
        for resource_type, stats in resource_stats.items():
            if resource_type == ResourceType.CPU and stats.current_value > self.cpu_bottleneck_threshold:
                severity = Severity.CRITICAL if stats.current_value > 95 else Severity.HIGH
                confidence = min(0.9, (stats.current_value - 70) / 30)  # Scale confidence with usage
                
                bottlenecks.append(BottleneckDetection(
                    bottleneck_type=BottleneckType.CPU_BOUND,
                    severity=severity,
                    confidence=confidence,
                    description=f"High CPU utilization detected: {stats.current_value:.1f}%",
                    affected_operations=["all_operations"],
                    metrics={
                        "current_cpu_percent": stats.current_value,
                        "mean_cpu_percent": stats.mean_value,
                        "trend": stats.trend
                    },
                    recommendations=[
                        "Scale horizontally by adding more instances",
                        "Optimize CPU-intensive algorithms",
                        "Implement request queuing to smooth load",
                        "Consider upgrading to higher CPU capacity"
                    ]
                ))
            
            elif resource_type == ResourceType.MEMORY and stats.current_value > self.memory_bottleneck_threshold:
                severity = Severity.CRITICAL if stats.current_value > 98 else Severity.HIGH
                confidence = min(0.95, (stats.current_value - 80) / 20)
                
                bottlenecks.append(BottleneckDetection(
                    bottleneck_type=BottleneckType.MEMORY_BOUND,
                    severity=severity,
                    confidence=confidence,
                    description=f"High memory utilization detected: {stats.current_value:.1f}%",
                    affected_operations=["all_operations"],
                    metrics={
                        "current_memory_percent": stats.current_value,
                        "mean_memory_percent": stats.mean_value,
                        "trend": stats.trend
                    },
                    recommendations=[
                        "Increase memory capacity or scale out",
                        "Implement memory-efficient data structures",
                        "Review and fix memory leaks",
                        "Optimize garbage collection settings"
                    ]
                ))
        
        return bottlenecks
    
    async def _detect_latency_bottlenecks(self, 
                                        latency_stats: Dict[LatencyCategory, LatencyStats]) -> List[BottleneckDetection]:
        """Detect bottlenecks based on latency patterns."""
        
        bottlenecks = []
        
        for category, stats in latency_stats.items():
            # Check for SLA violations
            if not stats.meets_sla_target(5000):  # 5 second SLA
                severity = self._determine_latency_severity(stats.p95_ms)
                confidence = min(0.9, (stats.p95_ms - 5000) / 10000)  # Scale with latency excess
                
                bottlenecks.append(BottleneckDetection(
                    bottleneck_type=self._map_category_to_bottleneck_type(category),
                    severity=severity,
                    confidence=confidence,
                    description=f"High latency in {category.value}: P95 = {stats.p95_ms:.0f}ms",
                    affected_operations=[category.value],
                    metrics={
                        "p95_latency_ms": stats.p95_ms,
                        "p99_latency_ms": stats.p99_ms,
                        "sla_compliance_rate": stats.sla_compliance_rate,
                        "request_count": stats.count
                    },
                    recommendations=self._get_latency_recommendations(category, stats)
                ))
        
        return bottlenecks
    
    async def _detect_pattern_bottlenecks(self,
                                        latency_stats: Dict[LatencyCategory, LatencyStats],
                                        resource_stats: Dict[ResourceType, ResourceStats]) -> List[BottleneckDetection]:
        """Detect bottlenecks based on correlation patterns."""
        
        bottlenecks = []
        
        # Detect I/O bound operations (high latency + low CPU)
        if (LatencyCategory.DATABASE_QUERY in latency_stats and 
            ResourceType.CPU in resource_stats):
            
            db_latency = latency_stats[LatencyCategory.DATABASE_QUERY]
            cpu_usage = resource_stats[ResourceType.CPU]
            
            if db_latency.p95_ms > 2000 and cpu_usage.current_value < 50:
                bottlenecks.append(BottleneckDetection(
                    bottleneck_type=BottleneckType.IO_BOUND,
                    severity=Severity.MEDIUM,
                    confidence=0.8,
                    description="I/O bound operations detected: High database latency with low CPU usage",
                    affected_operations=["database_operations"],
                    metrics={
                        "db_p95_latency_ms": db_latency.p95_ms,
                        "cpu_usage_percent": cpu_usage.current_value
                    },
                    recommendations=[
                        "Optimize database queries and indexes",
                        "Implement connection pooling",
                        "Consider database caching strategies",
                        "Review database schema design"
                    ]
                ))
        
        # Detect network bottlenecks (high API latency + high network usage)
        if (LatencyCategory.API_CALL in latency_stats and 
            ResourceType.NETWORK in resource_stats):
            
            api_latency = latency_stats[LatencyCategory.API_CALL]
            network_usage = resource_stats[ResourceType.NETWORK]
            
            if api_latency.p95_ms > 3000 and network_usage.current_value > 80:
                bottlenecks.append(BottleneckDetection(
                    bottleneck_type=BottleneckType.NETWORK_BOUND,
                    severity=Severity.HIGH,
                    confidence=0.85,
                    description="Network bottleneck detected: High API latency with network saturation",
                    affected_operations=["api_calls"],
                    metrics={
                        "api_p95_latency_ms": api_latency.p95_ms,
                        "network_usage_percent": network_usage.current_value
                    },
                    recommendations=[
                        "Implement request batching",
                        "Use CDN for static content",
                        "Optimize payload sizes",
                        "Consider geographic load balancing"
                    ]
                ))
        
        return bottlenecks
    
    async def _detect_system_specific_bottlenecks(self,
                                                latency_stats: Dict[LatencyCategory, LatencyStats],
                                                resource_stats: Dict[ResourceType, ResourceStats]) -> List[BottleneckDetection]:
        """Detect system-specific bottlenecks for Quantum MoE MAS."""
        
        bottlenecks = []
        
        # Detect routing inefficiency
        if LatencyCategory.ROUTING in latency_stats:
            routing_stats = latency_stats[LatencyCategory.ROUTING]
            
            if routing_stats.p95_ms > 1000:  # Routing should be fast
                bottlenecks.append(BottleneckDetection(
                    bottleneck_type=BottleneckType.ROUTING_INEFFICIENCY,
                    severity=Severity.MEDIUM,
                    confidence=0.8,
                    description=f"Quantum routing inefficiency: P95 = {routing_stats.p95_ms:.0f}ms",
                    affected_operations=["quantum_routing"],
                    metrics={
                        "routing_p95_ms": routing_stats.p95_ms,
                        "routing_requests": routing_stats.count
                    },
                    recommendations=[
                        "Optimize quantum gate calculations",
                        "Implement routing result caching",
                        "Review expert selection algorithms",
                        "Consider pre-computed routing tables"
                    ]
                ))
        
        # Detect expert overload (high expert inference latency)
        if LatencyCategory.EXPERT_INFERENCE in latency_stats:
            expert_stats = latency_stats[LatencyCategory.EXPERT_INFERENCE]
            
            if expert_stats.p95_ms > 8000:  # Expert inference taking too long
                bottlenecks.append(BottleneckDetection(
                    bottleneck_type=BottleneckType.EXPERT_OVERLOAD,
                    severity=Severity.HIGH,
                    confidence=0.9,
                    description=f"Expert overload detected: P95 = {expert_stats.p95_ms:.0f}ms",
                    affected_operations=["expert_inference"],
                    metrics={
                        "expert_p95_ms": expert_stats.p95_ms,
                        "expert_requests": expert_stats.count
                    },
                    recommendations=[
                        "Scale expert instances horizontally",
                        "Implement expert load balancing",
                        "Add expert response caching",
                        "Consider expert model optimization"
                    ]
                ))
        
        return bottlenecks
    
    def _map_category_to_bottleneck_type(self, category: LatencyCategory) -> BottleneckType:
        """Map latency category to bottleneck type."""
        
        mapping = {
            LatencyCategory.ROUTING: BottleneckType.ROUTING_INEFFICIENCY,
            LatencyCategory.EXPERT_INFERENCE: BottleneckType.EXPERT_OVERLOAD,
            LatencyCategory.RAG_RETRIEVAL: BottleneckType.DATABASE_BOUND,
            LatencyCategory.DATABASE_QUERY: BottleneckType.DATABASE_BOUND,
            LatencyCategory.API_CALL: BottleneckType.API_RATE_LIMITED,
            LatencyCategory.TOTAL_REQUEST: BottleneckType.CPU_BOUND
        }
        
        return mapping.get(category, BottleneckType.CPU_BOUND)
    
    def _determine_latency_severity(self, p95_latency_ms: float) -> Severity:
        """Determine severity based on P95 latency."""
        
        if p95_latency_ms > 20000:  # 20+ seconds
            return Severity.CRITICAL
        elif p95_latency_ms > 10000:  # 10+ seconds
            return Severity.HIGH
        elif p95_latency_ms > 5000:  # 5+ seconds (SLA violation)
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _get_latency_recommendations(self, 
                                   category: LatencyCategory, 
                                   stats: LatencyStats) -> List[str]:
        """Get specific recommendations for latency bottlenecks."""
        
        recommendations = []
        
        if category == LatencyCategory.ROUTING:
            recommendations.extend([
                "Optimize quantum routing algorithms",
                "Implement routing result caching",
                "Pre-compute common routing decisions",
                "Consider simpler routing heuristics for common cases"
            ])
        
        elif category == LatencyCategory.EXPERT_INFERENCE:
            recommendations.extend([
                "Scale expert instances horizontally",
                "Implement expert response caching",
                "Optimize expert model parameters",
                "Consider expert model quantization"
            ])
        
        elif category == LatencyCategory.RAG_RETRIEVAL:
            recommendations.extend([
                "Optimize vector database queries",
                "Implement semantic caching",
                "Review embedding model efficiency",
                "Consider approximate nearest neighbor search"
            ])
        
        elif category == LatencyCategory.DATABASE_QUERY:
            recommendations.extend([
                "Add database indexes",
                "Optimize query patterns",
                "Implement connection pooling",
                "Consider read replicas"
            ])
        
        elif category == LatencyCategory.API_CALL:
            recommendations.extend([
                "Implement API response caching",
                "Use request batching",
                "Consider API rate limit optimization",
                "Implement circuit breakers"
            ])
        
        return recommendations
    
    def _calculate_system_health_score(self,
                                     latency_stats: Dict[LatencyCategory, LatencyStats],
                                     resource_stats: Dict[ResourceType, ResourceStats],
                                     bottlenecks: List[BottleneckDetection]) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        
        # Start with perfect score
        health_score = 1.0
        
        # Penalize for SLA violations
        sla_penalty = 0.0
        for stats in latency_stats.values():
            if not stats.meets_sla_target(5000):
                sla_penalty += (1.0 - stats.sla_compliance_rate) * 0.3
        
        # Penalize for resource utilization
        resource_penalty = 0.0
        for resource_type, stats in resource_stats.items():
            if stats.current_value > 90:
                resource_penalty += 0.2
            elif stats.current_value > 80:
                resource_penalty += 0.1
        
        # Penalize for bottlenecks
        bottleneck_penalty = 0.0
        for bottleneck in bottlenecks:
            if bottleneck.severity == Severity.CRITICAL:
                bottleneck_penalty += 0.3 * bottleneck.confidence
            elif bottleneck.severity == Severity.HIGH:
                bottleneck_penalty += 0.2 * bottleneck.confidence
            elif bottleneck.severity == Severity.MEDIUM:
                bottleneck_penalty += 0.1 * bottleneck.confidence
        
        # Apply penalties
        health_score -= min(sla_penalty, 0.4)  # Cap SLA penalty
        health_score -= min(resource_penalty, 0.3)  # Cap resource penalty
        health_score -= min(bottleneck_penalty, 0.5)  # Cap bottleneck penalty
        
        return max(0.0, health_score)
    
    def _identify_primary_bottleneck(self, 
                                   bottlenecks: List[BottleneckDetection]) -> Optional[BottleneckDetection]:
        """Identify the primary bottleneck to focus on."""
        
        if not bottlenecks:
            return None
        
        # Score bottlenecks by severity and confidence
        def bottleneck_score(b: BottleneckDetection) -> float:
            severity_weights = {
                Severity.CRITICAL: 4.0,
                Severity.HIGH: 3.0,
                Severity.MEDIUM: 2.0,
                Severity.LOW: 1.0
            }
            return severity_weights[b.severity] * b.confidence
        
        return max(bottlenecks, key=bottleneck_score)
    
    def _prioritize_optimizations(self, 
                                bottlenecks: List[BottleneckDetection]) -> List[BottleneckDetection]:
        """Prioritize bottlenecks for optimization efforts."""
        
        # Sort by severity, then confidence, then impact
        def priority_key(b: BottleneckDetection) -> Tuple[int, float, int]:
            severity_order = {
                Severity.CRITICAL: 4,
                Severity.HIGH: 3,
                Severity.MEDIUM: 2,
                Severity.LOW: 1
            }
            
            # Estimate impact based on affected operations
            impact = len(b.affected_operations)
            
            return (severity_order[b.severity], b.confidence, impact)
        
        return sorted(bottlenecks, key=priority_key, reverse=True)
    
    def get_historical_patterns(self) -> Dict[str, Any]:
        """Get historical bottleneck patterns for trend analysis."""
        
        return {
            'detection_counts': dict(self._detection_patterns),
            'recent_detections': [
                b.to_dict() for b in self._historical_detections[-10:]
            ],
            'most_common_bottlenecks': [
                {'type': bt, 'count': count}
                for bt, count in Counter(self._detection_patterns).most_common(5)
            ]
        }
    
    def cleanup_historical_data(self, max_age: timedelta = timedelta(days=7)) -> int:
        """Clean up old historical detection data."""
        
        cutoff_time = datetime.now() - max_age
        original_count = len(self._historical_detections)
        
        self._historical_detections = [
            detection for detection in self._historical_detections
            if detection.detected_at > cutoff_time
        ]
        
        removed_count = original_count - len(self._historical_detections)
        
        logger.info("Cleaned up historical bottleneck data",
                   removed_count=removed_count,
                   remaining_count=len(self._historical_detections))
        
        return removed_count