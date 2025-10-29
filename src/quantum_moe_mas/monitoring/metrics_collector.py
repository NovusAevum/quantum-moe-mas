"""
Comprehensive Performance Metrics Collection System

Implements Prometheus-based metrics collection with sub-5-second latency tracking
and automated performance analysis for the Quantum MoE MAS system.

Requirements: 8.1, 8.2
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import structlog

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics tracked by the system."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    API_CALLS = "api_calls"
    ROUTING_DECISIONS = "routing_decisions"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics data structure."""
    
    # Core Performance Metrics
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput_qps: float = 0.0
    error_rate: float = 0.0
    
    # Resource Utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: float = 0.0
    
    # System-Specific Metrics
    cache_hit_rate: float = 0.0
    api_call_count: int = 0
    routing_accuracy: float = 0.0
    expert_utilization: Dict[str, float] = field(default_factory=dict)
    
    # Business Metrics
    user_satisfaction_score: float = 0.0
    cost_per_query: float = 0.0
    roi_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    collection_duration: float = 0.0
    
    def meets_sla_targets(self) -> bool:
        """Check if metrics meet SLA targets (sub-5s for 95% of queries)."""
        return self.latency_p95 < 5.0 and self.error_rate < 0.01


class MetricsCollector:
    """
    Comprehensive metrics collection system with Prometheus integration.
    
    Provides real-time performance monitoring, latency tracking, and
    automated bottleneck identification for the Quantum MoE MAS system.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize the metrics collector with Prometheus registry."""
        self.registry = registry or CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Internal state
        self._latency_buffer = deque(maxlen=10000)  # Last 10k requests
        self._metrics_cache = {}
        self._collection_lock = threading.RLock()
        self._is_collecting = False
        
        # Performance thresholds
        self.latency_sla_target = 5.0  # seconds
        self.error_rate_threshold = 0.01  # 1%
        self.cache_hit_target = 0.8  # 80%
        
        logger.info("MetricsCollector initialized", 
                   sla_target=self.latency_sla_target)
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics collectors."""
        
        # Latency metrics
        self.latency_histogram = Histogram(
            'quantum_moe_request_duration_seconds',
            'Request duration in seconds',
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.latency_summary = Summary(
            'quantum_moe_request_latency_seconds',
            'Request latency summary',
            registry=self.registry
        )
        
        # Throughput metrics
        self.request_counter = Counter(
            'quantum_moe_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'quantum_moe_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Resource utilization
        self.cpu_gauge = Gauge(
            'quantum_moe_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_gauge = Gauge(
            'quantum_moe_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        # System-specific metrics
        self.cache_hit_rate = Gauge(
            'quantum_moe_cache_hit_rate',
            'Cache hit rate percentage',
            registry=self.registry
        )
        
        self.api_calls_counter = Counter(
            'quantum_moe_api_calls_total',
            'Total API calls made',
            ['provider', 'endpoint'],
            registry=self.registry
        )
        
        self.routing_accuracy = Gauge(
            'quantum_moe_routing_accuracy',
            'MoE routing accuracy percentage',
            registry=self.registry
        )
        
        # Expert utilization
        self.expert_utilization = Gauge(
            'quantum_moe_expert_utilization',
            'Expert utilization percentage',
            ['expert_id', 'expert_type'],
            registry=self.registry
        )
    
    async def record_request_latency(self, 
                                   latency: float, 
                                   method: str = "POST",
                                   endpoint: str = "/query",
                                   status: str = "success") -> None:
        """Record request latency with automatic SLA monitoring."""
        
        # Record in Prometheus
        self.latency_histogram.observe(latency)
        self.latency_summary.observe(latency)
        self.request_counter.labels(
            method=method, 
            endpoint=endpoint, 
            status=status
        ).inc()
        
        # Update internal buffer
        with self._collection_lock:
            self._latency_buffer.append({
                'latency': latency,
                'timestamp': datetime.now(),
                'method': method,
                'endpoint': endpoint,
                'status': status
            })
        
        # Check SLA violation
        if latency > self.latency_sla_target:
            logger.warning("SLA violation detected",
                         latency=latency,
                         target=self.latency_sla_target,
                         endpoint=endpoint)
            
            await self._trigger_performance_alert(latency, endpoint)
    
    async def record_error(self, 
                          error_type: str, 
                          component: str, 
                          details: Optional[Dict[str, Any]] = None) -> None:
        """Record system errors with categorization."""
        
        self.error_counter.labels(
            error_type=error_type,
            component=component
        ).inc()
        
        logger.error("System error recorded",
                    error_type=error_type,
                    component=component,
                    details=details or {})
    
    def update_resource_metrics(self, 
                               cpu_percent: float,
                               memory_bytes: int,
                               disk_percent: float = 0.0,
                               network_io: float = 0.0) -> None:
        """Update system resource utilization metrics."""
        
        self.cpu_gauge.set(cpu_percent)
        self.memory_gauge.set(memory_bytes)
        
        logger.debug("Resource metrics updated",
                    cpu=cpu_percent,
                    memory_mb=memory_bytes / 1024 / 1024)
    
    def update_cache_metrics(self, hit_rate: float) -> None:
        """Update cache performance metrics."""
        
        self.cache_hit_rate.set(hit_rate)
        
        if hit_rate < self.cache_hit_target:
            logger.warning("Cache hit rate below target",
                         current=hit_rate,
                         target=self.cache_hit_target)
    
    def record_api_call(self, provider: str, endpoint: str) -> None:
        """Record external API calls for cost tracking."""
        
        self.api_calls_counter.labels(
            provider=provider,
            endpoint=endpoint
        ).inc()
    
    def update_routing_metrics(self, accuracy: float) -> None:
        """Update MoE routing accuracy metrics."""
        
        self.routing_accuracy.set(accuracy)
        
        logger.debug("Routing accuracy updated", accuracy=accuracy)
    
    def update_expert_utilization(self, 
                                 expert_metrics: Dict[str, Dict[str, float]]) -> None:
        """Update expert utilization metrics."""
        
        for expert_id, metrics in expert_metrics.items():
            utilization = metrics.get('utilization', 0.0)
            expert_type = metrics.get('type', 'unknown')
            
            self.expert_utilization.labels(
                expert_id=expert_id,
                expert_type=expert_type
            ).set(utilization)
    
    async def collect_comprehensive_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics snapshot."""
        
        start_time = time.time()
        
        with self._collection_lock:
            # Calculate latency percentiles
            if self._latency_buffer:
                latencies = [req['latency'] for req in self._latency_buffer]
                latency_p50 = statistics.median(latencies)
                latency_p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
                latency_p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            else:
                latency_p50 = latency_p95 = latency_p99 = 0.0
            
            # Calculate throughput (requests per second)
            recent_requests = [
                req for req in self._latency_buffer 
                if req['timestamp'] > datetime.now() - timedelta(seconds=60)
            ]
            throughput_qps = len(recent_requests) / 60.0
            
            # Calculate error rate
            error_requests = [req for req in recent_requests if req['status'] == 'error']
            error_rate = len(error_requests) / max(len(recent_requests), 1)
        
        collection_duration = time.time() - start_time
        
        metrics = PerformanceMetrics(
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput_qps=throughput_qps,
            error_rate=error_rate,
            collection_duration=collection_duration,
            timestamp=datetime.now()
        )
        
        logger.info("Comprehensive metrics collected",
                   latency_p95=latency_p95,
                   throughput=throughput_qps,
                   error_rate=error_rate,
                   meets_sla=metrics.meets_sla_targets())
        
        return metrics
    
    async def _trigger_performance_alert(self, latency: float, endpoint: str) -> None:
        """Trigger performance alert for SLA violations."""
        
        alert_data = {
            'alert_type': 'performance_sla_violation',
            'latency': latency,
            'target': self.latency_sla_target,
            'endpoint': endpoint,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if latency > self.latency_sla_target * 2 else 'medium'
        }
        
        # Here you would integrate with your alerting system
        # For now, we'll just log the alert
        logger.critical("Performance SLA violation alert", **alert_data)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary for dashboard display."""
        
        with self._collection_lock:
            if not self._latency_buffer:
                return {'status': 'no_data', 'message': 'No metrics available'}
            
            recent_latencies = [
                req['latency'] for req in self._latency_buffer
                if req['timestamp'] > datetime.now() - timedelta(minutes=5)
            ]
            
            if not recent_latencies:
                return {'status': 'stale_data', 'message': 'No recent metrics'}
            
            return {
                'status': 'healthy',
                'avg_latency': statistics.mean(recent_latencies),
                'p95_latency': statistics.quantiles(recent_latencies, n=20)[18] if len(recent_latencies) > 20 else max(recent_latencies),
                'request_count': len(recent_latencies),
                'sla_compliance': sum(1 for lat in recent_latencies if lat < self.latency_sla_target) / len(recent_latencies),
                'last_updated': datetime.now().isoformat()
            }
    
    async def start_background_collection(self, interval: int = 30) -> None:
        """Start background metrics collection task."""
        
        if self._is_collecting:
            logger.warning("Background collection already running")
            return
        
        self._is_collecting = True
        logger.info("Starting background metrics collection", interval=interval)
        
        while self._is_collecting:
            try:
                metrics = await self.collect_comprehensive_metrics()
                
                # Store in cache for quick access
                self._metrics_cache['latest'] = metrics
                
                # Clean old data from buffer
                cutoff_time = datetime.now() - timedelta(hours=1)
                with self._collection_lock:
                    self._latency_buffer = deque([
                        req for req in self._latency_buffer
                        if req['timestamp'] > cutoff_time
                    ], maxlen=10000)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error("Error in background metrics collection", error=str(e))
                await asyncio.sleep(interval)
    
    def stop_background_collection(self) -> None:
        """Stop background metrics collection."""
        
        self._is_collecting = False
        logger.info("Background metrics collection stopped")