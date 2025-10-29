"""
Latency Tracking System with Sub-5-Second SLA Monitoring

Provides comprehensive latency tracking with real-time SLA monitoring,
percentile calculations, and automated alerting for performance violations.

Requirements: 8.1, 8.2
"""

import time
import asyncio
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics
import threading
from contextlib import asynccontextmanager

import structlog

logger = structlog.get_logger(__name__)


class LatencyCategory(Enum):
    """Categories for latency tracking."""
    ROUTING = "routing"
    EXPERT_INFERENCE = "expert_inference"
    RAG_RETRIEVAL = "rag_retrieval"
    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    TOTAL_REQUEST = "total_request"


@dataclass
class LatencyMeasurement:
    """Individual latency measurement record."""
    
    category: LatencyCategory
    operation: str
    latency_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def latency_seconds(self) -> float:
        """Get latency in seconds."""
        return self.latency_ms / 1000.0
    
    def exceeds_sla(self, sla_threshold_ms: float = 5000.0) -> bool:
        """Check if measurement exceeds SLA threshold."""
        return self.latency_ms > sla_threshold_ms


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""
    
    category: LatencyCategory
    operation: str
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    sla_violations: int
    sla_compliance_rate: float
    time_window: timedelta
    
    def meets_sla_target(self, target_ms: float = 5000.0) -> bool:
        """Check if stats meet SLA target (95th percentile < target)."""
        return self.p95_ms < target_ms


class LatencyTracker:
    """
    Comprehensive latency tracking system with SLA monitoring.
    
    Tracks latency across different operation categories with real-time
    SLA monitoring, percentile calculations, and automated alerting.
    """
    
    def __init__(self, 
                 sla_threshold_ms: float = 5000.0,
                 max_measurements: int = 100000,
                 alert_callback: Optional[Callable] = None):
        """Initialize latency tracker."""
        
        self.sla_threshold_ms = sla_threshold_ms
        self.max_measurements = max_measurements
        self.alert_callback = alert_callback
        
        # Storage for measurements
        self._measurements: Dict[LatencyCategory, deque] = defaultdict(
            lambda: deque(maxlen=max_measurements)
        )
        self._operation_measurements: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_measurements)
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # SLA violation tracking
        self._sla_violations = defaultdict(int)
        self._consecutive_violations = defaultdict(int)
        
        # Performance optimization
        self._stats_cache: Dict[str, Tuple[LatencyStats, datetime]] = {}
        self._cache_ttl = timedelta(seconds=30)
        
        logger.info("LatencyTracker initialized", 
                   sla_threshold_ms=sla_threshold_ms,
                   max_measurements=max_measurements)
    
    @asynccontextmanager
    async def track_latency(self, 
                           category: LatencyCategory,
                           operation: str,
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for automatic latency tracking.
        
        Usage:
            async with tracker.track_latency(LatencyCategory.ROUTING, "quantum_route"):
                # Your operation here
                result = await some_operation()
        """
        
        start_time = time.perf_counter()
        start_timestamp = datetime.now()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0
            
            await self.record_latency(
                category=category,
                operation=operation,
                latency_ms=latency_ms,
                timestamp=start_timestamp,
                metadata=metadata or {}
            )
    
    async def record_latency(self,
                           category: LatencyCategory,
                           operation: str,
                           latency_ms: float,
                           timestamp: Optional[datetime] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a latency measurement."""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        measurement = LatencyMeasurement(
            category=category,
            operation=operation,
            latency_ms=latency_ms,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Store by category
            self._measurements[category].append(measurement)
            
            # Store by operation
            operation_key = f"{category.value}:{operation}"
            self._operation_measurements[operation_key].append(measurement)
        
        # Check for SLA violation
        if measurement.exceeds_sla(self.sla_threshold_ms):
            await self._handle_sla_violation(measurement)
        else:
            # Reset consecutive violations counter
            with self._lock:
                self._consecutive_violations[operation_key] = 0
        
        # Invalidate cache for this category/operation
        self._invalidate_cache(category, operation)
        
        logger.debug("Latency recorded",
                    category=category.value,
                    operation=operation,
                    latency_ms=latency_ms,
                    sla_violation=measurement.exceeds_sla(self.sla_threshold_ms))
    
    def get_latency_stats(self,
                         category: LatencyCategory,
                         operation: Optional[str] = None,
                         time_window: Optional[timedelta] = None) -> Optional[LatencyStats]:
        """Get latency statistics for a category or specific operation."""
        
        if time_window is None:
            time_window = timedelta(minutes=15)
        
        cache_key = f"{category.value}:{operation or 'all'}:{time_window.total_seconds()}"
        
        # Check cache first
        if cache_key in self._stats_cache:
            stats, cached_at = self._stats_cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return stats
        
        with self._lock:
            # Get measurements for the specified time window
            cutoff_time = datetime.now() - time_window
            
            if operation:
                operation_key = f"{category.value}:{operation}"
                measurements = [
                    m for m in self._operation_measurements[operation_key]
                    if m.timestamp > cutoff_time
                ]
            else:
                measurements = [
                    m for m in self._measurements[category]
                    if m.timestamp > cutoff_time
                ]
            
            if not measurements:
                return None
            
            # Calculate statistics
            latencies = [m.latency_ms for m in measurements]
            sla_violations = sum(1 for m in measurements if m.exceeds_sla(self.sla_threshold_ms))
            
            stats = LatencyStats(
                category=category,
                operation=operation or "all",
                count=len(measurements),
                min_ms=min(latencies),
                max_ms=max(latencies),
                mean_ms=statistics.mean(latencies),
                median_ms=statistics.median(latencies),
                p95_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
                p99_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies),
                sla_violations=sla_violations,
                sla_compliance_rate=(len(measurements) - sla_violations) / len(measurements),
                time_window=time_window
            )
            
            # Cache the result
            self._stats_cache[cache_key] = (stats, datetime.now())
            
            return stats
    
    def get_all_category_stats(self, 
                              time_window: Optional[timedelta] = None) -> Dict[LatencyCategory, LatencyStats]:
        """Get latency statistics for all categories."""
        
        if time_window is None:
            time_window = timedelta(minutes=15)
        
        stats = {}
        
        for category in LatencyCategory:
            category_stats = self.get_latency_stats(category, time_window=time_window)
            if category_stats:
                stats[category] = category_stats
        
        return stats
    
    def get_operation_breakdown(self, 
                               category: LatencyCategory,
                               time_window: Optional[timedelta] = None) -> Dict[str, LatencyStats]:
        """Get latency breakdown by operation within a category."""
        
        if time_window is None:
            time_window = timedelta(minutes=15)
        
        with self._lock:
            # Find all operations for this category
            operations = set()
            for key in self._operation_measurements.keys():
                if key.startswith(f"{category.value}:"):
                    operation = key.split(":", 1)[1]
                    operations.add(operation)
        
        breakdown = {}
        for operation in operations:
            stats = self.get_latency_stats(category, operation, time_window)
            if stats:
                breakdown[operation] = stats
        
        return breakdown
    
    def get_sla_compliance_summary(self) -> Dict[str, Any]:
        """Get overall SLA compliance summary."""
        
        with self._lock:
            total_measurements = sum(len(deque_obj) for deque_obj in self._measurements.values())
            total_violations = sum(self._sla_violations.values())
        
        if total_measurements == 0:
            return {
                'overall_compliance_rate': 1.0,
                'total_measurements': 0,
                'total_violations': 0,
                'categories': {}
            }
        
        overall_compliance = (total_measurements - total_violations) / total_measurements
        
        # Get compliance by category
        category_compliance = {}
        for category in LatencyCategory:
            stats = self.get_latency_stats(category)
            if stats:
                category_compliance[category.value] = {
                    'compliance_rate': stats.sla_compliance_rate,
                    'violations': stats.sla_violations,
                    'total_requests': stats.count,
                    'p95_latency_ms': stats.p95_ms,
                    'meets_sla': stats.meets_sla_target(self.sla_threshold_ms)
                }
        
        return {
            'overall_compliance_rate': overall_compliance,
            'total_measurements': total_measurements,
            'total_violations': total_violations,
            'sla_threshold_ms': self.sla_threshold_ms,
            'categories': category_compliance
        }
    
    async def _handle_sla_violation(self, measurement: LatencyMeasurement) -> None:
        """Handle SLA violation with alerting and tracking."""
        
        operation_key = f"{measurement.category.value}:{measurement.operation}"
        
        with self._lock:
            self._sla_violations[operation_key] += 1
            self._consecutive_violations[operation_key] += 1
        
        # Determine alert severity based on consecutive violations
        consecutive = self._consecutive_violations[operation_key]
        if consecutive >= 5:
            severity = "critical"
        elif consecutive >= 3:
            severity = "high"
        else:
            severity = "medium"
        
        alert_data = {
            'alert_type': 'sla_violation',
            'category': measurement.category.value,
            'operation': measurement.operation,
            'latency_ms': measurement.latency_ms,
            'threshold_ms': self.sla_threshold_ms,
            'consecutive_violations': consecutive,
            'severity': severity,
            'timestamp': measurement.timestamp.isoformat(),
            'metadata': measurement.metadata
        }
        
        logger.warning("SLA violation detected", **alert_data)
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                await self.alert_callback(alert_data)
            except Exception as e:
                logger.error("Error in SLA violation callback", error=str(e))
    
    def _invalidate_cache(self, category: LatencyCategory, operation: str) -> None:
        """Invalidate cached statistics for category/operation."""
        
        keys_to_remove = []
        for key in self._stats_cache.keys():
            if key.startswith(f"{category.value}:"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._stats_cache.pop(key, None)
    
    def cleanup_old_measurements(self, max_age: timedelta = timedelta(hours=24)) -> int:
        """Clean up old measurements to prevent memory bloat."""
        
        cutoff_time = datetime.now() - max_age
        removed_count = 0
        
        with self._lock:
            # Clean category measurements
            for category, measurements in self._measurements.items():
                original_len = len(measurements)
                # Convert to list, filter, convert back to deque
                filtered = [m for m in measurements if m.timestamp > cutoff_time]
                measurements.clear()
                measurements.extend(filtered)
                removed_count += original_len - len(measurements)
            
            # Clean operation measurements
            for operation_key, measurements in self._operation_measurements.items():
                original_len = len(measurements)
                filtered = [m for m in measurements if m.timestamp > cutoff_time]
                measurements.clear()
                measurements.extend(filtered)
                removed_count += original_len - len(measurements)
        
        # Clear cache
        self._stats_cache.clear()
        
        logger.info("Cleaned up old measurements", 
                   removed_count=removed_count,
                   max_age_hours=max_age.total_seconds() / 3600)
        
        return removed_count
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for dashboards."""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'sla_threshold_ms': self.sla_threshold_ms,
            'categories': {},
            'top_slowest_operations': [],
            'sla_compliance': self.get_sla_compliance_summary()
        }
        
        # Get stats for each category
        for category in LatencyCategory:
            stats = self.get_latency_stats(category)
            if stats:
                summary['categories'][category.value] = {
                    'count': stats.count,
                    'p95_ms': stats.p95_ms,
                    'p99_ms': stats.p99_ms,
                    'sla_compliance_rate': stats.sla_compliance_rate,
                    'meets_sla': stats.meets_sla_target(self.sla_threshold_ms)
                }
        
        # Find slowest operations across all categories
        all_operations = []
        for category in LatencyCategory:
            breakdown = self.get_operation_breakdown(category)
            for operation, stats in breakdown.items():
                all_operations.append({
                    'category': category.value,
                    'operation': operation,
                    'p95_ms': stats.p95_ms,
                    'count': stats.count,
                    'sla_compliance_rate': stats.sla_compliance_rate
                })
        
        # Sort by p95 latency and take top 10
        all_operations.sort(key=lambda x: x['p95_ms'], reverse=True)
        summary['top_slowest_operations'] = all_operations[:10]
        
        return summary