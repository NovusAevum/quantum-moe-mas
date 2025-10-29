"""
Resource Utilization Monitoring System

Provides comprehensive system resource monitoring including CPU, memory,
disk, and network utilization with automated alerting and optimization recommendations.

Requirements: 8.1, 8.2
"""

import asyncio
import psutil
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading
import statistics

import structlog

logger = structlog.get_logger(__name__)


class ResourceType(Enum):
    """Types of system resources monitored."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class AlertLevel(Enum):
    """Alert levels for resource utilization."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ResourceMeasurement:
    """Individual resource utilization measurement."""
    
    resource_type: ResourceType
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if measurement exceeds threshold."""
        return self.value > threshold


@dataclass
class ResourceThresholds:
    """Resource utilization thresholds for alerting."""
    
    cpu_warning: float = 70.0  # %
    cpu_critical: float = 90.0  # %
    memory_warning: float = 80.0  # %
    memory_critical: float = 95.0  # %
    disk_warning: float = 85.0  # %
    disk_critical: float = 95.0  # %
    network_warning: float = 80.0  # % of available bandwidth
    network_critical: float = 95.0  # % of available bandwidth


@dataclass
class ResourceStats:
    """Statistical summary of resource utilization."""
    
    resource_type: ResourceType
    count: int
    min_value: float
    max_value: float
    mean_value: float
    current_value: float
    trend: str  # "increasing", "decreasing", "stable"
    alert_level: AlertLevel
    time_window: timedelta


class ResourceMonitor:
    """
    Comprehensive system resource monitoring with alerting.
    
    Monitors CPU, memory, disk, and network utilization with configurable
    thresholds, trend analysis, and automated optimization recommendations.
    """
    
    def __init__(self,
                 thresholds: Optional[ResourceThresholds] = None,
                 collection_interval: int = 30,
                 max_measurements: int = 10000,
                 alert_callback: Optional[Callable] = None):
        """Initialize resource monitor."""
        
        self.thresholds = thresholds or ResourceThresholds()
        self.collection_interval = collection_interval
        self.max_measurements = max_measurements
        self.alert_callback = alert_callback
        
        # Storage for measurements
        self._measurements: Dict[ResourceType, deque] = {
            resource_type: deque(maxlen=max_measurements)
            for resource_type in ResourceType
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring state
        self._is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Alert state tracking
        self._last_alerts: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=5)
        
        # Network baseline (for percentage calculations)
        self._network_baseline_mbps = 100.0  # Default 100 Mbps
        
        logger.info("ResourceMonitor initialized",
                   collection_interval=collection_interval,
                   thresholds=thresholds.__dict__)
    
    async def start_monitoring(self) -> None:
        """Start continuous resource monitoring."""
        
        if self._is_monitoring:
            logger.warning("Resource monitoring already running")
            return
        
        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Resource monitoring started",
                   interval=self.collection_interval)
    
    async def stop_monitoring(self) -> None:
        """Stop continuous resource monitoring."""
        
        if not self._is_monitoring:
            logger.warning("Resource monitoring not running")
            return
        
        self._is_monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        
        while self._is_monitoring:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def collect_all_metrics(self) -> Dict[ResourceType, ResourceMeasurement]:
        """Collect all resource metrics."""
        
        measurements = {}
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_measurement = ResourceMeasurement(
            resource_type=ResourceType.CPU,
            value=cpu_percent,
            unit="percent",
            timestamp=datetime.now(),
            metadata={
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
        )
        measurements[ResourceType.CPU] = cpu_measurement
        await self._record_measurement(cpu_measurement)
        
        # Memory utilization
        memory = psutil.virtual_memory()
        memory_measurement = ResourceMeasurement(
            resource_type=ResourceType.MEMORY,
            value=memory.percent,
            unit="percent",
            timestamp=datetime.now(),
            metadata={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3)
            }
        )
        measurements[ResourceType.MEMORY] = memory_measurement
        await self._record_measurement(memory_measurement)
        
        # Disk utilization
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_measurement = ResourceMeasurement(
            resource_type=ResourceType.DISK,
            value=disk_percent,
            unit="percent",
            timestamp=datetime.now(),
            metadata={
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3)
            }
        )
        measurements[ResourceType.DISK] = disk_measurement
        await self._record_measurement(disk_measurement)
        
        # Network utilization
        network_stats = psutil.net_io_counters()
        # Calculate network utilization as percentage of baseline
        # This is a simplified calculation - in production you'd want more sophisticated bandwidth monitoring
        network_mbps = (network_stats.bytes_sent + network_stats.bytes_recv) / (1024**2) / self.collection_interval
        network_percent = min((network_mbps / self._network_baseline_mbps) * 100, 100)
        
        network_measurement = ResourceMeasurement(
            resource_type=ResourceType.NETWORK,
            value=network_percent,
            unit="percent",
            timestamp=datetime.now(),
            metadata={
                "bytes_sent": network_stats.bytes_sent,
                "bytes_recv": network_stats.bytes_recv,
                "packets_sent": network_stats.packets_sent,
                "packets_recv": network_stats.packets_recv,
                "current_mbps": network_mbps
            }
        )
        measurements[ResourceType.NETWORK] = network_measurement
        await self._record_measurement(network_measurement)
        
        logger.debug("Resource metrics collected",
                    cpu=cpu_percent,
                    memory=memory.percent,
                    disk=disk_percent,
                    network=network_percent)
        
        return measurements
    
    async def _record_measurement(self, measurement: ResourceMeasurement) -> None:
        """Record a resource measurement and check for alerts."""
        
        with self._lock:
            self._measurements[measurement.resource_type].append(measurement)
        
        # Check for threshold violations
        alert_level = self._check_thresholds(measurement)
        if alert_level != AlertLevel.INFO:
            await self._handle_resource_alert(measurement, alert_level)
    
    def _check_thresholds(self, measurement: ResourceMeasurement) -> AlertLevel:
        """Check if measurement exceeds thresholds."""
        
        if measurement.resource_type == ResourceType.CPU:
            if measurement.value >= self.thresholds.cpu_critical:
                return AlertLevel.CRITICAL
            elif measurement.value >= self.thresholds.cpu_warning:
                return AlertLevel.WARNING
        
        elif measurement.resource_type == ResourceType.MEMORY:
            if measurement.value >= self.thresholds.memory_critical:
                return AlertLevel.CRITICAL
            elif measurement.value >= self.thresholds.memory_warning:
                return AlertLevel.WARNING
        
        elif measurement.resource_type == ResourceType.DISK:
            if measurement.value >= self.thresholds.disk_critical:
                return AlertLevel.CRITICAL
            elif measurement.value >= self.thresholds.disk_warning:
                return AlertLevel.WARNING
        
        elif measurement.resource_type == ResourceType.NETWORK:
            if measurement.value >= self.thresholds.network_critical:
                return AlertLevel.CRITICAL
            elif measurement.value >= self.thresholds.network_warning:
                return AlertLevel.WARNING
        
        return AlertLevel.INFO
    
    async def _handle_resource_alert(self, 
                                   measurement: ResourceMeasurement,
                                   alert_level: AlertLevel) -> None:
        """Handle resource threshold violation alert."""
        
        alert_key = f"{measurement.resource_type.value}_{alert_level.value}"
        
        # Check cooldown period
        if alert_key in self._last_alerts:
            if datetime.now() - self._last_alerts[alert_key] < self._alert_cooldown:
                return
        
        self._last_alerts[alert_key] = datetime.now()
        
        alert_data = {
            'alert_type': 'resource_threshold_violation',
            'resource_type': measurement.resource_type.value,
            'current_value': measurement.value,
            'unit': measurement.unit,
            'alert_level': alert_level.value,
            'timestamp': measurement.timestamp.isoformat(),
            'metadata': measurement.metadata,
            'recommendations': self._get_optimization_recommendations(measurement)
        }
        
        logger.warning("Resource threshold violation",
                      resource=measurement.resource_type.value,
                      value=measurement.value,
                      level=alert_level.value)
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                await self.alert_callback(alert_data)
            except Exception as e:
                logger.error("Error in resource alert callback", error=str(e))
    
    def _get_optimization_recommendations(self, 
                                        measurement: ResourceMeasurement) -> List[str]:
        """Get optimization recommendations based on resource usage."""
        
        recommendations = []
        
        if measurement.resource_type == ResourceType.CPU:
            if measurement.value > 90:
                recommendations.extend([
                    "Consider scaling horizontally by adding more instances",
                    "Review CPU-intensive operations for optimization",
                    "Implement request queuing to smooth load spikes",
                    "Consider upgrading to higher CPU capacity instances"
                ])
            elif measurement.value > 70:
                recommendations.extend([
                    "Monitor for sustained high CPU usage",
                    "Consider implementing caching to reduce computation",
                    "Review algorithm efficiency in hot code paths"
                ])
        
        elif measurement.resource_type == ResourceType.MEMORY:
            if measurement.value > 95:
                recommendations.extend([
                    "Immediate action required - risk of OOM kills",
                    "Scale up memory or add more instances",
                    "Review memory leaks in application code",
                    "Implement memory-efficient data structures"
                ])
            elif measurement.value > 80:
                recommendations.extend([
                    "Monitor memory growth trends",
                    "Implement garbage collection tuning",
                    "Consider memory caching optimizations",
                    "Review large object allocations"
                ])
        
        elif measurement.resource_type == ResourceType.DISK:
            if measurement.value > 95:
                recommendations.extend([
                    "Critical - disk space nearly full",
                    "Clean up temporary files and logs",
                    "Implement log rotation policies",
                    "Consider adding storage capacity"
                ])
            elif measurement.value > 85:
                recommendations.extend([
                    "Monitor disk usage growth",
                    "Implement data archival strategies",
                    "Review large file storage patterns"
                ])
        
        elif measurement.resource_type == ResourceType.NETWORK:
            if measurement.value > 95:
                recommendations.extend([
                    "Network bandwidth nearly saturated",
                    "Consider CDN for static content",
                    "Implement response compression",
                    "Review data transfer patterns"
                ])
            elif measurement.value > 80:
                recommendations.extend([
                    "Monitor network usage patterns",
                    "Consider request batching",
                    "Implement efficient serialization"
                ])
        
        return recommendations
    
    def get_resource_stats(self,
                          resource_type: ResourceType,
                          time_window: Optional[timedelta] = None) -> Optional[ResourceStats]:
        """Get resource statistics for a specific resource type."""
        
        if time_window is None:
            time_window = timedelta(minutes=15)
        
        with self._lock:
            measurements = self._measurements[resource_type]
            
            if not measurements:
                return None
            
            # Filter by time window
            cutoff_time = datetime.now() - time_window
            recent_measurements = [
                m for m in measurements
                if m.timestamp > cutoff_time
            ]
            
            if not recent_measurements:
                return None
            
            values = [m.value for m in recent_measurements]
            current_value = recent_measurements[-1].value
            
            # Calculate trend
            if len(values) >= 3:
                recent_avg = statistics.mean(values[-3:])
                older_avg = statistics.mean(values[:3])
                if recent_avg > older_avg * 1.1:
                    trend = "increasing"
                elif recent_avg < older_avg * 0.9:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            # Determine alert level
            alert_level = self._check_thresholds(recent_measurements[-1])
            
            return ResourceStats(
                resource_type=resource_type,
                count=len(recent_measurements),
                min_value=min(values),
                max_value=max(values),
                mean_value=statistics.mean(values),
                current_value=current_value,
                trend=trend,
                alert_level=alert_level,
                time_window=time_window
            )
    
    def get_all_resource_stats(self,
                              time_window: Optional[timedelta] = None) -> Dict[ResourceType, ResourceStats]:
        """Get resource statistics for all resource types."""
        
        stats = {}
        for resource_type in ResourceType:
            resource_stats = self.get_resource_stats(resource_type, time_window)
            if resource_stats:
                stats[resource_type] = resource_stats
        
        return stats
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        
        all_stats = self.get_all_resource_stats()
        
        # Determine overall health status
        critical_resources = [
            stats for stats in all_stats.values()
            if stats.alert_level == AlertLevel.CRITICAL
        ]
        
        warning_resources = [
            stats for stats in all_stats.values()
            if stats.alert_level == AlertLevel.WARNING
        ]
        
        if critical_resources:
            overall_status = "critical"
        elif warning_resources:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Calculate resource utilization summary
        resource_summary = {}
        for resource_type, stats in all_stats.items():
            resource_summary[resource_type.value] = {
                'current_percent': stats.current_value,
                'trend': stats.trend,
                'alert_level': stats.alert_level.value,
                'mean_percent': stats.mean_value,
                'max_percent': stats.max_value
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'critical_count': len(critical_resources),
            'warning_count': len(warning_resources),
            'resources': resource_summary,
            'recommendations': self._get_system_recommendations(all_stats)
        }
    
    def _get_system_recommendations(self, 
                                  all_stats: Dict[ResourceType, ResourceStats]) -> List[str]:
        """Get system-wide optimization recommendations."""
        
        recommendations = []
        
        # Check for multiple resource constraints
        high_usage_resources = [
            resource_type for resource_type, stats in all_stats.items()
            if stats.current_value > 80
        ]
        
        if len(high_usage_resources) >= 2:
            recommendations.append(
                "Multiple resources under pressure - consider horizontal scaling"
            )
        
        # Check for trending issues
        increasing_resources = [
            resource_type for resource_type, stats in all_stats.items()
            if stats.trend == "increasing" and stats.current_value > 60
        ]
        
        if increasing_resources:
            recommendations.append(
                f"Resources showing increasing usage trend: {[r.value for r in increasing_resources]}"
            )
        
        # System-specific recommendations
        if ResourceType.CPU in all_stats and ResourceType.MEMORY in all_stats:
            cpu_stats = all_stats[ResourceType.CPU]
            memory_stats = all_stats[ResourceType.MEMORY]
            
            if cpu_stats.current_value > 80 and memory_stats.current_value < 50:
                recommendations.append(
                    "High CPU with low memory usage - consider CPU optimization or scaling"
                )
            elif memory_stats.current_value > 80 and cpu_stats.current_value < 50:
                recommendations.append(
                    "High memory with low CPU usage - review memory efficiency"
                )
        
        return recommendations
    
    def cleanup_old_measurements(self, max_age: timedelta = timedelta(hours=24)) -> int:
        """Clean up old measurements to prevent memory bloat."""
        
        cutoff_time = datetime.now() - max_age
        removed_count = 0
        
        with self._lock:
            for resource_type, measurements in self._measurements.items():
                original_len = len(measurements)
                # Convert to list, filter, convert back to deque
                filtered = [m for m in measurements if m.timestamp > cutoff_time]
                measurements.clear()
                measurements.extend(filtered)
                removed_count += original_len - len(measurements)
        
        logger.info("Cleaned up old resource measurements",
                   removed_count=removed_count,
                   max_age_hours=max_age.total_seconds() / 3600)
        
        return removed_count