"""
System-Wide Coordination and Monitoring

This module implements comprehensive system health monitoring, performance
metrics aggregation, configuration management, and distributed logging
and tracing capabilities for the MAS.

Requirements addressed: 5.4, 8.5

Author: Wan Mohamad Hanis bin Wan Hassan
"""

import asyncio
import json
import time
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from collections import defaultdict, deque

import structlog
from pydantic import BaseModel, Field, ConfigDict

from quantum_moe_mas.core.exceptions import QuantumMoEMASError
from quantum_moe_mas.core.logging_simple import get_logger


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class SystemMetric:
    """System metric data point."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unit: str = ""
    description: str = ""


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], bool]
    interval: float = 60.0  # seconds
    timeout: float = 30.0   # seconds
    critical: bool = False
    description: str = ""
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class Alert:
    """System alert."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""


@dataclass
class TraceSpan:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"


class SystemHealth(BaseModel):
    """Overall system health status."""
    overall_status: HealthStatus
    component_health: Dict[str, HealthStatus] = Field(default_factory=dict)
    active_alerts: int = 0
    critical_alerts: int = 0
    uptime: float = 0.0  # seconds
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    cpu_usage: float = Field(ge=0.0, le=100.0)
    memory_usage: float = Field(ge=0.0, le=100.0)
    disk_usage: float = Field(ge=0.0, le=100.0)
    network_io: Dict[str, float] = Field(default_factory=dict)
    active_connections: int = 0
    request_rate: float = 0.0  # requests per second
    error_rate: float = Field(ge=0.0, le=100.0, default=0.0)
    response_time_p95: float = 0.0  # 95th percentile response time
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConfigurationItem(BaseModel):
    """Configuration item."""
    key: str
    value: Any
    description: str = ""
    category: str = "general"
    sensitive: bool = False
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_by: str = "system"


class SystemMonitor:
    """
    System-wide coordination and monitoring service.
    
    Provides comprehensive monitoring capabilities including:
    - Health monitoring and alerting
    - Performance metrics collection and aggregation
    - System-wide configuration management
    - Distributed logging and tracing
    - Resource utilization tracking
    - Anomaly detection and alerting
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics_retention_hours: int = 24,
        alert_retention_hours: int = 168,  # 1 week
        trace_retention_hours: int = 24
    ):
        """
        Initialize the System Monitor.
        
        Args:
            config: Configuration dictionary
            metrics_retention_hours: How long to retain metrics
            alert_retention_hours: How long to retain alerts
            trace_retention_hours: How long to retain traces
        """
        self.config = config or {}
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_retention_hours = alert_retention_hours
        self.trace_retention_hours = trace_retention_hours
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.component_health: Dict[str, HealthStatus] = {}
        self.system_health = SystemHealth(overall_status=HealthStatus.UNKNOWN)
        
        # Metrics collection
        self.metrics: Dict[str, List[SystemMetric]] = defaultdict(list)
        self.performance_metrics: List[PerformanceMetrics] = []
        self.custom_metrics: Dict[str, Any] = {}
        
        # Alerting
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Configuration management
        self.configuration: Dict[str, ConfigurationItem] = {}
        self.config_watchers: List[Callable[[str, Any, Any], None]] = []
        
        # Distributed tracing
        self.traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self.active_spans: Dict[str, TraceSpan] = {}
        
        # System information
        self.start_time = datetime.now(timezone.utc)
        self.system_info = self._collect_system_info()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thresholds and limits
        self.cpu_warning_threshold = config.get("cpu_warning_threshold", 80.0) if config else 80.0
        self.cpu_critical_threshold = config.get("cpu_critical_threshold", 95.0) if config else 95.0
        self.memory_warning_threshold = config.get("memory_warning_threshold", 85.0) if config else 85.0
        self.memory_critical_threshold = config.get("memory_critical_threshold", 95.0) if config else 95.0
        self.disk_warning_threshold = config.get("disk_warning_threshold", 85.0) if config else 85.0
        self.disk_critical_threshold = config.get("disk_critical_threshold", 95.0) if config else 95.0
        
        self._logger = get_logger("system_monitor")
    
    async def start(self) -> None:
        """Start the system monitor."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize default health checks
        await self._initialize_default_health_checks()
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self._logger.info("System Monitor started")
    
    async def stop(self) -> None:
        """Stop the system monitor."""
        self._running = False
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._health_check_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._logger.info("System Monitor stopped")
    
    # Health Monitoring
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.component_health[health_check.name] = HealthStatus.UNKNOWN
        
        self._logger.info(
            "Health check registered",
            name=health_check.name,
            interval=health_check.interval,
            critical=health_check.critical
        )
    
    async def run_health_check(self, name: str) -> HealthStatus:
        """Run a specific health check."""
        health_check = self.health_checks.get(name)
        if not health_check:
            return HealthStatus.UNKNOWN
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                asyncio.create_task(self._run_check_function(health_check.check_function)),
                timeout=health_check.timeout
            )
            
            if result:
                status = HealthStatus.HEALTHY
                health_check.consecutive_failures = 0
            else:
                status = HealthStatus.WARNING
                health_check.consecutive_failures += 1
            
            # Check if we should mark as critical
            if health_check.consecutive_failures >= health_check.max_failures:
                status = HealthStatus.CRITICAL
            
            health_check.last_status = status
            health_check.last_check = datetime.now(timezone.utc)
            self.component_health[name] = status
            
            # Generate alert if critical
            if status == HealthStatus.CRITICAL and health_check.critical:
                await self._generate_health_alert(health_check, status)
            
            return status
            
        except asyncio.TimeoutError:
            status = HealthStatus.CRITICAL
            health_check.consecutive_failures += 1
            health_check.last_status = status
            health_check.last_check = datetime.now(timezone.utc)
            self.component_health[name] = status
            
            await self._generate_health_alert(health_check, status, "Health check timed out")
            
            return status
        
        except Exception as e:
            status = HealthStatus.CRITICAL
            health_check.consecutive_failures += 1
            health_check.last_status = status
            health_check.last_check = datetime.now(timezone.utc)
            self.component_health[name] = status
            
            await self._generate_health_alert(health_check, status, f"Health check failed: {str(e)}")
            
            return status
    
    async def get_system_health(self) -> SystemHealth:
        """Get overall system health status."""
        # Update overall status based on component health
        critical_components = sum(
            1 for status in self.component_health.values()
            if status == HealthStatus.CRITICAL
        )
        
        warning_components = sum(
            1 for status in self.component_health.values()
            if status == HealthStatus.WARNING
        )
        
        if critical_components > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_components > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Count active alerts
        active_alerts = len(self.active_alerts)
        critical_alerts = sum(
            1 for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.CRITICAL
        )
        
        # Calculate uptime
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        self.system_health = SystemHealth(
            overall_status=overall_status,
            component_health=self.component_health.copy(),
            active_alerts=active_alerts,
            critical_alerts=critical_alerts,
            uptime=uptime
        )
        
        return self.system_health
    
    # Metrics Collection
    
    def record_metric(self, metric: SystemMetric) -> None:
        """Record a system metric."""
        self.metrics[metric.name].append(metric)
        
        # Keep only recent metrics
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.metrics_retention_hours)
        self.metrics[metric.name] = [
            m for m in self.metrics[metric.name]
            if m.timestamp >= cutoff_time
        ]
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Record a counter metric."""
        metric = SystemMetric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a gauge metric."""
        metric = SystemMetric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None) -> None:
        """Record a timer metric."""
        metric = SystemMetric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            labels=labels or {},
            unit="seconds"
        )
        self.record_metric(metric)
    
    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": float(network.bytes_sent),
                "bytes_recv": float(network.bytes_recv),
                "packets_sent": float(network.packets_sent),
                "packets_recv": float(network.packets_recv)
            }
            
            # Active connections
            connections = len(psutil.net_connections())
            
            # Calculate request rate and error rate from recent metrics
            request_rate = self._calculate_request_rate()
            error_rate = self._calculate_error_rate()
            response_time_p95 = self._calculate_response_time_p95()
            
            metrics = PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=connections,
                request_rate=request_rate,
                error_rate=error_rate,
                response_time_p95=response_time_p95
            )
            
            # Store metrics
            self.performance_metrics.append(metrics)
            
            # Keep only recent metrics
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.metrics_retention_hours)
            self.performance_metrics = [
                m for m in self.performance_metrics
                if m.timestamp >= cutoff_time
            ]
            
            # Check thresholds and generate alerts
            await self._check_performance_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            self._logger.error("Failed to collect performance metrics", error=str(e))
            return PerformanceMetrics(cpu_usage=0, memory_usage=0, disk_usage=0)
    
    def get_metrics(self, name: str, time_range: Optional[timedelta] = None) -> List[SystemMetric]:
        """Get metrics by name within time range."""
        metrics = self.metrics.get(name, [])
        
        if time_range:
            cutoff_time = datetime.now(timezone.utc) - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_metric_summary(self, name: str, time_range: Optional[timedelta] = None) -> Dict[str, float]:
        """Get metric summary statistics."""
        metrics = self.get_metrics(name, time_range)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0.0
        }
    
    # Alerting
    
    async def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        labels: Dict[str, str] = None
    ) -> str:
        """Create a new alert."""
        alert_id = f"alert_{int(time.time() * 1000)}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            labels=labels or {}
        )
        
        self.alerts[alert_id] = alert
        self.active_alerts[alert_id] = alert
        
        self._logger.warning(
            "Alert created",
            alert_id=alert_id,
            title=title,
            severity=severity.value,
            source=source
        )
        
        return alert_id
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Resolve an alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        alert.resolution_notes = resolution_notes
        
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
        
        self._logger.info(
            "Alert resolved",
            alert_id=alert_id,
            resolution_notes=resolution_notes
        )
        
        return True
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    # Configuration Management
    
    def set_config(self, key: str, value: Any, description: str = "", category: str = "general", sensitive: bool = False) -> None:
        """Set a configuration value."""
        old_value = None
        if key in self.configuration:
            old_value = self.configuration[key].value
        
        config_item = ConfigurationItem(
            key=key,
            value=value,
            description=description,
            category=category,
            sensitive=sensitive,
            updated_by="system"
        )
        
        self.configuration[key] = config_item
        
        # Notify watchers
        for watcher in self.config_watchers:
            try:
                watcher(key, old_value, value)
            except Exception as e:
                self._logger.error("Config watcher failed", key=key, error=str(e))
        
        self._logger.info(
            "Configuration updated",
            key=key,
            category=category,
            sensitive=sensitive
        )
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        config_item = self.configuration.get(key)
        return config_item.value if config_item else default
    
    def get_config_category(self, category: str) -> Dict[str, Any]:
        """Get all configuration items in a category."""
        return {
            key: item.value
            for key, item in self.configuration.items()
            if item.category == category
        }
    
    def watch_config(self, watcher: Callable[[str, Any, Any], None]) -> None:
        """Register a configuration change watcher."""
        self.config_watchers.append(watcher)
    
    # Distributed Tracing
    
    def start_trace(self, operation_name: str, parent_span_id: Optional[str] = None) -> str:
        """Start a new trace span."""
        trace_id = f"trace_{int(time.time() * 1000000)}"
        span_id = f"span_{int(time.time() * 1000000)}"
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc)
        )
        
        self.traces[trace_id].append(span)
        self.active_spans[span_id] = span
        
        return span_id
    
    def finish_trace(self, span_id: str, status: str = "ok", tags: Dict[str, Any] = None) -> None:
        """Finish a trace span."""
        span = self.active_spans.get(span_id)
        if not span:
            return
        
        span.end_time = datetime.now(timezone.utc)
        span.duration = (span.end_time - span.start_time).total_seconds()
        span.status = status
        
        if tags:
            span.tags.update(tags)
        
        del self.active_spans[span_id]
    
    def add_trace_log(self, span_id: str, level: str, message: str, fields: Dict[str, Any] = None) -> None:
        """Add a log entry to a trace span."""
        span = self.active_spans.get(span_id)
        if not span:
            return
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "fields": fields or {}
        }
        
        span.logs.append(log_entry)
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        return self.traces.get(trace_id, [])
    
    # Private Implementation Methods
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect performance metrics
                await self.collect_performance_metrics()
                
                # Update system health
                await self.get_system_health()
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self._logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(30)
    
    async def _health_check_loop(self) -> None:
        """Health check execution loop."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Run due health checks
                for name, health_check in self.health_checks.items():
                    if (not health_check.last_check or
                        (current_time - health_check.last_check).total_seconds() >= health_check.interval):
                        
                        await self.run_health_check(name)
                
                # Sleep briefly
                await asyncio.sleep(10)
                
            except Exception as e:
                self._logger.error("Error in health check loop", error=str(e))
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for old data."""
        while self._running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self._logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(3600)
    
    async def _initialize_default_health_checks(self) -> None:
        """Initialize default system health checks."""
        # CPU health check
        cpu_check = HealthCheck(
            name="cpu_usage",
            check_function=lambda: psutil.cpu_percent(interval=1) < self.cpu_critical_threshold,
            interval=60.0,
            critical=True,
            description="Monitor CPU usage"
        )
        self.register_health_check(cpu_check)
        
        # Memory health check
        memory_check = HealthCheck(
            name="memory_usage",
            check_function=lambda: psutil.virtual_memory().percent < self.memory_critical_threshold,
            interval=60.0,
            critical=True,
            description="Monitor memory usage"
        )
        self.register_health_check(memory_check)
        
        # Disk health check
        disk_check = HealthCheck(
            name="disk_usage",
            check_function=lambda: (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100 < self.disk_critical_threshold,
            interval=300.0,  # Check every 5 minutes
            critical=True,
            description="Monitor disk usage"
        )
        self.register_health_check(disk_check)
    
    async def _run_check_function(self, check_function: Callable[[], bool]) -> bool:
        """Run a health check function."""
        if asyncio.iscoroutinefunction(check_function):
            return await check_function()
        else:
            return check_function()
    
    async def _generate_health_alert(self, health_check: HealthCheck, status: HealthStatus, message: str = "") -> None:
        """Generate an alert for health check failure."""
        severity = AlertSeverity.CRITICAL if health_check.critical else AlertSeverity.WARNING
        
        description = f"Health check '{health_check.name}' is {status.value}"
        if message:
            description += f": {message}"
        
        await self.create_alert(
            title=f"Health Check Failed: {health_check.name}",
            description=description,
            severity=severity,
            source="health_monitor",
            labels={"health_check": health_check.name, "status": status.value}
        )
    
    async def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check performance metrics against thresholds and generate alerts."""
        # CPU threshold checks
        if metrics.cpu_usage >= self.cpu_critical_threshold:
            await self.create_alert(
                title="Critical CPU Usage",
                description=f"CPU usage is {metrics.cpu_usage:.1f}% (threshold: {self.cpu_critical_threshold}%)",
                severity=AlertSeverity.CRITICAL,
                source="performance_monitor",
                labels={"metric": "cpu_usage", "value": str(metrics.cpu_usage)}
            )
        elif metrics.cpu_usage >= self.cpu_warning_threshold:
            await self.create_alert(
                title="High CPU Usage",
                description=f"CPU usage is {metrics.cpu_usage:.1f}% (threshold: {self.cpu_warning_threshold}%)",
                severity=AlertSeverity.WARNING,
                source="performance_monitor",
                labels={"metric": "cpu_usage", "value": str(metrics.cpu_usage)}
            )
        
        # Memory threshold checks
        if metrics.memory_usage >= self.memory_critical_threshold:
            await self.create_alert(
                title="Critical Memory Usage",
                description=f"Memory usage is {metrics.memory_usage:.1f}% (threshold: {self.memory_critical_threshold}%)",
                severity=AlertSeverity.CRITICAL,
                source="performance_monitor",
                labels={"metric": "memory_usage", "value": str(metrics.memory_usage)}
            )
        elif metrics.memory_usage >= self.memory_warning_threshold:
            await self.create_alert(
                title="High Memory Usage",
                description=f"Memory usage is {metrics.memory_usage:.1f}% (threshold: {self.memory_warning_threshold}%)",
                severity=AlertSeverity.WARNING,
                source="performance_monitor",
                labels={"metric": "memory_usage", "value": str(metrics.memory_usage)}
            )
        
        # Disk threshold checks
        if metrics.disk_usage >= self.disk_critical_threshold:
            await self.create_alert(
                title="Critical Disk Usage",
                description=f"Disk usage is {metrics.disk_usage:.1f}% (threshold: {self.disk_critical_threshold}%)",
                severity=AlertSeverity.CRITICAL,
                source="performance_monitor",
                labels={"metric": "disk_usage", "value": str(metrics.disk_usage)}
            )
        elif metrics.disk_usage >= self.disk_warning_threshold:
            await self.create_alert(
                title="High Disk Usage",
                description=f"Disk usage is {metrics.disk_usage:.1f}% (threshold: {self.disk_warning_threshold}%)",
                severity=AlertSeverity.WARNING,
                source="performance_monitor",
                labels={"metric": "disk_usage", "value": str(metrics.disk_usage)}
            )
    
    def _calculate_request_rate(self) -> float:
        """Calculate current request rate from metrics."""
        request_metrics = self.get_metrics("requests", timedelta(minutes=5))
        if len(request_metrics) < 2:
            return 0.0
        
        # Calculate rate over last 5 minutes
        total_requests = sum(m.value for m in request_metrics)
        time_window = 300  # 5 minutes in seconds
        
        return total_requests / time_window
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate from metrics."""
        request_metrics = self.get_metrics("requests", timedelta(minutes=5))
        error_metrics = self.get_metrics("errors", timedelta(minutes=5))
        
        total_requests = sum(m.value for m in request_metrics)
        total_errors = sum(m.value for m in error_metrics)
        
        if total_requests == 0:
            return 0.0
        
        return (total_errors / total_requests) * 100
    
    def _calculate_response_time_p95(self) -> float:
        """Calculate 95th percentile response time."""
        response_time_metrics = self.get_metrics("response_time", timedelta(minutes=5))
        
        if not response_time_metrics:
            return 0.0
        
        values = sorted([m.value for m in response_time_metrics])
        if not values:
            return 0.0
        
        # Calculate 95th percentile
        index = int(0.95 * len(values))
        return values[min(index, len(values) - 1)]
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect static system information."""
        try:
            return {
                "hostname": psutil.os.uname().nodename,
                "platform": psutil.os.uname().system,
                "architecture": psutil.os.uname().machine,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total,
                "boot_time": datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc).isoformat()
            }
        except Exception as e:
            self._logger.error("Failed to collect system info", error=str(e))
            return {}
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics, alerts, and traces."""
        current_time = datetime.now(timezone.utc)
        
        # Clean up old metrics
        metrics_cutoff = current_time - timedelta(hours=self.metrics_retention_hours)
        for metric_name in list(self.metrics.keys()):
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name]
                if m.timestamp >= metrics_cutoff
            ]
            
            if not self.metrics[metric_name]:
                del self.metrics[metric_name]
        
        # Clean up old alerts
        alerts_cutoff = current_time - timedelta(hours=self.alert_retention_hours)
        expired_alerts = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.resolved and alert.resolved_at and alert.resolved_at < alerts_cutoff
        ]
        
        for alert_id in expired_alerts:
            del self.alerts[alert_id]
        
        # Clean up old traces
        traces_cutoff = current_time - timedelta(hours=self.trace_retention_hours)
        expired_traces = []
        
        for trace_id, spans in self.traces.items():
            # Keep traces that have recent spans
            recent_spans = [
                span for span in spans
                if span.start_time >= traces_cutoff
            ]
            
            if recent_spans:
                self.traces[trace_id] = recent_spans
            else:
                expired_traces.append(trace_id)
        
        for trace_id in expired_traces:
            del self.traces[trace_id]
        
        # Clean up performance metrics
        perf_cutoff = current_time - timedelta(hours=self.metrics_retention_hours)
        self.performance_metrics = [
            m for m in self.performance_metrics
            if m.timestamp >= perf_cutoff
        ]
        
        if expired_alerts or expired_traces:
            self._logger.info(
                "Cleaned up old data",
                expired_alerts=len(expired_alerts),
                expired_traces=len(expired_traces)
            )