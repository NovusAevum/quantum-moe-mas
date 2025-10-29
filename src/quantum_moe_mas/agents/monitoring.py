"""
Agent Performance Monitoring and Health Checks

This module provides comprehensive monitoring capabilities for agents
including health status tracking, performance metrics collection,
and alerting systems.

Author: Wan Mohamad Hanis bin Wan Hassan
"""

import asyncio
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
from collections import deque, defaultdict

import structlog
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

from quantum_moe_mas.core.logging_simple import get_logger


class HealthStatus(Enum):
    """Agent health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent."""
    agent_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Task metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    
    # Performance metrics
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    
    # Communication metrics
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    
    # Error metrics
    error_count: int = 0
    error_rate: float = 0.0
    last_error_time: Optional[datetime] = None
    
    # Uptime metrics
    uptime_seconds: float = 0.0
    last_activity: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check configuration and result."""
    name: str
    description: str
    check_function: Callable[[], bool]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2
    
    # Runtime state
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None


@dataclass
class Alert:
    """System alert."""
    id: str
    agent_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """Collects and manages metrics for agents."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metric collector."""
        self.registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._logger = get_logger("metric_collector")
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self._task_counter = Counter(
            'agent_tasks_total',
            'Total number of tasks processed by agent',
            ['agent_id', 'status'],
            registry=self.registry
        )
        
        self._response_time_histogram = Histogram(
            'agent_response_time_seconds',
            'Agent response time in seconds',
            ['agent_id'],
            registry=self.registry
        )
        
        self._active_tasks_gauge = Gauge(
            'agent_active_tasks',
            'Number of currently active tasks',
            ['agent_id'],
            registry=self.registry
        )
        
        self._cpu_usage_gauge = Gauge(
            'agent_cpu_usage_percent',
            'Agent CPU usage percentage',
            ['agent_id'],
            registry=self.registry
        )
        
        self._memory_usage_gauge = Gauge(
            'agent_memory_usage_mb',
            'Agent memory usage in MB',
            ['agent_id'],
            registry=self.registry
        )
        
        self._error_counter = Counter(
            'agent_errors_total',
            'Total number of agent errors',
            ['agent_id', 'error_type'],
            registry=self.registry
        )
        
        self._health_status_gauge = Gauge(
            'agent_health_status',
            'Agent health status (0=unknown, 1=healthy, 2=warning, 3=critical)',
            ['agent_id'],
            registry=self.registry
        )
    
    def record_task_completion(
        self,
        agent_id: str,
        success: bool,
        response_time: float
    ) -> None:
        """Record task completion metrics."""
        status = 'success' if success else 'failure'
        self._task_counter.labels(agent_id=agent_id, status=status).inc()
        self._response_time_histogram.labels(agent_id=agent_id).observe(response_time)
    
    def update_active_tasks(self, agent_id: str, count: int) -> None:
        """Update active task count."""
        self._active_tasks_gauge.labels(agent_id=agent_id).set(count)
    
    def update_resource_usage(
        self,
        agent_id: str,
        cpu_percent: float,
        memory_mb: float
    ) -> None:
        """Update resource usage metrics."""
        self._cpu_usage_gauge.labels(agent_id=agent_id).set(cpu_percent)
        self._memory_usage_gauge.labels(agent_id=agent_id).set(memory_mb)
    
    def record_error(self, agent_id: str, error_type: str) -> None:
        """Record an error occurrence."""
        self._error_counter.labels(agent_id=agent_id, error_type=error_type).inc()
    
    def update_health_status(self, agent_id: str, status: HealthStatus) -> None:
        """Update agent health status."""
        status_value = {
            HealthStatus.UNKNOWN: 0,
            HealthStatus.HEALTHY: 1,
            HealthStatus.WARNING: 2,
            HealthStatus.CRITICAL: 3
        }[status]
        
        self._health_status_gauge.labels(agent_id=agent_id).set(status_value)
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        return generate_latest(self.registry).decode('utf-8')


class AgentMonitor:
    """
    Comprehensive monitoring system for agents.
    
    Provides:
    - Health status monitoring
    - Performance metrics collection
    - Resource usage tracking
    - Alert management
    - Automated health checks
    """
    
    def __init__(
        self,
        check_interval: int = 30,
        metrics_retention_hours: int = 24,
        alert_retention_hours: int = 168  # 1 week
    ):
        """
        Initialize agent monitor.
        
        Args:
            check_interval: Health check interval in seconds
            metrics_retention_hours: How long to retain metrics
            alert_retention_hours: How long to retain alerts
        """
        self.check_interval = check_interval
        self.metrics_retention = timedelta(hours=metrics_retention_hours)
        self.alert_retention = timedelta(hours=alert_retention_hours)
        
        # Monitoring data
        self._agents: Dict[str, Any] = {}  # agent_id -> agent instance
        self._health_checks: Dict[str, List[HealthCheck]] = defaultdict(list)
        self._metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._alerts: Dict[str, Alert] = {}
        self._alert_handlers: List[Callable[[Alert], None]] = []
        
        # Metric collector
        self._metric_collector = MetricCollector()
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Logger
        self._logger = get_logger("agent_monitor")
        
        # System resource monitoring
        self._system_metrics = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024 * 1024),  # MB
        }
    
    async def start(self) -> None:
        """Start the monitoring system."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self._logger.info("Agent monitor started")
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        self._running = False
        
        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._logger.info("Agent monitor stopped")
    
    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """Register an agent for monitoring."""
        self._agents[agent_id] = agent_instance
        
        # Add default health checks
        self._add_default_health_checks(agent_id)
        
        self._logger.info("Agent registered for monitoring", agent_id=agent_id)
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from monitoring."""
        if agent_id in self._agents:
            del self._agents[agent_id]
        
        if agent_id in self._health_checks:
            del self._health_checks[agent_id]
        
        if agent_id in self._metrics_history:
            del self._metrics_history[agent_id]
        
        self._logger.info("Agent unregistered from monitoring", agent_id=agent_id)
    
    def add_health_check(self, agent_id: str, health_check: HealthCheck) -> None:
        """Add a health check for an agent."""
        self._health_checks[agent_id].append(health_check)
        
        self._logger.info(
            "Health check added",
            agent_id=agent_id,
            check_name=health_check.name
        )
    
    def _add_default_health_checks(self, agent_id: str) -> None:
        """Add default health checks for an agent."""
        # Basic connectivity check
        connectivity_check = HealthCheck(
            name="connectivity",
            description="Check if agent is responsive",
            check_function=lambda: self._check_agent_connectivity(agent_id),
            interval_seconds=30
        )
        
        # Resource usage check
        resource_check = HealthCheck(
            name="resource_usage",
            description="Check agent resource usage",
            check_function=lambda: self._check_resource_usage(agent_id),
            interval_seconds=60
        )
        
        # Error rate check
        error_rate_check = HealthCheck(
            name="error_rate",
            description="Check agent error rate",
            check_function=lambda: self._check_error_rate(agent_id),
            interval_seconds=120
        )
        
        self._health_checks[agent_id].extend([
            connectivity_check,
            resource_check,
            error_rate_check
        ])
    
    def _check_agent_connectivity(self, agent_id: str) -> bool:
        """Check if agent is responsive."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        
        try:
            # Check if agent is in a healthy state
            if hasattr(agent, 'state'):
                from quantum_moe_mas.agents.base_agent import AgentState
                return agent.state not in [AgentState.ERROR, AgentState.SHUTDOWN]
            return True
        except Exception:
            return False
    
    def _check_resource_usage(self, agent_id: str) -> bool:
        """Check agent resource usage."""
        try:
            # Get current process info
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Update metrics
            self._metric_collector.update_resource_usage(
                agent_id, cpu_percent, memory_mb
            )
            
            # Check thresholds
            cpu_threshold = 80.0  # 80% CPU
            memory_threshold = 1024.0  # 1GB memory
            
            return cpu_percent < cpu_threshold and memory_mb < memory_threshold
            
        except Exception as e:
            self._logger.error(
                "Resource usage check failed",
                agent_id=agent_id,
                error=str(e)
            )
            return False
    
    def _check_error_rate(self, agent_id: str) -> bool:
        """Check agent error rate."""
        try:
            agent = self._agents.get(agent_id)
            if not agent or not hasattr(agent, 'metrics'):
                return True
            
            metrics = agent.metrics
            if metrics.total_tasks == 0:
                return True
            
            error_rate = metrics.failed_tasks / metrics.total_tasks
            error_threshold = 0.1  # 10% error rate
            
            return error_rate < error_threshold
            
        except Exception as e:
            self._logger.error(
                "Error rate check failed",
                agent_id=agent_id,
                error=str(e)
            )
            return False
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await self._collect_metrics()
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self._logger.error(
                    "Error in monitoring loop",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks for all agents."""
        for agent_id, health_checks in self._health_checks.items():
            for health_check in health_checks:
                await self._run_health_check(agent_id, health_check)
    
    async def _run_health_check(self, agent_id: str, health_check: HealthCheck) -> None:
        """Run a single health check."""
        current_time = datetime.now(timezone.utc)
        
        # Check if it's time to run this check
        if (health_check.last_check and 
            (current_time - health_check.last_check).total_seconds() < health_check.interval_seconds):
            return
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, health_check.check_function
                ),
                timeout=health_check.timeout_seconds
            )
            
            health_check.last_check = current_time
            
            if result:
                # Check passed
                health_check.consecutive_failures = 0
                health_check.consecutive_successes += 1
                health_check.last_error = None
                
                # Update status if recovered
                if (health_check.status != HealthStatus.HEALTHY and
                    health_check.consecutive_successes >= health_check.recovery_threshold):
                    
                    old_status = health_check.status
                    health_check.status = HealthStatus.HEALTHY
                    
                    self._logger.info(
                        "Health check recovered",
                        agent_id=agent_id,
                        check_name=health_check.name,
                        old_status=old_status.value,
                        new_status=health_check.status.value
                    )
                    
                    # Create recovery alert
                    await self._create_alert(
                        agent_id=agent_id,
                        severity=AlertSeverity.INFO,
                        title=f"Health check recovered: {health_check.name}",
                        description=f"Health check '{health_check.name}' has recovered",
                        metadata={
                            "check_name": health_check.name,
                            "old_status": old_status.value,
                            "new_status": health_check.status.value
                        }
                    )
            else:
                # Check failed
                health_check.consecutive_successes = 0
                health_check.consecutive_failures += 1
                
                # Update status if threshold reached
                if health_check.consecutive_failures >= health_check.failure_threshold:
                    old_status = health_check.status
                    
                    if health_check.consecutive_failures >= health_check.failure_threshold * 2:
                        health_check.status = HealthStatus.CRITICAL
                    else:
                        health_check.status = HealthStatus.WARNING
                    
                    if old_status != health_check.status:
                        self._logger.warning(
                            "Health check status changed",
                            agent_id=agent_id,
                            check_name=health_check.name,
                            old_status=old_status.value,
                            new_status=health_check.status.value,
                            consecutive_failures=health_check.consecutive_failures
                        )
                        
                        # Create alert
                        severity = (AlertSeverity.CRITICAL 
                                  if health_check.status == HealthStatus.CRITICAL 
                                  else AlertSeverity.WARNING)
                        
                        await self._create_alert(
                            agent_id=agent_id,
                            severity=severity,
                            title=f"Health check failed: {health_check.name}",
                            description=f"Health check '{health_check.name}' has failed {health_check.consecutive_failures} times",
                            metadata={
                                "check_name": health_check.name,
                                "consecutive_failures": health_check.consecutive_failures,
                                "status": health_check.status.value
                            }
                        )
        
        except asyncio.TimeoutError:
            health_check.last_error = "Health check timeout"
            health_check.consecutive_failures += 1
            
            self._logger.warning(
                "Health check timeout",
                agent_id=agent_id,
                check_name=health_check.name,
                timeout=health_check.timeout_seconds
            )
        
        except Exception as e:
            health_check.last_error = str(e)
            health_check.consecutive_failures += 1
            
            self._logger.error(
                "Health check error",
                agent_id=agent_id,
                check_name=health_check.name,
                error=str(e)
            )
        
        # Update Prometheus metric
        self._metric_collector.update_health_status(agent_id, health_check.status)
    
    async def _collect_metrics(self) -> None:
        """Collect metrics for all agents."""
        for agent_id, agent in self._agents.items():
            try:
                metrics = await self._collect_agent_metrics(agent_id, agent)
                self._metrics_history[agent_id].append(metrics)
                
            except Exception as e:
                self._logger.error(
                    "Failed to collect metrics",
                    agent_id=agent_id,
                    error=str(e)
                )
    
    async def _collect_agent_metrics(self, agent_id: str, agent: Any) -> PerformanceMetrics:
        """Collect metrics for a single agent."""
        metrics = PerformanceMetrics(agent_id=agent_id)
        
        # Get agent metrics if available
        if hasattr(agent, 'metrics'):
            agent_metrics = agent.metrics
            metrics.total_tasks = agent_metrics.total_tasks
            metrics.successful_tasks = agent_metrics.successful_tasks
            metrics.failed_tasks = agent_metrics.failed_tasks
            metrics.average_response_time = agent_metrics.average_response_time
            metrics.uptime_seconds = agent_metrics.uptime_seconds
            metrics.last_activity = agent_metrics.last_activity
        
        # Get system resource usage
        try:
            process = psutil.Process()
            metrics.cpu_usage_percent = process.cpu_percent()
            memory_info = process.memory_info()
            metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
            metrics.memory_usage_percent = (
                metrics.memory_usage_mb / self._system_metrics["memory_total"] * 100
            )
        except Exception as e:
            self._logger.debug(
                "Failed to get system metrics",
                agent_id=agent_id,
                error=str(e)
            )
        
        return metrics
    
    async def _create_alert(
        self,
        agent_id: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new alert."""
        alert_id = f"{agent_id}_{int(time.time())}_{len(self._alerts)}"
        
        alert = Alert(
            id=alert_id,
            agent_id=agent_id,
            severity=severity,
            title=title,
            description=description,
            metadata=metadata or {}
        )
        
        self._alerts[alert_id] = alert
        
        # Notify alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self._logger.error(
                    "Alert handler failed",
                    alert_id=alert_id,
                    error=str(e)
                )
        
        self._logger.info(
            "Alert created",
            alert_id=alert_id,
            agent_id=agent_id,
            severity=severity.value,
            title=title
        )
        
        return alert_id
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler."""
        self._alert_handlers.append(handler)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old metrics and alerts."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
                
            except Exception as e:
                self._logger.error(
                    "Error in cleanup loop",
                    error=str(e)
                )
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and alerts."""
        current_time = datetime.now(timezone.utc)
        
        # Cleanup old alerts
        expired_alerts = []
        for alert_id, alert in self._alerts.items():
            if (current_time - alert.timestamp) > self.alert_retention:
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            del self._alerts[alert_id]
        
        if expired_alerts:
            self._logger.info(
                "Cleaned up expired alerts",
                count=len(expired_alerts)
            )
    
    def get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Get health status for an agent."""
        if agent_id not in self._agents:
            return {"error": "Agent not found"}
        
        health_checks = self._health_checks.get(agent_id, [])
        overall_status = HealthStatus.HEALTHY
        
        # Determine overall status
        for check in health_checks:
            if check.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif check.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        return {
            "agent_id": agent_id,
            "overall_status": overall_status.value,
            "health_checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "last_check": check.last_check.isoformat() if check.last_check else None,
                    "consecutive_failures": check.consecutive_failures,
                    "last_error": check.last_error
                }
                for check in health_checks
            ]
        }
    
    def get_agent_metrics(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics for an agent."""
        metrics_history = self._metrics_history.get(agent_id, deque())
        
        # Get the most recent metrics
        recent_metrics = list(metrics_history)[-limit:]
        
        return [
            {
                "timestamp": metrics.timestamp.isoformat(),
                "total_tasks": metrics.total_tasks,
                "successful_tasks": metrics.successful_tasks,
                "failed_tasks": metrics.failed_tasks,
                "average_response_time": metrics.average_response_time,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "memory_usage_mb": metrics.memory_usage_mb,
                "uptime_seconds": metrics.uptime_seconds
            }
            for metrics in recent_metrics
        ]
    
    def get_alerts(
        self,
        agent_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        alerts = []
        
        for alert in self._alerts.values():
            # Apply filters
            if agent_id and alert.agent_id != agent_id:
                continue
            if severity and alert.severity != severity:
                continue
            if resolved is not None and alert.resolved != resolved:
                continue
            
            alerts.append({
                "id": alert.id,
                "agent_id": alert.agent_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "metadata": alert.metadata
            })
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self._alerts:
            return False
        
        alert = self._alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        
        self._logger.info(
            "Alert resolved",
            alert_id=alert_id,
            agent_id=alert.agent_id
        )
        
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        total_agents = len(self._agents)
        total_alerts = len(self._alerts)
        unresolved_alerts = sum(1 for alert in self._alerts.values() if not alert.resolved)
        
        # Health status distribution
        health_distribution = defaultdict(int)
        for agent_id in self._agents.keys():
            health_info = self.get_agent_health(agent_id)
            status = health_info.get("overall_status", "unknown")
            health_distribution[status] += 1
        
        return {
            "total_agents": total_agents,
            "total_alerts": total_alerts,
            "unresolved_alerts": unresolved_alerts,
            "health_distribution": dict(health_distribution),
            "system_metrics": self._system_metrics,
            "monitoring_uptime": time.time() - (time.time() if self._running else 0)
        }