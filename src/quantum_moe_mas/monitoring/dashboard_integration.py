"""
Dashboard Integration for Performance Monitoring

Provides Grafana dashboard integration and real-time performance data
for the Quantum MoE MAS monitoring system.

Requirements: 8.1, 8.2
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiohttp

import structlog

from .metrics_collector import MetricsCollector, PerformanceMetrics
from .latency_tracker import LatencyTracker, LatencyCategory
from .resource_monitor import ResourceMonitor, ResourceType
from .bottleneck_analyzer import BottleneckAnalyzer, AnalysisResult

logger = structlog.get_logger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard integration."""
    
    grafana_url: str = "http://localhost:3000"
    grafana_api_key: Optional[str] = None
    dashboard_refresh_interval: int = 30  # seconds
    alert_webhook_url: Optional[str] = None
    enable_real_time_updates: bool = True


class DashboardIntegration:
    """
    Grafana dashboard integration for real-time performance monitoring.
    
    Provides dashboard provisioning, real-time data updates, and
    alert integration for the Quantum MoE MAS monitoring system.
    """
    
    def __init__(self,
                 metrics_collector: MetricsCollector,
                 latency_tracker: LatencyTracker,
                 resource_monitor: ResourceMonitor,
                 bottleneck_analyzer: BottleneckAnalyzer,
                 config: Optional[DashboardConfig] = None):
        """Initialize dashboard integration."""
        
        self.metrics_collector = metrics_collector
        self.latency_tracker = latency_tracker
        self.resource_monitor = resource_monitor
        self.bottleneck_analyzer = bottleneck_analyzer
        self.config = config or DashboardConfig()
        
        # Dashboard state
        self._dashboard_data_cache: Dict[str, Any] = {}
        self._last_update = datetime.now()
        self._update_task: Optional[asyncio.Task] = None
        self._is_updating = False
        
        logger.info("DashboardIntegration initialized",
                   grafana_url=self.config.grafana_url,
                   refresh_interval=self.config.dashboard_refresh_interval)
    
    async def start_real_time_updates(self) -> None:
        """Start real-time dashboard data updates."""
        
        if not self.config.enable_real_time_updates:
            logger.info("Real-time updates disabled")
            return
        
        if self._is_updating:
            logger.warning("Real-time updates already running")
            return
        
        self._is_updating = True
        self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info("Real-time dashboard updates started")
    
    async def stop_real_time_updates(self) -> None:
        """Stop real-time dashboard data updates."""
        
        if not self._is_updating:
            logger.warning("Real-time updates not running")
            return
        
        self._is_updating = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time dashboard updates stopped")
    
    async def _update_loop(self) -> None:
        """Main update loop for dashboard data."""
        
        while self._is_updating:
            try:
                await self.update_dashboard_data()
                await asyncio.sleep(self.config.dashboard_refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in dashboard update loop", error=str(e))
                await asyncio.sleep(self.config.dashboard_refresh_interval)
    
    async def update_dashboard_data(self) -> Dict[str, Any]:
        """Update all dashboard data and cache results."""
        
        logger.debug("Updating dashboard data")
        
        # Collect all monitoring data
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_overview': await self._get_system_overview(),
            'latency_metrics': await self._get_latency_metrics(),
            'resource_metrics': await self._get_resource_metrics(),
            'bottleneck_analysis': await self._get_bottleneck_analysis(),
            'performance_trends': await self._get_performance_trends(),
            'alerts': await self._get_active_alerts()
        }
        
        # Cache the data
        self._dashboard_data_cache = dashboard_data
        self._last_update = datetime.now()
        
        logger.debug("Dashboard data updated successfully")
        
        return dashboard_data
    
    async def _get_system_overview(self) -> Dict[str, Any]:
        """Get high-level system overview metrics."""
        
        # Get comprehensive metrics
        performance_metrics = await self.metrics_collector.collect_comprehensive_metrics()
        sla_compliance = self.latency_tracker.get_sla_compliance_summary()
        system_health = self.resource_monitor.get_system_health_summary()
        
        return {
            'sla_compliance_rate': sla_compliance['overall_compliance_rate'],
            'avg_latency_p95': performance_metrics.latency_p95,
            'throughput_qps': performance_metrics.throughput_qps,
            'error_rate': performance_metrics.error_rate,
            'system_health_status': system_health['overall_status'],
            'active_alerts': len(system_health.get('critical_count', 0) + system_health.get('warning_count', 0)),
            'meets_sla_target': performance_metrics.meets_sla_targets()
        }
    
    async def _get_latency_metrics(self) -> Dict[str, Any]:
        """Get detailed latency metrics for dashboard."""
        
        latency_data = {}
        
        # Get stats for each category
        for category in LatencyCategory:
            stats = self.latency_tracker.get_latency_stats(category)
            if stats:
                latency_data[category.value] = {
                    'p50_ms': stats.median_ms,
                    'p95_ms': stats.p95_ms,
                    'p99_ms': stats.p99_ms,
                    'request_count': stats.count,
                    'sla_compliance_rate': stats.sla_compliance_rate,
                    'meets_sla': stats.meets_sla_target(5000)
                }
        
        # Get operation breakdown for key categories
        operation_breakdown = {}
        for category in [LatencyCategory.ROUTING, LatencyCategory.EXPERT_INFERENCE]:
            breakdown = self.latency_tracker.get_operation_breakdown(category)
            if breakdown:
                operation_breakdown[category.value] = {
                    op: {
                        'p95_ms': stats.p95_ms,
                        'count': stats.count,
                        'sla_compliance': stats.sla_compliance_rate
                    }
                    for op, stats in breakdown.items()
                }
        
        return {
            'categories': latency_data,
            'operations': operation_breakdown,
            'sla_summary': self.latency_tracker.get_sla_compliance_summary(),
            'performance_summary': self.latency_tracker.get_performance_summary()
        }
    
    async def _get_resource_metrics(self) -> Dict[str, Any]:
        """Get detailed resource utilization metrics."""
        
        resource_data = {}
        
        # Get stats for each resource type
        for resource_type in ResourceType:
            stats = self.resource_monitor.get_resource_stats(resource_type)
            if stats:
                resource_data[resource_type.value] = {
                    'current_percent': stats.current_value,
                    'mean_percent': stats.mean_value,
                    'max_percent': stats.max_value,
                    'trend': stats.trend,
                    'alert_level': stats.alert_level.value
                }
        
        return {
            'resources': resource_data,
            'system_health': self.resource_monitor.get_system_health_summary()
        }
    
    async def _get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Get bottleneck analysis results."""
        
        try:
            analysis_result = await self.bottleneck_analyzer.analyze_system_performance()
            return analysis_result.to_dict()
        except Exception as e:
            logger.error("Error getting bottleneck analysis", error=str(e))
            return {
                'bottlenecks': [],
                'system_health_score': 0.5,
                'primary_bottleneck': None,
                'optimization_priority': [],
                'error': str(e)
            }
    
    async def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trend data for charts."""
        
        # This would typically query historical data from a time-series database
        # For now, we'll provide current snapshot data
        
        trends = {
            'latency_trend': {
                'timestamps': [],
                'p95_values': [],
                'p99_values': []
            },
            'throughput_trend': {
                'timestamps': [],
                'qps_values': []
            },
            'resource_trends': {
                resource_type.value: {
                    'timestamps': [],
                    'values': []
                }
                for resource_type in ResourceType
            }
        }
        
        # In a real implementation, you would populate these with historical data
        # from your metrics storage system (e.g., Prometheus, InfluxDB)
        
        return trends
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        
        alerts = []
        
        # Get SLA violations
        sla_summary = self.latency_tracker.get_sla_compliance_summary()
        for category, data in sla_summary.get('categories', {}).items():
            if not data.get('meets_sla', True):
                alerts.append({
                    'type': 'sla_violation',
                    'severity': 'high',
                    'category': category,
                    'message': f"SLA violation in {category}: P95 = {data['p95_latency_ms']:.0f}ms",
                    'timestamp': datetime.now().isoformat()
                })
        
        # Get resource alerts
        system_health = self.resource_monitor.get_system_health_summary()
        for resource, data in system_health.get('resources', {}).items():
            if data['alert_level'] in ['warning', 'critical']:
                alerts.append({
                    'type': 'resource_threshold',
                    'severity': data['alert_level'],
                    'resource': resource,
                    'message': f"{resource.upper()} usage at {data['current_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def get_cached_dashboard_data(self) -> Dict[str, Any]:
        """Get cached dashboard data."""
        
        return {
            **self._dashboard_data_cache,
            'cache_timestamp': self._last_update.isoformat(),
            'cache_age_seconds': (datetime.now() - self._last_update).total_seconds()
        }
    
    async def create_grafana_dashboard(self) -> Optional[Dict[str, Any]]:
        """Create Grafana dashboard configuration."""
        
        if not self.config.grafana_api_key:
            logger.warning("Grafana API key not configured, skipping dashboard creation")
            return None
        
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Quantum MoE MAS Performance Dashboard",
                "tags": ["quantum", "moe", "performance"],
                "timezone": "browser",
                "refresh": f"{self.config.dashboard_refresh_interval}s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    self._create_system_overview_panel(),
                    self._create_latency_panel(),
                    self._create_resource_panel(),
                    self._create_bottleneck_panel(),
                    self._create_sla_compliance_panel()
                ]
            },
            "overwrite": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.config.grafana_api_key}',
                    'Content-Type': 'application/json'
                }
                
                async with session.post(
                    f"{self.config.grafana_url}/api/dashboards/db",
                    headers=headers,
                    json=dashboard_config
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Grafana dashboard created successfully",
                                   dashboard_id=result.get('id'),
                                   url=result.get('url'))
                        return result
                    else:
                        error_text = await response.text()
                        logger.error("Failed to create Grafana dashboard",
                                   status=response.status,
                                   error=error_text)
                        return None
                        
        except Exception as e:
            logger.error("Error creating Grafana dashboard", error=str(e))
            return None
    
    def _create_system_overview_panel(self) -> Dict[str, Any]:
        """Create system overview panel configuration."""
        
        return {
            "id": 1,
            "title": "System Overview",
            "type": "stat",
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
            "targets": [
                {
                    "expr": "quantum_moe_request_latency_seconds{quantile=\"0.95\"}",
                    "legendFormat": "P95 Latency"
                },
                {
                    "expr": "rate(quantum_moe_requests_total[5m])",
                    "legendFormat": "Throughput (QPS)"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "s",
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 2},
                            {"color": "red", "value": 5}
                        ]
                    }
                }
            }
        }
    
    def _create_latency_panel(self) -> Dict[str, Any]:
        """Create latency monitoring panel configuration."""
        
        return {
            "id": 2,
            "title": "Latency by Category",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "targets": [
                {
                    "expr": "quantum_moe_request_duration_seconds_bucket",
                    "legendFormat": "{{le}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "s",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear"
                    }
                }
            }
        }
    
    def _create_resource_panel(self) -> Dict[str, Any]:
        """Create resource utilization panel configuration."""
        
        return {
            "id": 3,
            "title": "Resource Utilization",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            "targets": [
                {
                    "expr": "quantum_moe_cpu_usage_percent",
                    "legendFormat": "CPU %"
                },
                {
                    "expr": "quantum_moe_memory_usage_bytes / 1024 / 1024 / 1024",
                    "legendFormat": "Memory GB"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 70},
                            {"color": "red", "value": 90}
                        ]
                    }
                }
            }
        }
    
    def _create_bottleneck_panel(self) -> Dict[str, Any]:
        """Create bottleneck analysis panel configuration."""
        
        return {
            "id": 4,
            "title": "Bottleneck Analysis",
            "type": "table",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
            "targets": [
                {
                    "expr": "quantum_moe_routing_accuracy",
                    "legendFormat": "Routing Accuracy"
                }
            ]
        }
    
    def _create_sla_compliance_panel(self) -> Dict[str, Any]:
        """Create SLA compliance panel configuration."""
        
        return {
            "id": 5,
            "title": "SLA Compliance",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
            "targets": [
                {
                    "expr": "rate(quantum_moe_requests_total{status=\"success\"}[5m]) / rate(quantum_moe_requests_total[5m])",
                    "legendFormat": "Success Rate"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 0.95},
                            {"color": "green", "value": 0.99}
                        ]
                    }
                }
            }
        }
    
    async def send_alert_webhook(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert to configured webhook."""
        
        if not self.config.alert_webhook_url:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.alert_webhook_url,
                    json=alert_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    
                    if response.status == 200:
                        logger.info("Alert webhook sent successfully")
                        return True
                    else:
                        logger.error("Alert webhook failed",
                                   status=response.status)
                        return False
                        
        except Exception as e:
            logger.error("Error sending alert webhook", error=str(e))
            return False