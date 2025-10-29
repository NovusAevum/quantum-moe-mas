"""
Kubernetes Horizontal Pod Autoscaler (HPA) Integration

Implements dynamic resource scaling based on demand using Kubernetes HPA
with custom metrics and intelligent scaling policies.

Requirements: 8.1, 8.3
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

import structlog
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = structlog.get_logger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Triggers that can initiate scaling operations."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Performance metrics
    request_rate: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    
    # Queue and load metrics
    queue_length: int = 0
    active_connections: int = 0
    
    # Custom business metrics
    api_calls_per_minute: float = 0.0
    expert_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    pod_count: int = 1
    target_pod_count: int = 1
    
    def should_scale_up(self, thresholds: Dict[str, float]) -> bool:
        """Determine if scaling up is needed based on thresholds."""
        return (
            self.cpu_utilization > thresholds.get('cpu_scale_up', 70.0) or
            self.memory_utilization > thresholds.get('memory_scale_up', 80.0) or
            self.avg_response_time > thresholds.get('response_time_scale_up', 2.0) or
            self.request_rate > thresholds.get('request_rate_scale_up', 100.0)
        )
    
    def should_scale_down(self, thresholds: Dict[str, float]) -> bool:
        """Determine if scaling down is needed based on thresholds."""
        return (
            self.cpu_utilization < thresholds.get('cpu_scale_down', 30.0) and
            self.memory_utilization < thresholds.get('memory_scale_down', 40.0) and
            self.avg_response_time < thresholds.get('response_time_scale_down', 1.0) and
            self.request_rate < thresholds.get('request_rate_scale_down', 20.0)
        )


@dataclass
class ScalingPolicy:
    """Configuration for scaling behavior."""
    
    # Scaling thresholds
    cpu_target_utilization: float = 70.0
    memory_target_utilization: float = 80.0
    response_time_threshold: float = 2.0
    
    # Scaling limits
    min_replicas: int = 2
    max_replicas: int = 20
    
    # Scaling behavior
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 600  # 10 minutes
    scale_up_increment: int = 2
    scale_down_increment: int = 1
    
    # Advanced settings
    enable_predictive_scaling: bool = True
    enable_custom_metrics: bool = True
    stabilization_window: int = 300  # 5 minutes


class KubernetesScaler:
    """
    Kubernetes Horizontal Pod Autoscaler integration.
    
    Provides dynamic resource scaling based on multiple metrics including
    CPU, memory, response time, and custom business metrics.
    """
    
    def __init__(self, 
                 namespace: str = "default",
                 deployment_name: str = "quantum-moe-mas",
                 policy: Optional[ScalingPolicy] = None):
        """Initialize the Kubernetes scaler."""
        
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.policy = policy or ScalingPolicy()
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()  # For in-cluster deployment
        except config.ConfigException:
            try:
                config.load_kube_config()  # For local development
            except config.ConfigException:
                logger.warning("Could not load Kubernetes config, using mock mode")
                self._mock_mode = True
            else:
                self._mock_mode = False
        else:
            self._mock_mode = False
        
        if not self._mock_mode:
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.autoscaling_v2 = client.AutoscalingV2Api()
        
        # Scaling state
        self._last_scale_time = {}
        self._scaling_history = []
        self._current_replicas = self.policy.min_replicas
        
        logger.info("KubernetesScaler initialized",
                   namespace=namespace,
                   deployment=deployment_name,
                   mock_mode=self._mock_mode)
    
    async def create_hpa(self) -> bool:
        """Create Horizontal Pod Autoscaler for the deployment."""
        
        if self._mock_mode:
            logger.info("Mock mode: HPA creation simulated")
            return True
        
        hpa_spec = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.deployment_name}-hpa",
                "namespace": self.namespace,
                "labels": {
                    "app": self.deployment_name,
                    "component": "autoscaler"
                }
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.deployment_name
                },
                "minReplicas": self.policy.min_replicas,
                "maxReplicas": self.policy.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": int(self.policy.cpu_target_utilization)
                            }
                        }
                    },
                    {
                        "type": "Resource", 
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": int(self.policy.memory_target_utilization)
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": self.policy.stabilization_window,
                        "policies": [
                            {
                                "type": "Pods",
                                "value": self.policy.scale_up_increment,
                                "periodSeconds": 60
                            }
                        ]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": self.policy.stabilization_window * 2,
                        "policies": [
                            {
                                "type": "Pods", 
                                "value": self.policy.scale_down_increment,
                                "periodSeconds": 60
                            }
                        ]
                    }
                }
            }
        }
        
        try:
            # Check if HPA already exists
            try:
                existing_hpa = self.autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                    name=f"{self.deployment_name}-hpa",
                    namespace=self.namespace
                )
                logger.info("HPA already exists, updating configuration")
                
                # Update existing HPA
                self.autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                    name=f"{self.deployment_name}-hpa",
                    namespace=self.namespace,
                    body=hpa_spec
                )
                
            except ApiException as e:
                if e.status == 404:
                    # Create new HPA
                    self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                        namespace=self.namespace,
                        body=hpa_spec
                    )
                    logger.info("HPA created successfully")
                else:
                    raise
            
            return True
            
        except ApiException as e:
            logger.error("Failed to create/update HPA",
                        error=str(e),
                        status=e.status)
            return False
    
    async def get_current_metrics(self) -> ScalingMetrics:
        """Get current scaling metrics from Kubernetes and custom sources."""
        
        if self._mock_mode:
            # Return mock metrics for testing
            return ScalingMetrics(
                cpu_utilization=45.0,
                memory_utilization=60.0,
                request_rate=50.0,
                avg_response_time=1.2,
                pod_count=3,
                target_pod_count=3
            )
        
        try:
            # Get deployment info
            deployment = self.apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            
            current_replicas = deployment.status.replicas or 0
            ready_replicas = deployment.status.ready_replicas or 0
            
            # Get HPA status
            try:
                hpa = self.autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                    name=f"{self.deployment_name}-hpa",
                    namespace=self.namespace
                )
                
                target_replicas = hpa.status.desired_replicas or current_replicas
                
                # Extract current metrics from HPA status
                cpu_utilization = 0.0
                memory_utilization = 0.0
                
                if hpa.status.current_metrics:
                    for metric in hpa.status.current_metrics:
                        if metric.resource and metric.resource.name == "cpu":
                            cpu_utilization = float(metric.resource.current.average_utilization or 0)
                        elif metric.resource and metric.resource.name == "memory":
                            memory_utilization = float(metric.resource.current.average_utilization or 0)
                
            except ApiException:
                target_replicas = current_replicas
                cpu_utilization = 0.0
                memory_utilization = 0.0
            
            # TODO: Integrate with monitoring system for additional metrics
            # For now, using placeholder values
            
            metrics = ScalingMetrics(
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization,
                request_rate=0.0,  # Will be populated by monitoring integration
                avg_response_time=0.0,  # Will be populated by monitoring integration
                pod_count=ready_replicas,
                target_pod_count=target_replicas,
                timestamp=datetime.now()
            )
            
            logger.debug("Current scaling metrics collected",
                        cpu=cpu_utilization,
                        memory=memory_utilization,
                        pods=ready_replicas,
                        target=target_replicas)
            
            return metrics
            
        except ApiException as e:
            logger.error("Failed to get current metrics",
                        error=str(e),
                        status=e.status)
            
            # Return default metrics on error
            return ScalingMetrics()
    
    async def manual_scale(self, target_replicas: int, reason: str = "manual") -> bool:
        """Manually scale the deployment to target replica count."""
        
        if target_replicas < self.policy.min_replicas:
            target_replicas = self.policy.min_replicas
        elif target_replicas > self.policy.max_replicas:
            target_replicas = self.policy.max_replicas
        
        if self._mock_mode:
            logger.info("Mock mode: Manual scaling simulated",
                       target_replicas=target_replicas,
                       reason=reason)
            self._current_replicas = target_replicas
            return True
        
        try:
            # Update deployment replica count
            deployment = self.apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            
            deployment.spec.replicas = target_replicas
            
            self.apps_v1.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            # Record scaling event
            self._record_scaling_event(
                from_replicas=deployment.status.replicas or 0,
                to_replicas=target_replicas,
                trigger="manual",
                reason=reason
            )
            
            logger.info("Manual scaling completed",
                       target_replicas=target_replicas,
                       reason=reason)
            
            return True
            
        except ApiException as e:
            logger.error("Failed to manually scale deployment",
                        error=str(e),
                        status=e.status,
                        target_replicas=target_replicas)
            return False
    
    async def evaluate_scaling_decision(self, metrics: ScalingMetrics) -> Dict[str, Any]:
        """Evaluate whether scaling is needed based on current metrics."""
        
        thresholds = {
            'cpu_scale_up': self.policy.cpu_target_utilization + 10,
            'cpu_scale_down': self.policy.cpu_target_utilization - 20,
            'memory_scale_up': self.policy.memory_target_utilization + 10,
            'memory_scale_down': self.policy.memory_target_utilization - 20,
            'response_time_scale_up': self.policy.response_time_threshold,
            'response_time_scale_down': self.policy.response_time_threshold * 0.5,
            'request_rate_scale_up': 100.0,
            'request_rate_scale_down': 20.0
        }
        
        # Check cooldown periods
        now = datetime.now()
        last_scale_up = self._last_scale_time.get('up', datetime.min)
        last_scale_down = self._last_scale_time.get('down', datetime.min)
        
        scale_up_ready = (now - last_scale_up).total_seconds() > self.policy.scale_up_cooldown
        scale_down_ready = (now - last_scale_down).total_seconds() > self.policy.scale_down_cooldown
        
        # Evaluate scaling need
        should_scale_up = metrics.should_scale_up(thresholds) and scale_up_ready
        should_scale_down = metrics.should_scale_down(thresholds) and scale_down_ready
        
        # Determine scaling direction and magnitude
        if should_scale_up and metrics.pod_count < self.policy.max_replicas:
            direction = ScalingDirection.UP
            target_replicas = min(
                metrics.pod_count + self.policy.scale_up_increment,
                self.policy.max_replicas
            )
            confidence = self._calculate_scaling_confidence(metrics, direction)
            
        elif should_scale_down and metrics.pod_count > self.policy.min_replicas:
            direction = ScalingDirection.DOWN
            target_replicas = max(
                metrics.pod_count - self.policy.scale_down_increment,
                self.policy.min_replicas
            )
            confidence = self._calculate_scaling_confidence(metrics, direction)
            
        else:
            direction = ScalingDirection.STABLE
            target_replicas = metrics.pod_count
            confidence = 1.0
        
        decision = {
            'direction': direction,
            'current_replicas': metrics.pod_count,
            'target_replicas': target_replicas,
            'confidence': confidence,
            'triggers': self._identify_triggers(metrics, thresholds),
            'cooldown_ready': {
                'scale_up': scale_up_ready,
                'scale_down': scale_down_ready
            },
            'timestamp': now
        }
        
        logger.debug("Scaling decision evaluated",
                    direction=direction.value,
                    current=metrics.pod_count,
                    target=target_replicas,
                    confidence=confidence)
        
        return decision
    
    def _calculate_scaling_confidence(self, 
                                    metrics: ScalingMetrics, 
                                    direction: ScalingDirection) -> float:
        """Calculate confidence score for scaling decision."""
        
        confidence_factors = []
        
        # CPU utilization confidence
        if direction == ScalingDirection.UP:
            cpu_confidence = min(metrics.cpu_utilization / 100.0, 1.0)
        else:
            cpu_confidence = max(1.0 - (metrics.cpu_utilization / 100.0), 0.0)
        confidence_factors.append(cpu_confidence)
        
        # Memory utilization confidence
        if direction == ScalingDirection.UP:
            mem_confidence = min(metrics.memory_utilization / 100.0, 1.0)
        else:
            mem_confidence = max(1.0 - (metrics.memory_utilization / 100.0), 0.0)
        confidence_factors.append(mem_confidence)
        
        # Response time confidence
        if metrics.avg_response_time > 0:
            if direction == ScalingDirection.UP:
                rt_confidence = min(metrics.avg_response_time / self.policy.response_time_threshold, 1.0)
            else:
                rt_confidence = max(1.0 - (metrics.avg_response_time / self.policy.response_time_threshold), 0.0)
            confidence_factors.append(rt_confidence)
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _identify_triggers(self, 
                          metrics: ScalingMetrics, 
                          thresholds: Dict[str, float]) -> List[str]:
        """Identify which metrics triggered the scaling decision."""
        
        triggers = []
        
        if metrics.cpu_utilization > thresholds.get('cpu_scale_up', 70):
            triggers.append(f"CPU high ({metrics.cpu_utilization:.1f}%)")
        elif metrics.cpu_utilization < thresholds.get('cpu_scale_down', 30):
            triggers.append(f"CPU low ({metrics.cpu_utilization:.1f}%)")
        
        if metrics.memory_utilization > thresholds.get('memory_scale_up', 80):
            triggers.append(f"Memory high ({metrics.memory_utilization:.1f}%)")
        elif metrics.memory_utilization < thresholds.get('memory_scale_down', 40):
            triggers.append(f"Memory low ({metrics.memory_utilization:.1f}%)")
        
        if metrics.avg_response_time > thresholds.get('response_time_scale_up', 2.0):
            triggers.append(f"Response time high ({metrics.avg_response_time:.2f}s)")
        elif metrics.avg_response_time < thresholds.get('response_time_scale_down', 1.0):
            triggers.append(f"Response time low ({metrics.avg_response_time:.2f}s)")
        
        return triggers
    
    def _record_scaling_event(self, 
                             from_replicas: int,
                             to_replicas: int,
                             trigger: str,
                             reason: str) -> None:
        """Record scaling event for history and analysis."""
        
        event = {
            'timestamp': datetime.now(),
            'from_replicas': from_replicas,
            'to_replicas': to_replicas,
            'trigger': trigger,
            'reason': reason,
            'direction': 'up' if to_replicas > from_replicas else 'down'
        }
        
        self._scaling_history.append(event)
        
        # Keep only last 100 events
        if len(self._scaling_history) > 100:
            self._scaling_history = self._scaling_history[-100:]
        
        # Update last scale time
        self._last_scale_time[event['direction']] = event['timestamp']
        
        logger.info("Scaling event recorded",
                   from_replicas=from_replicas,
                   to_replicas=to_replicas,
                   trigger=trigger,
                   reason=reason)
    
    async def get_scaling_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        
        return self._scaling_history[-limit:] if self._scaling_history else []
    
    async def get_hpa_status(self) -> Dict[str, Any]:
        """Get current HPA status and configuration."""
        
        if self._mock_mode:
            return {
                'status': 'mock_mode',
                'current_replicas': self._current_replicas,
                'desired_replicas': self._current_replicas,
                'min_replicas': self.policy.min_replicas,
                'max_replicas': self.policy.max_replicas
            }
        
        try:
            hpa = self.autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                name=f"{self.deployment_name}-hpa",
                namespace=self.namespace
            )
            
            return {
                'status': 'active',
                'current_replicas': hpa.status.current_replicas,
                'desired_replicas': hpa.status.desired_replicas,
                'min_replicas': hpa.spec.min_replicas,
                'max_replicas': hpa.spec.max_replicas,
                'last_scale_time': hpa.status.last_scale_time,
                'conditions': [
                    {
                        'type': condition.type,
                        'status': condition.status,
                        'reason': condition.reason,
                        'message': condition.message
                    }
                    for condition in (hpa.status.conditions or [])
                ]
            }
            
        except ApiException as e:
            logger.error("Failed to get HPA status",
                        error=str(e),
                        status=e.status)
            
            return {
                'status': 'error',
                'error': str(e)
            }