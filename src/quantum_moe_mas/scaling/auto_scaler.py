"""
Auto-Scaling System with Kubernetes HPA Integration

Provides dynamic resource scaling based on demand with intelligent
load balancing and predictive scaling capabilities.

Requirements: 8.1, 8.3, 8.4
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import threading

import structlog

logger = structlog.get_logger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling behavior."""
    
    # Basic scaling parameters
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_response_time_ms: float = 2000.0
    
    # Scaling behavior
    scale_up_threshold: float = 0.8  # Scale up when metric > threshold
    scale_down_threshold: float = 0.3  # Scale down when metric < threshold
    scale_up_cooldown: timedelta = timedelta(minutes=3)
    scale_down_cooldown: timedelta = timedelta(minutes=5)
    
    # Advanced settings
    scale_up_step: int = 2  # Number of instances to add
    scale_down_step: int = 1  # Number of instances to remove
    enable_predictive_scaling: bool = True
    aggressive_scaling: bool = False
    
    def validate(self) -> bool:
        """Validate scaling policy configuration."""
        return (
            self.min_instances >= 1 and
            self.max_instances > self.min_instances and
            0 < self.scale_up_threshold < 1 and
            0 < self.scale_down_threshold < self.scale_up_threshold and
            self.scale_up_step > 0 and
            self.scale_down_step > 0
        )


@dataclass
class ScalingMetrics:
    """Current system metrics for scaling decisions."""
    
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    request_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    queue_length: int = 0
    active_connections: int = 0
    error_rate: float = 0.0
    
    # Instance information
    current_instances: int = 1
    healthy_instances: int = 1
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_utilization_score(self) -> float:
        """Calculate overall utilization score (0.0 to 1.0)."""
        scores = [
            self.cpu_utilization / 100.0,
            self.memory_utilization / 100.0,
            min(self.request_rate / 100.0, 1.0),  # Normalize request rate
            min(self.avg_response_time_ms / 5000.0, 1.0)  # Normalize to 5s max
        ]
        return statistics.mean(scores)


@dataclass
class ScalingDecision:
    """Result of a scaling decision."""
    
    direction: ScalingDirection
    target_instances: int
    current_instances: int
    trigger: ScalingTrigger
    confidence: float
    reason: str
    metrics: ScalingMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def instance_change(self) -> int:
        """Number of instances to add/remove."""
        return self.target_instances - self.current_instances


class AutoScaler:
    """
    Intelligent auto-scaling system with Kubernetes HPA integration.
    
    Provides dynamic resource scaling based on multiple metrics with
    predictive capabilities and intelligent cooldown management.
    """
    
    def __init__(self,
                 policy: ScalingPolicy,
                 metrics_collector: Optional[Callable] = None,
                 scaling_executor: Optional[Callable] = None):
        """Initialize auto-scaler."""
        
        if not policy.validate():
            raise ValueError("Invalid scaling policy configuration")
        
        self.policy = policy
        self.metrics_collector = metrics_collector
        self.scaling_executor = scaling_executor
        
        # Scaling state
        self._current_instances = policy.min_instances
        self._last_scale_up = datetime.min
        self._last_scale_down = datetime.min
        self._scaling_history: List[ScalingDecision] = []
        
        # Metrics tracking
        self._metrics_history: List[ScalingMetrics] = []
        self._metrics_lock = threading.RLock()
        
        # Auto-scaling control
        self._is_enabled = True
        self._scaling_task: Optional[asyncio.Task] = None
        
        logger.info("AutoScaler initialized",
                   min_instances=policy.min_instances,
                   max_instances=policy.max_instances,
                   target_cpu=policy.target_cpu_percent)
    
    async def start_auto_scaling(self, check_interval: int = 30) -> None:
        """Start automatic scaling monitoring."""
        
        if self._scaling_task and not self._scaling_task.done():
            logger.warning("Auto-scaling already running")
            return
        
        self._scaling_task = asyncio.create_task(
            self._scaling_loop(check_interval)
        )
        
        logger.info("Auto-scaling started", interval=check_interval)
    
    async def stop_auto_scaling(self) -> None:
        """Stop automatic scaling monitoring."""
        
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaling stopped")
    
    async def _scaling_loop(self, check_interval: int) -> None:
        """Main auto-scaling monitoring loop."""
        
        while self._is_enabled:
            try:
                # Collect current metrics
                if self.metrics_collector:
                    metrics = await self.metrics_collector()
                else:
                    metrics = await self._collect_default_metrics()
                
                # Store metrics history
                with self._metrics_lock:
                    self._metrics_history.append(metrics)
                    # Keep last 1000 measurements
                    if len(self._metrics_history) > 1000:
                        self._metrics_history = self._metrics_history[-1000:]
                
                # Make scaling decision
                decision = await self.make_scaling_decision(metrics)
                
                # Execute scaling if needed
                if decision.direction != ScalingDirection.NONE:
                    await self._execute_scaling_decision(decision)
                
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in scaling loop", error=str(e))
                await asyncio.sleep(check_interval)
    
    async def make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Make intelligent scaling decision based on current metrics."""
        
        # Check if scaling is on cooldown
        now = datetime.now()
        scale_up_ready = now - self._last_scale_up > self.policy.scale_up_cooldown
        scale_down_ready = now - self._last_scale_down > self.policy.scale_down_cooldown
        
        # Analyze metrics for scaling triggers
        triggers = self._analyze_scaling_triggers(metrics)
        
        # Determine scaling direction and magnitude
        if triggers['scale_up'] and scale_up_ready:
            target_instances = min(
                metrics.current_instances + self.policy.scale_up_step,
                self.policy.max_instances
            )
            
            decision = ScalingDecision(
                direction=ScalingDirection.UP,
                target_instances=target_instances,
                current_instances=metrics.current_instances,
                trigger=triggers['primary_trigger'],
                confidence=triggers['confidence'],
                reason=triggers['reason'],
                metrics=metrics
            )
            
        elif triggers['scale_down'] and scale_down_ready:
            target_instances = max(
                metrics.current_instances - self.policy.scale_down_step,
                self.policy.min_instances
            )
            
            decision = ScalingDecision(
                direction=ScalingDirection.DOWN,
                target_instances=target_instances,
                current_instances=metrics.current_instances,
                trigger=triggers['primary_trigger'],
                confidence=triggers['confidence'],
                reason=triggers['reason'],
                metrics=metrics
            )
            
        else:
            decision = ScalingDecision(
                direction=ScalingDirection.NONE,
                target_instances=metrics.current_instances,
                current_instances=metrics.current_instances,
                trigger=ScalingTrigger.CUSTOM_METRIC,
                confidence=0.0,
                reason="No scaling needed or on cooldown",
                metrics=metrics
            )
        
        logger.debug("Scaling decision made",
                    direction=decision.direction.value,
                    target_instances=decision.target_instances,
                    trigger=decision.trigger.value,
                    confidence=decision.confidence)
        
        return decision
    
    def _analyze_scaling_triggers(self, metrics: ScalingMetrics) -> Dict[str, Any]:
        """Analyze metrics to determine scaling triggers."""
        
        triggers = {
            'scale_up': False,
            'scale_down': False,
            'primary_trigger': ScalingTrigger.CPU_UTILIZATION,
            'confidence': 0.0,
            'reason': ''
        }
        
        # CPU-based scaling
        cpu_score = metrics.cpu_utilization / 100.0
        if cpu_score > self.policy.scale_up_threshold:
            triggers['scale_up'] = True
            triggers['primary_trigger'] = ScalingTrigger.CPU_UTILIZATION
            triggers['confidence'] = min(0.9, cpu_score)
            triggers['reason'] = f"High CPU utilization: {metrics.cpu_utilization:.1f}%"
        elif cpu_score < self.policy.scale_down_threshold:
            triggers['scale_down'] = True
            triggers['primary_trigger'] = ScalingTrigger.CPU_UTILIZATION
            triggers['confidence'] = min(0.8, 1.0 - cpu_score)
            triggers['reason'] = f"Low CPU utilization: {metrics.cpu_utilization:.1f}%"
        
        # Memory-based scaling
        memory_score = metrics.memory_utilization / 100.0
        if memory_score > self.policy.scale_up_threshold:
            if not triggers['scale_up'] or memory_score > cpu_score:
                triggers['scale_up'] = True
                triggers['primary_trigger'] = ScalingTrigger.MEMORY_UTILIZATION
                triggers['confidence'] = min(0.95, memory_score)
                triggers['reason'] = f"High memory utilization: {metrics.memory_utilization:.1f}%"
        
        # Response time-based scaling
        if metrics.p95_response_time_ms > self.policy.target_response_time_ms:
            response_time_score = metrics.p95_response_time_ms / self.policy.target_response_time_ms
            if response_time_score > 1.5:  # 50% above target
                triggers['scale_up'] = True
                triggers['primary_trigger'] = ScalingTrigger.RESPONSE_TIME
                triggers['confidence'] = min(0.9, response_time_score - 1.0)
                triggers['reason'] = f"High response time: P95 = {metrics.p95_response_time_ms:.0f}ms"
        
        # Queue length-based scaling
        if metrics.queue_length > 10:  # Arbitrary threshold
            triggers['scale_up'] = True
            triggers['primary_trigger'] = ScalingTrigger.QUEUE_LENGTH
            triggers['confidence'] = min(0.8, metrics.queue_length / 50.0)
            triggers['reason'] = f"High queue length: {metrics.queue_length} requests"
        
        # Error rate consideration (prevents scaling down during issues)
        if metrics.error_rate > 0.05:  # 5% error rate
            triggers['scale_down'] = False
            if not triggers['scale_up']:
                triggers['reason'] = f"High error rate prevents scaling: {metrics.error_rate:.1%}"
        
        # Predictive scaling (if enabled)
        if self.policy.enable_predictive_scaling:
            predictive_trigger = self._analyze_predictive_scaling(metrics)
            if predictive_trigger['scale_up'] and not triggers['scale_up']:
                triggers.update(predictive_trigger)
        
        return triggers
    
    def _analyze_predictive_scaling(self, current_metrics: ScalingMetrics) -> Dict[str, Any]:
        """Analyze historical patterns for predictive scaling."""
        
        with self._metrics_lock:
            if len(self._metrics_history) < 10:
                return {'scale_up': False, 'scale_down': False}
            
            # Analyze recent trend
            recent_metrics = self._metrics_history[-10:]
            cpu_trend = [m.cpu_utilization for m in recent_metrics]
            memory_trend = [m.memory_utilization for m in recent_metrics]
            
            # Calculate trend slopes
            cpu_slope = self._calculate_trend_slope(cpu_trend)
            memory_slope = self._calculate_trend_slope(memory_trend)
            
            # Predict future utilization
            predicted_cpu = current_metrics.cpu_utilization + (cpu_slope * 3)  # 3 intervals ahead
            predicted_memory = current_metrics.memory_utilization + (memory_slope * 3)
            
            # Check if predicted values exceed thresholds
            if predicted_cpu > 85 or predicted_memory > 90:
                return {
                    'scale_up': True,
                    'scale_down': False,
                    'primary_trigger': ScalingTrigger.CUSTOM_METRIC,
                    'confidence': 0.7,
                    'reason': f"Predictive scaling: CPU trend {cpu_slope:.1f}%, Memory trend {memory_slope:.1f}%"
                }
        
        return {'scale_up': False, 'scale_down': False}
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope for a series of values."""
        
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression slope calculation
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision."""
        
        logger.info("Executing scaling decision",
                   direction=decision.direction.value,
                   current=decision.current_instances,
                   target=decision.target_instances,
                   reason=decision.reason)
        
        try:
            # Execute scaling through configured executor
            if self.scaling_executor:
                success = await self.scaling_executor(decision)
            else:
                success = await self._default_scaling_executor(decision)
            
            if success:
                # Update internal state
                self._current_instances = decision.target_instances
                
                # Update cooldown timers
                if decision.direction == ScalingDirection.UP:
                    self._last_scale_up = datetime.now()
                elif decision.direction == ScalingDirection.DOWN:
                    self._last_scale_down = datetime.now()
                
                # Store in history
                self._scaling_history.append(decision)
                
                logger.info("Scaling executed successfully",
                           new_instances=decision.target_instances)
                
                return True
            else:
                logger.error("Scaling execution failed")
                return False
                
        except Exception as e:
            logger.error("Error executing scaling decision", error=str(e))
            return False
    
    async def _default_scaling_executor(self, decision: ScalingDecision) -> bool:
        """Default scaling executor (placeholder implementation)."""
        
        # This would integrate with your container orchestration system
        # For now, just simulate the scaling operation
        
        logger.info("Simulating scaling operation",
                   direction=decision.direction.value,
                   instances=decision.target_instances)
        
        # Simulate scaling delay
        await asyncio.sleep(1)
        
        return True
    
    async def _collect_default_metrics(self) -> ScalingMetrics:
        """Collect default system metrics (placeholder implementation)."""
        
        # This would integrate with your monitoring system
        # For now, return simulated metrics
        
        return ScalingMetrics(
            cpu_utilization=50.0,
            memory_utilization=60.0,
            request_rate=10.0,
            avg_response_time_ms=1500.0,
            p95_response_time_ms=2500.0,
            queue_length=5,
            active_connections=20,
            error_rate=0.01,
            current_instances=self._current_instances,
            healthy_instances=self._current_instances
        )
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        
        return {
            'enabled': self._is_enabled,
            'current_instances': self._current_instances,
            'min_instances': self.policy.min_instances,
            'max_instances': self.policy.max_instances,
            'last_scale_up': self._last_scale_up.isoformat() if self._last_scale_up != datetime.min else None,
            'last_scale_down': self._last_scale_down.isoformat() if self._last_scale_down != datetime.min else None,
            'scaling_history_count': len(self._scaling_history),
            'metrics_history_count': len(self._metrics_history)
        }
    
    def get_scaling_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        
        recent_decisions = self._scaling_history[-limit:]
        
        return [
            {
                'timestamp': decision.timestamp.isoformat(),
                'direction': decision.direction.value,
                'target_instances': decision.target_instances,
                'current_instances': decision.current_instances,
                'trigger': decision.trigger.value,
                'confidence': decision.confidence,
                'reason': decision.reason
            }
            for decision in recent_decisions
        ]
    
    def enable_scaling(self) -> None:
        """Enable auto-scaling."""
        self._is_enabled = True
        logger.info("Auto-scaling enabled")
    
    def disable_scaling(self) -> None:
        """Disable auto-scaling."""
        self._is_enabled = False
        logger.info("Auto-scaling disabled")
    
    async def manual_scale(self, target_instances: int, reason: str = "Manual scaling") -> bool:
        """Manually scale to target instance count."""
        
        if target_instances < self.policy.min_instances or target_instances > self.policy.max_instances:
            logger.error("Manual scaling target outside policy limits",
                        target=target_instances,
                        min_instances=self.policy.min_instances,
                        max_instances=self.policy.max_instances)
            return False
        
        # Create manual scaling decision
        decision = ScalingDecision(
            direction=ScalingDirection.UP if target_instances > self._current_instances else ScalingDirection.DOWN,
            target_instances=target_instances,
            current_instances=self._current_instances,
            trigger=ScalingTrigger.CUSTOM_METRIC,
            confidence=1.0,
            reason=reason,
            metrics=await self._collect_default_metrics()
        )
        
        return await self._execute_scaling_decision(decision)