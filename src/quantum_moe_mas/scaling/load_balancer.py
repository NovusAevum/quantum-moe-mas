"""
Load Balancer with Health Checks

Implements intelligent load balancing across multiple instances with
comprehensive health monitoring and automatic failover capabilities.

Requirements: 8.1, 8.3
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class LoadBalancingStrategy(Enum):
    """Different load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASH = "consistent_hash"


class InstanceStatus(Enum):
    """Health status of service instances."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


@dataclass
class ServiceInstance:
    """Represents a service instance in the load balancer pool."""
    
    id: str
    host: str
    port: int
    weight: float = 1.0
    
    # Health status
    status: InstanceStatus = InstanceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # Performance metrics
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time: float = 0.0
    last_response_time: float = 0.0
    
    # Configuration
    max_connections: int = 100
    health_check_path: str = "/health"
    health_check_timeout: int = 5
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        """Get the base URL for this instance."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_check_url(self) -> str:
        """Get the health check URL for this instance."""
        return f"{self.url}{self.health_check_path}"
    
    @property
    def is_healthy(self) -> bool:
        """Check if instance is healthy and available."""
        return self.status == InstanceStatus.HEALTHY
    
    @property
    def is_available(self) -> bool:
        """Check if instance is available for requests."""
        return (
            self.status in [InstanceStatus.HEALTHY, InstanceStatus.DEGRADED] and
            self.active_connections < self.max_connections
        )
    
    @property
    def load_score(self) -> float:
        """Calculate load score for load balancing decisions."""
        if not self.is_available:
            return float('inf')
        
        # Combine connection count and response time
        connection_factor = self.active_connections / self.max_connections
        response_time_factor = min(self.avg_response_time / 1000.0, 1.0)  # Normalize to 0-1
        
        return (connection_factor * 0.6) + (response_time_factor * 0.4)


@dataclass
class LoadBalancerMetrics:
    """Metrics for load balancer performance."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    
    active_instances: int = 0
    total_instances: int = 0
    
    requests_per_second: float = 0.0
    errors_per_second: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return 100.0 - self.success_rate


class HealthChecker:
    """
    Health checker for monitoring service instance health.
    
    Performs periodic health checks and maintains instance status.
    """
    
    def __init__(self, 
                 check_interval: int = 30,
                 timeout: int = 5,
                 failure_threshold: int = 3,
                 success_threshold: int = 2):
        """Initialize health checker."""
        
        self.check_interval = check_interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        
        self._running = False
        self._check_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("HealthChecker initialized",
                   interval=check_interval,
                   timeout=timeout,
                   failure_threshold=failure_threshold,
                   success_threshold=success_threshold)
    
    async def start_monitoring(self, instances: List[ServiceInstance]) -> None:
        """Start health monitoring for all instances."""
        
        if self._running:
            logger.warning("Health monitoring already running")
            return
        
        self._running = True
        
        # Start health check tasks for each instance
        for instance in instances:
            task = asyncio.create_task(self._monitor_instance(instance))
            self._check_tasks[instance.id] = task
        
        logger.info("Health monitoring started", instance_count=len(instances))
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring for all instances."""
        
        self._running = False
        
        # Cancel all health check tasks
        for task in self._check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._check_tasks:
            await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)
        
        self._check_tasks.clear()
        logger.info("Health monitoring stopped")
    
    async def add_instance(self, instance: ServiceInstance) -> None:
        """Add new instance to health monitoring."""
        
        if self._running and instance.id not in self._check_tasks:
            task = asyncio.create_task(self._monitor_instance(instance))
            self._check_tasks[instance.id] = task
            
            logger.info("Added instance to health monitoring", instance_id=instance.id)
    
    async def remove_instance(self, instance_id: str) -> None:
        """Remove instance from health monitoring."""
        
        if instance_id in self._check_tasks:
            self._check_tasks[instance_id].cancel()
            del self._check_tasks[instance_id]
            
            logger.info("Removed instance from health monitoring", instance_id=instance_id)
    
    async def _monitor_instance(self, instance: ServiceInstance) -> None:
        """Monitor health of a single instance."""
        
        logger.info("Starting health monitoring for instance", instance_id=instance.id)
        
        while self._running:
            try:
                # Perform health check
                is_healthy = await self._perform_health_check(instance)
                
                # Update instance status based on health check result
                await self._update_instance_status(instance, is_healthy)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("Health monitoring cancelled for instance", instance_id=instance.id)
                break
            except Exception as e:
                logger.error("Error in health monitoring",
                           instance_id=instance.id,
                           error=str(e))
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self, instance: ServiceInstance) -> bool:
        """Perform health check on a single instance."""
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(instance.health_check_url) as response:
                    response_time = time.time() - start_time
                    
                    # Update response time metrics
                    instance.last_response_time = response_time
                    if instance.avg_response_time == 0:
                        instance.avg_response_time = response_time
                    else:
                        # Exponential moving average
                        instance.avg_response_time = (
                            instance.avg_response_time * 0.8 + response_time * 0.2
                        )
                    
                    # Check if response indicates health
                    is_healthy = (
                        response.status == 200 and
                        response_time < (self.timeout * 0.8)  # Response time should be reasonable
                    )
                    
                    logger.debug("Health check completed",
                               instance_id=instance.id,
                               status_code=response.status,
                               response_time=response_time,
                               is_healthy=is_healthy)
                    
                    return is_healthy
                    
        except asyncio.TimeoutError:
            logger.warning("Health check timeout", instance_id=instance.id)
            return False
        except Exception as e:
            logger.warning("Health check failed",
                         instance_id=instance.id,
                         error=str(e))
            return False
    
    async def _update_instance_status(self, instance: ServiceInstance, is_healthy: bool) -> None:
        """Update instance status based on health check result."""
        
        instance.last_health_check = datetime.now()
        
        if is_healthy:
            instance.consecutive_successes += 1
            instance.consecutive_failures = 0
            
            # Mark as healthy if we have enough consecutive successes
            if (instance.status != InstanceStatus.HEALTHY and 
                instance.consecutive_successes >= self.success_threshold):
                
                old_status = instance.status
                instance.status = InstanceStatus.HEALTHY
                
                logger.info("Instance marked as healthy",
                           instance_id=instance.id,
                           old_status=old_status.value,
                           consecutive_successes=instance.consecutive_successes)
        else:
            instance.consecutive_failures += 1
            instance.consecutive_successes = 0
            
            # Mark as unhealthy if we have enough consecutive failures
            if (instance.status == InstanceStatus.HEALTHY and 
                instance.consecutive_failures >= self.failure_threshold):
                
                instance.status = InstanceStatus.UNHEALTHY
                
                logger.warning("Instance marked as unhealthy",
                             instance_id=instance.id,
                             consecutive_failures=instance.consecutive_failures)
            
            # Mark as degraded for fewer failures
            elif (instance.status == InstanceStatus.HEALTHY and 
                  instance.consecutive_failures >= 1):
                
                instance.status = InstanceStatus.DEGRADED
                
                logger.warning("Instance marked as degraded",
                             instance_id=instance.id,
                             consecutive_failures=instance.consecutive_failures)


class LoadBalancer:
    """
    Intelligent load balancer with multiple strategies and health monitoring.
    
    Distributes requests across healthy service instances using various
    load balancing algorithms and automatic failover.
    """
    
    def __init__(self,
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME,
                 health_check_interval: int = 30):
        """Initialize the load balancer."""
        
        self.strategy = strategy
        self.instances: Dict[str, ServiceInstance] = {}
        self.health_checker = HealthChecker(check_interval=health_check_interval)
        
        # Load balancing state
        self._round_robin_index = 0
        self._request_counts: Dict[str, int] = {}
        
        # Metrics
        self._metrics = LoadBalancerMetrics()
        self._request_history = []
        
        logger.info("LoadBalancer initialized", strategy=strategy.value)
    
    async def add_instance(self, instance: ServiceInstance) -> None:
        """Add a service instance to the load balancer pool."""
        
        self.instances[instance.id] = instance
        self._request_counts[instance.id] = 0
        
        # Add to health monitoring
        await self.health_checker.add_instance(instance)
        
        logger.info("Instance added to load balancer",
                   instance_id=instance.id,
                   host=instance.host,
                   port=instance.port)
    
    async def remove_instance(self, instance_id: str) -> None:
        """Remove a service instance from the load balancer pool."""
        
        if instance_id in self.instances:
            del self.instances[instance_id]
            del self._request_counts[instance_id]
            
            # Remove from health monitoring
            await self.health_checker.remove_instance(instance_id)
            
            logger.info("Instance removed from load balancer", instance_id=instance_id)
    
    async def start(self) -> None:
        """Start the load balancer and health monitoring."""
        
        if self.instances:
            await self.health_checker.start_monitoring(list(self.instances.values()))
        
        logger.info("Load balancer started", instance_count=len(self.instances))
    
    async def stop(self) -> None:
        """Stop the load balancer and health monitoring."""
        
        await self.health_checker.stop_monitoring()
        logger.info("Load balancer stopped")
    
    async def select_instance(self, 
                            request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """Select the best instance for handling a request."""
        
        available_instances = [
            instance for instance in self.instances.values()
            if instance.is_available
        ]
        
        if not available_instances:
            logger.warning("No available instances for request routing")
            return None
        
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected = self._select_round_robin(available_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected = self._select_weighted_round_robin(available_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected = self._select_least_connections(available_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected = self._select_least_response_time(available_instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            selected = self._select_random(available_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            selected = self._select_weighted_random(available_instances)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            selected = self._select_consistent_hash(available_instances, request_context)
        else:
            selected = self._select_least_response_time(available_instances)
        
        if selected:
            logger.debug("Instance selected for request",
                        instance_id=selected.id,
                        strategy=self.strategy.value,
                        load_score=selected.load_score)
        
        return selected
    
    def _select_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance using round-robin strategy."""
        
        selected = instances[self._round_robin_index % len(instances)]
        self._round_robin_index += 1
        return selected
    
    def _select_weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance using weighted round-robin strategy."""
        
        # Create weighted list based on instance weights
        weighted_instances = []
        for instance in instances:
            weight_count = max(1, int(instance.weight * 10))
            weighted_instances.extend([instance] * weight_count)
        
        selected = weighted_instances[self._round_robin_index % len(weighted_instances)]
        self._round_robin_index += 1
        return selected
    
    def _select_least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least active connections."""
        
        return min(instances, key=lambda x: x.active_connections)
    
    def _select_least_response_time(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with lowest average response time."""
        
        return min(instances, key=lambda x: x.load_score)
    
    def _select_random(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance randomly."""
        
        return random.choice(instances)
    
    def _select_weighted_random(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance using weighted random strategy."""
        
        total_weight = sum(instance.weight for instance in instances)
        random_weight = random.uniform(0, total_weight)
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.weight
            if random_weight <= current_weight:
                return instance
        
        return instances[-1]  # Fallback
    
    def _select_consistent_hash(self, 
                              instances: List[ServiceInstance],
                              request_context: Optional[Dict[str, Any]]) -> ServiceInstance:
        """Select instance using consistent hashing."""
        
        # Use session ID or user ID for consistent routing
        hash_key = "default"
        if request_context:
            hash_key = (
                request_context.get('session_id') or
                request_context.get('user_id') or
                request_context.get('client_ip', 'default')
            )
        
        # Simple hash-based selection
        hash_value = hash(hash_key) % len(instances)
        return instances[hash_value]
    
    async def handle_request(self,
                           request_handler: Callable,
                           request_data: Any,
                           request_context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle a request through the load balancer."""
        
        start_time = time.time()
        selected_instance = None
        
        try:
            # Select instance
            selected_instance = await self.select_instance(request_context)
            
            if not selected_instance:
                raise Exception("No available instances")
            
            # Update connection count
            selected_instance.active_connections += 1
            
            try:
                # Execute request
                result = await request_handler(selected_instance, request_data)
                
                # Record successful request
                self._record_request_success(selected_instance, time.time() - start_time)
                
                return result
                
            finally:
                # Always decrement connection count
                selected_instance.active_connections -= 1
                
        except Exception as e:
            # Record failed request
            self._record_request_failure(selected_instance, time.time() - start_time, str(e))
            raise
    
    def _record_request_success(self, instance: ServiceInstance, response_time: float) -> None:
        """Record successful request metrics."""
        
        # Update instance metrics
        if instance:
            instance.total_requests += 1
            self._request_counts[instance.id] += 1
            
            # Update response time
            if instance.avg_response_time == 0:
                instance.avg_response_time = response_time
            else:
                instance.avg_response_time = (
                    instance.avg_response_time * 0.9 + response_time * 0.1
                )
        
        # Update load balancer metrics
        self._metrics.total_requests += 1
        self._metrics.successful_requests += 1
        
        # Update average response time
        if self._metrics.avg_response_time == 0:
            self._metrics.avg_response_time = response_time
        else:
            self._metrics.avg_response_time = (
                self._metrics.avg_response_time * 0.9 + response_time * 0.1
            )
        
        # Store for percentile calculations
        self._request_history.append({
            'timestamp': datetime.now(),
            'response_time': response_time,
            'success': True
        })
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self._request_history = [
            req for req in self._request_history
            if req['timestamp'] > cutoff_time
        ]
    
    def _record_request_failure(self, 
                              instance: Optional[ServiceInstance],
                              response_time: float,
                              error: str) -> None:
        """Record failed request metrics."""
        
        # Update instance metrics
        if instance:
            instance.total_requests += 1
            instance.total_errors += 1
        
        # Update load balancer metrics
        self._metrics.total_requests += 1
        self._metrics.failed_requests += 1
        
        # Store for analysis
        self._request_history.append({
            'timestamp': datetime.now(),
            'response_time': response_time,
            'success': False,
            'error': error
        })
        
        logger.warning("Request failed through load balancer",
                     instance_id=instance.id if instance else None,
                     error=error,
                     response_time=response_time)
    
    async def get_metrics(self) -> LoadBalancerMetrics:
        """Get current load balancer metrics."""
        
        # Update instance counts
        self._metrics.total_instances = len(self.instances)
        self._metrics.active_instances = sum(
            1 for instance in self.instances.values()
            if instance.is_healthy
        )
        
        # Calculate rates from recent history
        recent_requests = [
            req for req in self._request_history
            if req['timestamp'] > datetime.now() - timedelta(seconds=60)
        ]
        
        if recent_requests:
            self._metrics.requests_per_second = len(recent_requests) / 60.0
            failed_requests = [req for req in recent_requests if not req['success']]
            self._metrics.errors_per_second = len(failed_requests) / 60.0
            
            # Calculate P95 response time
            response_times = [req['response_time'] for req in recent_requests]
            if response_times:
                response_times.sort()
                p95_index = int(len(response_times) * 0.95)
                self._metrics.p95_response_time = response_times[p95_index]
        
        self._metrics.timestamp = datetime.now()
        return self._metrics
    
    async def get_instance_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all instances."""
        
        status = {}
        for instance_id, instance in self.instances.items():
            status[instance_id] = {
                'id': instance.id,
                'host': instance.host,
                'port': instance.port,
                'status': instance.status.value,
                'weight': instance.weight,
                'active_connections': instance.active_connections,
                'total_requests': instance.total_requests,
                'total_errors': instance.total_errors,
                'avg_response_time': instance.avg_response_time,
                'last_health_check': instance.last_health_check.isoformat() if instance.last_health_check else None,
                'consecutive_failures': instance.consecutive_failures,
                'consecutive_successes': instance.consecutive_successes,
                'is_available': instance.is_available,
                'load_score': instance.load_score if instance.is_available else None
            }
        
        return status