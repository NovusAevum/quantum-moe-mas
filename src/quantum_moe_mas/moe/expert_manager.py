"""
Expert Pool Management System.

This module provides comprehensive expert management including health monitoring,
failover mechanisms, and performance tracking.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum

from quantum_moe_mas.moe.expert import Expert, ExpertStatus, ExpertType
from quantum_moe_mas.core.logging_simple import get_logger


logger = get_logger(__name__)


class FailoverStrategy(Enum):
    """Failover strategies for expert management."""
    
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    HIGHEST_CONFIDENCE = "highest_confidence"
    RANDOM = "random"


class ExpertHealthMonitor:
    """
    Monitors expert health and triggers failover when needed.
    
    Attributes:
        check_interval_seconds: Interval between health checks
        unhealthy_threshold: Number of failed checks before marking unhealthy
        recovery_threshold: Number of successful checks before marking healthy
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 60,
        unhealthy_threshold: int = 3,
        recovery_threshold: int = 2
    ) -> None:
        """
        Initialize health monitor.
        
        Args:
            check_interval_seconds: Seconds between health checks
            unhealthy_threshold: Failed checks before marking unhealthy
            recovery_threshold: Successful checks before marking healthy
        """
        self.check_interval_seconds = check_interval_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self.recovery_threshold = recovery_threshold
        
        self.failed_checks: Dict[str, int] = {}
        self.successful_checks: Dict[str, int] = {}
        self.last_check_time: Dict[str, datetime] = {}
        self.monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"Initialized ExpertHealthMonitor: check_interval={check_interval_seconds}s, "
            f"unhealthy_threshold={unhealthy_threshold}"
        )
    
    async def check_expert_health(self, expert: Expert) -> bool:
        """
        Perform health check on an expert.
        
        Args:
            expert: Expert to check
        
        Returns:
            True if expert is healthy
        """
        try:
            # Check if expert has recent activity
            if expert.performance.last_request_timestamp:
                time_since_last = datetime.utcnow() - expert.performance.last_request_timestamp
                if time_since_last > timedelta(minutes=30):
                    logger.warning(
                        f"Expert {expert.name} has no recent activity",
                        last_request=expert.performance.last_request_timestamp
                    )
            
            # Check failure rate
            if expert.performance.failure_rate > 50.0:
                logger.warning(
                    f"Expert {expert.name} has high failure rate",
                    failure_rate=expert.performance.failure_rate
                )
                return False
            
            # Check if overloaded
            if expert.load_factor > 0.95:
                logger.warning(
                    f"Expert {expert.name} is overloaded",
                    load_factor=expert.load_factor
                )
                return False
            
            # All checks passed
            return True
            
        except Exception as e:
            logger.error(
                f"Health check failed for expert {expert.name}",
                error=str(e)
            )
            return False
    
    async def monitor_expert(self, expert: Expert) -> None:
        """
        Monitor a single expert's health.
        
        Args:
            expert: Expert to monitor
        """
        is_healthy = await self.check_expert_health(expert)
        
        if is_healthy:
            # Reset failed checks and increment successful checks
            self.failed_checks[expert.id] = 0
            self.successful_checks[expert.id] = self.successful_checks.get(expert.id, 0) + 1
            
            # Update status if recovered
            if (expert.status != ExpertStatus.HEALTHY and
                self.successful_checks[expert.id] >= self.recovery_threshold):
                expert.update_status(ExpertStatus.HEALTHY)
                logger.info(f"Expert {expert.name} recovered to healthy status")
        else:
            # Reset successful checks and increment failed checks
            self.successful_checks[expert.id] = 0
            self.failed_checks[expert.id] = self.failed_checks.get(expert.id, 0) + 1
            
            # Update status if unhealthy
            if self.failed_checks[expert.id] >= self.unhealthy_threshold:
                if expert.status == ExpertStatus.HEALTHY:
                    expert.update_status(ExpertStatus.DEGRADED)
                    logger.warning(f"Expert {expert.name} marked as degraded")
                elif expert.status == ExpertStatus.DEGRADED:
                    expert.update_status(ExpertStatus.UNHEALTHY)
                    logger.error(f"Expert {expert.name} marked as unhealthy")
        
        self.last_check_time[expert.id] = datetime.utcnow()
    
    async def start_monitoring(self, experts: Dict[str, Expert]) -> None:
        """
        Start continuous health monitoring.
        
        Args:
            experts: Dictionary of experts to monitor
        """
        self.monitoring_active = True
        logger.info("Started expert health monitoring")
        
        while self.monitoring_active:
            try:
                # Monitor all experts
                tasks = [self.monitor_expert(expert) for expert in experts.values()]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait for next check interval
                await asyncio.sleep(self.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Stopped expert health monitoring")
    
    def get_health_report(self) -> Dict[str, any]:
        """
        Get health monitoring report.
        
        Returns:
            Dictionary with health statistics
        """
        return {
            "monitoring_active": self.monitoring_active,
            "experts_monitored": len(self.last_check_time),
            "failed_checks": dict(self.failed_checks),
            "successful_checks": dict(self.successful_checks),
            "last_check_times": {
                k: v.isoformat() for k, v in self.last_check_time.items()
            }
        }


class ExpertPoolManager:
    """
    Manages the pool of experts with health monitoring and failover.
    
    Provides comprehensive expert lifecycle management including addition,
    removal, health monitoring, and automatic failover.
    """
    
    def __init__(
        self,
        failover_strategy: FailoverStrategy = FailoverStrategy.LEAST_LOADED,
        enable_auto_failover: bool = True
    ) -> None:
        """
        Initialize expert pool manager.
        
        Args:
            failover_strategy: Strategy for selecting failover experts
            enable_auto_failover: Whether to enable automatic failover
        """
        self.experts: Dict[str, Expert] = {}
        self.failover_strategy = failover_strategy
        self.enable_auto_failover = enable_auto_failover
        self.health_monitor = ExpertHealthMonitor()
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"Initialized ExpertPoolManager: failover_strategy={failover_strategy.value}, "
            f"auto_failover={enable_auto_failover}"
        )
    
    def add_expert(self, expert: Expert) -> bool:
        """
        Add an expert to the pool.
        
        Args:
            expert: Expert to add
        
        Returns:
            True if expert was added successfully
        
        Raises:
            ValueError: If expert with same ID already exists
        """
        if expert.id in self.experts:
            raise ValueError(f"Expert with ID {expert.id} already exists")
        
        self.experts[expert.id] = expert
        logger.info(
            f"Added expert to pool: {expert.name} (ID: {expert.id}, type: {expert.type.value})"
        )
        return True
    
    def remove_expert(self, expert_id: str) -> bool:
        """
        Remove an expert from the pool.
        
        Args:
            expert_id: ID of expert to remove
        
        Returns:
            True if expert was removed successfully
        """
        if expert_id not in self.experts:
            logger.warning(f"Expert {expert_id} not found in pool")
            return False
        
        expert = self.experts.pop(expert_id)
        logger.info(
            f"Removed expert from pool: {expert.name} (ID: {expert_id})"
        )
        return True
    
    def get_expert(self, expert_id: str) -> Optional[Expert]:
        """
        Get an expert by ID.
        
        Args:
            expert_id: ID of expert to retrieve
        
        Returns:
            Expert instance or None if not found
        """
        return self.experts.get(expert_id)
    
    def list_experts(
        self,
        status: Optional[ExpertStatus] = None,
        expert_type: Optional[ExpertType] = None
    ) -> List[Expert]:
        """
        List experts with optional filtering.
        
        Args:
            status: Filter by status
            expert_type: Filter by type
        
        Returns:
            List of Expert instances
        """
        experts = list(self.experts.values())
        
        if status:
            experts = [e for e in experts if e.status == status]
        
        if expert_type:
            experts = [e for e in experts if e.type == expert_type]
        
        return experts
    
    def get_available_experts(self) -> List[Expert]:
        """
        Get all available experts.
        
        Returns:
            List of available Expert instances
        """
        return [e for e in self.experts.values() if e.is_available()]
    
    def find_failover_expert(
        self,
        failed_expert_id: str,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[Expert]:
        """
        Find a suitable failover expert.
        
        Args:
            failed_expert_id: ID of the failed expert
            required_capabilities: Required capabilities for failover
        
        Returns:
            Failover Expert instance or None if not found
        """
        # Get available experts excluding the failed one
        candidates = [
            e for e in self.get_available_experts()
            if e.id != failed_expert_id
        ]
        
        # Filter by capabilities if specified
        if required_capabilities:
            candidates = [
                e for e in candidates
                if any(cap in e.capabilities for cap in required_capabilities)
            ]
        
        if not candidates:
            logger.warning("No failover candidates available")
            return None
        
        # Select based on strategy
        if self.failover_strategy == FailoverStrategy.LEAST_LOADED:
            return min(candidates, key=lambda e: e.load_factor)
        
        elif self.failover_strategy == FailoverStrategy.HIGHEST_CONFIDENCE:
            return max(candidates, key=lambda e: e.confidence_score)
        
        elif self.failover_strategy == FailoverStrategy.ROUND_ROBIN:
            # Simple round-robin: return first available
            return candidates[0]
        
        elif self.failover_strategy == FailoverStrategy.RANDOM:
            import random
            return random.choice(candidates)
        
        return candidates[0]
    
    async def handle_expert_failure(
        self,
        expert_id: str,
        error: Exception
    ) -> Optional[Expert]:
        """
        Handle expert failure and trigger failover if enabled.
        
        Args:
            expert_id: ID of failed expert
            error: Exception that caused the failure
        
        Returns:
            Failover expert or None
        """
        expert = self.get_expert(expert_id)
        if not expert:
            logger.error(f"Failed expert {expert_id} not found")
            return None
        
        # Update expert status
        expert.update_status(ExpertStatus.UNHEALTHY)
        logger.error(
            f"Expert {expert.name} failed",
            expert_id=expert_id,
            error=str(error)
        )
        
        # Find failover if enabled
        if self.enable_auto_failover:
            failover = self.find_failover_expert(
                expert_id,
                required_capabilities=expert.capabilities
            )
            
            if failover:
                logger.info(
                    f"Failover from {expert.name} (ID: {expert_id}) to {failover.name} (ID: {failover.id})"
                )
                return failover
        
        return None
    
    def update_expert_performance(
        self,
        expert_id: str,
        success: bool,
        latency_ms: float,
        confidence: float,
        tokens_used: int = 0,
        cost: float = 0.0
    ) -> None:
        """
        Update expert performance metrics.
        
        Args:
            expert_id: ID of expert
            success: Whether request was successful
            latency_ms: Request latency in milliseconds
            confidence: Response confidence score
            tokens_used: Number of tokens used
            cost: Cost of the request
        """
        expert = self.get_expert(expert_id)
        if expert:
            expert.record_request(success, latency_ms, confidence, tokens_used, cost)
    
    async def start_health_monitoring(self) -> None:
        """Start health monitoring for all experts."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Health monitoring already running")
            return
        
        self._monitoring_task = asyncio.create_task(
            self.health_monitor.start_monitoring(self.experts)
        )
        logger.info("Started health monitoring task")
    
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self.health_monitor.stop_monitoring()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Stopped health monitoring task")
    
    def get_pool_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        total = len(self.experts)
        if total == 0:
            return {
                "total_experts": 0,
                "available_experts": 0,
                "status_distribution": {},
                "type_distribution": {},
                "average_load": 0.0,
                "average_confidence": 0.0,
                "total_requests": 0,
                "total_cost": 0.0,
            }
        
        available = len(self.get_available_experts())
        
        status_dist = {}
        type_dist = {}
        total_load = 0.0
        total_confidence = 0.0
        total_requests = 0
        total_cost = 0.0
        
        for expert in self.experts.values():
            # Status distribution
            status = expert.status.value
            status_dist[status] = status_dist.get(status, 0) + 1
            
            # Type distribution
            exp_type = expert.type.value
            type_dist[exp_type] = type_dist.get(exp_type, 0) + 1
            
            # Aggregates
            total_load += expert.load_factor
            total_confidence += expert.confidence_score
            total_requests += expert.performance.total_requests
            total_cost += expert.performance.total_cost
        
        return {
            "total_experts": total,
            "available_experts": available,
            "status_distribution": status_dist,
            "type_distribution": type_dist,
            "average_load": total_load / total,
            "average_confidence": total_confidence / total,
            "total_requests": total_requests,
            "total_cost": total_cost,
            "health_report": self.health_monitor.get_health_report(),
        }
    
    def optimize_expert_pool(self) -> Dict[str, any]:
        """
        Optimize expert pool by rebalancing load and updating priorities.
        
        Returns:
            Dictionary with optimization results
        """
        optimizations = {
            "load_rebalanced": 0,
            "priorities_updated": 0,
            "experts_recovered": 0,
        }
        
        # Rebalance load
        for expert in self.experts.values():
            if expert.load_factor > 0.1:
                # Gradually reduce load
                new_load = max(0.0, expert.load_factor - 0.1)
                expert.update_load(new_load)
                optimizations["load_rebalanced"] += 1
        
        # Update priorities based on performance
        for expert in self.experts.values():
            if expert.performance.total_requests > 10:
                success_rate = expert.performance.success_rate
                if success_rate > 90:
                    expert.priority = min(100, expert.priority + 5)
                    optimizations["priorities_updated"] += 1
                elif success_rate < 50:
                    expert.priority = max(0, expert.priority - 5)
                    optimizations["priorities_updated"] += 1
        
        logger.info(f"Optimized expert pool: {optimizations}")
        return optimizations
