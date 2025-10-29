"""
Base Agent Architecture for Quantum MoE MAS

This module provides the abstract base class and core functionality for all
domain-specialized agents in the system. It includes state management,
communication protocols, and performance monitoring.

Author: Wan Mohamad Hanis bin Wan Hassan
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from contextlib import asynccontextmanager

import structlog
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, Gauge

# Import core components
from quantum_moe_mas.core.logging_simple import get_logger


class AgentState(Enum):
    """Agent operational states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MessageType(Enum):
    """Types of messages agents can exchange."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    HEALTH_CHECK = "health_check"
    COORDINATION = "coordination"
    ERROR_REPORT = "error_report"
    SHUTDOWN = "shutdown"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.TASK_REQUEST
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    priority: int = 0  # Higher number = higher priority
    ttl: Optional[int] = None  # Time to live in seconds


class AgentContext(BaseModel):
    """Agent execution context for state preservation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    agent_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    task_context: Dict[str, Any] = Field(default_factory=dict)
    performance_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentCapability(BaseModel):
    """Represents an agent's capability."""
    name: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class BaseAgent(ABC):
    """
    Abstract base class for all domain-specialized agents.
    
    Provides common functionality including:
    - State management and context preservation
    - Inter-agent communication protocols
    - Performance monitoring and health checks
    - Error handling and recovery
    - Lifecycle management
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the base agent."""
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = {cap.name: cap for cap in capabilities}
        self.config = config or {}
        
        # State management
        self._state = AgentState.INITIALIZING
        self._context = AgentContext(agent_id=agent_id)
        self._metrics = AgentMetrics()
        
        # Communication
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: List[str] = []
        
        # Monitoring
        self._logger = get_logger(f"agent.{agent_id}")
        self._start_time = time.time()
        self._health_check_interval = 30  # seconds
        self._last_health_check = time.time()
        
        # Prometheus metrics
        self._task_counter = Counter(
            'agent_tasks_total',
            'Total number of tasks processed',
            ['agent_id', 'status']
        )
        self._response_time_histogram = Histogram(
            'agent_response_time_seconds',
            'Agent response time in seconds',
            ['agent_id']
        )
        self._active_tasks_gauge = Gauge(
            'agent_active_tasks',
            'Number of currently active tasks',
            ['agent_id']
        )
        
        # Error handling
        self._error_count = 0
        self._max_errors = 10
        self._error_reset_time = 300  # 5 minutes
        self._last_error_time = 0
        
        self._logger.info(
            "Agent initialized",
            agent_id=agent_id,
            name=name,
            capabilities=list(self.capabilities.keys())
        )
    
    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state
    
    @property
    def context(self) -> AgentContext:
        """Get agent context."""
        return self._context
    
    @property
    def metrics(self) -> AgentMetrics:
        """Get agent metrics."""
        self._update_metrics()
        return self._metrics
    
    def _update_metrics(self) -> None:
        """Update internal metrics."""
        current_time = time.time()
        self._metrics.uptime_seconds = current_time - self._start_time
        self._metrics.last_activity = datetime.now(timezone.utc)
        
        # Update average response time if we have task data
        if self._metrics.total_tasks > 0:
            # This would be calculated from actual response times
            pass
    
    async def initialize(self) -> None:
        """Initialize the agent and start background tasks."""
        try:
            self._state = AgentState.INITIALIZING
            self._logger.info("Initializing agent", agent_id=self.agent_id)
            
            # Initialize agent-specific components
            await self._initialize_agent()
            
            # Start background tasks
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._health_monitor())
            
            self._state = AgentState.IDLE
            self._logger.info("Agent initialized successfully", agent_id=self.agent_id)
            
        except Exception as e:
            self._state = AgentState.ERROR
            self._logger.error("Agent initialization failed", error=str(e), agent_id=self.agent_id)
            raise
    
    @abstractmethod
    async def _initialize_agent(self) -> None:
        """Initialize agent-specific components. Must be implemented by subclasses."""
        pass
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        self._state = AgentState.SHUTDOWN
        self._logger.info("Shutting down agent", agent_id=self.agent_id)
        
        # Send shutdown message to subscribers
        shutdown_msg = AgentMessage(
            sender_id=self.agent_id,
            message_type=MessageType.SHUTDOWN,
            payload={"reason": "graceful_shutdown"}
        )
        await self._broadcast_message(shutdown_msg)
        
        # Cleanup agent-specific resources
        await self._cleanup_agent()
        
        self._logger.info("Agent shutdown complete", agent_id=self.agent_id)
    
    @abstractmethod
    async def _cleanup_agent(self) -> None:
        """Cleanup agent-specific resources. Must be implemented by subclasses."""
        pass
    
    async def process_task(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task with performance monitoring and error handling.
        
        Args:
            task: Task data to process
            context: Optional additional context
            
        Returns:
            Task result
        """
        task_id = task.get('id', str(uuid.uuid4()))
        start_time = time.time()
        
        self._logger.info(
            "Processing task",
            agent_id=self.agent_id,
            task_id=task_id,
            task_type=task.get('type', 'unknown')
        )
        
        try:
            self._state = AgentState.PROCESSING
            self._active_tasks_gauge.labels(agent_id=self.agent_id).inc()
            
            # Update context
            if context:
                self._context.task_context.update(context)
            
            # Process the task
            result = await self._process_task_impl(task, context)
            
            # Record success
            self._task_counter.labels(agent_id=self.agent_id, status='success').inc()
            self._metrics.successful_tasks += 1
            
            processing_time = time.time() - start_time
            self._response_time_histogram.labels(agent_id=self.agent_id).observe(processing_time)
            
            self._logger.info(
                "Task completed successfully",
                agent_id=self.agent_id,
                task_id=task_id,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            # Record failure
            self._task_counter.labels(agent_id=self.agent_id, status='failure').inc()
            self._metrics.failed_tasks += 1
            self._error_count += 1
            self._last_error_time = time.time()
            
            self._logger.error(
                "Task processing failed",
                agent_id=self.agent_id,
                task_id=task_id,
                error=str(e),
                exc_info=True
            )
            
            # Check if we need to enter error state
            if self._error_count >= self._max_errors:
                self._state = AgentState.ERROR
                self._logger.critical(
                    "Agent entering error state due to excessive failures",
                    agent_id=self.agent_id,
                    error_count=self._error_count
                )
            
            raise
            
        finally:
            self._active_tasks_gauge.labels(agent_id=self.agent_id).dec()
            self._metrics.total_tasks += 1
            self._state = AgentState.IDLE
    
    @abstractmethod
    async def _process_task_impl(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Implement task processing logic. Must be implemented by subclasses.
        
        Args:
            task: Task data to process
            context: Optional additional context
            
        Returns:
            Task result
        """
        pass
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent."""
        message.sender_id = self.agent_id
        self._logger.debug(
            "Sending message",
            agent_id=self.agent_id,
            recipient=message.recipient_id,
            message_type=message.message_type.value
        )
        
        # This would integrate with the message bus
        # For now, we'll just log it
        await self._handle_outgoing_message(message)
    
    async def _handle_outgoing_message(self, message: AgentMessage) -> None:
        """Handle outgoing message. Can be overridden by subclasses."""
        pass
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        await self._message_queue.put(message)
    
    async def _message_processor(self) -> None:
        """Background task to process incoming messages."""
        while self._state != AgentState.SHUTDOWN:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error(
                    "Error processing message",
                    agent_id=self.agent_id,
                    error=str(e)
                )
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle incoming message based on type."""
        handler = self._message_handlers.get(message.message_type)
        
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self._logger.error(
                    "Message handler failed",
                    agent_id=self.agent_id,
                    message_type=message.message_type.value,
                    error=str(e)
                )
        else:
            self._logger.warning(
                "No handler for message type",
                agent_id=self.agent_id,
                message_type=message.message_type.value
            )
    
    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[AgentMessage], Awaitable[None]]
    ) -> None:
        """Register a handler for a specific message type."""
        self._message_handlers[message_type] = handler
        self._logger.debug(
            "Registered message handler",
            agent_id=self.agent_id,
            message_type=message_type.value
        )
    
    async def _broadcast_message(self, message: AgentMessage) -> None:
        """Broadcast message to all subscribers."""
        for subscriber_id in self._subscribers:
            message.recipient_id = subscriber_id
            await self.send_message(message)
    
    async def _health_monitor(self) -> None:
        """Background task for health monitoring."""
        while self._state != AgentState.SHUTDOWN:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_check()
                
            except Exception as e:
                self._logger.error(
                    "Health check failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
    
    async def _perform_health_check(self) -> None:
        """Perform health check and update status."""
        current_time = time.time()
        
        # Reset error count if enough time has passed
        if (current_time - self._last_error_time) > self._error_reset_time:
            self._error_count = 0
            if self._state == AgentState.ERROR:
                self._state = AgentState.IDLE
                self._logger.info(
                    "Agent recovered from error state",
                    agent_id=self.agent_id
                )
        
        # Update last health check time
        self._last_health_check = current_time
        
        # Perform agent-specific health checks
        await self._agent_health_check()
    
    async def _agent_health_check(self) -> None:
        """Perform agent-specific health checks. Can be overridden by subclasses."""
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        current_time = time.time()
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self._state.value,
            "uptime_seconds": current_time - self._start_time,
            "last_health_check": self._last_health_check,
            "error_count": self._error_count,
            "metrics": self.metrics.model_dump(),
            "capabilities": [cap.name for cap in self.capabilities.values() if cap.enabled],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Get a specific capability by name."""
        return self.capabilities.get(name)
    
    def enable_capability(self, name: str) -> bool:
        """Enable a capability."""
        capability = self.capabilities.get(name)
        if capability:
            capability.enabled = True
            self._logger.info(
                "Capability enabled",
                agent_id=self.agent_id,
                capability=name
            )
            return True
        return False
    
    def disable_capability(self, name: str) -> bool:
        """Disable a capability."""
        capability = self.capabilities.get(name)
        if capability:
            capability.enabled = False
            self._logger.info(
                "Capability disabled",
                agent_id=self.agent_id,
                capability=name
            )
            return True
        return False
    
    @asynccontextmanager
    async def task_context(self, task_data: Dict[str, Any]):
        """Context manager for task execution with automatic cleanup."""
        task_id = task_data.get('id', str(uuid.uuid4()))
        
        # Setup
        self._context.task_context.update(task_data)
        self._logger.debug("Task context setup", agent_id=self.agent_id, task_id=task_id)
        
        try:
            yield self._context
        finally:
            # Cleanup
            self._context.task_context.clear()
            self._context.updated_at = datetime.now(timezone.utc)
            self._logger.debug("Task context cleanup", agent_id=self.agent_id, task_id=task_id)
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"id='{self.agent_id}', "
            f"name='{self.name}', "
            f"state='{self._state.value}', "
            f"capabilities={len(self.capabilities)}"
            f")"
        )