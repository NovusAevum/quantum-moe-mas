"""
MAS Orchestrator - Multi-Agent System Coordination and Task Distribution

This module implements the central orchestrator for the Multi-Agent System,
providing task distribution, conflict resolution, inter-agent communication,
and workflow management across all specialized agents.

Requirements addressed: 3.5, 4.1

Author: Wan Mohamad Hanis bin Wan Hassan
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from collections import defaultdict, deque

import structlog
from pydantic import BaseModel, Field, ConfigDict

from quantum_moe_mas.agents.base_agent import BaseAgent, AgentMessage, MessageType
from quantum_moe_mas.agents.communication import MessageBus, CommunicationProtocol, MessagePriority
from quantum_moe_mas.core.exceptions import QuantumMoEMASError
from quantum_moe_mas.core.logging_simple import get_logger


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


class ConflictType(Enum):
    """Types of conflicts that can occur."""
    RESOURCE_CONFLICT = "resource_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    PRIORITY_CONFLICT = "priority_conflict"
    CAPABILITY_CONFLICT = "capability_conflict"
    TIMING_CONFLICT = "timing_conflict"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition for agent execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    
    # Task requirements
    required_capabilities: List[str] = field(default_factory=list)
    required_resources: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[float] = None
    timeout: Optional[float] = None
    
    # Task data
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Assignment and execution
    assigned_agent_id: Optional[str] = None
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Dependencies and relationships
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    dependents: List[str] = field(default_factory=list)    # Task IDs that depend on this task
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    
    # Execution tracking
    attempts: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None
    execution_log: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Conflict:
    """Conflict between tasks or agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType = ConflictType.RESOURCE_CONFLICT
    description: str = ""
    
    # Involved entities
    task_ids: List[str] = field(default_factory=list)
    agent_ids: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    
    # Resolution
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""
    
    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: TaskPriority = TaskPriority.NORMAL


@dataclass
class Workflow:
    """Workflow definition containing multiple tasks."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: WorkflowStatus = WorkflowStatus.CREATED
    
    # Workflow structure
    tasks: List[str] = field(default_factory=list)  # Task IDs in execution order
    task_graph: Dict[str, List[str]] = field(default_factory=dict)  # Dependency graph
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    
    # Configuration
    parallel_execution: bool = True
    max_concurrent_tasks: int = 5
    timeout: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None


class AgentCapabilityMatch(BaseModel):
    """Agent capability matching result."""
    agent_id: str
    match_score: float = Field(ge=0.0, le=1.0)
    available_capabilities: List[str]
    missing_capabilities: List[str] = Field(default_factory=list)
    current_load: float = Field(ge=0.0, le=1.0, default=0.0)
    estimated_completion_time: Optional[float] = None


class TaskAssignment(BaseModel):
    """Task assignment result."""
    task_id: str
    agent_id: str
    assignment_score: float = Field(ge=0.0, le=1.0)
    estimated_start_time: datetime
    estimated_completion_time: datetime
    resource_allocation: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class ExecutionMetrics(BaseModel):
    """Execution metrics for monitoring."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    average_execution_time: float = 0.0
    success_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    throughput: float = 0.0  # Tasks per minute
    resource_utilization: Dict[str, float] = Field(default_factory=dict)
    agent_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MASOrchestrator:
    """
    Multi-Agent System Orchestrator.
    
    Central coordinator for the MAS that handles:
    - Task distribution and assignment to appropriate agents
    - Conflict detection and resolution
    - Inter-agent communication coordination
    - Workflow management and execution tracking
    - Resource allocation and optimization
    - Performance monitoring and analytics
    """
    
    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        max_concurrent_tasks: int = 50,
        task_timeout: float = 3600.0,  # 1 hour default
        conflict_resolution_timeout: float = 300.0,  # 5 minutes
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MAS Orchestrator.
        
        Args:
            message_bus: Message bus for inter-agent communication
            max_concurrent_tasks: Maximum concurrent tasks across all agents
            task_timeout: Default task timeout in seconds
            conflict_resolution_timeout: Timeout for conflict resolution
            config: Additional configuration options
        """
        self.message_bus = message_bus or MessageBus()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.conflict_resolution_timeout = conflict_resolution_timeout
        self.config = config or {}
        
        # Agent registry and management
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_load: Dict[str, float] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Workflow management
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Dict[str, Workflow] = {}
        
        # Conflict management
        self.conflicts: Dict[str, Conflict] = {}
        self.active_conflicts: Dict[str, Conflict] = {}
        
        # Resource management
        self.available_resources: Dict[str, Any] = {}
        self.resource_allocations: Dict[str, Dict[str, Any]] = {}tions: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring and metrics
        self.execution_metrics: ExecutionMetrics = ExecutionMetrics()
        self.metrics_history: List[ExecutionMetrics] = []
        
        # Background tasks
        self._orchestrator_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Configuration
        self.assignment_algorithm = config.get("assignment_algorithm", "capability_match") if config else "capability_match"
        self.load_balancing = config.get("load_balancing", True) if config else True
        self.auto_retry = config.get("auto_retry", True) if config else True
        self.conflict_resolution_strategy = config.get("conflict_resolution", "priority_based") if config else "priority_based"
        
        self._logger = get_logger("mas_orchestrator")
    
    async def start(self) -> None:
        """Start the MAS Orchestrator."""
        if self._running:
            return
        
        self._running = True
        
        # Start message bus
        await self.message_bus.start()
        
        # Start background tasks
        self._orchestrator_task = asyncio.create_task(self._orchestration_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self._logger.info("MAS Orchestrator started")
    
    async def stop(self) -> None:
        """Stop the MAS Orchestrator."""
        self._running = False
        
        # Cancel background tasks
        for task in [self._orchestrator_task, self._monitoring_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop message bus
        await self.message_bus.stop()
        
        self._logger.info("MAS Orchestrator stopped")
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: Agent instance to register
        """
        agent_id = agent.agent_id
        
        # Register with message bus
        await self.message_bus.register_agent(agent_id, agent)
        
        # Store agent information
        self.registered_agents[agent_id] = agent
        self.agent_capabilities[agent_id] = [cap.name for cap in agent.capabilities]
        self.agent_load[agent_id] = 0.0
        self.agent_performance[agent_id] = {
            "success_rate": 1.0,
            "average_execution_time": 0.0,
            "total_tasks": 0,
            "failed_tasks": 0
        }
        
        self._logger.info(
            "Agent registered",
            agent_id=agent_id,
            capabilities=self.agent_capabilities[agent_id]
        )
    
    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the orchestrator.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id not in self.registered_agents:
            self._logger.warning("Agent not registered", agent_id=agent_id)
            return
        
        # Cancel any active tasks assigned to this agent
        tasks_to_reassign = []
        for task in self.active_tasks.values():
            if task.assigned_agent_id == agent_id:
                tasks_to_reassign.append(task)
        
        for task in tasks_to_reassign:
            await self._reassign_task(task)
        
        # Unregister from message bus
        await self.message_bus.unregister_agent(agent_id)
        
        # Remove agent information
        del self.registered_agents[agent_id]
        del self.agent_capabilities[agent_id]
        del self.agent_load[agent_id]
        del self.agent_performance[agent_id]
        
        self._logger.info("Agent unregistered", agent_id=agent_id)
    
    async def submit_task(self, task: Task) -> str:
        """
        Submit a task for execution.
        
        Args:
            task: Task to execute
            
        Returns:
            Task ID
        """
        # Validate task
        await self._validate_task(task)
        
        # Store task
        self.tasks[task.id] = task
        
        # Add to queue
        self.task_queue.append(task.id)
        
        # Update metrics
        self.execution_metrics.total_tasks += 1
        
        self._logger.info(
            "Task submitted",
            task_id=task.id,
            task_type=task.task_type,
            priority=task.priority.value
        )
        
        return task.id
    
    async def submit_workflow(self, workflow: Workflow) -> str:
        """
        Submit a workflow for execution.
        
        Args:
            workflow: Workflow to execute
            
        Returns:
            Workflow ID
        """
        # Validate workflow
        await self._validate_workflow(workflow)
        
        # Store workflow
        self.workflows[workflow.id] = workflow
        
        # Add to active workflows
        self.active_workflows[workflow.id] = workflow
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now(timezone.utc)
        
        self._logger.info(
            "Workflow submitted",
            workflow_id=workflow.id,
            task_count=len(workflow.tasks)
        )
        
        return workflow.id
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if cancelled successfully
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        # Cancel task
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.now(timezone.utc)
        
        # Remove from active tasks
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        # Notify assigned agent if applicable
        if task.assigned_agent_id:
            await self._notify_agent_task_cancelled(task)
        
        self._logger.info("Task cancelled", task_id=task_id)
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if not found
        """
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    async def get_execution_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics."""
        # Update real-time metrics
        self.execution_metrics.active_tasks = len(self.active_tasks)
        self.execution_metrics.completed_tasks = len(self.completed_tasks)
        self.execution_metrics.failed_tasks = len(self.failed_tasks)
        
        if self.execution_metrics.total_tasks > 0:
            self.execution_metrics.success_rate = (
                self.execution_metrics.completed_tasks / self.execution_metrics.total_tasks
            )
        
        # Calculate average execution time
        if self.completed_tasks:
            total_time = 0.0
            count = 0
            for task in self.completed_tasks.values():
                if task.started_at and task.completed_at:
                    duration = (task.completed_at - task.started_at).total_seconds()
                    total_time += duration
                    count += 1
            
            if count > 0:
                self.execution_metrics.average_execution_time = total_time / count
        
        # Calculate throughput (tasks per minute)
        if len(self.metrics_history) > 0:
            time_window = 300  # 5 minutes
            recent_metrics = [
                m for m in self.metrics_history 
                if (datetime.now(timezone.utc) - m.timestamp).total_seconds() <= time_window
            ]
            
            if len(recent_metrics) > 1:
                time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
                task_diff = recent_metrics[-1].completed_tasks - recent_metrics[0].completed_tasks
                
                if time_diff > 0:
                    self.execution_metrics.throughput = (task_diff / time_diff) * 60  # per minute
        
        # Update agent performance
        self.execution_metrics.agent_performance = self.agent_performance.copy()
        
        # Update resource utilization
        self.execution_metrics.resource_utilization = {
            agent_id: load for agent_id, load in self.agent_load.items()
        }
        
        return self.execution_metrics    # P
rivate Implementation Methods
    
    async def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        while self._running:
            try:
                # Process task queue
                await self._process_task_queue()
                
                # Process active workflows
                await self._process_workflows()
                
                # Detect and resolve conflicts
                await self._detect_and_resolve_conflicts()
                
                # Update agent loads
                await self._update_agent_loads()
                
                # Sleep briefly to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self._logger.error("Error in orchestration loop", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _monitoring_loop(self) -> None:
        """Monitoring and metrics collection loop."""
        while self._running:
            try:
                # Update execution metrics
                current_metrics = await self.get_execution_metrics()
                
                # Store metrics history
                self.metrics_history.append(current_metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp >= cutoff_time
                ]
                
                # Check for performance issues
                await self._check_performance_issues()
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self._logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for expired tasks and resources."""
        while self._running:
            try:
                # Cleanup expired tasks
                await self._cleanup_expired_tasks()
                
                # Cleanup resolved conflicts
                await self._cleanup_resolved_conflicts()
                
                # Cleanup completed workflows
                await self._cleanup_completed_workflows()
                
                # Sleep for cleanup interval
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                self._logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(300)
    
    async def _process_task_queue(self) -> None:
        """Process tasks in the queue."""
        while self.task_queue and len(self.active_tasks) < self.max_concurrent_tasks:
            task_id = self.task_queue.popleft()
            task = self.tasks.get(task_id)
            
            if not task or task.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            if not await self._check_task_dependencies(task):
                # Re-queue if dependencies not met
                self.task_queue.append(task_id)
                break
            
            # Find suitable agent
            assignment = await self._assign_task_to_agent(task)
            
            if assignment:
                await self._execute_task_assignment(task, assignment)
            else:
                # No suitable agent available, re-queue
                self.task_queue.append(task_id)
                break
    
    async def _process_workflows(self) -> None:
        """Process active workflows."""
        for workflow_id, workflow in list(self.active_workflows.items()):
            try:
                await self._process_workflow(workflow)
            except Exception as e:
                self._logger.error(
                    "Error processing workflow",
                    workflow_id=workflow_id,
                    error=str(e)
                )
                workflow.status = WorkflowStatus.FAILED
                del self.active_workflows[workflow_id]
    
    async def _process_workflow(self, workflow: Workflow) -> None:
        """Process a single workflow."""
        if workflow.status != WorkflowStatus.RUNNING:
            return
        
        # Check if workflow is complete
        completed_tasks = 0
        failed_tasks = 0
        
        for task_id in workflow.tasks:
            task = self.tasks.get(task_id)
            if task:
                if task.status == TaskStatus.COMPLETED:
                    completed_tasks += 1
                elif task.status == TaskStatus.FAILED:
                    failed_tasks += 1
        
        # Update progress
        if workflow.tasks:
            workflow.progress = completed_tasks / len(workflow.tasks)
        
        # Check completion
        if completed_tasks == len(workflow.tasks):
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now(timezone.utc)
            del self.active_workflows[workflow.id]
            
            self._logger.info(
                "Workflow completed",
                workflow_id=workflow.id,
                duration=(workflow.completed_at - workflow.started_at).total_seconds()
            )
        
        elif failed_tasks > 0:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now(timezone.utc)
            del self.active_workflows[workflow.id]
            
            self._logger.error(
                "Workflow failed",
                workflow_id=workflow.id,
                failed_tasks=failed_tasks
            )
    
    async def _detect_and_resolve_conflicts(self) -> None:
        """Detect and resolve conflicts between tasks and agents."""
        # Detect resource conflicts
        await self._detect_resource_conflicts()
        
        # Detect dependency conflicts
        await self._detect_dependency_conflicts()
        
        # Detect priority conflicts
        await self._detect_priority_conflicts()
        
        # Resolve active conflicts
        for conflict in list(self.active_conflicts.values()):
            if not conflict.resolved:
                await self._resolve_conflict(conflict)
    
    async def _detect_resource_conflicts(self) -> None:
        """Detect resource conflicts between tasks."""
        resource_usage = defaultdict(list)
        
        # Map resource usage by active tasks
        for task in self.active_tasks.values():
            for resource_name in task.required_resources:
                resource_usage[resource_name].append(task.id)
        
        # Detect conflicts (multiple tasks using same exclusive resource)
        for resource_name, task_ids in resource_usage.items():
            if len(task_ids) > 1:
                # Check if resource allows concurrent access
                resource_info = self.available_resources.get(resource_name, {})
                if not resource_info.get("concurrent_access", False):
                    # Create conflict
                    conflict = Conflict(
                        conflict_type=ConflictType.RESOURCE_CONFLICT,
                        description=f"Multiple tasks competing for exclusive resource: {resource_name}",
                        task_ids=task_ids,
                        resources=[resource_name],
                        severity=TaskPriority.HIGH
                    )
                    
                    self.conflicts[conflict.id] = conflict
                    self.active_conflicts[conflict.id] = conflict
    
    async def _detect_dependency_conflicts(self) -> None:
        """Detect circular dependencies and other dependency conflicts."""
        # Build dependency graph
        dependency_graph = {}
        for task in self.tasks.values():
            dependency_graph[task.id] = task.dependencies
        
        # Detect circular dependencies using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check for cycles
        for task_id in dependency_graph:
            if task_id not in visited:
                if has_cycle(task_id):
                    # Create conflict for circular dependency
                    conflict = Conflict(
                        conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                        description=f"Circular dependency detected involving task: {task_id}",
                        task_ids=[task_id],
                        severity=TaskPriority.CRITICAL
                    )
                    
                    self.conflicts[conflict.id] = conflict
                    self.active_conflicts[conflict.id] = conflict
    
    async def _detect_priority_conflicts(self) -> None:
        """Detect priority conflicts between tasks."""
        # Group tasks by assigned agent
        agent_tasks = defaultdict(list)
        for task in self.active_tasks.values():
            if task.assigned_agent_id:
                agent_tasks[task.assigned_agent_id].append(task)
        
        # Check for priority conflicts within each agent
        for agent_id, tasks in agent_tasks.items():
            if len(tasks) > 1:
                # Sort by priority
                tasks.sort(key=lambda t: t.priority.value, reverse=True)
                
                # Check if high priority tasks are blocked by lower priority ones
                for i, high_task in enumerate(tasks[:-1]):
                    for low_task in tasks[i+1:]:
                        if (high_task.priority.value > low_task.priority.value and
                            low_task.status == TaskStatus.IN_PROGRESS and
                            high_task.status == TaskStatus.ASSIGNED):
                            
                            # Create priority conflict
                            conflict = Conflict(
                                conflict_type=ConflictType.PRIORITY_CONFLICT,
                                description=f"High priority task {high_task.id} blocked by lower priority task {low_task.id}",
                                task_ids=[high_task.id, low_task.id],
                                agent_ids=[agent_id],
                                severity=TaskPriority.HIGH
                            )
                            
                            self.conflicts[conflict.id] = conflict
                            self.active_conflicts[conflict.id] = conflict
    
    async def _resolve_conflict(self, conflict: Conflict) -> None:
        """Resolve a specific conflict."""
        try:
            if conflict.conflict_type == ConflictType.RESOURCE_CONFLICT:
                await self._resolve_resource_conflict(conflict)
            elif conflict.conflict_type == ConflictType.DEPENDENCY_CONFLICT:
                await self._resolve_dependency_conflict(conflict)
            elif conflict.conflict_type == ConflictType.PRIORITY_CONFLICT:
                await self._resolve_priority_conflict(conflict)
            
            # Mark as resolved
            conflict.resolved = True
            conflict.resolved_at = datetime.now(timezone.utc)
            
            if conflict.id in self.active_conflicts:
                del self.active_conflicts[conflict.id]
            
            self._logger.info(
                "Conflict resolved",
                conflict_id=conflict.id,
                conflict_type=conflict.conflict_type.value
            )
            
        except Exception as e:
            self._logger.error(
                "Failed to resolve conflict",
                conflict_id=conflict.id,
                error=str(e)
            )
    
    async def _resolve_resource_conflict(self, conflict: Conflict) -> None:
        """Resolve resource conflict."""
        if self.conflict_resolution_strategy == "priority_based":
            # Prioritize tasks by priority level
            tasks = [self.tasks[task_id] for task_id in conflict.task_ids if task_id in self.tasks]
            tasks.sort(key=lambda t: t.priority.value, reverse=True)
            
            # Keep highest priority task, reassign others
            for task in tasks[1:]:
                await self._reassign_task(task)
                
        elif self.conflict_resolution_strategy == "first_come_first_served":
            # Keep earliest assigned task
            tasks = [self.tasks[task_id] for task_id in conflict.task_ids if task_id in self.tasks]
            tasks.sort(key=lambda t: t.assigned_at or datetime.min.replace(tzinfo=timezone.utc))
            
            # Reassign all but the first
            for task in tasks[1:]:
                await self._reassign_task(task)
        
        conflict.resolution_strategy = self.conflict_resolution_strategy
        conflict.resolution_notes = f"Resolved using {self.conflict_resolution_strategy} strategy"
    
    async def _resolve_dependency_conflict(self, conflict: Conflict) -> None:
        """Resolve dependency conflict (circular dependencies)."""
        # For circular dependencies, we need to break the cycle
        # Strategy: Remove the dependency with the lowest priority difference
        
        tasks = [self.tasks[task_id] for task_id in conflict.task_ids if task_id in self.tasks]
        
        # Find the weakest dependency link to break
        min_priority_diff = float('inf')
        dependency_to_remove = None
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in conflict.task_ids:
                    dep_task = self.tasks.get(dep_id)
                    if dep_task:
                        priority_diff = abs(task.priority.value - dep_task.priority.value)
                        if priority_diff < min_priority_diff:
                            min_priority_diff = priority_diff
                            dependency_to_remove = (task.id, dep_id)
        
        # Remove the weakest dependency
        if dependency_to_remove:
            task_id, dep_id = dependency_to_remove
            task = self.tasks[task_id]
            task.dependencies.remove(dep_id)
            
            # Update dependent task
            dep_task = self.tasks[dep_id]
            if task_id in dep_task.dependents:
                dep_task.dependents.remove(task_id)
        
        conflict.resolution_strategy = "break_weakest_dependency"
        conflict.resolution_notes = f"Removed dependency: {dependency_to_remove}"
    
    async def _resolve_priority_conflict(self, conflict: Conflict) -> None:
        """Resolve priority conflict."""
        # Preempt lower priority task for higher priority one
        tasks = [self.tasks[task_id] for task_id in conflict.task_ids if task_id in self.tasks]
        tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        high_priority_task = tasks[0]
        low_priority_task = tasks[1]
        
        # Pause low priority task and start high priority one
        if low_priority_task.status == TaskStatus.IN_PROGRESS:
            await self._pause_task(low_priority_task)
        
        # Assign high priority task to the agent
        if high_priority_task.assigned_agent_id:
            await self._start_task_execution(high_priority_task)
        
        conflict.resolution_strategy = "priority_preemption"
        conflict.resolution_notes = f"Preempted task {low_priority_task.id} for {high_priority_task.id}"
    
    async def _check_task_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _assign_task_to_agent(self, task: Task) -> Optional[TaskAssignment]:
        """Assign task to the most suitable agent."""
        if self.assignment_algorithm == "capability_match":
            return await self._capability_based_assignment(task)
        elif self.assignment_algorithm == "load_balanced":
            return await self._load_balanced_assignment(task)
        elif self.assignment_algorithm == "performance_based":
            return await self._performance_based_assignment(task)
        else:
            return await self._capability_based_assignment(task)
    
    async def _capability_based_assignment(self, task: Task) -> Optional[TaskAssignment]:
        """Assign task based on agent capabilities."""
        best_match = None
        best_score = 0.0
        
        for agent_id, capabilities in self.agent_capabilities.items():
            # Check if agent has required capabilities
            match_score = await self._calculate_capability_match(task, agent_id, capabilities)
            
            if match_score > best_score:
                best_score = match_score
                best_match = agent_id
        
        if best_match and best_score > 0.5:  # Minimum match threshold
            estimated_start = datetime.now(timezone.utc)
            estimated_completion = estimated_start + timedelta(
                seconds=task.estimated_duration or 300  # Default 5 minutes
            )
            
            return TaskAssignment(
                task_id=task.id,
                agent_id=best_match,
                assignment_score=best_score,
                estimated_start_time=estimated_start,
                estimated_completion_time=estimated_completion,
                confidence=best_score
            )
        
        return None
    
    async def _load_balanced_assignment(self, task: Task) -> Optional[TaskAssignment]:
        """Assign task based on agent load balancing."""
        # First filter by capability
        capable_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            match_score = await self._calculate_capability_match(task, agent_id, capabilities)
            if match_score > 0.5:
                capable_agents.append((agent_id, match_score))
        
        if not capable_agents:
            return None
        
        # Select agent with lowest load
        best_agent = min(capable_agents, key=lambda x: self.agent_load.get(x[0], 0.0))
        agent_id, capability_score = best_agent
        
        # Calculate load-adjusted score
        load = self.agent_load.get(agent_id, 0.0)
        load_adjusted_score = capability_score * (1.0 - load)
        
        estimated_start = datetime.now(timezone.utc)
        estimated_completion = estimated_start + timedelta(
            seconds=(task.estimated_duration or 300) * (1.0 + load)  # Adjust for load
        )
        
        return TaskAssignment(
            task_id=task.id,
            agent_id=agent_id,
            assignment_score=load_adjusted_score,
            estimated_start_time=estimated_start,
            estimated_completion_time=estimated_completion,
            confidence=capability_score
        )
    
    async def _performance_based_assignment(self, task: Task) -> Optional[TaskAssignment]:
        """Assign task based on agent performance history."""
        # First filter by capability
        capable_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            match_score = await self._calculate_capability_match(task, agent_id, capabilities)
            if match_score > 0.5:
                capable_agents.append((agent_id, match_score))
        
        if not capable_agents:
            return None
        
        # Select agent with best performance
        best_agent = None
        best_performance_score = 0.0
        
        for agent_id, capability_score in capable_agents:
            performance = self.agent_performance.get(agent_id, {})
            success_rate = performance.get("success_rate", 0.5)
            avg_time = performance.get("average_execution_time", 300)
            
            # Calculate performance score (higher is better)
            time_factor = 1.0 / (1.0 + avg_time / 300)  # Normalize to 5 minutes
            performance_score = capability_score * success_rate * time_factor
            
            if performance_score > best_performance_score:
                best_performance_score = performance_score
                best_agent = agent_id
        
        if best_agent:
            performance = self.agent_performance.get(best_agent, {})
            estimated_duration = performance.get("average_execution_time", task.estimated_duration or 300)
            
            estimated_start = datetime.now(timezone.utc)
            estimated_completion = estimated_start + timedelta(seconds=estimated_duration)
            
            return TaskAssignment(
                task_id=task.id,
                agent_id=best_agent,
                assignment_score=best_performance_score,
                estimated_start_time=estimated_start,
                estimated_completion_time=estimated_completion,
                confidence=best_performance_score
            )
        
        return None
    
    async def _calculate_capability_match(self, task: Task, agent_id: str, capabilities: List[str]) -> float:
        """Calculate how well an agent's capabilities match task requirements."""
        if not task.required_capabilities:
            return 1.0  # No specific requirements
        
        matched_capabilities = 0
        for required_cap in task.required_capabilities:
            if required_cap in capabilities:
                matched_capabilities += 1
        
        match_ratio = matched_capabilities / len(task.required_capabilities)
        
        # Adjust for agent load
        load = self.agent_load.get(agent_id, 0.0)
        load_penalty = load * 0.5  # Reduce score by up to 50% based on load
        
        return max(0.0, match_ratio - load_penalty)
    
    async def _execute_task_assignment(self, task: Task, assignment: TaskAssignment) -> None:
        """Execute a task assignment."""
        # Update task
        task.assigned_agent_id = assignment.agent_id
        task.assigned_at = datetime.now(timezone.utc)
        task.status = TaskStatus.ASSIGNED
        task.updated_at = datetime.now(timezone.utc)
        
        # Add to active tasks
        self.active_tasks[task.id] = task
        
        # Update agent load
        self.agent_load[assignment.agent_id] = self.agent_load.get(assignment.agent_id, 0.0) + 0.1
        
        # Start task execution
        await self._start_task_execution(task)
        
        self._logger.info(
            "Task assigned",
            task_id=task.id,
            agent_id=assignment.agent_id,
            assignment_score=assignment.assignment_score
        )
    
    async def _start_task_execution(self, task: Task) -> None:
        """Start task execution on assigned agent."""
        if not task.assigned_agent_id:
            return
        
        # Create task message
        message = AgentMessage(
            sender_id="mas_orchestrator",
            recipient_id=task.assigned_agent_id,
            message_type=MessageType.TASK_REQUEST,
            payload={
                "task": {
                    "id": task.id,
                    "type": task.task_type,
                    "name": task.name,
                    "description": task.description,
                    "input_data": task.input_data,
                    "context": task.context,
                    "priority": task.priority.value,
                    "timeout": task.timeout
                }
            },
            correlation_id=task.id,
            priority=MessagePriority.HIGH if task.priority.value >= 3 else MessagePriority.NORMAL
        )
        
        # Send task to agent
        await self.message_bus.send_message(message, CommunicationProtocol.DIRECT)
        
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)
        task.updated_at = datetime.now(timezone.utc)
        task.execution_log.append(f"Task started on agent {task.assigned_agent_id}")
        
        self._logger.info(
            "Task execution started",
            task_id=task.id,
            agent_id=task.assigned_agent_id
        )
    
    async def _reassign_task(self, task: Task) -> None:
        """Reassign a task to a different agent."""
        # Remove from current agent
        if task.assigned_agent_id:
            current_load = self.agent_load.get(task.assigned_agent_id, 0.0)
            self.agent_load[task.assigned_agent_id] = max(0.0, current_load - 0.1)
        
        # Reset task assignment
        task.assigned_agent_id = None
        task.assigned_at = None
        task.status = TaskStatus.PENDING
        task.updated_at = datetime.now(timezone.utc)
        task.execution_log.append("Task reassigned due to conflict resolution")
        
        # Remove from active tasks
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        # Add back to queue
        self.task_queue.appendleft(task.id)  # Add to front for priority
        
        self._logger.info("Task reassigned", task_id=task.id)
    
    async def _pause_task(self, task: Task) -> None:
        """Pause a running task."""
        if task.status != TaskStatus.IN_PROGRESS:
            return
        
        # Send pause message to agent
        if task.assigned_agent_id:
            message = AgentMessage(
                sender_id="mas_orchestrator",
                recipient_id=task.assigned_agent_id,
                message_type=MessageType.TASK_PAUSE,
                payload={"task_id": task.id},
                correlation_id=task.id
            )
            
            await self.message_bus.send_message(message, CommunicationProtocol.DIRECT)
        
        # Update task status
        task.status = TaskStatus.PENDING  # Will be reassigned
        task.execution_log.append("Task paused for conflict resolution")
        
        self._logger.info("Task paused", task_id=task.id)
    
    async def _update_agent_loads(self) -> None:
        """Update agent load calculations."""
        for agent_id in self.registered_agents:
            # Count active tasks for this agent
            active_count = sum(
                1 for task in self.active_tasks.values()
                if task.assigned_agent_id == agent_id
            )
            
            # Calculate load (0.0 to 1.0)
            max_concurrent = 5  # Assume max 5 concurrent tasks per agent
            self.agent_load[agent_id] = min(1.0, active_count / max_concurrent)
    
    async def _check_performance_issues(self) -> None:
        """Check for performance issues and alerts."""
        # Check for stuck tasks
        current_time = datetime.now(timezone.utc)
        
        for task in self.active_tasks.values():
            if task.started_at:
                duration = (current_time - task.started_at).total_seconds()
                timeout = task.timeout or self.task_timeout
                
                if duration > timeout:
                    # Task has timed out
                    await self._handle_task_timeout(task)
        
        # Check agent performance
        for agent_id, performance in self.agent_performance.items():
            if performance.get("success_rate", 1.0) < 0.5:
                self._logger.warning(
                    "Agent performance degraded",
                    agent_id=agent_id,
                    success_rate=performance["success_rate"]
                )
    
    async def _handle_task_timeout(self, task: Task) -> None:
        """Handle task timeout."""
        task.status = TaskStatus.TIMEOUT
        task.completed_at = datetime.now(timezone.utc)
        task.error_message = "Task execution timed out"
        task.execution_log.append("Task timed out")
        
        # Remove from active tasks
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        # Add to failed tasks
        self.failed_tasks[task.id] = task
        
        # Update agent load
        if task.assigned_agent_id:
            current_load = self.agent_load.get(task.assigned_agent_id, 0.0)
            self.agent_load[task.assigned_agent_id] = max(0.0, current_load - 0.1)
        
        # Retry if configured
        if self.auto_retry and task.attempts < task.max_attempts:
            await self._retry_task(task)
        
        self._logger.warning("Task timed out", task_id=task.id)
    
    async def _retry_task(self, task: Task) -> None:
        """Retry a failed task."""
        task.attempts += 1
        task.status = TaskStatus.PENDING
        task.assigned_agent_id = None
        task.assigned_at = None
        task.started_at = None
        task.completed_at = None
        task.error_message = None
        task.updated_at = datetime.now(timezone.utc)
        task.execution_log.append(f"Retrying task (attempt {task.attempts})")
        
        # Remove from failed tasks
        if task.id in self.failed_tasks:
            del self.failed_tasks[task.id]
        
        # Add back to queue
        self.task_queue.appendleft(task.id)
        
        self._logger.info("Task queued for retry", task_id=task.id, attempt=task.attempts)
    
    async def _cleanup_expired_tasks(self) -> None:
        """Clean up expired and old tasks."""
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(hours=24)  # Keep tasks for 24 hours
        
        # Clean up completed tasks older than cutoff
        expired_completed = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in expired_completed:
            del self.completed_tasks[task_id]
            if task_id in self.tasks:
                del self.tasks[task_id]
        
        # Clean up failed tasks older than cutoff
        expired_failed = [
            task_id for task_id, task in self.failed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in expired_failed:
            del self.failed_tasks[task_id]
            if task_id in self.tasks:
                del self.tasks[task_id]
        
        if expired_completed or expired_failed:
            self._logger.info(
                "Cleaned up expired tasks",
                completed=len(expired_completed),
                failed=len(expired_failed)
            )
    
    async def _cleanup_resolved_conflicts(self) -> None:
        """Clean up resolved conflicts."""
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(hours=1)  # Keep resolved conflicts for 1 hour
        
        expired_conflicts = [
            conflict_id for conflict_id, conflict in self.conflicts.items()
            if conflict.resolved and conflict.resolved_at and conflict.resolved_at < cutoff_time
        ]
        
        for conflict_id in expired_conflicts:
            del self.conflicts[conflict_id]
        
        if expired_conflicts:
            self._logger.info("Cleaned up resolved conflicts", count=len(expired_conflicts))
    
    async def _cleanup_completed_workflows(self) -> None:
        """Clean up completed workflows."""
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(hours=24)  # Keep workflows for 24 hours
        
        expired_workflows = [
            workflow_id for workflow_id, workflow in self.workflows.items()
            if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and
                workflow.completed_at and workflow.completed_at < cutoff_time)
        ]
        
        for workflow_id in expired_workflows:
            del self.workflows[workflow_id]
        
        if expired_workflows:
            self._logger.info("Cleaned up completed workflows", count=len(expired_workflows))
    
    async def _validate_task(self, task: Task) -> None:
        """Validate task before submission."""
        if not task.name:
            raise ValueError("Task name is required")
        
        if not task.task_type:
            raise ValueError("Task type is required")
        
        # Validate dependencies exist
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                raise ValueError(f"Dependency task not found: {dep_id}")
        
        # Set defaults
        if task.estimated_duration is None:
            task.estimated_duration = 300.0  # 5 minutes default
        
        if task.timeout is None:
            task.timeout = self.task_timeout
    
    async def _validate_workflow(self, workflow: Workflow) -> None:
        """Validate workflow before submission."""
        if not workflow.name:
            raise ValueError("Workflow name is required")
        
        if not workflow.tasks:
            raise ValueError("Workflow must contain at least one task")
        
        # Validate all tasks exist
        for task_id in workflow.tasks:
            if task_id not in self.tasks:
                raise ValueError(f"Workflow task not found: {task_id}")
    
    async def _notify_agent_task_cancelled(self, task: Task) -> None:
        """Notify agent that a task has been cancelled."""
        if not task.assigned_agent_id:
            return
        
        message = AgentMessage(
            sender_id="mas_orchestrator",
            recipient_id=task.assigned_agent_id,
            message_type=MessageType.TASK_CANCEL,
            payload={"task_id": task.id},
            correlation_id=task.id
        )
        
        await self.message_bus.send_message(message, CommunicationProtocol.DIRECT)