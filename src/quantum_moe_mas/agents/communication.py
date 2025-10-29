"""
Agent Communication Protocols and Message Bus

This module provides the communication infrastructure for inter-agent
messaging, coordination, and distributed task execution.

Author: Wan Mohamad Hanis bin Wan Hassan
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable, Union
from collections import defaultdict, deque

import structlog
from pydantic import BaseModel, Field, ConfigDict

from quantum_moe_mas.agents.base_agent import AgentMessage, MessageType
from quantum_moe_mas.core.logging_simple import get_logger


class CommunicationProtocol(Enum):
    """Communication protocols supported by the message bus."""
    DIRECT = "direct"  # Direct agent-to-agent communication
    BROADCAST = "broadcast"  # One-to-many broadcasting
    PUBLISH_SUBSCRIBE = "publish_subscribe"  # Topic-based messaging
    REQUEST_RESPONSE = "request_response"  # Synchronous request-response
    QUEUE = "queue"  # Message queuing with persistence


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageStatus(Enum):
    """Message delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class MessageRoute:
    """Message routing information."""
    sender_id: str
    recipient_id: str
    protocol: CommunicationProtocol
    topic: Optional[str] = None
    routing_key: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MessageDeliveryReceipt:
    """Message delivery confirmation."""
    message_id: str
    recipient_id: str
    status: MessageStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None


class MessageFilter(BaseModel):
    """Filter for message subscription."""
    message_types: Optional[List[MessageType]] = None
    sender_ids: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    priority_threshold: MessagePriority = MessagePriority.LOW


class MessageBus:
    """
    Central message bus for inter-agent communication.
    
    Provides multiple communication patterns:
    - Direct messaging between agents
    - Broadcast messaging to multiple recipients
    - Publish-subscribe topic-based messaging
    - Request-response synchronous communication
    - Message queuing with persistence
    """
    
    def __init__(self, max_queue_size: int = 10000, message_ttl: int = 3600):
        """
        Initialize the message bus.
        
        Args:
            max_queue_size: Maximum number of messages in queue
            message_ttl: Default message time-to-live in seconds
        """
        self.max_queue_size = max_queue_size
        self.message_ttl = message_ttl
        
        # Agent registry
        self._agents: Dict[str, Any] = {}  # agent_id -> agent instance
        self._agent_queues: Dict[str, asyncio.Queue] = {}
        
        # Message routing
        self._message_history: deque = deque(maxlen=1000)
        self._pending_messages: Dict[str, AgentMessage] = {}
        self._delivery_receipts: Dict[str, List[MessageDeliveryReceipt]] = defaultdict(list)
        
        # Publish-subscribe
        self._topics: Dict[str, Set[str]] = defaultdict(set)  # topic -> subscriber_ids
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> topics
        self._message_filters: Dict[str, MessageFilter] = {}
        
        # Request-response
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._request_timeout = 30  # seconds
        
        # Monitoring
        self._logger = get_logger("message_bus")
        self._stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "active_agents": 0,
            "active_topics": 0
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the message bus and background tasks."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        
        self._logger.info("Message bus started")
    
    async def stop(self) -> None:
        """Stop the message bus and cleanup resources."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all queues and data
        self._agents.clear()
        self._agent_queues.clear()
        self._pending_messages.clear()
        self._topics.clear()
        self._subscriptions.clear()
        
        self._logger.info("Message bus stopped")
    
    async def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """Register an agent with the message bus."""
        if agent_id in self._agents:
            self._logger.warning(
                "Agent already registered, updating",
                agent_id=agent_id
            )
        
        self._agents[agent_id] = agent_instance
        self._agent_queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
        self._stats["active_agents"] = len(self._agents)
        
        self._logger.info("Agent registered", agent_id=agent_id)
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the message bus."""
        if agent_id not in self._agents:
            self._logger.warning("Agent not registered", agent_id=agent_id)
            return
        
        # Remove from all subscriptions
        for topic in list(self._subscriptions[agent_id]):
            await self.unsubscribe(agent_id, topic)
        
        # Clear agent data
        del self._agents[agent_id]
        del self._agent_queues[agent_id]
        if agent_id in self._subscriptions:
            del self._subscriptions[agent_id]
        if agent_id in self._message_filters:
            del self._message_filters[agent_id]
        
        self._stats["active_agents"] = len(self._agents)
        
        self._logger.info("Agent unregistered", agent_id=agent_id)
    
    async def send_message(
        self,
        message: AgentMessage,
        protocol: CommunicationProtocol = CommunicationProtocol.DIRECT
    ) -> str:
        """
        Send a message using the specified protocol.
        
        Args:
            message: Message to send
            protocol: Communication protocol to use
            
        Returns:
            Message ID for tracking
        """
        if not message.id:
            message.id = str(uuid.uuid4())
        
        # Set TTL if not specified
        if message.ttl is None:
            message.ttl = self.message_ttl
        
        self._stats["messages_sent"] += 1
        
        try:
            if protocol == CommunicationProtocol.DIRECT:
                await self._send_direct(message)
            elif protocol == CommunicationProtocol.BROADCAST:
                await self._send_broadcast(message)
            elif protocol == CommunicationProtocol.PUBLISH_SUBSCRIBE:
                await self._send_publish_subscribe(message)
            elif protocol == CommunicationProtocol.REQUEST_RESPONSE:
                await self._send_request_response(message)
            elif protocol == CommunicationProtocol.QUEUE:
                await self._send_queue(message)
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
            
            # Store in history
            self._message_history.append({
                "message_id": message.id,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "protocol": protocol.value,
                "timestamp": message.timestamp,
                "message_type": message.message_type.value
            })
            
            self._logger.debug(
                "Message sent",
                message_id=message.id,
                sender=message.sender_id,
                recipient=message.recipient_id,
                protocol=protocol.value
            )
            
            return message.id
            
        except Exception as e:
            self._stats["messages_failed"] += 1
            self._logger.error(
                "Failed to send message",
                message_id=message.id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _send_direct(self, message: AgentMessage) -> None:
        """Send message directly to a specific agent."""
        if message.recipient_id not in self._agents:
            raise ValueError(f"Recipient agent not found: {message.recipient_id}")
        
        queue = self._agent_queues[message.recipient_id]
        
        try:
            await asyncio.wait_for(queue.put(message), timeout=1.0)
            await self._record_delivery(message.id, message.recipient_id, MessageStatus.DELIVERED)
            self._stats["messages_delivered"] += 1
            
        except asyncio.TimeoutError:
            await self._record_delivery(
                message.id,
                message.recipient_id,
                MessageStatus.FAILED,
                "Queue full"
            )
            raise RuntimeError(f"Agent queue full: {message.recipient_id}")
    
    async def _send_broadcast(self, message: AgentMessage) -> None:
        """Broadcast message to all registered agents except sender."""
        recipients = [
            agent_id for agent_id in self._agents.keys()
            if agent_id != message.sender_id
        ]
        
        for recipient_id in recipients:
            try:
                message_copy = AgentMessage(
                    id=f"{message.id}_{recipient_id}",
                    sender_id=message.sender_id,
                    recipient_id=recipient_id,
                    message_type=message.message_type,
                    payload=message.payload.copy(),
                    timestamp=message.timestamp,
                    correlation_id=message.correlation_id,
                    priority=message.priority,
                    ttl=message.ttl
                )
                
                await self._send_direct(message_copy)
                
            except Exception as e:
                self._logger.error(
                    "Failed to broadcast to recipient",
                    recipient=recipient_id,
                    error=str(e)
                )
    
    async def _send_publish_subscribe(self, message: AgentMessage) -> None:
        """Send message to topic subscribers."""
        topic = message.payload.get("topic")
        if not topic:
            raise ValueError("Topic not specified in message payload")
        
        subscribers = self._topics.get(topic, set())
        
        for subscriber_id in subscribers:
            # Check message filter
            if not self._passes_filter(subscriber_id, message):
                continue
            
            try:
                message_copy = AgentMessage(
                    id=f"{message.id}_{subscriber_id}",
                    sender_id=message.sender_id,
                    recipient_id=subscriber_id,
                    message_type=message.message_type,
                    payload=message.payload.copy(),
                    timestamp=message.timestamp,
                    correlation_id=message.correlation_id,
                    priority=message.priority,
                    ttl=message.ttl
                )
                
                await self._send_direct(message_copy)
                
            except Exception as e:
                self._logger.error(
                    "Failed to send to subscriber",
                    subscriber=subscriber_id,
                    topic=topic,
                    error=str(e)
                )
    
    async def _send_request_response(self, message: AgentMessage) -> None:
        """Send request and wait for response."""
        # This is handled by the request_response method
        await self._send_direct(message)
    
    async def _send_queue(self, message: AgentMessage) -> None:
        """Send message to persistent queue."""
        # Store message for later delivery
        self._pending_messages[message.id] = message
        
        # Try immediate delivery
        try:
            await self._send_direct(message)
            # Remove from pending if successful
            if message.id in self._pending_messages:
                del self._pending_messages[message.id]
        except Exception:
            # Message remains in pending queue for retry
            pass
    
    async def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """
        Receive a message for the specified agent.
        
        Args:
            agent_id: Agent ID to receive message for
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            Received message or None if timeout
        """
        if agent_id not in self._agent_queues:
            raise ValueError(f"Agent not registered: {agent_id}")
        
        queue = self._agent_queues[agent_id]
        
        try:
            if timeout is None:
                message = await queue.get()
            else:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            
            # Check if message has expired
            if self._is_message_expired(message):
                await self._record_delivery(message.id, agent_id, MessageStatus.EXPIRED)
                return None
            
            return message
            
        except asyncio.TimeoutError:
            return None
    
    async def request_response(
        self,
        request: AgentMessage,
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for response.
        
        Args:
            request: Request message
            timeout: Response timeout in seconds
            
        Returns:
            Response message or None if timeout
        """
        if not request.correlation_id:
            request.correlation_id = str(uuid.uuid4())
        
        # Create future for response
        response_future = asyncio.Future()
        self._pending_requests[request.correlation_id] = response_future
        
        try:
            # Send request
            await self.send_message(request, CommunicationProtocol.REQUEST_RESPONSE)
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            self._logger.warning(
                "Request timeout",
                request_id=request.id,
                correlation_id=request.correlation_id
            )
            return None
            
        finally:
            # Cleanup
            if request.correlation_id in self._pending_requests:
                del self._pending_requests[request.correlation_id]
    
    async def send_response(self, response: AgentMessage) -> None:
        """Send response to a pending request."""
        if not response.correlation_id:
            raise ValueError("Response must have correlation_id")
        
        # Check if there's a pending request
        if response.correlation_id in self._pending_requests:
            future = self._pending_requests[response.correlation_id]
            if not future.done():
                future.set_result(response)
                return
        
        # If no pending request, send as regular message
        await self.send_message(response, CommunicationProtocol.DIRECT)
    
    async def subscribe(
        self,
        agent_id: str,
        topic: str,
        message_filter: Optional[MessageFilter] = None
    ) -> None:
        """Subscribe agent to a topic."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent not registered: {agent_id}")
        
        self._topics[topic].add(agent_id)
        self._subscriptions[agent_id].add(topic)
        
        if message_filter:
            self._message_filters[agent_id] = message_filter
        
        self._stats["active_topics"] = len(self._topics)
        
        self._logger.info(
            "Agent subscribed to topic",
            agent_id=agent_id,
            topic=topic
        )
    
    async def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe agent from a topic."""
        if topic in self._topics:
            self._topics[topic].discard(agent_id)
            if not self._topics[topic]:
                del self._topics[topic]
        
        if agent_id in self._subscriptions:
            self._subscriptions[agent_id].discard(topic)
            if not self._subscriptions[agent_id]:
                del self._subscriptions[agent_id]
        
        self._stats["active_topics"] = len(self._topics)
        
        self._logger.info(
            "Agent unsubscribed from topic",
            agent_id=agent_id,
            topic=topic
        )
    
    def _passes_filter(self, agent_id: str, message: AgentMessage) -> bool:
        """Check if message passes agent's filter."""
        message_filter = self._message_filters.get(agent_id)
        if not message_filter:
            return True
        
        # Check message type filter
        if (message_filter.message_types and 
            message.message_type not in message_filter.message_types):
            return False
        
        # Check sender filter
        if (message_filter.sender_ids and 
            message.sender_id not in message_filter.sender_ids):
            return False
        
        # Check priority filter
        if message.priority < message_filter.priority_threshold.value:
            return False
        
        return True
    
    def _is_message_expired(self, message: AgentMessage) -> bool:
        """Check if message has expired."""
        if message.ttl is None:
            return False
        
        age = (datetime.now(timezone.utc) - message.timestamp).total_seconds()
        return age > message.ttl
    
    async def _record_delivery(
        self,
        message_id: str,
        recipient_id: str,
        status: MessageStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Record message delivery status."""
        receipt = MessageDeliveryReceipt(
            message_id=message_id,
            recipient_id=recipient_id,
            status=status,
            error_message=error_message
        )
        
        self._delivery_receipts[message_id].append(receipt)
    
    async def _cleanup_expired_messages(self) -> None:
        """Background task to cleanup expired messages."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.now(timezone.utc)
                expired_messages = []
                
                # Check pending messages
                for message_id, message in self._pending_messages.items():
                    if self._is_message_expired(message):
                        expired_messages.append(message_id)
                
                # Remove expired messages
                for message_id in expired_messages:
                    del self._pending_messages[message_id]
                    await self._record_delivery(
                        message_id,
                        "system",
                        MessageStatus.EXPIRED
                    )
                
                if expired_messages:
                    self._logger.info(
                        "Cleaned up expired messages",
                        count=len(expired_messages)
                    )
                
            except Exception as e:
                self._logger.error(
                    "Error in message cleanup",
                    error=str(e)
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            **self._stats,
            "pending_messages": len(self._pending_messages),
            "pending_requests": len(self._pending_requests),
            "message_history_size": len(self._message_history),
            "total_subscriptions": sum(len(subs) for subs in self._subscriptions.values())
        }
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered agent."""
        if agent_id not in self._agents:
            return None
        
        queue = self._agent_queues[agent_id]
        subscriptions = list(self._subscriptions.get(agent_id, set()))
        
        return {
            "agent_id": agent_id,
            "queue_size": queue.qsize(),
            "max_queue_size": queue.maxsize,
            "subscriptions": subscriptions,
            "has_message_filter": agent_id in self._message_filters
        }
    
    def get_topic_info(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get information about a topic."""
        if topic not in self._topics:
            return None
        
        subscribers = list(self._topics[topic])
        
        return {
            "topic": topic,
            "subscriber_count": len(subscribers),
            "subscribers": subscribers
        }