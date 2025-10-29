"""
Comprehensive API Usage Tracking System.

This module provides detailed tracking of API usage across all integrated
providers with real-time analytics, cost calculation, and performance monitoring.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
from pathlib import Path

from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class UsageMetricType(Enum):
    """Types of usage metrics to track."""
    
    REQUEST_COUNT = "request_count"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"


class TimeWindow(Enum):
    """Time windows for aggregating metrics."""
    
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class UsageEvent:
    """Individual usage event record."""
    
    timestamp: datetime
    provider: str
    endpoint: str
    method: str
    success: bool
    status_code: Optional[int]
    latency_ms: float
    tokens_used: int
    cost: float
    request_size_bytes: int
    response_size_bytes: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageEvent':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class UsageMetrics:
    """Aggregated usage metrics for a time period."""
    
    provider: str
    time_window: TimeWindow
    start_time: datetime
    end_time: datetime
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    
    # Usage metrics
    total_tokens: int = 0
    total_cost: float = 0.0
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    
    # Error tracking
    error_codes: Dict[int, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return 100.0 - self.success_rate
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests
    
    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        duration_seconds = (self.end_time - self.start_time).total_seconds()
        if duration_seconds == 0:
            return 0.0
        return self.total_requests / duration_seconds
    
    @property
    def cost_per_request(self) -> float:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests
    
    @property
    def tokens_per_request(self) -> float:
        """Calculate average tokens per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_tokens / self.total_requests


class UsageTracker:
    """
    Comprehensive API usage tracking system.
    
    Provides real-time tracking of API usage with:
    - Event-level tracking for all API calls
    - Aggregated metrics by time windows
    - Performance analytics and trends
    - Cost tracking and optimization insights
    - SQLite storage for persistence
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_events_in_memory: int = 10000,
        auto_aggregate_interval_minutes: int = 5
    ) -> None:
        """
        Initialize usage tracker.
        
        Args:
            storage_path: Path to SQLite database file
            max_events_in_memory: Maximum events to keep in memory
            auto_aggregate_interval_minutes: Interval for auto-aggregation
        """
        self.storage_path = storage_path or os.path.expanduser("~/.quantum_moe_mas/usage_tracking.db")
        self.max_events_in_memory = max_events_in_memory
        self.auto_aggregate_interval = auto_aggregate_interval_minutes
        
        # In-memory storage for recent events
        self.recent_events: List[UsageEvent] = []
        self.aggregated_metrics: Dict[str, Dict[TimeWindow, List[UsageMetrics]]] = {}
        
        # Real-time counters
        self.realtime_counters: Dict[str, Dict[str, Union[int, float]]] = {}
        
        # Background tasks
        self.is_running = False
        self._aggregation_task: Optional[asyncio.Task] = None
        
        logger.info(
            "Initialized UsageTracker",
            storage_path=self.storage_path,
            max_events=max_events_in_memory
        )
    
    async def initialize(self) -> None:
        """Initialize the usage tracker and database."""
        try:
            # Create storage directory
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Initialize database
            await self._initialize_database()
            
            # Load recent data
            await self._load_recent_data()
            
            # Start background aggregation
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())
            
            self.is_running = True
            logger.info("UsageTracker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize UsageTracker: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the usage tracker."""
        self.is_running = False
        
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        
        # Save remaining events
        await self._flush_events_to_database()
        
        logger.info("UsageTracker shutdown complete")
    
    async def _initialize_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    status_code INTEGER,
                    latency_ms REAL NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    request_size_bytes INTEGER NOT NULL,
                    response_size_bytes INTEGER NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT
                )
            """)
            
            # Aggregated metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    time_window TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    total_requests INTEGER NOT NULL,
                    successful_requests INTEGER NOT NULL,
                    failed_requests INTEGER NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    min_latency_ms REAL NOT NULL,
                    max_latency_ms REAL NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    total_cost REAL NOT NULL,
                    total_request_bytes INTEGER NOT NULL,
                    total_response_bytes INTEGER NOT NULL,
                    error_codes TEXT
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON usage_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_provider ON usage_events(provider)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_provider_window ON usage_metrics(provider, time_window)")
            
            conn.commit()
            logger.info("Database schema initialized")
            
        finally:
            conn.close()
    
    async def track_usage(
        self,
        provider: str,
        endpoint: str,
        method: str,
        success: bool,
        status_code: Optional[int],
        latency_ms: float,
        tokens_used: int,
        cost: float,
        request_size_bytes: int = 0,
        response_size_bytes: int = 0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a usage event.
        
        Args:
            provider: API provider name
            endpoint: API endpoint
            method: HTTP method
            success: Whether request was successful
            status_code: HTTP status code
            latency_ms: Request latency in milliseconds
            tokens_used: Number of tokens used
            cost: Cost of the request
            request_size_bytes: Size of request in bytes
            response_size_bytes: Size of response in bytes
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional metadata
        """
        event = UsageEvent(
            timestamp=datetime.utcnow(),
            provider=provider,
            endpoint=endpoint,
            method=method,
            success=success,
            status_code=status_code,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost=cost,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # Add to recent events
        self.recent_events.append(event)
        
        # Update real-time counters
        await self._update_realtime_counters(event)
        
        # Flush to database if memory limit reached
        if len(self.recent_events) >= self.max_events_in_memory:
            await self._flush_events_to_database()
        
        logger.debug(f"Tracked usage event for {provider}: {success}, {latency_ms}ms, {tokens_used} tokens")
    
    async def _update_realtime_counters(self, event: UsageEvent) -> None:
        """Update real-time counters for an event."""
        if event.provider not in self.realtime_counters:
            self.realtime_counters[event.provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_latency_ms": 0.0,
                "last_request_time": None,
            }
        
        counters = self.realtime_counters[event.provider]
        counters["total_requests"] += 1
        
        if event.success:
            counters["successful_requests"] += 1
        else:
            counters["failed_requests"] += 1
        
        counters["total_tokens"] += event.tokens_used
        counters["total_cost"] += event.cost
        counters["total_latency_ms"] += event.latency_ms
        counters["last_request_time"] = event.timestamp.isoformat()
    
    async def _flush_events_to_database(self) -> None:
        """Flush recent events to database."""
        if not self.recent_events:
            return
        
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            for event in self.recent_events:
                cursor.execute("""
                    INSERT INTO usage_events (
                        timestamp, provider, endpoint, method, success, status_code,
                        latency_ms, tokens_used, cost, request_size_bytes, response_size_bytes,
                        user_id, session_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.timestamp.isoformat(),
                    event.provider,
                    event.endpoint,
                    event.method,
                    event.success,
                    event.status_code,
                    event.latency_ms,
                    event.tokens_used,
                    event.cost,
                    event.request_size_bytes,
                    event.response_size_bytes,
                    event.user_id,
                    event.session_id,
                    json.dumps(event.metadata)
                ))
            
            conn.commit()
            logger.debug(f"Flushed {len(self.recent_events)} events to database")
            
            # Clear recent events
            self.recent_events.clear()
            
        finally:
            conn.close()
    
    async def get_usage_metrics(
        self,
        provider: Optional[str] = None,
        time_window: TimeWindow = TimeWindow.HOUR,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[UsageMetrics]:
        """
        Get aggregated usage metrics.
        
        Args:
            provider: Optional provider filter
            time_window: Time window for aggregation
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of results
        
        Returns:
            List of usage metrics
        """
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=1)
        
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT * FROM usage_metrics
                WHERE time_window = ? AND start_time >= ? AND end_time <= ?
            """
            params = [time_window.value, start_time.isoformat(), end_time.isoformat()]
            
            if provider:
                query += " AND provider = ?"
                params.append(provider)
            
            query += " ORDER BY start_time DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to UsageMetrics objects
            metrics = []
            for row in rows:
                error_codes = json.loads(row[15]) if row[15] else {}
                
                metric = UsageMetrics(
                    provider=row[1],
                    time_window=TimeWindow(row[2]),
                    start_time=datetime.fromisoformat(row[3]),
                    end_time=datetime.fromisoformat(row[4]),
                    total_requests=row[5],
                    successful_requests=row[6],
                    failed_requests=row[7],
                    total_latency_ms=row[8],
                    min_latency_ms=row[9],
                    max_latency_ms=row[10],
                    total_tokens=row[11],
                    total_cost=row[12],
                    total_request_bytes=row[13],
                    total_response_bytes=row[14],
                    error_codes=error_codes
                )
                metrics.append(metric)
            
            return metrics
            
        finally:
            conn.close()
    
    async def get_realtime_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get real-time usage statistics.
        
        Args:
            provider: Optional provider filter
        
        Returns:
            Dictionary with real-time statistics
        """
        if provider:
            if provider in self.realtime_counters:
                stats = self.realtime_counters[provider].copy()
                
                # Calculate derived metrics
                if stats["total_requests"] > 0:
                    stats["success_rate"] = (stats["successful_requests"] / stats["total_requests"]) * 100
                    stats["average_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
                    stats["average_cost_per_request"] = stats["total_cost"] / stats["total_requests"]
                else:
                    stats["success_rate"] = 0.0
                    stats["average_latency_ms"] = 0.0
                    stats["average_cost_per_request"] = 0.0
                
                return {provider: stats}
            else:
                return {provider: {}}
        else:
            # Return all providers
            all_stats = {}
            for prov, counters in self.realtime_counters.items():
                stats = counters.copy()
                
                if stats["total_requests"] > 0:
                    stats["success_rate"] = (stats["successful_requests"] / stats["total_requests"]) * 100
                    stats["average_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
                    stats["average_cost_per_request"] = stats["total_cost"] / stats["total_requests"]
                else:
                    stats["success_rate"] = 0.0
                    stats["average_latency_ms"] = 0.0
                    stats["average_cost_per_request"] = 0.0
                
                all_stats[prov] = stats
            
            return all_stats
    
    async def get_usage_trends(
        self,
        provider: str,
        metric_type: UsageMetricType,
        time_window: TimeWindow = TimeWindow.HOUR,
        periods: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get usage trends for a specific metric.
        
        Args:
            provider: API provider name
            metric_type: Type of metric to analyze
            time_window: Time window for aggregation
            periods: Number of periods to include
        
        Returns:
            List of trend data points
        """
        end_time = datetime.utcnow()
        
        # Calculate time delta based on window
        if time_window == TimeWindow.MINUTE:
            delta = timedelta(minutes=1)
        elif time_window == TimeWindow.HOUR:
            delta = timedelta(hours=1)
        elif time_window == TimeWindow.DAY:
            delta = timedelta(days=1)
        else:
            delta = timedelta(hours=1)  # Default
        
        start_time = end_time - (delta * periods)
        
        metrics = await self.get_usage_metrics(
            provider=provider,
            time_window=time_window,
            start_time=start_time,
            end_time=end_time,
            limit=periods
        )
        
        # Extract trend data
        trend_data = []
        for metric in reversed(metrics):  # Reverse to get chronological order
            if metric_type == UsageMetricType.REQUEST_COUNT:
                value = metric.total_requests
            elif metric_type == UsageMetricType.TOKEN_USAGE:
                value = metric.total_tokens
            elif metric_type == UsageMetricType.COST:
                value = metric.total_cost
            elif metric_type == UsageMetricType.LATENCY:
                value = metric.average_latency_ms
            elif metric_type == UsageMetricType.ERROR_RATE:
                value = metric.error_rate
            elif metric_type == UsageMetricType.SUCCESS_RATE:
                value = metric.success_rate
            elif metric_type == UsageMetricType.THROUGHPUT:
                value = metric.requests_per_second
            else:
                value = 0
            
            trend_data.append({
                "timestamp": metric.start_time.isoformat(),
                "value": value,
                "period_start": metric.start_time.isoformat(),
                "period_end": metric.end_time.isoformat()
            })
        
        return trend_data
    
    async def _load_recent_data(self) -> None:
        """Load recent data from database."""
        # Load recent aggregated metrics
        recent_time = datetime.utcnow() - timedelta(hours=24)
        
        for time_window in TimeWindow:
            metrics = await self.get_usage_metrics(
                time_window=time_window,
                start_time=recent_time,
                limit=1000
            )
            
            for metric in metrics:
                if metric.provider not in self.aggregated_metrics:
                    self.aggregated_metrics[metric.provider] = {}
                if time_window not in self.aggregated_metrics[metric.provider]:
                    self.aggregated_metrics[metric.provider][time_window] = []
                
                self.aggregated_metrics[metric.provider][time_window].append(metric)
        
        logger.info("Loaded recent usage data from database")
    
    async def _aggregation_loop(self) -> None:
        """Background loop for aggregating metrics."""
        while self.is_running:
            try:
                await self._aggregate_recent_events()
                await asyncio.sleep(self.auto_aggregate_interval * 60)
                
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _aggregate_recent_events(self) -> None:
        """Aggregate recent events into metrics."""
        # Flush events to database first
        await self._flush_events_to_database()
        
        # Aggregate for different time windows
        for time_window in [TimeWindow.MINUTE, TimeWindow.HOUR, TimeWindow.DAY]:
            await self._aggregate_for_time_window(time_window)
    
    async def _aggregate_for_time_window(self, time_window: TimeWindow) -> None:
        """Aggregate events for a specific time window."""
        # Calculate time boundaries
        now = datetime.utcnow()
        
        if time_window == TimeWindow.MINUTE:
            # Aggregate last complete minute
            end_time = now.replace(second=0, microsecond=0)
            start_time = end_time - timedelta(minutes=1)
        elif time_window == TimeWindow.HOUR:
            # Aggregate last complete hour
            end_time = now.replace(minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(hours=1)
        elif time_window == TimeWindow.DAY:
            # Aggregate last complete day
            end_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(days=1)
        else:
            return
        
        # Get events for this time window
        conn = sqlite3.connect(self.storage_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT provider, COUNT(*) as total_requests,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests,
                       SUM(latency_ms) as total_latency_ms,
                       MIN(latency_ms) as min_latency_ms,
                       MAX(latency_ms) as max_latency_ms,
                       SUM(tokens_used) as total_tokens,
                       SUM(cost) as total_cost,
                       SUM(request_size_bytes) as total_request_bytes,
                       SUM(response_size_bytes) as total_response_bytes
                FROM usage_events
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY provider
            """, (start_time.isoformat(), end_time.isoformat()))
            
            rows = cursor.fetchall()
            
            # Create metrics for each provider
            for row in rows:
                provider = row[0]
                
                # Check if this metric already exists
                cursor.execute("""
                    SELECT COUNT(*) FROM usage_metrics
                    WHERE provider = ? AND time_window = ? AND start_time = ?
                """, (provider, time_window.value, start_time.isoformat()))
                
                if cursor.fetchone()[0] > 0:
                    continue  # Already aggregated
                
                # Get error codes for this period
                cursor.execute("""
                    SELECT status_code, COUNT(*) FROM usage_events
                    WHERE provider = ? AND timestamp >= ? AND timestamp < ? AND success = 0
                    GROUP BY status_code
                """, (provider, start_time.isoformat(), end_time.isoformat()))
                
                error_codes = dict(cursor.fetchall())
                
                # Insert aggregated metric
                cursor.execute("""
                    INSERT INTO usage_metrics (
                        provider, time_window, start_time, end_time,
                        total_requests, successful_requests, failed_requests,
                        total_latency_ms, min_latency_ms, max_latency_ms,
                        total_tokens, total_cost, total_request_bytes, total_response_bytes,
                        error_codes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    provider, time_window.value, start_time.isoformat(), end_time.isoformat(),
                    row[1], row[2], row[3], row[4], row[5], row[6],
                    row[7], row[8], row[9], row[10], json.dumps(error_codes)
                ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_tracker_status(self) -> Dict[str, Any]:
        """
        Get comprehensive tracker status.
        
        Returns:
            Dictionary with tracker status and statistics
        """
        return {
            "is_running": self.is_running,
            "recent_events_count": len(self.recent_events),
            "tracked_providers": len(self.realtime_counters),
            "storage_path": self.storage_path,
            "max_events_in_memory": self.max_events_in_memory,
            "auto_aggregate_interval_minutes": self.auto_aggregate_interval,
            "realtime_counters": self.realtime_counters,
        }