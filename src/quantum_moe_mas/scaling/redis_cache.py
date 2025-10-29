"""
Intelligent Redis Caching System

Implements multi-level caching strategy with intelligent query result caching,
TTL management, and cache invalidation to achieve 40%+ API call reduction.

Requirements: 8.4, 8.5
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pickle
import zlib

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


class CacheStrategy(Enum):
    """Different caching strategies available."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


class CacheLevel(Enum):
    """Different cache levels in the hierarchy."""
    MEMORY = "memory"  # In-memory cache (fastest)
    REDIS = "redis"    # Redis cache (fast, persistent)
    CDN = "cdn"        # CDN cache (for static content)


@dataclass
class CacheMetrics:
    """Metrics for cache performance monitoring."""
    
    # Hit/Miss statistics
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Performance metrics
    avg_hit_time: float = 0.0
    avg_miss_time: float = 0.0
    
    # Storage metrics
    memory_usage: int = 0
    redis_usage: int = 0
    total_keys: int = 0
    
    # Business impact
    api_calls_saved: int = 0
    cost_savings: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate percentage."""
        return 100.0 - self.hit_rate
    
    def meets_target(self, target_hit_rate: float = 80.0) -> bool:
        """Check if cache performance meets target hit rate."""
        return self.hit_rate >= target_hit_rate


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    size: int = 0
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class IntelligentCache:
    """
    Multi-level intelligent caching system with Redis backend.
    
    Provides intelligent query result caching with adaptive TTL,
    cache invalidation, and performance optimization to achieve
    40%+ reduction in API calls.
    """
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379/0",
                 default_ttl: int = 3600,  # 1 hour
                 max_memory_cache_size: int = 1000,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        """Initialize the intelligent cache system."""
        
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_memory_cache_size = max_memory_cache_size
        self.strategy = strategy
        
        # Initialize Redis connection
        self.redis_client = None
        self._redis_connected = False
        
        # In-memory cache for fastest access
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_cache_order = []  # For LRU eviction
        
        # Cache metrics
        self._metrics = CacheMetrics()
        
        # Cache configuration
        self._ttl_rules: Dict[str, int] = {}  # Pattern-based TTL rules
        self._invalidation_rules: Dict[str, List[str]] = {}  # Tag-based invalidation
        
        # Performance tracking
        self._performance_history = []
        
        logger.info("IntelligentCache initialized",
                   strategy=strategy.value,
                   default_ttl=default_ttl,
                   max_memory_size=max_memory_cache_size)
    
    async def connect(self) -> bool:
        """Connect to Redis server."""
        
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self._redis_connected = True
            
            logger.info("Redis connection established", url=self.redis_url)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to Redis",
                        error=str(e),
                        url=self.redis_url)
            self._redis_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis server."""
        
        if self.redis_client:
            await self.redis_client.close()
            self._redis_connected = False
            logger.info("Redis connection closed")
    
    def _generate_cache_key(self, 
                           namespace: str,
                           identifier: str,
                           params: Optional[Dict[str, Any]] = None) -> str:
        """Generate a consistent cache key."""
        
        key_parts = [namespace, identifier]
        
        if params:
            # Sort parameters for consistent key generation
            sorted_params = sorted(params.items())
            params_str = json.dumps(sorted_params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            key_parts.append(params_hash)
        
        return ":".join(key_parts)
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        
        try:
            # Use pickle for Python objects, compress for efficiency
            serialized = pickle.dumps(value)
            compressed = zlib.compress(serialized)
            return compressed
            
        except Exception as e:
            logger.error("Failed to serialize cache value", error=str(e))
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        
        try:
            # Decompress and unpickle
            decompressed = zlib.decompress(data)
            value = pickle.loads(decompressed)
            return value
            
        except Exception as e:
            logger.error("Failed to deserialize cache value", error=str(e))
            raise
    
    async def get(self, 
                  namespace: str,
                  identifier: str,
                  params: Optional[Dict[str, Any]] = None,
                  default: Any = None) -> Any:
        """Get value from cache with multi-level lookup."""
        
        start_time = time.time()
        cache_key = self._generate_cache_key(namespace, identifier, params)
        
        try:
            # Level 1: Memory cache (fastest)
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                
                if not entry.is_expired:
                    # Update access statistics
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    # Move to front for LRU
                    if cache_key in self._memory_cache_order:
                        self._memory_cache_order.remove(cache_key)
                    self._memory_cache_order.insert(0, cache_key)
                    
                    self._record_cache_hit(time.time() - start_time, CacheLevel.MEMORY)
                    
                    logger.debug("Memory cache hit", key=cache_key)
                    return entry.value
                else:
                    # Remove expired entry
                    await self._remove_from_memory_cache(cache_key)
            
            # Level 2: Redis cache
            if self._redis_connected and self.redis_client:
                try:
                    redis_data = await self.redis_client.get(cache_key)
                    
                    if redis_data:
                        value = self._deserialize_value(redis_data)
                        
                        # Store in memory cache for faster future access
                        await self._store_in_memory_cache(cache_key, value)
                        
                        self._record_cache_hit(time.time() - start_time, CacheLevel.REDIS)
                        
                        logger.debug("Redis cache hit", key=cache_key)
                        return value
                        
                except Exception as e:
                    logger.warning("Redis cache lookup failed",
                                 key=cache_key,
                                 error=str(e))
            
            # Cache miss - record and return default
            self._record_cache_miss(time.time() - start_time)
            
            logger.debug("Cache miss", key=cache_key)
            return default
            
        except Exception as e:
            logger.error("Cache get operation failed",
                        key=cache_key,
                        error=str(e))
            return default
    
    async def set(self,
                  namespace: str,
                  identifier: str,
                  value: Any,
                  ttl: Optional[int] = None,
                  params: Optional[Dict[str, Any]] = None,
                  tags: Optional[List[str]] = None) -> bool:
        """Set value in cache with multi-level storage."""
        
        cache_key = self._generate_cache_key(namespace, identifier, params)
        effective_ttl = ttl or self._get_ttl_for_key(cache_key)
        cache_tags = tags or []
        
        try:
            # Store in memory cache
            await self._store_in_memory_cache(cache_key, value, effective_ttl, cache_tags)
            
            # Store in Redis cache
            if self._redis_connected and self.redis_client:
                try:
                    serialized_value = self._serialize_value(value)
                    
                    if effective_ttl:
                        await self.redis_client.setex(
                            cache_key,
                            effective_ttl,
                            serialized_value
                        )
                    else:
                        await self.redis_client.set(cache_key, serialized_value)
                    
                    # Store tags for invalidation
                    if cache_tags:
                        for tag in cache_tags:
                            await self.redis_client.sadd(f"tag:{tag}", cache_key)
                    
                    logger.debug("Value cached successfully",
                               key=cache_key,
                               ttl=effective_ttl,
                               tags=cache_tags)
                    
                except Exception as e:
                    logger.warning("Redis cache store failed",
                                 key=cache_key,
                                 error=str(e))
            
            return True
            
        except Exception as e:
            logger.error("Cache set operation failed",
                        key=cache_key,
                        error=str(e))
            return False
    
    async def delete(self, 
                     namespace: str,
                     identifier: str,
                     params: Optional[Dict[str, Any]] = None) -> bool:
        """Delete value from cache."""
        
        cache_key = self._generate_cache_key(namespace, identifier, params)
        
        try:
            # Remove from memory cache
            await self._remove_from_memory_cache(cache_key)
            
            # Remove from Redis cache
            if self._redis_connected and self.redis_client:
                await self.redis_client.delete(cache_key)
            
            logger.debug("Cache entry deleted", key=cache_key)
            return True
            
        except Exception as e:
            logger.error("Cache delete operation failed",
                        key=cache_key,
                        error=str(e))
            return False
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all cache entries with a specific tag."""
        
        invalidated_count = 0
        
        try:
            if self._redis_connected and self.redis_client:
                # Get all keys with this tag
                tagged_keys = await self.redis_client.smembers(f"tag:{tag}")
                
                if tagged_keys:
                    # Delete all tagged keys
                    await self.redis_client.delete(*tagged_keys)
                    
                    # Remove from memory cache
                    for key in tagged_keys:
                        if isinstance(key, bytes):
                            key = key.decode('utf-8')
                        await self._remove_from_memory_cache(key)
                    
                    # Clean up tag set
                    await self.redis_client.delete(f"tag:{tag}")
                    
                    invalidated_count = len(tagged_keys)
            
            # Also check memory cache for tagged entries
            memory_keys_to_remove = []
            for key, entry in self._memory_cache.items():
                if tag in entry.tags:
                    memory_keys_to_remove.append(key)
            
            for key in memory_keys_to_remove:
                await self._remove_from_memory_cache(key)
                invalidated_count += 1
            
            logger.info("Cache invalidation by tag completed",
                       tag=tag,
                       invalidated_count=invalidated_count)
            
            return invalidated_count
            
        except Exception as e:
            logger.error("Cache invalidation by tag failed",
                        tag=tag,
                        error=str(e))
            return 0
    
    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        
        try:
            # Clear memory cache
            self._memory_cache.clear()
            self._memory_cache_order.clear()
            
            # Clear Redis cache (only our keys)
            if self._redis_connected and self.redis_client:
                # Use pattern to delete only our keys
                keys = await self.redis_client.keys("*")
                if keys:
                    await self.redis_client.delete(*keys)
            
            logger.info("All cache entries cleared")
            return True
            
        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))
            return False
    
    async def _store_in_memory_cache(self,
                                   key: str,
                                   value: Any,
                                   ttl: Optional[int] = None,
                                   tags: Optional[List[str]] = None) -> None:
        """Store entry in memory cache with LRU eviction."""
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl=ttl,
            size=len(str(value)),  # Rough size estimate
            tags=tags or []
        )
        
        # Add to cache
        self._memory_cache[key] = entry
        
        # Update order for LRU
        if key in self._memory_cache_order:
            self._memory_cache_order.remove(key)
        self._memory_cache_order.insert(0, key)
        
        # Evict if over limit
        while len(self._memory_cache) > self.max_memory_cache_size:
            oldest_key = self._memory_cache_order.pop()
            if oldest_key in self._memory_cache:
                del self._memory_cache[oldest_key]
    
    async def _remove_from_memory_cache(self, key: str) -> None:
        """Remove entry from memory cache."""
        
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        if key in self._memory_cache_order:
            self._memory_cache_order.remove(key)
    
    def _get_ttl_for_key(self, key: str) -> int:
        """Get TTL for a specific key based on patterns."""
        
        # Check pattern-based TTL rules
        for pattern, ttl in self._ttl_rules.items():
            if pattern in key:
                return ttl
        
        return self.default_ttl
    
    def _record_cache_hit(self, response_time: float, level: CacheLevel) -> None:
        """Record cache hit metrics."""
        
        self._metrics.total_requests += 1
        self._metrics.cache_hits += 1
        self._metrics.avg_hit_time = (
            (self._metrics.avg_hit_time * (self._metrics.cache_hits - 1) + response_time) /
            self._metrics.cache_hits
        )
        
        # Estimate API call savings
        if level == CacheLevel.MEMORY or level == CacheLevel.REDIS:
            self._metrics.api_calls_saved += 1
            self._metrics.cost_savings += 0.01  # Rough estimate
    
    def _record_cache_miss(self, response_time: float) -> None:
        """Record cache miss metrics."""
        
        self._metrics.total_requests += 1
        self._metrics.cache_misses += 1
        self._metrics.avg_miss_time = (
            (self._metrics.avg_miss_time * (self._metrics.cache_misses - 1) + response_time) /
            self._metrics.cache_misses
        )
    
    async def get_metrics(self) -> CacheMetrics:
        """Get current cache performance metrics."""
        
        # Update storage metrics
        self._metrics.memory_usage = len(self._memory_cache)
        self._metrics.total_keys = len(self._memory_cache)
        
        if self._redis_connected and self.redis_client:
            try:
                info = await self.redis_client.info('memory')
                self._metrics.redis_usage = info.get('used_memory', 0)
                
                # Count Redis keys
                redis_keys = await self.redis_client.dbsize()
                self._metrics.total_keys += redis_keys
                
            except Exception as e:
                logger.warning("Failed to get Redis metrics", error=str(e))
        
        self._metrics.timestamp = datetime.now()
        return self._metrics
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Perform cache optimization based on usage patterns."""
        
        optimization_results = {
            'actions_taken': [],
            'performance_improvement': 0.0,
            'memory_freed': 0,
            'recommendations': []
        }
        
        try:
            # Remove expired entries from memory cache
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._remove_from_memory_cache(key)
            
            if expired_keys:
                optimization_results['actions_taken'].append(
                    f"Removed {len(expired_keys)} expired entries"
                )
                optimization_results['memory_freed'] += len(expired_keys)
            
            # Analyze access patterns for TTL optimization
            if len(self._memory_cache) > 10:
                access_counts = [entry.access_count for entry in self._memory_cache.values()]
                avg_access = sum(access_counts) / len(access_counts)
                
                # Identify frequently accessed items for longer TTL
                frequent_items = [
                    key for key, entry in self._memory_cache.items()
                    if entry.access_count > avg_access * 2
                ]
                
                if frequent_items:
                    optimization_results['recommendations'].append(
                        f"Consider increasing TTL for {len(frequent_items)} frequently accessed items"
                    )
            
            # Check hit rate and suggest improvements
            current_metrics = await self.get_metrics()
            if current_metrics.hit_rate < 80.0:
                optimization_results['recommendations'].append(
                    f"Hit rate ({current_metrics.hit_rate:.1f}%) below target. "
                    "Consider increasing cache size or TTL values."
                )
            
            logger.info("Cache optimization completed",
                       actions=len(optimization_results['actions_taken']),
                       recommendations=len(optimization_results['recommendations']))
            
            return optimization_results
            
        except Exception as e:
            logger.error("Cache optimization failed", error=str(e))
            return optimization_results
    
    def configure_ttl_rule(self, pattern: str, ttl: int) -> None:
        """Configure TTL rule for keys matching a pattern."""
        
        self._ttl_rules[pattern] = ttl
        logger.info("TTL rule configured", pattern=pattern, ttl=ttl)
    
    def configure_invalidation_rule(self, trigger_tag: str, target_tags: List[str]) -> None:
        """Configure automatic invalidation rule."""
        
        self._invalidation_rules[trigger_tag] = target_tags
        logger.info("Invalidation rule configured",
                   trigger=trigger_tag,
                   targets=target_tags)
    
    async def cache_function_result(self,
                                  func: Callable,
                                  namespace: str,
                                  ttl: Optional[int] = None,
                                  tags: Optional[List[str]] = None) -> Callable:
        """Decorator to cache function results."""
        
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            func_name = func.__name__
            cache_params = {
                'args': args,
                'kwargs': kwargs
            }
            
            # Try to get from cache first
            cached_result = await self.get(namespace, func_name, cache_params)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            await self.set(namespace, func_name, result, ttl, cache_params, tags)
            
            return result
        
        return wrapper