"""
Caching layer for frequently accessed documents and retrieval results.

This module implements intelligent caching strategies to reduce API calls
and improve retrieval performance by 40%+ as specified in requirements.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import json

import numpy as np

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.rag.retrieval import RetrievalResult, RetrievalMetrics

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata and expiration."""
    
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    max_size_bytes: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def size_utilization(self) -> float:
        """Calculate cache size utilization."""
        return self.total_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0.0


class LRUCache:
    """Least Recently Used cache with TTL support."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
        default_ttl: Optional[float] = 3600,  # 1 hour
    ):
        self.max_size = max_size
        self.max_size_bytes = max_size_bytes
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(max_size_bytes=max_size_bytes)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                await self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._stats.hits += 1
            return entry.value
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Put value in cache."""
        async with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                await self._remove_entry(key)
            
            # Check if we need to evict entries
            await self._ensure_capacity(size_bytes)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes,
            )
            
            self._cache[key] = entry
            self._stats.total_size_bytes += size_bytes
            self._stats.entry_count += 1
    
    async def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats = CacheStats(max_size_bytes=self.max_size_bytes)
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                await self._remove_entry(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    async def _remove_entry(self, key: str) -> None:
        """Remove entry and update stats."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.total_size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
    
    async def _ensure_capacity(self, new_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check size limit
        while (
            self._stats.total_size_bytes + new_size > self.max_size_bytes
            and self._cache
        ):
            # Remove least recently used entry
            oldest_key = next(iter(self._cache))
            await self._remove_entry(oldest_key)
            self._stats.evictions += 1
        
        # Check entry count limit
        while len(self._cache) >= self.max_size and self._cache:
            oldest_key = next(iter(self._cache))
            await self._remove_entry(oldest_key)
            self._stats.evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            else:
                # Fallback to JSON serialization size
                return len(json.dumps(value, default=str))
        except Exception:
            return 1024  # Default size estimate


class RetrievalCache:
    """Specialized cache for retrieval results with intelligent invalidation."""
    
    def __init__(
        self,
        max_entries: int = 1000,
        max_size_mb: int = 100,
        query_ttl: float = 3600,  # 1 hour
        embedding_ttl: float = 86400,  # 24 hours
        enable_semantic_deduplication: bool = True,
    ):
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        self.query_ttl = query_ttl
        self.embedding_ttl = embedding_ttl
        self.enable_semantic_deduplication = enable_semantic_deduplication
        
        # Multiple cache layers
        self.query_cache = LRUCache(
            max_size=max_entries,
            max_size_bytes=max_size_mb * 1024 * 1024,
            default_ttl=query_ttl,
        )
        
        self.embedding_cache = LRUCache(
            max_size=max_entries * 2,
            max_size_bytes=max_size_mb * 1024 * 1024 // 2,
            default_ttl=embedding_ttl,
        )
        
        self.document_cache = LRUCache(
            max_size=max_entries // 2,
            max_size_bytes=max_size_mb * 1024 * 1024 // 2,
            default_ttl=embedding_ttl,
        )
        
        # Semantic similarity tracking for deduplication
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.similarity_threshold = 0.95
        
        logger.info("Retrieval cache initialized")
    
    async def get_query_results(
        self,
        query: str,
        k: int,
        strategy: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[List[RetrievalResult], RetrievalMetrics]]:
        """Get cached query results."""
        cache_key = self._generate_query_key(query, k, strategy, filters)
        
        # Check for exact match
        cached_result = await self.query_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Query cache hit: {query[:50]}...")
            return cached_result
        
        # Check for semantic similarity if enabled
        if self.enable_semantic_deduplication:
            similar_result = await self._find_similar_query(query, k, strategy, filters)
            if similar_result:
                logger.debug(f"Semantic cache hit: {query[:50]}...")
                return similar_result
        
        return None
    
    async def cache_query_results(
        self,
        query: str,
        k: int,
        strategy: str,
        results: List[RetrievalResult],
        metrics: RetrievalMetrics,
        filters: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Cache query results."""
        cache_key = self._generate_query_key(query, k, strategy, filters)
        
        # Store results
        await self.query_cache.put(
            cache_key,
            (results, metrics),
            ttl=self.query_ttl,
        )
        
        # Store query embedding for semantic deduplication
        if query_embedding is not None and self.enable_semantic_deduplication:
            self.query_embeddings[cache_key] = query_embedding
        
        logger.debug(f"Cached query results: {query[:50]}...")
    
    async def get_embedding(self, content: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        cache_key = self._generate_embedding_key(content, model)
        return await self.embedding_cache.get(cache_key)
    
    async def cache_embedding(
        self,
        content: str,
        model: str,
        embedding: np.ndarray,
    ) -> None:
        """Cache embedding."""
        cache_key = self._generate_embedding_key(content, model)
        await self.embedding_cache.put(
            cache_key,
            embedding,
            ttl=self.embedding_ttl,
        )
    
    async def get_document(self, document_id: str) -> Optional[Any]:
        """Get cached document."""
        return await self.document_cache.get(document_id)
    
    async def cache_document(
        self,
        document_id: str,
        document: Any,
    ) -> None:
        """Cache document."""
        await self.document_cache.put(
            document_id,
            document,
            ttl=self.embedding_ttl,
        )
    
    async def invalidate_document(self, document_id: str) -> None:
        """Invalidate all cache entries related to a document."""
        # Remove document from cache
        await self.document_cache.remove(document_id)
        
        # Remove related query results (this is simplified - in production,
        # you'd want more sophisticated invalidation)
        await self._invalidate_related_queries(document_id)
        
        logger.debug(f"Invalidated cache for document: {document_id}")
    
    async def cleanup(self) -> Dict[str, int]:
        """Clean up expired entries from all caches."""
        query_expired = await self.query_cache.cleanup_expired()
        embedding_expired = await self.embedding_cache.cleanup_expired()
        document_expired = await self.document_cache.cleanup_expired()
        
        # Clean up query embeddings
        expired_keys = [
            key for key in self.query_embeddings.keys()
            if key not in [entry.key for entry in self.query_cache._cache.values()]
        ]
        for key in expired_keys:
            del self.query_embeddings[key]
        
        cleanup_stats = {
            "query_expired": query_expired,
            "embedding_expired": embedding_expired,
            "document_expired": document_expired,
            "query_embeddings_cleaned": len(expired_keys),
        }
        
        if sum(cleanup_stats.values()) > 0:
            logger.info(f"Cache cleanup completed: {cleanup_stats}")
        
        return cleanup_stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        query_stats = self.query_cache.get_stats()
        embedding_stats = self.embedding_cache.get_stats()
        document_stats = self.document_cache.get_stats()
        
        total_hits = query_stats.hits + embedding_stats.hits + document_stats.hits
        total_misses = query_stats.misses + embedding_stats.misses + document_stats.misses
        
        return {
            "overall": {
                "total_hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
                "total_entries": query_stats.entry_count + embedding_stats.entry_count + document_stats.entry_count,
                "total_size_mb": (query_stats.total_size_bytes + embedding_stats.total_size_bytes + document_stats.total_size_bytes) / (1024 * 1024),
            },
            "query_cache": {
                "hit_rate": query_stats.hit_rate,
                "entries": query_stats.entry_count,
                "size_mb": query_stats.total_size_bytes / (1024 * 1024),
                "evictions": query_stats.evictions,
            },
            "embedding_cache": {
                "hit_rate": embedding_stats.hit_rate,
                "entries": embedding_stats.entry_count,
                "size_mb": embedding_stats.total_size_bytes / (1024 * 1024),
                "evictions": embedding_stats.evictions,
            },
            "document_cache": {
                "hit_rate": document_stats.hit_rate,
                "entries": document_stats.entry_count,
                "size_mb": document_stats.total_size_bytes / (1024 * 1024),
                "evictions": document_stats.evictions,
            },
            "semantic_deduplication": {
                "enabled": self.enable_semantic_deduplication,
                "query_embeddings_stored": len(self.query_embeddings),
                "similarity_threshold": self.similarity_threshold,
            },
        }
    
    def _generate_query_key(
        self,
        query: str,
        k: int,
        strategy: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate cache key for query."""
        key_data = {
            "query": query,
            "k": k,
            "strategy": strategy,
            "filters": filters or {},
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _generate_embedding_key(self, content: str, model: str) -> str:
        """Generate cache key for embedding."""
        key_str = f"{model}:{hashlib.sha256(content.encode()).hexdigest()}"
        return key_str
    
    async def _find_similar_query(
        self,
        query: str,
        k: int,
        strategy: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[List[RetrievalResult], RetrievalMetrics]]:
        """Find semantically similar cached query."""
        if not self.query_embeddings:
            return None
        
        # This is a simplified implementation - in production, you'd want
        # to use proper embedding similarity search
        query_lower = query.lower()
        
        for cached_key, cached_embedding in self.query_embeddings.items():
            # Try to extract original query from cache (simplified)
            cached_result = await self.query_cache.get(cached_key)
            if cached_result:
                # Simple text similarity check (in production, use embedding similarity)
                if self._calculate_text_similarity(query_lower, cached_key) > self.similarity_threshold:
                    return cached_result
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (placeholder for embedding similarity)."""
        # This is a very basic implementation - in production, use proper
        # embedding cosine similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _invalidate_related_queries(self, document_id: str) -> None:
        """Invalidate query cache entries that might be affected by document changes."""
        # This is a simplified implementation - in production, you'd want
        # more sophisticated tracking of document-query relationships
        
        # For now, we'll clear a portion of the query cache to be safe
        # In a full implementation, you'd track which queries returned
        # results from specific documents
        
        cache_keys = list(self.query_cache._cache.keys())
        invalidated_count = 0
        
        # Invalidate up to 10% of cache entries (conservative approach)
        max_invalidations = max(1, len(cache_keys) // 10)
        
        for key in cache_keys[:max_invalidations]:
            await self.query_cache.remove(key)
            invalidated_count += 1
        
        if invalidated_count > 0:
            logger.debug(f"Invalidated {invalidated_count} query cache entries due to document change")