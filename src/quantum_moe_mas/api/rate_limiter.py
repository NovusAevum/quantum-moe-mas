"""
Rate Limiting and Quota Management System.

This module provides comprehensive rate limiting and quota management
using token bucket algorithms and sliding window techniques.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os

from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class LimitType(Enum):
    """Types of rate limits."""
    
    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    TOKENS_PER_HOUR = "tokens_per_hour"
    TOKENS_PER_DAY = "tokens_per_day"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    
    limit_type: LimitType
    limit_value: int
    window_size_seconds: int
    burst_allowance: float = 1.5  # Allow 50% burst above limit
    
    def __post_init__(self) -> None:
        """Validate rate limit configuration."""
        if self.limit_value <= 0:
            raise ValueError("Limit value must be positive")
        if self.window_size_seconds <= 0:
            raise ValueError("Window size must be positive")
        if self.burst_allowance < 1.0:
            raise ValueError("Burst allowance must be >= 1.0")


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    
    def __post_init__(self) -> None:
        """Initialize token bucket."""
        if self.tokens == 0.0:
            self.tokens = float(self.capacity)
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self) -> None:
        """Refill the token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_status(self) -> Dict[str, Any]:
        """Get bucket status."""
        self._refill()
        return {
            "capacity": self.capacity,
            "current_tokens": self.tokens,
            "refill_rate": self.refill_rate,
            "utilization_percent": ((self.capacity - self.tokens) / self.capacity) * 100,
        }


@dataclass
class SlidingWindow:
    """Sliding window for tracking requests."""
    
    window_size_seconds: int
    max_requests: int
    requests: List[float] = field(default_factory=list)
    
    def add_request(self, timestamp: Optional[float] = None) -> bool:
        """
        Add a request to the window.
        
        Args:
            timestamp: Request timestamp (defaults to current time)
        
        Returns:
            True if request was added, False if limit exceeded
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Clean old requests
        self._clean_old_requests(timestamp)
        
        # Check if we can add the request
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(timestamp)
        return True
    
    def _clean_old_requests(self, current_time: float) -> None:
        """Remove requests outside the window."""
        cutoff_time = current_time - self.window_size_seconds
        self.requests = [req for req in self.requests if req > cutoff_time]
    
    def get_current_count(self) -> int:
        """Get current request count in window."""
        self._clean_old_requests(time.time())
        return len(self.requests)
    
    def get_status(self) -> Dict[str, Any]:
        """Get window status."""
        current_count = self.get_current_count()
        return {
            "window_size_seconds": self.window_size_seconds,
            "max_requests": self.max_requests,
            "current_requests": current_count,
            "utilization_percent": (current_count / self.max_requests) * 100,
            "remaining_requests": self.max_requests - current_count,
        }


class RateLimiter:
    """
    Comprehensive rate limiting system using token buckets and sliding windows.
    
    Provides flexible rate limiting for different API providers with
    multiple limit types and burst handling.
    """
    
    def __init__(
        self,
        default_limits: Optional[Dict[str, List[RateLimit]]] = None,
        storage_path: Optional[str] = None
    ) -> None:
        """
        Initialize rate limiter.
        
        Args:
            default_limits: Default rate limits by provider
            storage_path: Path to persist rate limit state
        """
        self.storage_path = storage_path or os.path.expanduser("~/.quantum_moe_mas/rate_limits.json")
        
        # Rate limit configurations
        self.limits: Dict[str, List[RateLimit]] = default_limits or {}
        
        # Token buckets for each provider and limit type
        self.token_buckets: Dict[str, Dict[LimitType, TokenBucket]] = {}
        
        # Sliding windows for request tracking
        self.sliding_windows: Dict[str, Dict[LimitType, SlidingWindow]] = {}
        
        # Load default limits if none provided
        if not self.limits:
            self._load_default_limits()
        
        # Initialize buckets and windows
        self._initialize_buckets_and_windows()
        
        logger.info(
            "Initialized RateLimiter",
            providers=len(self.limits),
            storage_path=self.storage_path
        )
    
    def _load_default_limits(self) -> None:
        """Load default rate limits for common providers."""
        self.limits = {
            "openai_playground": [
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 60, 60),
                RateLimit(LimitType.TOKENS_PER_MINUTE, 10000, 60),
            ],
            "hugging_face": [
                RateLimit(LimitType.REQUESTS_PER_SECOND, 10, 1),
                RateLimit(LimitType.REQUESTS_PER_HOUR, 1000, 3600),
            ],
            "google_ai_studio": [
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 60, 60),
                RateLimit(LimitType.TOKENS_PER_MINUTE, 32000, 60),
            ],
            "groq": [
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 30, 60),
                RateLimit(LimitType.TOKENS_PER_MINUTE, 6000, 60),
            ],
            "cerebras": [
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 20, 60),
                RateLimit(LimitType.TOKENS_PER_MINUTE, 4000, 60),
            ],
            "deepseek": [
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 50, 60),
                RateLimit(LimitType.TOKENS_PER_MINUTE, 8000, 60),
            ],
            "cohere": [
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 100, 60),
                RateLimit(LimitType.TOKENS_PER_MINUTE, 20000, 60),
            ],
            "anthropic_claude": [
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 50, 60),
                RateLimit(LimitType.TOKENS_PER_MINUTE, 40000, 60),
            ],
            "replicate": [
                RateLimit(LimitType.REQUESTS_PER_SECOND, 5, 1),
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 100, 60),
            ],
            "stability_ai": [
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 150, 60),
            ],
        }
    
    def _initialize_buckets_and_windows(self) -> None:
        """Initialize token buckets and sliding windows for all providers."""
        for provider, limits in self.limits.items():
            self.token_buckets[provider] = {}
            self.sliding_windows[provider] = {}
            
            for limit in limits:
                # Create token bucket
                refill_rate = limit.limit_value / limit.window_size_seconds
                capacity = int(limit.limit_value * limit.burst_allowance)
                
                self.token_buckets[provider][limit.limit_type] = TokenBucket(
                    capacity=capacity,
                    refill_rate=refill_rate
                )
                
                # Create sliding window
                self.sliding_windows[provider][limit.limit_type] = SlidingWindow(
                    window_size_seconds=limit.window_size_seconds,
                    max_requests=limit.limit_value
                )
    
    async def check_rate_limit(
        self,
        provider: str,
        tokens: int = 1,
        limit_types: Optional[List[LimitType]] = None
    ) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            provider: API provider name
            tokens: Number of tokens for token-based limits
            limit_types: Specific limit types to check (defaults to all)
        
        Returns:
            True if request is allowed, False if rate limited
        """
        if provider not in self.limits:
            # No limits configured, allow request
            return True
        
        # Check all applicable limits
        limits_to_check = limit_types or [limit.limit_type for limit in self.limits[provider]]
        
        for limit_type in limits_to_check:
            if not await self._check_single_limit(provider, limit_type, tokens):
                logger.warning(
                    f"Rate limit exceeded for {provider}",
                    limit_type=limit_type.value,
                    tokens=tokens
                )
                return False
        
        return True
    
    async def _check_single_limit(
        self,
        provider: str,
        limit_type: LimitType,
        tokens: int
    ) -> bool:
        """Check a single rate limit."""
        # Check token bucket
        if provider in self.token_buckets and limit_type in self.token_buckets[provider]:
            bucket = self.token_buckets[provider][limit_type]
            
            # For token-based limits, consume the actual token count
            tokens_to_consume = tokens if "TOKEN" in limit_type.value else 1
            
            if not bucket.consume(tokens_to_consume):
                return False
        
        # Check sliding window
        if provider in self.sliding_windows and limit_type in self.sliding_windows[provider]:
            window = self.sliding_windows[provider][limit_type]
            
            if not window.add_request():
                return False
        
        return True
    
    def add_provider_limits(
        self,
        provider: str,
        limits: List[RateLimit]
    ) -> None:
        """
        Add rate limits for a provider.
        
        Args:
            provider: API provider name
            limits: List of rate limits
        """
        self.limits[provider] = limits
        
        # Initialize buckets and windows for new provider
        self.token_buckets[provider] = {}
        self.sliding_windows[provider] = {}
        
        for limit in limits:
            refill_rate = limit.limit_value / limit.window_size_seconds
            capacity = int(limit.limit_value * limit.burst_allowance)
            
            self.token_buckets[provider][limit.limit_type] = TokenBucket(
                capacity=capacity,
                refill_rate=refill_rate
            )
            
            self.sliding_windows[provider][limit.limit_type] = SlidingWindow(
                window_size_seconds=limit.window_size_seconds,
                max_requests=limit.limit_value
            )
        
        logger.info(f"Added rate limits for {provider}: {len(limits)} limits")
    
    def update_provider_limits(
        self,
        provider: str,
        limit_type: LimitType,
        new_limit_value: int
    ) -> None:
        """
        Update rate limit for a provider.
        
        Args:
            provider: API provider name
            limit_type: Type of limit to update
            new_limit_value: New limit value
        """
        if provider not in self.limits:
            logger.warning(f"Provider {provider} not found")
            return
        
        # Update limit configuration
        for limit in self.limits[provider]:
            if limit.limit_type == limit_type:
                limit.limit_value = new_limit_value
                
                # Update token bucket
                if provider in self.token_buckets and limit_type in self.token_buckets[provider]:
                    bucket = self.token_buckets[provider][limit_type]
                    bucket.capacity = int(new_limit_value * limit.burst_allowance)
                    bucket.refill_rate = new_limit_value / limit.window_size_seconds
                
                # Update sliding window
                if provider in self.sliding_windows and limit_type in self.sliding_windows[provider]:
                    window = self.sliding_windows[provider][limit_type]
                    window.max_requests = new_limit_value
                
                logger.info(f"Updated {limit_type.value} limit for {provider} to {new_limit_value}")
                break
    
    def get_remaining_quota(
        self,
        provider: str,
        limit_type: LimitType
    ) -> Optional[int]:
        """
        Get remaining quota for a provider and limit type.
        
        Args:
            provider: API provider name
            limit_type: Type of limit
        
        Returns:
            Remaining quota or None if not found
        """
        if provider in self.token_buckets and limit_type in self.token_buckets[provider]:
            bucket = self.token_buckets[provider][limit_type]
            return int(bucket.tokens)
        
        if provider in self.sliding_windows and limit_type in self.sliding_windows[provider]:
            window = self.sliding_windows[provider][limit_type]
            return window.max_requests - window.get_current_count()
        
        return None
    
    def reset_limits(self, provider: str) -> None:
        """
        Reset all limits for a provider.
        
        Args:
            provider: API provider name
        """
        if provider in self.token_buckets:
            for bucket in self.token_buckets[provider].values():
                bucket.tokens = float(bucket.capacity)
                bucket.last_refill = time.time()
        
        if provider in self.sliding_windows:
            for window in self.sliding_windows[provider].values():
                window.requests.clear()
        
        logger.info(f"Reset rate limits for {provider}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive rate limiter status.
        
        Returns:
            Dictionary with rate limiter status
        """
        status = {
            "providers": len(self.limits),
            "total_limits": sum(len(limits) for limits in self.limits.values()),
            "provider_status": {},
        }
        
        for provider in self.limits:
            provider_status = {
                "limits": len(self.limits[provider]),
                "token_buckets": {},
                "sliding_windows": {},
            }
            
            # Token bucket status
            if provider in self.token_buckets:
                for limit_type, bucket in self.token_buckets[provider].items():
                    provider_status["token_buckets"][limit_type.value] = bucket.get_status()
            
            # Sliding window status
            if provider in self.sliding_windows:
                for limit_type, window in self.sliding_windows[provider].items():
                    provider_status["sliding_windows"][limit_type.value] = window.get_status()
            
            status["provider_status"][provider] = provider_status
        
        return status


@dataclass
class QuotaConfig:
    """Quota configuration for a provider."""
    
    daily_requests: Optional[int] = None
    daily_tokens: Optional[int] = None
    monthly_requests: Optional[int] = None
    monthly_tokens: Optional[int] = None
    monthly_cost: Optional[float] = None
    reset_day: int = 1  # Day of month for monthly reset


@dataclass
class QuotaUsage:
    """Current quota usage."""
    
    daily_requests: int = 0
    daily_tokens: int = 0
    monthly_requests: int = 0
    monthly_tokens: int = 0
    monthly_cost: float = 0.0
    last_daily_reset: datetime = field(default_factory=datetime.utcnow)
    last_monthly_reset: datetime = field(default_factory=datetime.utcnow)


class QuotaManager:
    """
    Quota management system for tracking API usage and costs.
    
    Provides comprehensive quota tracking including:
    - Daily and monthly request limits
    - Token usage tracking
    - Cost monitoring and alerts
    - Automatic quota resets
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        default_quotas: Optional[Dict[str, QuotaConfig]] = None
    ) -> None:
        """
        Initialize quota manager.
        
        Args:
            storage_path: Path to persist quota data
            default_quotas: Default quota configurations
        """
        self.storage_path = storage_path or os.path.expanduser("~/.quantum_moe_mas/quotas.json")
        
        # Quota configurations
        self.quotas: Dict[str, QuotaConfig] = default_quotas or {}
        
        # Current usage tracking
        self.usage: Dict[str, QuotaUsage] = {}
        
        # Load default quotas if none provided
        if not self.quotas:
            self._load_default_quotas()
        
        # Initialize usage tracking
        self._initialize_usage_tracking()
        
        logger.info(
            "Initialized QuotaManager",
            providers=len(self.quotas),
            storage_path=self.storage_path
        )
    
    def _load