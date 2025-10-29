"""
API Orchestrator - Main orchestration class for managing 30+ free AI APIs.

This module provides the central API_Orchestrator class that coordinates
all API integrations with unified interface, failover, and optimization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from quantum_moe_mas.api.key_manager import APIKeyManager, APICredentials
from quantum_moe_mas.api.rate_limiter import RateLimiter, QuotaManager
from quantum_moe_mas.api.health_monitor import APIHealthMonitor
from quantum_moe_mas.api.integrations.base import BaseAPIIntegration, APIResponse, APIError
from quantum_moe_mas.moe.expert import Expert, ExpertType, ExpertStatus
from quantum_moe_mas.moe.expert_manager import ExpertPoolManager, FailoverStrategy
from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class APIProvider(Enum):
    """Supported API providers."""
    
    # Language Models
    OPENAI_PLAYGROUND = "openai_playground"
    HUGGING_FACE = "hugging_face"
    GOOGLE_AI_STUDIO = "google_ai_studio"
    GROQ = "groq"
    CEREBRAS = "cerebras"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    MISTRAL = "mistral"
    TOGETHER_AI = "together_ai"
    
    # Vision & Multimodal
    FLUX_11 = "flux_11"
    STABILITY_AI = "stability_ai"
    REPLICATE = "replicate"
    MIDJOURNEY = "midjourney"
    DALL_E = "dall_e"
    
    # Code & Reasoning
    CODESTRAL = "codestral"
    QWEN_CODER = "qwen_coder"
    DEEPSEEK_CODER = "deepseek_coder"
    GITHUB_COPILOT = "github_copilot"
    
    # Specialized APIs
    PERPLEXITY = "perplexity"
    YOU_COM = "you_com"
    BRAVE_SEARCH = "brave_search"
    SERPER = "serper"
    TAVILY = "tavily"
    
    # Embedding & Vector
    VOYAGE_AI = "voyage_ai"
    JINA_AI = "jina_ai"
    NOMIC_EMBED = "nomic_embed"
    
    # Audio & Speech
    ELEVEN_LABS = "eleven_labs"
    WHISPER_API = "whisper_api"


@dataclass
class APIUsageStats:
    """API usage statistics."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0


@dataclass
class OrchestrationConfig:
    """Configuration for API orchestration."""
    
    max_concurrent_requests: int = 10
    default_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    enable_failover: bool = True
    enable_load_balancing: bool = True
    enable_cost_optimization: bool = True
    health_check_interval_seconds: int = 60
    quota_reset_interval_hours: int = 24


class API_Orchestrator:
    """
    Central API orchestrator managing 30+ free AI APIs with unified interface.
    
    Provides comprehensive API management including:
    - Unified interface for all integrated APIs
    - Automatic failover and load balancing
    - Rate limiting and quota management
    - Cost tracking and optimization
    - Health monitoring and performance analytics
    """
    
    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        key_manager: Optional[APIKeyManager] = None,
        expert_pool_manager: Optional[ExpertPoolManager] = None
    ) -> None:
        """
        Initialize API orchestrator.
        
        Args:
            config: Orchestration configuration
            key_manager: API key manager instance
            expert_pool_manager: Expert pool manager for integration
        """
        self.config = config or OrchestrationConfig()
        self.key_manager = key_manager or APIKeyManager()
        self.expert_pool_manager = expert_pool_manager or ExpertPoolManager(
            failover_strategy=FailoverStrategy.LEAST_LOADED,
            enable_auto_failover=True
        )
        
        # Core components
        self.rate_limiter = RateLimiter()
        self.quota_manager = QuotaManager()
        self.health_monitor = APIHealthMonitor()
        
        # API integrations registry
        self.integrations: Dict[APIProvider, BaseAPIIntegration] = {}
        self.usage_stats: Dict[APIProvider, APIUsageStats] = {}
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.active_requests: Dict[str, datetime] = {}
        
        # Monitoring
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(
            "Initialized API_Orchestrator",
            max_concurrent=self.config.max_concurrent_requests,
            failover_enabled=self.config.enable_failover
        )
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and all components."""
        try:
            # Initialize key manager
            await self.key_manager.initialize()
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Initialize usage stats for all providers
            for provider in APIProvider:
                self.usage_stats[provider] = APIUsageStats()
            
            # Start monitoring task
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            
            self.is_running = True
            logger.info("API_Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API_Orchestrator: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator and cleanup resources."""
        self.is_running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        await self.health_monitor.stop_monitoring()
        logger.info("API_Orchestrator shutdown complete")
    
    def register_integration(
        self,
        provider: APIProvider,
        integration: BaseAPIIntegration
    ) -> None:
        """
        Register an API integration.
        
        Args:
            provider: API provider enum
            integration: Integration instance
        """
        self.integrations[provider] = integration
        
        # Create corresponding expert for MoE routing
        expert = Expert(
            name=f"{provider.value}_expert",
            type=self._get_expert_type(provider),
            api_endpoint=integration.base_url,
            api_key_env_var=integration.api_key_env_var,
            capabilities=integration.capabilities,
            cost_per_token=integration.cost_per_token,
            max_tokens=integration.max_tokens,
            metadata={"provider": provider.value}
        )
        
        self.expert_pool_manager.add_expert(expert)
        logger.info(f"Registered integration for {provider.value}")
    
    def _get_expert_type(self, provider: APIProvider) -> ExpertType:
        """Map API provider to expert type."""
        vision_providers = {APIProvider.FLUX_11, APIProvider.STABILITY_AI, APIProvider.DALL_E}
        code_providers = {APIProvider.CODESTRAL, APIProvider.QWEN_CODER, APIProvider.DEEPSEEK_CODER}
        
        if provider in vision_providers:
            return ExpertType.VISION_MODEL
        elif provider in code_providers:
            return ExpertType.CODE_MODEL
        else:
            return ExpertType.LANGUAGE_MODEL
    
    async def make_request(
        self,
        provider: APIProvider,
        request_data: Dict[str, Any],
        timeout: Optional[int] = None,
        enable_failover: bool = True
    ) -> APIResponse:
        """
        Make a request to a specific API provider with orchestration features.
        
        Args:
            provider: Target API provider
            request_data: Request payload
            timeout: Request timeout in seconds
            enable_failover: Whether to enable automatic failover
        
        Returns:
            APIResponse with result or error information
        
        Raises:
            APIError: If request fails and no failover available
        """
        request_id = f"{provider.value}_{datetime.utcnow().timestamp()}"
        timeout = timeout or self.config.default_timeout_seconds
        
        async with self.semaphore:
            try:
                # Check rate limits
                if not await self.rate_limiter.check_rate_limit(provider.value):
                    raise APIError(f"Rate limit exceeded for {provider.value}")
                
                # Check quota
                if not self.quota_manager.check_quota(provider.value):
                    raise APIError(f"Quota exceeded for {provider.value}")
                
                # Get integration
                integration = self.integrations.get(provider)
                if not integration:
                    raise APIError(f"No integration found for {provider.value}")
                
                # Track active request
                self.active_requests[request_id] = datetime.utcnow()
                
                # Make the request
                start_time = datetime.utcnow()
                response = await asyncio.wait_for(
                    integration.make_request(request_data),
                    timeout=timeout
                )
                end_time = datetime.utcnow()
                
                # Update statistics
                latency_ms = (end_time - start_time).total_seconds() * 1000
                await self._update_usage_stats(provider, True, latency_ms, response)
                
                # Update quota
                tokens_used = response.metadata.get("tokens_used", 0)
                self.quota_manager.consume_quota(provider.value, tokens_used)
                
                return response
                
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout for {provider.value}")
                await self._update_usage_stats(provider, False, timeout * 1000)
                
                if enable_failover and self.config.enable_failover:
                    return await self._handle_failover(provider, request_data, timeout)
                
                raise APIError(f"Request timeout for {provider.value}")
                
            except Exception as e:
                logger.error(f"Request failed for {provider.value}: {e}")
                await self._update_usage_stats(provider, False, 0)
                
                if enable_failover and self.config.enable_failover:
                    return await self._handle_failover(provider, request_data, timeout)
                
                raise APIError(f"Request failed for {provider.value}: {str(e)}")
                
            finally:
                # Clean up active request tracking
                self.active_requests.pop(request_id, None)
    
    async def make_optimized_request(
        self,
        request_data: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        max_cost: Optional[float] = None,
        preferred_providers: Optional[List[APIProvider]] = None
    ) -> APIResponse:
        """
        Make an optimized request using the best available provider.
        
        Args:
            request_data: Request payload
            required_capabilities: Required API capabilities
            max_cost: Maximum acceptable cost
            preferred_providers: Preferred providers in order
        
        Returns:
            APIResponse from the optimal provider
        """
        # Get available providers
        candidates = await self._get_optimal_providers(
            required_capabilities,
            max_cost,
            preferred_providers
        )
        
        if not candidates:
            raise APIError("No suitable providers available")
        
        # Try providers in order of preference
        last_error = None
        for provider in candidates:
            try:
                return await self.make_request(provider, request_data, enable_failover=False)
            except APIError as e:
                last_error = e
                logger.warning(f"Provider {provider.value} failed, trying next: {e}")
                continue
        
        # All providers failed
        raise APIError(f"All providers failed. Last error: {last_error}")
    
    async def _get_optimal_providers(
        self,
        required_capabilities: Optional[List[str]] = None,
        max_cost: Optional[float] = None,
        preferred_providers: Optional[List[APIProvider]] = None
    ) -> List[APIProvider]:
        """Get optimal providers based on criteria."""
        candidates = []
        
        # Start with preferred providers if specified
        if preferred_providers:
            candidates.extend(preferred_providers)
        else:
            candidates.extend(list(APIProvider))
        
        # Filter by capabilities
        if required_capabilities:
            candidates = [
                p for p in candidates
                if p in self.integrations and
                any(cap in self.integrations[p].capabilities for cap in required_capabilities)
            ]
        
        # Filter by cost
        if max_cost is not None:
            candidates = [
                p for p in candidates
                if p in self.integrations and
                self.integrations[p].cost_per_token <= max_cost
            ]
        
        # Filter by availability and health
        available_candidates = []
        for provider in candidates:
            if provider in self.integrations:
                integration = self.integrations[provider]
                if await self.health_monitor.is_healthy(provider.value):
                    available_candidates.append(provider)
        
        # Sort by performance and cost
        available_candidates.sort(
            key=lambda p: (
                -self.usage_stats[p].success_rate,  # Higher success rate first
                self.integrations[p].cost_per_token,  # Lower cost first
                self.usage_stats[p].average_latency_ms  # Lower latency first
            )
        )
        
        return available_candidates
    
    async def _handle_failover(
        self,
        failed_provider: APIProvider,
        request_data: Dict[str, Any],
        timeout: int
    ) -> APIResponse:
        """Handle failover to alternative provider."""
        logger.info(f"Attempting failover from {failed_provider.value}")
        
        # Get failover candidates
        candidates = await self._get_optimal_providers()
        candidates = [p for p in candidates if p != failed_provider]
        
        if not candidates:
            raise APIError("No failover providers available")
        
        # Try the best alternative
        failover_provider = candidates[0]
        logger.info(f"Failing over to {failover_provider.value}")
        
        return await self.make_request(
            failover_provider,
            request_data,
            timeout,
            enable_failover=False  # Prevent infinite failover loops
        )
    
    async def _update_usage_stats(
        self,
        provider: APIProvider,
        success: bool,
        latency_ms: float,
        response: Optional[APIResponse] = None
    ) -> None:
        """Update usage statistics for a provider."""
        stats = self.usage_stats[provider]
        
        stats.total_requests += 1
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
        
        # Update running average latency
        if stats.total_requests == 1:
            stats.average_latency_ms = latency_ms
        else:
            stats.average_latency_ms = (
                (stats.average_latency_ms * (stats.total_requests - 1) + latency_ms) /
                stats.total_requests
            )
        
        if response:
            tokens = response.metadata.get("tokens_used", 0)
            cost = response.metadata.get("cost", 0.0)
            stats.total_tokens += tokens
            stats.total_cost += cost
        
        stats.last_request_time = datetime.utcnow()
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_running:
            try:
                # Clean up old active requests
                cutoff_time = datetime.utcnow() - timedelta(minutes=5)
                expired_requests = [
                    req_id for req_id, start_time in self.active_requests.items()
                    if start_time < cutoff_time
                ]
                for req_id in expired_requests:
                    self.active_requests.pop(req_id, None)
                
                # Reset quotas if needed
                self.quota_manager.reset_quotas_if_needed()
                
                # Log statistics
                await self._log_statistics()
                
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _log_statistics(self) -> None:
        """Log orchestrator statistics."""
        total_requests = sum(stats.total_requests for stats in self.usage_stats.values())
        total_cost = sum(stats.total_cost for stats in self.usage_stats.values())
        active_integrations = len(self.integrations)
        
        logger.info(
            "API Orchestrator Statistics",
            total_requests=total_requests,
            total_cost=total_cost,
            active_integrations=active_integrations,
            active_requests=len(self.active_requests)
        )
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """
        Get comprehensive orchestrator status.
        
        Returns:
            Dictionary with orchestrator status and statistics
        """
        return {
            "is_running": self.is_running,
            "active_integrations": len(self.integrations),
            "active_requests": len(self.active_requests),
            "total_providers": len(APIProvider),
            "usage_statistics": {
                provider.value: {
                    "total_requests": stats.total_requests,
                    "success_rate": stats.success_rate,
                    "average_latency_ms": stats.average_latency_ms,
                    "total_cost": stats.total_cost,
                    "total_tokens": stats.total_tokens,
                }
                for provider, stats in self.usage_stats.items()
            },
            "health_status": self.health_monitor.get_health_report(),
            "rate_limits": self.rate_limiter.get_status(),
            "quotas": self.quota_manager.get_status(),
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """
        Optimize orchestrator performance based on usage patterns.
        
        Returns:
            Dictionary with optimization results
        """
        optimizations = {
            "providers_optimized": 0,
            "quotas_adjusted": 0,
            "rate_limits_adjusted": 0,
        }
        
        # Optimize based on success rates
        for provider, stats in self.usage_stats.items():
            if stats.total_requests > 10:
                if stats.success_rate < 50:
                    # Reduce quota for poorly performing providers
                    self.quota_manager.adjust_quota(provider.value, 0.8)
                    optimizations["quotas_adjusted"] += 1
                elif stats.success_rate > 95:
                    # Increase quota for well-performing providers
                    self.quota_manager.adjust_quota(provider.value, 1.2)
                    optimizations["quotas_adjusted"] += 1
        
        logger.info(f"Performance optimization completed: {optimizations}")
        return optimizations