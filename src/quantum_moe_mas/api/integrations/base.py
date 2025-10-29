"""
Base API Integration Interface and Common Classes.

This module provides the foundation for all API integrations with
consistent interfaces, error handling, and response structures.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import httpx
from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class APICapability(Enum):
    """API capabilities for different types of requests."""
    
    # Text Generation
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    CODE_GENERATION = "code_generation"
    
    # Vision & Multimodal
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    VISION_CHAT = "vision_chat"
    
    # Audio & Speech
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"
    
    # Embeddings & Search
    TEXT_EMBEDDINGS = "text_embeddings"
    SEMANTIC_SEARCH = "semantic_search"
    
    # Specialized
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"


@dataclass
class APIResponse:
    """Standardized API response structure."""
    
    success: bool
    data: Any
    error: Optional[str] = None
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Performance metrics
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    
    @classmethod
    def success_response(
        cls,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "APIResponse":
        """Create a successful response."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            **kwargs
        )
    
    @classmethod
    def error_response(
        cls,
        error: str,
        status_code: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "APIResponse":
        """Create an error response."""
        return cls(
            success=False,
            data=None,
            error=error,
            status_code=status_code,
            metadata=metadata or {},
            **kwargs
        )


class APIError(Exception):
    """Custom exception for API integration errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        self.provider = provider
        self.timestamp = datetime.utcnow()
    
    def __str__(self) -> str:
        provider_info = f"[{self.provider}] " if self.provider else ""
        status_info = f" (Status: {self.status_code})" if self.status_code else ""
        return f"{provider_info}{self.message}{status_info}"


@dataclass
class IntegrationConfig:
    """Configuration for API integrations."""
    
    base_url: str
    api_key_env_var: str
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    rate_limit_per_minute: int = 60
    cost_per_token: float = 0.0
    max_tokens: int = 4096
    supports_streaming: bool = False
    requires_auth: bool = True
    custom_headers: Optional[Dict[str, str]] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.base_url:
            raise ValueError("base_url is required")
        if self.requires_auth and not self.api_key_env_var:
            raise ValueError("api_key_env_var is required when auth is required")


class BaseAPIIntegration(ABC):
    """
    Abstract base class for all API integrations.
    
    Provides common functionality including:
    - HTTP client management
    - Authentication handling
    - Error handling and retries
    - Response standardization
    - Performance tracking
    """
    
    def __init__(
        self,
        config: IntegrationConfig,
        api_key: Optional[str] = None
    ) -> None:
        """
        Initialize API integration.
        
        Args:
            config: Integration configuration
            api_key: Optional API key (overrides env var)
        """
        self.config = config
        self.api_key = api_key
        self.provider_name = self.__class__.__name__.replace("Integration", "").lower()
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Performance tracking
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        logger.info(
            f"Initialized {self.provider_name} integration",
            base_url=config.base_url,
            timeout=config.timeout_seconds
        )
    
    async def initialize(self) -> None:
        """Initialize the integration and HTTP client."""
        headers = {
            "User-Agent": "QuantumMoEMAS/0.1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        # Add custom headers
        if self.config.custom_headers:
            headers.update(self.config.custom_headers)
        
        # Add authentication
        if self.config.requires_auth:
            auth_header = await self._get_auth_header()
            if auth_header:
                headers.update(auth_header)
        
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        logger.info(f"Initialized HTTP client for {self.provider_name}")
    
    async def shutdown(self) -> None:
        """Shutdown the integration and cleanup resources."""
        if self.client:
            await self.client.aclose()
            self.client = None
        
        logger.info(f"Shutdown {self.provider_name} integration")
    
    @abstractmethod
    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """
        Get authentication header for requests.
        
        Returns:
            Dictionary with authentication headers or None
        """
        pass
    
    @abstractmethod
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """
        Make API request to specific endpoint.
        
        Args:
            endpoint: API endpoint path
            data: Request payload
            method: HTTP method
        
        Returns:
            APIResponse with result
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[APICapability]:
        """
        Get list of capabilities supported by this integration.
        
        Returns:
            List of supported capabilities
        """
        pass
    
    async def make_request(
        self,
        request_data: Dict[str, Any],
        **kwargs
    ) -> APIResponse:
        """
        Main entry point for making requests.
        
        Args:
            request_data: Request payload
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with result
        """
        if not self.client:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Validate request data
            validated_data = await self._validate_request(request_data)
            
            # Make the API request with retries
            response = await self._make_request_with_retries(validated_data, **kwargs)
            
            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            response.latency_ms = latency_ms
            
            self.request_count += 1
            self.total_latency += latency_ms
            
            if not response.success:
                self.error_count += 1
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Request failed for {self.provider_name}: {e}")
            
            return APIResponse.error_response(
                error=str(e),
                metadata={
                    "provider": self.provider_name,
                    "latency_ms": (time.time() - start_time) * 1000
                }
            )
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and transform request data.
        
        Args:
            request_data: Raw request data
        
        Returns:
            Validated request data
        
        Raises:
            APIError: If validation fails
        """
        # Default implementation - subclasses can override
        return request_data
    
    async def _make_request_with_retries(
        self,
        request_data: Dict[str, Any],
        **kwargs
    ) -> APIResponse:
        """Make request with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Determine endpoint and method
                endpoint = kwargs.get("endpoint", "/")
                method = kwargs.get("method", "POST")
                
                response = await self._make_api_request(endpoint, request_data, method)
                
                if response.success:
                    return response
                
                # If not successful but not a retryable error, return immediately
                if not self._is_retryable_error(response):
                    return response
                
                last_error = response.error
                
            except Exception as e:
                last_error = str(e)
                
                if not self._is_retryable_exception(e):
                    raise
            
            # Wait before retry (except on last attempt)
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay_seconds * (2 ** attempt)  # Exponential backoff
                await asyncio.sleep(delay)
                logger.warning(
                    f"Retrying request for {self.provider_name}",
                    attempt=attempt + 1,
                    delay=delay
                )
        
        # All retries exhausted
        error_msg = f"Request failed after {self.config.max_retries + 1} attempts: {last_error}"
        return APIResponse.error_response(
            error=error_msg,
            metadata={"provider": self.provider_name, "attempts": self.config.max_retries + 1}
        )
    
    def _is_retryable_error(self, response: APIResponse) -> bool:
        """Check if error is retryable."""
        if not response.status_code:
            return True
        
        # Retry on server errors and rate limits
        return response.status_code >= 500 or response.status_code == 429
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        # Retry on network errors
        return isinstance(exception, (httpx.NetworkError, httpx.TimeoutException))
    
    async def health_check(self) -> bool:
        """
        Perform health check for the API.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Simple health check - subclasses can override
            if not self.client:
                await self.initialize()
            
            # Make a minimal request to check connectivity
            response = await self.client.get("/", timeout=5.0)
            return response.status_code < 500
            
        except Exception as e:
            logger.warning(f"Health check failed for {self.provider_name}: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this integration.
        
        Returns:
            Dictionary with performance statistics
        """
        avg_latency = (
            self.total_latency / self.request_count
            if self.request_count > 0 else 0.0
        )
        
        success_rate = (
            ((self.request_count - self.error_count) / self.request_count) * 100
            if self.request_count > 0 else 0.0
        )
        
        return {
            "provider": self.provider_name,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate_percent": success_rate,
            "average_latency_ms": avg_latency,
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "config": {
                "base_url": self.config.base_url,
                "timeout_seconds": self.config.timeout_seconds,
                "max_retries": self.config.max_retries,
                "cost_per_token": self.config.cost_per_token,
                "max_tokens": self.config.max_tokens,
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        logger.info(f"Reset metrics for {self.provider_name}")
    
    @property
    def base_url(self) -> str:
        """Get base URL."""
        return self.config.base_url
    
    @property
    def api_key_env_var(self) -> str:
        """Get API key environment variable name."""
        return self.config.api_key_env_var
    
    @property
    def capabilities(self) -> List[str]:
        """Get capabilities as string list."""
        return [cap.value for cap in self.get_capabilities()]
    
    @property
    def cost_per_token(self) -> float:
        """Get cost per token."""
        return self.config.cost_per_token
    
    @property
    def max_tokens(self) -> int:
        """Get maximum tokens."""
        return self.config.max_tokens