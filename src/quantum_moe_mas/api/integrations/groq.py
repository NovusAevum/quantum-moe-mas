"""
Groq API Integration.

This module provides integration with Groq's ultra-fast inference API
for lightning-speed language model inference.
"""

import os
from typing import Dict, List, Optional, Any

import httpx
from quantum_moe_mas.api.integrations.base import (
    BaseAPIIntegration,
    APIResponse,
    APIError,
    APICapability,
    IntegrationConfig
)
from quantum_moe_mas.core.logging_simple import get_logger

logger = get_logger(__name__)


class GroqIntegration(BaseAPIIntegration):
    """
    Groq API integration for ultra-fast inference.
    
    Provides access to optimized models running on Groq's
    Language Processing Units (LPUs) for extremely fast inference.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Groq integration.
        
        Args:
            api_key: Optional API key (overrides GROQ_API_KEY env var)
        """
        config = IntegrationConfig(
            base_url="https://api.groq.com/openai/v1",
            api_key_env_var="GROQ_API_KEY",
            timeout_seconds=30,
            max_retries=3,
            retry_delay_seconds=1.0,
            rate_limit_per_minute=30,
            cost_per_token=0.00027,  # Competitive pricing
            max_tokens=8192,
            supports_streaming=True,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Available models on Groq
        self.models = {
            "llama2-70b-4096": {
                "max_tokens": 4096,
                "cost_per_token": 0.00070,
                "context_window": 4096
            },
            "mixtral-8x7b-32768": {
                "max_tokens": 32768,
                "cost_per_token": 0.00027,
                "context_window": 32768
            },
            "gemma-7b-it": {
                "max_tokens": 8192,
                "cost_per_token": 0.00015,
                "context_window": 8192
            },
            "llama3-8b-8192": {
                "max_tokens": 8192,
                "cost_per_token": 0.00005,
                "context_window": 8192
            },
            "llama3-70b-8192": {
                "max_tokens": 8192,
                "cost_per_token": 0.00059,
                "context_window": 8192
            }
        }
        
        self.default_model = "llama3-8b-8192"
    
    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get Groq authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.error("Groq API key not found")
            return None
        
        return {"Authorization": f"Bearer {api_key}"}
    
    def get_capabilities(self) -> List[APICapability]:
        """Get Groq capabilities."""
        return [
            APICapability.TEXT_GENERATION,
            APICapability.CHAT_COMPLETION,
            APICapability.CODE_GENERATION,
            APICapability.STREAMING,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Groq request data."""
        # Ensure required fields
        if "messages" not in request_data and "prompt" not in request_data:
            raise APIError("Either 'messages' or 'prompt' is required")
        
        # Set default model if not specified
        if "model" not in request_data:
            request_data["model"] = self.default_model
        
        # Validate model
        model = request_data["model"]
        if model not in self.models:
            logger.warning(f"Unknown model {model}, using default")
            request_data["model"] = self.default_model
        
        # Set reasonable defaults
        request_data.setdefault("max_tokens", 1000)
        request_data.setdefault("temperature", 0.7)
        request_data.setdefault("top_p", 1.0)
        
        return request_data
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to Groq API."""
        try:
            # Determine endpoint based on request type
            if "messages" in data:
                endpoint = "/chat/completions"
            else:
                endpoint = "/completions"
            
            response = await self.client.request(
                method=method,
                url=endpoint,
                json=data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Parse response
                result = self._parse_success_response(response_data, data)
                
                return APIResponse.success_response(
                    data=result,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metadata={
                        "provider": "groq",
                        "model": data.get("model", self.default_model),
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                        "cost": self._calculate_cost(result.get("usage", {}), data.get("model")),
                    }
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                return APIResponse.error_response(
                    error=f"Groq API error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "groq"}
                )
                
        except httpx.HTTPStatusError as e:
            return APIResponse.error_response(
                error=f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                metadata={"provider": "groq"}
            )
        except Exception as e:
            logger.error(f"Groq request failed: {e}")
            raise APIError(f"Groq request failed: {str(e)}", provider="groq")
    
    def _parse_success_response(self, response_data: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse successful Groq response."""
        result = {
            "id": response_data.get("id"),
            "object": response_data.get("object"),
            "created": response_data.get("created"),
            "model": response_data.get("model"),
            "usage": response_data.get("usage", {}),
        }
        
        # Extract content from choices
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            
            if "message" in choice:
                # Chat completion response
                result["content"] = choice["message"].get("content", "")
                result["role"] = choice["message"].get("role", "assistant")
                result["finish_reason"] = choice.get("finish_reason")
            elif "text" in choice:
                # Text completion response
                result["content"] = choice.get("text", "")
                result["finish_reason"] = choice.get("finish_reason")
        
        return result
    
    def _calculate_cost(self, usage: Dict[str, Any], model: Optional[str]) -> float:
        """Calculate request cost based on token usage."""
        if not usage or not model:
            return 0.0
        
        model_config = self.models.get(model, {"cost_per_token": self.config.cost_per_token})
        total_tokens = usage.get("total_tokens", 0)
        
        return total_tokens * model_config["cost_per_token"]
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3-8b-8192",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> APIResponse:
        """
        Create a chat completion with ultra-fast inference.
        
        Args:
            messages: List of message objects
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with chat completion
        """
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def text_completion(
        self,
        prompt: str,
        model: str = "llama3-8b-8192",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> APIResponse:
        """
        Create a text completion.
        
        Args:
            prompt: Text prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with text completion
        """
        # Convert to chat format for consistency
        messages = [{"role": "user", "content": prompt}]
        
        return await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def code_generation(
        self,
        prompt: str,
        language: str = "python",
        model: str = "llama3-70b-8192",  # Use larger model for code
        **kwargs
    ) -> APIResponse:
        """
        Generate code with ultra-fast inference.
        
        Args:
            prompt: Code generation prompt
            language: Programming language
            model: Model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated code
        """
        # Format prompt for code generation
        messages = [
            {
                "role": "system",
                "content": f"You are an expert {language} programmer. Generate clean, efficient, and well-commented code."
            },
            {
                "role": "user", 
                "content": f"Generate {language} code for: {prompt}"
            }
        ]
        
        return await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.2,  # Lower temperature for code
            max_tokens=2000,
            **kwargs
        )
    
    async def health_check(self) -> bool:
        """Check Groq API health."""
        try:
            # Test with a minimal request
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.default_model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1
                },
                timeout=10.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.models.get(model)
    
    def get_fastest_model(self) -> str:
        """Get the fastest model for quick responses."""
        return "llama3-8b-8192"  # Fastest on Groq LPUs
    
    def get_most_capable_model(self) -> str:
        """Get the most capable model for complex tasks."""
        return "llama3-70b-8192"  # Largest available model