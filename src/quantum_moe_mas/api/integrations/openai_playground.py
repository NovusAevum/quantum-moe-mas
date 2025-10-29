"""
OpenAI Playground API Integration.

This module provides integration with OpenAI's Playground API for
text generation, chat completion, and code generation capabilities.
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


class OpenAIPlaygroundIntegration(BaseAPIIntegration):
    """
    OpenAI Playground API integration.
    
    Provides access to OpenAI's models including GPT-4, GPT-3.5-turbo,
    and other language models through the playground interface.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize OpenAI Playground integration.
        
        Args:
            api_key: Optional API key (overrides OPENAI_API_KEY env var)
        """
        config = IntegrationConfig(
            base_url="https://api.openai.com/v1",
            api_key_env_var="OPENAI_API_KEY",
            timeout_seconds=60,
            max_retries=3,
            retry_delay_seconds=1.0,
            rate_limit_per_minute=60,
            cost_per_token=0.00002,  # Approximate cost for GPT-3.5-turbo
            max_tokens=4096,
            supports_streaming=True,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Model configurations
        self.models = {
            "gpt-4": {"max_tokens": 8192, "cost_per_token": 0.00006},
            "gpt-4-turbo": {"max_tokens": 128000, "cost_per_token": 0.00003},
            "gpt-3.5-turbo": {"max_tokens": 4096, "cost_per_token": 0.000002},
            "gpt-3.5-turbo-16k": {"max_tokens": 16384, "cost_per_token": 0.000004},
            "text-davinci-003": {"max_tokens": 4097, "cost_per_token": 0.00002},
            "code-davinci-002": {"max_tokens": 8001, "cost_per_token": 0.00002},
        }
        
        self.default_model = "gpt-3.5-turbo"
    
    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get OpenAI authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.error("OpenAI API key not found")
            return None
        
        return {"Authorization": f"Bearer {api_key}"}
    
    def get_capabilities(self) -> List[APICapability]:
        """Get OpenAI capabilities."""
        return [
            APICapability.TEXT_GENERATION,
            APICapability.CHAT_COMPLETION,
            APICapability.CODE_GENERATION,
            APICapability.FUNCTION_CALLING,
            APICapability.JSON_MODE,
            APICapability.STREAMING,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OpenAI request data."""
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
        
        return request_data
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to OpenAI API."""
        try:
            # Determine the correct endpoint based on request type
            if "messages" in data:
                endpoint = "/chat/completions"
            elif "prompt" in data:
                endpoint = "/completions"
            else:
                endpoint = "/chat/completions"  # Default to chat
            
            response = await self.client.request(
                method=method,
                url=endpoint,
                json=data
            )
            
            response_data = response.json()
            
            if response.status_code == 200:
                # Extract relevant information
                result = self._parse_success_response(response_data, data)
                return APIResponse.success_response(
                    data=result,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metadata={
                        "provider": "openai_playground",
                        "model": data.get("model", self.default_model),
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                        "cost": self._calculate_cost(result.get("usage", {}), data.get("model")),
                    }
                )
            else:
                error_msg = response_data.get("error", {}).get("message", "Unknown error")
                return APIResponse.error_response(
                    error=f"OpenAI API error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "openai_playground"}
                )
                
        except httpx.HTTPStatusError as e:
            return APIResponse.error_response(
                error=f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                metadata={"provider": "openai_playground"}
            )
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise APIError(f"OpenAI request failed: {str(e)}", provider="openai_playground")
    
    def _parse_success_response(self, response_data: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse successful OpenAI response."""
        result = {
            "id": response_data.get("id"),
            "object": response_data.get("object"),
            "created": response_data.get("created"),
            "model": response_data.get("model"),
            "usage": response_data.get("usage", {}),
        }
        
        # Extract content based on response type
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
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> APIResponse:
        """
        Create a chat completion.
        
        Args:
            messages: List of message objects
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with chat completion
        """
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def text_completion(
        self,
        prompt: str,
        model: str = "text-davinci-003",
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
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def code_completion(
        self,
        prompt: str,
        language: str = "python",
        max_tokens: int = 1000,
        **kwargs
    ) -> APIResponse:
        """
        Generate code completion.
        
        Args:
            prompt: Code prompt
            language: Programming language
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with code completion
        """
        # Format prompt for code generation
        formatted_prompt = f"# {language.title()} code:\n{prompt}\n\n"
        
        request_data = {
            "model": "code-davinci-002",
            "prompt": formatted_prompt,
            "temperature": 0.2,  # Lower temperature for code
            "max_tokens": max_tokens,
            "stop": ["\n\n", "# End"],
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            response = await self.client.get("/models", timeout=10.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.models.get(model)