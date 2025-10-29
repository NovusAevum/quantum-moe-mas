"""
DeepSeek API Integration.

This module provides integration with DeepSeek's API for
advanced reasoning and code generation capabilities.
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


class DeepSeekIntegration(BaseAPIIntegration):
    """
    DeepSeek API integration.
    
    Provides access to DeepSeek's advanced models including:
    - DeepSeek-V3 for general tasks
    - DeepSeek-Coder for programming
    - DeepSeek-Math for mathematical reasoning
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize DeepSeek integration.
        
        Args:
            api_key: Optional API key (overrides DEEPSEEK_API_KEY env var)
        """
        config = IntegrationConfig(
            base_url="https://api.deepseek.com/v1",
            api_key_env_var="DEEPSEEK_API_KEY",
            timeout_seconds=60,
            max_retries=3,
            retry_delay_seconds=2.0,
            rate_limit_per_minute=50,
            cost_per_token=0.00014,  # Very competitive pricing
            max_tokens=8192,
            supports_streaming=True,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Available models on DeepSeek
        self.models = {
            "deepseek-chat": {
                "max_tokens": 8192,
                "cost_per_token": 0.00014,
                "context_window": 8192,
                "description": "General purpose chat model"
            },
            "deepseek-coder": {
                "max_tokens": 8192,
                "cost_per_token": 0.00014,
                "context_window": 8192,
                "description": "Specialized for code generation and programming"
            },
            "deepseek-math": {
                "max_tokens": 4096,
                "cost_per_token": 0.00014,
                "context_window": 4096,
                "description": "Specialized for mathematical reasoning"
            },
            "deepseek-v3": {
                "max_tokens": 8192,
                "cost_per_token": 0.00027,
                "context_window": 8192,
                "description": "Latest and most capable model"
            }
        }
        
        self.default_model = "deepseek-chat"
    
    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get DeepSeek authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.error("DeepSeek API key not found")
            return None
        
        return {"Authorization": f"Bearer {api_key}"}
    
    def get_capabilities(self) -> List[APICapability]:
        """Get DeepSeek capabilities."""
        return [
            APICapability.TEXT_GENERATION,
            APICapability.CHAT_COMPLETION,
            APICapability.CODE_GENERATION,
            APICapability.STREAMING,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DeepSeek request data."""
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
        request_data.setdefault("top_p", 0.9)
        
        return request_data
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to DeepSeek API."""
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
                        "provider": "deepseek",
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
                    error=f"DeepSeek API error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "deepseek"}
                )
                
        except httpx.HTTPStatusError as e:
            return APIResponse.error_response(
                error=f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                metadata={"provider": "deepseek"}
            )
        except Exception as e:
            logger.error(f"DeepSeek request failed: {e}")
            raise APIError(f"DeepSeek request failed: {str(e)}", provider="deepseek")
    
    def _parse_success_response(self, response_data: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse successful DeepSeek response."""
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
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> APIResponse:
        """
        Create a chat completion with DeepSeek.
        
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
    
    async def code_generation(
        self,
        prompt: str,
        language: str = "python",
        model: str = "deepseek-coder",
        **kwargs
    ) -> APIResponse:
        """
        Generate code using DeepSeek-Coder.
        
        Args:
            prompt: Code generation prompt
            language: Programming language
            model: Model to use (defaults to deepseek-coder)
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated code
        """
        # Format prompt for code generation
        messages = [
            {
                "role": "system",
                "content": f"You are DeepSeek-Coder, an expert {language} programmer. Generate clean, efficient, and well-documented code."
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
    
    async def mathematical_reasoning(
        self,
        problem: str,
        model: str = "deepseek-math",
        **kwargs
    ) -> APIResponse:
        """
        Solve mathematical problems using DeepSeek-Math.
        
        Args:
            problem: Mathematical problem to solve
            model: Model to use (defaults to deepseek-math)
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with mathematical solution
        """
        messages = [
            {
                "role": "system",
                "content": "You are DeepSeek-Math, an expert mathematician. Solve problems step by step with clear explanations."
            },
            {
                "role": "user",
                "content": f"Solve this mathematical problem step by step: {problem}"
            }
        ]
        
        return await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.1,  # Very low temperature for math
            max_tokens=2000,
            **kwargs
        )
    
    async def reasoning_task(
        self,
        problem: str,
        model: str = "deepseek-v3",
        **kwargs
    ) -> APIResponse:
        """
        Perform complex reasoning tasks.
        
        Args:
            problem: Problem requiring reasoning
            model: Model to use (defaults to deepseek-v3)
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with reasoning solution
        """
        messages = [
            {
                "role": "system",
                "content": "You are DeepSeek-V3, an advanced AI with strong reasoning capabilities. Think carefully and provide detailed analysis."
            },
            {
                "role": "user",
                "content": f"Analyze and solve this problem: {problem}"
            }
        ]
        
        return await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=2000,
            **kwargs
        )
    
    async def code_review(
        self,
        code: str,
        language: str = "python",
        model: str = "deepseek-coder",
        **kwargs
    ) -> APIResponse:
        """
        Review code for quality, bugs, and improvements.
        
        Args:
            code: Code to review
            language: Programming language
            model: Model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with code review
        """
        messages = [
            {
                "role": "system",
                "content": f"You are an expert {language} code reviewer. Analyze code for bugs, performance issues, security vulnerabilities, and suggest improvements."
            },
            {
                "role": "user",
                "content": f"Please review this {language} code:\n\n```{language}\n{code}\n```"
            }
        ]
        
        return await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=2000,
            **kwargs
        )
    
    async def health_check(self) -> bool:
        """Check DeepSeek API health."""
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
            logger.warning(f"DeepSeek health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.models.get(model)
    
    def get_best_model_for_task(self, task: str) -> str:
        """Get the best model for a specific task."""
        task_models = {
            "code": "deepseek-coder",
            "programming": "deepseek-coder",
            "math": "deepseek-math",
            "mathematics": "deepseek-math",
            "reasoning": "deepseek-v3",
            "analysis": "deepseek-v3",
            "general": "deepseek-chat",
            "chat": "deepseek-chat"
        }
        
        return task_models.get(task.lower(), self.default_model)