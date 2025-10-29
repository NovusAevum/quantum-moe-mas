"""
Google AI Studio (Gemini) API Integration.

This module provides integration with Google's AI Studio API for
accessing Gemini models and other Google AI capabilities.
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


class GoogleAIStudioIntegration(BaseAPIIntegration):
    """
    Google AI Studio (Gemini) API integration.
    
    Provides access to Google's Gemini models including:
    - Gemini Pro (text and code)
    - Gemini Pro Vision (multimodal)
    - PaLM 2 models
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Google AI Studio integration.
        
        Args:
            api_key: Optional API key (overrides GOOGLE_AI_STUDIO_API_KEY env var)
        """
        config = IntegrationConfig(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env_var="GOOGLE_AI_STUDIO_API_KEY",
            timeout_seconds=60,
            max_retries=3,
            retry_delay_seconds=2.0,
            rate_limit_per_minute=60,
            cost_per_token=0.00025,  # Approximate cost for Gemini Pro
            max_tokens=32768,
            supports_streaming=True,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Model configurations
        self.models = {
            "gemini-pro": {
                "max_tokens": 32768,
                "cost_per_token": 0.00025,
                "supports_vision": False,
                "supports_function_calling": True
            },
            "gemini-pro-vision": {
                "max_tokens": 16384,
                "cost_per_token": 0.00025,
                "supports_vision": True,
                "supports_function_calling": False
            },
            "gemini-1.5-pro": {
                "max_tokens": 1048576,  # 1M tokens
                "cost_per_token": 0.0035,
                "supports_vision": True,
                "supports_function_calling": True
            },
            "text-bison-001": {
                "max_tokens": 8192,
                "cost_per_token": 0.0005,
                "supports_vision": False,
                "supports_function_calling": False
            },
            "chat-bison-001": {
                "max_tokens": 4096,
                "cost_per_token": 0.0005,
                "supports_vision": False,
                "supports_function_calling": False
            }
        }
        
        self.default_model = "gemini-pro"
    
    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get Google AI Studio authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.error("Google AI Studio API key not found")
            return None
        
        # Google AI Studio uses API key as query parameter, not header
        return None
    
    def get_capabilities(self) -> List[APICapability]:
        """Get Google AI Studio capabilities."""
        return [
            APICapability.TEXT_GENERATION,
            APICapability.CHAT_COMPLETION,
            APICapability.CODE_GENERATION,
            APICapability.IMAGE_ANALYSIS,
            APICapability.VISION_CHAT,
            APICapability.FUNCTION_CALLING,
            APICapability.STREAMING,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Google AI Studio request data."""
        # Ensure model is specified
        if "model" not in request_data:
            request_data["model"] = self.default_model
        
        # Validate model
        model = request_data["model"]
        if model not in self.models:
            logger.warning(f"Unknown model {model}, using default")
            request_data["model"] = self.default_model
        
        # Convert messages format if needed
        if "messages" in request_data:
            request_data["contents"] = self._convert_messages_to_contents(request_data.pop("messages"))
        elif "prompt" in request_data:
            # Convert simple prompt to contents format
            request_data["contents"] = [{
                "parts": [{"text": request_data.pop("prompt")}]
            }]
        
        # Ensure contents exist
        if "contents" not in request_data:
            raise APIError("Either 'contents', 'messages', or 'prompt' is required")
        
        # Set generation config
        if "generationConfig" not in request_data:
            request_data["generationConfig"] = {}
        
        # Set reasonable defaults
        gen_config = request_data["generationConfig"]
        gen_config.setdefault("maxOutputTokens", 1000)
        gen_config.setdefault("temperature", 0.7)
        gen_config.setdefault("topP", 0.8)
        gen_config.setdefault("topK", 40)
        
        return request_data
    
    def _convert_messages_to_contents(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Google contents format."""
        contents = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map roles
            if role == "assistant":
                role = "model"
            elif role == "system":
                # System messages are handled differently in Gemini
                role = "user"
                content = f"System: {content}"
            
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        return contents
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to Google AI Studio API."""
        try:
            # Get API key for query parameter
            api_key = self.api_key or os.getenv(self.config.api_key_env_var)
            if not api_key:
                raise APIError("Google AI Studio API key not found")
            
            # Extract model and build endpoint
            model = data.pop("model", self.default_model)
            
            # Determine endpoint based on request type
            if "contents" in data:
                endpoint = f"/models/{model}:generateContent"
            else:
                endpoint = f"/models/{model}:generateText"
            
            # Add API key as query parameter
            params = {"key": api_key}
            
            response = await self.client.request(
                method=method,
                url=endpoint,
                json=data,
                params=params
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Parse response
                result = self._parse_success_response(response_data, model)
                
                return APIResponse.success_response(
                    data=result,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metadata={
                        "provider": "google_ai_studio",
                        "model": model,
                        "tokens_used": result.get("usage", {}).get("totalTokens", 0),
                        "cost": self._calculate_cost(result.get("usage", {}), model),
                    }
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                return APIResponse.error_response(
                    error=f"Google AI Studio error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "google_ai_studio", "model": model}
                )
                
        except httpx.HTTPStatusError as e:
            return APIResponse.error_response(
                error=f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                metadata={"provider": "google_ai_studio"}
            )
        except Exception as e:
            logger.error(f"Google AI Studio request failed: {e}")
            raise APIError(f"Google AI Studio request failed: {str(e)}", provider="google_ai_studio")
    
    def _parse_success_response(self, response_data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Parse successful Google AI Studio response."""
        result = {
            "model": model,
            "usage": response_data.get("usageMetadata", {}),
        }
        
        # Extract content from candidates
        if "candidates" in response_data and response_data["candidates"]:
            candidate = response_data["candidates"][0]
            
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    result["content"] = parts[0]["text"]
            
            # Add safety ratings and finish reason
            result["finishReason"] = candidate.get("finishReason")
            result["safetyRatings"] = candidate.get("safetyRatings", [])
        
        # Handle prompt feedback
        if "promptFeedback" in response_data:
            result["promptFeedback"] = response_data["promptFeedback"]
        
        return result
    
    def _calculate_cost(self, usage: Dict[str, Any], model: str) -> float:
        """Calculate request cost based on token usage."""
        if not usage or not model:
            return 0.0
        
        model_config = self.models.get(model, {"cost_per_token": self.config.cost_per_token})
        
        # Google uses different token counting
        prompt_tokens = usage.get("promptTokenCount", 0)
        candidates_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = prompt_tokens + candidates_tokens
        
        return total_tokens * model_config["cost_per_token"]
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> APIResponse:
        """
        Create a chat completion using Gemini.
        
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
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        return await self.make_request(request_data)
    
    async def text_generation(
        self,
        prompt: str,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> APIResponse:
        """
        Generate text using Gemini.
        
        Args:
            prompt: Text prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated text
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        return await self.make_request(request_data)
    
    async def vision_chat(
        self,
        text: str,
        image_data: bytes,
        model: str = "gemini-pro-vision",
        **kwargs
    ) -> APIResponse:
        """
        Chat with vision capabilities.
        
        Args:
            text: Text prompt
            image_data: Image bytes (JPEG, PNG, WebP)
            model: Vision model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with vision analysis
        """
        import base64
        
        # Encode image to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        request_data = {
            "model": model,
            "contents": [{
                "parts": [
                    {"text": text},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",  # Assume JPEG, could be detected
                            "data": image_b64
                        }
                    }
                ]
            }],
            "generationConfig": kwargs
        }
        
        return await self.make_request(request_data)
    
    async def code_generation(
        self,
        prompt: str,
        language: str = "python",
        model: str = "gemini-pro",
        **kwargs
    ) -> APIResponse:
        """
        Generate code using Gemini.
        
        Args:
            prompt: Code generation prompt
            language: Programming language
            model: Model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated code
        """
        # Format prompt for code generation
        formatted_prompt = f"Generate {language} code for: {prompt}\n\nCode:"
        
        request_data = {
            "model": model,
            "prompt": formatted_prompt,
            "generationConfig": {
                "maxOutputTokens": 2000,
                "temperature": 0.2,  # Lower temperature for code
                **kwargs
            }
        }
        
        return await self.make_request(request_data)
    
    async def health_check(self) -> bool:
        """Check Google AI Studio API health."""
        try:
            api_key = self.api_key or os.getenv(self.config.api_key_env_var)
            if not api_key:
                return False
            
            # Test with a simple request
            response = await self.client.get(
                "/models",
                params={"key": api_key},
                timeout=10.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Google AI Studio health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.models.get(model)
    
    def supports_vision(self, model: str) -> bool:
        """Check if model supports vision capabilities."""
        model_config = self.models.get(model, {})
        return model_config.get("supports_vision", False)
    
    def supports_function_calling(self, model: str) -> bool:
        """Check if model supports function calling."""
        model_config = self.models.get(model, {})
        return model_config.get("supports_function_calling", False)