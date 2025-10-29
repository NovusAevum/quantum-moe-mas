"""
Replicate API Integration.

This module provides integration with Replicate for
running various AI models including image generation,
text processing, and more.
"""

import os
import asyncio
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


class ReplicateIntegration(BaseAPIIntegration):
    """
    Replicate API integration.
    
    Provides access to thousands of AI models including:
    - Image generation (SDXL, Midjourney, etc.)
    - Text generation (Llama, CodeLlama, etc.)
    - Audio processing (Whisper, MusicGen, etc.)
    - Video generation (Stable Video Diffusion, etc.)
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Replicate integration.
        
        Args:
            api_key: Optional API key (overrides REPLICATE_API_TOKEN env var)
        """
        config = IntegrationConfig(
            base_url="https://api.replicate.com/v1",
            api_key_env_var="REPLICATE_API_TOKEN",
            timeout_seconds=300,  # Models can take a while
            max_retries=3,
            retry_delay_seconds=5.0,
            rate_limit_per_minute=100,
            cost_per_token=0.001,  # Varies by model
            max_tokens=4096,
            supports_streaming=False,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Popular models on Replicate
        self.models = {
            # Image Generation
            "sdxl": {
                "model_id": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                "type": "image_generation",
                "cost_estimate": 0.0025
            },
            "flux-schnell": {
                "model_id": "black-forest-labs/flux-schnell:bf2f2e683d03a9549f484a37a0df1581514b28b3b4731b154c7b36790a4c5b1b",
                "type": "image_generation", 
                "cost_estimate": 0.003
            },
            "midjourney": {
                "model_id": "tstramer/midjourney-diffusion:436b051ebd8f68d23e83d22de5e198e0995357afef113768c20f0b6fcef23c8b",
                "type": "image_generation",
                "cost_estimate": 0.005
            },
            
            # Text Generation
            "llama-2-70b": {
                "model_id": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                "type": "text_generation",
                "cost_estimate": 0.00065
            },
            "codellama-34b": {
                "model_id": "meta/codellama-34b-instruct:8fefae79b74e40e8b8c538c9e4057b7a6a4395a9b6b6e8e0e0e0e0e0e0e0e0e0",
                "type": "code_generation",
                "cost_estimate": 0.0005
            },
            
            # Audio
            "whisper": {
                "model_id": "openai/whisper:4d50797290df275329f202e48c76360b3f22b08d28c196cbc54600319435f8d2",
                "type": "speech_to_text",
                "cost_estimate": 0.0001
            },
            "musicgen": {
                "model_id": "meta/musicgen:b05b1dff1d8c6dc63d14b0cdb42135378dcb87f6373b0d3d341ede46e59e2dbe",
                "type": "audio_generation",
                "cost_estimate": 0.002
            }
        }
        
        self.default_model = "sdxl"    

    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get Replicate authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.error("Replicate API token not found")
            return None
        
        return {"Authorization": f"Token {api_key}"}
    
    def get_capabilities(self) -> List[APICapability]:
        """Get Replicate capabilities."""
        return [
            APICapability.IMAGE_GENERATION,
            APICapability.TEXT_GENERATION,
            APICapability.CODE_GENERATION,
            APICapability.SPEECH_TO_TEXT,
            APICapability.IMAGE_ANALYSIS,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Replicate request data."""
        # Ensure model is specified
        if "model" not in request_data:
            request_data["model"] = self.default_model
        
        # Ensure input is provided
        if "input" not in request_data:
            # Try to construct input from common fields
            if "prompt" in request_data:
                request_data["input"] = {"prompt": request_data.pop("prompt")}
            else:
                raise APIError("'input' field is required for Replicate API")
        
        return request_data
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to Replicate API."""
        try:
            model = data.pop("model", self.default_model)
            model_config = self.models.get(model, {})
            model_id = model_config.get("model_id", model)
            
            # Create prediction
            prediction_data = {
                "version": model_id,
                "input": data.get("input", {})
            }
            
            response = await self.client.post(
                "/predictions",
                json=prediction_data
            )
            
            if response.status_code == 201:
                prediction = response.json()
                prediction_id = prediction["id"]
                
                # Wait for completion
                result = await self._wait_for_completion(prediction_id)
                
                return APIResponse.success_response(
                    data=result,
                    status_code=200,
                    metadata={
                        "provider": "replicate",
                        "model": model,
                        "model_id": model_id,
                        "prediction_id": prediction_id,
                        "cost": model_config.get("cost_estimate", 0.001),
                    }
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                return APIResponse.error_response(
                    error=f"Replicate API error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "replicate"}
                )
                
        except Exception as e:
            logger.error(f"Replicate request failed: {e}")
            raise APIError(f"Replicate request failed: {str(e)}", provider="replicate")
    
    async def _wait_for_completion(self, prediction_id: str, max_wait: int = 300) -> Dict[str, Any]:
        """Wait for prediction to complete."""
        for _ in range(max_wait // 5):  # Check every 5 seconds
            response = await self.client.get(f"/predictions/{prediction_id}")
            
            if response.status_code == 200:
                prediction = response.json()
                status = prediction.get("status")
                
                if status == "succeeded":
                    return {
                        "status": "completed",
                        "output": prediction.get("output"),
                        "input": prediction.get("input", {}),
                        "prediction_id": prediction_id,
                        "metrics": prediction.get("metrics", {})
                    }
                elif status == "failed":
                    error_msg = prediction.get("error", "Prediction failed")
                    raise APIError(f"Prediction failed: {error_msg}")
                elif status in ["starting", "processing"]:
                    await asyncio.sleep(5)
                    continue
            
            await asyncio.sleep(5)
        
        raise APIError("Prediction timed out")
    
    async def generate_image(
        self,
        prompt: str,
        model: str = "sdxl",
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> APIResponse:
        """
        Generate an image using Replicate models.
        
        Args:
            prompt: Text description of the image
            model: Model to use
            width: Image width
            height: Image height
            **kwargs: Additional model-specific parameters
        
        Returns:
            APIResponse with generated image URLs
        """
        request_data = {
            "model": model,
            "input": {
                "prompt": prompt,
                "width": width,
                "height": height,
                **kwargs
            }
        }
        
        return await self.make_request(request_data)
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "llama-2-70b",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> APIResponse:
        """
        Generate text using Replicate language models.
        
        Args:
            prompt: Text prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated text
        """
        request_data = {
            "model": model,
            "input": {
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        return await self.make_request(request_data)
    
    async def transcribe_audio(
        self,
        audio_url: str,
        model: str = "whisper",
        **kwargs
    ) -> APIResponse:
        """
        Transcribe audio using Whisper on Replicate.
        
        Args:
            audio_url: URL to audio file
            model: Model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with transcription
        """
        request_data = {
            "model": model,
            "input": {
                "audio": audio_url,
                **kwargs
            }
        }
        
        return await self.make_request(request_data)
    
    async def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        model: str = "musicgen",
        **kwargs
    ) -> APIResponse:
        """
        Generate music using MusicGen on Replicate.
        
        Args:
            prompt: Description of the music
            duration: Duration in seconds
            model: Model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated music URL
        """
        request_data = {
            "model": model,
            "input": {
                "prompt": prompt,
                "duration": duration,
                **kwargs
            }
        }
        
        return await self.make_request(request_data)
    
    async def run_custom_model(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> APIResponse:
        """
        Run a custom model on Replicate.
        
        Args:
            model_id: Full model ID (owner/name:version)
            input_data: Input parameters for the model
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with model output
        """
        request_data = {
            "model": model_id,
            "input": input_data
        }
        
        return await self.make_request(request_data)
    
    async def get_model_info(self, model_id: str) -> APIResponse:
        """
        Get information about a model.
        
        Args:
            model_id: Model ID to query
        
        Returns:
            APIResponse with model information
        """
        try:
            response = await self.client.get(f"/models/{model_id}")
            
            if response.status_code == 200:
                model_info = response.json()
                return APIResponse.success_response(
                    data=model_info,
                    metadata={"provider": "replicate", "operation": "model_info"}
                )
            else:
                return APIResponse.error_response(
                    error=f"Failed to get model info: HTTP {response.status_code}",
                    status_code=response.status_code
                )
        except Exception as e:
            return APIResponse.error_response(
                error=f"Model info error: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Check Replicate API health."""
        try:
            # Check if we can access the API
            response = await self.client.get("/", timeout=10.0)
            return response.status_code in [200, 404]  # 404 is OK for root
        except Exception as e:
            logger.warning(f"Replicate health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_config(self, model: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        return self.models.get(model)
    
    def get_models_by_type(self, model_type: str) -> List[str]:
        """Get models of a specific type."""
        return [
            model for model, config in self.models.items()
            if config.get("type") == model_type
        ]