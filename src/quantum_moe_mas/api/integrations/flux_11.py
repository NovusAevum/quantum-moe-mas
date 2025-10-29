"""
Flux 1.1 API Integration.

This module provides integration with Flux 1.1 for
advanced image generation capabilities.
"""

import os
import base64
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


class Flux11Integration(BaseAPIIntegration):
    """
    Flux 1.1 API integration for image generation.
    
    Provides access to Flux 1.1's advanced image generation
    models through Replicate or direct API.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Flux 1.1 integration.
        
        Args:
            api_key: Optional API key (overrides FLUX_API_KEY env var)
        """
        config = IntegrationConfig(
            base_url="https://api.replicate.com/v1",
            api_key_env_var="FLUX_API_KEY",
            timeout_seconds=120,  # Image generation takes longer
            max_retries=3,
            retry_delay_seconds=5.0,
            rate_limit_per_minute=10,
            cost_per_token=0.003,  # Per image cost
            max_tokens=1,  # Not applicable for images
            supports_streaming=False,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Flux 1.1 model configurations
        self.models = {
            "flux-1.1-pro": {
                "model_id": "black-forest-labs/flux-1.1-pro",
                "max_resolution": "2048x2048",
                "cost_per_image": 0.05,
                "description": "Highest quality Flux model"
            },
            "flux-1.1-dev": {
                "model_id": "black-forest-labs/flux-dev",
                "max_resolution": "1024x1024", 
                "cost_per_image": 0.025,
                "description": "Development version of Flux"
            },
            "flux-schnell": {
                "model_id": "black-forest-labs/flux-schnell",
                "max_resolution": "1024x1024",
                "cost_per_image": 0.003,
                "description": "Fast generation Flux model"
            }
        }
        
        self.default_model = "flux-1.1-pro"    
  
  async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get Flux authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.error("Flux API key not found")
            return None
        
        return {"Authorization": f"Token {api_key}"}
    
    def get_capabilities(self) -> List[APICapability]:
        """Get Flux capabilities."""
        return [
            APICapability.IMAGE_GENERATION,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Flux request data."""
        # Ensure prompt is provided
        if "prompt" not in request_data:
            raise APIError("'prompt' is required for image generation")
        
        # Set default model if not specified
        if "model" not in request_data:
            request_data["model"] = self.default_model
        
        # Validate model
        model = request_data["model"]
        if model not in self.models:
            logger.warning(f"Unknown model {model}, using default")
            request_data["model"] = self.default_model
        
        # Set reasonable defaults
        request_data.setdefault("width", 1024)
        request_data.setdefault("height", 1024)
        request_data.setdefault("num_outputs", 1)
        request_data.setdefault("guidance_scale", 3.5)
        request_data.setdefault("num_inference_steps", 28)
        
        return request_data
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to Flux API via Replicate."""
        try:
            model = data.pop("model", self.default_model)
            model_config = self.models[model]
            model_id = model_config["model_id"]
            
            # Create prediction
            prediction_data = {
                "version": model_id,
                "input": data
            }
            
            response = await self.client.post(
                "/predictions",
                json=prediction_data
            )
            
            if response.status_code == 201:
                prediction = response.json()
                prediction_id = prediction["id"]
                
                # Poll for completion
                result = await self._wait_for_completion(prediction_id)
                
                return APIResponse.success_response(
                    data=result,
                    status_code=200,
                    metadata={
                        "provider": "flux_11",
                        "model": model,
                        "prediction_id": prediction_id,
                        "cost": model_config["cost_per_image"],
                    }
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                return APIResponse.error_response(
                    error=f"Flux API error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "flux_11"}
                )
                
        except Exception as e:
            logger.error(f"Flux request failed: {e}")
            raise APIError(f"Flux request failed: {str(e)}", provider="flux_11")
    
    async def _wait_for_completion(self, prediction_id: str, max_wait: int = 300) -> Dict[str, Any]:
        """Wait for prediction to complete."""
        import asyncio
        
        for _ in range(max_wait // 5):  # Check every 5 seconds
            response = await self.client.get(f"/predictions/{prediction_id}")
            
            if response.status_code == 200:
                prediction = response.json()
                status = prediction.get("status")
                
                if status == "succeeded":
                    return {
                        "status": "completed",
                        "images": prediction.get("output", []),
                        "prompt": prediction.get("input", {}).get("prompt", ""),
                        "prediction_id": prediction_id
                    }
                elif status == "failed":
                    error_msg = prediction.get("error", "Generation failed")
                    raise APIError(f"Image generation failed: {error_msg}")
                elif status in ["starting", "processing"]:
                    await asyncio.sleep(5)
                    continue
            
            await asyncio.sleep(5)
        
        raise APIError("Image generation timed out")
    
    async def generate_image(
        self,
        prompt: str,
        model: str = "flux-1.1-pro",
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        **kwargs
    ) -> APIResponse:
        """
        Generate an image using Flux 1.1.
        
        Args:
            prompt: Text description of the image
            model: Model to use
            width: Image width
            height: Image height
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated image URLs
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def generate_multiple_images(
        self,
        prompt: str,
        num_images: int = 4,
        model: str = "flux-1.1-pro",
        **kwargs
    ) -> APIResponse:
        """
        Generate multiple images from the same prompt.
        
        Args:
            prompt: Text description
            num_images: Number of images to generate
            model: Model to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with multiple image URLs
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "num_outputs": num_images,
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def health_check(self) -> bool:
        """Check Flux API health."""
        try:
            # Check if we can access the API
            response = await self.client.get("/", timeout=10.0)
            return response.status_code in [200, 404]  # 404 is OK for root endpoint
        except Exception as e:
            logger.warning(f"Flux health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.models.get(model)
    
    def get_supported_resolutions(self, model: str) -> List[str]:
        """Get supported resolutions for a model."""
        # Common resolutions supported by Flux
        return [
            "512x512", "768x768", "1024x1024",
            "512x768", "768x512", "1024x768", "768x1024",
            "512x1024", "1024x512", "1536x1024", "1024x1536"
        ]