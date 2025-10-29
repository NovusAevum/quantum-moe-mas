"""
Stability AI API Integration.

This module provides integration with Stability AI for
image generation using Stable Diffusion models.
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


class StabilityAIIntegration(BaseAPIIntegration):
    """
    Stability AI API integration for image generation.
    
    Provides access to Stable Diffusion models including:
    - SDXL models
    - SD 3.0 models
    - Stable Video Diffusion
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Stability AI integration.
        
        Args:
            api_key: Optional API key (overrides STABILITY_AI_API_KEY env var)
        """
        config = IntegrationConfig(
            base_url="https://api.stability.ai/v1",
            api_key_env_var="STABILITY_AI_API_KEY",
            timeout_seconds=120,
            max_retries=3,
            retry_delay_seconds=3.0,
            rate_limit_per_minute=150,
            cost_per_token=0.02,  # Per image cost
            max_tokens=1,
            supports_streaming=False,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Available engines/models
        self.engines = {
            "stable-diffusion-xl-1024-v1-0": {
                "cost_per_image": 0.02,
                "max_resolution": "1024x1024",
                "description": "SDXL 1.0 - High quality images"
            },
            "stable-diffusion-v1-6": {
                "cost_per_image": 0.02,
                "max_resolution": "512x512",
                "description": "SD 1.6 - Fast generation"
            },
            "stable-diffusion-xl-beta-v2-2-2": {
                "cost_per_image": 0.02,
                "max_resolution": "512x512",
                "description": "SDXL Beta - Experimental"
            }
        }
        
        self.default_engine = "stable-diffusion-xl-1024-v1-0"   
 
    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get Stability AI authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.error("Stability AI API key not found")
            return None
        
        return {"Authorization": f"Bearer {api_key}"}
    
    def get_capabilities(self) -> List[APICapability]:
        """Get Stability AI capabilities."""
        return [
            APICapability.IMAGE_GENERATION,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Stability AI request data."""
        # Ensure text prompts are provided
        if "text_prompts" not in request_data and "prompt" not in request_data:
            raise APIError("'text_prompts' or 'prompt' is required")
        
        # Convert simple prompt to text_prompts format
        if "prompt" in request_data and "text_prompts" not in request_data:
            request_data["text_prompts"] = [
                {"text": request_data.pop("prompt"), "weight": 1.0}
            ]
        
        # Set default engine if not specified
        if "engine" not in request_data:
            request_data["engine"] = self.default_engine
        
        # Set reasonable defaults
        request_data.setdefault("cfg_scale", 7)
        request_data.setdefault("height", 1024)
        request_data.setdefault("width", 1024)
        request_data.setdefault("samples", 1)
        request_data.setdefault("steps", 30)
        
        return request_data
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to Stability AI API."""
        try:
            engine = data.pop("engine", self.default_engine)
            endpoint = f"/generation/{engine}/text-to-image"
            
            response = await self.client.request(
                method=method,
                url=endpoint,
                json=data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Parse response
                result = self._parse_success_response(response_data, engine)
                
                return APIResponse.success_response(
                    data=result,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metadata={
                        "provider": "stability_ai",
                        "engine": engine,
                        "cost": self.engines.get(engine, {}).get("cost_per_image", 0.02),
                    }
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                return APIResponse.error_response(
                    error=f"Stability AI error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "stability_ai"}
                )
                
        except httpx.HTTPStatusError as e:
            return APIResponse.error_response(
                error=f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                metadata={"provider": "stability_ai"}
            )
        except Exception as e:
            logger.error(f"Stability AI request failed: {e}")
            raise APIError(f"Stability AI request failed: {str(e)}", provider="stability_ai")
    
    def _parse_success_response(self, response_data: Dict[str, Any], engine: str) -> Dict[str, Any]:
        """Parse successful Stability AI response."""
        result = {
            "engine": engine,
            "images": [],
        }
        
        # Extract images from artifacts
        if "artifacts" in response_data:
            for artifact in response_data["artifacts"]:
                if artifact.get("finishReason") == "SUCCESS":
                    image_data = {
                        "base64": artifact.get("base64"),
                        "seed": artifact.get("seed"),
                        "finish_reason": artifact.get("finishReason")
                    }
                    result["images"].append(image_data)
        
        return result
    
    async def generate_image(
        self,
        prompt: str,
        engine: str = "stable-diffusion-xl-1024-v1-0",
        width: int = 1024,
        height: int = 1024,
        cfg_scale: float = 7.0,
        steps: int = 30,
        samples: int = 1,
        **kwargs
    ) -> APIResponse:
        """
        Generate an image using Stability AI.
        
        Args:
            prompt: Text description of the image
            engine: Engine/model to use
            width: Image width
            height: Image height
            cfg_scale: How strictly to follow the prompt
            steps: Number of diffusion steps
            samples: Number of images to generate
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated images (base64 encoded)
        """
        request_data = {
            "engine": engine,
            "text_prompts": [{"text": prompt, "weight": 1.0}],
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "samples": samples,
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def generate_with_negative_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        engine: str = "stable-diffusion-xl-1024-v1-0",
        **kwargs
    ) -> APIResponse:
        """
        Generate image with positive and negative prompts.
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt (what to avoid)
            engine: Engine to use
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated image
        """
        request_data = {
            "engine": engine,
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
                {"text": negative_prompt, "weight": -1.0}
            ],
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def upscale_image(
        self,
        image_base64: str,
        width: int = 2048,
        height: int = 2048,
        engine: str = "esrgan-v1-x2plus"
    ) -> APIResponse:
        """
        Upscale an image using Stability AI.
        
        Args:
            image_base64: Base64 encoded image
            width: Target width
            height: Target height
            engine: Upscaling engine
        
        Returns:
            APIResponse with upscaled image
        """
        try:
            # Use image-to-image endpoint for upscaling
            files = {
                "init_image": base64.b64decode(image_base64)
            }
            
            data = {
                "image_strength": 0.35,
                "init_image_mode": "IMAGE_STRENGTH",
                "text_prompts[0][text]": "high quality, detailed",
                "text_prompts[0][weight]": 1.0,
                "cfg_scale": 7,
                "samples": 1,
                "steps": 30,
                "width": width,
                "height": height
            }
            
            response = await self.client.post(
                f"/generation/{engine}/image-to-image",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                result = self._parse_success_response(response_data, engine)
                
                return APIResponse.success_response(
                    data=result,
                    metadata={"provider": "stability_ai", "operation": "upscale"}
                )
            else:
                return APIResponse.error_response(
                    error=f"Upscaling failed: HTTP {response.status_code}",
                    status_code=response.status_code
                )
                
        except Exception as e:
            return APIResponse.error_response(
                error=f"Upscaling error: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Check Stability AI API health."""
        try:
            # Check engines endpoint
            response = await self.client.get("/engines/list", timeout=10.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Stability AI health check failed: {e}")
            return False
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engines."""
        return list(self.engines.keys())
    
    def get_engine_info(self, engine: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific engine."""
        return self.engines.get(engine)
    
    def get_supported_resolutions(self, engine: str) -> List[str]:
        """Get supported resolutions for an engine."""
        if "xl" in engine.lower():
            return ["1024x1024", "1152x896", "896x1152", "1216x832", "832x1216"]
        else:
            return ["512x512", "768x768", "512x768", "768x512"]