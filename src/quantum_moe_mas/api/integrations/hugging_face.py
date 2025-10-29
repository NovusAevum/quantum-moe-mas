"""
Hugging Face Inference API Integration.

This module provides integration with Hugging Face's Inference API for
accessing thousands of pre-trained models for various AI tasks.
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


class HuggingFaceIntegration(BaseAPIIntegration):
    """
    Hugging Face Inference API integration.
    
    Provides access to thousands of pre-trained models including:
    - Language models (BERT, GPT, T5, etc.)
    - Vision models (CLIP, DETR, etc.)
    - Audio models (Wav2Vec2, etc.)
    - Multimodal models
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Hugging Face integration.
        
        Args:
            api_key: Optional API key (overrides HUGGINGFACE_API_KEY env var)
        """
        config = IntegrationConfig(
            base_url="https://api-inference.huggingface.co",
            api_key_env_var="HUGGINGFACE_API_KEY",
            timeout_seconds=30,
            max_retries=3,
            retry_delay_seconds=2.0,
            rate_limit_per_minute=1000,  # Generous free tier
            cost_per_token=0.0,  # Free tier
            max_tokens=2048,
            supports_streaming=False,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Popular model configurations
        self.models = {
            # Text Generation
            "gpt2": {"task": "text-generation", "max_tokens": 1024},
            "microsoft/DialoGPT-medium": {"task": "conversational", "max_tokens": 1024},
            "facebook/blenderbot-400M-distill": {"task": "conversational", "max_tokens": 512},
            
            # Text Classification
            "cardiffnlp/twitter-roberta-base-sentiment-latest": {"task": "sentiment-analysis"},
            "facebook/bart-large-mnli": {"task": "zero-shot-classification"},
            
            # Question Answering
            "deepset/roberta-base-squad2": {"task": "question-answering"},
            "microsoft/DialoGPT-large": {"task": "question-answering"},
            
            # Text Summarization
            "facebook/bart-large-cnn": {"task": "summarization", "max_tokens": 512},
            "t5-base": {"task": "text2text-generation", "max_tokens": 512},
            
            # Code Generation
            "microsoft/CodeBERT-base": {"task": "feature-extraction"},
            "codeparrot/codeparrot-small": {"task": "text-generation", "max_tokens": 1024},
            
            # Vision
            "google/vit-base-patch16-224": {"task": "image-classification"},
            "facebook/detr-resnet-50": {"task": "object-detection"},
            
            # Audio
            "facebook/wav2vec2-base-960h": {"task": "automatic-speech-recognition"},
            "espnet/hindi_male_fgl": {"task": "text-to-speech"},
        }
        
        self.default_model = "gpt2"
    
    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get Hugging Face authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.warning("Hugging Face API key not found, using public inference")
            return None
        
        return {"Authorization": f"Bearer {api_key}"}
    
    def get_capabilities(self) -> List[APICapability]:
        """Get Hugging Face capabilities."""
        return [
            APICapability.TEXT_GENERATION,
            APICapability.CHAT_COMPLETION,
            APICapability.CODE_GENERATION,
            APICapability.IMAGE_ANALYSIS,
            APICapability.TEXT_EMBEDDINGS,
            APICapability.SEMANTIC_SEARCH,
            APICapability.SPEECH_TO_TEXT,
            APICapability.TEXT_TO_SPEECH,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Hugging Face request data."""
        # Ensure model is specified
        if "model" not in request_data:
            request_data["model"] = self.default_model
        
        # Ensure inputs are provided
        if "inputs" not in request_data:
            if "prompt" in request_data:
                request_data["inputs"] = request_data.pop("prompt")
            elif "text" in request_data:
                request_data["inputs"] = request_data.pop("text")
            else:
                raise APIError("'inputs' field is required for Hugging Face API")
        
        # Set parameters if not provided
        if "parameters" not in request_data:
            request_data["parameters"] = {}
        
        # Set reasonable defaults based on model
        model = request_data["model"]
        model_config = self.models.get(model, {})
        
        if "max_new_tokens" not in request_data["parameters"]:
            max_tokens = model_config.get("max_tokens", 100)
            request_data["parameters"]["max_new_tokens"] = max_tokens
        
        return request_data
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to Hugging Face API."""
        try:
            # Construct endpoint for specific model
            model = data.pop("model", self.default_model)
            endpoint = f"/models/{model}"
            
            response = await self.client.request(
                method=method,
                url=endpoint,
                json=data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Parse response based on task type
                result = self._parse_success_response(response_data, model)
                
                return APIResponse.success_response(
                    data=result,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metadata={
                        "provider": "hugging_face",
                        "model": model,
                        "task": self.models.get(model, {}).get("task", "unknown"),
                        "tokens_used": self._estimate_tokens(data.get("inputs", "")),
                        "cost": 0.0,  # Free tier
                    }
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                return APIResponse.error_response(
                    error=f"Hugging Face API error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "hugging_face", "model": model}
                )
                
        except httpx.HTTPStatusError as e:
            return APIResponse.error_response(
                error=f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                metadata={"provider": "hugging_face"}
            )
        except Exception as e:
            logger.error(f"Hugging Face request failed: {e}")
            raise APIError(f"Hugging Face request failed: {str(e)}", provider="hugging_face")
    
    def _parse_success_response(self, response_data: Any, model: str) -> Dict[str, Any]:
        """Parse successful Hugging Face response."""
        model_config = self.models.get(model, {})
        task = model_config.get("task", "unknown")
        
        result = {
            "model": model,
            "task": task,
            "raw_response": response_data,
        }
        
        # Parse based on task type
        if isinstance(response_data, list) and response_data:
            first_result = response_data[0]
            
            if task == "text-generation":
                result["content"] = first_result.get("generated_text", "")
            elif task == "conversational":
                result["content"] = first_result.get("generated_text", "")
            elif task == "sentiment-analysis":
                result["sentiment"] = first_result.get("label", "")
                result["confidence"] = first_result.get("score", 0.0)
            elif task == "question-answering":
                result["answer"] = first_result.get("answer", "")
                result["confidence"] = first_result.get("score", 0.0)
            elif task == "summarization":
                result["summary"] = first_result.get("summary_text", "")
            elif task == "image-classification":
                result["label"] = first_result.get("label", "")
                result["confidence"] = first_result.get("score", 0.0)
            else:
                result["content"] = str(first_result)
        
        elif isinstance(response_data, dict):
            if task == "question-answering":
                result["answer"] = response_data.get("answer", "")
                result["confidence"] = response_data.get("score", 0.0)
            else:
                result["content"] = str(response_data)
        
        else:
            result["content"] = str(response_data)
        
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    async def text_generation(
        self,
        prompt: str,
        model: str = "gpt2",
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> APIResponse:
        """
        Generate text using a language model.
        
        Args:
            prompt: Input text prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with generated text
        """
        request_data = {
            "model": model,
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                **kwargs
            }
        }
        
        return await self.make_request(request_data)
    
    async def sentiment_analysis(
        self,
        text: str,
        model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ) -> APIResponse:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            model: Sentiment analysis model
        
        Returns:
            APIResponse with sentiment analysis
        """
        request_data = {
            "model": model,
            "inputs": text
        }
        
        return await self.make_request(request_data)
    
    async def question_answering(
        self,
        question: str,
        context: str,
        model: str = "deepset/roberta-base-squad2"
    ) -> APIResponse:
        """
        Answer a question based on context.
        
        Args:
            question: Question to answer
            context: Context text
            model: QA model to use
        
        Returns:
            APIResponse with answer
        """
        request_data = {
            "model": model,
            "inputs": {
                "question": question,
                "context": context
            }
        }
        
        return await self.make_request(request_data)
    
    async def summarization(
        self,
        text: str,
        model: str = "facebook/bart-large-cnn",
        max_length: int = 150,
        min_length: int = 30
    ) -> APIResponse:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            model: Summarization model
            max_length: Maximum summary length
            min_length: Minimum summary length
        
        Returns:
            APIResponse with summary
        """
        request_data = {
            "model": model,
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False
            }
        }
        
        return await self.make_request(request_data)
    
    async def image_classification(
        self,
        image_data: bytes,
        model: str = "google/vit-base-patch16-224"
    ) -> APIResponse:
        """
        Classify an image.
        
        Args:
            image_data: Image bytes
            model: Image classification model
        
        Returns:
            APIResponse with classification
        """
        # For image data, we need to send as binary
        try:
            response = await self.client.post(
                f"/models/{model}",
                content=image_data,
                headers={"Content-Type": "application/octet-stream"}
            )
            
            if response.status_code == 200:
                response_data = response.json()
                result = self._parse_success_response(response_data, model)
                
                return APIResponse.success_response(
                    data=result,
                    status_code=response.status_code,
                    metadata={
                        "provider": "hugging_face",
                        "model": model,
                        "task": "image-classification"
                    }
                )
            else:
                error_msg = f"HTTP {response.status_code}"
                return APIResponse.error_response(
                    error=f"Image classification failed: {error_msg}",
                    status_code=response.status_code
                )
                
        except Exception as e:
            return APIResponse.error_response(
                error=f"Image classification error: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """Check Hugging Face API health."""
        try:
            # Test with a simple model
            response = await self.client.post(
                "/models/gpt2",
                json={"inputs": "Hello", "parameters": {"max_new_tokens": 1}},
                timeout=10.0
            )
            return response.status_code in [200, 503]  # 503 is "model loading"
        except Exception as e:
            logger.warning(f"Hugging Face health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.models.get(model)
    
    def get_models_by_task(self, task: str) -> List[str]:
        """Get models that support a specific task."""
        return [
            model for model, config in self.models.items()
            if config.get("task") == task
        ]