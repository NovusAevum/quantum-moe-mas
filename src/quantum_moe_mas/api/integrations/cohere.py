"""
Cohere API Integration.

This module provides integration with Cohere's API for
text generation, embeddings, and classification capabilities.
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


class CohereIntegration(BaseAPIIntegration):
    """
    Cohere API integration.
    
    Provides access to Cohere's models including:
    - Command models for text generation
    - Embed models for embeddings
    - Classify models for classification
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize Cohere integration.
        
        Args:
            api_key: Optional API key (overrides COHERE_API_KEY env var)
        """
        config = IntegrationConfig(
            base_url="https://api.cohere.ai/v1",
            api_key_env_var="COHERE_API_KEY",
            timeout_seconds=60,
            max_retries=3,
            retry_delay_seconds=2.0,
            rate_limit_per_minute=100,
            cost_per_token=0.00015,  # Competitive pricing
            max_tokens=4096,
            supports_streaming=True,
            requires_auth=True
        )
        
        super().__init__(config, api_key)
        
        # Available models on Cohere
        self.models = {
            "command": {
                "max_tokens": 4096,
                "cost_per_token": 0.00015,
                "description": "Cohere's flagship generative model"
            },
            "command-light": {
                "max_tokens": 4096,
                "cost_per_token": 0.00015,
                "description": "Faster, lighter version of Command"
            },
            "command-nightly": {
                "max_tokens": 4096,
                "cost_per_token": 0.00015,
                "description": "Latest experimental Command model"
            },
            "command-r": {
                "max_tokens": 128000,
                "cost_per_token": 0.00050,
                "description": "Command model optimized for RAG"
            },
            "command-r-plus": {
                "max_tokens": 128000,
                "cost_per_token": 0.00300,
                "description": "Most capable Command model"
            }
        }
        
        # Embedding models
        self.embed_models = {
            "embed-english-v3.0": {"dimensions": 1024, "cost_per_token": 0.0001},
            "embed-multilingual-v3.0": {"dimensions": 1024, "cost_per_token": 0.0001},
            "embed-english-light-v3.0": {"dimensions": 384, "cost_per_token": 0.0001},
            "embed-multilingual-light-v3.0": {"dimensions": 384, "cost_per_token": 0.0001}
        }
        
        self.default_model = "command"
        self.default_embed_model = "embed-english-v3.0"
    
    async def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get Cohere authentication header."""
        api_key = self.api_key or os.getenv(self.config.api_key_env_var)
        
        if not api_key:
            logger.error("Cohere API key not found")
            return None
        
        return {"Authorization": f"Bearer {api_key}"}
    
    def get_capabilities(self) -> List[APICapability]:
        """Get Cohere capabilities."""
        return [
            APICapability.TEXT_GENERATION,
            APICapability.CHAT_COMPLETION,
            APICapability.TEXT_EMBEDDINGS,
            APICapability.SEMANTIC_SEARCH,
            APICapability.STREAMING,
        ]
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Cohere request data."""
        # Handle different request types
        if "message" in request_data or "messages" in request_data:
            # Chat request
            if "model" not in request_data:
                request_data["model"] = self.default_model
        elif "prompt" in request_data or "text" in request_data:
            # Generation request
            if "model" not in request_data:
                request_data["model"] = self.default_model
        elif "texts" in request_data:
            # Embedding request
            if "model" not in request_data:
                request_data["model"] = self.default_embed_model
        
        return request_data
    
    async def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST"
    ) -> APIResponse:
        """Make request to Cohere API."""
        try:
            # Determine endpoint based on request type
            if "texts" in data:
                endpoint = "/embed"
            elif "message" in data or "messages" in data:
                endpoint = "/chat"
            else:
                endpoint = "/generate"
            
            response = await self.client.request(
                method=method,
                url=endpoint,
                json=data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Parse response based on endpoint
                result = self._parse_success_response(response_data, endpoint, data)
                
                return APIResponse.success_response(
                    data=result,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metadata={
                        "provider": "cohere",
                        "model": data.get("model", self.default_model),
                        "endpoint": endpoint,
                        "tokens_used": self._estimate_tokens(data),
                        "cost": self._calculate_cost(data),
                    }
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                return APIResponse.error_response(
                    error=f"Cohere API error: {error_msg}",
                    status_code=response.status_code,
                    metadata={"provider": "cohere"}
                )
                
        except httpx.HTTPStatusError as e:
            return APIResponse.error_response(
                error=f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                metadata={"provider": "cohere"}
            )
        except Exception as e:
            logger.error(f"Cohere request failed: {e}")
            raise APIError(f"Cohere request failed: {str(e)}", provider="cohere")
    
    def _parse_success_response(self, response_data: Dict[str, Any], endpoint: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse successful Cohere response."""
        result = {
            "endpoint": endpoint,
            "model": request_data.get("model"),
        }
        
        if endpoint == "/generate":
            # Text generation response
            if "generations" in response_data and response_data["generations"]:
                generation = response_data["generations"][0]
                result["content"] = generation.get("text", "")
                result["finish_reason"] = generation.get("finish_reason")
            result["id"] = response_data.get("id")
            
        elif endpoint == "/chat":
            # Chat response
            result["content"] = response_data.get("text", "")
            result["conversation_id"] = response_data.get("conversation_id")
            result["finish_reason"] = response_data.get("finish_reason")
            result["citations"] = response_data.get("citations", [])
            
        elif endpoint == "/embed":
            # Embedding response
            result["embeddings"] = response_data.get("embeddings", [])
            result["id"] = response_data.get("id")
            result["texts"] = request_data.get("texts", [])
        
        # Add metadata if available
        if "meta" in response_data:
            result["meta"] = response_data["meta"]
        
        return result
    
    def _estimate_tokens(self, request_data: Dict[str, Any]) -> int:
        """Estimate token count for request."""
        text_content = ""
        
        if "prompt" in request_data:
            text_content = request_data["prompt"]
        elif "message" in request_data:
            text_content = request_data["message"]
        elif "messages" in request_data:
            text_content = " ".join([msg.get("message", "") for msg in request_data["messages"]])
        elif "texts" in request_data:
            text_content = " ".join(request_data["texts"])
        
        # Rough estimation: ~4 characters per token
        return len(text_content) // 4
    
    def _calculate_cost(self, request_data: Dict[str, Any]) -> float:
        """Calculate request cost."""
        model = request_data.get("model", self.default_model)
        
        if model in self.models:
            cost_per_token = self.models[model]["cost_per_token"]
        elif model in self.embed_models:
            cost_per_token = self.embed_models[model]["cost_per_token"]
        else:
            cost_per_token = self.config.cost_per_token
        
        tokens = self._estimate_tokens(request_data)
        return tokens * cost_per_token
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "command",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> APIResponse:
        """
        Generate text using Cohere.
        
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
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def chat(
        self,
        message: str,
        model: str = "command",
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> APIResponse:
        """
        Chat with Cohere model.
        
        Args:
            message: User message
            model: Model to use
            conversation_id: Optional conversation ID for context
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with chat response
        """
        request_data = {
            "model": model,
            "message": message,
            **kwargs
        }
        
        if conversation_id:
            request_data["conversation_id"] = conversation_id
        
        return await self.make_request(request_data)
    
    async def embed_texts(
        self,
        texts: List[str],
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        **kwargs
    ) -> APIResponse:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            input_type: Type of input (search_document, search_query, classification, clustering)
            **kwargs: Additional parameters
        
        Returns:
            APIResponse with embeddings
        """
        request_data = {
            "model": model,
            "texts": texts,
            "input_type": input_type,
            **kwargs
        }
        
        return await self.make_request(request_data)
    
    async def semantic_search(
        self,
        query: str,
        documents: List[str],
        model: str = "embed-english-v3.0",
        top_k: int = 5
    ) -> APIResponse:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            documents: List of documents to search
            model: Embedding model to use
            top_k: Number of top results to return
        
        Returns:
            APIResponse with search results
        """
        # Get query embedding
        query_response = await self.embed_texts(
            texts=[query],
            model=model,
            input_type="search_query"
        )
        
        if not query_response.success:
            return query_response
        
        # Get document embeddings
        doc_response = await self.embed_texts(
            texts=documents,
            model=model,
            input_type="search_document"
        )
        
        if not doc_response.success:
            return doc_response
        
        # Calculate similarities (simplified - in practice use proper cosine similarity)
        query_embedding = query_response.data["embeddings"][0]
        doc_embeddings = doc_response.data["embeddings"]
        
        # Return structured results
        results = {
            "query": query,
            "results": [
                {
                    "document": doc,
                    "index": i,
                    "similarity": 0.8  # Placeholder - implement proper similarity calculation
                }
                for i, doc in enumerate(documents[:top_k])
            ]
        }
        
        return APIResponse.success_response(
            data=results,
            metadata={
                "provider": "cohere",
                "model": model,
                "query_embedding_dims": len(query_embedding),
                "documents_processed": len(documents)
            }
        )
    
    async def health_check(self) -> bool:
        """Check Cohere API health."""
        try:
            # Test with a simple generation request
            response = await self.client.post(
                "/generate",
                json={
                    "model": self.default_model,
                    "prompt": "Hello",
                    "max_tokens": 1
                },
                timeout=10.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Cohere health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available generation models."""
        return list(self.models.keys())
    
    def get_available_embed_models(self) -> List[str]:
        """Get list of available embedding models."""
        return list(self.embed_models.keys())
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        if model in self.models:
            return self.models[model]
        elif model in self.embed_models:
            return self.embed_models[model]
        return None
    
    def get_best_model_for_task(self, task: str) -> str:
        """Get the best model for a specific task."""
        task_models = {
            "generation": "command",
            "chat": "command",
            "rag": "command-r",
            "complex": "command-r-plus",
            "fast": "command-light",
            "experimental": "command-nightly"
        }
        
        return task_models.get(task.lower(), self.default_model)