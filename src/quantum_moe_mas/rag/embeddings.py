"""
Multi-modal embeddings system for the RAG engine.

This module provides unified embedding generation for text, images, and other
content types using various embedding models and techniques.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.rag.document import Document, DocumentType

logger = get_logger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    
    # Model configuration
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model: str = "sentence-transformers/clip-ViT-B-32"
    code_model: str = "microsoft/codebert-base"
    
    # Embedding parameters
    embedding_dimension: int = 384
    max_sequence_length: int = 512
    batch_size: int = 32
    
    # Performance settings
    use_gpu: bool = False
    cache_embeddings: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Normalization
    normalize_embeddings: bool = True
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""
    
    embedding: np.ndarray
    model_name: str
    dimension: int
    processing_time: float
    cache_hit: bool = False
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
        }


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._cache: Dict[str, EmbeddingResult] = {}
        self._model = None
    
    @abstractmethod
    async def encode(
        self,
        inputs: Union[str, List[str]],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode inputs into embeddings."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model."""
        pass
    
    def _get_cache_key(self, input_text: str, **kwargs) -> str:
        """Generate cache key for input."""
        key_data = f"{input_text}_{kwargs}_{self.__class__.__name__}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[EmbeddingResult]:
        """Get cached embedding if available and not expired."""
        if not self.config.cache_embeddings:
            return None
        
        if cache_key in self._cache:
            result = self._cache[cache_key]
            # Check if cache is still valid
            if time.time() - result.processing_time < self.config.cache_ttl:
                result.cache_hit = True
                return result
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        
        return None
    
    def _cache_embedding(self, cache_key: str, result: EmbeddingResult) -> None:
        """Cache embedding result."""
        if self.config.cache_embeddings:
            self._cache[cache_key] = result


class TextEmbeddingModel(BaseEmbeddingModel):
    """Text embedding model using SentenceTransformers."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model_name = config.text_model
        self.load_model()
    
    def load_model(self) -> None:
        """Load the text embedding model."""
        try:
            logger.info(f"Loading text embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device='cuda' if self.config.use_gpu else 'cpu'
            )
            logger.info(f"Text embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text embedding model: {e}")
            raise
    
    async def encode(
        self,
        inputs: Union[str, List[str]],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode text inputs into embeddings."""
        start_time = time.time()
        
        # Handle single string input
        if isinstance(inputs, str):
            cache_key = self._get_cache_key(inputs, **kwargs)
            cached_result = self._get_cached_embedding(cache_key)
            
            if cached_result:
                logger.debug(f"Cache hit for text embedding")
                return cached_result.embedding
            
            # Generate embedding
            embedding = await self._encode_single(inputs, **kwargs)
            processing_time = time.time() - start_time
            
            # Cache result
            result = EmbeddingResult(
                embedding=embedding,
                model_name=self.model_name,
                dimension=len(embedding),
                processing_time=processing_time
            )
            self._cache_embedding(cache_key, result)
            
            return embedding
        
        # Handle batch input
        else:
            embeddings = await self._encode_batch(inputs, **kwargs)
            return embeddings
    
    async def _encode_single(self, text: str, **kwargs) -> np.ndarray:
        """Encode single text input."""
        # Truncate text if too long
        if len(text) > self.config.max_sequence_length * 4:  # Rough character estimate
            text = text[:self.config.max_sequence_length * 4]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                **kwargs
            )
        )
        
        return embedding.astype(np.float32)
    
    async def _encode_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """Encode batch of text inputs."""
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Truncate texts in batch
            truncated_batch = []
            for text in batch:
                if len(text) > self.config.max_sequence_length * 4:
                    text = text[:self.config.max_sequence_length * 4]
                truncated_batch.append(text)
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    truncated_batch,
                    normalize_embeddings=self.config.normalize_embeddings,
                    batch_size=len(truncated_batch),
                    **kwargs
                )
            )
            
            all_embeddings.extend([emb.astype(np.float32) for emb in batch_embeddings])
        
        return all_embeddings


class ImageEmbeddingModel(BaseEmbeddingModel):
    """Image embedding model using CLIP."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model_name = config.image_model
        self.load_model()
    
    def load_model(self) -> None:
        """Load the image embedding model."""
        try:
            logger.info(f"Loading image embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device='cuda' if self.config.use_gpu else 'cpu'
            )
            logger.info(f"Image embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image embedding model: {e}")
            # Fallback to text model for image descriptions
            logger.warning("Falling back to text model for image processing")
            self._model = SentenceTransformer(
                self.config.text_model,
                device='cuda' if self.config.use_gpu else 'cpu'
            )
    
    async def encode(
        self,
        inputs: Union[str, List[str]],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode image inputs (as descriptions) into embeddings."""
        # For now, treat image descriptions as text
        # In a full implementation, this would process actual image data
        if isinstance(inputs, str):
            # Assume input is image description
            return await self._encode_image_description(inputs, **kwargs)
        else:
            # Batch of image descriptions
            embeddings = []
            for img_desc in inputs:
                emb = await self._encode_image_description(img_desc, **kwargs)
                embeddings.append(emb)
            return embeddings
    
    async def _encode_image_description(self, description: str, **kwargs) -> np.ndarray:
        """Encode image description into embedding."""
        # Run in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                description,
                normalize_embeddings=self.config.normalize_embeddings,
                **kwargs
            )
        )
        
        return embedding.astype(np.float32)


class CodeEmbeddingModel(BaseEmbeddingModel):
    """Code embedding model for source code."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model_name = config.code_model
        # For now, use text model as fallback
        self._text_model = TextEmbeddingModel(config)
    
    def load_model(self) -> None:
        """Load the code embedding model."""
        # For now, delegate to text model
        logger.info(f"Using text model for code embeddings: {self.config.text_model}")
    
    async def encode(
        self,
        inputs: Union[str, List[str]],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode code inputs into embeddings."""
        # Delegate to text model for now
        return await self._text_model.encode(inputs, **kwargs)


class MultiModalEmbeddings:
    """Unified multi-modal embedding system."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        
        # Initialize embedding models
        self.text_model = TextEmbeddingModel(self.config)
        self.image_model = ImageEmbeddingModel(self.config)
        self.code_model = CodeEmbeddingModel(self.config)
        
        logger.info("Multi-modal embedding system initialized")
    
    async def embed_document(self, document: Document) -> np.ndarray:
        """Generate embedding for a document based on its type."""
        try:
            if document.document_type == DocumentType.TEXT:
                return await self.embed_text(document.content)
            elif document.document_type == DocumentType.IMAGE:
                # For images, we'd typically extract features or use descriptions
                # For now, use metadata or generate a description
                description = self._generate_image_description(document)
                return await self.embed_text(description)
            elif document.document_type == DocumentType.PDF:
                # Extract text content from PDF
                text_content = self._extract_pdf_text(document)
                return await self.embed_text(text_content)
            elif document.document_type == DocumentType.CODE:
                return await self.embed_code(document.content)
            else:
                # Default to text embedding
                content = document.content if isinstance(document.content, str) else str(document.content)
                return await self.embed_text(content)
        
        except Exception as e:
            logger.error(f"Failed to embed document {document.id}: {e}")
            raise
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding."""
        if not text or not text.strip():
            # Return zero embedding for empty text
            return np.zeros(self.config.embedding_dimension, dtype=np.float32)
        
        return await self.text_model.encode(text)
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return await self.text_model.encode(texts)
    
    async def embed_image(self, image_description: str) -> np.ndarray:
        """Generate image embedding from description."""
        return await self.image_model.encode(image_description)
    
    async def embed_code(self, code: str) -> np.ndarray:
        """Generate code embedding."""
        return await self.code_model.encode(code)
    
    def _generate_image_description(self, document: Document) -> str:
        """Generate description for image document."""
        # Use metadata to create description
        metadata = document.metadata
        description_parts = []
        
        if metadata.title:
            description_parts.append(f"Image titled: {metadata.title}")
        
        if metadata.tags:
            description_parts.append(f"Tags: {', '.join(metadata.tags)}")
        
        if metadata.custom_fields:
            for key, value in metadata.custom_fields.items():
                if isinstance(value, str):
                    description_parts.append(f"{key}: {value}")
        
        if not description_parts:
            description_parts.append("Image content")
        
        return ". ".join(description_parts)
    
    def _extract_pdf_text(self, document: Document) -> str:
        """Extract text content from PDF document."""
        # This is a placeholder - in a real implementation,
        # you'd use libraries like PyPDF2, pdfplumber, etc.
        if isinstance(document.content, str):
            return document.content
        else:
            # For binary PDF content, you'd extract text here
            return f"PDF document: {document.metadata.title or 'Untitled'}"
    
    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    async def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """Find most similar embeddings to query."""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embedding cache and usage."""
        return {
            "text_model": self.text_model.model_name,
            "image_model": self.image_model.model_name,
            "code_model": self.code_model.model_name,
            "embedding_dimension": self.config.embedding_dimension,
            "cache_enabled": self.config.cache_embeddings,
            "text_cache_size": len(self.text_model._cache),
            "image_cache_size": len(self.image_model._cache),
            "code_cache_size": len(self.code_model._cache),
        }