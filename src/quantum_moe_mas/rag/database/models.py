"""
Database models for Supabase integration.

This module provides Pydantic models that map to database tables,
with proper validation and type safety for all database operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field, validator

from quantum_moe_mas.rag.document import DocumentType, ProcessingStatus


class DocumentRecord(BaseModel):
    """Database record model for documents table."""
    
    # Primary fields
    id: UUID
    content: str
    document_type: DocumentType
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    
    # File metadata
    title: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    
    # Content metadata
    language: Optional[str] = None
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    duration: Optional[float] = None
    
    # Processing metadata
    processing_time: Optional[float] = None
    chunk_count: int = 0
    embedding_model: Optional[str] = None
    
    # Vector embedding
    embedding: Optional[np.ndarray] = None
    
    # Relationships
    parent_id: Optional[UUID] = None
    
    # Security and access
    access_level: str = "public"
    contains_pii: bool = False
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if v is not None else None,
            datetime: lambda v: v.isoformat(),
        }
    
    @validator('embedding', pre=True)
    def validate_embedding(cls, v):
        """Validate and convert embedding to numpy array."""
        if v is None:
            return None
        if isinstance(v, list):
            return np.array(v, dtype=np.float32)
        if isinstance(v, np.ndarray):
            return v.astype(np.float32)
        raise ValueError("Embedding must be a list or numpy array")
    
    @validator('access_level')
    def validate_access_level(cls, v):
        """Validate access level."""
        valid_levels = ['public', 'private', 'restricted', 'confidential']
        if v not in valid_levels:
            raise ValueError(f"Access level must be one of: {valid_levels}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = self.dict()
        
        # Convert UUID to string
        data['id'] = str(self.id)
        if self.parent_id:
            data['parent_id'] = str(self.parent_id)
        
        # Convert embedding to list
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        
        # Convert enums to strings
        data['document_type'] = self.document_type.value
        data['status'] = self.status.value
        
        # Convert timestamps to ISO format
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.processed_at:
            data['processed_at'] = self.processed_at.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentRecord:
        """Create instance from database row."""
        # Convert string IDs to UUID
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'parent_id' in data and data['parent_id']:
            data['parent_id'] = UUID(data['parent_id'])
        
        # Convert embedding list to numpy array
        if 'embedding' in data and data['embedding']:
            data['embedding'] = np.array(data['embedding'], dtype=np.float32)
        
        # Convert enum strings
        if 'document_type' in data:
            data['document_type'] = DocumentType(data['document_type'])
        if 'status' in data:
            data['status'] = ProcessingStatus(data['status'])
        
        # Convert timestamp strings
        for field in ['created_at', 'updated_at', 'processed_at']:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class ChunkRecord(BaseModel):
    """Database record model for document_chunks table."""
    
    # Primary fields
    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    
    # Chunk positioning
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    
    # Semantic metadata
    semantic_type: Optional[str] = None
    importance_score: Optional[float] = None
    
    # Vector embedding (required for chunks)
    embedding: np.ndarray
    
    # Timestamps
    created_at: datetime
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            datetime: lambda v: v.isoformat(),
        }
    
    @validator('embedding', pre=True)
    def validate_embedding(cls, v):
        """Validate and convert embedding to numpy array."""
        if v is None:
            raise ValueError("Chunk embedding is required")
        if isinstance(v, list):
            return np.array(v, dtype=np.float32)
        if isinstance(v, np.ndarray):
            return v.astype(np.float32)
        raise ValueError("Embedding must be a list or numpy array")
    
    @validator('importance_score')
    def validate_importance_score(cls, v):
        """Validate importance score range."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Importance score must be between 0 and 1")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = self.dict()
        
        # Convert UUIDs to strings
        data['id'] = str(self.id)
        data['document_id'] = str(self.document_id)
        
        # Convert embedding to list
        data['embedding'] = self.embedding.tolist()
        
        # Convert timestamp
        data['created_at'] = self.created_at.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChunkRecord:
        """Create instance from database row."""
        # Convert string IDs to UUID
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'document_id' in data and isinstance(data['document_id'], str):
            data['document_id'] = UUID(data['document_id'])
        
        # Convert embedding list to numpy array
        if 'embedding' in data and data['embedding']:
            data['embedding'] = np.array(data['embedding'], dtype=np.float32)
        
        # Convert timestamp
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


class EmbeddingRecord(BaseModel):
    """Database record model for embeddings table."""
    
    # Primary fields
    id: UUID
    content_id: UUID  # References document or chunk
    content_type: str  # 'document' or 'chunk'
    
    # Embedding metadata
    model_name: str
    model_version: Optional[str] = None
    dimensions: int
    
    # Vector data
    embedding: np.ndarray
    
    # Performance metadata
    generation_time: Optional[float] = None
    token_count: Optional[int] = None
    cost: Optional[float] = None
    
    # Timestamps
    created_at: datetime
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            datetime: lambda v: v.isoformat(),
        }
    
    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate content type."""
        if v not in ['document', 'chunk']:
            raise ValueError("Content type must be 'document' or 'chunk'")
        return v
    
    @validator('embedding', pre=True)
    def validate_embedding(cls, v):
        """Validate and convert embedding to numpy array."""
        if v is None:
            raise ValueError("Embedding is required")
        if isinstance(v, list):
            return np.array(v, dtype=np.float32)
        if isinstance(v, np.ndarray):
            return v.astype(np.float32)
        raise ValueError("Embedding must be a list or numpy array")
    
    @validator('dimensions')
    def validate_dimensions(cls, v, values):
        """Validate dimensions match embedding size."""
        if 'embedding' in values and values['embedding'] is not None:
            if len(values['embedding']) != v:
                raise ValueError("Dimensions must match embedding size")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = self.dict()
        
        # Convert UUIDs to strings
        data['id'] = str(self.id)
        data['content_id'] = str(self.content_id)
        
        # Convert embedding to list
        data['embedding'] = self.embedding.tolist()
        
        # Convert timestamp
        data['created_at'] = self.created_at.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingRecord:
        """Create instance from database row."""
        # Convert string IDs to UUID
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'content_id' in data and isinstance(data['content_id'], str):
            data['content_id'] = UUID(data['content_id'])
        
        # Convert embedding list to numpy array
        if 'embedding' in data and data['embedding']:
            data['embedding'] = np.array(data['embedding'], dtype=np.float32)
        
        # Convert timestamp
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


class MetadataRecord(BaseModel):
    """Database record model for document_metadata table."""
    
    # Primary fields
    id: UUID
    document_id: UUID
    
    # Metadata fields
    metadata_key: str
    metadata_value: Optional[str] = None
    metadata_type: str = "string"
    
    # Categorization
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @validator('metadata_type')
    def validate_metadata_type(cls, v):
        """Validate metadata type."""
        valid_types = ['string', 'number', 'boolean', 'json', 'array']
        if v not in valid_types:
            raise ValueError(f"Metadata type must be one of: {valid_types}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = self.dict()
        
        # Convert UUIDs to strings
        data['id'] = str(self.id)
        data['document_id'] = str(self.document_id)
        
        # Convert timestamps
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetadataRecord:
        """Create instance from database row."""
        # Convert string IDs to UUID
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'document_id' in data and isinstance(data['document_id'], str):
            data['document_id'] = UUID(data['document_id'])
        
        # Convert timestamps
        for field in ['created_at', 'updated_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class SearchResult(BaseModel):
    """Model for vector similarity search results."""
    
    document_id: UUID
    chunk_id: Optional[UUID] = None
    content: str
    similarity: float
    document_type: DocumentType
    
    # Additional metadata
    title: Optional[str] = None
    chunk_index: Optional[int] = None
    section_title: Optional[str] = None
    importance_score: Optional[float] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            UUID: str,
        }
    
    @validator('similarity')
    def validate_similarity(cls, v):
        """Validate similarity score range."""
        if v < 0 or v > 1:
            raise ValueError("Similarity score must be between 0 and 1")
        return v


class BatchOperationResult(BaseModel):
    """Result model for batch operations."""
    
    total_items: int
    successful_items: int
    failed_items: int
    errors: List[str] = Field(default_factory=list)
    processing_time: float
    
    # Item-specific results
    inserted_ids: List[UUID] = Field(default_factory=list)
    updated_ids: List[UUID] = Field(default_factory=list)
    failed_ids: List[UUID] = Field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100
    
    @property
    def is_successful(self) -> bool:
        """Check if operation was successful."""
        return self.failed_items == 0
    
    def add_error(self, error: str, item_id: Optional[UUID] = None) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.failed_items += 1
        if item_id:
            self.failed_ids.append(item_id)
    
    def add_success(self, item_id: UUID, operation: str = "insert") -> None:
        """Add a successful operation to the result."""
        self.successful_items += 1
        if operation == "insert":
            self.inserted_ids.append(item_id)
        elif operation == "update":
            self.updated_ids.append(item_id)