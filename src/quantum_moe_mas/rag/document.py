"""
Document models and types for multi-modal RAG system.

This module defines the core document structures used throughout the RAG system,
supporting text, image, PDF, video, and audio content types.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Supported document types for multi-modal processing."""
    
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"
    CODE = "code"
    STRUCTURED = "structured"  # JSON, CSV, etc.


class ProcessingStatus(str, Enum):
    """Document processing status."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""
    
    # Core metadata
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    
    # File metadata
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    
    # Content metadata
    language: Optional[str] = None
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    duration: Optional[float] = None  # For video/audio
    
    # Processing metadata
    processing_time: Optional[float] = None
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None
    
    # Custom metadata
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # Security metadata
    access_level: str = "public"
    contains_pii: bool = False
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class DocumentChunk(BaseModel):
    """A chunk of a document with its embedding and metadata."""
    
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_index: int
    content: str
    embedding: Optional[np.ndarray] = None
    
    # Chunk-specific metadata
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    
    # Semantic metadata
    semantic_type: Optional[str] = None  # paragraph, heading, table, etc.
    importance_score: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
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


class Document(BaseModel):
    """Core document model for multi-modal RAG system."""
    
    id: UUID = Field(default_factory=uuid4)
    content: Union[str, bytes]
    document_type: DocumentType
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    
    # Processing information
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    
    # Embeddings and chunks
    embedding: Optional[np.ndarray] = None
    chunks: List[DocumentChunk] = Field(default_factory=list)
    
    # Relationships
    parent_id: Optional[UUID] = None  # For document hierarchies
    related_documents: List[UUID] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if v is not None else None,
            datetime: lambda v: v.isoformat(),
            bytes: lambda v: v.decode('utf-8', errors='ignore'),
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
    
    @property
    def content_hash(self) -> str:
        """Generate a hash of the document content."""
        if isinstance(self.content, str):
            content_bytes = self.content.encode('utf-8')
        else:
            content_bytes = self.content
        
        return hashlib.sha256(content_bytes).hexdigest()
    
    @property
    def is_processed(self) -> bool:
        """Check if document has been successfully processed."""
        return self.status == ProcessingStatus.COMPLETED
    
    @property
    def has_embedding(self) -> bool:
        """Check if document has an embedding."""
        return self.embedding is not None
    
    @property
    def has_chunks(self) -> bool:
        """Check if document has been chunked."""
        return len(self.chunks) > 0
    
    def add_chunk(
        self,
        content: str,
        chunk_index: int,
        embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> DocumentChunk:
        """Add a chunk to the document."""
        chunk = DocumentChunk(
            document_id=self.id,
            chunk_index=chunk_index,
            content=content,
            embedding=embedding,
            **kwargs
        )
        self.chunks.append(chunk)
        return chunk
    
    def get_chunk_by_index(self, index: int) -> Optional[DocumentChunk]:
        """Get a chunk by its index."""
        for chunk in self.chunks:
            if chunk.chunk_index == index:
                return chunk
        return None
    
    def update_status(
        self,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update document processing status."""
        self.status = status
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
        
        if status == ProcessingStatus.COMPLETED:
            self.processed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for storage."""
        return {
            "id": str(self.id),
            "content": self.content if isinstance(self.content, str) else self.content.decode('utf-8', errors='ignore'),
            "document_type": self.document_type.value,
            "metadata": self.metadata.dict(),
            "status": self.status.value,
            "error_message": self.error_message,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "chunks": [chunk.dict() for chunk in self.chunks],
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "related_documents": [str(doc_id) for doc_id in self.related_documents],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Document:
        """Create document from dictionary."""
        # Convert string IDs back to UUIDs
        if "id" in data and isinstance(data["id"], str):
            data["id"] = UUID(data["id"])
        
        if "parent_id" in data and data["parent_id"]:
            data["parent_id"] = UUID(data["parent_id"])
        
        if "related_documents" in data:
            data["related_documents"] = [UUID(doc_id) for doc_id in data["related_documents"]]
        
        # Convert embedding back to numpy array
        if "embedding" in data and data["embedding"]:
            data["embedding"] = np.array(data["embedding"], dtype=np.float32)
        
        # Convert chunks
        if "chunks" in data:
            chunks = []
            for chunk_data in data["chunks"]:
                if "id" in chunk_data and isinstance(chunk_data["id"], str):
                    chunk_data["id"] = UUID(chunk_data["id"])
                if "document_id" in chunk_data and isinstance(chunk_data["document_id"], str):
                    chunk_data["document_id"] = UUID(chunk_data["document_id"])
                if "embedding" in chunk_data and chunk_data["embedding"]:
                    chunk_data["embedding"] = np.array(chunk_data["embedding"], dtype=np.float32)
                chunks.append(DocumentChunk(**chunk_data))
            data["chunks"] = chunks
        
        # Convert datetime strings
        for field in ["created_at", "updated_at", "processed_at"]:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        document_type: Optional[DocumentType] = None
    ) -> Document:
        """Create document from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect document type if not provided
        if document_type is None:
            document_type = cls._detect_document_type(file_path)
        
        # Read file content
        if document_type in [DocumentType.IMAGE, DocumentType.VIDEO, DocumentType.AUDIO]:
            with open(file_path, 'rb') as f:
                content = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        # Create metadata
        stat = file_path.stat()
        metadata = DocumentMetadata(
            title=file_path.stem,
            file_path=str(file_path),
            file_size=stat.st_size,
            mime_type=cls._get_mime_type(file_path),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )
        
        return cls(
            content=content,
            document_type=document_type,
            metadata=metadata
        )
    
    @staticmethod
    def _detect_document_type(file_path: Path) -> DocumentType:
        """Auto-detect document type from file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.txt', '.md', '.rst']:
            return DocumentType.TEXT
        elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return DocumentType.IMAGE
        elif suffix == '.pdf':
            return DocumentType.PDF
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return DocumentType.VIDEO
        elif suffix in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
            return DocumentType.AUDIO
        elif suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']:
            return DocumentType.CODE
        elif suffix in ['.json', '.csv', '.xml', '.yaml', '.yml']:
            return DocumentType.STRUCTURED
        else:
            return DocumentType.TEXT  # Default fallback
    
    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Get MIME type from file extension."""
        suffix = file_path.suffix.lower()
        
        mime_types = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.mp4': 'video/mp4',
            '.mp3': 'audio/mpeg',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.xml': 'application/xml',
        }
        
        return mime_types.get(suffix, 'application/octet-stream')