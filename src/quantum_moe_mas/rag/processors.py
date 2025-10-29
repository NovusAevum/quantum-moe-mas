"""
Document processors for different content types.

This module provides specialized processors for handling various document types
including text, images, PDFs, and videos.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from quantum_moe_mas.rag.document import Document, DocumentType, DocumentMetadata


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    def process(self, document: Document) -> Document:
        """Process a document and return the processed version."""
        pass
    
    @abstractmethod
    def can_process(self, document_type: DocumentType) -> bool:
        """Check if this processor can handle the given document type."""
        pass


class TextProcessor(DocumentProcessor):
    """Processor for text documents."""
    
    def process(self, document: Document) -> Document:
        """Process text document."""
        if not self.can_process(document.document_type):
            raise ValueError(f"Cannot process document type: {document.document_type}")
        
        # For text documents, just ensure content is string
        if isinstance(document.content, bytes):
            document.content = document.content.decode('utf-8', errors='ignore')
        
        # Update metadata
        if isinstance(document.content, str):
            document.metadata.word_count = len(document.content.split())
        
        return document
    
    def can_process(self, document_type: DocumentType) -> bool:
        """Check if can process text documents."""
        return document_type in [DocumentType.TEXT, DocumentType.CODE, DocumentType.STRUCTURED]


class ImageProcessor(DocumentProcessor):
    """Processor for image documents."""
    
    def process(self, document: Document) -> Document:
        """Process image document."""
        if not self.can_process(document.document_type):
            raise ValueError(f"Cannot process document type: {document.document_type}")
        
        # For images, we would typically extract features or generate descriptions
        # For now, just add basic metadata
        if isinstance(document.content, bytes):
            document.metadata.file_size = len(document.content)
        
        return document
    
    def can_process(self, document_type: DocumentType) -> bool:
        """Check if can process image documents."""
        return document_type == DocumentType.IMAGE


class PDFProcessor(DocumentProcessor):
    """Processor for PDF documents."""
    
    def process(self, document: Document) -> Document:
        """Process PDF document."""
        if not self.can_process(document.document_type):
            raise ValueError(f"Cannot process document type: {document.document_type}")
        
        # For PDFs, we would typically extract text and images
        # For now, just add basic metadata
        if isinstance(document.content, bytes):
            document.metadata.file_size = len(document.content)
        
        return document
    
    def can_process(self, document_type: DocumentType) -> bool:
        """Check if can process PDF documents."""
        return document_type == DocumentType.PDF


class VideoProcessor(DocumentProcessor):
    """Processor for video documents."""
    
    def process(self, document: Document) -> Document:
        """Process video document."""
        if not self.can_process(document.document_type):
            raise ValueError(f"Cannot process document type: {document.document_type}")
        
        # For videos, we would typically extract frames and audio
        # For now, just add basic metadata
        if isinstance(document.content, bytes):
            document.metadata.file_size = len(document.content)
        
        return document
    
    def can_process(self, document_type: DocumentType) -> bool:
        """Check if can process video documents."""
        return document_type == DocumentType.VIDEO