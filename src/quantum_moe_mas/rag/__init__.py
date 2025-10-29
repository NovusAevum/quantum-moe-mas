"""
Adaptive Multi-Modal RAG (Retrieval-Augmented Generation) module.

This module provides advanced multi-modal retrieval capabilities with hybrid
vector-graph search, cross-modal understanding, and adaptive learning.
"""

from quantum_moe_mas.rag.adaptive_rag import AdaptiveMultiModalRAG
from quantum_moe_mas.rag.document import Document, DocumentType, DocumentMetadata
from quantum_moe_mas.rag.embeddings import MultiModalEmbeddings, EmbeddingConfig
from quantum_moe_mas.rag.retrieval import (
    RetrievalEngine,
    HybridSearchStrategy,
    RetrievalResult,
    RetrievalMetrics,
)
from quantum_moe_mas.rag.processors import (
    DocumentProcessor,
    TextProcessor,
    ImageProcessor,
    PDFProcessor,
    VideoProcessor,
)

# Supabase database integration
from quantum_moe_mas.rag.database import (
    SupabaseConnection,
    DocumentRecord,
    ChunkRecord,
    EmbeddingRecord,
    VectorOperations,
    BatchOperations,
    SearchOperations,
    DatabaseSchema,
)

__all__ = [
    "AdaptiveMultiModalRAG",
    "Document",
    "DocumentType",
    "DocumentMetadata",
    "MultiModalEmbeddings",
    "EmbeddingConfig",
    "RetrievalEngine",
    "HybridSearchStrategy",
    "RetrievalResult",
    "RetrievalMetrics",
    "DocumentProcessor",
    "TextProcessor",
    "ImageProcessor",
    "PDFProcessor",
    "VideoProcessor",
    # Database integration
    "SupabaseConnection",
    "DocumentRecord",
    "ChunkRecord",
    "EmbeddingRecord",
    "VectorOperations",
    "BatchOperations",
    "SearchOperations",
    "DatabaseSchema",
]