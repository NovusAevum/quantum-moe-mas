"""
Database integration module for RAG system.

This module provides database connectivity and operations for the multi-modal
RAG system, with primary support for Supabase with pgvector extension.
"""

from quantum_moe_mas.rag.database.connection import SupabaseConnection
from quantum_moe_mas.rag.database.models import (
    DocumentRecord,
    ChunkRecord,
    EmbeddingRecord,
)
from quantum_moe_mas.rag.database.operations import (
    VectorOperations,
    BatchOperations,
    SearchOperations,
)
from quantum_moe_mas.rag.database.schema import DatabaseSchema

__all__ = [
    "SupabaseConnection",
    "DocumentRecord",
    "ChunkRecord", 
    "EmbeddingRecord",
    "VectorOperations",
    "BatchOperations",
    "SearchOperations",
    "DatabaseSchema",
]