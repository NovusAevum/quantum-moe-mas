"""
Database operations for Supabase vector database integration.

This module provides high-level database operations including vector storage,
similarity search, and batch processing with optimal performance.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from quantum_moe_mas.core.exceptions import DatabaseError, ValidationError
from quantum_moe_mas.rag.database.connection import SupabaseConnection
from quantum_moe_mas.rag.database.models import (
    BatchOperationResult,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRecord,
    MetadataRecord,
    SearchResult,
)
from quantum_moe_mas.rag.document import Document, DocumentChunk, DocumentType

logger = logging.getLogger(__name__)


class VectorOperations:
    """
    High-performance vector operations for similarity search and storage.
    
    Features:
    - Optimized vector similarity search with multiple distance metrics
    - Batch vector operations for improved performance
    - Automatic index optimization and maintenance
    - Comprehensive error handling and retry logic
    """
    
    def __init__(self, connection: SupabaseConnection):
        """Initialize vector operations."""
        self.connection = connection
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def store_document_embedding(
        self,
        document_record: DocumentRecord
    ) -> UUID:
        """
        Store document with vector embedding.
        
        Args:
            document_record: Document record with embedding
            
        Returns:
            UUID of stored document
            
        Raises:
            DatabaseError: If storage fails
            ValidationError: If document data is invalid
        """
        try:
            if document_record.embedding is None:
                raise ValidationError("Document embedding is required")
            
            # Convert to dictionary for insertion
            doc_data = document_record.to_dict()
            
            # Insert document
            async with self.connection.get_transaction() as conn:
                insert_query = """
                INSERT INTO documents (
                    id, content, document_type, status, error_message,
                    title, file_path, file_size, mime_type, encoding,
                    language, word_count, page_count, duration,
                    processing_time, chunk_count, embedding_model,
                    embedding, parent_id, access_level, contains_pii,
                    created_at, updated_at, processed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                    $22, $23, $24
                ) RETURNING id
                """
                
                result = await conn.fetchrow(
                    insert_query,
                    document_record.id,
                    document_record.content,
                    document_record.document_type.value,
                    document_record.status.value,
                    document_record.error_message,
                    document_record.title,
                    document_record.file_path,
                    document_record.file_size,
                    document_record.mime_type,
                    document_record.encoding,
                    document_record.language,
                    document_record.word_count,
                    document_record.page_count,
                    document_record.duration,
                    document_record.processing_time,
                    document_record.chunk_count,
                    document_record.embedding_model,
                    document_record.embedding.tolist(),
                    document_record.parent_id,
                    document_record.access_level,
                    document_record.contains_pii,
                    document_record.created_at,
                    document_record.updated_at,
                    document_record.processed_at,
                )
                
                logger.info(f"Stored document with embedding: {result['id']}")
                return UUID(result['id'])
                
        except Exception as e:
            logger.error(f"Failed to store document embedding: {e}")
            raise DatabaseError(f"Document storage failed: {e}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def store_chunk_embeddings(
        self,
        chunk_records: List[ChunkRecord]
    ) -> List[UUID]:
        """
        Store multiple document chunks with embeddings.
        
        Args:
            chunk_records: List of chunk records with embeddings
            
        Returns:
            List of stored chunk UUIDs
        """
        try:
            stored_ids = []
            
            async with self.connection.get_transaction() as conn:
                insert_query = """
                INSERT INTO document_chunks (
                    id, document_id, chunk_index, content,
                    start_position, end_position, page_number, section_title,
                    semantic_type, importance_score, embedding, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                ) RETURNING id
                """
                
                for chunk in chunk_records:
                    result = await conn.fetchrow(
                        insert_query,
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.start_position,
                        chunk.end_position,
                        chunk.page_number,
                        chunk.section_title,
                        chunk.semantic_type,
                        chunk.importance_score,
                        chunk.embedding.tolist(),
                        chunk.created_at,
                    )
                    
                    stored_ids.append(UUID(result['id']))
                
                logger.info(f"Stored {len(stored_ids)} chunk embeddings")
                return stored_ids
                
        except Exception as e:
            logger.error(f"Failed to store chunk embeddings: {e}")
            raise DatabaseError(f"Chunk storage failed: {e}") from e
    
    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        document_types: Optional[List[DocumentType]] = None,
        search_chunks: bool = True,
        search_documents: bool = False,
    ) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query vector embedding
            similarity_threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results
            document_types: Filter by document types
            search_chunks: Search in document chunks
            search_documents: Search in full documents
            
        Returns:
            List of search results ordered by similarity
        """
        try:
            results = []
            
            # Convert document types to strings
            type_filter = None
            if document_types:
                type_filter = [dt.value for dt in document_types]
            
            async with self.connection.get_connection() as conn:
                if search_chunks:
                    # Search in document chunks (primary search)
                    chunk_query = """
                    SELECT 
                        d.id as document_id,
                        c.id as chunk_id,
                        c.content,
                        c.chunk_index,
                        c.section_title,
                        c.importance_score,
                        d.document_type,
                        d.title,
                        1 - (c.embedding <=> $1) as similarity
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE 
                        ($2::text[] IS NULL OR d.document_type = ANY($2))
                        AND (1 - (c.embedding <=> $1)) >= $3
                        AND d.status = 'completed'
                    ORDER BY c.embedding <=> $1
                    LIMIT $4
                    """
                    
                    chunk_results = await conn.fetch(
                        chunk_query,
                        query_embedding.tolist(),
                        type_filter,
                        similarity_threshold,
                        max_results
                    )
                    
                    for row in chunk_results:
                        results.append(SearchResult(
                            document_id=UUID(row['document_id']),
                            chunk_id=UUID(row['chunk_id']),
                            content=row['content'],
                            similarity=float(row['similarity']),
                            document_type=DocumentType(row['document_type']),
                            title=row['title'],
                            chunk_index=row['chunk_index'],
                            section_title=row['section_title'],
                            importance_score=row['importance_score'],
                        ))
                
                if search_documents and len(results) < max_results:
                    # Search in full documents (secondary search)
                    remaining_results = max_results - len(results)
                    
                    doc_query = """
                    SELECT 
                        d.id as document_id,
                        d.content,
                        d.document_type,
                        d.title,
                        1 - (d.embedding <=> $1) as similarity
                    FROM documents d
                    WHERE 
                        d.embedding IS NOT NULL
                        AND ($2::text[] IS NULL OR d.document_type = ANY($2))
                        AND (1 - (d.embedding <=> $1)) >= $3
                        AND d.status = 'completed'
                    ORDER BY d.embedding <=> $1
                    LIMIT $4
                    """
                    
                    doc_results = await conn.fetch(
                        doc_query,
                        query_embedding.tolist(),
                        type_filter,
                        similarity_threshold,
                        remaining_results
                    )
                    
                    for row in doc_results:
                        results.append(SearchResult(
                            document_id=UUID(row['document_id']),
                            chunk_id=None,
                            content=row['content'][:1000] + "..." if len(row['content']) > 1000 else row['content'],
                            similarity=float(row['similarity']),
                            document_type=DocumentType(row['document_type']),
                            title=row['title'],
                        ))
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x.similarity, reverse=True)
            
            logger.info(f"Vector similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            raise DatabaseError(f"Similarity search failed: {e}") from e
    
    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        similarity_threshold: float = 0.7,
        text_similarity_weight: float = 0.3,
        vector_similarity_weight: float = 0.7,
        max_results: int = 10,
        document_types: Optional[List[DocumentType]] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid vector + text similarity search.
        
        Args:
            query_embedding: Query vector embedding
            query_text: Query text for full-text search
            similarity_threshold: Minimum similarity score
            text_similarity_weight: Weight for text similarity (0-1)
            vector_similarity_weight: Weight for vector similarity (0-1)
            max_results: Maximum number of results
            document_types: Filter by document types
            
        Returns:
            List of search results with combined similarity scores
        """
        try:
            # Normalize weights
            total_weight = text_similarity_weight + vector_similarity_weight
            text_weight = text_similarity_weight / total_weight
            vector_weight = vector_similarity_weight / total_weight
            
            type_filter = None
            if document_types:
                type_filter = [dt.value for dt in document_types]
            
            async with self.connection.get_connection() as conn:
                hybrid_query = """
                SELECT 
                    d.id as document_id,
                    c.id as chunk_id,
                    c.content,
                    c.chunk_index,
                    c.section_title,
                    c.importance_score,
                    d.document_type,
                    d.title,
                    -- Vector similarity
                    (1 - (c.embedding <=> $1)) as vector_similarity,
                    -- Text similarity using trigram matching
                    similarity(c.content, $2) as text_similarity,
                    -- Combined similarity score
                    ($3 * (1 - (c.embedding <=> $1)) + $4 * similarity(c.content, $2)) as combined_similarity
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE 
                    ($5::text[] IS NULL OR d.document_type = ANY($5))
                    AND d.status = 'completed'
                    AND (
                        (1 - (c.embedding <=> $1)) >= $6 
                        OR similarity(c.content, $2) > 0.1
                    )
                ORDER BY combined_similarity DESC
                LIMIT $7
                """
                
                results = await conn.fetch(
                    hybrid_query,
                    query_embedding.tolist(),
                    query_text,
                    vector_weight,
                    text_weight,
                    type_filter,
                    similarity_threshold,
                    max_results
                )
                
                search_results = []
                for row in results:
                    if row['combined_similarity'] >= similarity_threshold:
                        search_results.append(SearchResult(
                            document_id=UUID(row['document_id']),
                            chunk_id=UUID(row['chunk_id']),
                            content=row['content'],
                            similarity=float(row['combined_similarity']),
                            document_type=DocumentType(row['document_type']),
                            title=row['title'],
                            chunk_index=row['chunk_index'],
                            section_title=row['section_title'],
                            importance_score=row['importance_score'],
                        ))
                
                logger.info(f"Hybrid search returned {len(search_results)} results")
                return search_results
                
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise DatabaseError(f"Hybrid search failed: {e}") from e


class BatchOperations:
    """
    High-performance batch operations for large-scale document processing.
    
    Features:
    - Concurrent batch processing with configurable parallelism
    - Transaction management for data consistency
    - Progress tracking and error handling
    - Memory-efficient processing for large datasets
    """
    
    def __init__(self, connection: SupabaseConnection, batch_size: int = 100):
        """Initialize batch operations."""
        self.connection = connection
        self.batch_size = batch_size
        self.max_concurrent_batches = 5
    
    async def batch_store_documents(
        self,
        documents: List[Document],
        include_embeddings: bool = True
    ) -> BatchOperationResult:
        """
        Store multiple documents in batches.
        
        Args:
            documents: List of documents to store
            include_embeddings: Whether to include embeddings
            
        Returns:
            Batch operation result with statistics
        """
        start_time = time.time()
        result = BatchOperationResult(
            total_items=len(documents),
            successful_items=0,
            failed_items=0,
            processing_time=0.0
        )
        
        try:
            # Process documents in batches
            batches = [
                documents[i:i + self.batch_size]
                for i in range(0, len(documents), self.batch_size)
            ]
            
            # Process batches concurrently
            semaphore = asyncio.Semaphore(self.max_concurrent_batches)
            
            async def process_batch(batch: List[Document]) -> None:
                async with semaphore:
                    await self._process_document_batch(batch, result, include_embeddings)
            
            # Execute all batches
            await asyncio.gather(*[process_batch(batch) for batch in batches])
            
            result.processing_time = time.time() - start_time
            
            logger.info(
                f"Batch document storage completed: "
                f"{result.successful_items}/{result.total_items} successful "
                f"in {result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            result.processing_time = time.time() - start_time
            result.add_error(f"Batch operation failed: {e}")
            logger.error(f"Batch document storage failed: {e}")
            return result
    
    async def _process_document_batch(
        self,
        batch: List[Document],
        result: BatchOperationResult,
        include_embeddings: bool
    ) -> None:
        """Process a single batch of documents."""
        try:
            async with self.connection.get_transaction() as conn:
                for document in batch:
                    try:
                        # Convert to database record
                        doc_record = DocumentRecord(
                            id=document.id,
                            content=document.content if isinstance(document.content, str) else document.content.decode('utf-8', errors='ignore'),
                            document_type=document.document_type,
                            status=document.status,
                            error_message=document.error_message,
                            title=document.metadata.title,
                            file_path=document.metadata.file_path,
                            file_size=document.metadata.file_size,
                            mime_type=document.metadata.mime_type,
                            encoding=document.metadata.encoding,
                            language=document.metadata.language,
                            word_count=document.metadata.word_count,
                            page_count=document.metadata.page_count,
                            duration=document.metadata.duration,
                            processing_time=document.metadata.processing_time,
                            chunk_count=len(document.chunks),
                            embedding_model=document.metadata.embedding_model,
                            embedding=document.embedding if include_embeddings else None,
                            parent_id=document.parent_id,
                            access_level=document.metadata.access_level,
                            contains_pii=document.metadata.contains_pii,
                            created_at=document.created_at,
                            updated_at=document.updated_at,
                            processed_at=document.processed_at,
                        )
                        
                        # Insert document
                        insert_query = """
                        INSERT INTO documents (
                            id, content, document_type, status, error_message,
                            title, file_path, file_size, mime_type, encoding,
                            language, word_count, page_count, duration,
                            processing_time, chunk_count, embedding_model,
                            embedding, parent_id, access_level, contains_pii,
                            created_at, updated_at, processed_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                            $22, $23, $24
                        )
                        """
                        
                        await conn.execute(
                            insert_query,
                            doc_record.id,
                            doc_record.content,
                            doc_record.document_type.value,
                            doc_record.status.value,
                            doc_record.error_message,
                            doc_record.title,
                            doc_record.file_path,
                            doc_record.file_size,
                            doc_record.mime_type,
                            doc_record.encoding,
                            doc_record.language,
                            doc_record.word_count,
                            doc_record.page_count,
                            doc_record.duration,
                            doc_record.processing_time,
                            doc_record.chunk_count,
                            doc_record.embedding_model,
                            doc_record.embedding.tolist() if doc_record.embedding is not None else None,
                            doc_record.parent_id,
                            doc_record.access_level,
                            doc_record.contains_pii,
                            doc_record.created_at,
                            doc_record.updated_at,
                            doc_record.processed_at,
                        )
                        
                        result.add_success(document.id, "insert")
                        
                    except Exception as e:
                        result.add_error(f"Failed to store document {document.id}: {e}", document.id)
                        
        except Exception as e:
            # Transaction failed, mark all items in batch as failed
            for document in batch:
                result.add_error(f"Batch transaction failed: {e}", document.id)
    
    async def batch_store_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: UUID
    ) -> BatchOperationResult:
        """
        Store multiple document chunks in batches.
        
        Args:
            chunks: List of document chunks to store
            document_id: Parent document ID
            
        Returns:
            Batch operation result with statistics
        """
        start_time = time.time()
        result = BatchOperationResult(
            total_items=len(chunks),
            successful_items=0,
            failed_items=0,
            processing_time=0.0
        )
        
        try:
            # Convert chunks to records
            chunk_records = []
            for chunk in chunks:
                chunk_record = ChunkRecord(
                    id=chunk.id,
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    start_position=chunk.start_position,
                    end_position=chunk.end_position,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    semantic_type=chunk.semantic_type,
                    importance_score=chunk.importance_score,
                    embedding=chunk.embedding,
                    created_at=chunk.created_at,
                )
                chunk_records.append(chunk_record)
            
            # Process in batches
            batches = [
                chunk_records[i:i + self.batch_size]
                for i in range(0, len(chunk_records), self.batch_size)
            ]
            
            for batch in batches:
                try:
                    async with self.connection.get_transaction() as conn:
                        insert_query = """
                        INSERT INTO document_chunks (
                            id, document_id, chunk_index, content,
                            start_position, end_position, page_number, section_title,
                            semantic_type, importance_score, embedding, created_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                        )
                        """
                        
                        for chunk_record in batch:
                            await conn.execute(
                                insert_query,
                                chunk_record.id,
                                chunk_record.document_id,
                                chunk_record.chunk_index,
                                chunk_record.content,
                                chunk_record.start_position,
                                chunk_record.end_position,
                                chunk_record.page_number,
                                chunk_record.section_title,
                                chunk_record.semantic_type,
                                chunk_record.importance_score,
                                chunk_record.embedding.tolist(),
                                chunk_record.created_at,
                            )
                            
                            result.add_success(chunk_record.id, "insert")
                            
                except Exception as e:
                    for chunk_record in batch:
                        result.add_error(f"Failed to store chunk: {e}", chunk_record.id)
            
            result.processing_time = time.time() - start_time
            
            logger.info(
                f"Batch chunk storage completed: "
                f"{result.successful_items}/{result.total_items} successful "
                f"in {result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            result.processing_time = time.time() - start_time
            result.add_error(f"Batch chunk operation failed: {e}")
            logger.error(f"Batch chunk storage failed: {e}")
            return result


class SearchOperations:
    """
    Advanced search operations with multiple search strategies.
    
    Features:
    - Multi-modal search across different content types
    - Semantic search with contextual understanding
    - Faceted search with filtering and aggregation
    - Search result ranking and relevance scoring
    """
    
    def __init__(self, connection: SupabaseConnection):
        """Initialize search operations."""
        self.connection = connection
        self.vector_ops = VectorOperations(connection)
    
    async def multi_modal_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        document_types: Optional[List[DocumentType]] = None,
        max_results: int = 20,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, List[SearchResult]]:
        """
        Perform multi-modal search across different content types.
        
        Args:
            query_embedding: Query vector embedding
            query_text: Query text
            document_types: Filter by document types
            max_results: Maximum results per modality
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary of search results by modality
        """
        try:
            results = {}
            
            # Define modality groups
            modality_groups = {
                "text": [DocumentType.TEXT, DocumentType.CODE, DocumentType.STRUCTURED],
                "visual": [DocumentType.IMAGE, DocumentType.PDF],
                "media": [DocumentType.VIDEO, DocumentType.AUDIO],
            }
            
            # Search each modality group
            for modality, types in modality_groups.items():
                if document_types is None or any(dt in document_types for dt in types):
                    search_types = types if document_types is None else [dt for dt in types if dt in document_types]
                    
                    modality_results = await self.vector_ops.hybrid_search(
                        query_embedding=query_embedding,
                        query_text=query_text,
                        document_types=search_types,
                        max_results=max_results,
                        similarity_threshold=similarity_threshold,
                    )
                    
                    results[modality] = modality_results
            
            logger.info(f"Multi-modal search completed across {len(results)} modalities")
            return results
            
        except Exception as e:
            logger.error(f"Multi-modal search failed: {e}")
            raise DatabaseError(f"Multi-modal search failed: {e}") from e
    
    async def get_document_by_id(self, document_id: UUID) -> Optional[DocumentRecord]:
        """Retrieve document by ID."""
        try:
            async with self.connection.get_connection() as conn:
                query = "SELECT * FROM documents WHERE id = $1"
                row = await conn.fetchrow(query, document_id)
                
                if row:
                    return DocumentRecord.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise DatabaseError(f"Document retrieval failed: {e}") from e
    
    async def get_document_chunks(self, document_id: UUID) -> List[ChunkRecord]:
        """Retrieve all chunks for a document."""
        try:
            async with self.connection.get_connection() as conn:
                query = """
                SELECT * FROM document_chunks 
                WHERE document_id = $1 
                ORDER BY chunk_index
                """
                rows = await conn.fetch(query, document_id)
                
                return [ChunkRecord.from_dict(dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            raise DatabaseError(f"Chunk retrieval failed: {e}") from e